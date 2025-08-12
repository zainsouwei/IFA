import numpy as np
import nibabel as nib
import hcp_utils as hcp 
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import r2_score
import pickle

from skopt import gp_minimize
from skopt.space import Real

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pso import PSO  
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import StarmapParallelization
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

from preprocessing import load_subject

def evaluate_subject_helper(args):
    """
    A module-level helper function that unpacks arguments and calls
    the instance's evaluate_subject method.
    """
    instance, sub_path, hyperparams = args
    return instance.evaluate_subject(sub_path, hyperparams)

class DualRegressionOptimizer:
    def __init__(self, subject_paths, spatial_map, search_space=None, 
                 parallel_points=10, parallel_subs=3):
        """
        Parameters:
          - subject_paths: list of file paths for subjects.
          - spatial_map: spatial map array (e.g., from IFA/ICA), shape [V, C].
          - search_space: Optional; either a dict with keys 'alpha' and 'l1_ratio' 
                          or a list of skopt dimensions.
          - parallel_points: Number of parallel evaluations for Bayesian optimization.
          - parallel_subs: Number of parallel subject evaluations.
        """
        self.subject_paths = subject_paths
        self.spatial_map = spatial_map
        self.spatial_map_dm = self.spatial_map - self.spatial_map.mean(axis=0, keepdims=True)
        self.spatial_map_dm_plus = np.linalg.pinv(self.spatial_map_dm.T)
        self.parallel_points = parallel_points
        self.parallel_subs = parallel_subs

        if search_space is None:
            self.search_space_dict = {'alpha': (1e-3, 10), 'l1_ratio': ((1e-4), 1-(1e-4))}
        elif isinstance(search_space, dict):
            self.search_space_dict = search_space
        else:
            raise ValueError("search_space must be either None or a dict")

        # Build a list of skopt dimensions for Bayesian optimization.
        self.search_space_dims = [
            Real(self.search_space_dict['alpha'][0], 
                 self.search_space_dict['alpha'][1], 
                 prior='log-uniform', 
                 name='alpha'),
            Real(self.search_space_dict['l1_ratio'][0], 
                 self.search_space_dict['l1_ratio'][1],
                #  prior='log-uniform',
                 name='l1_ratio')
        ]


    def evaluate_subject(self, sub_path, hyperparams):
        """
        For a given subject file path, load data, compute the network matrix A,
        compute predictors (using the chosen mode), perform a train-test split,
        train an ElasticNet model with given hyperparameters, and return the R² score.
        """
        try:
            Xn = load_subject(sub_path)
            # Demean each time point (row) of Xn.
            Xn_demeaned = Xn - Xn.mean(axis=1, keepdims=True)
            # Compute the network matrix A.
            A = Xn_demeaned @ self.spatial_map_dm_plus
            # Compute predictors.
            predictors = hcp.normalize(A)
            # Split data into train and test sets.
            X_train, X_test, y_train, y_test = train_test_split(
                predictors, Xn, test_size=0.3, random_state=42
            )
            # Train ElasticNet.
            model = ElasticNet(alpha=hyperparams[0], l1_ratio=hyperparams[1], fit_intercept=False, selection='random')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred, multioutput='uniform_average')
            return score
        except Exception as e:
            print(f"Error processing subject {sub_path}: {e}")
            return 0.0

    def objective(self, hyperparams):
        """
        Shared objective function for both Bayesian and PSO optimization.
        Receives hyperparams as a list [alpha, l1_ratio] and returns the negative
        average R² score (to be minimized).
        """
        # Prepare the arguments for each subject evaluation.
        args = [(self, sub, hyperparams) for sub in self.subject_paths]
        with ProcessPoolExecutor(max_workers=self.parallel_subs) as executor:
            scores = list(executor.map(evaluate_subject_helper, args))
        avg_score = np.mean(scores)
        return -avg_score  # Negative because we minimize
    
    def objective_pso(self, hyperparams):
        # Evaluate each subject sequentially.
        scores = [evaluate_subject_helper((self, sub, hyperparams)) for sub in self.subject_paths]
        avg_score = np.mean(scores)
        return -avg_score  # Negative because we minimize


    def optimize_bayesian(self, n_calls=15, random_state=42):
        res = gp_minimize(
            func=self.objective,
            dimensions=self.search_space_dims,
            n_calls=n_calls,
            n_initial_points = int(n_calls/2),
            random_state=random_state,
            n_jobs=self.parallel_points,
            acq_optimizer="lbfgs" 
        )
        best_params = res.x
        best_cv_score = -res.fun
        return best_params, best_cv_score
    
    class MyOutput(Output):
        def __init__(self):
            super().__init__()
            self.best_f = Column("best_f", width=12)
            self.best_alpha = Column("best_alpha", width=12)
            self.best_l1 = Column("best_l1", width=12)
            # Add the new columns to the output columns.
            self.columns += [self.best_f, self.best_alpha, self.best_l1]

        def update(self, algorithm):
            super().update(algorithm)
            F = algorithm.pop.get("F")
            best_index = np.argmin(F)
            best_value = np.min(F)
            X = algorithm.pop.get("X")
            best_candidate = X[best_index]
            self.best_f.set(best_value)
            self.best_alpha.set(best_candidate[0])
            self.best_l1.set(best_candidate[1])

    class AggregatedElasticNetCVProblem(ElementwiseProblem):
        def __init__(self, outer, **kwargs):
            """
            Parameters:
              - outer: a reference to the outer DualRegressionOptimizer instance.
            """
            self.outer = outer
            # Set bounds: convert alpha to log10 space.
            xl = [np.log10(self.outer.search_space_dict['alpha'][0]),
                  self.outer.search_space_dict['l1_ratio'][0]]
            xu = [np.log10(self.outer.search_space_dict['alpha'][1]),
                  self.outer.search_space_dict['l1_ratio'][1]]
            super().__init__(n_var=2, n_obj=1, n_constr=0, xl=xl, xu=xu, **kwargs)

        def _evaluate(self, x, out):
            alpha_original = 10 ** x[0]
            out["F"] = self.outer.objective_pso([alpha_original, x[1]])

    def optimize_pso(self, random_state=42, particles=15,ftol=1e-2, period=1, n_calls=75):
        # Use a multiprocessing pool for parallel evaluations.
        pool = multiprocessing.Pool(processes=self.parallel_points)
        runner = StarmapParallelization(pool.starmap)
        problem_instance = self.AggregatedElasticNetCVProblem(outer=self,elementwise_runner=runner)
        algorithm = PSO(pop_size=particles)
        termination = DefaultSingleObjectiveTermination(ftol=ftol, period=period, n_max_gen=int(n_calls/particles))
        res = minimize(problem_instance, algorithm, termination=termination,
                       seed=random_state, verbose=True, output=self.MyOutput())

        pool.close()
        pool.join()
        best_params = res.X
        best_cv_score = -res.F[0]
        print("PSO optimization:")
        print("  Best hyperparameters found:", best_params)
        print("  Best CV R² score:", best_cv_score)
        # Note: best_params[0] is in log-space; convert it.
        best_params[0] = 10 ** best_params[0]
        return best_params, best_cv_score

    def optimize(self, optimizer="bayesian", **kwargs):
        """
        Optimize the ElasticNet hyperparameters over all subjects using the selected method.
        
        Parameters:
          - optimizer: "bayesian" or "pso"
          - kwargs: additional parameters passed to the specific optimizer method.
        
        Returns:
          A tuple: (best_params, best_cv_score)
        """
        if optimizer.lower() == "bayesian":
            return self.optimize_bayesian(**kwargs)
        elif optimizer.lower() == "pso":
            return self.optimize_pso(**kwargs)
        else:
            raise ValueError("optimizer must be either 'bayesian' or 'pso'")

# https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00115/full
class DualRegress:
    def __init__(self,subs,spatial_maps,train_index, train_labels,outputfolders, workers=20, sample=50, method="bayesian", parallel_points=10, parallel_subs=2, n_calls=20, random_state=42):
        self.subs = subs
        self.spatial_maps = spatial_maps
        self.train_index = train_index
        self.train_labels = train_labels
        self.workers = workers
        self.sample = sample
        self.method = method
        
        self.spatial_maps_dm = [s_map - s_map.mean(axis=0, keepdims=True) for s_map in self.spatial_maps] # Demean the columns of z_maps (V x C)
        self.spatial_map_dm_plus = [np.linalg.pinv(s_map.T) for s_map in self.spatial_maps_dm]
        
        self.parallel_points = parallel_points
        self.parallel_subs = parallel_subs
        self.hyperparams = []
        
        self.n_calls = n_calls
        self.random_state = random_state
        self.outputfolders = outputfolders
        for map_outputfolder in self.outputfolders:
            if not os.path.exists(map_outputfolder):
                os.makedirs(map_outputfolder)

        self.dual_regression_results = []
        for i in range(len(self.spatial_maps)):
            self.dual_regression_results.append({'A': [], 'spatial_map': [], 'reconstruction_error': []})


    def stage_two_elastic(self,A,X,map_index=0,norm_index=0):
        hyper = self.hyperparams[map_index][norm_index]
        Elastic = ElasticNet(alpha=hyper[0], l1_ratio=hyper[1], fit_intercept=False, selection='random')
        Elastic.fit(A, X)
        spatial_map = Elastic.coef_.T  # Coefficients are Components x Grayordinates (C x V)
        return spatial_map
    
    def recon(self, A, spatial_map, Xn):
        # Reconstruct the data using the spatial map
        reconstructed = A @ spatial_map.T
        # Compute residuals
        residuals = Xn - reconstructed
        # Compute variances
        var_residuals = np.var(residuals, ddof=1)
        var_original = np.var(Xn, ddof=1)
        # Calculate variance explained
        variance_explained = (1 - (var_residuals / var_original)) * 100
        
        return variance_explained
    
    def calculate_netmat_and_spatial_map(self,Xn, map_index=0):
        """
        Calculate the network matrix (netmat) and spatial map for a given subject and z_maps.
        
        Parameters:
        Xn (array): Time x Grayordinates normalized data matrix (Time x V)
        map_index: which map in list to use 
        Returns:
        spatial_map (array): Components x Grayordinates matrix (C x V)
        """
        map_plus = self.spatial_map_dm_plus[map_index]
        
        # Time x Components
        A = (Xn-Xn.mean(axis=1,keepdims=True)) @ map_plus  # A is Time x Components (T x C)
        A = A - A.mean(axis=0,keepdims=True)

        # Normalized Time x Components matrix
        An = hcp.normalize(A)  # An is Time x Components (T x C)
        # An = Adm/np.percentile(np.abs(Adm),95,axis=0)
        
        spatial_map_ = self.stage_two_elastic(An,Xn,map_index=map_index,norm_index=0)
        
        rec_error = self.recon(An, spatial_map_.T, Xn)
    
        return [A, spatial_map_, rec_error]

        # Components x Grayordinates spatial map
        # spatial_map = np.linalg.pinv(An) @ hcp.normalize(Xn)  # Spatial map is Components x Grayordinates (C x V)
        # Automatically determine alpha using RidgeCV
        # Elastic = ElasticNet(alpha=1.0, l1_ratio=0.01, fit_intercept=False)
        # Elastic.fit(An, hcp.normalize(Xn))
        # spatial_map = Elastic.coef_.T  # Coefficients are Components x Grayordinates (C x V)
        # coef = Elastic.coef_.T
        # y_hat = Elastic.predict(An)  # Fitted response for normalized data
        # del Elastic
        # # # print(y_hat.shape, hcp.normalize(Xn).shape)
        # ScaleReg = LinearRegression(fit_intercept=False)
        # kappa = ScaleReg.fit(y_hat.reshape(-1,1), hcp.normalize(Xn).reshape(-1,1)).coef_.T
        # # # print(Elastic.coef_.T.shape, kappa.shape)
        # # # print(kappa)
        # del ScaleReg
        # spatial_map = coef * kappa  # Apply correction

        # spatial_mapdm = np.linalg.pinv(Adm) @ hcp.normalize(Xn)
        # Elastic_dm = ElasticNet(alpha=1.0, l1_ratio=0.01, fit_intercept=False)
        # Elastic_dm.fit(Adm, hcp.normalize(Xn))
        # spatial_mapdm = Elastic_dm.coef_.T  # Coefficients are Components x Grayordinates (C x V)
        # coef_dm = Elastic_dm.coef_.T
        # y_hat_dm = Elastic_dm.predict(Adm)  # Fitted response for normalized data
        # del Elastic_dm
        # # # print(y_hat_dm.shape, hcp.normalize(Xn).shape)
        # ScaleReg_dm = LinearRegression(fit_intercept=False)
        # kappa_dm = ScaleReg_dm.fit(y_hat_dm.reshape(-1,1), hcp.normalize(Xn).reshape(-1,1)).coef_.T
        # # # # print(Elastic_dm.coef_.T.shape, kappa_dm.shape)
        # # # print(kappa_dm)
        # del ScaleReg_dm
        # spatial_mapdm = coef_dm * kappa_dm  # Apply correction

    def dual_regress_sub(self, sub_path):
        try:
            Xn = load_subject(sub_path)
            Xn.flags.writeable = False  # Mark the array as read-only.        
            # Calculate for IFA z_maps
            dual_results = []
            for i in range(len(self.spatial_maps)):
                dual_result_i = self.calculate_netmat_and_spatial_map(Xn, map_index=i)
                dual_results.append(dual_result_i)
            
            return dual_results
    
        except Exception as e:
            print(f"Error processing subject: {e}")
            return None

    def optimize_hyperparameters(self):
        # Create an instance of DualRegressionOptimizer.
        train_paths = self.subs[self.train_index]
        
        # Ensure Class Distribution Remains
        if self.sample >= len(train_paths):
            sampled_subjects = train_paths
        else:
            # Only one split
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.sample, random_state=self.random_state)
            for sample_idx, _ in sss.split(train_paths, self.train_labels):
                sampled_subjects = train_paths[sample_idx]
        for i, s_map in enumerate(self.spatial_maps):
            map_hyperparams = []
            best_params_path = os.path.join(self.outputfolders[i], "best_params.npy")
            best_cv_path     = os.path.join(self.outputfolders[i], "best_cv.npy")
            
            if os.path.exists(best_params_path) and os.path.exists(best_cv_path):
                print(f"[cache] Hyperparameters for map {i}")
                best_params = np.load(best_params_path)
                best_cv_score = np.load(best_cv_path)
            else:
                print(f"[running] Hyperparameters for map {i}")
                optimizer_instance = DualRegressionOptimizer(sampled_subjects, s_map,parallel_points=self.parallel_points, parallel_subs=self.parallel_subs)

                best_params, best_cv_score = optimizer_instance.optimize(optimizer=self.method, n_calls=self.n_calls, random_state=self.random_state)
                np.save(best_params_path,np.array(best_params))
                np.save(best_cv_path,np.array(best_cv_score))
            
            map_hyperparams.append(best_params)   
            self.hyperparams.append(map_hyperparams)
        
            # Choose an optimization method: "bayesian" or "pso"
            # method = "pso"  # or "bayesian"
            # best_params, best_cv_score = optimizer_instance.optimize(
            #     optimizer=method, random_state=42, particles=100,ftol=1e-2, period=1, n_max_gen=5
            # )
            
    def dual_regress(self):
        """
        Run dual regression for all subjects in parallel and store & save the aggregated results.
        For each spatial map (indexed by i), the following are stored:
        - Normalized results: 'An' (network matrices) and 'spatial_map'
        - Reconstruction error: reconstruction_error
        The results are saved to the corresponding output folder specified in self.outputfolders.
        """
        self.optimize_hyperparameters()

        # Process all subjects in parallel.
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = list(executor.map(self.dual_regress_sub, self.subs))
        
        num_maps = len(self.spatial_maps)
        
        # Aggregate results from each subject.
        # Each subject's result is expected to be a list with one element per spatial map:
        # [A, spatial_map, variance_explained]
        for subject_result in results:
            if subject_result is None:
                continue
            for i, map_result in enumerate(subject_result):
                self.dual_regression_results[i]['A'].append(np.asarray(map_result[0], dtype=np.float32))
                self.dual_regression_results[i]['spatial_map'].append(np.asarray(map_result[1], dtype=np.float32))
                self.dual_regression_results[i]['reconstruction_error'].append(np.asarray(map_result[2], dtype=np.float32))

        for i in range(num_maps):
            out_folder = self.outputfolders[i]
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            # Keep A as a list of (T_i, C) arrays
            with open(os.path.join(out_folder, "A.pkl"), "wb") as f:
                pickle.dump(self.dual_regression_results[i]['A'], f, protocol=pickle.HIGHEST_PROTOCOL)

            spatial_map_arr = np.asarray(self.dual_regression_results[i]['spatial_map'], dtype=np.float32)
            recon_err_arr   = np.asarray(self.dual_regression_results[i]['reconstruction_error'], dtype=np.float32)

            np.save(os.path.join(out_folder, "spatial_map.npy"), spatial_map_arr)
            np.save(os.path.join(out_folder, "reconstruction_error.npy"), recon_err_arr)