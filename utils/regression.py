# # Confounders 
# # confounds detailed in https://www.sciencedirect.com/science/article/pii/S1053811920300914 & https://www.humanconnectome.org/storage/app/media/documentation/s500/HCP500_MegaTrawl_April2015.pdf
# # In Data Table: Age (Age_in_Yrs), Sex (Gender), Ethnicity (Ethnicity), Weight (Weight), Brain Size (FS_BrainSeg_Vol), Intracranial Volume (FS_IntraCranial_Vol), Confounds Modelling Slow Drift (TestRetestInterval), reconstruction code version (fMRI_3T_ReconVrs) or Acquisition Quarter (Acquisition)
# # In pathfile: Head Motion (a summation over all timepoints of timepoint-to-timepoint relative head motion or average) Movement_RelativeRMS_mean.txt (Since LR RL and session scans are concateanted, take average of this average)
# # Mentioned in papers but not found: variables (x, y, z, table) related to bed position in scanner
# confounders =  ["Age_in_Yrs", "Gender", "Race", "Ethnicity", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"]
# continuous_confounders = ["Age_in_Yrs", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"]
# categorical_confounders = ["Gender","fMRI_3T_ReconVrs", "Race", "Ethnicity"]

# phen_confounders = ["Age_in_Yrs", "Gender", "Race", "Ethnicity", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"]
# phen_continuous_confounders = ["Age_in_Yrs", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"]
# phen_categorical_confounders = ["Gender","fMRI_3T_ReconVrs", "Race", "Ethnicity"]

# Confounds for sex
# # From https://www.sciencedirect.com/science/article/pii/S1053811920300914#appsec1 "For sex prediction, the crucial confounds are age, height, weight, head motion and head size"
# confounders = ["Age_in_Yrs", "Height", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"]
# continuous_confounders = ["Age_in_Yrs", "Height", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"]
# categorical_confounders = ["fMRI_3T_ReconVrs"]

# Function to deconfound features by regressing out the effects of continuous and categorical confounders,
# including non-linear interactions of age and sex

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def visualize_beta_weights(beta_weights, phenotype_labels, confound_labels, output_path):
    plt.figure(figsize=(12, 8))
    
    # Use a bar chart for a single phenotype
    plt.barh(confound_labels, beta_weights, color='skyblue')
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f"Beta Weights for Phenotype: {phenotype_labels}")
    plt.xlabel("Beta Weight")
    plt.ylabel("Confounders")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"beta_weights_bar_{phenotype_labels}.svg"), format="svg")
    
    plt.show()
    plt.close()

def deconfound(X_train, con_confounder_train, cat_confounder_train, X_test=None, con_confounder_test=None, cat_confounder_test=None, output_path=""):
    age_var="age"
    sex_var="sex"

    is_df = isinstance(X_train, pd.DataFrame)

    # Normalize continuous confounders
    scaler = StandardScaler()
    con_conf_train_scaled = scaler.fit_transform(con_confounder_train)

    # One-hot encode categorical confounders
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    cat_conf_train_encoded = encoder.fit_transform(cat_confounder_train)
    
    age_idx = con_confounder_train.columns.get_loc(age_var)
    cat_feats = encoder.get_feature_names_out()
    
    if sex_var in cat_confounder_train.columns:
        sex_idxs = [i for i, f in enumerate(cat_feats) if f.startswith(f"{sex_var}_")]

        # Extract Age and Sex variables and Construct interaction terms
        age = con_conf_train_scaled[:, age_idx:(age_idx+1)] 
        sex = cat_conf_train_encoded[:, sex_idxs] 
        age_squared = age ** 2
        interaction_age_sex = age * sex 
        interaction_age_squared_sex = age_squared * sex

        interaction_terms_train = np.column_stack([age_squared, interaction_age_sex, interaction_age_squared_sex])
        conf_train_combined = np.hstack([con_conf_train_scaled, cat_conf_train_encoded, interaction_terms_train])
    else:
        conf_train_combined = np.hstack([con_conf_train_scaled, cat_conf_train_encoded])

    # Fit a linear regression model to predict X_train from combined confounders
    model = LinearRegression()
    model.fit(conf_train_combined, X_train)

    # Predict the confounder effects on both training and test features
    predicted_train = model.predict(conf_train_combined)

    # Calculate the residuals (deconfounded features)
    X_train_dc = X_train - predicted_train

    # if X was a DataFrame, it represents target phenotypes. Plot beta‐weights per target phenotype
    if is_df:
        con_labels = con_confounder_train.columns.tolist()
        cat_labels  = cat_feats.tolist()
        if sex_var in cat_confounder_train.columns:
            sex_feats   = [f for f in cat_feats if f.startswith(f"{sex_var}_")]
            interaction_labels = [f"{age_var}^2"]
            interaction_labels += [f"{age_var}*{f}"   for f in sex_feats]
            interaction_labels += [f"{age_var}^2*{f}" for f in sex_feats]
            confound_labels = con_labels + cat_labels + interaction_labels
        else:
            confound_labels = con_labels + cat_labels

        phenotype_labels = X_train.columns.tolist()

        # TODO Verify this transpose
        # coefs come back as (n_targets, n_conf) so transpose to (n_conf, n_targets/phenotypes)
        betas = model.coef_.T
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            for j, phen in enumerate(phenotype_labels):
                visualize_beta_weights(betas[:, j], phen, confound_labels,output_path)

    if X_test is not None:
        con_conf_test_scaled = scaler.transform(con_confounder_test)
        cat_conf_test_encoded = encoder.transform(cat_confounder_test)
        if sex_var in cat_confounder_train.columns:
            age_test = con_conf_test_scaled[:, age_idx:(age_idx+1)]
            sex_test = cat_conf_test_encoded[:, sex_idxs] 
            age_squared_test = age_test ** 2
            interaction_age_sex_test = age_test * sex_test 
            interaction_age_squared_sex_test = age_squared_test * sex_test 

            interaction_terms_test = np.column_stack([age_squared_test, interaction_age_sex_test, interaction_age_squared_sex_test])
            confounders_test_combined = np.hstack([con_conf_test_scaled, cat_conf_test_encoded, interaction_terms_test])
        else:
            confounders_test_combined = np.hstack([con_conf_test_scaled, cat_conf_test_encoded])

        predicted_test = model.predict(confounders_test_combined)
        X_test_dc = X_test - predicted_test

        return X_train_dc, X_test_dc

    return X_train_dc