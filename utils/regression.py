# # Confounders 
# # confounds detailed in https://www.sciencedirect.com/science/article/pii/S1053811920300914 & https://www.humanconnectome.org/storage/app/media/documentation/s500/HCP500_MegaTrawl_April2015.pdf
# # In Data Table: Age (Age_in_Yrs), Sex (Gender), Ethnicity (Ethnicity), Weight (Weight), Brain Size (FS_BrainSeg_Vol), Intracranial Volume (FS_IntraCranial_Vol), Confounds Modelling Slow Drift (TestRetestInterval), reconstruction code version (fMRI_3T_ReconVrs) or Acquisition Quarter (Acquisition)
# # In pathfile: Head Motion (a summation over all timepoints of timepoint-to-timepoint relative head motion or average) Movement_RelativeRMS_mean.txt (Since LR RL and session scans are concateanted, take average of this average)
# # Mentioned in papers but not found: variables (x, y, z, table) related to bed position in scanner
# confounders =  ["Age_in_Yrs", "Gender", "Race", "Ethnicity", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"] # TODO uncomment
# continuous_confounders = ["Age_in_Yrs", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"] # TODO uncomment
# categorical_confounders = ["Gender","fMRI_3T_ReconVrs", "Race", "Ethnicity"] # TODO uncomment

# phen_confounders = ["Age_in_Yrs", "Gender", "Race", "Ethnicity", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"] # TODO uncomment
# phen_continuous_confounders = ["Age_in_Yrs", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"] # TODO uncomment
# phen_categorical_confounders = ["Gender","fMRI_3T_ReconVrs", "Race", "Ethnicity"] # TODO uncomment

# confounders =  ["Age_in_Yrs", "Gender", "Ethnicity", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"] # TODO uncomment
# continuous_confounders = ["Age_in_Yrs", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"] # TODO uncomment
# categorical_confounders = ["Gender","fMRI_3T_ReconVrs", "Ethnicity"] # TODO uncomment

# phen_confounders = ["Age_in_Yrs", "Gender", "Ethnicity", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"] # TODO uncomment
# phen_continuous_confounders = ["Age_in_Yrs", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"] # TODO uncomment
# phen_categorical_confounders = ["Gender","fMRI_3T_ReconVrs", "Ethnicity"] # TODO uncomment

# "Race","Ethnicity"
# phen_confounders = ["Age_in_Yrs", "Gender","Race","Ethnicity"] # TODO uncomment
# phen_continuous_confounders = ["Age_in_Yrs"] # TODO uncomment
# phen_categorical_confounders = ["Gender","Race","Ethnicity",] # TODO uncomment
# phen_confounders = ["Age_in_Yrs", "Gender","Handedness","FS_BrainSeg_Vol","fMRI_3T_ReconVrs","motion"] # TODO uncomment
# phen_continuous_confounders = ["Age_in_Yrs","Handedness","FS_BrainSeg_Vol","motion"] # TODO uncomment
# phen_categorical_confounders = ["Gender","fMRI_3T_ReconVrs"] # TODO uncomment
# phen_confounders = ["Age_in_Yrs", "Gender"] # TODO uncomment
# phen_continuous_confounders = ["Age_in_Yrs"] # TODO uncomment
# phen_categorical_confounders = ["Gender"] # TODO uncomment

# TODO Confounds for sex
# # From https://www.sciencedirect.com/science/article/pii/S1053811920300914#appsec1 "For sex prediction, the crucial confounds are age, height, weight, head motion and head size" # TODO delete
# confounders = ["Age_in_Yrs", "Height", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"] # TODO delete
# continuous_confounders = ["Age_in_Yrs", "Height", "Weight", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"] # TODO delete
# categorical_confounders = ["fMRI_3T_ReconVrs"] # TODO delete

# Function to deconfound features by regressing out the effects of continuous and categorical confounders,
# including non-linear interactions of age and sex

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

confounders =  ["Age_in_Yrs", "Gender", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"]
continuous_confounders = ["Age_in_Yrs", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"]
categorical_confounders = ["Gender","fMRI_3T_ReconVrs"]


# phen_confounders =  ["Age_in_Yrs", "Gender", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "fMRI_3T_ReconVrs", "motion"]
# phen_continuous_confounders = ["Age_in_Yrs", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol", "motion"]
# phen_categorical_confounders = ["Gender","fMRI_3T_ReconVrs"]
phen_confounders = ["Age_in_Yrs", "Gender"]
phen_continuous_confounders = ["Age_in_Yrs"]
phen_categorical_confounders = ["Gender"]

def deconfound(X_train, con_confounder_train, cat_confounder_train, X_test=None, con_confounder_test=None, cat_confounder_test=None, age_var="Age_in_Yrs", sex_var="Gender",phenotype_labels=None,output_path=""):
    # Step 1: Normalize continuous confounders
    scaler = StandardScaler()
    con_confounder_train_scaled = scaler.fit_transform(con_confounder_train)

    # Step 2: One-hot encode the categorical confounders
    encoder = OneHotEncoder(drop='first', sparse_output=False,handle_unknown='ignore')  # drop='first' to avoid multicollinearity
    cat_confounder_train_encoded = encoder.fit_transform(cat_confounder_train)
    age_index = continuous_confounders.index(age_var)
    sex_index = categorical_confounders.index(sex_var)

    # Extract Age and Sex variables
    age = con_confounder_train_scaled[:, age_index]  # Age from continuous confounders
    # TODO Aligns since drop=first so only one column (first) for sex but not stable
    sex = cat_confounder_train_encoded[:, sex_index]  # Sex from encoded categorical confounders 

    # Construct non-linear interaction terms
    age_squared = age ** 2
    interaction_age_sex = age * sex 
    interaction_age_squared_sex = age_squared * sex

    # Include these interaction terms in the combined confounders for training
    interaction_terms_train = np.column_stack([age_squared, interaction_age_sex, interaction_age_squared_sex])

    confounders_train_combined = np.hstack([con_confounder_train_scaled, cat_confounder_train_encoded, interaction_terms_train])
    
    # Step 4: Fit a linear regression model to predict X_train from combined confounders
    model = LinearRegression()
    model.fit(confounders_train_combined, X_train)

    # Step 5: Predict the confounder effects on both training and test features
    predicted_train = model.predict(confounders_train_combined)

    # Step 6: Calculate the residuals (deconfounded features)
    X_train_dc = X_train - predicted_train

    if phenotype_labels:
        continuous_labels = con_confounder_train.columns.tolist()
        categorical_labels = encoder.get_feature_names_out(cat_confounder_train.columns).tolist()
        interaction_labels = ["Age^2", "Age*Sex", "Age^2*Sex"]
        confound_labels = continuous_labels + categorical_labels + interaction_labels
        visualize_beta_weights(model.coef_, phenotype_labels, confound_labels, output_path)

    if X_test is not None:
        con_confounder_test_scaled = scaler.transform(con_confounder_test)
        cat_confounder_test_encoded = encoder.transform(cat_confounder_test)
        # Repeat for the test data
        age_test = con_confounder_test_scaled[:, age_index]
        sex_test = cat_confounder_test_encoded[:, sex_index] 
        age_squared_test = age_test ** 2
        interaction_age_sex_test = age_test * sex_test 
        interaction_age_squared_sex_test = age_squared_test * sex_test 

        interaction_terms_test = np.column_stack([age_squared_test, interaction_age_sex_test, interaction_age_squared_sex_test])
        confounders_test_combined = np.hstack([con_confounder_test_scaled, cat_confounder_test_encoded, interaction_terms_test])
        predicted_test = model.predict(confounders_test_combined)
        X_test_dc = X_test - predicted_test

        return X_train_dc, X_test_dc

    return X_train_dc


def visualize_beta_weights(beta_weights, phenotype_labels, confound_labels, output_path):
    """
    Visualize the beta weights of linear regression using a heatmap or bar chart.

    Parameters:
    - beta_weights (np.ndarray): A 1D or 2D array of beta weights (confounders x phenotypes).
    - phenotype_labels (list): List of phenotype names (columns of beta_weights).
    - confound_labels (list): List of confounder names (rows of beta_weights).
    - output_path (str): File path to save the plot.

    Returns:
    - None
    """
    # Ensure beta_weights is a 2D array
    if beta_weights.ndim == 1:
        beta_weights = beta_weights[:, np.newaxis]
        
    num_phenotypes = beta_weights.shape[1]
    
    plt.figure(figsize=(12, 8))
    
    if num_phenotypes == 1:
        # Use a bar chart for a single phenotype
        plt.barh(confound_labels, beta_weights[:, 0], color='skyblue')
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.title(f"Beta Weights for Phenotype: {phenotype_labels}")
        plt.xlabel("Beta Weight")
        plt.ylabel("Confounders")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"beta_weights_bar_{phenotype_labels}.svg"), format="svg")
    else:
        # Use a heatmap for multiple phenotypes
        sns.heatmap(
            beta_weights, annot=True, cmap="coolwarm", cbar=True,
            xticklabels=[phenotype_labels], yticklabels=confound_labels, center=0
        )
        plt.title("Beta Weights of Linear Regression")
        plt.xlabel("Phenotypes")
        plt.ylabel("Confounders")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "beta_weights_heatmap.svg"), format="svg")
    
    plt.show()
    plt.close()
