from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

from preprocessing import continuous_confounders, categorical_confounders

# Function to deconfound features by regressing out the effects of continuous and categorical confounders,
# including non-linear interactions of age and sex
def deconfound(X_train, con_confounder_train, cat_confounder_train, X_test=None, con_confounder_test=None, cat_confounder_test=None, age_var="Age_in_Yrs", sex_var="Gender"):
    # Step 1: Normalize continuous confounders
    scaler = StandardScaler()
    con_confounder_train_scaled = scaler.fit_transform(con_confounder_train)

    # Step 2: One-hot encode the categorical confounders
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' to avoid multicollinearity
    cat_confounder_train_encoded = encoder.fit_transform(cat_confounder_train)

    age_index = continuous_confounders.index(age_var)
    sex_index = categorical_confounders.index(sex_var)

    # Extract Age and Sex variables
    age = con_confounder_train_scaled[:, age_index]  # Age from continuous confounders
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


