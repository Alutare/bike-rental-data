import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib.pyplot as plt

# Import the preprocess_data function from app.py
from app import preprocess_data
warnings.filterwarnings("ignore", category=DeprecationWarning)

def preprocess_for_model(data, target_column):
    # Selecting features and target variable
    train_feature_space = data.iloc[:, data.columns != target_column]
    target_class = data[target_column]

    # Drop specified columns
    if target_column == 'casual':
        train_feature_space = train_feature_space.drop(["cnt", "workingday", "registered"], axis=1)
    elif target_column == 'registered':
        train_feature_space = train_feature_space.drop(["cnt", "workingday", "casual"], axis=1)

    # One-Hot Encoding of Categorical Features
    train_feature_space_encoded = pd.get_dummies(train_feature_space, drop_first=True)

    # Normalize the numeric columns
    scaler = MinMaxScaler()
    train_feature_space_scaled = scaler.fit_transform(train_feature_space_encoded)
    train_feature_space_scaled = pd.DataFrame(train_feature_space_scaled, columns=train_feature_space_encoded.columns)

    # Split the data into training and test sets
    training_set, test_set, train_target, test_target = train_test_split(
        train_feature_space_scaled,
        target_class,
        test_size=0.30,
        random_state=456
    )

    # Flatten target variables
    train_target = train_target.values.ravel()
    test_target = test_target.values.ravel()
    
    return training_set, test_set, train_target, test_target

def train_ols_model(X, y):
    # Add a constant for the intercept
    X_encoded = sm.add_constant(X)

    # Log-transform the target variable (add 1 to avoid log(0))
    y_log = np.log(y + 1)

    # Fit the OLS model
    model = sm.OLS(y_log, X_encoded.astype(float)).fit()

    return model

def cross_validate_ols(X, y, cv=10):
    kf = KFold(n_splits=cv, shuffle=True, random_state=456)
    rmse_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model and calculate RMSE for the test fold
        model = train_ols_model(X_train, y_train)
        rmse = calculate_rmse(model, X_test, y_test)
        rmse_scores.append(rmse)

    return np.array(rmse_scores)

def calculate_rmse(model, X, y):
    # Predict the target variable
    y_pred_log = model.predict(sm.add_constant(X.astype(float)))
    
    # Transform predictions back from log scale
    y_pred = np.exp(y_pred_log) - 1  # subtract 1 to reverse the log transformation

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse

def train_random_forest(X, y):
    rf = RandomForestRegressor(random_state=12345)
    rf.fit(X, y)
    return rf

def cross_validate_random_forest(X, y, cv=10):
    rf = RandomForestRegressor(random_state=12345)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)  # Convert negative MSE to positive RMSE
    return rmse_scores

def evaluate_random_forest(model, X, y):
    rf_predictions = model.predict(X)
    
    # Calculate RMSE
    rmse_rf = np.sqrt(mean_squared_error(y, rf_predictions))
    
    # Calculate R²
    r2_rf = r2_score(y, rf_predictions)
    
    return rf_predictions, rmse_rf, r2_rf

def find_best_k_for_ols(X, y, k_range=range(2, 11)):
    best_k = None
    lowest_rmse = float('inf')
    rmse_results = {}

    for k in k_range:
        rmse_scores = cross_validate_ols(X, y, cv=k)
        avg_rmse = rmse_scores.mean()
        rmse_results[k] = avg_rmse

        if avg_rmse < lowest_rmse:
            lowest_rmse = avg_rmse
            best_k = k

    print(f"Best k for OLS: {best_k} with RMSE: {lowest_rmse:.2f}")
    return best_k, rmse_results

# Helper function to find the best k for Random Forest
def find_best_k_for_rf(X, y, k_range=range(2, 11)):
    best_k = None
    lowest_rmse = float('inf')
    rmse_results = {}

    for k in k_range:
        rmse_scores = cross_validate_random_forest(X, y, cv=k)
        avg_rmse = rmse_scores.mean()
        rmse_results[k] = avg_rmse

        if avg_rmse < lowest_rmse:
            lowest_rmse = avg_rmse
            best_k = k

    print(f"Best k for Random Forest: {best_k} with RMSE: {lowest_rmse:.2f}")
    return best_k, rmse_results

def main():
    # Data from app.py
    raw_data = preprocess_data()

    # Preprocess data for casual rentals
    training_set_casual, test_set_casual, train_target_casual, test_target_casual = preprocess_for_model(raw_data, 'casual')
    
    # Find the best k for OLS and Random Forest on Casual Rentals
    best_k_ols_casual, ols_rmse_results_casual = find_best_k_for_ols(training_set_casual, train_target_casual)
    best_k_rf_casual, rf_rmse_results_casual = find_best_k_for_rf(training_set_casual, train_target_casual)
    
    # Perform final evaluation for Casual Rentals with the best k
    print("\nFinal Evaluation for Casual Rentals using best k values:")
    print(f"OLS Casual Rentals - Best k: {best_k_ols_casual}, Average RMSE: {ols_rmse_results_casual[best_k_ols_casual]:.2f}")
    print(f"Random Forest Casual Rentals - Best k: {best_k_rf_casual}, Average RMSE: {rf_rmse_results_casual[best_k_rf_casual]:.2f}")
    
    # Preprocess data for registered rentals
    training_set_registered, test_set_registered, train_target_registered, test_target_registered = preprocess_for_model(raw_data, 'registered')
    
    # Find the best k for OLS and Random Forest on Registered Rentals
    best_k_ols_registered, ols_rmse_results_registered = find_best_k_for_ols(training_set_registered, train_target_registered)
    best_k_rf_registered, rf_rmse_results_registered = find_best_k_for_rf(training_set_registered, train_target_registered)
    
    # Perform final evaluation for Registered Rentals with the best k
    print("\nFinal Evaluation for Registered Rentals using best k values:")
    print(f"OLS Registered Rentals - Best k: {best_k_ols_registered}, Average RMSE: {ols_rmse_results_registered[best_k_ols_registered]:.2f}")
    print(f"Random Forest Registered Rentals - Best k: {best_k_rf_registered}, Average RMSE: {rf_rmse_results_registered[best_k_rf_registered]:.2f}")

if __name__ == "__main__":
    main()