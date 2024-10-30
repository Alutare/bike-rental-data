import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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

def evaluate_random_forest(model, X, y):

    rf_predictions = model.predict(X)
    
    # Calculate RMSE
    rmse_rf = np.sqrt(mean_squared_error(y, rf_predictions))
    
    # Calculate R²
    r2_rf = r2_score(y, rf_predictions)
    
    return rf_predictions, rmse_rf, r2_rf

def plot_feature_importance(model, feature_names):
    # Calculate feature importance
    feature_importance = pd.Series(model.feature_importances_, index=feature_names)
    
    # Sort the feature importance
    feature_importance_sorted = feature_importance.sort_values(ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feature_importance_sorted.plot(kind='barh')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

def main():
    # Data from app.py
    raw_data = preprocess_data()

    # Preprocess and train OLS model for casual rentals
    training_set_casual, test_set_casual, train_target_casual, test_target_casual = preprocess_for_model(raw_data, 'casual')
    model_casual = train_ols_model(training_set_casual, train_target_casual)
    rmse_casual = calculate_rmse(model_casual, test_set_casual, test_target_casual)

    # Print the summary of the casual model and RMSE
    print("Casual Rentals Model Summary:")
    print(model_casual.summary())
    print(f"Casual Rentals RMSE: {rmse_casual:.2f}")

    # Fit and evaluate Random Forest for casual rentals
    rf_casual = train_random_forest(training_set_casual, train_target_casual)
    rf_predictions_casual, rmse_rf_casual, r2_rf_casual = evaluate_random_forest(rf_casual, test_set_casual, test_target_casual)
    print(f"Random Forest Casual Rentals RMSE: {rmse_rf_casual:.2f}")
    print(f"Random Forest Casual Rentals R²: {r2_rf_casual:.2f}")

    # Plot feature importance for casual rentals
    plot_feature_importance(rf_casual, training_set_casual.columns)

    # Preprocess and train OLS model for registered rentals
    training_set_registered, test_set_registered, train_target_registered, test_target_registered = preprocess_for_model(raw_data, 'registered')
    model_registered = train_ols_model(training_set_registered, train_target_registered)
    rmse_registered = calculate_rmse(model_registered, test_set_registered, test_target_registered)

    # Print the summary of the registered model and RMSE
    print("\nRegistered Rentals Model Summary:")
    print(model_registered.summary())
    print(f"Registered Rentals RMSE: {rmse_registered:.2f}")

    # Fit and evaluate Random Forest for registered rentals
    rf_registered = train_random_forest(training_set_registered, train_target_registered)
    rf_predictions_registered, rmse_rf_registered, r2_rf_registered = evaluate_random_forest(rf_registered, test_set_registered, test_target_registered)
    print(f"Random Forest Registered Rentals RMSE: {rmse_rf_registered:.2f}")
    print(f"Random Forest Registered Rentals R²: {r2_rf_registered:.2f}")

    # Plot feature importance for registered rentals
    plot_feature_importance(rf_registered, training_set_registered.columns)

if __name__ == "__main__":
    main()