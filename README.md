## Bike Sharing Demand Prediction

This project aims to predict bike sharing demand using machine learning models. The project involves performing Exploratory Data Analysis (EDA) and building an end-to-end machine learning pipeline. The dataset used for this project is the `bike_sharing.csv`, and the project covers data ingestion, preprocessing, feature engineering and model evaluation

## Project Overview

In this project, two machine learning models (Linear Regression and Random Forests) are evaluated to predict bike-sharing demand, focusing on the following key components:
1. Data ingestion and EDA using Jupyter Notebook.
2. Data preprocessing, feature engineering, and model building in Python scripts.
3. Model evaluation using RMSE metric

## Dataset

The dataset `bike_sharing.csv` contains historical bike-sharing data. The data dictionary explaining the variables can be found in `data_details.txt`.

**Target variable**: Number of bike rentals (`cnt`), which is comprised of summing (`casual`) and (`registered`) users

Casual and Registered users may have different behaviors and patterns. Training separate models for casual and registered rentals allows for more accurate predictions, a better understanding of user behavior, tailored marketing strategies, and the ability to manage and analyze each segment more effectively.

## Feature Engineering

**Manipulation of Predictor Variables**
Converted (`hr`) based on Morning, Afternoon, Evening, Night. Removed (`hr`) and replaced with (`time_interval`)

Converted (`weekday`) to 2 groups, either Weekday or Weekend. Removed (`weekday`) and replaced with (`weekend`)

**Removal of Variables**
Removed (`instant`) - Irrelevant as a predictor 

Removed (`dteday`) - Too many unique entries, irrelevant as a predictor as time is captured in other variables already

Removed (`mnth`) - Already clustered using (`seasons`) variable

Removed (`atemp`) - High collinearity with (`temp`)

Removed (`workingday`) - High collinearity with new (`weekday`) variable


## Key Observations after EDA

<Categorical Variables>
# Low rentals for Season 1 (Spring Season)
# Higher total rentals (Registered and Casual) in Year 2012 
# Higher Registered rentals on non-holidays
# Higher registered rentals on working days
# Cnt drops for both casual and registered as weather becomes worse from 1-4
# Cnt is highest in Afternoon (12pm - 6pm)
# Higher Casual rentals on weekend, but higher registered rentals on weekdays

<Numerical Variables>
# Higher Casual and Registered rentals as temperature increases
# Higher Casual and Registered rentals at lower humidity
# Cnt are quite similar for all the windspeed ranges

## Results of Model evaluation

Linear Regression and Random Forests were used to predict both (`casual`) and (`registered`) as the target variables

Models on (`casual`) rentals:
OLS R²: 0.677 
OLS RMSE: 20.30 
Random Forest R²: 0.70 
Random Forest RMSE: 14.45

Models on (`registered`) rentals:
OLS R²: 0.673
OLS RMSE: 78.10
Random Forest R²: 0.55
Random Forest RMSE: 70.08

## Results of 10-fold Cross Validation
Final Evaluation for Casual Rentals using best k values:
Best k for OLS: 8 with RMSE: 19.35
Best k for Random Forest: 9 with RMSE: 14.79

Final Evaluation for Registered Rentals using best k values:
Best k for OLS: 2 with RMSE: 77.48
Best k for Random Forest: 10 with RMSE: 71.62

## Conclusion
For (`casual`) rentals:
RF Model has a lower RMSE (14.79) and slightly better R²

For (`registered`) rentals:
RF Model has a lower RMSE (71.62) However, it has lower R², indicating that the linear model explains more variance and has better goodness of fit but is slight less accurate

Possible Improvements:
1) Hyperparameter tuning, grid search to search for the best hyperparameters for the model to attain better results.
2) Comparision with more models
3) Alternative Feature Engineering 


