import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None

def map_hour_to_interval(hr):
    if hr in range(0, 6):
        return 'Night'
    elif hr in range(6, 12):
        return 'Morning'
    elif hr in range(12, 18):
        return 'Afternoon'
    else:
        return 'Evening'

def map_dayofweek_to_weekdayorweekend(weekday):
    if weekday == 6 or weekday == 0:
        return 'Weekend'
    else:
        return 'Weekday'

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def preprocess_data():
    # Load Dataset, Change File Path Accordingly
    raw_data = pd.read_csv('C:/Users/e1121605/Desktop/Kevin Chan/section 2/bike_sharing.csv')
    
    # Convert categorical columns to category dtype
    categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
    for col in categorical_cols:
        raw_data[col] = raw_data[col].astype('category')

    # Convert 'dteday' to datetime
    raw_data['dteday'] = pd.to_datetime(raw_data['dteday'])

    # Drop unnecessary columns
    raw_data = raw_data.drop(["instant"], axis=1)

    # Map hour to time intervals
    raw_data['time_interval'] = raw_data['hr'].apply(map_hour_to_interval)
    raw_data['time_interval'] = raw_data['time_interval'].astype('category')

    # Map weekday to weekend or weekday
    raw_data['weekend'] = raw_data['weekday'].apply(map_dayofweek_to_weekdayorweekend)
    raw_data['weekend'] = raw_data['weekend'].astype('category')

    # Drop more unnecessary columns
    raw_data = raw_data.drop(["mnth", "hr", "dteday", "weekday", "atemp"], axis=1)

    # Remove outliers for specified numerical columns
    numerical_columns = ['temp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    outlier_raw_data = raw_data.copy()

    for column in numerical_columns:
        outlier_raw_data = remove_outliers(outlier_raw_data, column)

    return outlier_raw_data  # Return the processed data

def main():
    processed_data = preprocess_data()  # Call the function to process data
    print(processed_data.head())  # Check the processed data

if __name__ == "__main__":
    main()
