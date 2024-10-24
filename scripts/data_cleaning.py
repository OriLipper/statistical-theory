# scripts/data_cleaning.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at the path: {filepath}")
        return None

def inspect_data(df):
    """Inspect the DataFrame for initial understanding."""
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nDescriptive Statistics:")
    print(df.describe())

def check_number_of_views(df):
    """Check the unique values in 'number of views'."""
    unique_values = df['number of views'].unique()
    print("\nUnique values in 'number of views':", unique_values)
    return unique_values

def handle_missing_values(df):
    """Handle missing values by imputing or removing."""
    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing Values Before Handling:")
    print(missing)
    
    # Impute numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numerical_cols:
        numerical_cols.remove('id')  # Assuming 'id' is unique and should be dropped
    
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"Filled missing values in {col} with median value {median}")
    
    # Impute categorical columns with mode (if any)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"Filled missing values in {col} with mode value '{mode}'")
    
    print("\nMissing Values After Handling:")
    print(df.isnull().sum())
    
    return df

def convert_date(df):
    """Convert serial date to datetime."""
    # Assuming 'Date' is an Excel serial date number
    # Excel's epoch starts on 1899-12-30
    excel_epoch = pd.Timestamp('1899-12-30')
    
    # Convert 'Date' to datetime
    df['Date'] = pd.to_timedelta(df['Date'], unit='D') + excel_epoch
    print("\nConverted 'Date' to datetime format.")
    
    return df

def drop_irrelevant_columns(df):
    """Drop columns that are not needed for analysis."""
    columns_to_drop = ['id', 'Date', 'Postal Code']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(f"\nDropped columns: {columns_to_drop}")
    return df

def encode_categorical_variables(df):
    """Encode categorical variables using one-hot encoding."""
    # Identify binary categorical variables
    binary_cols = ['waterfront present']
    
    # Identify ordinal categorical variables
    ordinal_cols = ['condition of the house', 'grade of the house']
    
    # No one-hot encoding needed for binary or ordinal if they are numerical
    print("\nEncoded categorical variables (if necessary).")
    
    return df

def handle_renovation_year(df):
    """Convert 'Renovation Year' 0 to NaN and handle missing values."""
    df['Renovation Year'] = df['Renovation Year'].replace(0, np.nan)
    # Fill missing 'Renovation Year' with 'Built Year' (assuming no renovation)
    df['Renovation Year'].fillna(df['Built Year'], inplace=True)
    print("\nHandled 'Renovation Year' by replacing 0 with 'Built Year'.")
    return df

def feature_engineering(df):
    """Create new features for better analysis."""
    current_year = pd.Timestamp.now().year
    
    # Age of the house
    df['House_Age'] = current_year - df['Built Year']
    
    # Time since renovation
    df['Time_Since_Renovation'] = current_year - df['Renovation Year']
    
    # Total Area
    df['Total_Area'] = df['living area'] + df['Area of the house(excluding basement)'] + df['Area of the basement']
    
    print("\nCreated new features: 'House_Age', 'Time_Since_Renovation', 'Total_Area'")
    return df

def handle_outliers(df, columns):
    """Cap outliers in specified columns using the IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        original_count = df[col].shape[0]
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        print(f"Capped outliers in {col} between {lower_bound} and {upper_bound}")
    return df

def scale_features(df, target):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    
    # Identify numerical features excluding the target
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numerical_cols:
        numerical_cols.remove(target)
    
    # Scale features
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("\nScaled numerical features using StandardScaler.")
    
    return df

def save_clean_data(df, filepath):
    """Save the cleaned dataframe to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"\nCleaned data saved to {filepath}")

def final_checks(df):
    """Perform final checks on the cleaned DataFrame."""
    print("\nFinal DataFrame Info:")
    print(df.info())
    
    print("\nFinal Descriptive Statistics:")
    print(df.describe())

def main():
    # Define file paths
    raw_data_path = os.path.join('data', 'House Price India.csv')
    clean_data_path = os.path.join('data', 'House Price India_clean.csv')
    
    # Load data
    df = load_data(raw_data_path)
    if df is None:
        return
    
    # Inspect data
    inspect_data(df)
    
    # Check 'number of views'
    unique_views = check_number_of_views(df)
    
    # Proceed based on the findings
    if len(unique_views) == 1 and unique_views[0] == 0:
        print("\n'number of views' is constant (0) across all records and will be dropped from analysis.")
        df = df.drop(columns=['number of views'])
    else:
        print("\n'number of views' has varying values and will be retained for analysis.")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Convert 'Date' to datetime
    df = convert_date(df)
    
    # Drop irrelevant columns
    df = drop_irrelevant_columns(df)
    
    # Handle 'Renovation Year'
    df = handle_renovation_year(df)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Handle outliers in numerical columns (optional)
    numerical_cols = ['number of bedrooms', 'number of bathrooms', 'living area', 'lot area',
                      'number of floors', 'condition of the house', 'grade of the house',
                      'Area of the house(excluding basement)', 'Area of the basement',
                      'Distance from the airport', 'House_Age', 'Time_Since_Renovation', 'Total_Area']
    df = handle_outliers(df, numerical_cols)
    
    # Encode categorical variables (if necessary)
    df = encode_categorical_variables(df)
    
    # Scale numerical features
    target = 'Price'
    df = scale_features(df, target)
    
    # Final checks
    final_checks(df)
    
    # Save cleaned data
    save_clean_data(df, clean_data_path)
    
if __name__ == "__main__":
    main()
