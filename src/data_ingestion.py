import pandas as pd
import os

def load_data(input_path):
    """
    Loads the raw CSV data into a DataFrame.
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Data shape: {df.shape}")
    return df

def validate_columns(df, expected_columns):
    """
    Checks if all expected columns are in the DataFrame.
    """
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    print("All expected columns are present.")

def clean_data(df):
    """
    Cleans the data (basic example: drop nulls).
    """
    print(f"Before cleaning: {df.shape}")
    df = df.dropna()
    print(f"After cleaning: {df.shape}")
    return df

def save_data(df, output_path):
    """
    Saves the DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")

def main():
    raw_path = r"../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_path = r"../data/processed/cleaned_telco_churn.csv"

    expected_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    df = load_data(raw_path)
    validate_columns(df, expected_columns)
    df = clean_data(df)
    save_data(df, processed_path)

if __name__ == "__main__":
    main()
