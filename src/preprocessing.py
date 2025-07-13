import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_clean_data(path):
    print(f"Loading cleaned data from: {path}")
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    print(f"Data shape after cleaning: {df.shape}")
    return df

def split_X_y(df, target_column='Churn'):
    print(f"Splitting features and target: {target_column}")
    X = df.drop(columns=[target_column, 'customerID'])
    y = df[target_column].apply(lambda x: 1 if x == 'Yes' else 0)
    return X, y

def preprocess_features(X):
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    print(f"Categorical columns: {list(cat_cols)}")
    print(f"Numerical columns: {list(num_cols)}")

    # OneHotEncode categoricals
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(X[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    # Scale numerics
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[num_cols])

    # Combine
    X_processed = np.hstack([X_num, X_cat])
    feature_names = list(num_cols) + list(cat_feature_names)

    X_df = pd.DataFrame(X_processed, columns=feature_names)
    print(f"Processed feature shape: {X_df.shape}")
    return X_df

def save_splits(X_train, X_test, y_train, y_test, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    X_train.to_csv(os.path.join(output_folder, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_folder, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_folder, "y_test.csv"), index=False)
    print(f"Saved splits to {output_folder}")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, "..", "data", "processed", "cleaned_telco_churn.csv")
    output_folder = os.path.join(BASE_DIR, "..", "data", "features")

    df = load_clean_data(input_path)
    X, y = split_X_y(df)
    X_processed = preprocess_features(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    save_splits(X_train, X_test, y_train, y_test, output_folder)

if __name__ == "__main__":
    main()