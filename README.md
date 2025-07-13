# customer-churn-ml-pipeline
An automated ML pipeline for customer churn prediction using the Telco dataset.
## **Project Overview**
- Predict whether a customer will churn (leave the service).
- Kaggle's Telco Customer Churn dataset.
- Automated pipeline built in **Python**.
- Production-style, reproducible, modular code.

---

## **Features**
- Data ingestion from raw Excel/CSV
- Cleaning & processing
- Feature engineering (encoding, scaling)
- Train/test split
- Model training (RandomForestClassifier)
- Model evaluation (accuracy, classification report)
- Model saving for deployment

---
## **Data Source**
- Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## ⚙**Pipeline Steps**
1. **Data Ingestion**
   - Loads raw Excel/CSV data
   - Validates schema
   - Cleans and saves to `data/processed/`
   
2. **Preprocessing**
   - Loads cleaned data
   - Converts data types
   - Encodes categoricals
   - Scales numericals
   - Splits train/test
   - Saves to `data/features/`
   
3. **Model Training**
   - Loads features
   - Trains RandomForestClassifier
   - Evaluates on test set
   - Saves trained model to `models/`

---
