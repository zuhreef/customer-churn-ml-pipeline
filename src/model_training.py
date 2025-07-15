import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

def load_features(folder):
    print(f"Loading features from: {folder}")
    X_train = pd.read_csv(os.path.join(folder, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(folder, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(folder, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(folder, "y_test.csv")).values.ravel()
    print(f"Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training RandomForestClassifier with class_weight='balanced'...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

def plot_and_save_confusion_matrix(model, X_test, y_test, output_folder):
    print("Generating confusion matrix plot...")
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
        values_format='d'
    )
    plt.title("Confusion Matrix")
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    features_folder = os.path.join(BASE_DIR, "..", "data", "features")
    model_output = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")
    results_folder = os.path.join(BASE_DIR, "..", "results")

    X_train, X_test, y_train, y_test = load_features(features_folder)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_and_save_confusion_matrix(model, X_test, y_test, results_folder)
    save_model(model, model_output)

if __name__ == "__main__":
    main()
