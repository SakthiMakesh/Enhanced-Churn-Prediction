# cc_prediction.py

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

def preprocess_data(raw_data):
    non_numeric_cols = raw_data.select_dtypes(include=['object']).columns.tolist()
    raw_data_numeric = raw_data.drop(non_numeric_cols, axis=1)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(raw_data_numeric)
    raw_data_numeric_imputed = pd.DataFrame(imp.fit_transform(raw_data_numeric), columns=raw_data_numeric.columns)

    raw_data_imputed = pd.concat([raw_data[non_numeric_cols], raw_data_numeric_imputed], axis=1)

    le = LabelEncoder()
    raw_data_imputed['Churn'] = le.fit_transform(raw_data_imputed['Churn'].astype(str))

    df = pd.DataFrame(raw_data_imputed)

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype("category").cat.codes.astype("float")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    processed_data = df.copy()
    processed_data['cluster'] = clusters

    return processed_data

def train_models(X_train_scaled, y_train_smote):
    lr = LogisticRegression(max_iter=1000)
    svm = SVC(kernel='linear', probability=True, random_state=0)
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    bagging = BaggingClassifier(RandomForestClassifier(n_estimators=100, random_state=0),n_estimators=10, random_state=0)

    sm = SMOTE(random_state=0)
    X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train_smote)

    models = [('Logistic Regression', lr),
              ('Support Vector Machine', svm),
              ('Random Forest', rf),
              ('Bagging Classifier', bagging)]

    results = []
    for name, model in models:
        model.fit(X_train_smote, y_train_smote)
        results.append((name, model))

    return results

def evaluate_models(models, X_test_scaled, y_test):
    evaluation_results = []
    for name, model in models:
        y_pred = model.predict(X_test_scaled)
        evaluation_result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }
        evaluation_results.append(evaluation_result)

    return evaluation_results

def write_results_to_file(best_model_name, processed_data):
    result_text = ""
    result_text += f"Best Model: {best_model_name}\n"
    result_text += f"Total Customers: {len(processed_data)}\n"
    result_text += f"Churn Customers: {len(processed_data[processed_data['Churn'] == 1])}\n"
    result_text += f"Non-Churn Customers: {len(processed_data[processed_data['Churn'] == 0])}\n"

    return result_text

def main(dataset_file):
    raw_data = pd.read_csv(dataset_file)
    processed_data = preprocess_data(raw_data)
    
    X = processed_data.drop(['Churn', 'cluster'], axis=1)
    y = processed_data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    models = train_models(X_train_scaled, y_train)
    evaluation_results = evaluate_models(models, X_test_scaled, y_test)
    best_model = max(evaluation_results, key=lambda x: x["F1-Score"])
    best_model_name = best_model["Model"]

    result_text = write_results_to_file(best_model_name, processed_data)
    return result_text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cc_prediction.py <dataset_file>")
        exit()
    
    dataset_file = sys.argv[1]
    main(dataset_file)
