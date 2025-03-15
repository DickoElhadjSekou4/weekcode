import pandas as pd
from utils.data_preprocessing import load_and_clean_data
from models.random_forest import train_model
from models.random_forest import evaluate_model
from shap_analysis import analyze_shap

# Charger les données
df = load_and_clean_data()

# Prétraiter les données
X_train, X_test, y_train, y_test = preprocess_data(df)

# Entraîner le modèle
train_random_forest = train_model(X_train, y_train)

# Évaluer le modèle
evaluate_model(svm_model, X_test, y_test)

# Analyse SHAP
analyze_shap(svm_model, X_train, X_test, feature_names=df.drop(columns=['Biopsy']).columns)
