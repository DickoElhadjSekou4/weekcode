import pandas as pd
from utils.data_preprocessing import load_and_clean_data
from models.random_forest import train_model
from models.random_forest import evaluate_model
from models.random_forest import explain_model

# Charger les données
df = load_and_clean_data()

# Prétraiter les données
X_train, X_test, y_train, y_test = prepa_data(df)

# Entraîner le modèle
train_random_forest = train_model(X_train, y_train)

# Évaluer le modèle
evaluate_model(model, X_test, y_test)

# Analyse SHAP
explain_model(model, X_train)
