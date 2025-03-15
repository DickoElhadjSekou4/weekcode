import shap
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data():
    """Divise les données en X (features) et y (target), puis les sépare en train/test."""
    df = get_data_cleaned()

    # Définition des features (X) et de la variable cible (y)
    X = df.drop(columns=["Biopsy"])  # Variable cible = Biopsy
    y = df["Biopsy"]

    # Division des données en train et test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
    
# Fonction pour entraîner le modèle XGBoost
def train_model(X_train, y_train):
    """Entraîne un XGBClassifier sur les données."""
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Fonction pour évaluer le modèle
def evaluate_model(model, X_test, y_test):
    """Évalue le modèle en affichant la précision et le rapport de classification."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Fonction pour expliquer les prédictions avec SHAP
def explain_model(model, X_train):
    """Explique les décisions du modèle avec SHAP."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    # Affichage des features les plus importantes
    shap.summary_plot(shap_values, X_train)

# Exécution complète du pipeline 
X_train, X_test, y_train, y_test = prepare_data()  # Préparation des données
model = train_model(X_train, y_train)  # Entraînement du modèle
evaluate_model(model, X_test, y_test)  # Évaluation du modèle
explain_model(model, X_train)  # Explication avec SHAP
