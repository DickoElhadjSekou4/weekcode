import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve
import xgboost as xgb
import shap

def charger_donnees(chemin_fichier):
    """Charger les données à partir d'un fichier CSV."""
    df = pd.read_csv(chemin_fichier)
    print("Colonnes disponibles :", df.columns)
    return df

def pretraiter_donnees(df):
    """Prétraiter les données : remplacer les '?' par des NaN, supprimer les colonnes avec plus de 50 % de valeurs manquantes, et imputer les valeurs manquantes."""
    df.replace("?", np.nan, inplace=True)
    seuil_colonnes = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=seuil_colonnes)
    
    colonnes_numeriques = df.select_dtypes(include=['float64', 'int64']).columns
    df[colonnes_numeriques] = df[colonnes_numeriques].fillna(df[colonnes_numeriques].mean())
    
    colonnes_categorielles = df.select_dtypes(include=['object']).columns
    df[colonnes_categorielles] = df[colonnes_categorielles].fillna(df[colonnes_categorielles].mode().iloc[0])
    
    return df

def verifier_colonne_cible(df, target_colonne):
    """Vérifier si la colonne cible existe dans le DataFrame."""
    if target_colonne not in df.columns:
        raise ValueError(f"La colonne cible '{target_colonne}' n'existe pas dans le DataFrame.")
    return df[target_colonne]

def separer_caracteristiques_et_cible(df, target_colonne):
    """Séparer les caractéristiques et la cible."""
    X = df.drop(target_colonne, axis=1)
    y = df[target_colonne]
    return X, y

def encoder_variables_categorielles(X):
    """Encoder les variables catégorielles."""
    return pd.get_dummies(X, drop_first=True)

def diviser_donnees(X, y, test_size=0.2, random_state=42):
    """Diviser les données en ensembles d'entraînement et de test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def entrainer_modele_xgboost(X_train, y_train, param, num_round=100):
    """Entraîner un modèle XGBoost."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(param, dtrain, num_round)
    return model

def evaluer_modele(model, X_test, y_test):
    """Faire des prédictions et évaluer le modèle."""
    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = model.predict(dtest)
    predictions = [1 if value > 0.5 else 0 for value in preds]
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, preds)
    
    print(f"Exactitude du modèle : {accuracy:.2f}")
    print(f"Précision : {precision:.2f}")
    print(f"Rappel : {recall:.2f}")
    print(f"Score F1 : {f1:.2f}")
    print(f"ROC-AUC : {roc_auc:.2f}")
    
    # Tracer la courbe ROC
    fpr, tpr, _ = roc_curve(y_test, preds)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (aire = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    # Rapport de classification détaillé
    print("\nRapport de classification :")
    print(classification_report(y_test, predictions))

def generer_graphiques_shap(model, X_test):
    """Générer des graphiques récapitulatifs SHAP pour interpréter le modèle."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values.values, X_test)

def main():
    chemin_fichier = r"C:\Users\dontr\OneDrive\Bureau\Cours 1A\SEMESTER 2\Coding week\risk_factors_cervical_cancer.csv"
    target_colonne = 'Biopsy'
    
    df = charger_donnees(chemin_fichier)
    df = pretraiter_donnees(df)
    y = verifier_colonne_cible(df, target_colonne)
    X, y = separer_caracteristiques_et_cible(df, target_colonne)
    X = encoder_variables_categorielles(X)
    X_train, X_test, y_train, y_test = diviser_donnees(X, y)
    
    param = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    model = entrainer_modele_xgboost(X_train, y_train, param)
    evaluer_modele(model, X_test, y_test)
    generer_graphiques_shap(model, X_test)

if __name__ == "__main__":
    main()