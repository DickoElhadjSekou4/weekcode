import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Fonction pour charger et visualiser les données
def charger_donnees(chemin_fichier):
    """Charge le fichier CSV et affiche les colonnes disponibles."""
    df = pd.read_csv(chemin_fichier)
    print("=== Colonnes disponibles ===")
    print(df.columns)
    return df

# Fonction pour remplacer les valeurs manquantes ("?") et supprimer des colonnes
def pretraitement(df):
    """Remplace les '?' par NaN, supprime les colonnes avec >50% de NaN, et impute les valeurs manquantes restantes."""
    df.replace("?", np.nan, inplace=True)
    
    # Suppression des colonnes avec plus de 50% de valeurs manquantes
    seuil_colonnes = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=seuil_colonnes)
    
    # Imputation des colonnes numériques par la moyenne
    colonnes_numeriques = df.select_dtypes(include=['float64', 'int64']).columns
    df[colonnes_numeriques] = df[colonnes_numeriques].fillna(df[colonnes_numeriques].mean())
    
    # Imputation des colonnes catégorielles par le mode
    colonnes_categorielles = df.select_dtypes(include=['object']).columns
    df[colonnes_categorielles] = df[colonnes_categorielles].fillna(df[colonnes_categorielles].mode().iloc[0])
    
    return df

# Fonction pour vérifier et séparer les caractéristiques et la cible
def separer_features_target(df, target_colonne):
    """Sépare les caractéristiques (X) et la cible (y) après vérification de la colonne cible."""
    if target_colonne not in df.columns:
        raise ValueError(f"La colonne cible '{target_colonne}' n'existe pas dans le DataFrame.")
    
    X = df.drop(target_colonne, axis=1)
    y = df[target_colonne]
    
    # Encodage des variables catégorielles
    X = pd.get_dummies(X, drop_first=True)
    return X, y

# Fonction pour diviser les données en ensembles d'entraînement et de test
def diviser_ensemble(X, y, test_size=0.2, random_state=42):
    """Divise les données en ensembles d'entraînement et de test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Fonction pour entraîner le modèle XGBoost
def entrainer_xgboost(X_train, y_train, num_round=100):
    """Entraîne un modèle XGBoost sur les données d'entraînement."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    param = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    model = xgb.train(param, dtrain, num_round)
    return model

# Fonction pour faire des prédictions et évaluer le modèle
def evaluer_modele(model, X_test, y_test):
    """Évalue le modèle XGBoost avec des prédictions et des métriques de performance."""
    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = model.predict(dtest)
    predictions = [1 if value > 0.5 else 0 for value in preds]
    
    # Calculer l'exactitude
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nExactitude du modèle : {accuracy:.2f}")
    
    # Rapport de classification
    print("\n=== Rapport de classification ===")
    print(classification_report(y_test, predictions))
    
    # Matrice de confusion
    print("\n=== Matrice de confusion ===")
    print(confusion_matrix(y_test, predictions))
    
    # Afficher l'importance des caractéristiques
    xgb.plot_importance(model)
    plt.title("Importance des caractéristiques")
    plt.show()

# Fonction principale pour exécuter le processus complet
def main():
    chemin_fichier = r"C:\Users\dontr\OneDrive\Bureau\Cours 1A\SEMESTER 2\Coding week\risk_factors_cervical_cancer.csv"
    target_colonne = 'Biopsy'  # Remplacez par le nom exact de la colonne cible
    
    # Étapes du pipeline
    df = charger_donnees(chemin_fichier)
    df = pretraitement(df)
    X, y = separer_features_target(df, target_colonne)
    X_train, X_test, y_train, y_test = diviser_ensemble(X, y)
    model = entrainer_xgboost(X_train, y_train)
    evaluer_modele(model, X_test, y_test)

# Exécuter le programme principal
if __name__ == "__main__":
    main()
