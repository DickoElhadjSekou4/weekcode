import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from layouts.footer import footer
from layouts.header import header
from layouts.data import get_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

##################### Pré-traitement des données #########################

def data_preprocessing():
    df = get_data()

    # Supprimer les colonnes avec plus de 50% de valeurs manquantes
    df = df.dropna(thresh=len(df)*0.5, axis=1)

    # Remplacer les valeurs manquantes restantes par des médianes pour les colonnes numériques
    imputer = SimpleImputer(strategy="median")
    df.iloc[:, :] = imputer.fit_transform(df)

    # Séparation des données en X (features) et y (target)
    X = df.drop(columns=["Biopsy"]) 
    y = df["Biopsy"]

    return X, y


############################ Système de prédiction  #############################

def train_model():
    X, y = data_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Suréchantillonnage avec SMOTE
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Initialisation et entraînement du modèle Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_resampled, y_resampled)

    # Prédictions sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    return y_test, y_pred, accuracy, X_test, clf


############################ SHAP Waterfall Plot #############################

def make_shap_waterfall_plot(shap_values, features, num_display=None):
    '''
    Fonction pour créer un SHAP Waterfall Plot.
    Affiche l'impact des différentes caractéristiques de manière décroissante.

    Parameters:
    shap_values (list): Valeurs SHAP obtenues d'un modèle
    features (pandas DataFrame): Liste des caractéristiques utilisées dans le modèle
    num_display (int): Nombre de caractéristiques à afficher, sinon toutes les caractéristiques sont affichées

    Returns:
    matplotlib.pyplot plot: Graphique SHAP waterfall
    '''
    
    # Si num_display est None, nous affichons toutes les caractéristiques
    if num_display is None:
        num_display = len(features.columns)

    column_list = features.columns
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]
    
    num_height = 0
    if (num_display >= 20) & (len(column_list) >= 20):
        num_height = (len(column_list) - 20) * 0.4

    fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)

    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1) + 1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1) + 1, 10))
    ax1.set_xlabel('Cumulative Ratio')
    ax2.set_xlabel('Composition Ratio')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list))

    return fig

######################## Modélisation et prédiction pour le docteur ########################

def model():
    # Entraînement du modèle et récupération des résultats
    y_test, y_pred, accuracy, X_test, clf = train_model()

    # Affichage de l'accuracy en grand
    st.markdown(f"## 🔹 La Précision du modèle est de  : **{accuracy:.4f}**")

    # Décision à prendre en fonction de la précision
    if accuracy > 0.85:  # Exemple d'une condition arbitraire pour une bonne précision
        st.markdown("**Le modèle a une très bonne précision, continuez à suivre les recommandations.**")
    else:
        st.markdown("**Précision modérée, veuillez examiner d'autres facteurs avant de prendre une décision.**")

    # Sélection du patient à analyser
    st.markdown("###  Résultat de la prédiction :")
    patient_index = 0  # Choix de l'index du patient à analyser, ici le premier dans l'ensemble de test
    patient_data = X_test.iloc[patient_index:patient_index+1]  # Données du patient
    patient_pred = clf.predict(patient_data)[0]  # Prédiction pour ce patient

    # Obtenir les probabilités de prédiction
    proba = clf.predict_proba(patient_data)  # Prédictions en probabilité
    proba_positive = proba[0][1]  # Probabilité que le patient ait le cancer (classe 1)

    # Affichage des résultats en probabilité
    st.markdown(f"**Probabilité que le patient soit atteint du cancer cervical : {proba_positive * 100:.2f}%**")
    if patient_pred == 1:
        st.markdown("**Le modèle prédit que le patient est atteint du cancer cervical.** :warning:")
    else:
        st.markdown("**Le modèle prédit que le patient n'est pas atteint du cancer cervical.** :white_check_mark:")

    # Affichage du SHAP Waterfall Plot pour ce patient
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):  # Cas pour un modèle binaire (2 classes)
        shap_values_class_1 = shap_values[1]  # Classe positive (1)
    else:
        shap_values_class_1 = shap_values

    st.markdown("### 🔍 SHAP Waterfall Plot")
    fig = make_shap_waterfall_plot(shap_values_class_1[patient_index], X_test, num_display=len(X_test.columns))
    st.pyplot(fig)

    st.markdown("### 🔎 Caractéristiques importantes")
    st.write("Les caractéristiques les plus influentes dans cette prédiction sont :")
    st.write("- **Le sexe**")
    st.write("- **Le nombre de rapports sexuels**")

#################### Fonction principale pour le lancement ###################

def main():
    header()
    model()
    footer()

# La condition correcte pour exécuter le programme
if __name__ == "__main__":
    main()  # Appel de la fonction pour afficher les résultats
