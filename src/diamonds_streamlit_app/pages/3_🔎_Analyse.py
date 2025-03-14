import streamlit as st
import pandas as pd
from layouts.footer import footer
from layouts.header import header
from layouts.data import get_data
from flask import Flask, render_template, request


# Fonction pour récupérer dynamiquement les features
def get_features_from_data():
    df = get_data()  # Récupérer la base de données
    return df.dtypes.to_dict()  # Extraire les noms des colonnes

# Fonction pour collecter les données du patient
def collect_patient_data():
    st.title("Formulaire Patient - Facteurs de Risque du Cancer du Col")
    st.write("Veuillez entrer vos informations dans les champs ci-dessous.")

    # Récupérer dynamiquement les features depuis les données
    feature_names = get_features_from_data()

    # Dictionnaire pour stocker les réponses
    patient_data = {}

    # Création dynamique des champs d'entrée
    for feature, dtype in feature_names.items():
        if dtype == "int64":  # Si le type est int
            patient_data[feature] = st.number_input(f"{feature}:", min_value=0, value=0, step=1)
        elif dtype == "float64":  # Si le type est float
            patient_data[feature] = st.number_input(f"{feature}:", min_value=0.0, value=0.00, step=0.01)
        else:  # Par défaut, gérer comme un texte (si d'autres types existent)
            patient_data[feature] = st.text_input(f"{feature}:")

    # Bouton pour enregistrer les informations
    if st.button("Enregistrer les informations"):
        st.success("✅ Données enregistrées avec succès !")
        st.dataframe(pd.DataFrame([patient_data]))  # Afficher les données sous forme de tableau

# Lancer l'application
if __name__ == "_main_":
    collect_patient_data()



def analyse():
    data = get_data()

    select = st.sidebar.selectbox("Select", ["Head", "Tail", "Shape", "Type", "Isnull", "Describe", "DataPatient"])
    
    if select == "Head":
         st.write("Head", data.head())
    elif select == "Tail":
        st.write("Tail: ", data.tail())
    elif select == "Shape":
        st.write("Shape:", data.shape)
    elif select == "Type":
        st.write("Type:", data.dtypes.to_frame("Types"))
    elif select == "Isnull":
        st.write("Isnull:", data.isnull().sum().to_frame("Null"))
    elif select == "Describe":
        st.write("Describe: ", data.describe())
    elif select== "DataPatient" :
        patient_data= collect_patient_data()
        st.write("DataPatient", patient_data)
   

def main():
    header()
    analyse()
    footer()

if __name__ == "__main__":
    main()