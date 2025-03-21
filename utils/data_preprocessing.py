import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def get_data():
    from sklearn.impute import SimpleImputer
    cervical_cancer_risk_factors = fetch_ucirepo(id=383) 
    df = cervical_cancer_risk_factors.data.features 
    df = df.dropna(thresh=len(df)*0.5, axis=1)
    # Remplacer les valeurs manquantes restantes par des médianes pour les colonnes numériques
    imputer = SimpleImputer(strategy="median")
    df.iloc[:, :] = imputer.fit_transform(df)
    return df

def get_data_origninal():
    df=get_data()
    cols_to_drop = ["Smokes (years)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs: Number of diagnosis", "Dx"]
    df= df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df

def load_and_clean_data():
    df= get_data()
    df=get_data_original()
    return df

import pandas as pd

def optimize_memory(df):
    
    for col in df.columns:  # Parcours toutes les colonnes du DataFrame
        col_type = df[col].dtype  # Récupère le type de la colonne
        
        # Optimisation des entiers (int64 → int32 ou plus petit)
        if col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')  
        
        # Optimisation des flottants (float64 → float32)
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')  

    return df  # Retourne le DataFrame optimisé


   


    
