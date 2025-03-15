import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_clean_data(file_path):
    """ Charger et nettoyer les données """
    df = pd.read_csv(file_path)

    # Convertir les colonnes "object" en nombres
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Supprimer les colonnes fortement corrélées
    df.drop(columns=["Smokes (years)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs: Number of diagnosis", "Dx"], inplace=True)

    # Remplacer les valeurs NaN par la médiane
    df.fillna(df.median(), inplace=True)

    return df

def preprocess_data(df, n_components=5):
    """ Normalisation et réduction dimensionnelle (PCA) """
    X = df.drop(columns=['Biopsy'])  
    y = df['Biopsy']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y

def clean_dataset(file_path):
    """ Charger et nettoyer les données """
    df = pd.read_csv(file_path)
    
    # Convertir les colonnes "object" en nombres
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Supprimer les colonnes fortement corrélées
    df.drop(columns=["Smokes (years)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs: Number of diagnosis", "Dx"], inplace=True)
    
    # Remplacer les valeurs NaN par la médiane
    df.fillna(df.median(), inplace=True)
    
    return df

def transform_data(df):
    """ Normalisation et réduction de dimension des données """
    X = df.drop(columns=['Biopsy'])  
    y = df['Biopsy']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    
    return train_test_split(X_pca, y, test_size=0.2, random_state=42)