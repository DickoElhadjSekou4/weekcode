import pandas as pd #pandas est une bibliothèque utilisée pour manipuler des données sous forme de tableaux (DataFrame).
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataloader import get_downloaded_dataset

df = get_downloaded_dataset()

print('Afficher les informations générales sur le dataset')
print(df.info())
print('Convertir les colonnes object en nombres (ignorer les erreurs)')
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = pd.to_numeric(df[col], errors="coerce")

print('Vérifier si la conversion a fonctionné')

print(df.dtypes)
print(df.isnull().sum())# donne les nombre de NAN pqr colone 

df.drop(columns=["STDs: Time since first diagnosis", "STDs: Time since last diagnosis"], inplace=True) #Suppri,er les colones qui ont beaucoup de NAN
df.fillna(df.median(), inplace=True)
print('Vérifie sil reste des NaN')
print(df.isnull().sum())  
print('Vérifie les types de données')
print(df.info())  
print('  Vérifie les statistiques des colonne')
print(df.describe())
# Histogramme de l'âge
plt.figure(figsize=(16, 15))
sns.histplot(df['Age'], bins=20, kde=True, color='blue')
plt.title("Distribution de l'Âge")
plt.xlabel("Âge")
plt.ylabel("Nombre de personnes")
plt.show()

# Heatmap des corrélations
plt.figure(figsize=(16, 15))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de Corrélation")
plt.show()

# Boxplot du nombre de partenaires sexuels
plt.figure(figsize=(12, 11))
sns.boxplot(x=df['Number of sexual partners'])
plt.title("Boxplot - Nombre de partenaires sexuels")
plt.show()

# Nuage de points entre l'âge et le nombre de partenaires sexuels
plt.figure(figsize=(15, 13))
sns.scatterplot(x=df['Age'], y=df['Number of sexual partners'], alpha=0.5)
plt.title("Relation entre l'âge et le nombre de partenaires sexuels")
plt.xlabel("Âge")
plt.ylabel("Nombre de partenaires sexuels")
plt.show()