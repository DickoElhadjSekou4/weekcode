import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def train_svm(X_train, y_train):
    """Entraîne un modèle SVM avec un noyau linéaire."""
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(svm, X_test, y_test):
    """Fait des prédictions et évalue les performances du modèle."""
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Afficher la matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    
    # Affichage de la frontière de décision
    plot_decision_boundary(X_train, X_test, y_train, y_test, svm)
    
    # Afficher les performances
    print("📊 Rapport de classification :\n", classification_report(y_test, y_pred))
    print(f"Précision du modèle : {accuracy:.2f}")
    print(f"🔹 ROC-AUC Score : {roc_auc_score(y_test, y_pred):.2f}")

def plot_decision_boundary(X_train, X_test, y_train, y_test, svm):
    """Affiche la frontière de décision du modèle SVM."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k', label='Entraînement')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k', marker='x', s=100, label='Test')
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Tracer la ligne de séparation et les marges
    plt.contour(xx, yy, Z, colors='black', levels=[-1, 0, 1], linestyles=['dashed', 'solid', 'dashed'])
    plt.legend()
    plt.title("SVM - Frontière de décision")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.show()

def print_class_distribution(df):
    """Affiche la répartition des classes après prétraitement."""
    print("\n📊 Répartition des classes après prétraitement :")
    print(df['Biopsy'].value_counts(normalize=True))

