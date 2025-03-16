import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_svm(X_train, y_train, class_weight={0: 1, 1: 10}):
    """ Entraîner un modèle SVM """
    model = SVC(kernel='linear', class_weight=class_weight)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """ Évaluer le modèle avec les métriques courantes """
    y_pred = model.predict(X_test)


    print(" Rapport de classification :\n", classification_report(y_test, y_pred))

    print("Rapport de classification :\n", classification_report(y_test, y_pred))

    print(f" Précision du modèle : {accuracy_score(y_test, y_pred):.2f}")
    print(f" ROC-AUC Score : {roc_auc_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    plt.show()

def save_model(model, path="models/svm_model.pkl"):
    """ Sauvegarder le modèle """
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f" Modèle sauvegardé sous {path}")

def load_model(path="models/svm_model.pkl"):
    """ Charger un modèle sauvegardé """
    with open(path, "rb") as f:
        return pickle.load(f)
