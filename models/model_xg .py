from sklearn.svm import SVC

def train_svm(X_train, y_train):
    """ Entraîner un modèle SVM avec class_weight ajusté """
    svm = SVC(kernel='linear', class_weight={0: 1, 1: 10})
    svm.fit(X_train, y_train)
    return svm
