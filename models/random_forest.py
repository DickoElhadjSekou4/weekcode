def train_model(X, y):
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

    # Prédiction pour un patient spécifique
    st.markdown("###  Résultat de la prédiction :")

    # Sélection du patient à analyser
    patient_index = 0  # Choix de l'index du patient à analyser, ici le premier dans l'ensemble de test
    patient_data = X_test.iloc[patient_index:patient_index+1]  # Données du patient
    patient_pred = clf.predict(patient_data)[0]  # Prédiction pour ce patient

    # Affichage du résultat
    if patient_pred == 1:
        st.markdown("**Le modèle prédit que le patient est atteint du cancer cervical.** :warning:")
    else:
        st.markdown("**Le modèle prédit que le patient n'est pas atteint du cancer cervical.** :white_check_mark:")

