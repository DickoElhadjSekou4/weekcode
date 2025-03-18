Prédiction du Cancer du Col de l'Utérus avec Machine Learning

    Description du Projet
Ce projet vise à prédire la présence d'un cancer du col de l'utérus à partir d'un ensemble de facteurs de risque en utilisant plusieurs modèles de Machine Learning. Nous avons comparé différentes approches afin d'identifier le modèle offrant les meilleures performances.

 Optimisation de l'utilisation de la mémoire : Pour l'optimisation de la memoire nous avons utiliser comme indiquer sur la fiche du projet une fonction optimize_memory aue nous avons appliqués aux dataset après l'avoir extrait ; 
 Nous obtenons:
 Utilisation mémoire avant optimisation :
1222985 octets
 Utilisation mémoire après optimisation :
1162925 octets
L’optimisation de la mémoire a permis de réduire l’utilisation de 1 005 060 octets à 962 180 octets, soit une réduction d’environ 5%.

Prétraitement des Données
•	Conversion des valeurs non numériques en nombres.
•	Suppression des colonnes fortement corrélées.
•	Remplacement des valeurs manquantes par la médiane.
•	Normalisation des données.
•	Réduction de dimension avec PCA pour la visualisation.

 
 Gestion du Déséquilibre des Classes
L'ensemble de données était déséquilibré : la classe 1 (biopsie positive) était sous-représentée par rapport à la classe 0 (biopsie négative). Ce déséquilibre a causé un problème, car le modèle prédisait majoritairement la classe 0 et n'apprenait pas correctement les caractéristiques de la classe minoritaire.
Pour corriger cela, nous avons utilisé class_weight={0: 1, 1: 10}, ce qui signifie que les erreurs sur la classe 1 coûtent 10 fois plus cher que celles sur la classe 0. Cela a permis au modèle de mieux détecter les cas positifs tout en maintenant un bon équilibre avec la classe majoritaire.

  Etude de corrélation
Après visualisation de la matrice de corrlation et en n'utilisant l'analyse ci dessous Corrélations élevées (proches de 1 ou -1) indiquent une forte relation entre deux variables.
Corrélations faibles (proches de 0) indiquent peu ou pas de relation.
:
  Stratégie de gestion des variables corrélées
1.	Suppression des variables redondantes
	Ex : Si "STDs (nombre)" est très corrélé avec plusieurs STDs individuels, on peut le supprimer pour éviter la redondance.
	Même chose pour "Smokes (years)" ou "Hormonal Contraceptives" si l’une des deux variables contient déjà assez d’information.
2.	Réduction dimensionnelle avec PCA (Analyse en Composantes Principales)
	Si beaucoup de corrélations fortes sont présentes, on peut utiliser PCA pour réduire le nombre de dimensions sans trop perdre d’information.
3.	Garder les variables corrélées si elles apportent une valeur différente
	Exemple : "STDs: HPV" et "Biopsy" peuvent rester même s’ils sont corrélés, car ils peuvent influencer différemment les modèles.

•	Corrélations importantes repérées : 
	"STDs (nombre)" et plusieurs types de STDs : Normal, car le nombre total de STDs dépend des sous-catégories.
	"Smokes (years)" et "Smokes (packs/year)" : Logique, car plus une personne fume longtemps, plus elle accumule de paquets.
	"Hormonal Contraceptives" et "Hormonal Contraceptives (years)" : Une personne prenant des contraceptifs a logiquement un nombre d’années associé. Après c'est Analyse nous nous sommes permis de suppri,er certaines colonnes


Évaluation des Performances
Nous avons évalué chaque modèle à l'aide des métriques suivantes :
•	ROC-AUC
•	Exactitude (Accuracy)
•	Précision (Precision)
•	Rappel (Recall)
•	Score F1
 
         Modèles Utilisés
Nous avons testé et comparé les modèles suivants :
•	Support Vector Machine (SVM) 
 Rapport de classification :
               precision    recall  f1-score   support

           0       0.99      0.94      0.96       161
           1       0.50      0.91      0.65        11

    accuracy                           0.94       172
   macro avg       0.75      0.92      0.81       172
weighted avg       0.96      0.94      0.94       172

 Précision du modèle : 0.94
 ROC-AUC Score : 0.92
•	CatBoost 
Rapport de classification...
              precision    recall  f1-score   support

           0       0.98      0.97      0.97       161
           1       0.62      0.73      0.67        11

    accuracy                           0.95       172
   macro avg       0.80      0.85      0.82       172
weighted avg       0.96      0.95      0.96       172
•	XGBoost
Exactitude du modèle : 0.95
Précision : 0.56
Rappel : 0.91
Score F1 : 0.69
ROC-AUC : 0.94
Rapport de classification...
              precision    recall  f1-score   support

           0       0.99      0.95      0.97       161
           1       0.56      0.91      0.69        11

    accuracy                           0.95       172
   macro avg       0.77      0.93      0.83       172
weighted avg       0.97      0.95      0.95       172
•	Random Forest
  precision    recall  f1-score   support

           0       0.97      0.97      0.97       155
           1       0.71      0.71      0.71        17

    accuracy                           0.94       172
   macro avg       0.84      0.84      0.84       172
weighted avg       0.94      0.94      0.94       172
AUC-ROC: 0.9725
Modèle choisi : Random Forest
Après comparaison, le modèle Random Forest a obtenu les meilleures performances.

Analyse SHAP : Caractéristiques les plus influentes
D'après l'analyse SHAP, les facteurs ayant le plus d'impact sur la prédiction du cancer du col de l'utérus sont :
•	Nombre de partenaires sexuels
•	Antécédents d'IST (Infections Sexuellement Transmissibles)
•	Utilisation de contraceptifs hormonaux
•	Âge du premier rapport sexuel
l’ingénierie rapide est celle aui nous a per,it de faire tout le netoyage de donner c'est à dire la Conversion des variables non numériques en valeurs numériques.
Remplacement des valeurs manquantes par la médiane pour éviter les biais.Suppression des colonnes fortement corrélées pour éviter la redondance, l'application d’un class_weight pour donner plus d’importance à la classe minoritaire et améliorer la détection des cas positif et également l'utilisation de SHAP, qui nous a permis d'identifié les variables ayant le plus d’impact sur les prédictions, ce qui a permis d’affiner le modèle et d’améliorer ses performances.Ce travail est crucial, car même le meilleur modèle ne donnera pas de bons résultats si les données sont mal préparées.

Installation
1.	Clonez le projet :
git clone https://github.com/utilisateur/nom-du-projet.git
2.	Installez les dépendances :
pip install -r requirements.txt
3.	Exécutez l'entraînement :
python train_model.py
Utilisation
1.	Chargez vos données dans risk_factors_cervical_cancer.csv
2.	Exécutez le script d'entraînement
3.	Analysez les performances affichées
Résultats & Visualisation
•	Matrice de confusion
•	Courbe ROC-AUC
•	Importance des variables (Feature Importance)
Mendre du Groupe :
DICKO ELHADJ SEKOU
DON TITO TRESOR
SAADOUI HELMI
MEITE SAID AYMAN
