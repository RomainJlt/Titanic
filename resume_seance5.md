# Résumé de la Séance 5 - Titanic Survival Prediction

## 1. Objectif de la Séance
Approfondissement de l'analyse du dataset Titanic en utilisant des techniques de machine learning, en particulier la classification binaire avec un arbre de décision.

## 2. Contexte du Dataset
- **Titanic**: Classification binaire (survécu / pas survécu)
- **Caractéristiques principales**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

## 3. Vérification de l'Équilibre de la Cible
- **Survivants**: 38.38%
- **Non-survivants**: 61.62%
- **Constat**: Déséquilibre modéré justifiant l'utilisation du F1-score comme métrique complémentaire à l'accuracy

## 4. Prétraitement des Données
1. **Gestion des valeurs manquantes**:
   - Age: remplacé par la médiane
   - Embarked: remplacé par le mode

2. **Encodage des variables catégorielles**:
   - Sex: {'female': 1, 'male': 0}
   - Embarked: {'S': 0, 'C': 1, 'Q': 2}

3. **Caractéristiques sélectionnées**:
   - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

## 5. Pipeline d'Entraînement et Validation
1. **Séparation train/test**:
   - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   - 80% des données pour l'entraînement, 20% pour le test

2. **Validation croisée**:
   - Utilisation de cross_val_score avec 5 folds
   - Score moyen: 81.37% avec un écart-type de 2.89%

## 6. Formation du Premier Modèle
- **Algorithme choisi**: Arbre de décision (DecisionTreeClassifier)
- **Hyperparamètres**: max_depth=5, random_state=42
- **Raison du choix**: Bon équilibre entre performance et interprétabilité

## 7. Évaluation des Performances
1. **Métriques utilisées**:
   - **Accuracy**: 79.89% (proportion de prédictions correctes)
   - **F1-score**: 72.73% (moyenne harmonique de la précision et du rappel)
   - **Matrice de confusion**: Visualisation des vrais/faux positifs/négatifs

2. **Justification du choix des métriques**:
   - Accuracy: Métrique standard pour évaluer la performance globale
   - F1-score: Plus pertinent en cas de déséquilibre des classes
   - Matrice de confusion: Permet d'identifier les types d'erreurs du modèle

## 8. Analyse de l'Arbre et Importance des Caractéristiques
1. **Importance des caractéristiques**:
   - **Sex**: 53.74% - Le facteur le plus déterminant
   - **Pclass**: 19.14% - La classe socio-économique
   - **Fare**: 12.49% - Le prix du billet
   - **Age**: 9.06% - L'âge du passager
   - **SibSp**: 4.27% - Nombre de frères/sœurs/conjoints
   - **Embarked**: 1.29% - Port d'embarquement
   - **Parch**: 0.00% - Nombre de parents/enfants (non utilisé par le modèle)

2. **Interprétation de l'arbre**:
   - La première division se fait sur le sexe, confirmant l'hypothèse "les femmes et les enfants d'abord"
   - La classe socio-économique est un facteur important pour la survie
   - L'âge joue un rôle dans certaines branches, confirmant la priorité donnée aux enfants

## 9. Pistes d'Amélioration
1. **Optimisation des hyperparamètres**:
   - Ajuster max_depth, min_samples_split, etc.

2. **Ingénierie de caractéristiques**:
   - Créer FamilySize = SibSp + Parch + 1
   - Extraire le titre à partir du nom
   - Créer des groupes d'âge

3. **Algorithmes plus avancés**:
   - Random Forest
   - Gradient Boosting (XGBoost)
   - Techniques d'ensemble

4. **Gestion du déséquilibre**:
   - Techniques de rééchantillonnage
   - Pondération des classes

## 10. Conclusion
L'arbre de décision offre un bon équilibre entre performance et interprétabilité pour ce problème de classification. L'analyse confirme plusieurs hypothèses historiques sur les facteurs de survie lors du naufrage du Titanic, avec le sexe et la classe sociale comme déterminants principaux.

Le modèle actuel est une base solide qui pourrait être améliorée par une ingénierie de caractéristiques plus poussée et l'utilisation d'algorithmes plus sophistiqués lors de la prochaine séance.
