# Fiche Résumé (Séance 5)

## 1. Choisir le type d'algorithme
- **Type choisi**: Classification binaire
- **Dataset**: Titanic (survécu / pas survécu)
- **Algorithme**: Arbre de décision (DecisionTreeClassifier)

## 2. Séparer le dataset (train/test)
- **Méthode**: train_test_split de scikit-learn
- **Proportion**: 80% entraînement, 20% test
- **Code utilisé**:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Entraîner un premier modèle simple
- **Modèle**: DecisionTreeClassifier
- **Hyperparamètres**: max_depth=5, random_state=42
- **Code utilisé**:
```python
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
```

## 4. Mesurer la performance et la commenter
- **Accuracy**: 79.89% (proportion de prédictions correctes)
- **F1-score**: 72.73% (moyenne harmonique de la précision et du rappel)
- **Commentaire**: Performance satisfaisante pour un premier modèle. L'écart entre accuracy et F1-score indique que le modèle a plus de difficulté à prédire correctement la classe minoritaire (survivants).
- **Validation croisée**: Score moyen de 81.37% avec un écart-type de 2.89%, indiquant une bonne stabilité du modèle.

## 5. Préparer la suite
- **Hyperparamètres à tuner**: max_depth, min_samples_split, min_samples_leaf
- **Modèles à comparer**: Random Forest, Gradient Boosting
- **Ingénierie de caractéristiques**: Créer FamilySize, extraire le titre du nom, groupes d'âge
- **Techniques à explorer**: Gestion du déséquilibre des classes, ensemble learning
