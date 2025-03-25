"""
Titanic Survival Prediction - Classification avec Decision Tree
Séance 5: Approfondissement de l'analyse (techniques statistiques ou ML)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# 1. Chargement des données
print("1. Chargement des données")
df = pd.read_csv('titanic/data/train.csv')

print(f"Dimensions du dataset: {df.shape}")
print("\nAperçu des données:")
print(df.head())

# 2. Vérifier l'équilibre de la cible
print("\n2. Vérification de l'équilibre de la cible")
target_counts = df['Survived'].value_counts(normalize=True) * 100
print(f"Survivants: {target_counts[1]:.2f}%")
print(f"Non-survivants: {target_counts[0]:.2f}%")

# Visualiser la distribution de la cible
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=df)
plt.title('Distribution des survivants')
plt.savefig('survival_distribution.png')
plt.close()

# 3. Prétraitement des données
print("\n3. Prétraitement des données")

# Gestion des valeurs manquantes
print("Valeurs manquantes avant traitement:")
print(df.isnull().sum())

# Remplir les valeurs manquantes
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Conversion des variables catégorielles
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Sélectionner les caractéristiques pour le modèle
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

print("Caractéristiques sélectionnées:", features)

# 4. Séparation train/test
print("\n4. Séparation train/test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Taille du jeu d'entraînement: {X_train.shape}")
print(f"Taille du jeu de test: {X_test.shape}")

# 5. Formation d'un premier modèle (Decision Tree)
print("\n5. Formation du modèle Decision Tree")
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 6. Évaluation des performances
print("\n6. Évaluation des performances")
y_pred = model.predict(X_test)

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# Afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion:")
print(cm)

# Visualiser la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-survivant', 'Survivant'],
            yticklabels=['Non-survivant', 'Survivant'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.savefig('confusion_matrix_dt.png')
plt.close()

# Afficher le rapport de classification
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

# 7. Validation croisée
print("\n7. Validation croisée")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Scores de validation croisée: {cv_scores}")
print(f"Score moyen: {cv_scores.mean():.4f}")
print(f"Écart-type: {cv_scores.std():.4f}")

# 8. Visualisation de l'arbre de décision
print("\n8. Visualisation de l'arbre de décision")
plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=features, 
          class_names=['Non-survivant', 'Survivant'],
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title('Arbre de décision pour la prédiction de survie sur le Titanic')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Importance des caractéristiques
print("\n9. Importance des caractéristiques")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Visualiser l'importance des caractéristiques
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Importance des caractéristiques')
plt.savefig('feature_importance_dt.png')
plt.close()

# 10. Conclusion
print("\n10. Conclusion")
print("Nous avons entraîné un modèle d'arbre de décision pour prédire la survie sur le Titanic.")
print(f"Le modèle a obtenu une accuracy de {accuracy:.4f} et un F1-score de {f1:.4f} sur le jeu de test.")

# Commentaire sur l'équilibre des classes
if abs(target_counts[1] - target_counts[0]) > 20:
    print("\nLe jeu de données présente un déséquilibre significatif entre les classes.")
    print("Le F1-score est donc une métrique plus pertinente que l'accuracy.")
else:
    print("\nLe jeu de données est relativement équilibré entre les classes.")
    print("L'accuracy est donc une métrique appropriée, mais le F1-score reste informatif.")

# Analyse de l'arbre
print("\nAnalyse de l'arbre de décision:")
print("1. La première division de l'arbre se fait probablement sur le sexe (Sex), ")
print("   ce qui confirme l'hypothèse 'les femmes et les enfants d'abord'.")
print("2. La classe (Pclass) est également un facteur important, ")
print("   les passagers de première classe ayant eu plus de chances de survie.")
print("3. L'âge (Age) joue un rôle dans certaines branches de l'arbre, ")
print("   confirmant que les enfants avaient une priorité.")

# Pistes d'amélioration
print("\nPistes d'amélioration:")
print("1. Ajuster les hyperparamètres de l'arbre (max_depth, min_samples_split, etc.)")
print("2. Créer de nouvelles caractéristiques (ex: FamilySize = SibSp + Parch + 1)")
print("3. Essayer d'autres algorithmes (Random Forest, Gradient Boosting, etc.)")
print("4. Combiner plusieurs modèles (ensemble learning)")
