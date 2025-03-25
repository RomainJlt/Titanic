import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

print("1. Chargement des données")
df = pd.read_csv('titanic/data/train.csv')

print(f"Dimensions du dataset: {df.shape}")
print("\nAperçu des données:")
print(df.head())

print("\n2. Vérification de l'équilibre de la cible")
target_counts = df['Survived'].value_counts(normalize=True) * 100
print(f"Survivants: {target_counts[1]:.2f}%")
print(f"Non-survivants: {target_counts[0]:.2f}%")

print("\n3. Prétraitement des données")

print("Valeurs manquantes avant traitement:")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

print("Caractéristiques sélectionnées:", features)


print("\n4. Application du code récapitulatif")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = df[features]  
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy :", acc, "F1 :", f1)

print("\n5. Matrice de confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-survivant', 'Survivant'],
            yticklabels=['Non-survivant', 'Survivant'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.savefig('confusion_matrix_logistic.png')
plt.close()
print("Matrice de confusion sauvegardée dans 'confusion_matrix_logistic.png'")

print("\n6. Rapport de classification:")
report = classification_report(y_test, y_pred)
print(report)

print("\n7. Coefficients du modèle (importance des caractéristiques):")
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(coef_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df)
plt.title('Importance des caractéristiques (Coefficients de la régression logistique)')
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png')
plt.close()
print("Visualisation des coefficients sauvegardée dans 'logistic_regression_coefficients.png'")

print("\n8. Validation croisée:")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Scores de validation croisée: {cv_scores}")
print(f"Score moyen: {cv_scores.mean():.4f}")
print(f"Écart-type: {cv_scores.std():.4f}")

print("\n9. Conclusion")
print(f"Le modèle de régression logistique a obtenu une accuracy de {acc:.4f} et un F1-score de {f1:.4f}.")

if abs(target_counts[1] - target_counts[0]) > 20:
    print("\nLe jeu de données présente un déséquilibre significatif entre les classes.")
    print("Le F1-score est donc une métrique plus pertinente que l'accuracy.")
else:
    print("\nLe jeu de données est relativement équilibré entre les classes.")
    print("L'accuracy est donc une métrique appropriée, mais le F1-score reste informatif.")

with open('resultats_logistic_regression.txt', 'w') as f:
    f.write(f"Dimensions du dataset: {df.shape}\n\n")
    f.write(f"Survivants: {target_counts[1]:.2f}%\n")
    f.write(f"Non-survivants: {target_counts[0]:.2f}%\n\n")
    f.write(f"Caractéristiques sélectionnées: {features}\n\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n\n")
    f.write("Matrice de confusion:\n")
    f.write(str(cm) + "\n\n")
    f.write("Rapport de classification:\n")
    f.write(report + "\n\n")
    f.write("Coefficients du modèle (importance des caractéristiques):\n")
    f.write(str(coef_df) + "\n\n")
    f.write("Scores de validation croisée:\n")
    f.write(f"Scores: {cv_scores}\n")
    f.write(f"Score moyen: {cv_scores.mean():.4f}\n")
    f.write(f"Écart-type: {cv_scores.std():.4f}\n")

print("\nRésultats sauvegardés dans 'resultats_logistic_regression.txt'")

print("\n10. Comparaison avec l'arbre de décision:")
print("Arbre de décision (séance précédente):")
print("- Accuracy: 0.7989")
print("- F1-score: 0.7273")
print("- Caractéristique la plus importante: Sex (53.74%)")
print("\nRégression logistique (séance actuelle):")
print(f"- Accuracy: {acc:.4f}")
print(f"- F1-score: {f1:.4f}")
print(f"- Caractéristique la plus importante: {coef_df.iloc[0]['Feature']} (coefficient: {coef_df.iloc[0]['Coefficient']:.4f})")
