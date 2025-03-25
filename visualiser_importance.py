"""
Visualisation de l'importance des caractéristiques du modèle d'arbre de décision
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic/data/train.csv')


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)


feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)


plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')


for i, v in enumerate(feature_importance['Importance']):
    ax.text(v + 0.01, i, f"{v:.4f}", va='center')

plt.title('Importance des caractéristiques dans le modèle d\'arbre de décision', fontsize=16)
plt.xlabel('Importance relative', fontsize=14)
plt.ylabel('Caractéristique', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance_detailed.png', dpi=300)
plt.show()


plt.figure(figsize=(10, 10))
plt.pie(feature_importance['Importance'], 
        labels=feature_importance['Feature'], 
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.1 if i == 0 else 0 for i in range(len(features))],  
        colors=plt.cm.viridis(np.linspace(0, 1, len(features))))
plt.title('Proportion de l\'importance des caractéristiques', fontsize=16)
plt.axis('equal')  
plt.tight_layout()
plt.savefig('feature_importance_pie.png', dpi=300)
plt.show()

print("Visualisations sauvegardées dans 'feature_importance_detailed.png' et 'feature_importance_pie.png'")
