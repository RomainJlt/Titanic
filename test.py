import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv('titanic/train.csv')


missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :")
print(missing_values[missing_values > 0])

# Exemple de décisions :
if missing_values['Cabin'] > len(df) * 0.5:
    print("On drop la colonne Cabin car trop de valeurs manquantes")
    df = df.drop(columns=['Cabin'])
else:
    print("On tente une imputation pour Cabin (pas retenu ici car trop de missing)")

# Imputation sur Age par la médiane
if missing_values['Age'] > 0:
    median_age = df['Age'].median()
    print(f"Imputation des valeurs manquantes de Age par la médiane : {median_age}")
    df['Age'] = df['Age'].fillna(median_age)

# Suppression des lignes avec embarquement manquant
if missing_values['Embarked'] > 0:
    print("On supprime les lignes avec Embarked manquant")
    df = df.dropna(subset=['Embarked'])


plt.figure(figsize=(8,6))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survie par sexe')
plt.savefig('survie_par_sexe.png')
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(df[df['Survived']==1]['Age'], kde=True, color='green', label='Survived', bins=30)
sns.histplot(df[df['Survived']==0]['Age'], kde=True, color='red', label='Not Survived', bins=30)
plt.legend()
plt.title('Distribution des âges selon la survie')
plt.savefig('age_vs_survie.png')
plt.show()


plt.figure(figsize=(8,6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survie par classe')
plt.savefig('survie_par_classe.png')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Âge par classe')
plt.savefig('age_par_classe.png')
plt.show()
