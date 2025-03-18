import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv('train.csv')

# Diagnostic des valeurs manquantes
print(df.isnull().sum())

# Traitement des valeurs manquantes
df.drop(columns=['Cabin'], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

# Graphes
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survie par sexe')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survie par classe')
plt.show()

# Hypothèses confirmées :
# - Les femmes et la 1ère classe survivent plus souvent.
# - Les valeurs manquantes de Cabin trop nombreuses, colonne supprimée.
