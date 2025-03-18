#%% Importation des librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Chargement des données
df = pd.read_csv("data/train.csv")

#%% Analyse générale
print(" Aperçu des données")
print(df.head())

print("\n Infos générales")
print(df.info())  # Type des colonnes et valeurs manquantes

print("\n Statistiques descriptives")
print(df.describe())  # Moyennes, médianes, min, max

#%% Détection des valeurs manquantes
print("\n Valeurs manquantes")
print(df.isnull().sum())  # Nombre de valeurs nulles par colonne

#%% Visualisation des distributions

# Histogramme des âges
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"].dropna(), bins=30, kde=True)
plt.title("Distribution des âges")
plt.show()

# Répartition des survivants
plt.figure(figsize=(6, 4))
sns.countplot(x="Survived", data=df)
plt.title("Nombre de survivants (1 = survécu, 0 = non)")
plt.show()

# Boxplot de l'âge par classe
plt.figure(figsize=(8, 5))
sns.boxplot(x="Pclass", y="Age", data=df)
plt.title("Répartition des âges par classe")
plt.show()

#%% Premières hypothèses
print("\n Taux de survie par sexe")
print(df.groupby("Sex")["Survived"].mean())

print("\n Taux de survie par classe")
print(df.groupby("Pclass")["Survived"].mean())

#%% Rapport automatisé avec ydata_profiling
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("report2.html")

# %%
