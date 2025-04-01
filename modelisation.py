#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Chargement des données
df = pd.read_csv("data/train.csv")

# Vérification de l'équilibre des classes
print("Répartition des classes :")
print(df['Survived'].value_counts(normalize=True))  # Vérifier si les classes sont équilibrées

# Sélection des features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"
df = df[features + [target]]

# Séparation X et y
X = df.drop(columns=[target])
y = df[target]

# Prétraitement des données
numerical_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Pclass", "Sex", "Embarked"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Remplace les valeurs manquantes par la médiane
    ("scaler", StandardScaler())  # Standardisation
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Remplace les NaN par la valeur la plus fréquente
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # Encodage one-hot des variables catégorielles
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entraînement de plusieurs modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilités pour ROC AUC
    
    # Évaluation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n🔹 Modèle : {name}")
    print(f"🔹 Accuracy : {acc:.4f}")
    print(f"🔹 F1-score : {f1:.4f}")
    print(f"🔹 AUC-ROC : {auc:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non survécu", "Survécu"], yticklabels=["Non survécu", "Survécu"])
    plt.title(f"Matrice de confusion - {name}")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Courbe de référence
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbes ROC")
plt.legend()
plt.show()

# Validation croisée
best_model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
print(f"🔹 Cross-validation accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %%
