import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel

def main():
    # Chargement des données
    df = pd.read_csv("data/train.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    # Prétraitement
    prep = TitanicPreprocessor()
    X_train, y_train = prep.fit_transform(df_train)
    X_test = prep.transform(df_test)
    y_test = df_test["Survived"]
    # Entraînement du modèle
    model = TitanicModel()
    model.train(X_train, y_train)
    
    evaluation = model.evaluate(X_test, y_test)
    print("🔹 Performance sur l'ensemble d'entraînement:")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"F1-score: {evaluation['f1_score']:.4f}")
    print(f"AUC-ROC: {evaluation['roc_auc']:.4f}")
    
    mean_cv, std_cv = model.cross_validate(X_train, y_train)
    print(f"\n🔹 Cross-validation: {mean_cv:.4f} ± {std_cv:.4f}")

if __name__ == "__main__":
    main()