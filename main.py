import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel

def main():
    # Chargement des données
    df = pd.read_csv("data/train.csv")
    
    # Prétraitement
    prep = TitanicPreprocessor()
    X, y = prep.fit_transform(df)
    
    # Entraînement du modèle
    model = TitanicModel()
    model.train(X, y)
    
    # Évaluation
    evaluation = model.evaluate(X, y)
    print("🔹 Performance sur l'ensemble d'entraînement:")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"F1-score: {evaluation['f1_score']:.4f}")
    print(f"AUC-ROC: {evaluation['roc_auc']:.4f}")
    
    # Validation croisée
    mean_cv, std_cv = model.cross_validate(X, y)
    print(f"\n🔹 Cross-validation: {mean_cv:.4f} ± {std_cv:.4f}")

if __name__ == "__main__":
    main()