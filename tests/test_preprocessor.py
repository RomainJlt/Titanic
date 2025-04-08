import pandas as pd
import numpy as np
from src.preprocessor import TitanicPreprocessor

def test_preprocessor_fit_transform():
    """Test la transformation complète des données"""
    # Création de données de test
    data = {
        "Pclass": [1, 2, 3, 1],
        "Sex": ["male", "female", "male", "female"],
        "Age": [22, 38, 26, None],
        "SibSp": [1, 1, 0, 1],
        "Parch": [0, 0, 0, 0],
        "Fare": [7.25, 71.28, 7.92, 53.10],
        "Embarked": ["S", "C", "S", None],
        "Survived": [0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Test du préprocesseur
    prep = TitanicPreprocessor()
    X, y = prep.fit_transform(df)
    
    # Vérifications
    assert X.shape[0] == len(y) == 4
    assert not pd.isna(X).any()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.Series)

def test_preprocessor_transform():
    """Test la transformation de nouvelles données"""
    # Données d'entraînement
    train_data = {
        "Pclass": [1, 2],
        "Sex": ["male", "female"],
        "Age": [22, 38],
        "SibSp": [1, 1],
        "Parch": [0, 0],
        "Fare": [7.25, 71.28],
        "Embarked": ["S", "C"],
        "Survived": [0, 1]
    }
    train_df = pd.DataFrame(train_data)
    
    # Données de test
    test_data = {
        "Pclass": [3],
        "Sex": ["male"],
        "Age": [26],
        "SibSp": [0],
        "Parch": [0],
        "Fare": [7.92],
        "Embarked": ["S"]
    }
    test_df = pd.DataFrame(test_data)
    
    # Test du préprocesseur
    prep = TitanicPreprocessor()
    prep.fit_transform(train_df)  # Entraînement sur les données d'entraînement
    X_test = prep.transform(test_df)  # Transformation des données de test
    
    # Vérifications
    assert X_test.shape[0] == 1
    assert not pd.isna(X_test).any()
    assert isinstance(X_test, np.ndarray)

def test_preprocessor_missing_values():
    """Test la gestion des valeurs manquantes"""
    # Données avec beaucoup de valeurs manquantes
    data = {
        "Pclass": [1, None, 3],
        "Sex": ["male", None, "male"],
        "Age": [None, None, 26],
        "SibSp": [1, None, 0],
        "Parch": [None, 0, 0],
        "Fare": [None, 71.28, 7.92],
        "Embarked": [None, "C", None],
        "Survived": [0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Test du préprocesseur
    prep = TitanicPreprocessor()
    X, y = prep.fit_transform(df)
    
    # Vérifications
    assert X.shape[0] == len(y) == 3
    assert not pd.isna(X).any()  # Vérifie qu'il n'y a plus de valeurs manquantes
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.Series)
