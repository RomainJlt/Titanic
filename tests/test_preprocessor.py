import pandas as pd
from src.preprocessor import TitanicPreprocessor

def test_preprocessor():
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