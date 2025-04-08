import pytest
import pandas as pd
import numpy as np
from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_titanic_data():
    """Fixture that provides a small sample of Titanic data for testing"""
    return pd.DataFrame({
        'Survived': [1, 0, 1, 0],
        'Pclass': [1, 3, 1, 2],
        'Sex': ['female', 'male', 'female', 'male'],
        'Age': [38, 26, 35, 27],
        'SibSp': [1, 0, 1, 0],
        'Parch': [0, 0, 0, 0],
        'Fare': [71.2833, 7.9250, 53.1000, 10.5000],
        'Embarked': ['S', 'S', 'C', 'Q']
    })

@pytest.fixture
def preprocessor():
    """Fixture that provides a preprocessor instance"""
    return TitanicPreprocessor()

@pytest.fixture
def trained_model(sample_titanic_data, preprocessor):
    """Fixture that provides a trained model using the sample data"""
    X, y = preprocessor.fit_transform(sample_titanic_data)
    model = TitanicModel()
    model.train(X, y)
    return model

def test_model_prediction(trained_model, preprocessor):
    """Test that the model can make predictions"""
    # Create a single sample for prediction
    sample = pd.DataFrame({
        'Pclass': [1],
        'Sex': ['female'],
        'Age': [30],
        'SibSp': [0],
        'Parch': [0],
        'Fare': [50.0],
        'Embarked': ['S']
    })
    X = preprocessor.transform(sample)
    prediction = trained_model.predict(X)
    proba = trained_model.predict_proba(X)
    assert prediction is not None
    assert prediction.shape[0] == 1
    assert proba.shape == (1, 2)  # Vérifie les probabilités pour les deux classes
    assert np.all((proba >= 0) & (proba <= 1))  # Vérifie que les probabilités sont entre 0 et 1

def test_model_evaluation(trained_model, sample_titanic_data, preprocessor):
    """Test that the model can be evaluated"""
    X, y = preprocessor.transform(sample_titanic_data), sample_titanic_data['Survived']
    evaluation = trained_model.evaluate(X, y)
    
    # Vérifie la présence et la validité de toutes les métriques
    assert 'accuracy' in evaluation
    assert 'f1_score' in evaluation
    assert 'roc_auc' in evaluation
    assert 0 <= evaluation['accuracy'] <= 1
    assert 0 <= evaluation['f1_score'] <= 1
    assert 0 <= evaluation['roc_auc'] <= 1
    
    # Vérifie que les prédictions sont cohérentes
    predictions = trained_model.predict(X)
    assert len(predictions) == len(y)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
    assert all(pred in [0, 1] for pred in predictions)

def test_model_train(sample_titanic_data, preprocessor):
    """Test that the model can be trained correctly"""
    X, y = preprocessor.fit_transform(sample_titanic_data)
    model = TitanicModel()
    model.train(X, y)
    
    # Vérifie que le modèle est bien initialisé et entraîné
    assert model.model is not None, "Le modèle n'a pas été entraîné correctement."
    assert hasattr(model.model, 'predict'), "Le modèle n'a pas la méthode predict"
    assert hasattr(model.model, 'predict_proba'), "Le modèle n'a pas la méthode predict_proba"
    
    # Vérifie les prédictions sur les données d'entraînement
    prediction = model.predict(X)
    assert len(prediction) == len(y), "Le modèle ne prédit pas correctement sur les données d'entraînement."
    
    # Vérifie la validation croisée
    mean_score, std_score = model.cross_validate(X, y, cv=2)
    assert isinstance(mean_score, float)
    assert isinstance(std_score, float)
    assert 0 <= mean_score <= 1
    assert 0 <= std_score <= 1
