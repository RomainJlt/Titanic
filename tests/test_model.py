import pytest
import pandas as pd
from src.preprocessor import TitanicPreprocessor
from src.model import TitanicModel

@pytest.fixture
def sample_titanic_data():
    """Fixture that provides a small sample of Titanic data for testing"""
    return pd.DataFrame({
        'Survived': [1, 0, 1, 0],
        'Pclass': [1, 3, 1, 2],
        'Sex': ['female', 'male', 'female', 'male'],
        'Age': [38, 26, 35, 27],
        'Fare': [71.2833, 7.9250, 53.1000, 10.5000]
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
        'Fare': [50.0]
    })
    X = preprocessor.transform(sample)
    prediction = trained_model.predict(X)
    assert prediction is not None
    assert prediction.shape[0] == 1

def test_model_evaluation(trained_model, sample_titanic_data, preprocessor):
    """Test that the model can be evaluated"""
    X, y = preprocessor.transform(sample_titanic_data), sample_titanic_data['Survived']
    evaluation = trained_model.evaluate(X, y)
    assert 'accuracy' in evaluation
    assert 'f1_score' in evaluation
    assert 'roc_auc' in evaluation