import numpy as np
from src.model import TitanicModel

def test_model():
    # Données d'exemple
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Test du modèle
    model = TitanicModel()
    model.train(X, y)
    preds = model.predict(X)
    probas = model.predict_proba(X)
    
    # Vérifications
    assert len(preds) == 100
    assert probas.shape == (100, 2)
    assert (probas.sum(axis=1) - 1 < 1e-6).all()