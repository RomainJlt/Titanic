import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class TitanicPreprocessor:
    def __init__(self):
        self.features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        self.target = "Survived"
        
        numerical_features = ["Age", "Fare", "SibSp", "Parch"]
        categorical_features = ["Pclass", "Sex", "Embarked"]

        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    def fit_transform(self, df):
        df = df[self.features + [self.target]]
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        X_processed = self.preprocessor.fit_transform(X)
        return X_processed, y

    def transform(self, df):
        df = df[self.features]
        return self.preprocessor.transform(df)
