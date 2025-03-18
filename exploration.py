# import pandas as pd

# df = pd.read_csv('titanic/train.csv')

# print(df.head())



# from ydata_profiling import ProfileReport

# profile = ProfileReport(df, title="Profiling Report")
# profile.to_file("your_report.html")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create title feature from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5,
        'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 2, 'Countess': 3,
        'Ms': 2, 'Lady': 3, 'Jonkheer': 1, 'Don': 1, 'Mme': 3, 'Capt': 5, 'Sir': 5
    }
    df['Title'] = df['Title'].map(title_mapping)
    
    # Convert categorical features
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Select features for model
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
               'FamilySize', 'Title']
    
    return df[features]

def main():
    # Load data
    try:
        train_df = pd.read_csv('titanic/train.csv')
        test_df = pd.read_csv('titanic/test.csv')
        print("Data loaded successfully!")
        
        # Preprocess data
        X_train = preprocess_data(train_df)
        y_train = train_df['Survived']
        X_test = preprocess_data(test_df)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )
        
        # Validate model
        val_predictions = model.predict(X_val)
        print("\nValidation Results:")
        print(f"Accuracy: {accuracy_score(y_val, val_predictions):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_predictions))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Make predictions on test set
        test_predictions = model.predict(X_test_scaled)
        
        # Save predictions
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': test_predictions
        })
        submission.to_csv('titanic_predictions.csv', index=False)
        print("\nPredictions saved to 'titanic_predictions.csv'")
        
    except FileNotFoundError:
        print("Error: Could not find the dataset files. Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()