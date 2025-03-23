import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_data():
    """Load the breast cancer dataset and return X, y"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

def train_model(model_name='RandomForest', test_size=0.25, random_state=42):
    """
    Train a model on the breast cancer dataset
    
    Parameters:
    -----------
    model_name : str
        Name of the model to train ('RandomForest', 'SVM', 'GradientBoosting', 'NaiveBayes')
    test_size : float
        Size of the test set
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing the trained pipeline, metrics, and predictions
    """
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Select model
    if model_name == 'RandomForest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name == 'SVM':
        classifier = SVC(probability=True, random_state=random_state)
    elif model_name == 'GradientBoosting':
        classifier = GradientBoostingClassifier(random_state=random_state)
    elif model_name == 'NaiveBayes':
        classifier = GaussianNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    model_path = os.path.join('models', f'{model_name.lower()}_model.joblib')
    joblib.dump(pipeline, model_path)
    
    # Return results
    return {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'X_test': X_test,
        'y_test': y_test,
        'model_path': model_path
    }

def load_model(model_name='RandomForest'):
    """
    Load a trained model
    
    Parameters:
    -----------
    model_name : str
        Name of the model to load ('RandomForest', 'SVM', 'GradientBoosting', 'NaiveBayes')
    
    Returns:
    --------
    object
        The trained scikit-learn pipeline
    """
    model_path = os.path.join('models', f'{model_name.lower()}_model.joblib')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found. Training new model...")
        result = train_model(model_name)
        return result['pipeline']
    
    # Load model
    return joblib.load(model_path)

def get_feature_names():
    """Return the feature names from the breast cancer dataset"""
    return load_breast_cancer().feature_names

if __name__ == '__main__':
    # Train all models
    models = ['RandomForest', 'SVM', 'GradientBoosting', 'NaiveBayes']
    results = {}
    
    for model_name in models:
        print(f"Training {model_name}...")
        results[model_name] = train_model(model_name)
        print(f"Accuracy: {results[model_name]['accuracy']:.4f}")
        
    # Find best model
    best_model = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model} with accuracy: {results[best_model]['accuracy']:.4f}")