import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import os
import joblib
from model.train_model import load_model, load_data

def evaluate_best_model():
    st.title("Best Model Evaluation")
    
    # Check if models exist
    models_path = "models"
    model_files = []
    if os.path.exists(models_path):
        model_files = [f for f in os.listdir(models_path) if f.endswith('_model.joblib')]
    
    if not model_files:
        st.error("No trained models found. Please go to the 'Model Comparison' tab and train a model first.")
        return
    
    # Select model to evaluate
    model_name = st.selectbox(
        "Select model to evaluate",
        [f.split('_')[0] for f in model_files]
    )
    
    # Load the model
    model = load_model(model_name)
    
    # Get test data
    X, y = load_data()
    test_size = st.slider("Test Size", 0.1, 0.5, 0.25, 0.05)
    random_state = st.slider("Random State", 0, 100, 42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['Precision']:.4f}")
    col3.metric("Recall", f"{metrics['Recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    st.pyplot(fig)
    
    # ROC Curve if model has predict_proba
    if hasattr(model, 'predict_proba'):
        st.subheader("ROC Curve")
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(fig)
        
        # Precision-Recall curve
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        st.pyplot(fig)
        
    # Learning Curve
    st.subheader("Learning Curve")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy"
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.plot(train_sizes, test_mean, label="Cross-validation score", color="red")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")
    
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    evaluate_best_model()