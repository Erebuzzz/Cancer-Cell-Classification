import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import joblib
import os
from model.train_model import load_data, train_model

def train_and_evaluate_models():
    st.title("Model Comparison and Selection")
    
    # Model selection and parameters
    st.sidebar.subheader("Model Configuration")
    
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.25, 0.05)
    random_state = st.sidebar.slider("Random State", 0, 100, 42)
    
    # Model selection checkboxes
    st.sidebar.subheader("Select Models")
    model_nb = st.sidebar.checkbox("Gaussian Naive Bayes", value=True)
    model_rf = st.sidebar.checkbox("Random Forest", value=True)
    model_gb = st.sidebar.checkbox("Gradient Boosting", value=True)
    model_svm = st.sidebar.checkbox("Support Vector Machine", value=True)
    
    # Button to trigger training
    train_button = st.sidebar.button("Train Models")
    
    # Display pre-trained models results or train new ones
    if train_button:
        # Load data
        X, y = load_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        st.write(f"Training set shape: {X_train.shape}")
        st.write(f"Testing set shape: {X_test.shape}")
        
        # Create dictionary for models
        model_names = []
        
        if model_nb:
            model_names.append("NaiveBayes")
        
        if model_rf:
            model_names.append("RandomForest")
        
        if model_gb:
            model_names.append("GradientBoosting")
        
        if model_svm:
            model_names.append("SVM")
        
        if not model_names:
            st.warning("Please select at least one model to train.")
            return
        
        # Dictionary to store results
        results = {}
        best_model = None
        best_score = 0
        
        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Train and evaluate each model
        for i, model_name in enumerate(model_names):
            progress_text.text(f"Training {model_name}...")
            progress_value = (i / len(model_names))
            progress_bar.progress(progress_value)
            
            # Train model
            result = train_model(model_name, test_size, random_state)
            results[model_name] = result
            
            # Track best model
            if result['accuracy'] > best_score:
                best_score = result['accuracy']
                best_model = model_name
        
        progress_bar.progress(1.0)
        progress_text.text("Training complete!")
        
        # Compare model metrics
        st.subheader("Model Performance Comparison")
        
        metric_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results.keys()]
        })
        
        st.write(metric_df)
        
        # Plot metric comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Model', y='Accuracy', data=metric_df)
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0.8, 1.0)  # Adjust y-axis for better visualization
        plt.tight_layout()
        st.pyplot(fig)
        
        # ROC curve comparison
        st.subheader("ROC Curve Comparison")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name in results.keys():
            if results[name]['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(results[name]['y_test'], results[name]['y_prob'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        cols = st.columns(min(2, len(results)))
        
        for i, (name, result) in enumerate(results.items()):
            col_idx = i % 2
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            
            with cols[col_idx]:
                st.write(f"**{name}**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Malignant', 'Benign'],
                            yticklabels=['Malignant', 'Benign'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix - {name}')
                plt.tight_layout()
                st.pyplot(fig)
    else:
        # Check if models exist
        models_path = "models"
        if os.path.exists(models_path) and os.listdir(models_path):
            st.info("Pre-trained models are available. Click 'Train Models' to retrain or use the Prediction tab to make predictions.")
        else:
            st.info("Click 'Train Models' to train and compare different classification models.")

if __name__ == "__main__":
    train_and_evaluate_models()