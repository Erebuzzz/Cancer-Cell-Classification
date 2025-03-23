import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import os

# Create a folder to save plots if it doesn't exist
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Load the breast cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Create a separate dataframe for visualization that includes the string labels
df_viz = df.copy()
df_viz['diagnosis_label'] = df_viz['diagnosis'].map({0: 'Malignant', 1: 'Benign'})

def display_data_overview():
    st.title("Breast Cancer Dataset Overview")
    st.write("### Dataset Shape:")
    st.write(df.shape)
    
    st.write("### Number of Benign samples:")
    st.write(sum(df['diagnosis'] == 1))
    
    st.write("### Number of Malignant samples:")
    st.write(sum(df['diagnosis'] == 0))
    
    st.write("### Feature Statistics:")
    st.write(df.describe().T)

def plot_class_distribution():
    st.write("### Class Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='diagnosis_label', data=df_viz, palette='viridis', ax=ax)
    plt.title('Class Distribution in Breast Cancer Dataset')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points')
    
    st.pyplot(fig)

def plot_feature_correlations():
    st.write("### Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = df.corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)

def plot_feature_importance():
    st.write("### Feature Importance with Diagnosis")
    
    # Sort features by correlation with diagnosis
    corr_with_target = df.corr()['diagnosis'].abs().sort_values(ascending=False)
    top_features = corr_with_target.index[1:16]  # Skip diagnosis itself
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=corr_with_target[top_features].values, y=top_features, palette='viridis', ax=ax)
    plt.title('Top 15 Features by Correlation with Diagnosis')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ["Data Overview", "Class Distribution", "Feature Correlations", "Feature Importance"])
    
    if options == "Data Overview":
        display_data_overview()
    elif options == "Class Distribution":
        plot_class_distribution()
    elif options == "Feature Correlations":
        plot_feature_correlations()
    elif options == "Feature Importance":
        plot_feature_importance()

if __name__ == "__main__":
    main()