import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    return df

def plot_class_distribution(df):
    df_viz = df.copy()
    df_viz['diagnosis_label'] = df_viz['diagnosis'].map({0: 'Malignant', 1: 'Benign'})
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='diagnosis_label', data=df_viz, palette='viridis')
    plt.title('Class Distribution in Breast Cancer Dataset')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points')
    
    st.pyplot(plt)

def plot_feature_correlation(df):
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(plt)

def plot_feature_importance(df):
    correlation_with_target = df.corr()['diagnosis'].sort_values(ascending=False).drop('diagnosis')
    plt.figure(figsize=(12, 8))
    sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index, palette='viridis')
    plt.title('Feature Correlation with Diagnosis')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    st.pyplot(plt)

def feature_analysis():
    st.title("Feature Analysis")
    df = load_data()
    
    st.subheader("Class Distribution")
    plot_class_distribution(df)
    
    st.subheader("Feature Correlation Matrix")
    plot_feature_correlation(df)
    
    st.subheader("Feature Importance")
    plot_feature_importance(df)

def display_feature_analysis():
    st.title("Feature Analysis")
    
    # Load data
    df = load_data()
    all_features = df.columns.tolist()
    
    # Feature selection
    st.sidebar.subheader("Select Features to Analyze")
    
    # Allow user to select features
    selected_features = st.sidebar.multiselect(
        "Choose features to examine",
        all_features,
        default=all_features[:3]  # Default to first 3 features
    )
    
    if not selected_features:
        st.warning("Please select at least one feature to analyze.")
        return
    
    # Feature distribution by diagnosis
    st.subheader("Feature Distribution by Diagnosis")
    
    for feature in selected_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=df, 
            x=feature, 
            hue="diagnosis", 
            multiple="stack",
            palette={0: "red", 1: "green"},
            bins=30,
            alpha=0.7
        )
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.legend(labels=["Malignant", "Benign"])
        st.pyplot(fig)
        
        # Feature statistics by diagnosis
        st.write(f"### {feature} Statistics by Diagnosis")
        
        feature_stats = df.groupby("diagnosis")[feature].describe().T
        feature_stats.columns = ["Malignant", "Benign"]
        st.write(feature_stats)
    
    # Correlation Heatmap
    if len(selected_features) > 1:
        st.subheader("Feature Correlation Heatmap")
        
        selected_df = df[selected_features]
        corr_matrix = selected_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    df = load_data()
    display_feature_analysis()