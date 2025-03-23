import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import os
import joblib
from model.train_model import load_model, get_feature_names

def make_prediction():
    st.title("Cancer Cell Prediction")
    
    # Check if models exist
    models_path = "models"
    model_files = []
    if os.path.exists(models_path):
        model_files = [f for f in os.listdir(models_path) if f.endswith('_model.joblib')]
    
    if not model_files:
        st.error("No trained models found. Please go to the 'Model Comparison' tab and train a model first.")
        return
    
    # Select model
    model_name = st.selectbox(
        "Select model for prediction",
        [f.split('_')[0] for f in model_files]
    )
    
    # Load model
    model = load_model(model_name)
    
    # Get feature names and statistics
    feature_names = get_feature_names()
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Display input methods
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Sample Data", "Upload Data"]
    )
    
    if input_method == "Manual Input":
        st.write("Enter feature values manually:")
        
        # Create columns for better layout
        cols = st.columns(3)
        input_values = {}
        
        for i, feature in enumerate(feature_names):
            # Calculate statistics for the feature to set sensible default values and ranges
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            
            # Set default to mean with range from min to max
            col_idx = i % 3
            input_values[feature] = cols[col_idx].slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                format="%.4f"
            )
        
        # Create input dataframe
        input_df = pd.DataFrame([input_values])
        
    elif input_method == "Sample Data":
        # Let user select from random samples in the dataset
        st.write("Select a sample from the dataset:")
        random_indices = np.random.choice(len(X), size=10, replace=False)
        selected_index = st.selectbox("Choose a sample:", random_indices)
        
        input_df = X.iloc[[selected_index]]
        
        st.write("### Sample Feature Values:")
        st.write(input_df)
        
    else:  # Upload Data
        st.write("Upload a CSV file with feature values (should match the feature names):")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is None:
            return
        
        # Read the uploaded file
        input_df = pd.read_csv(uploaded_file)
        
        # Check if the columns match
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            st.error(f"Uploaded file is missing these required columns: {', '.join(missing_cols)}")
            return
        
        # Use only the needed columns in the right order
        input_df = input_df[feature_names]
        
        st.write("### Uploaded Data:")
        st.write(input_df)
    
    # Make prediction
    if st.button("Predict"):
        with st.spinner("Making prediction..."):
            # Use the model to predict
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None
            
            # Display results
            st.subheader("Prediction Results")
            
            for i in range(len(input_df)):
                result_text = "Benign (Non-cancerous)" if prediction[i] == 1 else "Malignant (Cancerous)"
                result_color = "green" if prediction[i] == 1 else "red"
                
                st.markdown(f"### Sample {i+1}: <span style='color:{result_color};'>{result_text}</span>", unsafe_allow_html=True)
                
                if prediction_proba is not None:
                    malignant_prob = prediction_proba[i][0]
                    benign_prob = prediction_proba[i][1]
                    
                    # Create probability gauge charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Malignant Probability:")
                        st.progress(malignant_prob)
                        st.write(f"{malignant_prob:.2%}")
                    
                    with col2:
                        st.write("Benign Probability:")
                        st.progress(benign_prob)
                        st.write(f"{benign_prob:.2%}")
                    
                    # Plot the probability distribution
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(["Malignant", "Benign"], [malignant_prob, benign_prob], color=["red", "green"])
                    plt.ylim(0, 1)
                    plt.title("Prediction Probability")
                    plt.ylabel("Probability")
                    plt.tight_layout()
                    st.pyplot(fig)

if __name__ == "__main__":
    make_prediction()