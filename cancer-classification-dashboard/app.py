import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Import components
from components.data_explorer import display_data_overview, plot_class_distribution, plot_feature_correlations, plot_feature_importance
from components.feature_analysis import display_feature_analysis
from components.model_comparison import train_and_evaluate_models
from components.model_evaluation import evaluate_best_model
from components.prediction import make_prediction

# Set page config
st.set_page_config(
    page_title="Cancer Cell Classification Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
if os.path.exists(os.path.join("assets", "styles.css")):
    with open(os.path.join("assets", "styles.css"), "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create the sidebar
st.sidebar.title("Cancer Cell Classification")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4812/4812487.png", width=100)

# Navigation
page = st.sidebar.selectbox(
    "Select a page",
    ["Introduction", "Data Explorer", "Feature Analysis", "Model Comparison", "Model Evaluation", "Make Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This interactive dashboard visualizes breast cancer classification "
    "using machine learning models to distinguish between benign and malignant tumors."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit")

# Load data - shared across all pages
@st.cache_data
def get_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    df_viz = df.copy()
    df_viz['diagnosis_label'] = df_viz['diagnosis'].map({0: 'Malignant', 1: 'Benign'})
    return data, df, df_viz

# Load data
data, df, df_viz = get_data()

# Display the selected page
if page == "Introduction":
    st.title("Breast Cancer Classification Dashboard")
    st.write("""
    ### Welcome to the Breast Cancer Classification Dashboard!
    
    This application demonstrates how machine learning can be used to classify breast cancer tumors as benign or malignant
    based on features extracted from digitized images of fine needle aspirates (FNA) of breast masses.
    
    #### Dataset Information:
    - **Features**: 30 numeric features computed from the digitized image
    - **Target**: Diagnosis (M = malignant, B = benign)
    - **Samples**: 569 instances
    
    #### What You Can Do:
    - Explore the dataset and its features
    - Analyze feature distributions and correlations
    - Compare different machine learning models
    - Evaluate model performance
    - Make predictions on new data
    
    #### Navigation:
    Use the sidebar on the left to navigate between different sections of the dashboard.
    """)
    
    st.image("https://www.cancer.gov/sites/g/files/xnrzdm211/files/styles/cgov_article/public/cgov_image/media_image/2019-11/Breast%20Cancer%20Cells%2C%20credit%20NCI%20Visuals%20Online.jpg?h=4f45205e&itok=l_gINRW1", 
             caption="Breast cancer cells under microscope. Credit: National Cancer Institute")

elif page == "Data Explorer":
    st.title("Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Class Distribution", "Feature Correlations", "Feature Importance"])
    
    with tab1:
        display_data_overview()
    
    with tab2:
        plot_class_distribution()
    
    with tab3:
        plot_feature_correlations()
    
    with tab4:
        plot_feature_importance()

elif page == "Feature Analysis":
    display_feature_analysis()

elif page == "Model Comparison":
    train_and_evaluate_models()

elif page == "Model Evaluation":
    evaluate_best_model()

elif page == "Make Prediction":
    make_prediction()