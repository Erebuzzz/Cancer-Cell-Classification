# Breast Cancer Classification Dashboard

This project is an interactive dashboard for breast cancer classification using machine learning. It allows users to explore the dataset, analyze features, evaluate model performance, and make predictions based on user input.

## Project Structure

```
cancer-classification-dashboard
├── app.py                     # Main entry point for the Streamlit application
├── model
│   ├── train_model.py         # Code for training the breast cancer classification model
│   ├── utils.py               # Utility functions for data processing and model evaluation
│   └── __init__.py            # Marks the model directory as a package
├── components
│   ├── data_explorer.py       # Functions for visualizing and exploring the dataset
│   ├── feature_analysis.py     # Visualizations related to feature correlations and importance
│   ├── model_evaluation.py     # Functions for evaluating model performance
│   ├── prediction.py           # Handles prediction functionality
│   └── __init__.py            # Marks the components directory as a package
├── assets
│   └── styles.css             # Custom CSS styles for the Streamlit application
├── requirements.txt            # Lists the Python dependencies required for the project
├── README.md                  # Documentation for the project
└── .gitignore                  # Specifies files and directories to be ignored by Git
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cancer-classification-dashboard
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command in your terminal:
```
streamlit run app.py
```

Once the application is running, you can access it in your web browser at `http://localhost:8501`.

## Features

- **Data Exploration**: Visualize class distributions and feature statistics.
- **Feature Analysis**: Analyze feature correlations and importance.
- **Model Evaluation**: Evaluate model performance with confusion matrices, classification reports, and ROC curves.
- **Prediction**: Input new data and receive predictions from the trained model.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.