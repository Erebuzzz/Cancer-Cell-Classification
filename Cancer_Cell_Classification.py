import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import os

# Create a folder to save plots if it doesn't exist
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Function to create a styled title for terminal output
def print_title(title):
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")

# Load and explore the dataset
print_title("BREAST CANCER CLASSIFICATION")
print_title("1. DATA EXPLORATION")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Create a separate dataframe for visualization that includes the string labels
df_viz = df.copy()
df_viz['diagnosis_label'] = df_viz['diagnosis'].map({0: 'Malignant', 1: 'Benign'})

# Display basic dataset information
print(f"Dataset Shape: {df.shape}")
print(f"Number of Benign samples: {sum(df['diagnosis'] == 1)}")
print(f"Number of Malignant samples: {sum(df['diagnosis'] == 0)}")
print("\nFeature Statistics:")
print(df.describe().T)

# Visualize class distribution
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='diagnosis_label', data=df_viz, palette='viridis')
plt.title('Class Distribution in Breast Cancer Dataset')
plt.xlabel('Diagnosis')
plt.ylabel('Count')

# Add count labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom', 
                xytext = (0, 5), textcoords = 'offset points')
                
plt.savefig(os.path.join(plot_dir, "class_distribution.png"), dpi=300, bbox_inches='tight')

# Visualize feature correlations with target - use only numeric columns
print_title("2. FEATURE ANALYSIS")

# Calculate correlation using only numeric columns
plt.figure(figsize=(12, 10))
correlation = df.corr()  # This will only use numeric columns
mask = np.triu(correlation)
sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "feature_correlation_matrix.png"), dpi=300, bbox_inches='tight')

# Feature importance visualization
plt.figure(figsize=(12, 8))
correlation_with_target = correlation['diagnosis'].sort_values(ascending=False)
correlation_with_target = correlation_with_target.drop('diagnosis')
sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index, palette='viridis')
plt.title('Feature Correlation with Diagnosis')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "feature_correlation_with_diagnosis.png"), dpi=300, bbox_inches='tight')

# Data Preparation
print_title("3. DATA PREPARATION & MODELING")

X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create a pipeline with preprocessing and multiple models
models = {
    'Gaussian Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

# Dictionary to store results
results = {}
best_model = None
best_score = 0

for name, model in models.items():
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_prob': pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    }
    
    # Track best model
    if accuracy > best_score:
        best_score = accuracy
        best_model = name
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and display confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"), dpi=300, bbox_inches='tight')

# Visualize model comparison
print_title("4. MODEL COMPARISON")

# Compare model accuracies
plt.figure(figsize=(10, 6))
accuracies = [results[model]['accuracy'] for model in models.keys()]
sns.barplot(x=list(models.keys()), y=accuracies, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0.9, 1.0)  # Adjust y-axis for better visualization
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "model_accuracy_comparison.png"), dpi=300, bbox_inches='tight')

# ROC curve comparison
plt.figure(figsize=(10, 8))
for name in models.keys():
    if results[name]['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "roc_curve_comparison.png"), dpi=300, bbox_inches='tight')

# Feature importance for the best model (if applicable)
print_title("5. FEATURE IMPORTANCE ANALYSIS")

if best_model == 'Random Forest' or best_model == 'Gradient Boosting':
    # Extract the classifier from the pipeline
    best_classifier = results[best_model]['pipeline'].named_steps['classifier']
    
    # Get feature importances
    importances = best_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices][:15], y=[X.columns[i] for i in indices][:15], palette='viridis')
    plt.title(f'Top 15 Feature Importances - {best_model}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"feature_importances_{best_model.replace(' ', '_').lower()}.png"), dpi=300, bbox_inches='tight')

# Final model evaluation on test set
print_title("6. FINAL MODEL EVALUATION")

best_pipeline = results[best_model]['pipeline']
y_pred = results[best_model]['y_pred']
y_prob = results[best_model]['y_prob']

print(f"Best Model: {best_model}")
print(f"Accuracy: {best_score:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Precision-Recall curve
if y_prob is not None:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')

print_title("SUMMARY")
print(f"The best performing model is {best_model} with an accuracy of {best_score:.4f}")
print("This model can effectively classify breast cancer tumors as benign or malignant")
print("based on features extracted from digitized images of fine needle aspirates (FNA) of breast mass.")

# Save all figures that were created and stored in plt.get_fignums()
for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(plot_dir, f"plot_{i}.png"), dpi=300, bbox_inches='tight')
    
plt.show()