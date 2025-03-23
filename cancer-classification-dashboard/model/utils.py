def load_data():
    from sklearn.datasets import load_breast_cancer
    import pandas as pd

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    return df

def preprocess_data(df):
    from sklearn.model_selection import train_test_split

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, report, cm

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)