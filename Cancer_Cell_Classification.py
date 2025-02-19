import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# loading the dataset
data = load_breast_cancer()

# Data Organisation
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

print(label_names)
print(labels)
print(feature_names)
print(features)


# splitting the data
train, test, train_labels, test_labels = train_test_split(features, labels,test_size = 0.33, random_state = 42)

gnb = GaussianNB()

# training the classifier
model = gnb.fit(train, train_labels)

# making the predictions
predictions = gnb.predict(test)
print(predictions)

# evaluating the accuracy
print(accuracy_score(test_labels, predictions))