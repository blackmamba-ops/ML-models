import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import parallel_backend

# Step 1: Load the dataset
data = pd.read_csv("star_classification.csv")

# Step 2: Prepare the data
X = data.drop(columns=['class'])
y = data['class']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the SVM classifier with parallel processing
with parallel_backend('threading', n_jobs=-1):
    svm_classifier = LinearSVC(C=1.0, random_state=42)
    svm_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = svm_classifier.predict(X_test)


# Step 6: Display the predicted class labels for each test sample
print("Predictions for Test Samples:")
print(y_pred)

# Step 7: Evaluate the SVM classifier
accuracy = accuracy_score(y_test, y_pred)
print("\nSVM Accuracy:", accuracy)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred))


