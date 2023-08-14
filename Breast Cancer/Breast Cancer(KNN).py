import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Breast_Cancer.csv')

# Separate features and target
X = data.drop(columns=['Status'])
y = data['Status']

# Encode the categorical features using one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert data to numpy arrays
X_train_np = X_train.values
X_val_np = X_val.values
y_train_np = y_train.values
y_val_np = y_val.values

# Create and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_np, y_train_np)

# Make predictions on the validation set
y_pred = knn_classifier.predict(X_val_np)

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_val_np, y_pred)
print("Validation accuracy:", accuracy)

# Convert the test data to NumPy array
X_test_np = np.ascontiguousarray(X_test)

# Make predictions on the test set
y_pred_test = knn_classifier.predict(X_test_np)

# Calculate accuracy on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test accuracy:", accuracy_test)

# Create a DataFrame to store predictions with assigned patient numbers
predictions_df = pd.DataFrame({'Prediction': y_pred_test})

# Display all predictions (no truncation)
pd.set_option('display.max_rows', None)
print("Predictions on the test set:")
print(predictions_df)


