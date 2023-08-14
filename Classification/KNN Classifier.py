import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv("star_classification.csv")

# Step 2: Prepare the data
X = data.drop(columns=['class'])  # Use all columns except 'class' as input features
y = data['class']  # Use the 'class' column as the target variable

# Step 3: Convert the target variable to numerical using label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Convert the data to NumPy arrays with 'C' memory layout
X = np.array(X, order='C')
y_encoded = np.array(y_encoded, order='C')

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 6: Create and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred_encoded = knn_classifier.predict(X_test)

# Step 8: Convert the predicted numerical labels back to categorical form
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Step 9: Display the predicted class labels for each test sample
print("Predictions for Test Samples:")
print(y_pred)

# Step 10: Evaluate the KNN classifier
accuracy = accuracy_score(y_test, y_pred_encoded)
print("\nKNN Accuracy:", accuracy)

print("KNN Classification Report:")
print(classification_report(y_test, y_pred_encoded))
