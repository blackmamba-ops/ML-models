import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv('creditcard.csv')

# Assuming 'Class' column represents fraud labels (0: normal, 1: fraud)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train an Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination based on dataset characteristics
model.fit(X_train)

# Predict anomalies (fraudulent transactions)
predictions = model.predict(X_test)

# Convert predictions: 1 for normal, -1 for anomaly
predictions = [1 if pred == 1 else 0 for pred in predictions]

# Print the predictions
print("Predictions for Anomalies (1 for normal, 0 for anomaly):")
print(predictions)



