# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = pd.read_csv(url)

# Print the first few rows of the dataset
print(data.head())
# Print information about the dataset, including columns and data types
print(data.info())

# Preprocessing: Handle missing values and convert categorical variables to numeric
data["Age"].fillna(data["Age"].median(), inplace=True)
data["Sex"] = data["Sex"].map({"female": 0, "male": 1})

# Separate features and target variable
X = data.drop(["Survived", "Name"], axis=1)
y = data["Survived"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier on the training data
random_forest_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = random_forest_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print the predicted class labels along with the true class labels
print("\nPredicted Class Labels | True Class Labels")
for pred, true in zip(y_pred, y_test):
    pred_label = "Survived" if pred == 1 else "Not Survived"
    true_label = "Survived" if true == 1 else "Not Survived"
    print(f"{pred_label} | {true_label}")
