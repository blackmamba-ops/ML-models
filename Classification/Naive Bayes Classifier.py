import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv("star_classification.csv")

# Step 2: Prepare the data
X = data.drop(columns=['class'])  # Use all columns except 'class' as input features
y = data['class']  # Use the 'class' column as the target variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Naive Bayes classifier
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test)

# Step 6: Display the predicted class labels for each test sample
print("Predictions for Test Samples:")
print(y_pred)
# add (Y_pred.tolist())

# Step 7: Evaluate the Naive Bayes classifier
accuracy = accuracy_score(y_test, y_pred)
print("\nNaive Bayes Accuracy:", accuracy)

print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred))


