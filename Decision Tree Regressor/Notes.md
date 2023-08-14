# **Decision Tree Regressor**


This code snippet demonstrates the process of building and evaluating a Decision Tree Regressor model using a dataset of car sales. Let's break down the code step by step:

**Import Libraries:**

pandas: Library for data manipulation and analysis.
train_test_split: Function for splitting data into training and testing sets.
DecisionTreeRegressor: Class for the Decision Tree Regressor algorithm.
mean_squared_error: Function to calculate the mean squared error for regression evaluation.

**Load and Prepare the Dataset:**

data = pd.read_csv('Car_sales.csv') loads the car sales dataset from a CSV file.
X contains selected features, and y contains the target variable ('Sales_in_thousands').

**Convert Categorical Variables:**

pd.get_dummies(X, columns=['Manufacturer', 'Model'], drop_first=True) converts categorical variables (manufacturer and model) into numerical format using one-hot encoding.

**Check for Missing Values:**

missing_values = X.isnull().sum() calculates the count of missing values in each column of the feature matrix X.

**Remove Missing Values or Impute Them:**

X.dropna(inplace=True) drops rows with missing values from the feature matrix X.
y = y[X.index] updates the target vector y to match the indices of the remaining rows in X.

**Split the Dataset:**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) splits the dataset into training and testing sets.

**Define Parameter Grid:**

Define a grid of hyperparameters to search over using Grid Search. Here, you specify different values for max_depth, min_samples_split, and min_samples_leaf.

**Create Grid Search Instance:**

Create an instance of GridSearchCV, passing the DecisionTreeRegressor model, the parameter grid, the number of cross-validation folds (cv=5), the scoring metric (scoring='neg_mean_squared_error'), verbosity level (verbose=2), and the number of parallel jobs (n_jobs=-1).

**Create and Train the Decision Tree Regressor:**

regressor = DecisionTreeRegressor() creates an instance of the Decision Tree Regressor.
regressor.fit(X_train, y_train) trains the model on the training data.

**Fit Grid Search to Data:** 

Fit the Grid Search instance to the training data, performing cross-validation and searching for the best combination of hyperparameters.

**Get Best Model:** 

Retrieve the best model from the Grid Search, which corresponds to the combination of hyperparameters that yielded the lowest mean squared error.

**Make Predictions on the Test Set:**

y_pred = regressor.predict(X_test) makes predictions using the trained Decision Tree Regressor on the test data.

**Print Predictions:**

The code prints the true and predicted sales values side by side for each data point in the test set.

**Evaluate the Model:**

mse = mean_squared_error(y_test, y_pred) calculates the mean squared error between the true and predicted sales values.
The mean squared error is printed to evaluate the performance of the Decision Tree Regressor.

In summary, this code demonstrates how to preprocess a dataset, train a Decision Tree Regressor model for predicting car sales, and evaluate the model's performance using mean squared error.

**The code is little bit off in predictions due to the poor data quality and lot of missing values,to increase Accuracy,refer Accuracy issues.md file**
