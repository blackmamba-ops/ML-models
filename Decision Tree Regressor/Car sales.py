import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load and prepare the dataset
data = pd.read_csv('Car_sales.csv')

X = data[['Manufacturer', 'Model', 'Price_in_thousands', 'Engine_size', 'Horsepower',
          'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency']]
y = data['Sales_in_thousands']

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Manufacturer', 'Model'], drop_first=True)

# Step 2: Check for missing values
missing_values = X.isnull().sum()
print("Missing Values:\n", missing_values)

# Step 3: Remove missing values or impute them
X.dropna(inplace=True)
y = y[X.index]  # Update y accordingly after dropping rows in X

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define the parameter grid for Grid Search
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Step 6: Create the Grid Search instance
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Step 7: Fit the Grid Search to the data
grid_search.fit(X_train, y_train)

# Step 8: Get the best model from the search
best_model = grid_search.best_estimator_

# Step 9: Make predictions on the test set
y_pred = best_model.predict(X_test)

# Step 10: Print the predictions and evaluate the model
print("Predictions:")
for true_value, pred_value in zip(y_test, y_pred):
    print(f"True Value: {true_value:.2f} | Predicted Value: {pred_value:.2f}")

# Step 11: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
