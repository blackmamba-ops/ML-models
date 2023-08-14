import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the dataset
data = pd.read_csv("Housing.csv")

# Separate features and target variable
X = data.drop("price", axis=1)
y = data["price"]

# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)

# Identify categorical columns
categorical_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]

# Perform one-hot encoding
encoder = OneHotEncoder(sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# Concatenate the encoded features with remaining numerical features
X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=100)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)

# Scale the features to a range of 0-1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, verbose=1)

# Evaluate the model on the validation set
loss_val = model.evaluate(X_val_scaled, y_val, verbose=0)
print("Mean Squared Error (Validation):", loss_val)

# Predict on all data points
X_all_scaled = scaler.transform(X)
predictions = model.predict(X_all_scaled)

# Print predicted prices for all data points
for i, pred in enumerate(predictions):
    print("Predicted price for data point {}: {}".format(i + 1, pred[0]))
