# **House Price Prediction**


**Import Libraries:**

TensorFlow is imported for building and training the neural network.
pandas is imported to handle data in DataFrame format.
train_test_split from sklearn.model_selection is imported for splitting the dataset.
MinMaxScaler and OneHotEncoder from sklearn.preprocessing are imported for data scaling and one-hot encoding.

**Load and Prepare Data:**

The dataset is loaded from a CSV file using pd.read_csv.
Features (X) and target variable (y) are separated.

**Identify Categorical Columns:**

A list (categorical_cols) containing the names of categorical columns is created.

**Perform One-Hot Encoding:**

OneHotEncoder is instantiated with sparse=False to perform one-hot encoding.
Categorical features in X are encoded using one-hot encoding.
Encoded features are concatenated with the remaining numerical features.

**Split Data into Train-Validation-Test Sets:**

The dataset is split into training, validation, and testing sets using train_test_split.
The data is split twice: first into training and a temporary set, then the temporary set into validation and testing.

**Scale Features:**

MinMaxScaler is used to scale features to a range of 0 to 1.
Training data is used to fit the scaler, and then it's used to transform validation and test data.

**Define Neural Network Model:**

A sequential neural network model is defined using TensorFlow's Sequential API.
It consists of multiple layers: input, hidden, and output.
Activation functions like ReLU are applied in hidden layers.

**Compile the Model:**

The model is compiled using an optimizer (e.g., Adam) and a loss function (e.g., mean squared error) suitable for regression tasks.

**Train the Model:**

The model is trained using the scaled training data.
Training continues for a specified number of epochs and batch size.

**Evaluate on Validation Set:**

The trained model is evaluated on the scaled validation data.
The evaluation metric (e.g., mean squared error) gives an indication of how well the model performs on unseen data.

**Predict on All Data Points:**

The trained model is used to make predictions on all data points, including training, validation, and test data.

**Print Predicted Prices:**

The predicted prices for all data points are printed using a loop.
Each predicted price is associated with its corresponding data point index.
In summary, the code follows the typical machine learning pipeline for a regression task. It preprocesses data, constructs a neural network model, trains it, evaluates its performance, and generates predictions for all data points. The goal is to predict housing prices based on various features.


To convert the provided deep neural network model into a traditional linear regression model, you need to remove the
hidden layers and use a linear activation function in the output layer.

Define the model (linear regression)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', input_shape=(X_train_scaled.shape[1],)),
])


The linear activation function is simply the identity function, which means it returns
the input as it is without any transformation.