import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the IMDb movie review dataset
max_features = 5000  # number of most frequent words to use in the dataset
maxlen = 200  # Maximum sequence length (truncate or pad sequences to this length)
batch_size = 32

print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Set the number of training samples
num_train_samples = 5000
x_train = sequence.pad_sequences(x_train[:num_train_samples], maxlen=maxlen)
y_train = y_train[:num_train_samples]

# Pad sequences to ensure all have the same length
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Create the LSTM model
model = Sequential()
model.add(Embedding(max_features, 128))  # Embedding layer for word representations
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer with dropout for regularization
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = model.predict(x_test)

# Convert predicted probabilities to binary class labels (0 or 1)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Print some of the predictions
for i in range(10):
    print(f"Review {i+1}: Predicted Label: {y_pred_binary[i]}, True Label: {y_test[i]}")


# Long Short-Term Memory (LSTM): A type of RNN that addresses the vanishing gradient
# problem and can handle long-term dependencies. LSTMs are commonly used for tasks
# requiring memory, such as language translation and sentiment analysis.