import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.losses import BinaryCrossentropy

# Load the data from Reddit and Twitter datasets
reddit_data = pd.read_csv("Reddit_Data.csv")
twitter_data = pd.read_csv("Twitter_Data.csv")

# Combine the data from both datasets
combined_data = pd.concat([reddit_data[['clean_comment', 'category']], twitter_data[['clean_text', 'category']]])

# Print information about the dataset, including columns and data types
print(combined_data.info())

# Preprocess the text data
def preprocess_text(text):
    if pd.notna(text):
        text = text.lower()
    else:
        text = ""
    return text

combined_data['clean_comment'] = combined_data['clean_comment'].apply(preprocess_text)

# Separate the features and labels
X = combined_data['clean_comment'].values
y = combined_data['category'].values

# Convert the labels to binary format: -1 to [0, 0], 1 to [0, 1]
y_binary = np.array([[1, 0] if label == -1 else [0, 1] for label in y])

# Tokenize the text data and pad the sequences
vocab_size = 10000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Split the data into training and testing sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(X_padded, y_binary, test_size=0.2, random_state=42)

# Build the model
embedding_dim = 100
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=64))
model.add(Dense(units=2, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(train_sequences, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Print predictions on the test set
predictions = model.predict(test_sequences)
predicted_labels = [np.argmax(pred) - 1 for pred in predictions]

# Convert predicted_labels to -1, 0, or 1 format
predicted_labels = np.array(predicted_labels)

print("Predicted Labels:")
print(predicted_labels)





