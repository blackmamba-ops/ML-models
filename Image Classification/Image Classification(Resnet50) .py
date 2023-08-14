import os
import numpy as np
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

# Set the path to the folder containing the images
image_folder = "images"

# Get the list of image file names in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Set batch size
batch_size = 32

# Create empty lists to store the images and their predictions
images = []
predictions = []

# Process images in batches
for i in range(0, len(image_files), batch_size):
    batch_images = []
    start_index = i
    end_index = min(i + batch_size, len(image_files))

    # Iterate over the images in the current batch
    for j in range(start_index, end_index):
        img_file = image_files[j]
        img_path = os.path.join(image_folder, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        batch_images.append(img_array)

    # Concatenate the image arrays into a single array for the batch
    batch_images = np.concatenate(batch_images, axis=0)

    # Make predictions on the batch of images
    batch_preds = model.predict(batch_images)
    decoded_batch_preds = decode_predictions(batch_preds, top=1)

    # Append the batch predictions to the overall predictions list
    predictions.extend(decoded_batch_preds)

# Iterate over the predictions and print the results
for i, pred in enumerate(predictions):
    image_name = image_files[i]
    predicted_label = pred[0][1]
    confidence = pred[0][2]
    print(f"Image: {image_name} | Predicted Label: {predicted_label} | Confidence: {confidence}")


