# **Image Classification(ResNet50)**

**Import Libraries:** The code starts by importing necessary libraries for image processing and using the ResNet50 model.

**Set Image Folder Path:** The path to the folder containing the images is specified.

Get Image File Names: The code lists all the image files (with '.jpg' extension) present in the specified folder.

**Load Pre-trained ResNet50 Model:** The ResNet50 model is loaded with pre-trained weights from the ImageNet dataset. ResNet50 is a deep convolutional neural network commonly used for image classification tasks.

**Set Batch Size:** A batch size is determined, which defines the number of images processed together in each iteration.

**Empty Lists Initialization:** Empty lists are created to store images and their corresponding predictions.

**Process Images in Batches:** The code iterates through the list of image files in batches. For each batch, it loads images, preprocesses them, and creates a batch of images.

**Concatenate Image Batches:** The individual image arrays in the batch are concatenated to create a single array representing the batch of images.

**Predictions with ResNet50:** The model predicts the labels of the images in the batch using the ResNet50 model. The decode_predictions function is used to decode the model predictions into human-readable labels.

**Append Batch Predictions:** The predictions from the batch are appended to the overall list of predictions.

**Display Predictions:** The code iterates over the list of predictions and displays the image name, predicted label, and confidence level for each image.

Overall, this code utilizes the pre-trained ResNet50 model to classify images in the specified folder. It processes the images in batches to improve efficiency and accuracy, and then displays the predicted labels along with confidence scores for each image. The ResNet50 model has been trained on a large dataset (ImageNet) and can recognize a wide range of objects in images.
ResNet (Residual Network) is a type of Convolutional Neural Network (CNN) architecture. It is a deep neural network architecture that was introduced to address the vanishing gradient problem and enable the training of very deep neural networks.



