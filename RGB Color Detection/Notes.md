# **RGB COLOR DETECTION**

**Importing Libraries:**

cv2: OpenCV library for computer vision tasks.
numpy as np: Numpy library for numerical computations.

**get_most_prominent_color(frame) Function:**

This function takes a frame (image) as input and calculates the most prominent color in the image using K-means clustering.
The frame is reshaped to a 2D array of pixels.
K-means clustering is applied to group pixels into a specified number of clusters (colors).
The cluster with the highest count is considered as the most prominent color, and its center is returned.

**get_color_name(bgr) Function:**

This function takes a BGR (Blue, Green, Red) color value as input and determines the color's name (Red, Green, Blue, Other) based on the channel values.

**Capturing Video from Webcam:**

cv2.VideoCapture(0): Opens the default camera (webcam) for capturing video.

**Main Loop:**

Continuously captures frames from the webcam.

**Getting Most Prominent Color:**

Calls get_most_prominent_color() to find the most prominent color in the current frame.

**Converting BGR to RGB:**

Converts the BGR color format to RGB format for displaying on the screen.
Ensures that the color values are within the valid range of 0-255.

**Creating Rectangle Image:**

Creates a new image filled with the most prominent color using numpy operations.
This will be used to display a rectangle with the most prominent color.

**Combining Frame and Rectangle Image:**

Combines the original frame and the rectangle image to highlight the most prominent color.
cv2.addWeighted() function is used to blend the images with specific weights.

**Displaying Color Name and Combined Frame:**

Adds text to the combined frame indicating the color name.
Displays the combined frame with the most prominent color and the color name.

**Exiting the Program:**

The program stops when the 'q' key is pressed.
Releases the video capture and closes all windows.
In summary, this code captures video from the webcam, identifies the most prominent color in each frame using K-means clustering, creates a rectangle image filled with the prominent color, and displays the combined frame with the color name. The program continues to run until the 'q' key is pressed.




