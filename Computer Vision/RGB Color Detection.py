import cv2
import numpy as np

# Function to get the most prominent color
def get_most_prominent_color(frame):
    pixels = frame.reshape(-1, 3)
    num_clusters = 3  # Number of clusters (colors) to find
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    prominent_color = centers[np.argmax(counts)]
    return prominent_color

# Function to get the color name based on the RGB value
def get_color_name(rgb):
    r, g, b = rgb
    if r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    else:
        return "Other"

# Taking input from webcam
vid = cv2.VideoCapture(0)

while True:
    # Capturing the current frame
    _, frame = vid.read()

    # Get the most prominent color
    prominent_color = get_most_prominent_color(frame)

    # Draw a rectangle with the most prominent color
    cv2.rectangle(frame, (10, 50), (110, 150), prominent_color.tolist(), -1)

    # Display the color name and accuracy
    color_name = get_color_name(prominent_color)
    cv2.putText(frame, f"Prominent Color: {color_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame with the most prominent color
    cv2.imshow("frame", frame)

    # Stop the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
vid.release()
cv2.destroyAllWindows()


