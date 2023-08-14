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

# Function to get the color name based on the BGR value
def get_color_name(bgr):
    b, g, r = bgr
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

    # Convert BGR to RGB for displaying
    rgb_color = (prominent_color[2], prominent_color[1], prominent_color[0])
    rgb_color = tuple(np.clip(rgb_color, 0, 255))  # Ensure values are within 0-255 range

    # Create a rectangle image filled with the most prominent color
    rect_image = np.zeros_like(frame)
    rect_image[:] = rgb_color

    # Combine the frame and the rectangle image
    combined_frame = cv2.addWeighted(frame, 0.7, rect_image, 0.3, 0)

    # Display the color name and accuracy
    color_name = get_color_name(prominent_color)
    cv2.putText(combined_frame, f"Prominent Color: {color_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame with the most prominent color
    cv2.imshow("frame", combined_frame)

    # Stop the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
vid.release()
cv2.destroyAllWindows()
