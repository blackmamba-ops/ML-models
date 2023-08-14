import cv2
import torch
import torch.hub

# Load the YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Set the sharing strategy for PyTorch DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')

while True:
    # Get a frame from the webcam
    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model(frame)

    # Get the detections
    detections = results.pred[0]

    # Loop over the detections and draw rectangles on the frame
    for detection in detections:
        label = results.names[int(detection[5])]
        confidence = detection[4].item()
        bbox = detection[:4].cpu().numpy()

        if confidence > 0.3:  # Lower the confidence threshold to display more detections
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0)  # Green color for the bounding box
            thickness = 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Show the frame with detections
    cv2.imshow('Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()


