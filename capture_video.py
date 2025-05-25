# import cv2
# from ultralytics import YOLO

# # Load the YOLOv11 model
# model = YOLO('yolo11n.pt')

# # Open a video stream (webcam)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     #Perform object detection
#     results = model(frame)
#     # Draw bounding boxes on the frame
#     for result in results:
#         for box in result.boxes:
#             # Safely unpack xyxy values
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers for OpenCV

#             # Draw rectangle
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Display the frame with detections
#     cv2.imshow('YOLOv11 Live Monitoring', frame)

#     if cv2.waitKey(1) & 0xFF != 0xFF:
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO('yolo11n.pt')

# Open a video stream (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # class ID

            # Only process if class is 'person' (class ID 0 in COCO)
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optionally: display label "person"
                cv2.putText(frame, "person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv11 - Person Detection Only', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
