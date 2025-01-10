import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
            "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

    def detect_objects(self, video_path):

        try:

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run the YOLO model on the current frame
                results = self.model.predict(frame, conf=0.5)

                for result in results:
                    # Get detection boxes
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        # Extract box coordinates and confidence score
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])

                        # Draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

                        # Add the confidence score and class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Display the frame with bounding boxes
                cv2.imshow("Detection", frame)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error in detection: {str(e)}")

if __name__ == "__main__":
    detector = ObjectDetector(model_path=r"yolo-Weights/yolov8m.pt")
    video_path = r"D:\Projects\ComputerVisionProjects\YoloV3_Deepsort\Videos\cars.mp4"  # Replace with the path to your video file
    result = detector.detect_objects(video_path)
    print(result)