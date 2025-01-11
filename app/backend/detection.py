import cv2
from ultralytics import YOLO
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

class ObjectDetector:
    def __init__(self, model_path=r"yolo-Weights/yolov8m.pt"):
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
    def detect_object(self, frame):
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

        return frame

    def run(self, video_path, video_label, status_bar):

        try:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            output_path = video_path.split('.')[0] + "_output.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

                    # Convert frame to QPixmap and update QLabel
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    video_label.setPixmap(pixmap.scaled(video_label.size(), Qt.KeepAspectRatio))

                    # Update the GUI
                    QApplication.processEvents()

                cap.release()
                status_bar.showMessage("Processing Complete")
                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error in detection: {str(e)}")

