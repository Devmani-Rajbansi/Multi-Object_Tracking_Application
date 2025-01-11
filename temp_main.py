from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QFileDialog, QWidget, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import cv2
import sys


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

    def detect_frame(self, frame):
        """Run YOLO model on a single frame and return processed frame."""
        results = self.model.predict(frame, conf=0.5)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

                # Add class label and confidence
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return frame


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Object Tracking")
        self.setGeometry(100, 100, 800, 600)

        # UI Elements
        self.video_label = QLabel("No Video Loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")

        self.load_button = QPushButton("Load Video")
        self.process_button = QPushButton("Process Video")

        # Detector and Timer
        self.detector = ObjectDetector()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_frame)

        self.load_button.clicked.connect(self.load_video)
        self.process_button.clicked.connect(self.start_processing)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.process_button)

        # Central Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Status Bar
        self.statusBar().showMessage("Ready")

        # Video capture
        self.cap = None

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.statusBar().showMessage("Failed to open video file.")
                self.cap = None
                return
            self.statusBar().showMessage(f"Loaded video: {file_name}")
            self.video_label.setText(f"Video: {file_name.split('/')[-1]}")
        else:
            self.statusBar().showMessage("No video selected")

    def start_processing(self):
        if self.cap is not None:
            self.statusBar().showMessage("Processing video...")
            self.timer.start(30)  # Process frame every 30 ms
        else:
            self.statusBar().showMessage("No video loaded. Please load a video first.")

    def process_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.statusBar().showMessage("Processing complete")
            return

        # Run object detection on the frame
        processed_frame = self.detector.detect_frame(frame)

        # Convert frame to QPixmap and display
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
