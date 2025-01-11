# All imports
import sys
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from app.backend.detection import ObjectDetector
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Object Tracking")
        self.setGeometry(100, 100, 1000, 750)

        self.video_label = QLabel("No Video Loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.video_label.setFixedSize(980, 720)

        self.load_button = QPushButton("Load Video")
        self.process_button = QPushButton("Process Video")
        self.stop_button = QPushButton("Stop")

        self.detector = ObjectDetector()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_frame)

        self.load_button.clicked.connect(self.load_video)
        self.process_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.stop_button)

        # Central Wideget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Status Bar
        self.statusBar().showMessage("Ready")

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
            self.timer.start(10)
        else:
            self.statusBar().showMessage("No video loaded. Please load a video first.")

    def stop_processing(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("No Video Loaded")
        self.statusBar().showMessage("Processing stopped")

    def process_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.statusBar().showMessage("Processing complete")
            return

        # Run object detection on the frame
        processed_frame = self.detector.detect_object(frame)

        # Convert frame to QPixmap and display
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))