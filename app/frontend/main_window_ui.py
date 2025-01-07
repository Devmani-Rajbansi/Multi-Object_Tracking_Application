# All imports
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, QPushButton,
                             QVBoxLayout, QWidget)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Object Tracking")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel("No Video Loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.video_label.setFixedSize(700, 400)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.load_button)

        # Central Wideget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Status Bar
        self.statusBar().showMessage("Ready")

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.statusBar().showMessage(f"Loaded video: {file_name}")
            self.video_label.setText(f"Video: {file_name.split('/')[-1]}")
        else:
            self.statusBar().showMessage("No video selected")

