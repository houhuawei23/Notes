import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QFileDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QPixmap


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # QGraphicsView
        self.graphics_view = QGraphicsView(self)
        self.layout.addWidget(self.graphics_view)

        # Button to open file dialog
        self.open_button = QPushButton("Open Image", self)
        self.layout.addWidget(self.open_button)
        self.open_button.clicked.connect(self.open_image)

        # QGraphicsScene
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

    def open_image(self):
        # Open a file dialog to select an image
        file_name, selectedFilter = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        print(f"Selected file: {file_name}")
        print(f"Selected filter: {selectedFilter}")
        
        if file_name:  # If a file is selected
            pixmap = QPixmap(file_name)  # Load the image
            self.scene.clear()  # Clear the previous scene content
            pixmap_item = QGraphicsPixmapItem(pixmap)  # Create a pixmap item
            self.scene.addItem(pixmap_item)  # Add the pixmap item to the scene
            self.graphics_view.fitInView(
                pixmap_item, mode=0
            )  # Fit the image within the view


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
