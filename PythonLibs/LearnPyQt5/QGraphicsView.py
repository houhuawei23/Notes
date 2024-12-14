import sys
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
)
from PyQt5.QtGui import QPixmap


class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()

        # Create a QGraphicsScene
        self.scene = QGraphicsScene()

        # Load the image using QPixmap
        pixmap = QPixmap("dog.jpg")  # Replace with your image file path

        # Add the image to the scene
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)

        # Set the scene to the QGraphicsView
        self.setScene(self.scene)

        # Set a fixed size for the QGraphicsView
        # self.setFixedSize(pixmap.width() + 10, pixmap.height() + 10)
        self.setFixedSize(800, 600)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.setWindowTitle("QGraphicsScene Image Example")
    viewer.show()
    sys.exit(app.exec_())
