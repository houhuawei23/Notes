import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
)
from PyQt5.QtGui import QPixmap, QPainterPath, QPolygonF, QColor, QBrush
from PyQt5.QtCore import QPointF


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Draw Points and Polygons on Image")
        self.setGeometry(100, 100, 800, 600)

        # QGraphicsView and QGraphicsScene
        self.graphics_view = QGraphicsView(self)
        self.setCentralWidget(self.graphics_view)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # Load image
        pixmap = QPixmap("dog.jpg")  # Replace with your image path
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        # Draw points
        self.draw_points([(100, 100), (150, 150), (200, 50)])

        # Draw polygon
        self.draw_polygon([(300, 300), (400, 350), (350, 450), (250, 400)])

    def draw_points(self, points):
        """Draw points on the image using QGraphicsPathItem."""
        path = QPainterPath()
        for x, y in points:
            path.addEllipse(QPointF(x, y), 5, 5)  # Add small circles at the points
        path_item = QGraphicsPathItem(path)
        path_item.setBrush(QColor.fromRgb(255, 0, 0))  # Set the color of the points
        self.scene.addItem(path_item)

    def draw_polygon(self, points):
        """Draw a polygon on the image using QGraphicsPolygonItem."""
        polygon = QPolygonF([QPointF(x, y) for x, y in points])
        polygon_item = QGraphicsPolygonItem(polygon)
        # polygon_item.setBrush(QBrush(QColor.transparent)) # Transparent fill
        polygon_item.setPen(QColor.fromRgb(0, 0, 255))  # Blue border
        self.scene.addItem(polygon_item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
