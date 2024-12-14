import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("PyQt5 Image Example")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        # Create a QLabel widget
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 700, 500)  # x, y, width, height

        # Load the image
        pixmap = QPixmap("dog.jpg")  # Replace with your image file path

        # Resize the QLabel to fit the image
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # Allows scaling of the image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
