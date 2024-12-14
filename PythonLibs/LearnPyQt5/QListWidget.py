import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QListWidget,
    QVBoxLayout,
    QWidget,
    QLabel,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QListWidget Example")
        self.setGeometry(100, 100, 400, 300)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # QListWidget
        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        # Add items to QListWidget
        self.list_widget.addItems(["Item 1", "Item 2", "Item 3", "Item 4"])

        # QLabel to display clicked item
        self.label = QLabel("Click an item to see its text")
        self.layout.addWidget(self.label)

        # Connect itemClicked signal to a slot
        self.list_widget.itemClicked.connect(self.on_item_clicked)

    def on_item_clicked(self, item):
        # Update the label with the clicked item's text
        self.label.setText(f"You clicked: {item.text()}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
