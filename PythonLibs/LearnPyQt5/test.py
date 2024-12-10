import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def window():
    app = QApplication(sys.argv)
    w = QWidget()
    b = QLabel(w)
    b.setText("Hello World!")
    w.setGeometry(100, 100, 200, 50)
    b.move(50, 20)
    w.setWindowTitle("PyQt5")
    w.show()
    sys.exit(app.exec_())


def window2():
    app = QApplication(sys.argv)
    win = QDialog()
    b1 = QPushButton(win)
    b1.setText("Button1")
    b1.move(50, 20)
    b1.clicked.connect(b1_clicked)

    b2 = QPushButton(win)
    b2.setText("Button2")
    b2.move(50, 50)
    b2.clicked.connect(b2_clicked)

    win.setGeometry(100, 100, 200, 100)

    win.setWindowTitle("PyQt5")
    win.show()
    sys.exit(app.exec_())


def b1_clicked():
    print("Button 1 clicked")


def b2_clicked():
    print("Button 2 clicked")


if __name__ == "__main__":
    # window()
    window2()
