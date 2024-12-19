# Qt Style Sheets

- [stylesheet-examples](https://doc.qt.io/qt-5/stylesheet-examples.html)
- [Customizing Qt Widgets Using Style Sheets](https://doc.qt.io/qt-5/stylesheet-customizing.html)
- [Qt Style Sheets Reference](https://doc.qt.io/qt-5/stylesheet-reference.html)
- [blog: 使用 QSS 美化 PyQt5 界面](https://blog.csdn.net/mziing/article/details/119054948)
- [qt-material](https://qt-material.readthedocs.io/)

Qt-Meterial:

```bash
pip install qt-material
```

```python
import sys
from PySide6 import QtWidgets
# from PySide2 import QtWidgets
# from PyQt5 import QtWidgets
from qt_material import apply_stylesheet

# create the application and the main window
app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()

# setup stylesheet
apply_stylesheet(app, theme='dark_teal.xml')

# run
window.show()
app.exec_()
```
