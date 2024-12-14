from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)

Form = QtWidgets.QWidget()
Form.setWindowTitle("oxxo.studio")
Form.resize(300, 200)

menubar = QtWidgets.QMenuBar(Form)  # 建立 menubar

menu_file = QtWidgets.QMenu("File")  # 建立一個 File 選項 ( QMenu )

action_open = QtWidgets.QAction("Open")  # 建立一個 Open 選項 ( QAction )
menu_file.addAction(action_open)  # 將 Open 選項放入 File 選項裡

action_close = QtWidgets.QAction("Close")  # 建立一個 Close 選項 ( QAction )
menu_file.addAction(action_close)  # 將 Close 選項放入 File 選項裡

menubar.addMenu(menu_file)  # 將 File 選項放入 menubar 裡

Form.show()
sys.exit(app.exec_())
