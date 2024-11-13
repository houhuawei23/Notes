debian系统对中文输入法的支持少之又少，很多人会选择使用搜狗，但是对于大多数来说，会有各种各样的问题，所以这里将会介绍使用系统自带的fcitx输入法。
 

首先软件源更新（选）：

1.在终端中输入： sudo gedit  /etc/apt/sources.list

2.在打开的文本中删除全部内容，粘贴上以下文本：
```
deb http://mirrors.163.com/debian/ jessie-updates main non-free contrib
deb http://mirrors.163.com/debian/ jessie-backports main non-free contrib
deb-src http://mirrors.163.com/debian/ jessie main non-free contrib
deb-src http://mirrors.163.com/debian/ jessie-updates main non-free contrib
deb-src http://mirrors.163.com/debian/ jessie-backports main non-free contrib
deb http://mirrors.163.com/debian-security/ jessie/updates main non-free contrib
deb-src http://mirrors.163.com/debian-security/ jessie/updates main non-free contrib
deb http://ftp.cn.debian.org/debian wheezy main contrib non-free
```
（此处包含163以及debian官方软件源）

3.点击保存，并关闭，回到终端，进行软件源同步

输入指令：sudo apt-get update

apt-get install fcitx-ui-classic && apt-get install fcitx-ui-light

*  5.点击菜单，找到应用: 输入法，并打开。

*  6.在用户设置中 点击  fctix选项  ，并点击确定。

*  7.根据输入法配置中的提示，打开终端，输入指令（根据自身要求选择）：

            sudo apt-get install fcitx-sunpinyin fcitx-googlepinyin fcitx-pinyin

（这里有三种拼音输入法：fcitx-sunpinyin ，fcitx-googlepinyin 和 fcitx-pinyin ，不需要的可以删掉）

               sudo apt-get fcitx-table-wubi fcitx-table-wbpy

(两种五笔输入法：fcitx-table-wubi 和fcitx-table-wbpy)

                sudo apt-get fcitx-table-cangjie

(繁体中文输入，只有一种)

*  8.安装通用的输入法码表: fcitx-table* 套件（必装！）

               sudo apt-get fcitx-table*

*  9.应用程序支持（必装！）

               sudo apt-get install fcitx-frontend-gtk2 fcitx-frontend-gtk3 fcitx-frontend-qt4

(fcitx-frontend-gtk2 和 fcitx-frontend-gtk3 必装，同时  fcitx-frontend-qt4  也建议一起装上)

*  10.最后重启，根据自己的快捷键启动输入法（默认    Ctrl+空格   ）

在右下脚会有小键盘，右键 --配置    可以选择各种输入选项
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/ieeso/article/details/105274943