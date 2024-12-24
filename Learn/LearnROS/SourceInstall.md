# Install ROS from source

## ROS Noetic on Debian 12

Install from source:

- [wiki.ros: Installation](http://wiki.ros.org/noetic/Installation/Source)
- [csdn blog: Debian12 通过源码安装 ros-noetic](https://blog.csdn.net/u011391361/article/details/136471637)
- [Compiling ros1 noetic from source on Ubuntu 22.04](https://www.reddit.com/r/ROS/comments/158icpy/compiling_ros1_noetic_from_source_on_ubuntu_2204/)

Debian 12, Bookworm

Dependencies:

```bash
# use apt or pip to install:
python3-
  rosdep
  rosinstall-generator
  vcstools
  vcstool

# init rosdep
sudo rosdep init
rosdep update

# use apt
build-essential
libboost1.74-all-dev

# 版本不能太高, debian12 bookworm default v1.0.0
liblog4cxx10v5_0.10.0
liblog4cxx-dev_0.10.0

# need lower version -> v1.11.2, can download from pkgs.org
libogre-1.12-dev
ogre-1.12-tools

liburdfdom-tools
liburdfdom-headers-dev
liburdfdom-dev

libbz2-dev
libgpgme-dev

liborocos-kdl-dev/stable 1.5.1-2+b4 amd64
  Kinematics and Dynamics Library development files

liborocos-kdl1.5/stable,now 1.5.1-2+b4 amd64 [installed,automatic]
  Kinematics and Dynamics Library runtime

```

Installation:

```bash
# create catkin workspace
mkdir ~/ros_catkin_ws
cd ~/ros_catkin_ws

# download source code for ros noetic, use vcstool, build all od Desktop
# generates rosinstall file for noetic-desktop
rosinstall_generator desktop --rosdistro noetic --deps --tar > noetic-desktop.rosinstall
mkdir src
# use vcs to download all source code base on xx.rosinstall
vcs import --input noetic-desktop.rosinstall ./src

# resolve dependencies (in official doc)
# in debian 12 bookworm, please manually install the dependencies in the former section
rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro noetic -y

# build, use catkin_make_isolate
./src/catkin/bin/catkin_make_isolated --install \
-DCMAKE_BUILD_TYPE=Release \
-DPYTHON_EXECUTABLE=/usr/bin/python3 # use your python3 path
--install-space path/to/install # default is ~/ros_catkin_ws/install_isolated

# after build success, all files have been installed in ~/ros_catkin_ws/install_isolated
# source the setup.sh file, or add it to your ~/.bashrc
source ~/ros_catkin_ws/install_isolated/setup.sh
```

Other problems:

```bash
roscore

# if stack on roscore, do:
pip uninstall rosgraph
sudo apt-get install python3-rosgraph python3-rosgraph-msgs
```
