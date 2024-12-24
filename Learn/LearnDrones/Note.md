# Learn Drones

- [HKUST-Aerial-Robotics](https://github.com/HKUST-Aerial-Robotics)
- [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab)
- [Fast-Drone-250](https://github.com/ZJU-FAST-Lab/Fast-Drone-250)
- [ego-planner-swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)
- [ego-planner](https://github.com/ZJU-FAST-Lab/ego-planner)
- [multi_uav_simulator](https://github.com/malintha/multi_uav_simulator)


## Tools

- RealSense Driver
  - [github](https://github.com/IntelRealSense/librealsense)
  - [source installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)
  - The shared object will be installed in `/usr/local/lib`, header files in `/usr/local/include`.
  - sudo apt install libpcl-dev

- Mavros: Micro Air Vehicle Robot Operating System
  - **MAVLink** extendable communication node for ROS with proxy for **Ground Control Station**.
  - This package provides communication driver for various autopilots with MAVLink communication protocol. Additional it provides UDP MAVLink bridge for ground control stations (e.g. QGroundControl).
  - mavros用于无人机通信，可以将飞控与主控的信息进行交换
- [ceres-solver](http://ceres-solver.org/):
  - [googlesource](https://ceres-solver.googlesource.com/ceres-solver)
  - An open source lib for modeling and solving large, complicated optimization problems.
  - It can be used to solve Non-linear Least Squares problems with bounds constraints and general unconstrained optimization problems. 它可用于求解具有边界约束的非线性最小二乘法问题和一般的无约束优化问题。
- [glog](https://github.com/google/glog)
  - C++ implementation of the Google logging module.

```bash
pip uninstall em
pip install empy
```

```bash
# install ros noetic dependencies
# [ddynamic_reconfigure](https://github.com/pal-robotics/ddynamic_reconfigure)
catkin_make_isolated --install -DPYTHON_EXECUTABLE=/usr/bin/python3 --force-cmake

# must be: dont be higher than 0.10.0
Package: liblog4cxx-dev
Version: 0.10.0-15ubuntu2
```