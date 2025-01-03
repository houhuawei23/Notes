# ROS: Robot Operating System

- [index.ros.org](https://index.ros.org/)
- [wiki.ros: Tutorials](https://wiki.ros.org/ROS/Tutorials)
- [wiki.ros: cn Introduction](https://wiki.ros.org/cn/ROS/Introduction)
- [github: ros-infrastructure](https://github.com/ros-infrastructure)
- [github: ROS core stacks](https://github.com/ros)
- [github: ros-gbp](https://github.com/ros-gbp)
- [rep: learn-ros](https://github.com/iConor/learn-ros/)
- [blog: ros-tutorials](https://songapore.gitbook.io/ros-tutorials)

ROS（机器人操作系统）提供库和工具来帮助软件开发人员创建机器人应用程序。它提供硬件抽象、设备驱动程序、库、可视化工具、消息传递、包管理等。

## ROS Releases/Distributions

- [ros: Distributions](https://wiki.ros.org/Distributions)
- [ros2: Releases](https://docs.ros.org/en/humble/Releases.html)
- [ros - os version match](https://blog.csdn.net/maizousidemao/article/details/119846292)

ROS:

- ROS Box Turtle 2010.03.02
- ROS C Turtle 2010.08.02
- ROS Diamondback 2011.03.01
- ROS Electric Emys 2011.08.30
- ROS Fuerte Turtle 2012.04.23
- ROS Groovy Galapagos 2012.12.31
- ROS Hydro Medusa 2013.09.04
- ROS Indigo Igloo 2014.07.22
- ROS Jade Turtle 2015.05.23
- ROS Kinetic Kame 2016.05.23
- ROS Lunar Loggerhead 2017.05.23
- ROS Melodic Morenia 2018.05.23
- ROS Noetic Ninjemys 2020.05.23

ROS2:

- ROS 2 Ardent Apalone 2017.12.08
- ROS 2 Bouncy Bolson 2018.05.31
- ROS 2 Crystal Clemmys 2018.12.12
- ROS 2 Dashing Diademata 2019.05.31
- ROS 2 Eloquent Elusor 2019.12.12
- ROS 2 Foxy Fitzroy 2020.06.05
- ROS 2 Galactic Geochelone 2021.05.23
- ROS 2 Humble Hawksbill 2022.05.23
- ROS 2 Iron Irwini 2023.05.23
- ROS 2 Jazzy Jalisco 2024.05.23

There is a new ROS 2 distribution released yearly on May 23rd (World Turtle Day).

### ROS Noetic Ninjemys

- [wiki](http://wiki.ros.org/noetic)

ROS Noetic Ninjemys is the thirteenth ROS distribution release. It was released on May 23rd, 2020.

## [Concepts](https://wiki.ros.org/ROS/Concepts)

- REP: ROS Enhancement Proposals
  - REPs are documents that define standards, conventions, and best practices for the ROS ecosystem. They are similar to RFCs (Request for Comments) in the internet protocol community or PEPs (Python Enhancement Proposals) in the Python community.
- Filesystem Level
  - Packages
  - Metapackages
  - Package Manifests
  - Repositories
  - Message (msg) types: Message Description, stored in `my_package/msg/MyMessageType.msg`
  - Service (srv) types: Service Description, stored in `my_package/srv/MyServiceType.srv`
- Graph Level
  - Nodes: process that performs computation
  - Master: provides name registration and lookup to the rest of the Computation Graph
  - Parameter Server: allows data to be stored by key in a central location
  - Messages: data structure for communication
  - Topics: messages are routed via a transport system with publish/subscribe semantics
    - a node sends a message by publishing it to a given topic
    - a node receives a message by subscribing to the appropriate topic
  - Services: request/reply is done via a service
    - a node offers a service under a specific name
    - a client uses the service by sending the request message and awaiting the reply
  - Bags: a format for saving and playing back ROS message data
    - mechanism for storing ROS message data, such as sensor data
- Community Level
  - Distributions
  - Repositories
  - ROS Wiki
  - ...
- names: Package Resource Names and Graph Resource Names
  - Graph Resource Names:
    - provides a hierarchical naming structure that is used for all resources in ROS Computation Graph
      - Graph Resource Names are an important mechanism in ROS for providing `encapsulation`.
      - Each resource is defined within a namespace, which it may share with many other resources.
      - In general, resources can create resources within their namespace and they can access resources within or above their own namespace.
      - Connections can be made between resources in distinct namespaces, but this is generally done by integration code above both namespaces.
      - This encapsulation isolates different portions of the system from accidentally grabbing the wrong named resource or globally hijacking names.
      - 每个资源都在一个命名空间中定义，该命名空间可以与许多其他资源共享。
      - 通常，资源可以在其命名空间中创建资源，并且可以访问其自己的命名空间内或之上的资源。
      - 可以在不同命名空间中的资源之间建立连接，但这通常是通过两个命名空间上方的集成代码完成的。
      - 这种封装将系统的不同部分与意外获取错误的命名资源或全局劫持名称隔离开来。
    - `/`: global namespace
    - four types of Graph Resource Names:
      - base, relative, global, and private
      - `base`
      - `relative/name`
      - `/global/name`
      - `~private/name`
  - Package Resource Names
    - "std_msgs/String" refers to the "String" message type in the "std_msgs" Package.

## Higher-Level Concepts

- [wiki](https://wiki.ros.org/ROS/Higher-Level%20Concepts)

- Coordinate Frames/Transforms
  - The `tf` package provides a distributed, ROS-based framework for calculating the positions of multiple coordinate frames over time.
- Actions/Tasks
  - The `actionlib` package defines a common, topic-based interface for preemptible tasks in ROS.
- Message Ontology
  - `common_msgs` stack provides a set of common message types for interacting with robots.
    - `actionlib_msgs`: messages for representing actions
    - `diagnostic_msgs`: messages for sending diagnostic data.
    - `geometry_msgs`: messages for representing common geometric primitives.
    - `nav_msgs`: messages for navigation.
    - `sensor_msgs`: messages for representing sensor data.
- Plugins
  - `pluginlib` package provides tools for writing and dynamically loading plugins using the ROS build system.
- Filters
  - `filters` package provides a set of filters for processing data streams.
- Robot Model
  - The `urdf` package defines an XML format for representing a robot model and provides a C++ parser.

## Client Libraries

A ROS client library is a collection of code that eases the job of the ROS programmer. It takes many of the ROS concepts and makes them accessible via code. In general, these libraries let you write ROS nodes, publish and subscribe to topics, write and call services, and use the Parameter Server. Such a library can be implemented in any programming language, though the current focus is on providing robust C++ and Python support.

ROS 客户端库是简化 ROS 程序员工作的代码集合。它采用了许多 ROS 概念，并使其可以通过代码访问。通常，这些库允许您编写 ROS 节点、发布和订阅主题、编写和调用服务以及使用 Parameter Server。这样的库可以用任何编程语言实现，尽管目前的重点是提供强大的 C++ 和 Python 支持。

- [roscpp](https://wiki.ros.org/roscpp)
- [rospy](https://wiki.ros.org/rospy)
- [roslisp](https://wiki.ros.org/roslisp)
- ...

## [Technical Overview](https://wiki.ros.org/ROS/Technical%20Overview)

## Tools

### [`rosdep`](https://wiki.ros.org/rosdep)

rosdep is a command-line tool for installing system dependencies.

```bash
# install
sudo apt-get install python3-rosdep
# or
pip install rosdep
# source install
git clone https://github.com/ros-infrastructure/rosdep
cd rosdep
source setup.sh

# init rosdep, needs to call only once after installation
sudo rosdep init
# update
rosdep update
# install system dependencies

# install dependency of a package
rosdep install AMAZING_PACKAGE
# install dependency of all packages in the workspace
# cd into the catkin workspace, run:
rosdep install --from-paths src --ignore-src -r -y
```

### [`catkin`](https://wiki.ros.org/catkin)

Low-level build system macros and infrastructure for ROS.

- [wiki.ros: catkin](https://wiki.ros.org/catkin)
- [wiki.ros: catkin conceptual overview](https://wiki.ros.org/catkin/conceptual_overview)
- [catkin](https://wiki.ros.org/catkin/commands/catkin_make)
- [ros: rep-0128](https://ros.org/reps/rep-0128.html)

[catkin](https://wiki.ros.org/catkin) is the official build system of ROS and the successor to the original ROS build system, [rosbuild](https://wiki.ros.org/rosbuild). [catkin](https://wiki.ros.org/catkin) combines [CMake](http://www.cmake.org/) macros and Python scripts to provide some functionality on top of CMake's normal workflow. [catkin](https://wiki.ros.org/catkin) was designed to be more conventional than [rosbuild](https://wiki.ros.org/rosbuild), allowing for better distribution of packages, better cross-compiling support, and better portability. [catkin](https://wiki.ros.org/catkin)'s workflow is very similar to [CMake](http://www.cmake.org/)'s but adds support for automatic 'find package' infrastructure and building multiple, dependent projects at the same time.

[catkin](https://wiki.ros.org/catkin) 是 ROS 的官方构建系统，也是原始 ROS 构建系统 [rosbuild](https://wiki.ros.org/rosbuild) 的继承者。[catkin](https://wiki.ros.org/catkin) 结合了 [CMake](http://www.cmake.org/) 宏和 Python 脚本，在 CMake 的正常工作流程之上提供了一些功能。[Catkin](https://wiki.ros.org/catkin) 的设计比 [rosbuild](https://wiki.ros.org/rosbuild) 更传统，允许更好的包分发、更好的交叉编译支持和更好的可移植性。[catkin](https://wiki.ros.org/catkin) 的工作流程与 [CMake](http://www.cmake.org/) 的工作流程非常相似，但增加了对自动“查找包”基础设施的支持，并同时构建多个依赖的项目。

```bash
# debian 12 bookworm
catkin/stable 0.8.10-9 all
# python3
python3-catkin/stable,now 0.8.10-9 al
```

Usage:

```bash
cd path/to/your/catkin_workspace
# will build any packages in /catkin_workspace/src
catkin_make

# equivalent to
cd path/to/your/catkin_workspace
cd src
catkin_init_workspace
cd ..
mkdir build
cd build
cmake ../src -DCMAKE_INSTALL_PREFIX=../install -DCATKIN_DEVEL_PREFIX=../devel
make


# build specific package
catkin_make -DCATKIN_WHITELIST_PACKAGES="package1;package2"

# revert back to building all packages:
catkin_make -DCATKIN_WHITELIST_PACKAGES=""

# generate build and devel dir under workspace root

# install
catkin_make install

# specific source
catkin_make --source my_src
catkin_make install --source my_src
```

### [`rosinstall_generator`](https://wiki.ros.org/rosinstall_generator)

generattes `.rosinstall` files containing information about repositories with ROS packages/stacks.

```bash

# usage
rosinstall_generator PACKAGE DEPENDENCY1 DEPENDENCY2 > PACKAGE.rosinstall

# example
rosinstall_generator desktop --rosdistro noetic --deps --tar > noetic-desktop.rosinstall
```

### [`vcstool`](https://wiki.ros.org/vcstool)

Command-line tools for maintaining a workspace of projects from multiple version-control systems.

vcstool provides commands to manage several local SCM repositories (supports git, mercurial, subversion, bazaar) based on a single workspace definition file (`.repos` or `.rosinstall`).

```bash
vcs help

# example
vcs import --input noetic-desktop.rosinstall ./src
```
