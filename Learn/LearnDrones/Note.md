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
  - mavros 用于无人机通信，可以将飞控与主控的信息进行交换
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

## Concepts

### SDF Signed Distance Field 符号距离场

- 描述或存储点与场景物体表面位置关系的方式
- 当点在物体的内部时距离为负数，在物体表面时距离为 0，在物体外部时距离为正数
- 作用: 用于快速查询任意一点到场景中物体表面的最短距离
- 如何建立和使用？

- [Intro to Signed Distance Fields](https://www.rustadventure.dev/introduction-to-signed-distance-fields)

### Gradient-based trajectory planning 基于梯度的轨迹规划

- [blog: Gradient-based trajectory planning](https://kyunghyuncho.me/gradient-based-trajectory-planning/)
- [CHOMP: Gradient Optimization Techniques for Efficient Motion Planning](https://ieeexplore.ieee.org/document/5152817)

GPT4o: Gradient-Based Trajectory Planning: Overview 基于梯度的轨迹规划：概述

**Gradient-based trajectory planning** is a method used to determine a path or trajectory for an object (like a robot or vehicle) by iteratively improving an initial trajectory based on gradient information. The goal is to optimize a trajectory with respect to certain objectives, such as minimizing energy, avoiding obstacles, or reaching a target point efficiently.

基于梯度的轨迹规划是一种通过基于梯度信息迭代改进初始轨迹来确定物体（如机器人或车辆）的路径或轨迹的方法。目标是针对某些目标优化轨迹，例如最小化能量、避开障碍物或有效地到达目标点。

---

#### Key Thoughts of the Method 该方法的核心思想

1. Optimization via Gradients: 通过梯度进行优化：
   - The trajectory is treated as a variable in an optimization problem.
     - 轨迹被视为优化问题中的变量。
   - Gradients of a cost function (e.g., distance to obstacles, time taken, energy expenditure) with respect to the trajectory are computed and used to iteratively improve the trajectory.
     - 计算成本函数相对于轨迹的梯度（例如，到障碍物的距离、所花费的时间、能量消耗）并用于迭代地改进轨迹。
2. Cost Function Design: 代价函数设计：
   - A carefully designed cost function combines all relevant objectives, such as collision avoidance, smoothness, and time efficiency.
     - 精心设计的成本函数结合了所有相关目标，例如避免碰撞、平滑度和时间效率。
   - Commonly, the cost function includes terms for proximity to obstacles, adherence to desired dynamics, and overall trajectory smoothness.
     - 通常，成本函数包括与障碍物的接近程度、对所需动态的遵守以及总体轨迹平滑度等项。
3. Iterative Adjustment: 迭代调整：
   - Starting with an initial trajectory, gradient descent or a similar optimization algorithm adjusts the trajectory step-by-step.
     - 从初始轨迹开始，梯度下降或类似的优化算法逐步调整轨迹。
   - Constraints (e.g., dynamics, kinematics) are often included to ensure feasible and safe paths.
     - 通常包括约束（例如动力学、运动学）以确保可行且安全的路径。
4. Scalability and Flexibility: 可扩展性和灵活性：
   - The method can handle high-dimensional trajectory spaces, making it suitable for robotic arms, drones, and autonomous vehicles.
     - 该方法可以处理高维轨迹空间，使其适用于机械臂、无人机和自动驾驶车辆。

---

#### Advantages 优点

1. **Adaptability**: Gradient-based methods can adapt trajectories to dynamic environments by continuously updating the solution as conditions change.
   - **适应性**：基于梯度的方法可以通过随着条件变化不断更新解决方案来使轨迹适应动态环境。
2. **Optimality**: These methods aim to find a locally (and sometimes globally) optimal solution based on the cost function.
   - **最优性**：这些方法旨在基于成本函数找到局部（有时是全局）最优解。
3. **Mathematical Rigorousness**: Strong theoretical foundation allows for analysis and guarantees under certain conditions.
   - **数学严谨性**：强大的理论基础可以在一定条件下进行分析和保证。
4. **Wide Applicability**: Useful for a range of domains like robotics, autonomous driving, and animation.
   - **广泛的适用性**：可用于机器人、自动驾驶和动画等一系列领域。
5. **Smoothness**: Incorporating smoothness constraints often results in trajectories that are smooth and practical for real-world use.
   - **平滑度**：结合平滑度约束通常会产生平滑且适合实际使用的轨迹。

---

#### Disadvantages 缺点

1. **Local Minima**: The method may converge to a suboptimal solution due to the presence of local minima in the cost function.
   - **局部最小值**：由于成本函数中存在局部最小值，该方法可能会收敛到次优解。
2. **Computational Cost**: Computing gradients and solving high-dimensional optimization problems can be computationally expensive.
   - **计算成本**：计算梯度和解决高维优化问题的计算成本可能很高。
3. **Dependence on Initial Guess**: Performance and results heavily depend on the quality of the initial trajectory.
   - **对初始猜测的依赖**：性能和结果在很大程度上取决于初始轨迹的质量。
4. **Complex Cost Function Design**: Designing an appropriate and effective cost function can be challenging and problem-specific.
   - **复杂的成本函数设计**：设计适当且有效的成本函数可能具有挑战性并且针对具体问题。
5. **Dynamic Environments**: Handling highly dynamic and uncertain environments can be difficult without extensive computational resources.
   - **动态环境**：如果没有大量的计算资源，处理高度动态和不确定的环境可能会很困难。

---

#### Origin and Key Papers 起源和关键论文

- The principles of gradient-based optimization trace back to classical optimization theory.
- In trajectory planning, this concept has been extensively developed in robotics and control literature.
- A key early reference is **"CHOMP: Covariant Hamiltonian Optimization for Motion Planning"** by Marc Toussaint and Sachin Chitta, which formalized gradient-based optimization in the context of trajectory planning.
- Other significant contributions include works on **STOMP (Stochastic Trajectory Optimization for Motion Planning)** and **TrajOpt (Trajectory Optimization)**.

<br>

- 基于梯度的优化原理可以追溯到经典优化理论。
- 在轨迹规划中，这个概念已在机器人技术和控制文献中得到广泛发展。
- 一个重要的早期参考文献是 Marc Toussaint 和 Sachin Chitta 撰写的“CHOMP：运动规划的协变哈密顿优化” ，它在轨迹规划的背景下形式化了基于梯度的优化。
- 其他重要贡献包括**STOMP（运动规划随机轨迹优化）**和**TrajOpt（轨迹优化）**方面的工作。

---

#### Applications 应用领域

1. **Autonomous Vehicles**: Planning collision-free and efficient routes in complex environments.
2. **Robotics**: Motion planning for robotic arms, humanoid robots, and drones.
3. **Computer Graphics**: Smooth trajectory generation for animations.
4. **Space Exploration**: Path optimization for spacecraft and planetary rovers.
5. **Medical Robotics**: Trajectory planning for surgical robots to ensure precision.

<br>

1. **医疗机器人**：手术机器人的轨迹规划以确保精度。
2. **自动驾驶车辆**：在复杂环境中规划无碰撞且高效的路线。
3. **机器人技术**：机械臂、人形机器人和无人机的运动规划。
4. **计算机图形学**：动画的平滑轨迹生成。
5. **太空探索**：航天器和行星漫游器的路径优化。

---

#### Related Techniques 相关技术

Gradient-based trajectory planning is part of a broader class of **optimization-based motion planning techniques**, including:

基于梯度的轨迹规划是更广泛的基于优化的运动规划技术的一部分，包括：

- **Sampling-based Planning (e.g., RRT, PRM)**: Focus on randomly sampling the configuration space.
  - **基于采样的规划（例如，RRT、PRM）** ：重点是对配置空间进行随机采样。
- **Optimization-based Planning (e.g., CHOMP, STOMP)**: Emphasize iterative refinement using optimization.
  - **基于优化的规划（例如 CHOMP、STOMP）** ：强调使用优化进行迭代细化。
- **Machine Learning Approaches**: Use learned models to assist or replace traditional optimization techniques.
  - **机器学习方法**：使用学习模型来辅助或取代传统的优化技术。
