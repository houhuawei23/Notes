# Learn Drones

## Labs

- [机器人研究团队](https://zhuanlan.zhihu.com/p/647982110)
- [multi_uav_simulator](https://github.com/malintha/multi_uav_simulator)

- [UPenn GRASP kumarrobotics](https://www.kumarrobotics.org/)
  - 宾夕法尼亚大学工程系 General Robotics, Automation, Sensing and Perception (GRASP) Laboratory
  - Vijay Kumar 维杰·库马尔
  - 致力于创建自主地面和空中机器人、设计集体行为的仿生算法以及机器人群
- [HKUST Aerial Robotics](https://uav.hkust.edu.hk/)
  - 香港科技大学 空中机器人研究组
  - [github](https://github.com/HKUST-Aerial-Robotics)
  - [沈劭劼](https://seng.hkust.edu.hk/about/people/faculty/shaojie-shen)
  - 状态估计、传感器融合、定位和地图绘制以及复杂环境中的自主导航
  - VINS-MONO, Fiesta, FAST-planner
- [SUSTech SmarT Autonomous Robotics Group](https://sysu-star.com/)
  - [github](https://github.com/SYSU-STAR)
  - 南方科技大学（SUSTech）
  - 周博宇 助理教授（副研究员）
- [ZJU FAST Lab](http://zju-fast.com/)
  - 浙江大学 无人系统与自主计算实验室
  - 高飞 许超
  - 无人机运动规划
  - [github](https://github.com/ZJU-FAST-Lab)
    - [Fast-Drone-250](https://github.com/ZJU-FAST-Lab/Fast-Drone-250)
    - [ego-planner-swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)
    - [ego-planner](https://github.com/ZJU-FAST-Lab/ego-planner)
- [IWIN](https://iwin-fins.com/)
  - Center for Intelligent Wireless Networks and Cooperative Control (IWIN Center)
  - 上海交通大学自动化系 智能无线网络与协同控制中心
  - [何建平](https://automation.sjtu.edu.cn/JPHE)
- [ETH-Z ASL](https://asl.ethz.ch/)
  - 苏黎世联邦理工大学自主系统实验室
  - Roland Siegwart
  - Rotors_simulator，Next best view planner
  - 运动规划、建图、微型无人机、全向飞行器、太阳能无人机、无人机编队飞行
- [ETH RPG](https://rpg.ifi.uzh.ch/index.html)
  - 苏黎世联邦理工大学自主系统实验室
  - 视觉无人机自主导航（无 GPS、激光雷达）、多机协同、运动规划、控制策略、敏捷飞行、侧重于环境感知等
- [UZH-RPG](https://rpg.ifi.uzh.ch/)
  - 苏黎世大学机器人与感知研究组
  - Scaramuzza 主持的实验室，
  - 实验室工作包括 SVO，事件相机，基于学习的一系列无人机自主飞行与竞速等工作，Flightmare 无人机仿平台。
- [MIT-ACL](https://acl.mit.edu/)
  - 麻省理工大学空间控制实验室
  - Jonathan How 教授主持的实验室，实验室在多机路径规划，多智能体强化学习，主动感知运动规划等领域有很好的工作
- 多伦多大学飞行系统与控制实验室（[UofT FSC](https://www.flight.utias.utoronto.ca/fsc/)）
  - Hugh liu 教授主持的实验室。
  - 实验室这几年工作集中在无人机绳系负载控制，多机协作绳系负载控制，多机运动规划等领域。有一批相当好的工作。对控制感兴趣的朋友可以关注此实验室。
- 卡耐基梅隆大学机器人学院（[Air Lab](https://theairlab.org/)）
  - 研究方向为多旋翼飞行器自主导航系统、移动平台无人机自主降落等。
- [rislab](https://rislab.org/)
  - Resilient Intelligent Systems Lab (RISLab)
  - Carnegie Mellon University’s Robotics Institute

## Papers

- RACER: rapid collaborative exploration with a decentralized multi-UAV system
  - [ieeexplore](https://ieeexplore.ieee.org/document/10038280)
  - 2023 IEEE Transactions on Robotics King-Sun Fu Memorial Best Paper Award
  - 周博宇
  - 该文章介绍了一个去中心化的多无人机（Unmanned Aerial Vehicles, UAVs）系统，旨在通过协作快速探索未知环境。该系统首先通过在线的未探索区域分解和基于成对交互的方式调度多无人机，即便在异步和不稳定的通信条件下也能合理分配探索任务。此外，通过一种考虑容量的车辆路径问题，优化了无人机群覆盖未知空间的路径，合理平衡各无人机的工作载荷。在该系统中，每个无人机根据分配的任务，通过层级规划器快速响应环境变化，从而安全高效地探索未知空间。无人机间通过融合视觉、惯性测量单元、超宽带等多种传感器信息实现相互定位。该系统在仿真和真实世界的严峻环境中进行了大量测试，可以适应室内、野外等多种复杂场景（如下图 1），显示出了高效率、可扩展性和在通信受限情况下的稳健性。该工作首次实现了在真实世界复杂场景中多无人机完全去中心化的协作探索，对于各种机器人应用，如巡检、搜救等，具有重要的实际意义。

## Journal 期刊

- [zhihu: 机器人领域主要期刊及国际会议汇总 2021](https://zhuanlan.zhihu.com/p/457055314)
- [机器人领域主要国际会议与期刊列表](https://blog.csdn.net/weixin_41598607/article/details/121084397)

<br>

- _Science Robotics_
  - 26.1 Q1 1Q
  - [science](https://www.science.org/journal/scirobotics)
- _IEEE Transactions on Robotics (TRO)_
  - IF 9.4, JCR Q1, CAS Q1
  - Computer Science Applications, Control and Systems Engineering, Electrical and Electronic Engineering
  - [ieee](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8860)
- _International Journal of Robotics Research (IJRR)_
  - 7.5 Q1 Q2
  - [sagepub](https://journals.sagepub.com/home/ijr)
- _Drones_
  - IF 4.4, JCR Q1 (Remote Sensing), CAS Q2
  - [mdpi](https://www.mdpi.com/journal/drones)
  - UAV, UAS, RPAS
- _Aerospace Science and Technology_
  - IF 5 JCR Q1, CAS Q1
  - [sciencedirect](https://www.sciencedirect.com/journal/aerospace-science-and-technology)
- _Soft Robotics_
  - Q1
- _Robotics and Computer-Integrated Manufacturing_
  - 9.1 Q1 1 区
  - 注重计算机科学和机械制造系统的结合
- _Vehicular Communications_ (Engineering: Automotive Engineering)
  - IF 5.8, JCR Q1, CAS Q2
  - [sciencedirect](https://www.sciencedirect.com/journal/vehicular-communications)

### _Robotics and Computer-Integrated Manufacturing_

The emphasis of the journal Robotics and Computer-Integrated Manufacturing is on disseminating the application of research to the development of new or improved industrially-relevant robotics, manufacturing technologies, and innovative manufacturing strategies. Preference is given to papers describing original research that includes both theory and experimental validation. Comprehensive review papers on topical issues related to robotics and manufacturing will also be considered. Papers on conventional machining processes, modelling and simulation, supply chain management, and resource optimisation, will generally be considered out of scope, as there are other more appropriate journals in these areas. Overly theoretical or mathematical papers will be directed to other more appropriate journals as well. Original papers are welcomed in the areas of industrial robotics, human-robot collaborative manufacturing, cloud-based manufacturing, cyber-physical production systems, big data analytics in manufacturing, smart mechatronics, machine learning, adaptive and sustainable manufacturing, and other fields involving unique manufacturing technologies.

《机器人与计算机集成制造》杂志的重点是传播研究在新型或改进的工业相关机器人、制造技术和创新制造策略的开发中的应用。优先考虑描述原创研究的论文，包括理论和实验验证。还将考虑与机器人和制造相关的热门问题的综合评论论文。关于传统加工工艺、建模和仿真、供应链管理和资源优化的论文通常被认为超出范围，因为这些领域还有其他更合适的期刊。过于理论或数学的论文也将被定向到其他更合适的期刊。工业机器人、人机协同制造、云制造、网络物理生产系统、制造大数据分析、智能机电一体化、机器学习、自适应和可持续制造以及其他涉及独特领域的原创论文受到欢迎制造技术。

### _Vehicular Communications_ (Engineering: Automotive Engineering)

types of communications involving vehicles, including **vehicle–to–vehicle** and **vehicle–to–infrastructure**. The scope includes (but not limited to) the following topics related to **vehicular communications**:

该期刊的目标是发表**车辆通信**领域的高质量同行评审论文。范围涵盖涉及车辆的所有类型的通信，包括**车辆对车辆**和**车辆对基础设施**。范围包括（但不限于）以下与**车辆通信**相关的主题：

- **Vehicle to vehicle** and **vehicle to infrastructure communications** **车辆到车辆**和**车辆到基础设施通信**
- Channel modelling, modulating and coding 信道建模、调制和编码
- **Congestion Control** and scalability issues **拥塞控制**和可扩展性问题
- Protocol design, testing and verification 协议设计、测试和验证
- Routing in **vehicular networks** **车载网络**中的路由
- Security issues and countermeasures 安全问题及对策
- Deployment and field testing 部署和现场测试
- Reducing energy consumption and enhancing safety of vehicles 降低能源消耗并提高车辆安全性
- Wireless **in–car networks** 无线**车载网络**
- Data collection and dissemination methods 数据收集和传播方法
- Mobility and handover issues 移动性和移交问题
- Safety and **driver assistance applications** 安全和**驾驶员辅助应用**
- UAV 无人机
- Underwater communications 水下通讯
- Autonomous cooperative driving 自动协同驾驶
- Social networks 社交网络
- Internet of vehicles 车联网
- Standardization of protocols 协议标准化

### _Drones_

- [scimagojr](https://www.scimagojr.com/journalsearch.php?q=21101017244&tip=sid&clean=0#google_vignette)

Drones (ISSN 2504-446X) is an international, peer-reviewed, open access journal published monthly online. The journal focuses on design and applications of drones, including unmanned aerial vehicle (UAV), Unmanned Aircraft Systems (UAS), and Remotely Piloted Aircraft Systems (RPAS), etc. Likewise, contributions based on unmanned water/underwater drones and unmanned ground vehicles are also welcomed.

《无人机》（ISSN 2504-446X）是一份国际同行评审、开放获取期刊，每月在线出版。该杂志重点关注无人机的设计和应用，包括无人机（UAV）、无人机系统（UAS）和遥控驾驶飞机系统（RPAS）等。同样，基于无人水上/水下无人机和无人地面车辆的贡献也受到欢迎。

### _IEEE Transactions on Robotics_

The IEEE Transactions on Robotics (T-RO) publishes fundamental papers on all aspects of Robotics, featuring interdisciplinary approaches from computer science, control systems, electrical engineering, mathematics, mechanical engineering, and other fields. Robots and intelligent machines and systems are critical in areas such as industrial applications; service and personal assistants; surgical operations; space, underwater, and remote exploration; entertainment; safety, search, and rescue; military applications; agriculture applications; and intelligent vehicles. Special emphasis in the T-RO is placed on intelligent machines and systems for unstructured environments, where a significant portion of the environment is unknown and cannot be directly sensed or controlled.

IEEE 机器人学汇刊 (T-RO) 发表机器人学各个方面的基础论文，涵盖计算机科学、控制系统、电气工程、数学、机械工程和其他领域的跨学科方法。机器人、智能机器和系统在工业应用等领域至关重要；服务和私人助理；外科手术；太空、水下和远程探索；娱乐;安全、搜索和救援；军事应用；农业应用；和智能汽车。 T-RO 特别强调非结构化环境中的智能机器和系统，其中环境的很大一部分是未知的，无法直接感知或控制。

## Conference 会议

- IEEE International Conference on Robotics and Automation (ICRA)
- IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
- Robotic: Science and Systems (RSS)
- Conference on Robot Learning (CoRL)

## CAA 中国自动化学会推荐学术会议、科技期刊目录

- [2022-12-06 中国自动化学会推荐学术会议、科技期刊目录发布](https://www.caa.org.cn/article/290/3225.html)

《会议目录》分为 A 类、B 类、C 类三档。A 类代表国际上极少数的顶级会议；B 类代表国际上著名和非常重要的会议；C 类代表国际学术界所认可的重要会议。《会议目录》包含自动化学科领域 12 个细分领域的 230 个学术会议，12 个细分领域为：
（1）控制理论与控制工程；
（2）模式识别与智能系统；
（3）检测技术与自动化装置；
（4）导航、制导与控制；
（5）系统工程；
（6）生物信息学；
（7）企业信息化；
（8）智能感知与自主控制；
（9）机器人与无人系统；
（10）仿真科学与工程；
（11）综合交叉；
（12）前沿高端。

《期刊目录》延续 2018 年版，分为 A 类、B 类两档，共 315 个刊物，A 类代表国内外顶级学术刊物，B 类代表国内外著名学术刊物。其中，A 类中前 20%的顶尖期刊划分为 A+类，以期国内学术期刊对标突破。

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
