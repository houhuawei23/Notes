# 基于梯度的运动规划

## CHOMP

CHOMP: Gradient Optimization Techniques for Efficient Motion Planning

摘要：

现有的高维运动规划算法在某种程度上既过度又不足。在障碍物稀疏分布的领域中，基于采样的规划器用于导航“狭窄通道”的启发式方法可能过于复杂；此外，还需要进一步后处理，以去除这些规划器生成路径中的颠簸或多余的动作。在本文中，我们介绍了 CHOMP，一种使用 `协变梯度技术`covariant gradient techniques 持续优化路径的新方法，用于提高采样轨迹的质量。我们的优化技术既可以优化高阶动态性，又可以相对于以前的路径优化策略在更广泛的输入路径上收敛。特别地，我们放宽了这些策略要求的输入路径上的无碰撞可行性先决条件。因此，CHOMP 可以用作许多实际规划查询中的独立运动规划器。我们演示了我们提议的方法在一个 6-DOF 机器臂的操纵规划以及在一个行走的四足机器人的轨迹生成中的有效性。

---

## Stomp

Stomp: Stochastic trajectory optimization for motion planning

摘要 — 我们提出了一种使用 `随机轨迹优化框架`进行运动规划的新方法。此方法依赖于生成噪声轨迹来探索初始（可能不可行）轨迹周围的空间，然后结合它们产生一个成本更低的更新轨迹。在每次迭代中，基于障碍和平滑性成本的组合优化一个成本函数。我们使用的特定优化算法不需要梯度信息，因此可以在成本函数中包括可能无法获得导数的一般成本（例如，对应于约束和电机扭矩的成本）。我们在模拟和移动操作系统上展示了这种方法，用于无约束和受约束的任务。我们实验性地显示，STOMP 的随机性使其能够克服 CHOMP 这样的基于梯度的方法可能陷入的局部最小值。

---

Continuous-time trajectory optimization for online uav
replanning

Realtime trajectory replanning for mavs using uniform b-splines and a 3d circular buffer

An efficient b-spline-based kinodynamic replanning framework for quadrotors

Robust and efficient quadrotor trajectory generation for fast autonomous flight

Raptor: Robust and perception aware trajectory replanning for quadrotor fast flight
