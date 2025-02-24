# Drones or Multi-UAV

多无人机协同任务规划

采用分层控制：决策层、协调层、执行层等

- 决策层 Decision Making Layer
  - 多 UAV 系统顶层的任务决策、避障、冲突消解、任务重分配和指标评估等
- 路径规划层 Path Planning Layer
  - 任务执行中的运动规划，生成航路点，以引导 UAV 避障
- 轨迹生成层 Trajectory Gen Layer
  - 根据 UAV 状态、输入和初始条件，为 UAV 生成通过航路点的可飞路径
- 内环控制层 Inner-Loop Control Layer
  - 保证 UAV 准确地沿着生成的轨迹飞行 并进行一定的荣誉管理以降低干扰等因素的影响
  - 飞控？

从多 UAV 协同路径规划的角度将任务规划的层次结构划分为：

- 机群协同任务规划与分配层
- 机群协同路径规划层
- 单机控制层
