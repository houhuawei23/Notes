# UAV Swarm Task Assignment Problem

- [无人机集群任务分配技术研究综述 系统工程与电子技术, 2024, 46(3): 922-934 ](https://www.sys-ele.com/article/2024/1001-506X/20240318.shtml)

## Modeling

无人机集群任务分配的数学描述模型可概括性地表述为:

异构无人机平台(Who)在特定位置(Where)上, 为完成某种集群任务而按照一定的约束条件/作战规则(Why)执行自身分配到的任务(How)并产生与消耗作战资源(What), 即无人机集群任务分配“4W1H”关系。

组合优化问题模型

- 旅行商问题 (traveling salesman problem, TSP) 模型
- 网络流优化 (network flow optimization, NFO) 模型
- 车辆路由问题 (vehicle routing problem, VRP) 模型
- 协同多任务分配问题 (cooperative multi-tasks assignment problem, CMTAP) 模型
- 混合整数线性规划 (mixed integer linear programming, MILP) 模型
- 基于马尔可夫决策过程 (Markov decision process, MDP) 模型

优劣

- TSP 和 VRP 模型主要用于求解单一任务的分配问题, 而对多任务情况适用性较差;
- NFO 模型较早运用于弹药较少的广域搜索弹药任务分配问题上;
- MILP 模型描述简洁, 很容易表示涉及到数值的全局约束, 将任务规划问题描述为一个组合优化问题, 实用性较强, 但计算成本会随问题规模增大而呈指数型增长;
- 而基于 NFO 和 MILP 模型提出的 CMTAP 模型考虑多无人机编队完成探测识别、打击、毁伤评估等一系列时序任务, 通过优化完成任务的总时间或者飞行的总距离来实现任务分配, 更适用于复杂任务分配问题建模, 但可扩展性低;
- 在考虑系统存在不确定因素和多智能体协同系统时, 可分别通过部分可观测的 MDP(partially observable MDP, POMDP) 及多智能体的 MDP(multi-agent MDP, MMDP)对协同任务分配问题进行建模, 但上述模型均存在通用性较差的缺点。

## Solving Algorithm

- 集中式 vs 分布式
- 预分配 vs 重分配
- 同构 vs 异构
- 中小规模 vs 大规模
- 考虑的约束条件: 资源 通信 任务时序...
- 静态 vs 动态: 能否在线处理

### 集中式

- 最优化方法
  - 穷举法、分支定界、整数规划、动态规划等
  - NP 难问题, 计算量随规模指数增加; 难以求解大规模问题
  - 不能处理真实环境中的随机性和动态性
- 启发式方法
  - 可行解 次优解
  - 遗传算法、禁忌搜索算法、粒子群优化算法、模拟退火算法等
  - 灰狼算法 仿生狼群无人机群任务分配方法

#### 启发式方法

##### 遗传算法

##### 粒子群优化算法

### 分布式

- 自顶向下
  - 利用分层递阶求解的思路, 将复杂任务协同分配问题逐层分解为若干个更简单的子任务分配问题, 各无人机通过协商与合作实现问题求解
  - 分布式 MDP(decentralized MDP, Dec-MDP)方法
  - 基于市场机制的方法
  - 动态分布式约束优化问题 (dynamic distributed constraint optimization problem, DDCOP) 方法
- 自下而上
  - 通过研究无人机个体的局部感知和动态反应, 设计基于反应和行为的协同优化策略, 实现多无人机整体自组织任务分配
  - 自组织算法(self-organized algorithm, SOA)
  - 阈值响应法
  - 蚁群优化(ant colony optimization, ACO)算法

### 重分配

随着战场对抗不断升级, 无人机集群在任务执行过程中遇到突发事件时, 任务重分配算法需要通过各无人机平台之间的信息交互对战场态势和动态任务快速做出反应, 在有限时间内完成任务的重构与优化, 并达成无冲突的任务重分配方案。

- 基于市场机制的算法
- 基于博弈的算法
- 其他

### 异构系统

## Misc
