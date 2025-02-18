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

- Response threshold model based UAV search planning and task allocation [J]. Journal of Intelligent & Robotic Systems, 2014

---

### 异构无人机集群与任务集合的数学描述

#### 无人机建模：

- $n$ 个无人机集合: $\boldsymbol{U}=\left\{u_{1}, u_{2}, \ldots, u_{n}\right\}$
- 利用四元组来描绘无人机状态信息 UIN:
  - $U I N\langle\text { Res, position, value, } \mathrm{v}\rangle$
  - $U I N=\left\{U I N_{u_{i}} \mid i=1,2, \ldots, n\right\}$
- $l$ 种资源类型, $R e_{u_{i}}^{k}$ 为无人机携带第 k 种资源的数目
  - 无人机 $u_i$ 的性能描述为: $\operatorname{Res}_{u_{i}}=\left\{R e_{u_{i}}^{1}, ~ R e_{u_{i}}^{2}, \ldots, R e_{u_{i}}^{l}\right\}$
- 无人机价值信息 $value _{u_{\mathrm{i}}}$
- 无人机速度约束:
  - $v_{u_{i}} \leqslant v_{u_{i}}^{\max }, \forall u_{i} \in U$
  - $v_{u_{i}}^{\max}$ 为常数，表示各无人机最大速度约束

则无人机 $u_i$ 的状态信息如下:

$$
\begin{aligned}
U I N_{u_{i}} & =\left\langle\operatorname{Res}_{u_{i}}, \text { position }_{\mathrm{u}_{\mathrm{i}}}, \text { value }_{\mathrm{u}_{\mathrm{i}}}, \mathrm{v}_{\mathrm{u}_{\mathrm{i}}}\right\rangle
\\& =\left\langle\left\{\operatorname{Re}_{u_{i}}^{1}, \ldots, \operatorname{Re}_{u_{i}}^{l}\right\},\left(x_{u_{i}}, y_{u_{i}}, z_{u_{i}}\right), \text { value }_{\mathrm{u}_{\mathrm{i}}}, \mathrm{v}_{\mathrm{u}_{\mathrm{i}}}\right\rangle
\end{aligned}
$$

#### 任务建模：

- $m$ 个任务的任务集合, $\boldsymbol{T}=\left\{t_{1}, t_{2}, \ldots, t_{m}\right\}$
- 任务状态信息 $TIN$ 描述如下
  - $\operatorname{TIN}\langle\text { Res, position, TW, a }\rangle$
  - $\operatorname{TIN}=\left\{\operatorname{TIN}_{t_{j}} \mid j=1,2, \ldots, m\right\}$
- $\operatorname{Res}_{t_{j}}=\left\{R e_{t_{j}}^{1}, R e_{t_{j}}^{2}, \ldots, R e_{t_{j}}^{l}\right\}$ 是任务所需的资源类型及数目，
- $R e_{t_{j}}^{k}$ 是任务所需要的第 k 种资源的数目。
- 任务 $t_{j}$ 的位置信息记为 $\operatorname{position}_{\mathrm{t}_{\mathrm{j}}}$ 。
- $TW$ 是任务的硬时间窗口信息，指对无人机到达任务时间范围提出硬性要求，若晚于某个时间节点，任务失败。
  - $\mathrm{TW}=\left[t^{\text {minstart }}, t^{\text {maxstart }}\right]$
- 威胁指数 $a_{t_{j}}$
  - 当无人机执行不同任务时，任务本身可能对无人机有一定的破坏能力，因此在建模中引入威胁指数 $a_{t_{j}}$ ，表示任务 $t_{j}$ 对无人机的威胁系数。

则任务 $t_j$ 的状态信息描述 $TIN_{t_j}$ 如下:

$$
\begin{array}{l}\operatorname{TIN}_{t_{j}}=\left\langle\text { Res }_{t_{j}}, \text { position }_{\mathrm{t}_{\mathrm{j}}}, \mathrm{TW}_{\mathrm{t}_{\mathrm{j}}}, \mathrm{a}_{\mathrm{t}_{\mathrm{j}}}\right\rangle \\=\left\langle\left\{\operatorname{Re}_{t_{j}}^{1}, \ldots, \text { Re }_{t_{j}}^{l}\right\},\left(x_{t_{j}}, y_{t_{j}}, z_{t_{j}}\right),\left[t_{t_{j}}^{\text {minstart }}, t_{t_{j}}^{\text {maxstart }}\right], a_{t_{j}}\right\rangle .\end{array}
$$

以下是异构无人机集群任务分配问题在不同数学模型中的基本数学形式：

---

### **1. 旅行商问题 (TSP) 模型**

**目标**：为每个无人机规划访问其分配任务的顺序，最小化总路径成本（时间或距离）。  
**变量**：

- $x_{ijk} \in \{0,1\}$: 无人机$u_i$是否从任务$t_j$移动到$t_k$。
- $s_{ij}$: 无人机$u_i$到达任务$t_j$的时间。

**目标函数**：

$$
\min \sum_{i=1}^n \sum_{j=0}^m \sum_{k=0}^m d_{jk} \cdot x_{ijk}
$$

（其中$d_{jk}$是任务$t_j$到$t_k$的距离，$j=0$或$k=0$表示无人机起点）

**约束**：

1. **任务覆盖**：每个任务被访问一次：
   $$
    \sum_{i=1}^n \sum_{k=0}^m x_{ikj} = 1, \quad \forall t_j \in T.
   $$
2. **路径连续性**：无人机路径形成闭合回路：
   $$
    \sum_{j=0}^m x_{ijk} = \sum_{j=0}^m x_{ikj}, \quad \forall u_i \in U, t_k \in T.
   $$
3. **资源约束**：
   $$
    \sum_{t_j \in T} \text{Re}_{t_j}^k \cdot \sum_{l=0}^m x_{ilj} \leq \text{Re}_{u_i}^k, \quad \forall u_i \in U, k=1,\ldots,l.
   $$
4. **时间窗口**：
   $$
    t_{t_j}^{\text{minstart}} \leq s_{ij} \leq t_{t_j}^{\text{maxstart}}, \quad \text{if } x_{ijk}=1.
   $$

---

### **2. 网络流优化 (NFO) 模型**

**目标**：将任务分配建模为多商品流问题，优化资源与时间约束下的流量分配。  
**变量**：

- $f_{i,jk} \in \{0,1\}$: 无人机$u_i$是否经过边$(t_j, t_k)$。

**目标函数**：

$$
\min \sum_{i=1}^n \sum_{j=0}^m \sum_{k=0}^m d_{jk} \cdot f_{i,jk}
$$

**约束**：

1. **流量守恒**：
   $$
    \sum_{k=0}^m f_{i,jk} = \sum_{k=0}^m f_{i,kj}, \quad \forall u_i \in U, t_j \in T.
   $$
2. **资源容量**：
   $$
    \sum_{u_i \in U} \text{Re}_{u_i}^k \cdot \sum_{j=0}^m f_{i,jt} \geq \text{Re}_{t}^k, \quad \forall t \in T, k=1,\ldots,l.
   $$
3. **时间窗口**：通过时间扩展网络建模。

---

### **3. 车辆路由问题 (VRP) 模型**

**目标**：为异构无人机规划路径，满足资源、时间窗口约束，最小化总成本。  
**变量**：

- $x_{ij} \in \{0,1\}$: 无人机$u_i$是否分配至任务$t_j$。
- $y_{ijk} \in \{0,1\}$: 无人机$u_i$从$t_j$移动到$t_k$。

**目标函数**：

$$
\min \sum_{i=1}^n \sum_{j=0}^m \sum_{k=0}^m d_{jk} \cdot y_{ijk}
$$

**约束**：

1. **任务分配唯一性**：
   $$
    \sum_{i=1}^n x_{ij} = 1, \quad \forall t_j \in T.
   $$
2. **资源约束**：
   $$
    \sum_{t_j \in T} \text{Re}_{t_j}^k \cdot x_{ij} \leq \text{Re}_{u_i}^k, \quad \forall u_i \in U, k=1,\ldots,l.
   $$
3. **时间协调**：
   $$
    s_{ik} \geq s_{ij} + \frac{d_{jk}}{v_{u_i}}, \quad \text{if } y_{ijk}=1.
   $$

---

### **4. 协同多任务分配问题 (CMTAP) 模型**

**目标**：多无人机协同完成任务，优化任务完成率与威胁代价。  
**变量**：

- $z_{ij} \in \{0,1\}$: 无人机$u_i$是否参与任务$t_j$。

**目标函数**：

$$
\max \sum_{t_j \in T} \left( \prod_{k=1}^l \mathbf{1}_{\left\{\sum_{i=1}^n \text{Re}_{u_i}^k \cdot z_{ij} \geq \text{Re}_{t_j}^k\right\}} \right) - \sum_{i=1}^n \sum_{j=1}^m a_{t_j} \cdot z_{ij}
$$

**约束**：

1. **资源协同**：
   $$
    \sum_{i=1}^n \text{Re}_{u_i}^k \cdot z_{ij} \geq \text{Re}_{t_j}^k, \quad \forall t_j \in T, k=1,\ldots,l.
   $$
2. **时间协同**：
   $$
    \max_{u_i \in U} \left( \frac{\| \text{position}_{u_i} - \text{position}_{t_j} \|}{v_{u_i}} \right) \leq t_{t_j}^{\text{maxstart}}, \quad \text{if } z_{ij}=1.
   $$

---

### **5. 混合整数线性规划 (MILP) 模型**

**目标**：通过线性约束建模任务分配、路径规划与资源协同。  
**变量**：

- $x_{ij} \in \{0,1\}$: 无人机$u_i$分配至任务$t_j$。
- $s_{ij} \geq 0$: 无人机$u_i$开始执行任务$t_j$的时间。

**目标函数**：

$$
\min \sum_{i=1}^n \sum_{j=1}^m a_{t_j} \cdot x_{ij} + \sum_{i=1}^n \sum_{j=1}^m \frac{\| \text{position}_{u_i} - \text{position}_{t_j} \|}{v_{u_i}} \cdot x_{ij}
$$

**约束**：

1. **资源约束**：
   $$
    \sum_{j=1}^m \text{Re}_{t_j}^k \cdot x_{ij} \leq \text{Re}_{u_i}^k, \quad \forall u_i \in U, k=1,\ldots,l.
   $$
2. **时间窗口**：
   $$
    t_{t_j}^{\text{minstart}} \leq s_{ij} \leq t_{t_j}^{\text{maxstart}}, \quad \text{if } x_{ij}=1.
   $$

---

### **6. 基于马尔可夫决策过程 (MDP) 模型**

**五元组**：$\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$

- **状态空间$\mathcal{S}$**：无人机位置、资源、时间；任务时间窗口、完成状态。
- **动作空间$\mathcal{A}$**：为无人机分配任务或路径点。
- **转移函数$P(s' \mid s, a)$**：移动时间、资源消耗与任务完成状态更新。
- **奖励函数$R(s, a)$**：
  $$
   R = \sum_{t_j \in T} r_j^{\text{complete}} - \sum_{u_i \in U} \sum_{t_j \in T} a_{t_j} \cdot \mathbf{1}_{\{u_i \text{ 参与 } t_j\}}.
  $$
- **策略$\pi: \mathcal{S} \to \mathcal{A}$**：最大化期望累积奖励$\mathbb{E}[\sum_{t=0}^\infty \gamma^t R_t]$。

---

以上模型均基于无人机和任务的状态信息$\text{UIN}$和$\text{TIN}$，结合问题特性与约束条件构建。

## Solving Algorithm

- 集中式 vs 分布式
- 预分配 vs 重分配
- 同构 vs 异构
- 中小规模 vs 大规模
- 考虑的约束条件: 资源 通信 任务时序...
- 静态 vs 动态: 能否在线处理
  - 静态: agent 在执行前就知晓所有待分配任务的信息
  - 动态: 在执行过程中出现新的任务 or 参与任务的机器人遭到破坏而必须将其任务序列中待完成的任务进行重新分配

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

## 数学问题的模型与常见求解算法

1. **TSP模型**：基本形式是寻找最短回路，访问所有城市一次。变量通常是二元变量表示边是否被选中。约束包括每个节点进出一次，子回路消除等。解法可能包括精确算法（分支定界）、启发式（最近邻）和元启发式（遗传算法）。

2. **网络流优化（NFO）**：核心是流量守恒、容量限制，目标是最小化或最大化流量相关成本。常见模型如最小费用流，求解方法包括单纯形法、消圈算法等。

3. **VRP模型**：类似于TSP但多车辆，有容量约束。变量包括分配和路径变量。解法包括启发式（节约算法）、元启发式（模拟退火）和精确方法。

4. **CMTAP模型**：需要多智能体协同，资源匹配，可能涉及组合优化。目标函数可能最大化任务完成数，约束包括资源和时间。解法有拍卖算法、分布式优化等。

5. **MILP模型**：包含连续和整数变量，线性目标函数和约束。求解方法包括分支定界、割平面，以及启发式。

6. **MDP模型**：基于状态、动作、转移概率和奖励，策略优化。解法包括值迭代、策略迭代、Q学习等强化学习方法。

以下是六种数学模型的一般数学形式及其常见求解方法的系统总结：

---

### 1. **旅行商问题（TSP）模型**

**一般数学形式**：

- **目标**：在完全图中找到访问所有节点且总距离最短的哈密顿回路。
  - 寻找一条最短的路径，使得旅行商能够访问每个城市恰好一次，并最终返回起点。
- **变量**：$x_{ij} \in \{0,1\}$，表示边 $(i,j)$ 是否被选中。
- **目标函数**：
  $$
  \min \sum_{i=1}^n \sum_{j \neq i} d_{ij} x_{ij}
  $$
- **约束**：
  $$
  \begin{cases}
  \sum_{i=1}^n x_{ij} = 1, & \forall j \ (\text{每个节点入度=1}) \\
  \sum_{j=1}^n x_{ij} = 1, & \forall i \ (\text{每个节点出度=1}) \\
  \sum_{i \in S, j \notin S} x_{ij} \geq 1, & \forall S \subset \{1,\ldots,n\}, S \neq \emptyset \ (\text{消除子回路})
  \end{cases}
  $$

**常见求解方法**：

- **精确算法**：分支定界、动态规划（Held-Karp 算法）。
- **启发式算法**：最近邻法、插入法。
- **元启发式算法**：遗传算法、蚁群优化、模拟退火。
- **近似算法**：Christofides 算法（保证 1.5 倍最优解）。

---

### 2. **网络流优化（NFO）模型**

**一般数学形式**：

- **目标**：在带容量的网络中优化流量分配，如最小成本流、最大流。
  - 在图中找到从源到汇的最大流，同时满足容量约束。
- **变量**：$f_{ij} \geq 0$，表示边 $(i,j)$ 上的流量。
- **目标函数**（以最小费用流为例）：
  $$
  \min \sum_{(i,j)} c_{ij} f_{ij}
  $$
- **约束**：
  $$
  \begin{cases}
  \sum_j f_{ij} - \sum_j f_{ji} = b_i, & \forall i \ (\text{流量守恒}) \\
  l_{ij} \leq f_{ij} \leq u_{ij}, & \forall (i,j) \ (\text{容量约束})
  \end{cases}
  $$
  其中 $b_i$ 为节点的供给/需求（$\sum b_i = 0$）。

**常见求解方法**：

- **单纯形法**：针对线性网络流问题。
- **消圈算法（Cycle Canceling）**：消除负费用环。
- **最短增广路算法**（如 Successive Shortest Path）。
- **预流推进算法**：用于最大流问题。

---

### 3. **车辆路由问题（VRP）模型**

**一般数学形式**：

- **目标**：多车辆服务所有客户，最小化总成本（距离、时间等）。
- **变量**：
  - $x_{ijk} \in \{0,1\}$，车辆 $k$ 是否从 $i$ 到 $j$。
  - $q_k \leq Q_k$，车辆 $k$ 的载重约束。
- **目标函数**：
  $$
  \min \sum_{k=1}^K \sum_{i,j} d_{ij} x_{ijk}
  $$
- **约束**：
  $$
  \begin{cases}
  \sum_{k} \sum_j x_{ijk} = 1, & \forall i \ (\text{客户被服务}) \\
  \sum_{i} d_i \sum_j x_{ijk} \leq Q_k, & \forall k \ (\text{容量约束}) \\
  \text{路径连续性约束} & (\text{类似TSP})
  \end{cases}
  $$

**常见求解方法**：

- **精确算法**：分支定价（Branch-and-Price）。
- **启发式算法**：Clarke-Wright 节约算法、扫描算法。
- **元启发式算法**：禁忌搜索、大规模邻域搜索（LNS）。
- **强化学习**：基于神经网络的端到端求解。

---

### 4. **协同多任务分配问题（CMTAP）模型**

**一般数学形式**：

- **目标**：多智能体协同完成任务，最大化任务完成率或最小化代价。
- **变量**：$z_{ik} \in \{0,1\}$，智能体 $i$ 是否执行任务 $k$。
- **目标函数**（以最小化威胁代价为例）：
  $$
  \min \sum_{i,k} a_k z_{ik} + \sum_{k} \lambda \cdot \mathbf{1}_{\{\sum_i z_{ik} < R_k\}}
  $$
- **约束**：
  $$
  \begin{cases}
  \sum_i z_{ik} \geq R_k, & \forall k \ (\text{资源需求}) \\
  \sum_k z_{ik} \leq C_i, & \forall i \ (\text{智能体能力限制})
  \end{cases}
  $$
  其中 $R_k$ 为任务 $k$ 所需智能体数，$a_k$ 为威胁系数。

**常见求解方法**：

- **集中式优化**：整数规划求解器（如 CPLEX、Gurobi）。
- **分布式算法**：合同网协议（Contract Net）、拍卖算法。
- **群智能算法**：粒子群优化（PSO）、蜂群算法。
- **博弈论方法**：Nash 均衡求解。

---

### 5. **混合整数线性规划（MILP）模型**

**一般数学形式**：

- **目标**：线性目标函数，含连续和整数变量。
- **标准形式**：
  $$
  \begin{aligned}
  \min \quad & c^T x + d^T y \\
  \text{s.t.} \quad & A x + B y \leq b \\
  & x \in \mathbb{Z}^p, \ y \in \mathbb{R}^q
  \end{aligned}
  $$
  其中 $x$ 为整数变量，$y$ 为连续变量。

**常见求解方法**：

- **精确算法**：
  - 分支定界（Branch-and-Bound）。
  - 分支切割（Branch-and-Cut）。
- **启发式算法**：
  - 松弛固定法（Fix-and-Relax）。
  - 大邻域搜索（Large Neighborhood Search）。
- **分解算法**：Benders 分解、列生成（Column Generation）。

---

### 6. **基于马尔可夫决策过程（MDP）模型**

**一般数学形式**：

- **五元组**：$\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$
  - **状态空间** $\mathcal{S}$：系统所有可能状态。
  - **动作空间** $\mathcal{A}$：智能体可选动作。
  - **转移函数** $P(s'|s,a)$：状态转移概率。
  - **奖励函数** $R(s,a)$：即时奖励。
  - **折扣因子** $\gamma \in [0,1]$。
- **目标**：找到策略 $\pi: \mathcal{S} \to \mathcal{A}$，最大化期望累积奖励：
  $$
  \max_\pi \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]
  $$

**常见求解方法**：

- **动态规划**：
  - 值迭代（Value Iteration）。
  - 策略迭代（Policy Iteration）。
- **蒙特卡洛方法**：基于采样的策略评估。
- **时序差分学习**：Q 学习、SARSA。
- **深度强化学习**：DQN、PPO、A3C。
- **逆向强化学习**：从专家演示中学习奖励函数。

---

### 总结对比表

| 模型      | 核心变量类型        | 典型约束类型         | 求解方法特点                   |
| --------- | ------------------- | -------------------- | ------------------------------ |
| **TSP**   | 二元边选择变量      | 路径闭合、子回路消除 | 精确算法+启发式组合            |
| **NFO**   | 连续/整数流量变量   | 流量守恒、容量限制   | 基于图论的优化算法             |
| **VRP**   | 路径与分配变量      | 容量、时间窗口       | 分层优化（先分配后路由）       |
| **CMTAP** | 任务-智能体分配变量 | 资源协同、时间协同   | 分布式或群智能优化             |
| **MILP**  | 混合整数变量        | 线性不等式           | 分支定界框架+割平面增强        |
| **MDP**   | 状态-动作策略映射   | 贝尔曼方程           | 基于值函数或策略梯度的迭代优化 |

每种模型的求解方法选择需结合问题规模、实时性要求和资源约束。实际应用中常采用**混合策略**（如 MILP+启发式初始化）以平衡效率与精度。
