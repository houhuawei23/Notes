# 基于联盟形成博弈的异构无人机集群分布式任务分配算法 中国科学:信息科学 南京航空航天大学自动化学院 2024

Distributed task allocation algorithm for heterogeneous unmanned aerial vehicle swarm based on coalition formation game.

- 薛舒心 1,3, 马亚杰 1,3\*, 姜斌 1,3, 李文博 1,2, 刘成瑞 1,2

1. 南京航空航天大学自动化学院, 南京 210016
2. 北京控制工程研究所空间智能控制技术全国重点实验室, 北京 100094
3. 飞行器自主控制技术教育部工程研究中心, 南京 210016
4. \* 通信作者. E-mail: yajiema@nuaa.edu.cn

- 关键词： 任务分配; 异构无人机集群; 异构资源; 聚类算法; 联盟形成博弈;
- Keywords: task allocation, heterogeneous UAV swarms, heterogeneous resources, clustering algorithm, coalition formation game
- 摘要：
  - 针对无人机集群携带异构资源以及任务的异构需求下的复杂任务分配优化求解问题,提出一种基于联盟形成博弈的分布式任务分配算法.
  - 首先针对任务分配问题规模较大的特点以及资源的异构性,提出一种基于异构资源的改进 K-medoids 聚类算法,通过对无人机集群和任务进行聚类的预处理,降低了任务分配的规模和难度.
  - 考虑任务需求、机载资源以及路径成本等条件建立任务分配模型,将原有任务分配问题转化为联盟划分问题,设计了一种基于联盟形成博弈的分布式任务分配算法进行求解.
  - 最后,将 30 个具有异构需求的任务分配给 100 架携带 3 种异构资源的无人机的仿真结果表明,所提算法能够实现较好的任务分配效果,同时极大提高任务分配的实时性,充分发挥集群效能.
- 基金资助： 国家自然科学基金(批准号:62273177,62020106003,62233009)； 江苏省自然科学基金(批准号:BK20211566,BK20222012)； 高校学科创新引智基地(批准号:B20007)； 空间智能控制技术全国重点实验室开放基金(批准号:HTKJ2023KL502006)； 中央高校基本科研业务费(批准号:NI2024001)资助项目；
- [doi-link](https://doi.org/10.1360/ssi-2024-0167)

1. 引言
2. 问题描述
   1. 数学建模
      1. 无人机建模
      2. 任务建模
      3. 联盟形成博弈数学模型
   2. 目标函数与约束条件
3. 联盟形成博弈算法设计
   1. 基于异构资源的改进 K-medoids 聚类算法
      1. 无人机集群聚类
      2. 任务聚类
   2. 基于联盟形成博弈的任务分配算法
   3. 算法复杂度分析
4. 仿真验证
   1. 基于异构资源的改进 K-medoids 聚类算法结果分析
   2. 基于联盟形成博弈的任务分配算法结果分析
5. 结论

## others

- [从匈牙利算法到 KM 算法](https://zhuanlan.zhihu.com/p/214072424)

## 引言

常见的分布式任务分配算法主要有博弈论方法、基于市场机制的方法 [7] 和分布式马尔可夫决策过程方法.

- **基于市场机制的方法**：
  - 合同网算法和拍卖算法是应用较为广泛的基于市场机制的方法.
  - 文献 [8] 针对异构无人机和未知目标背景提出了一种**基于合作网协议的任务分配算法**, 能够获得较大的系统效能, 但算法缺少对某无人机可能同属多个任务最优联盟解的冲突消除考虑.
  - 文献 [9] 针对不同作战任务, 建立了一种**面向异构且具有时序约束任务的多轮次分布式拍卖算法**, 本文模型中无人机可执行不同类型的任务,但缺少对执行相同类型任务的无人机能力的区分. 基于分布式马尔可夫法通常训练难度大, 收敛较慢.
- **分布式马尔可夫决策过程方法**：
  - 文献 [10] 针对边缘服务器的分布式任务和资源分配问题, 将其建模为**部分观测的马尔可夫决策过程**,提出一种依赖于多智能体的深度强化两步策略, 所提出的解决方案在训练和收敛方面优于基准.
- **博弈论方法**：
  - 目前, 得益于博弈论能够为无人机之间的合作协商提供良好性能, 博弈方法已成为热门研究方向,应用于各种优化问题 [11].
  - 文献 [5] 设计了智能体之间的组合策略, 将全局效用映射为势函数获得最优分配方案.
  - 文献 [12] 采用了**基于势博弈的分布式多智能体动态任务分配方法**, 该算法在全局搜索能力方面表现良好.
  - 文献 [13] 设计了**基于偏好联盟形成博弈的任务分配算法**, 利用偏好程度定义机器人加入联盟后的效用, 算法保证了多项式时间复杂性和解的纳什稳定性.
    - 其中, 联盟形成博弈 (coalition formation game, CFG) 模型是常用的博弈模型之一, 目的在于利用无人机之间的相互合作最大化系统性能, 为参与者之间的决策提供了一个高效的模型与工具. 在整个博弈过程中, 参与者会自行组成若干个联盟, 从全局来看形成若干分组.
  - 文献 [14] 设计了一种**基于联盟博弈的面向组的分布式聚类算法**, 算法经验证在通信链路稳定性、聚类数量和负载平衡等方面优于现有算法.
  - 文献 [15] 针对无人机前置群干扰提出一种**基于分布式联盟形成博弈的动态任务分配算法**,将任务分配问题转化为多智能体协同问题, 算法可以实现与集中式优化相似的性能.
  - 文献 [16] 考虑资源属性和任务执行顺序的重叠和互补关系, 在基于顺序重叠联盟形成博弈模型基础上开发了一种**偏好重力引导禁忌搜索算法**, 可获得稳定的联盟结构.

综上所述, 分布式任务分配算法已取得较多研究成果, 尤其在博弈方面. 然而, 目前工作仍存在以下问题:

- 一是对于机载资源和任务需求资源的**异构性研究较为薄弱**;
- 二是大规模无人机集群在进行任务分配时求解难度剧增, 导致**算法实时性较差**;
- 三是单个无人机运载能力很难满足任务需求, 需要考虑**无人机之间的协作**.

基于联盟形成博弈解决任务分配问题能够：

- 根据无人机在性能、能力和资源方面的差异灵活进行任务分配, 提高系统整体效率;
- 其次具有较高的可拓展性, 能够适应不同规模的无人机集群. 同时联盟形成博弈支持分布式决策, 无人机能够自主进行任务分配, 提升系统的鲁棒性和可靠性;
- 通过合理的收益分配机制, 激励无人机积极参与任务执行, 增强协作效果.

因此, 本文提出一种基于联盟形成博弈的分布式任务分配算法, 主要贡献总结如下:

1. 针对原任务分配问题规模大以及资源的异构性, 在 K-medoids 聚类算法的基础上加入关于异构资源数目均衡度的收敛条件以进行改进, 对无人机集群和任务进行合理的预处理, 降低原问题的规模, 提高任务分配的实时性;
2. 建立了联盟形成博弈数学模型, 将任务分配问题转化为联盟划分问题; 通过证明本文联盟形成博弈模型为势博弈, 证明了纳什均衡解的存在性并为算法的设计提供了理论基础;
3. 设计了基于联盟形成博弈的分布式任务分配算法, 主要包含了最大加权匹配阶段和无人机转移联盟判断阶段, 使得算法相比随机寻优更具有导向性, 能够获得较好的分配效果并且提高任务分配的实时性.

<p align="center"> 
<img src="images/2024-Distributed-Coalition-Xue/Overall_flowchart_of_the_coalition_formation_game_model.png" width=60%/>
<img src="images/2024-Distributed-Coalition-Xue/Overall_flowchart_of_the_task_allocation_algorithm_based_on_coalition_formation_game.png" width=25%/>
</p>

Overall_flowchart_of_the_coalition_formation_game_model

<!-- <p align="center"> <img src="images/2024-Distributed-Coalition-Xue/Overall_flowchart_of_the_task_allocation_algorithm_based_on_coalition_formation_game.png" width=30%/> </p> -->

Overall_flowchart_of_the_task_allocation_algorithm_based_on_coalition_formation_game

基于联盟形成博弈的任务分配算法整体流程图

1. Initial task allocation result.
2. Calculate or update the benefit matric.
3. Matching based on maximum weighed principle.
4. Update the required resources of task, design rules for profit checking.
5. Obtain the final stable result of this layer.
6. Update the carried resources and requirements of tasks.
7. Requirements of a certain task are an empty set.
   1. if true, jump to 2.
   2. else go to 8.
8. Exit.When all tasks exit, obtain the final coalition structure.

## 2 问题描述

1. 数学建模
   1. 无人机建模
   2. 任务建模
   3. 联盟形成博弈数学模型
2. 目标函数与约束条件

- m 项任务分配给 n 个异构无人机
- 旋翼无人机: 异构无人机 - 量化为携带资源的类型与数目的差别, 即资源向量
- 任务:
  - 每个任务包含多种子任务, 且每个子任务对资源需求不同, 量化为资源需求向量
  - 一个任务需要多个多种无人机协同完成, 协作完成同一个任务的无人机称作一个联盟 coalition
- 假设:
  - 无人机 任务 状态信息已知且不变
  - 无人机可两两通讯

### 2.1 数学建模

#### 2.1.1 无人机建模

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

#### 2.1.2 任务建模

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

#### 2.1.3 联盟形成博弈数学模型

- 通过将无人机集群任务分配问题建模为联盟形成博弈，证明了该博弈模型可以构造为势博弈。
- 势博弈的性质保证了纳什均衡解的存在性，并且可以通过最小化或最大化势函数 $S R$ 来求解。
- 基于此理论，可以设计合理的联盟形成博弈算法，实现最终的任务分配。

##### 1. 联盟形成博弈模型

将无人机集群任务分配问题建模为联盟形成博弈模型，定义如下：

- 博弈模型：$\boldsymbol{G}=(\boldsymbol{U}, \boldsymbol{E}, \varepsilon, \boldsymbol{R})$
  - $\boldsymbol{U}$：无人机集合。
  - $\boldsymbol{E}=\left[e_{u_{1}}, e_{u_{2}}, \ldots, e_{u_{n}}\right]$：无人机选择的任务集合，等价于任务集合 $\boldsymbol{T}$，即 $\boldsymbol{E}_{U}=\boldsymbol{T}$。
  - $\varepsilon$：评估无人机收益的效用函数。
  - $\boldsymbol{R}$：评估各个联盟效用的函数。

##### 2. 联盟划分问题

原问题转化为联盟划分问题，无人机选择策略后形成 $(m+1)$ 个任务联盟。博弈目标是无人机集合选择合适策略，得到稳定的联盟结构 $\boldsymbol{C S}$：

- 联盟结构：$\boldsymbol{C S}=\left\{\boldsymbol{c}_{t_{0}}, \boldsymbol{c}_{t_{1}}, \boldsymbol{c}_{t_{2}}, \ldots, \boldsymbol{c}_{t_{m}}\right\}$
  - $\boldsymbol{c}_{t_{0}}$：未分配任务的无人机联盟集合。
  - $\boldsymbol{c}_{t_{j}}$：执行任务 $t_{j}$ 的无人机联盟集合。

##### 3. 无人机收益

无人机 $u_{i}$ 加入联盟 $c_{t_{j}}$ 时，收益 $r_{u_{i}}\left(t_{j}\right)$ 包含三部分：

- 资源贡献：$\operatorname{val}\left(u_{i}, t_{j}\right)$
- 路径成本：$\operatorname{cost}\left(u_{i}, t_{j}\right)$
- 威胁代价：$\operatorname{risk}\left(u_{i}, t_{j}\right)$

###### 3.1 资源贡献

资源贡献 $\operatorname{val}\left(u_{i}, t_{j}\right)$ 定义为：

$$
\operatorname{val}\left(u_{i}, t_{j}\right)=\left\{\begin{array}{ll}
\boldsymbol{K}(j,:) \boldsymbol{I}-P O, & \text { if } \boldsymbol{K}(j,:) \boldsymbol{I}-P O>0 \\
0, & \text { otherwise }
\end{array}\right.
$$

- $\boldsymbol{K}$：权重矩阵，权衡资源在价值收益中的比重。
- $\boldsymbol{K}(j,:)=\left\{k_{j 1}, k_{j 2}, \ldots, k_{j l}\right\}$：任务 $t_{j}$ 各异构资源的重要程度。
- $\boldsymbol{I}=\left\{i_{1}, i_{2}, \ldots, i_{l}\right\}$：无人机可利用的每类资源的数目。
- $O$：无人机未利用的总资源数目。

###### 3.2 路径成本

路径成本 $\operatorname{cost}\left(u_{i}, t_{j}\right)$ 定义为：

$$
\operatorname{cost}\left(u_{i}, t_{j}\right)=\left\{\begin{array}{ll}
1-\boldsymbol{d}\left(u_{i}, t_{j}\right) / \sqrt{x^{2}+y^{2}}, & \operatorname{val}\left(u_{i}, t_{j}\right)>0 \\
\mu, & \text { otherwise }
\end{array}\right.
$$

- $\boldsymbol{d}$：无人机与任务之间的欧氏距离。
- $x$ 和 $y$：任务环境区域大小。
- $\mu$：小于 0 的常数。
  - 当 $\operatorname{val}\left(u_{i}, t_{j}\right)$ 为 0 时，设计 $r_{u_{i}}\left(t_{j}\right)$ 小于 0 。含义是当无人机 $u_{i}$ 加入任务 $t_{j}$ 联盟无法贡献资源时，加入该联盟的收益小于在 $c_{t_{0}}$ 中的收益。

###### 3.3 威胁代价

威胁代价 $\operatorname{risk}\left(u_{i}, t_{j}\right)$ 定义为：

$$
\operatorname{risk}\left(u_{i}, t_{j}\right)=\text { value }_{\mathrm{u}_{\mathrm{i}}} \mathrm{a}_{\mathrm{t}_{\mathrm{j}}}
$$

- 表示任务的探测雷达和攻击能力等因素对无人机造成的威胁代价评估。

##### 4. 任务收益

任务收益 $r_{u_{i}}\left(t_{j}\right)$ 定义为：

$$
r_{u_{i}}\left(t_{j}\right)=\alpha \mathbf{v a l}\left(u_{i}, t_{j}\right)+\beta \operatorname{cost}\left(u_{i}, t_{j}\right)-\gamma \operatorname{risk}\left(u_{i}, t_{j}\right)
$$

- $\alpha, ~ \beta, ~ \gamma$：常数权重值，分别决定资源重叠度、路径成本和威胁代价在收益中的比重。

##### 5. 联盟效用

任务 $t_{j}$ 的联盟效用 $\boldsymbol{R}\left(c_{t_{j}}\right)$ 定义为：

$$
\boldsymbol{R}\left(c_{t_{j}}\right)=\sum_{u_{i} \in c_{t_{j}}} r_{u_{i}}\left(t_{j}\right)
$$

任务分配问题的总收益 $S R$ 定义为：

$$
S R=\sum_{c_{t_{j}} \in C S} \boldsymbol{R}\left(c_{t_{j}}\right)
$$

##### 6. 无人机效用函数

无人机效用函数 $\varepsilon_{u_{i}}\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)$ 定义为：

$$
\varepsilon_{u_{i}}\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)=\boldsymbol{R}\left(c_{u_{i}}\right)-\boldsymbol{R}\left(c_{u_{i}} \mid u_{i}\right)
$$

- $c_{u_{i}}$：无人机 $u_{i}$ 选择策略 $e_{u_{i}}$ 时加入的任务联盟。
- $\boldsymbol{R}\left(c_{u_{i}} \mid u_{i}\right)$：将 $u_{i}$ 从原所属联盟中删除后的剩余联盟效用。

##### 7. 联盟博弈理论和建模

###### 纳什均衡

博弈的目标是获得最终稳定的纳什均衡解 $\boldsymbol{C S}^{*}$。纳什均衡是博弈论中的一种解概念，定义如下：

**定义 1：纳什均衡**

纳什均衡是指满足以下性质的策略组合：在其他玩家策略不变的前提下，任何一位玩家在此策略组合下单方面改变自己的策略都不会提高自身的收益。在本文中，纳什均衡定义为：

$$
\boldsymbol{C S}^{*}=\arg \max S R(\boldsymbol{C S}) \Rightarrow \varepsilon_{u_{i}}\left(e_{u_{i}}^{*}, \boldsymbol{E}_{-u_{i}}^{*}\right) \geqslant \varepsilon_{u_{i}}\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}^{*}\right), \forall u_{i} \in \boldsymbol{U}
$$

- $e_{u_{i}}^{*}$：联盟结构 $\boldsymbol{C S}^{*}$ 中无人机 $u_{i}$ 的策略。
- $\boldsymbol{E}_{-u_{i}}^{*}$：$u_{i}$ 之外的无人机在稳定联盟结果中的策略集合。
- $\bar{e}_{u_{i}}$：除 $e_{u_{i}}^{*}$ 外的任意策略，$\bar{e}_{u_{i}} \in \boldsymbol{T}$。

当任何无人机都无法通过独自调整所在联盟来提高收益时，达到稳定的联盟结构。联盟形成博弈的目标是得到稳定的联盟结构，与任务分配的目标一致。

###### 势博弈

势博弈是一种特殊的博弈模型，其定义如下：

**定义 2：势博弈**

在博弈模型中，若存在一个势函数 $P$，满足：

$$
\varepsilon_{u_{i}}\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)-\varepsilon_{u_{i}}\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)=P\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)-P\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right), \forall u_{i} \in \boldsymbol{U}, \forall e_{u_{i}}, \bar{e}_{u_{i}} \in \boldsymbol{E}_{\boldsymbol{U}}
$$

则该博弈称为势博弈。势函数 $P$ 能够反映参与者改变策略后的效用变化。

**性质 1：势博弈的性质**

- 势博弈必然存在纳什均衡点。
- 通过最小化或最大化势函数，可以达到博弈的纳什均衡。
- 在有限递增特性下，势博弈能够在有限时间内收敛到纳什均衡状态。

###### 博弈模型的势博弈构造

本文建立的博弈模型 $\boldsymbol{G}=(\boldsymbol{U}, \boldsymbol{E}, \varepsilon, \boldsymbol{R})$ 可以构造为势博弈，且其纳什均衡解存在。

**定理 1：博弈模型的势博弈构造**

博弈模型 $\boldsymbol{G}$ 可以构造为势博弈，且其纳什均衡解存在。

**证明：**

设计势函数为：

$$
P\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)=S R\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)=\sum_{c_{t_{j}} \in C S} \boldsymbol{R}\left(c_{t_{j}}\right)
$$

当无人机 $u_{i}$ 的策略由 $e_{u_{i}}$ 改为 $\bar{e}_{u_{i}}$ 时，势函数的变化为：

$$
\begin{aligned}
P\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)-P\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right) &= S R\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)-S R\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right) \\
&= \left[\boldsymbol{R}\left(c_{e_{u_{i}}}\right)+\boldsymbol{R}\left(c_{\bar{e}_{u_{i}}}\right)+\sum_{c_{m} \in C S_{\Delta}} \boldsymbol{R}\left(c_{m}\right)\right] -\left[\boldsymbol{R}\left(\bar{c}_{e_{u_{i}}}\right)+\boldsymbol{R}\left(\bar{c}_{\bar{e}_{u_{i}}}\right)+\sum_{c_{m} \in C S_{\Delta}} \boldsymbol{R}\left(c_{m}\right)\right]
\end{aligned}
$$

其中：

- $\boldsymbol{C S}_{\Delta}=\boldsymbol{C S} \mid\left\{c_{e_{u_{i}}}, c_{\bar{e}_{u_{i}}}\right\}$：不包含任务 $e_{u_{i}}$ 和 $\bar{e}_{u_{i}}$ 的所有任务联盟。
- $\bar{c}_{e_{u_{i}}}=c_{e_{u_{i}}} \mid u_{i}$：从联盟 $c_{e_{u_{i}}}$ 中移除无人机 $u_{i}$。
- $c_{\bar{e}_{u_{i}}}=\bar{c}_{\bar{e}_{u_{i}}} \mid u_{i}$：从联盟 $\bar{c}_{\bar{e}_{u_{i}}}$ 中移除无人机 $u_{i}$。

进一步推导可得：

$$
\begin{aligned}
P\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)-P\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right) &= \left(\boldsymbol{R}\left(c_{e_{u_{i}}}\right)-\boldsymbol{R}\left(\bar{c}_{e_{u_{i}}}\right)\right)-\left(\boldsymbol{R}\left(\bar{c}_{\bar{e}_{u_{i}}}\right)+\boldsymbol{R}\left(c_{\bar{e}_{u_{i}}}\right)\right) \\
&= \varepsilon_{u_{i}}\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)-\varepsilon_{u_{i}}\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)
\end{aligned}
$$

因此，存在势函数 $P$ 能够等量反映单个无人机策略的变化。根据定义 2，博弈模型 $\boldsymbol{G}$ 可构造为势博弈，且其纳什均衡解必然存在。

### 2.2 目标函数与约束条件

#### 1. 目标函数

无人机集群任务分配的目标是最大化总收益 $S R$，具体表示为：

$$
\max S R = \max \sum_{c_{t_{j}} \in C S} \boldsymbol{R}\left(c_{t_{j}}\right)
$$

- $S R$：任务分配问题的总收益。
- $\boldsymbol{R}\left(c_{t_{j}}\right)$：任务 $t_{j}$ 对应的联盟 $c_{t_{j}}$ 的效用。

#### 2. 约束条件

任务分配问题需要满足以下约束条件：

##### （1）无人机飞行速度约束

$$
v_{u_{i}} \leqslant v_{u_{i}}^{\max }, \forall u_{i} \in U
$$

- $v_{u_{i}}$：无人机 $u_{i}$ 的飞行速度。
- $v_{u_{i}}^{\max }$：无人机 $u_{i}$ 的最大飞行速度。
- **含义**：执行任务的无人机具有最大速度约束，意味着无人机到达任务地点具有最短所需时间约束。

##### （2）任务时效性约束

$$
\operatorname{mint}\left(u_{i}, t_{j}\right) \leqslant t_{t_{j}}^{\operatorname{maxstart}}, \forall u_{i} \in U, t_{j} \in T
$$

- $\operatorname{mint}\left(u_{i}, t_{j}\right)$：无人机 $u_{i}$ 到达任务 $t_{j}$ 的最短所需时间。
- $t_{t_{j}}^{\operatorname{maxstart}}$：任务 $t_{j}$ 的最晚开始时间。
- **含义**：任务的时效性约束，即无人机需要在任务截止时间前到达任务地点。

##### （3）无人机执行任务模式约束

$$
c_{t_{j}} \cap c_{t_{i}}=\varnothing, \forall t_{i}, t_{j} \in \boldsymbol{T}
$$

- $c_{t_{j}}$ 和 $c_{t_{i}}$：分别表示执行任务 $t_{j}$ 和 $t_{i}$ 的无人机联盟。
- **含义**：不同任务对应的联盟之间交集为空集，即每架无人机最多只能执行一个任务。

#### 3. 数学模型

基于上述目标函数和约束条件，任务分配问题的数学模型可表示为：

$$
\begin{array}{c}
\max _{C S} S R \\\\
\text { s.t. } \quad v_{u_{i}} \leqslant v_{u_{i}}^{\max }, \forall u_{i} \in U, \\\\
\operatorname{mint}\left(u_{i}, t_{j}\right) \leqslant t_{t_{j}}^{\operatorname{maxstart}}, \forall u_{i} \in U, t_{j} \in T, \\\\
c_{t_{j}} \cap c_{t_{i}}=\varnothing, \forall t_{i}, t_{j} \in \boldsymbol{T}
\end{array}
$$

#### 4. 任务分配算法的目标

任务分配算法的目标是求取任务对应的无人机联盟，满足以下条件：

1. **任务需求与约束**：联盟内的无人机能够满足任务需求和约束条件。
2. **集群效能**：充分发挥无人机集群的效能。
3. **路径成本与资源优化**：降低路径成本，减少联盟内冗余资源的数量。

至此，任务分配问题已建模为最大化势函数的联盟划分问题。接下来的工作是基于联盟形成博弈的算法设计，以实现最优任务分配。

## 3 联盟形成博弈算法设计

1. 基于异构资源的改进 K-medoids 聚类算法
   1. 无人机集群聚类
   2. 任务聚类
2. 基于联盟形成博弈的任务分配算法
3. 算法复杂度分析

<p align="center"> 
<img src="./images/2024-Distributed-Coalition-Xue/Alg1.png" width=50%/>
<img src="./images/2024-Distributed-Coalition-Xue/Alg2.png" width=25%/>
</p>

算法 1: 基于异构资源的改进 K-medoids 聚类算法

```cpp
输入: 无人机集合 U 的机载资源信息 Res_u 和位置信息 position_u，类簇个数: k, 初始聚类结果
主迭代:
for j = 1:l do
 feature_j <- bsxfun(@minus, Res_U^j, mean(Res_U^j)), 计算无人机每类机载资源的特征数据值;
end for

for i = 1:k do
  while 同一类簇内某类资源的特征数据之和 sum(feature_j(find(cluster == i))) >= β do
    随机选择 k 个无人机作为初始类簇中心 prev_medoids_index;
    重复 K-medoids 聚类算法得到新的聚类结果；
  end while
end for
输出: 最终无人机集群聚类结果
```

算法 2: 无人机转移联盟判断算法

```cpp
输入 最大加权匹配结果 E' 和 任务信息 TIN
主迭代:
for j = 1:m do
    for ui \in ctj and ui == arg_{uk \in ctj} min ruk(tj) do
        epsilon_ui(eui, E_{-ui}) <- R(cui) - R(cui | ui)
        for \bar{cui} != cui do
            epsilon_ui(\bar{eui}, E_{-ui}) <- R(\bar{cui} \union ui) - R(\bar{cui})
            if epsilon_ui(eui, E_{-ui}) >= epsilon_ui(\bar{eui}, E_{-ui}) then
                ui \in cui
            else
                ui \in ct0
            end if
        end for
    end for
    TINtj <- TIN^'tj
end for
输出: 最终联盟结果和更新后的 V
```

### 3.1 基于异构资源的改进 K-medoids 聚类算法

聚类是按照某个指标将数据集划分为不同的簇或类, 最常见的是基于距离进行分割, 尽可能增大类内元素的相似性, 同时降低类间元素的相似性.

对相似的数据进行分割归簇, 实现聚类. 为了降低原问题的规模大小, 本文在进行全局任务分配前, 作基于距离和资源对任务和无人机集群聚类的预处理, 从而缩短任务分配的计算时间, 提高任务分配效率 [18].


#### 3.1.1 无人机集群聚类

##### 1. 聚类目标

为了保证后续类簇内局部任务分配的合理性，聚类结果需要满足以下条件：

- 资源均衡：每个类簇内的资源相对均衡。
- 距离与异构资源结合：基于距离和异构资源对无人机进行合理聚类。

---

##### 2. K-medoids 聚类算法

本文采用 K-medoids 算法对无人机进行初始聚类。K-medoids 算法相比 K-means 具有更强的鲁棒性，异常点不会严重影响聚类结果。

2.1 算法定义

- 数据集：$\boldsymbol{P}=\left\{p_{1}, p_{2}, \ldots, p_{n}\right\}$，表示 $n$ 个无人机点。
- 类簇划分：将数据集分为 $k$ 个类簇，记为 $\boldsymbol{C}=\left\{\boldsymbol{c}_{1}, \boldsymbol{c}_{2}, \ldots, \boldsymbol{c}_{k}\right\}$。
- 类簇中心：每个类簇 $\boldsymbol{c}_{i}$ 的中心点记为 $o_{i}$。

  2.2 代价函数
  设计代价函数 $E$ 评估聚类结果，使其尽可能小：

  $$
  E=\sum_{j=1}^{k} \sum_{P \in c_{j}}\left\|p-o_{i}\right\|_{2}
  $$

- $E$：各个样本点距离各类簇中心的误差之和。
- $\left\|p-o_{i}\right\|_{2}$：样本点 $p$ 与类簇中心 $o_{i}$ 的欧氏距离。

---

##### 3. 异构资源均衡约束

为了在聚类结果中考虑异构资源的均衡性，本文在收敛条件中加入对异构资源的约束。

###### 3.1 资源零均值化处理

对每类资源进行零均值化处理，将资源数目转化为一组和为零的正负特征数据：

$$
\sum_{u_{i} \in U} \bar{R}_{u_{i}}^{j}=0
$$

- $\bar{R}_{u_{i}}^{j}$：无人机 $u_{i}$ 的第 $j$ 类资源经过零均值化处理后的值。

###### 3.2 资源均衡条件

类簇内每类资源的均衡度通过以下条件约束：

$$
\left|\sum_{U \in \boldsymbol{c}_{i}} \bar{R}_{u_{i}}^{j}\right|<\beta, \quad i=\{1,2, \ldots, k\}, \quad j=\{1,2, \ldots, l\}
$$

- $\beta$：设定的常值，用于控制资源均衡的阈值。
- $\sum_{U \in \boldsymbol{c}_{i}} \bar{R}_{u_{i}}^{j}$：类簇 $\boldsymbol{c}_{i}$ 内第 $j$ 类资源的总和。

当聚类结果满足上述条件时，迭代结束，此时类簇内每类资源满足均衡要求。

##### 4. 改进的 K-medoids 聚类算法

基于异构资源的改进 K-medoids 聚类算法的核心步骤如下：

1. 初始化：随机选择 $k$ 个类簇中心。
2. 迭代聚类：
   - 将每个无人机点分配到最近的类簇中心。
   - 更新类簇中心，选择使代价函数 $E$ 最小的点作为新的中心。
3. 资源均衡检查：
   - 检查每个类簇的资源是否满足均衡条件 $\left|\sum_{U \in \boldsymbol{c}_{i}} \bar{R}_{u_{i}}^{j}\right|<\beta$。
   - 如果满足，聚类结束；否则，重新初始化类簇中心并进入下一轮迭代。

##### 5. 算法优势

- 降低问题规模：通过合理的预处理（如资源零均值化处理），降低原任务分配问题的规模。
- 资源均衡性：在聚类过程中考虑异构资源的均衡性，为后续任务分配提供更合理的类簇划分。
- 鲁棒性：K-medoids 算法对异常点具有较强的鲁棒性，能够更好地适应实际场景中的噪声数据。

#### 3.1.2 任务聚类

任务聚类的具体实现步骤与上文一致, 将 m 个任务进行聚类划分为 k 个类簇. 经过聚类后的无
人机集群和任务类簇可一一对应, 构成一个完整分区. 当无人机集群的总资源满足任务需求且单无人
机和单任务资源相对均衡时, 经过基于资源聚类后的分区内无人机资源亦能够满足任务需求.
通过聚类的预处理极大地降低了原问题的规模, 从而缩短任务分配的计算时间, 显著提高任务分
配效率.

### 3.2 基于联盟形成博弈的任务分配算法

本文提出的基于联盟形成博弈的分布式任务分配算法，通过最大加权匹配和无人机转移联盟判断，实现了无人机集群的高效任务分配。算法具有纳什稳定性和收敛性，能够全局优化效用并提高实时性。

#### 1. 算法概述

本文提出了一种基于联盟形成博弈的分布式任务分配算法，通过聚类预处理和分阶段求解，实现无人机集群的任务分配。算法的主要流程包括：

1. 最大加权匹配阶段：基于收益矩阵进行任务与无人机的初始匹配。
2. 无人机转移联盟判断阶段：检查无人机的联盟稳定性，更新任务需求和收益矩阵。
3. 迭代直至稳定：重复匹配和检查，直到所有无人机达到稳定状态。

#### 2. 算法定义与核心概念

**定义 3**：匹配（Matching）

给定一个二分图 $G$，$M$ 是 $G$ 的一个子图，且其边集 $\{E\}$ 中的任意两条边都不依附于同一个顶点，则 $M$ 称为 $G$ 的一个匹配。

**定义 4**：最大加权匹配（Maximum Weighted Matching）

在带有权值边的二分图中，使匹配边上的权值和最大的匹配。

#### 3. 算法流程

1. 最大加权匹配阶段
2. 无人机转移联盟判断阶段
3. 迭代直至稳定

##### 3.1 最大加权匹配阶段

目标:

将任务和无人机集合看作二分图中的 $X$ 和 $Y$ 集合，基于收益矩阵进行最大加权匹配。

步骤

1. 初始化：
   - 所有任务标记为未分配，对应的联盟结构为空集。
   - 所有无人机均未被分配任务，初始联盟 $c_{t_{0}}=\boldsymbol{U}$。
2. 计算收益矩阵：
   - 计算每个无人机单独执行每项任务的收益，得到 $M \times N$ 维收益矩阵 $V$。
   - 若无人机无法在任务截止时间前到达，则收益记为 $0$。
3. 构建加权二分图：
   - 将收益矩阵转化为加权二分图，初始化顶标：
     - 任务顶点设置为各无人机单独加入该联盟后收益的最大值 $x[i]$。
     - 无人机顶点预设为 $y[i]=0$。
   - 匹配原则：顶点值相加等于与其相连边的权重 $w[i][j]$，即 $x[i]+y[j]=w[i][j]$。
4. 虚拟节点处理：
   - 增加虚拟节点，权重设为特别小的值，确保不影响正常匹配。
   - 匹配完成后删除虚拟节点匹配部分。

##### 3.2 无人机转移联盟判断阶段

目标: 检查无人机的联盟稳定性，更新任务需求和收益矩阵。

步骤:

1. 更新任务需求：
   - 根据匹配结果更新任务剩余所需资源。
2. 检查无人机效用：
   - 计算无人机在当前联盟中的效用：
     $$
     \varepsilon_{u_{i}}\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)=\boldsymbol{R}\left(c_{u_{i}}\right)-\boldsymbol{R}\left(c_{u_{i}} \mid u_{i}\right)
     $$
     - $c_{u_{i}}$：无人机 $u_{i}$ 当前加入的任务联盟。
     - $\boldsymbol{R}\left(c_{u_{i}} \mid u_{i}\right)$：将 $u_{i}$ 从原联盟中删除后的联盟效用。
   - 计算无人机加入其他任务联盟时的效用：
     $$
     \varepsilon_{u_{i}}\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)=\boldsymbol{R}\left(\bar{c}_{u_{i}} \cup u_{i}\right)-\boldsymbol{R}\left(\bar{c}_{u_{i}}\right)
     $$
     - $\bar{c}_{u_{i}}$：无人机 $u_{i}$ 选择策略 $\bar{e}_{u_{i}}$ 时加入的任务联盟。
3. 转移规则：
   - 若存在 $\bar{e}_{u_{i}}$ 使 $\varepsilon_{u_{i}}\left(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right) > \varepsilon_{u_{i}}\left(e_{u_{i}}, \boldsymbol{E}_{-u_{i}}\right)$，则无人机退出原联盟，等待重新分配。
   - 否则，无人机保持当前联盟关系不变。
4. 更新收益矩阵：
   - 根据检查结果更新未分配无人机集合 $c_{t_{0}}$ 和任务剩余需求，计算新的收益矩阵 $V$。


##### 3.3 迭代直至稳定

目标: 重复匹配和检查，直到所有无人机达到稳定状态。

步骤:

1. 重复匹配：
   - 根据新的收益矩阵进行最大加权匹配。
2. 检查稳定性：
   - 对新匹配的无人机进行效用检查，更新联盟结构。
3. 收敛条件：
   - 所有无人机稳定，即无法通过更改所属联盟提高自身收益。

#### 4. 算法性质

##### 4.1 纳什稳定性

最终联盟结构 $\boldsymbol{C S}^{*}$ 是纳什稳定的。证明如下：

- 假设 $\boldsymbol{C S}^{*}$ 不稳定，则存在至少一个无人机 $u_{i}$ 可以通过切换联盟提高收益。
  - 即存在 $\bar{e}_{u_{i}}$:
    $$
    \varepsilon_{u_{i}}(\bar{e}_{u_{i}}, \boldsymbol{E}_{-u_{i}}) > \varepsilon_{u_{i}}(e_{u_{i}}, \boldsymbol{E}_{-u_{i}})
    $$
- 根据转移规则，$u_{i}$ 可以转换至新联盟，与假设矛盾。
- 因此，$\boldsymbol{C S}^{*}$ 是纳什稳定的。

##### 4.2 收敛性

- 根据势博弈的性质，联盟结构经过有限次改变会收敛至纳什均衡。
- 算法具有良好的鲁棒性和实时性

### 3.3 算法复杂度分析

问题描述

将 $m$ 项任务分配给 $n$ 个无人机的任务分配问题，算法的主要工作量在于对联盟效用的计算。假设每次联盟效用的计算复杂度为 $CP$。

复杂度对比

- 枚举法：$O_{\text{enum}} = (m+1)^{n} \times CP$（指数级）。
- 本文算法：$O_{CFG} \leqslant (m n + m(m-1)) \times CP \times l$（多项式级）。

#### 枚举法的复杂度

枚举法的算法复杂度为：

$$
O_{\text{enum}} = (m+1)^{n} \times CP
$$

- 特点：这是一个指数级的复杂度。
- 问题：对于较大的无人机数量 $n$，计算量会急剧增加，难以在实际应用中高效运行。

#### 本文算法的复杂度

本文提出的基于联盟形成博弈的分布式任务分配算法的复杂度为：

$$
\begin{aligned}
O_{CFG} &= (m n + m(m-1)) \times CP + \left(m n^{\prime} + m(m-1)\right) \times CP + \cdots \\
&\leqslant (m n + m(m-1)) \times CP \times l
\end{aligned}
$$

- 参数说明：
  - $m$：任务数量。
  - $n$：无人机数量。
  - $n^{\prime}$：未分配无人机的数量（随着迭代减少）。
  - $l$：迭代次数，取决于无人机和任务的参数。
- 特点：这是一个多项式型复杂度。
- 优势：相比枚举法的指数级复杂度，本文算法显著降低了问题的复杂度和计算量。

总结

本文提出的基于联盟形成博弈的分布式任务分配算法，通过多项式级的复杂度，显著降低了任务分配问题的计算量，能够高效处理较大规模的无人机集群任务分配问题。

## 4 仿真验证

1. 基于异构资源的改进 K-medoids 聚类算法结果分析
2. 基于联盟形成博弈的任务分配算法结果分析

### 基于异构资源的改进 K-medoids 聚类算法结果分析

<p align="center"> 
<img src="./images/2024-Distributed-Coalition-Xue/cluster1.png" width=49%/>
<img src="./images/2024-Distributed-Coalition-Xue/cluster2.png" width=45%/>
</p>

### 基于联盟形成博弈的任务分配算法结果分析

为了验证本文基于联盟形成博弈任务分配算法，仿真场景设置在某一次聚类结果的分区内，共有九架异构无人机和三个任务，初始参数不具体展开．在相同参数设置下与枚举法进行对比，从而验证算法的有效性。

基于联盟形成博亦算法运行结果如表 3 所示，重复运行 10 次，平均仿真时间为 0.094 s ．上述结果中，每个联盟内的资源能够满足任务需求；无人机不需要全部参与到任务联盟中，避免了资源的浪费．基于相同仿真参数设置，利用枚举法来验证上述结果．运行结果相同，重复运行十次，平均仿真时间为 1.392 s ．

![alt text](./images/2024-Distributed-Coalition-Xue/image5-6.png)

为了进一步验证算法的有效性和实时性，设置在不同的任务和无人机数量下与枚举法，粒子群算法，遗传算法进行联盟总收益和算法运行时间的对比．首先固定任务数量 m=3 ，取无人机数量 $n \in\{5,6,7,8,9,10,11\}$ ，各算法重复运行 50 次．图 5 是本文算法与遗传算法，粒子群算法相比于全局最优的收益衰减量，图 6 给出了算法相对于枚举法的平均运行时间之比．

由图 5 可知，当任务数量固定时，遗传算法和粒子群算法在解决无人机数量较少的任务分配问题具有优势，而随着无人机数量的增加，本文算法的优势更加明显．当无人机数量 n=11 时，本文算法得到的联盟总收益衰减量仍未超过 3.30 \% ．由图 6 可以看出当任务数量 m=3 固定时，无人机数量 n>8 时，算法运行时间已不足枚举法运行时间的 0.1 ，具有较好的实时性。无人机 - 任务的组合数量增加意味着粒子群算法和遗传算法的搜索空间变大，计算量增加且算法搜索最优解难度变大；而通过本文算法逻辑的设计使寻优具有导向性，算法经过有限次的迭代即可收玫到一个稳定且质量较高的纳什均衡解。

![alt text](./images/2024-Distributed-Coalition-Xue/image7-8.png)

其次分别设置在任务数量 $m \in\{1,2,3,4,5,6\}$ ，对应的无人机数量 $n \in\{3,6,9,10,12,14\}$ 的场景下．各算法重复运行 50 次，图 7 是不同算法相对于枚举法的目标函数衰减量仿真结果，图 8 给出了各算法运行时间示意图．需要特别说明的是由于粒子群算法和遗传算法的运行时间与设置的迭代次数相关，本文取联盟总收益稳定收敛时平均所需迭代次数。

由图 7 和图 8 可得当无人机和任务数量较少时，基于联盟形成博弈的分布式任务分配算法可以获得与枚举法相同的收益。随着无人机和任务数量的增加，本文算法仍能在收益上表现出与遗传算法相似较好的性能，得到较高的总联盟收益值，同时在实时性上更加优越。从图中可以看出当任务数量 m=6 ，无人机数量 n=14 时，本文算法收益可近似达到全局最优的 96 \% 以上，同时算法运行时间接近遗传算法所需运行时间的 1 / 2 ．粒子群算法收敛速度较快，但最终解的质量平均质量有待提高，过早收敛现象使其容易陷入局部最优，算法性能依赖于参数的设计。

**仿真表明，本文提出的基于联盟形成博弈的异构无人机集群分布式任务分配算法能够综合考虑资源的异构性，保证较低复杂度的同时得到较优的任务分配方案，极大提高了任务分配的实时性，充分发挥集群效能。**

## 5 结论

本文给出了一种基于联盟形成博弈的异构无人机集群分布式任务分配算法，考虑了大规模无人机集群机载资源的异构性以及任务的异构需求。

基于异构资源的改进 K－medoids 聚类方法能够更合理的对问题进行预处理，降低原问题的规模；建立的联盟形成博亦模型将原任务分配问题转化为联盟划分问题，其具有势博弈的特性为任务分配算法提供理论基础；利用设计的基于联盟形成博弈的分布式任务分配算法进行求解，无需全局信息交互，提高了时效性和对通信故障的容错性。

恶劣的工作环境或突发事件的发生有时会导致无人机故障，导致整体性能的降低，众多国内外学者围绕故障检测以及容错控制问题展开研究 { }^{[19]} 。从控制层面已有很多较为成熟的针对各种执行器故障的容错自适应方案 { }^{[20]} ，此时还需要考虑集群对环境态势和动态任务迅速做出反应，在有限时间内完成联盟的重构与优化，达成新的无冲突的任务重分配方案。因此，研究任务重分配是后续研究重点。

## 6 List of references

1. Poudel S, Moh S. Task assignment algorithms for unmanned aerial vehicle networks: A comprehensive survey. IEEE Trans Veh Commun, 2022, 35: 100469.
   [doi-link](https://doi.org/10.1016/j.vehcom.2022.100469)
2. 李 鸿一, 陈 锦涛, 任 鸿儒. Random-sampling-based multi-UAV cooperative search planning for high-rise firefighting. Sci Sin-Inf, 2022, 52: 1610-1626.
   [doi-link](https://doi.org/10.1360/SSI-2022-0038)
3. Zhou W, Kuang M, Zhu J. An unmanned air combat system based on swarm intelligence. Sci Sin-Inf, 2020, 50: 363-374.
   [doi-link](https://doi.org/10.1360/SSI-2019-0196)
4. Lei Y Q, Duan H B. Decision-making of multi-UAV combat game via enhanced competitive learning pigeon-inspired optimization. Sci Sin Tech, 2024, 54: 136-148.
   [doi-link](https://doi.org/10.1360/SST-2022-0032)
5. Ju K, Mao Z H, Jiang B, et al. Task allocation and reallocation for heterogeneous multiagent systems based on potential game. Act Autom Sin, 2022, 48: 2416-2428.
6. Cui W, Li R, Feng Y. Distributed Task Allocation for a Multi-UAV System with Time Window Constraints. Drones, 2022, 6:
   [doi-link](https://doi.org/10.3390/drones6090226)
7. Chen X, Wei X M, Xu G Y. Multiple unmanned aerial vehicle decentralized cooperative air combat decision making with fuzzy situation. J Shanghai Jiaotong Univ. 2014, 48: 907-913+921.
8. Yan F, Zhu X, Zhou Z. Real-time task allocation for a heterogeneous multi-UAV simultaneous attack. Sci Sin-Inf, 2019, 49: 555-569.
   [doi-link](https://doi.org/10.1360/N112018-00338)
9. Lyu Y, Zhou R, Li X, et al. Multi-task assignment algorithm based on multi-round distributed auction. J Beijing Univ Aeronaut Astronaut, 2023, 1-14.
10. Chen Y, Sun Y, Yu H. Joint Task and Computing Resource Allocation in Distributed Edge Computing Systems via Multi-Agent Deep Reinforcement Learning. IEEE Trans Netw Sci Eng, 2024, 11: 3479-3494.
    [doi-link](https://doi.org/10.1109/TNSE.2024.3375374)
11. Xu Y, Jiang B, Yang H. Two-Level Game-Based Distributed Optimal Fault-Tolerant Control for Nonlinear Interconnected Systems. IEEE Trans Neural Netw Learn Syst, 2020, 31: 4892-4906.
    [doi-link](https://doi.org/10.1109/TNNLS.2019.2958948)
12. Wu H, Shang H. Potential game for dynamic task allocation in multi-agent system. ISA Trans, 2020, 102: 208-220.
    [doi-link](https://doi.org/10.1016/j.isatra.2020.03.004)
13. Czarnecki E, Dutta A. Scalable hedonic coalition formation for task allocation with heterogeneous robots. Intel Serv Robotics, 2021, 14: 501-517.
    [doi-link](https://doi.org/10.1007/s11370-021-00372-9)
14. Zhang M, Li J, Wu X. Coalition Game Based Distributed Clustering Approach for Group Oriented Unmanned Aerial Vehicle Networks. Drones, 2023, 7:
    [doi-link](https://doi.org/10.3390/drones7020091)
15. Zhang T, Wang Y, Ma Z. Task Assignment in UAV-Enabled Front Jammer Swarm: A Coalition Formation Game Approach. IEEE Trans Aerosp Electron Syst, 2023, 59: 9562-9575.
    [doi-link](https://doi.org/10.1109/TAES.2023.3323441)
16. Qi N, Huang Z, Zhou F. A Task-Driven Sequential Overlapping Coalition Formation Game for Resource Allocation in Heterogeneous UAV Networks. IEEE Trans Mobile Comput, 2023, 22: 4439-4455.
    [doi-link](https://doi.org/10.1109/TMC.2022.3165965)
17. Wang J, Jia G, Lin J. Cooperative task allocation for heterogeneous multi-UAV using multi-objective optimization algorithm. J Cent South Univ, 2020, 27: 432-448.
    [doi-link](https://doi.org/10.1007/s11771-020-4307-0)
18. Gao C, Du Y L, Bu Y N, et al. Heterogeneous UAV swarm grouping deployment for complex multiple tasks. J Syst Eng Electron, 2024, 46: 972-981.
19. Ma Y, Jiang B, Tao G. Uncertainty decomposition-based fault-tolerant adaptive control of flexible spacecraft. IEEE Trans Aerosp Electron Syst, 2015, 51: 1053-1068.
    [doi-link](https://doi.org/10.1109/TAES.2014.130032)
20. Mao Z, Jiang B, Shi P. Fault-tolerant control for a class of nonlinear sampled-data systems via a Euler approximate observer. Automatica, 2010, 46: 1852-1859.
    [doi-link](https://doi.org/10.1016/j.automatica.2010.06.052)
