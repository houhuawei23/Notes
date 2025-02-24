# Coalitional Games: 联盟博弈

A coalitional game, also known as a cooperative game, is a game theory model where players form groups to achieve a collective goal. In a coalitional game, players form coalitions, or binding agreements, to strengthen their positions and act as a single entity. Coalitional games are different from non-cooperative games, where players cannot form alliances or agreements must be self-enforced.

联盟博弈，也称为合作博弈，是一种博弈论模型，其中玩家组成团队以实现集体目标。 在联盟博弈中，玩家组成联盟或具有约束力的协议，以加强他们的地位并作为一个实体行动。 联盟博弈与非合作博弈不同，在非合作博弈中，玩家不能结成联盟或协议必须自行执行。

Here are some key concepts in coalitional games:

- Coalition: A group of players that form a binding agreement
- Coalition value: The worth of a coalition in a game, which is denoted by the symbol $v$
- The core: The set of payoff allocations that ensures no group of players has an incentive to leave their coalition
- Shapley value: An efficient solution concept that is recommended for games with a low number of players

### Hedonic Game 享乐博弈

- [wiki](https://en.wikipedia.org/wiki/Hedonic_game)

在[合作博弈论](https://en.wikipedia.org/wiki/Cooperative_game_theory)中，**享乐博弈**[[1]](https://en.wikipedia.org/wiki/Hedonic_game#cite_note-:0-1)[[2]](https://en.wikipedia.org/wiki/Hedonic_game#cite_note-:3-2)（也称为享乐**联盟形成博弈**）是一种当玩家对他们所属的群体有偏好时，对玩家[联盟](https://en.wikipedia.org/wiki/Coalition)（组）形成进行建模的博弈。享乐博弈的指定方式是给一组有限的玩家，并且对于每个玩家，给出一个对该玩家所属的所有玩家联盟（子集）的[优先排序](https://en.wikipedia.org/wiki/Preference_relation)。享乐博弈的结果包括将玩家[划分](https://en.wikipedia.org/wiki/Partition_of_a_set)为[不相交](https://en.wikipedia.org/wiki/Disjoint_sets)的联盟，也就是说，每个玩家都被分配一个唯一的组。此类分区通常称为联盟结构。

### Preference-Driven Hedonic Game 偏好驱动享乐博弈

- [偏好驱动享乐博弈（Preference-Driven Hedonic Game）的全面解析](https://metaso.cn/s/sTEaIJQ)

#### 一、定义与核心思想

偏好驱动享乐博弈（Preference-Driven Hedonic Game）是合作博弈论的一个分支，研究玩家如何基于个体偏好形成联盟结构（coalition structure）。其核心特征在于，每个玩家对包含自身的联盟具有明确的偏好关系，且偏好由其对其他玩家的分类（如朋友、敌人、中立）和排序驱动。

**关键定义**：

1. **玩家分类**：每个玩家将其他玩家分为三类：朋友（Friends）、敌人（Enemies）、中立（Neutral），并对朋友和敌人进行排名（如朋友中的优先级）。
2. **偏好扩展**：通过**双极响应式扩展原理**（Bipolar Responsive Extension），将玩家对个体的偏好扩展为对联盟的偏好。例如，玩家可能偏好包含更多高优先级朋友的联盟，或更少敌人的联盟。
3. **阈值与稳定性**：引入阈值（Thresholds）区分联盟的优劣，例如判断哪些联盟优于单独行动（个体理性阈值），或哪些联盟结构满足纳什稳定性、核心稳定性等。

#### 二、提出与发展历程

1. **起源**：享乐博弈的概念最早由 Drèze 和 Greenberg（1980）提出，后由 Bogomolnaia 和 Jackson（2002）等学者形式化。早期模型关注联盟结构的稳定性，但偏好表示较为简单。
2. **偏好驱动模型的提出**：Dimitrov 等（2006）首次将玩家分为朋友和敌人，并基于此提出偏好扩展模型。Kerkmann 等近年提出的**FEN 模型**（Friends-Enemies-Neutral）进一步完善了这一框架，允许玩家对朋友和敌人进行排序，并引入阈值机制。
3. **计算复杂性突破**：针对偏好表示的指数级复杂度问题，学者提出**布尔享乐博弈**（Boolean Hedonic Games）和**匿名编码**（Anonymous Encoding）等紧凑表示方法，将偏好转化为逻辑公式或匿名计数（如朋友数量），以降低计算难度。

#### 三、核心研究问题与理论框架

1. **稳定性概念**：
   - **个体理性**：玩家是否愿意留在当前联盟而非单独行动。
   - **纳什稳定性**：玩家是否无法通过单方面转移到其他联盟获得更高收益。
   - **核心稳定性**：是否存在群体性偏离（如多个玩家共同重组联盟）的可能。
2. **算法挑战**：
   - **复杂度**：确定稳定联盟结构的存在性通常为 NP 难问题，例如核心稳定性的判定在高维偏好下不可解。
   - **启发式方法**：针对二元偏好（Dichotomous Preferences）等受限场景，学者提出多项式时间算法，例如基于命题逻辑的 SAT 求解器优化。
3. **工程导向视角**：不同于传统博弈论的预测性分析，偏好驱动享乐博弈常被用于设计中央权威的分配机制，例如在无线网络资源分配中计算最优联盟结构。

#### 四、应用现状

1. **无线通信安全**：在 **协作干扰（Cooperative Jamming）** 场景中，源节点与干扰节点通过享乐博弈形成稳定联盟，最大化保密传输速率。北京邮电大学的研究表明，与非合作机制相比，该模型可使平均保密速率提升 14.29%。
2. **团队形成与资源分配**：
   - **科研团队组建**：根据研究人员的合作偏好（如领域匹配、历史合作效果），通过 FEN 模型划分高效团队，避免利益冲突。
   - **云资源分配**：在云计算中，用户与服务提供商的博弈可建模为享乐博弈，通过定价机制和联盟稳定性分析实现资源高效分配。
3. **社会场景优化**：
   - **高校宿舍分配**：基于学生的生活习惯、兴趣等偏好，利用决策树算法生成和谐度最高的分配方案。例如，优先匹配睡眠时间相近、卫生习惯相似的学生。
   - **联邦学习模型共享**：在分布式机器学习中，智能体根据数据分布相似性选择加入全局或本地模型联盟，通过个体稳定性保证协作效率。

#### 五、经典实例分析

1. **无线网络协作干扰联盟**（案例来源）：
   - **背景**：源节点需与干扰节点合作，通过发射噪声干扰窃听者，提升保密传输速率。
   - **博弈建模**：
     - **玩家**：源节点（N）和干扰节点（T）。
     - **偏好函数**：源节点偏好包含高干扰效率的联盟，干扰节点偏好获得更高报酬。
     - **稳定性准则**：采用“严进宽出”（FX-AE）偏好，确保联盟成员无法通过单方面退出或加入其他联盟提升效用。
   - **结果**：通过分布式享乐博弈算法收敛至稳定联盟结构，保密速率提升显著。
2. **高校宿舍智能分配系统**（案例来源）：
   - **背景**：学生因生活习惯差异易引发宿舍矛盾。
   - **模型设计**：
     - **偏好分类**：学生填写问卷，标识对室友的偏好（如作息时间、卫生习惯、兴趣等）。
     - **算法**：基于决策树分类和聚类算法，最大化整体和谐度。
   - **效果**：通过满意度反馈机制优化分配策略，减少冲突发生率。

#### 六、未来研究方向

1. **动态偏好建模**：现有研究多假设偏好静态，未来可探索动态偏好（如随时间变化的友谊关系）对联盟稳定性的影响。
2. **多目标优化**：在稳定性之外引入公平性、社会福利等目标，例如在资源分配中平衡效率与公平。
3. **跨学科应用**：将模型扩展至生物群体行为分析（如动物群体形成）或政治联盟构建（如政党合作）。

#### 七、总结

偏好驱动享乐博弈通过精细刻画个体偏好与群体互动的动态关系，为解决联盟形成中的复杂决策问题提供了理论工具。其在工程优化、社会科学等领域的成功应用，验证了模型的实用性与扩展潜力。然而，计算复杂性与动态环境适应性仍是未来研究的关键挑战。
