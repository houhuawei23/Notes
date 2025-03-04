# Game Theory

- Coalitional Games: 联盟博弈
- 享乐博弈
- 双层博弈

## A Course in Game Theory

### Contents

A course in game theory
by Martin J. Osborne and Ariel Rubinstein

- 1 Introduction 1
  - 1.1 Game theory 1
  - 1.2 Games and Solutions 2
  - 1.3 Game Theory and the Theory of Competitive Equilibrium 3
  - 1.4 Rational Behavior 4
  - 1.5 The Steady State and Deductive Interpretations 5
  - 1.6 Bounded Rationality 6
  - 1.7 Terminology and Notation 6
  - Notes 8
- I Strategic Games 9
- 2 Nash Equilibrium 11
  - 2.1 Strategic Games 11
  - 2.2 Nash Equilibrium 14
  - 2.3 Examples 15
  - 2.4 Existence of a Nash Equilibrium 19
  - 2.5 Strictly Competitive Games 21
  - 2.6 Bayesian Games: Strategic Games with Imperfect Information 24
  - Notes 29
- 3 Mixed, Correlated, and Evolutionary Equilibrium 31
  - 3.1 Mixed Strategy Nash Equilibrium 31
  - 3.2 Interpretations of Mixed Strategy Nash Equilibrium 37
  - 3.3 Correlated Equilibrium 44
  - 3.4 Evolutionary Equilibrium 48
  - Notes 51
- 4 Rationalizability and Iterated Elimination of Dominated Actions 53
  - 4.1 Rationalizability 53
  - 4.2 Iterated Elimination of Strictly Dominated Actions 58
  - 4.3 Iterated Elimination of Weakly Dominated Actions 62
  - Notes 64
- 5 Knowledge and Equilibrium 67
  - 5.1 A Model of Knowledge 67
  - 5.2 Common Knowledge 73
  - 5.3 Can People Agree to Disagree? 75
  - 5.4 Knowledge and Solution Concepts 76
  - 5.5 The Electronic Mail Game 81
  - Notes 84
- II Extensive Games with Perfect Information 9
- 6 Extensive Games with Perfect Information 89
  - 6.1 Extensive Games with Perfect Information 89
  - 6.2 Subgame Perfect Equilibrium 97
  - 6.3 Two Extensions of the Definition of a Game 101
  - 6.4 The Interpretation of a Strategy 103
  - 6.5 Two Notable Finite Horizon Games 105
  - 6.6 Iterated Elimination of Weakly Dominated Strategies 108
  - Notes 114
- 7 Bargaining Games 117
  - 7.1 Bargaining and Game Theory 117
  - 7.2 A Bargaining Game of Alternating Offers 118
  - 7.3 Subgame Perfect Equilibrium 121
  - 7.4Variations and Extensions 127
  - Notes 131
- 8 Repeated Games 133
  - 8.1 The Basic Idea 133
  - 8.2 Infinitely Repeated Games vs. Finitely Repeated Games 134
  - 8.3 Infinitely Repeated Games: Definitions 136
  - 8.4 Strategies as Machines 140
  - 8.5 Trigger Strategies: Nash Folk Theorems 143
  - 8.6 Punishing for a Limited Length of Time: A Perfect Folk Theorem for the Limit of Means Criterion 146
  - 8.7 Punishing the Punisher: A Perfect Folk Theorem for the Overtaking Criterion 149
  - 8.2 Rewarding Players Who Punish: A Perfect Folk Theorem for the Discounting Criterion 150
  - 8.9 The Structure of Subgame Perfect Equilibria Under the Discounting Criterion 153
  - 8.10 Finitely Repeated Games 155
  - Notes 160
- 9 Complexity Considerations in Repeated Games 163
  - 9.1 Introduction 163
  - 9.2 Complexity and the Machine Game 164
  - 9.3 The Structure of the Equilibria of a Machine Game 168
  - 9.4 The Case of Lexicographic Preferences 172
  - Notes 175
- 10 Implementation Theory 177
  - 10.1 Introduction 177
  - 10.2 The Implementation Problem 178
  - 10.3 Implementation in Dominant Strategies 180
  - 10.4 Nash Implementation 185
  - 10.5 Subgame Perfect Equilibrium Implementation 191
  - Notes 195
- III Extensive Games with Imperfect Information 9
- 11 Extensive Games with Imperfect Information 199
  - 11.2 Principles for the Equivalence of Extensive Games 204
  - 11.3 Framing Effects and the Equivalence of Extensive Games 209
  - 11.4 Mixed and Behavioral Strategies 212
  - 11.5 Nash Equilibrium 216
  - Notes 217
- 12 Sequential Equilibrium 219
  - 12.1 Strategies and Beliefs 219
  - 12.2 Sequential Equilibrium 222
  - 12.3 Games with Observable Actions: Perfect Bayesian Equilibrium 231
  - 12.4 Refinements of Sequential Equilibrium 243
  - 12.5 Trembling Hand Perfect Equilibrium 246
  - Notes 254
- IV Coalitional Games 9
- 13 The Core 257
  - 13.1 Coalitional Games with Transferable Payoff 257
  - 13.2 The Core 258
  - 13.3 Nonemptiness of the Core 262
  - 13.4 Markets with Transferable Payoff 263
  - 13.5 Coalitional Games without Transferable Payoff 268
  - 13.6 Exchange Economies 269
  - Notes 274
- 14 Stable Sets, the Bargaining Set, and the Shapley Value 277
  - 14.1 Two Approaches 277
  - 14.1 The Stable Sets of von Neumann and Morgenstern 278
  - 14.1 The Bargaining Set, Kernel, and Nucleolus 281
  - 14.1 The Shapley Value 289
  - Notes 297
- 15 The Nash Solution 299
  - 15.1 Bargaining Problems 299
  - 15.3 The Nash Solution: Definition and Characterization 301
  - 15.3 An Axiomatic Definition 305
  - 15.4 The Nash Solution and the Bargaining Game of Alternating Offers 310
  - 15.5 An Exact Implementation of the Nash Solution 311
  - Notes 312
- List of Results 313
- References 321
- Index 341

### Intro

The models we study assume that each decision-maker is “rational” in the sense that he is aware of his alternatives, forms expectations about any unknowns, has clear preferences, and chooses his action deliberately after some process of optimization. In the absence of uncertainty the following elements constitute a model of rational choice.

- A set $A$ of actions from which the decision-maker makes a choice.
- A set $C$ of possible consequences of these actions.
- A consequence function $g: A\to C$ that associates a consequence
  with each action.
- A preference relation (a complete transitive reflexive binary relation) $\succsim$ on the set $C$.

Sometimes the decision-maker's preferences are specified by giving a utility function $U: C\to \mathbb{R}$, which defines a preference relation $\succsim$by the condition $x\succsim y$ if and only if $U(x)\geq U(y)$.

Given any set $B\subseteq A$ of actions that are feasible in some particulan case, a $rational decision-maker chooses an action $a^*$ that is feasible (belongs to $B)$ and optimal in the sense that $g(a^*)\succsim g(a)$ for all $a\in B;$ alternatively he solves the problem $\max _{a\in BU}( g( a) ) $. An assumption upon which the usefulness of this model of decision-making depends is that the individual uses the same preference relation when choosing from different sets $B$.

In the models we study, individuals often have to make decisions under
conditions of uncertainty. The players may be

- uncertain about the objective parameters of the environment
- imperfectly informed about events that happen in the game
- uncertain about actions of the other players that are not deterministic
- uncertain about the reasoning of the other players.

#### Temiology

We denote the set of real numbers by $\mathbb{R}$, the set of nonnegative real numbers by $\mathbb{R}_{+}$, the set of vectors of $n$ real numbers by $\mathbb{R}^{n}$, and the set of vectors of $n$ nonnegative real numbers by $\mathbb{R}_+^n$.

- For $x\in\mathbb{R}^n$ and $y\in\mathbb{R}^n$ we use $x\geq y$ to mean $x_i\geq y_i$ for $i=1,\ldots,n$ and $x>y$ to mean $x_i>y_i$ for $i=1,\ldots,n$.
- We say that a function $f{:}\mathbb{R}\to\mathbb{R}$ is increasing if $f(x)>f(y)$ whenever $x>y$ and is nondecreasing if $f(x)\geq f(y)$ whenever $x>y$.
- A function $f{:}\mathbb{R}\to\mathbb{R}$ is **concave** if $f(\alpha x+(1-\alpha)x^\prime)\geq$ $\alpha f(x)+(1-\alpha)f(x')$ for all $x\in\mathbb{R}$, all $x'\in\mathbb{R}$, and all $\alpha\in[0,1]$.
- Given a function $f{:}X\to\mathbb{R}$ we denote by arg $\max_x\in Xf(x)$ the set of maximizers of $f$;
  - for any $Y\subseteq X$ we denote by $f(Y)$ the set $\{f(x):x\in Y\}$.

Throughout we use $N$ to denote the set of players. We refer to a collection of values of some variable, one for each player, as a **profle**; we denote such a profile by $(x_i)_{i\in N}$, or, if the qualifer "$i\in N$" is clear, simply $(x_i)$.

- For any profile $x=(x_j)_{j\in N}$ and any $i\in N$ we let $x_{-i}$ be the list $(x_j)_{j\in N\setminus\{i\}}$ of elements of the profile $x$ for all players except $i$.
- Given a list $x_{-i}=(x_j)_{j\in N\setminus\{i\}}$ and an element $x_i$ we denote by $(x_{-i},x_i)$ the profile $(x_i)_{i\in N}$.
- If $X_i$ is a set for each $i\in N$ then we denote by $X_{-i}$ the set $\times_{j\in N\setminus\{i\}}X_j$.

A binary relation $\succsim$ on a set $A$ is complete if $a\succsim b$ or $b\succsim a$ for every $a\in A$ and $b\in A$, reflexive if $a\succsim a$ for every $a\in A$, and transitive if $a\succsim c$ whenever $a\succsim b$ and $b\succsim c$.

- A preference relation is a complete reflexive transitive binary relation.
- If $a\succsim b$ but not $b\succsim a$ then we write $a\succ b$;
- if $a\succsim b$ and $b\succsim a$ then we write $a\sim b$.
- A preference relation $\succsim$ on $A$ is continuous if $a\succsim b$ whenever there are sequences $(a^k)_k$ and $(b^k)_k$ in $A$ that converge to $a$ and $b$ respectively for which $a^k\succsim b^k$ for all $k$.
- A preference relation $\succsim$on $\mathbb{R}^n$ is **quasi-concave** if for every $b\in\mathbb{R}^n$ the set $\{a\in\mathbb{R}^n{:}a\succsim b\}$ is **convex**;
  - it is strictly **quasi-concave** if every such set is **strictly convex**.

Let $X$ be a set. We denote by $|X|$ the number of members of $X$. A partition of $X$ is a collection of disjoint subsets of $X$ whose union is $X$. Let $N$ be a finite set and let $X\subseteq\mathbb{R}^N$ be a set.

- Then $x\in X$ is **Pareto effcient** if there is no $y\in X$ for which $y_i>x_i$ for all $i\in N$;
- $x\in X$ is **strongly Pareto efficient** if there is no $y\in X$ for which $y_i\geq x_i$ for all $i\in N$ and $y_i>x_i$ for some $i\in N$.

A **probability measure** $\mu$ on a finite (or countable) set $X$ is an additive function that associates a nonnegative real number with every subset of $X$ (that is, $\mu(B \cup C) = \mu(B) + \mu(C)$ whenever $B$ and $C$ are disjoint) and satisfies $\mu(X) = 1$. In some cases we work with probability measures over spaces that are not necessarily finite.

If you are unfamiliar with such measures, little is lost by restricting attention to the finite case; for a definition of more general measures see, for example, Chung (1974, Ch. 2).

**1. 数学符号与定义**

- **实数与非负实数**：
  - $\mathbb{R}$ 表示实数集合，$\mathbb{R}_{+}$ 表示非负实数集合。
  - $\mathbb{R}^{n}$ 表示 $n$ 维实向量集合，$\mathbb{R}_+^n$ 表示 $n$ 维非负实向量集合。
- **向量比较**：
  - 对于 $x, y \in \mathbb{R}^n$，$x \geq y$ 表示 $x_i \geq y_i$（对所有 $i=1,\ldots,n$）；$x > y$ 表示 $x_i > y_i$（对所有 $i=1,\ldots,n$）。
- **函数性质**：
  - 函数 $f{:}\mathbb{R}\to\mathbb{R}$ 是**递增的**（increasing），如果 $x > y$ 时 $f(x) > f(y)$；是**非递减的**（nondecreasing），如果 $x > y$ 时 $f(x) \geq f(y)$。
  - 函数 $f{:}\mathbb{R}\to\mathbb{R}$ 是**凹函数**（concave），如果对任意 $x, x' \in \mathbb{R}$ 和 $\alpha \in [0,1]$，满足 $f(\alpha x + (1-\alpha)x') \geq \alpha f(x) + (1-\alpha)f(x')$。
- **最大值与集合映射**：
  - 对于函数 $f{:}X\to\mathbb{R}$，$\arg\max_{x \in X} f(x)$ 表示 $f$ 的最大值点集合。
  - 对于任意 $Y \subseteq X$，$f(Y)$ 表示集合 $\{f(x): x \in Y\}$。

---

**2. 博弈论中的符号与定义**

- **玩家集合**：
  - 用 $N$ 表示玩家集合。
- **策略组合**：
  - 策略组合（profile）表示为 $(x_i)_{i\in N}$，其中每个 $x_i$ 对应一个玩家的策略。
  - 对于任意策略组合 $x = (x_j)_{j\in N}$ 和玩家 $i \in N$，$x_{-i}$ 表示除玩家 $i$ 外其他玩家的策略组合 $(x_j)_{j\in N\setminus\{i\}}$。
  - 给定 $x_{-i}$ 和 $x_i$，$(x_{-i}, x_i)$ 表示完整的策略组合 $(x_i)_{i\in N}$。
  - 如果每个玩家 $i \in N$ 有一个集合 $X_i$，则 $X_{-i}$ 表示其他玩家集合的笛卡尔积 $\times_{j\in N\setminus\{i\}}X_j$。

将 **其他玩家集合的笛卡尔积** $\times_{j\in N\setminus\{i\}}X_j$ 展开成若干项相乘的形式，可以表示为：

$$
X_{-i} = X_1 \times X_2 \times \cdots \times X_{i-1} \times X_{i+1} \times \cdots \times X_n
$$

**解释**

- $N$ 是玩家集合，$i$ 是当前玩家。
- $N \setminus \{i\}$ 表示除玩家 $i$ 外的其他所有玩家。
- 每个 $X_j$ 是玩家 $j$ 的策略集合。
- 笛卡尔积 $\times_{j\in N\setminus\{i\}}X_j$ 表示所有其他玩家策略集合的乘积，即每个玩家的策略组合在一起形成的集合。

**示例**

假设有 3 个玩家 $N = \{1, 2, 3\}$，且当前玩家是 $i = 2$，则：

$$
X_{-2} = X_1 \times X_3
$$

即玩家 1 和玩家 3 的策略集合的笛卡尔积。

如果玩家更多，例如 $N = \{1, 2, 3, 4\}$，且当前玩家是 $i = 3$，则：

$$
X_{-3} = X_1 \times X_2 \times X_4
$$

这种展开形式在博弈论中常用于描述其他玩家的策略组合对当前玩家策略的影响。

**3. 偏好关系**

- **二元关系**：
  - 集合 $A$ 上的二元关系 $\succsim$ 是**完全的**（complete），如果对任意 $a, b \in A$，$a \succsim b$ 或 $b \succsim a$ 成立；
  - 是**自反的**（reflexive），如果对任意 $a \in A$，$a \succsim a$ 成立；
  - 是**传递的**（transitive），如果 $a \succsim b$ 且 $b \succsim c$ 时，$a \succsim c$ 成立。
- **偏好关系**：
  - 偏好关系是完全、自反且传递的二元关系。
  - 如果 $a \succsim b$ 但 $b \succsim a$ 不成立，则记为 $a \succ b$；
  - 如果 $a \succsim b$ 且 $b \succsim a$，则记为 $a \sim b$。
- **连续性**：
  - 偏好关系 $\succsim$ 是**连续的**（continuous），如果当序列 $(a^k)_k$ 和 $(b^k)_k$ 分别收敛于 $a$ 和 $b$ 且 $a^k \succsim b^k$ 对所有 $k$ 成立时，$a \succsim b$ 成立。
- **拟凹性**：
  - 偏好关系 $\succsim$ 在 $\mathbb{R}^n$ 上是**拟凹的**（quasi-concave），如果对任意 $b \in \mathbb{R}^n$，集合 $\{a \in \mathbb{R}^n: a \succsim b\}$ 是**凸的**（convex）；
  - 如果该集合是**严格凸的**（strictly convex），则偏好关系是**严格拟凹的**（strictly quasi-concave）。

---

**4. 帕累托效率**

- **帕累托效率**：
  - 在集合 $X \subseteq \mathbb{R}^N$ 中，$x \in X$ 是**帕累托有效的**（Pareto efficient），如果不存在 $y \in X$ 使得 $y_i > x_i$ 对所有 $i \in N$ 成立；
  - $x \in X$ 是**强帕累托有效的**（strongly Pareto efficient），如果不存在 $y \in X$ 使得 $y_i \geq x_i$ 对所有 $i \in N$ 成立且 $y_i > x_i$ 对某些 $i \in N$ 成立。

---

**5. 概率测度**

- **概率测度**：
  - 在有限（或可数）集合 $X$ 上，概率测度 $\mu$ 是一个加性函数，将每个子集映射到一个非负实数，满足 $\mu(B \cup C) = \mu(B) + \mu(C)$（当 $B$ 和 $C$ 不相交时），且 $\mu(X) = 1$。
  - 在某些情况下，概率测度可以定义在非有限空间上。

---

以上内容主要定义了数学和博弈论中常用的符号与概念，包括实数集合、向量比较、函数性质（如凹函数、递增函数）、策略组合、偏好关系（如完全性、传递性、拟凹性）、帕累托效率以及概率测度。这些定义为博弈论和经济学中的模型构建与分析提供了基础工具。

### I Strategic Games 9 策略博弈

- Nash 均衡
- 混合战略均衡 and 相关均衡 （行动不必是确定的）
- 可理性化 and 反复剔除劣行动（假定人不知道均衡）
- 知识模型（正式检验在已定义好各种解之下的种种假设）

### 2 Nash Equilibrium 11

- 2.1 Strategic Games 11
- 2.2 Nash Equilibrium 14
- 2.3 Examples 15
- 2.4 Existence of a Nash Equilibrium 19
- 2.5 Strictly Competitive Games 21
- 2.6 Bayesian Games: Strategic Games with Imperfect Information 24
- Notes 29

#### Strategic Games

[Definition] **Strategic Games**
A strategic game consists of

- a finite set N (the set of players)
- for each player $i \in N$ a nonempty set $A_i$ (the set of actions available to player i)
- for each player $i \in N$ a preference relation $\gtrsim_i$ on $A = \times_{j \in N} A_j$ (the preference relation of player i).

If the set $A_i$ of actions of every player i is finite then the game is **finite**.

In some situations the players' preferences are most naturally defined not over action profiles but over their consequences. When modeling an oligopoly, for example, we may take the set of players to be a set of firms and the set of actions of each fırm to be the set of prices; but we may wish to model the assumption that each fırm cares only about its profit not about the profile of prices that generates that proft. To do so we introduce a set $C$ of $consequences$, a function $g{:}A\to C$ that associates consequences with action profiles, and a profile $(\succsim_i^*)$ of preference relations over $C$. Then the preference relation $\succsim_i$ of each player $i$ in the strategic game is defined as follows: $a\succsim_ib$ if and only if $g(a)\succsim_i^*g(b)$.

Sometimes we wish to model a situation in which the consequence of an action profile is affected by an exogenous random variable whose realization is not known to the players before they take their actions. We can model such a situation as a strategic game by introducing a set $C$ of consequences, a probability space $\Omega$, and a function $g{:}A\times\Omega\to C$ with the interpretation that $g(a,\omega)$ is the consequence when the action profile is $a\in A$ and the realization of the random variable is $\omega\in\Omega$. A profile of actions induces a lottery on $C;$ for each player $i$ a preference relation $\succsim_i^*$ must be specified over the set of all such lotteries. Player $i’$s preference relation in the strategic game is defined as follows: $a\succsim_ib$ if and only if the lottery over $C$ induced by $g(a,\cdot)$ is at least as good according to $\succsim_i^*$ as the lottery induced by $g(b,\cdot)$.

Under a wide range of circumstances the preference relation $\succsim_i$ of player $i$ in a strategic game can be represented by a **payoff function** $u_i{:}A\to\mathbb{R}$ also called a **utility function** , in the sense that $u_i( a) \geq u_i( b)$ whenever $a\succsim_ib$. We refer to values of such a function as payoffs (or utilities). Frequently we specify a player's preference relation by giving a payoffunction that represents it. In such a case we denote the game by $\langle N,(A_{i}),(u_{i})\rangle$ rather than $\langle N,(A_{i}),(\succsim_{i})\rangle$.

战略博弈的定义

- **战略博弈**（Strategic Game）由以下三部分组成：
  1. **玩家集合**：一个有限的集合 $N$，表示所有玩家。
  2. **行动集合**：对每个玩家 $i \in N$，存在一个非空的行动集合 $A_i$，表示玩家 $i$ 可选的行动。
  3. **偏好关系**：对每个玩家 $i \in N$，存在一个偏好关系 $\gtrsim_i$，定义在行动组合集合 $A = \times_{j \in N} A_j$ 上，表示玩家 $i$ 对不同行动组合的偏好。
- **有限博弈**：如果每个玩家的行动集合 $A_i$ 都是有限的，则该博弈称为**有限博弈**。

---

偏好关系的扩展

- **基于后果的偏好**：在某些情况下，玩家的偏好不是直接定义在行动组合上，而是定义在行动组合的**后果**（consequences）上。例如，在寡头垄断模型中，玩家的行动可能是价格，但玩家真正关心的是利润（后果）。此时，可以引入：
  - 后果集合 $C$；
  - 映射函数 $g{:}A \to C$，将行动组合映射到后果；
  - 玩家在后果上的偏好关系 $(\succsim_i^*)$。
    然后，玩家 $i$ 在战略博弈中的偏好关系定义为：$a \succsim_i b$ 当且仅当 $g(a) \succsim_i^* g(b)$。
- **随机变量的影响**：如果行动组合的后果受到外生随机变量的影响（例如市场波动），可以通过以下方式建模：
  - 后果集合 $C$；
  - 概率空间 $\Omega$，表示随机变量的可能取值；
  - 映射函数 $g{:}A \times \Omega \to C$，表示在行动组合 $a$ 和随机变量实现 $\omega$ 下的后果。
    此时，行动组合会诱导一个关于后果的**彩票**（lottery），玩家 $i$ 的偏好关系 $\succsim_i^*$ 定义在所有可能的彩票上。玩家 $i$ 在战略博弈中的偏好关系定义为：$a \succsim_i b$ 当且仅当 $g(a, \cdot)$ 诱导的彩票至少与 $g(b, \cdot)$ 诱导的彩票一样好（根据 $\succsim_i^*$）。

---

效用函数表示

- **效用函数**：在许多情况下，玩家 $i$ 的偏好关系 $\succsim_i$ 可以用一个**效用函数**（或**收益函数**）$u_i{:}A \to \mathbb{R}$ 来表示，满足 $u_i(a) \geq u_i(b)$ 当且仅当 $a \succsim_i b$。效用函数的值称为**收益**（payoffs）或**效用**（utilities）。
- **博弈的简化表示**：当用效用函数表示偏好关系时，博弈可以简记为 $\langle N, (A_i), (u_i) \rangle$，而不是 $\langle N, (A_i), (\succsim_i) \rangle$。

---

补充说明

- **战略博弈的核心**：战略博弈的核心是玩家在给定其他玩家行动的情况下，选择最优行动以实现自身偏好或效用最大化。
- **应用场景**：战略博弈广泛应用于经济学、政治学、生物学等领域，例如寡头竞争、拍卖机制设计、进化博弈等。
- **随机性与不确定性**：当引入随机变量时，博弈的后果具有不确定性，玩家的偏好需要基于期望效用或风险态度进行分析。
- 参与人独立做决策且所有参与人在做决策前不知道其他人的选择。

---

翻译总结
战略博弈由玩家集合、行动集合和偏好关系三部分组成。偏好关系可以直接定义在行动组合上，也可以基于行动组合的后果或随机变量的影响来定义。在许多情况下，偏好关系可以用效用函数表示，从而简化博弈的描述。这种建模方法为分析竞争、合作和不确定性下的决策提供了强大工具。

#### Nash Equilibrium

The most commonly used solution concept in game theory is that of Nash equilibrium. This notion captures a **steady state** of the play of a strategic game in which each player holds the correct expectation about the other players' behavior and acts rationally. It does not attempt to examine the process by which a steady state is reached.

[DEFINITION] A Nash equilibrium of a strategic game $\langle N,(A_i)$, $(\succsim_i)\rangle$ is a profile $a^*\in A$ of actions with the property that for every player $i\in N$ we have

$$
(a_{-i}^*,a_i^*)\succsim_i(a_{-i}^*,a_i)\text{ for all }a_i\in A_i
$$

Thus for $a^*$ to be a Nash equilibrium it must be that no player $i$ has an action yielding an outcome that he prefers to that generated when he chooses $a_i^*$, given that every other player $j$ chooses his equilibrium action $a_j^*$. Bricfly, no playcr can profitably dcviatc, givcn thc actions of the other players.

The following restatement of the defnition is sometimes useful. For any $a_{-i}\in A_{-i}$ define $B_i(a_{-i})$ to be the set of player $i$'s best actions given $a_{-i}{:}$

$$
B_i(a_{-i})=\{a_i\in A_i\colon(a_{-i},a_i)\succeq_i(a_{-i},a_i')\text{ for all }a_i'\in A_i\}
$$

We call the set-valued function $B_i$ the best-response function of player $i$. A Nash equilibrium is a profile $a^*$ of actions for which

$$
a_i^*\in B_i(a_{-i}^*)\text{ for all }i\in N
$$

This alternative formulation of the defnition points us to a (not necessarily efficient) method of fnding Nash equilibria: first calculate the best response function of each player, then find a profile $a^*$ of actions for which $a_i^*\in B_i(a_{-i}^*)$ for all $i\in N$. If the functions $B_i$ are singleton-valued then the second step entails solving $|N|$ equations in the $|N|$ unknowns $(a_i^*)_{i\in N}$.

**1. 纳什均衡的定义**

- **纳什均衡**（Nash Equilibrium）是博弈论中最常用的解概念，描述了一种**稳态**（steady state）。在这种状态下：
  - 每个玩家对其他玩家的行为有正确的预期；
  - 每个玩家在给定其他玩家行为的情况下，选择最优行动（即理性行为）。
- 纳什均衡不关注稳态是如何达到的，而是关注稳态本身的性质。

---

**2. 纳什均衡的数学定义**

对于战略博弈 $\langle N, (A_i), (\succsim_i)\rangle$，一个行动组合 $a^* \in A$ 是纳什均衡，如果对于每个玩家 $i \in N$，满足：

$$
(a_{-i}^*, a_i^*) \succsim_i (a_{-i}^*, a_i) \quad \text{对于所有 } a_i \in A_i
$$

**解释**：

- $a^* = (a_1^*, a_2^*, \ldots, a_n^*)$ 是一个行动组合，其中 $a_i^*$ 是玩家 $i$ 的均衡行动。
- $a_{-i}^*$ 表示除玩家 $i$ 外其他玩家的均衡行动组合。
- $(a_{-i}^*, a_i^*)$ 表示在均衡状态下，玩家 $i$ 选择 $a_i^*$ 时的行动组合。
- $(a_{-i}^*, a_i)$ 表示在均衡状态下，玩家 $i$ 选择其他行动 $a_i$ 时的行动组合。
- 条件 $(a_{-i}^*, a_i^*) \succsim_i (a_{-i}^*, a_i)$ 表示，给定其他玩家的均衡行动 $a_{-i}^*$，玩家 $i$ 选择 $a_i^*$ 的效用不低于选择任何其他行动 $a_i$ 的效用。

**核心思想**：

- 在纳什均衡中，没有任何玩家可以通过单方面改变自己的行动来提高自己的效用。
- 换句话说，给定其他玩家的行动，每个玩家都已经选择了最优行动。

---

**3. 最佳响应函数**

- **最佳响应函数**（Best-Response Function）：对于任意 $a_{-i} \in A_{-i}$，玩家 $i$ 的最佳行动集合定义为：

$$
B_i(a_{-i}) = \{a_i \in A_i \colon (a_{-i}, a_i) \succeq_i (a_{-i}, a_i') \text{ for all } a_i' \in A_i\}
$$

- **解释**：

  - $B_i(a_{-i})$ 表示在给定其他玩家行动 $a_{-i}$ 的情况下，玩家 $i$ 的所有最优行动。
  - 如果 $B_i(a_{-i})$ 是单值的（即只有一个最优行动），则称其为**最佳响应**。

- **纳什均衡的另一种定义**：
  一个行动组合 $a^*$ 是纳什均衡，如果对于每个玩家 $i \in N$，满足：

$$
a_i^* \in B_i(a_{-i}^*)
$$

**解释**：

- 在均衡状态下，每个玩家的行动 $a_i^*$ 都是对其他玩家均衡行动 $a_{-i}^*$ 的最佳响应。

---

**4. 寻找纳什均衡的方法**

- **步骤**：
  1. 计算每个玩家的最佳响应函数 $B_i$；
  2. 寻找一个行动组合 $a^*$，使得对于所有玩家 $i \in N$，$a_i^* \in B_i(a_{-i}^*)$。
- **特殊情况**：
  - 如果每个玩家的最佳响应函数 $B_i$ 是单值的，则纳什均衡可以通过求解 $|N|$ 个方程（每个玩家一个方程）来找到。

---

**翻译总结**
纳什均衡是博弈论中的核心概念，描述了一种稳态，其中每个玩家在给定其他玩家行动的情况下，选择最优行动。其数学定义要求每个玩家的均衡行动是对其他玩家均衡行动的最佳响应。通过最佳响应函数，可以系统地寻找纳什均衡，尤其是在最佳响应函数为单值的情况下，纳什均衡可以通过求解方程组得到。这一概念为分析竞争与合作中的稳定状态提供了重要工具。

#### 2.4 Existence of a Nash Equilibrium

Not every strategic game has a Nash equilibrium, as the game Matching Pennies (Figure 17.3) shows. The conditions under which the set of Nash equilibria of a game is nonempty have been investigated extensively. We now present an existence result that is one of the simplest of the genre. (Nevertheless its mathematical level is more advanced than most of the rest of the book, which does not depend on the details.)

An existence result has two purposes. First, if we have a game that satisfies the hypothesis of the result then we know that there is some hope that our efforts to find an equilibrium will meet with success. Second, and more important, the existence of an equilibrium shows that the game is consistent with a steady state solution. Further, the existence of equilibria for a family of games allows us to study properties of these equilibria (by using, for example, “comparative static” techniques) without finding them explicitly and without taking the risk that we are studying the empty set.

To show that a game has a Nash equilibrium it suffices to show that there is a profile $a^* \in B_i(a_{-i}^*)$ for all $i \in N$ (see (15.2)). Define the set-valued function $B: A \rightarrow A$ by $B(a) = \times_{i \in N} B_i(a_{-i})$. Then (15.2) can be written in vector form simply as $a^* \in B(a^*)$. Fixed point theorems give conditions on $B$ under which indeed exists a value of $a^*$ for which $a^* \in B(a^*)$. The fixed point theorem that we use is the following (due to Kakutani (1941)).

[LEMMA] (Kakutani's fixed point theorem): Let $X$ be a compact convex subset of $\mathbb{R}^n$ and let $f:X \to X$ be a set-valued function for which

- for all $x \in X$, the set $f(x)$ is nonempty and convex
- the graph of $f$ is closed (i.e., for all sequences $\{x_n\}$ and $\{y_n\}$ such that $y_n \in f(x_n)$ for all $n$, $x_n \to x$, and $y_n \to y$, we have $y \in f(x)$). Then there exists $x^* \in X$ such that $x^* \in f(x^*)$.

Define a preference relation $\succsim_i$ over $A$ to be quasi-concave on $A_i$ if
for every $a^*\in A$ the set $\{a_i\in A_i{:}(a_{-i}^*,a_i)\succsim_ia^*\}$ is convex.

[PROPOSITION 20.3] The strategic game $\langle N, (A_i), (\succsim_i) \rangle$ has a Nash equilibrium if for all $i \in N$

- the set $A_i$ of actions of player $i$ is a nonempty compact convex subset of a Euclidean space
  and the preference relation $\succsim_i$ is
- continuous
- quasi-concave on $A_i$.

[Proof]. Define $B{:}A\to A$ by $B(a)=\times_i\in NB_i(a_{-i})$ (where $B_i$ is the best response function of player $i$, defined in (15.1)). For every $i\in N$ the set $B_{i}(a_{-i})$ is nonempty since $\succsim_i$ is continuous and $A_i$ is compact, and is convex since $\succsim_i$ is quasi-concave on $A_i;B$ has a closed graph since each $\succsim_i$ is continuous. Thus by Kakutani's theorem $B$ has a fixed point; as we have noted any fixed point is a Nash equilibrium of the game. $\square$

Note that this result asserts that a strategic game satisfying certain conditions has aleast one Nash equilibrium; as we have seen, a game can have more than one equilibrium. (Results that we do not discuss identify conditions under which a game has a unique Nash equilibrium.) Note also that Proposition 20.3 does not apply to any game in which some player has finitely many actions, since such a game violates the condition that the set of actions of every player be convex.

[EXERCISE]: Show that each of the following four conditions is necessary for Kakutani's theorem. $(i)X$ is compact. $(ii)X$ is convex. $( iii)$ $f( x)$ is convex for each $x\in X$. $(iv)$ $f$ has a closed graph.

[EXERCISE] Symmetric games: Consider a two-person strategic game that satisfes the conditions of Proposition 20.3. Let $N=\{1,2\}$ and assume that the game is $symmetric:A_1=A_2$ and $(a_1,a_2)\succsim_1(b_1,b_2)$ if and only if $(a_2,a_1)\succsim_2(b_2,b_1)$ for all $a\in A$ and $b\in A$. Use Kakutani's theorem to prove that there is an action $a_1^*\in A_1$ such that $(a_1^*,a_1^*)$ is a Nash equilibrium of the game. (Such an equilibrium is called a symmetric equilibrium. ) Give an example of a finite symmetric game that has only asymmetric equilibria.

**1. 纳什均衡的存在性**

- **背景**：并非所有战略博弈都存在纳什均衡（例如“匹配硬币”博弈）。研究纳什均衡存在的条件具有重要意义：
  1. **实践意义**：若博弈满足存在性条件，则寻找均衡是可行的；
  2. **理论意义**：均衡存在性表明博弈存在稳态解，且可通过比较静态分析研究均衡性质，无需显式求解。
- **存在性证明方法**：  
  通过定义集合值函数 $B(a) = \times_{i \in N} B_i(a_{-i})$（$B_i$ 为玩家 $i$ 的最佳响应函数），纳什均衡即为 $B$ 的固定点（即 $a^* \in B(a^*)$）。利用 **角谷不动点定理**（Kakutani’s fixed point theorem）可证明固定点的存在性。

---

**2. 角谷不动点定理**

[**定理**]（角谷不动点定理）：  
设 $X$ 是 $\mathbb{R}^n$ 中的 **紧凸集**，集合值函数 $f: X \to X$ 满足：

1. **非空凸值**：对任意 $x \in X$，$f(x)$ 非空且凸；
2. **闭图像**：若序列 $\{x_n\} \to x$ 和 $\{y_n\} \to y$ 满足 $y_n \in f(x_n)$，则 $y \in f(x)$。

则存在 $x^* \in X$ 使得 $x^* \in f(x^*)$。

---

**3. 纳什均衡存在性定理**

[**命题 20.3**]：战略博弈 $\langle N, (A_i), (\succsim_i) \rangle$ 存在纳什均衡，若对每个玩家 $i \in N$：

1. **行动集条件**：$A_i$ 是欧几里得空间的非空紧凸集；
2. **偏好关系条件**：$\succsim_i$ 是连续的，且在 $A_i$ 上是 **拟凹的**（quasi-concave）。

**证明思路**：

- 定义集合值函数 $B(a) = \times_{i \in N} B_i(a_{-i})$，其中 $B_i(a_{-i})$ 是玩家 $i$ 的最佳响应集合。
- 由于 $\succsim_i$ 的连续性和拟凹性，$B_i(a_{-i})$ 非空且凸；闭图像条件由连续性保证。
- 应用角谷定理，$B$ 存在固定点 $a^*$，即纳什均衡。

**注意**：

- 命题仅保证至少存在一个均衡，但博弈可能有多个均衡。
- 若某些玩家的行动集非凸（如有限集），命题不适用。

---

**练习题解答**

---

**练习题 1**

**要求**：证明角谷不动点定理的四个条件均为必要。

**分析**：需构造反例，说明缺少任一条件时定理可能不成立。

1. **条件 $(i)$：$X$ 是紧集**
   - **反例**：设 $X = \mathbb{R}$（非紧），定义 $f(x) = x + 1$。显然 $f$ 无固定点，但满足其他条件。
   - **结论**：紧性不可缺失。
2. **条件 $(ii)$：$X$ 是凸集**
   - **反例**：设 $X = S^1$（单位圆周，非凸），定义 $f(x)$ 为将 $x$ 绕原点旋转固定角度。$f$ 无固定点，但满足其他条件。
   - **结论**：凸性不可缺失。
3. **条件 $(iii)$：$f(x)$ 是凸集**
   - **反例**：设 $X = [0,1]$，定义 $f(x) = \{0,1\}$（非凸值）。显然 $f$ 无固定点，但满足其他条件。
   - **结论**：凸值性不可缺失。
4. **条件 $(iv)$：$f$ 有闭图像**
   - **反例**：设 $X = [0,1]$，定义
     $$
     f(x) =
     \begin{cases}
         \{1\} & \text{if } x < 0.5, \\
         \{0\} & \text{if } x \geq 0.5.
     \end{cases}
     $$
     当 $x_n \to 0.5^-$ 时 $y_n = 1 \to 1 \notin f(0.5) = \{0\}$，闭图像条件不满足。此时 $f$ 无固定点。
   - **结论**：闭图像条件不可缺失。

---

**练习题 2**

**要求**：对称博弈中证明存在对称纳什均衡，并构造有限对称博弈仅有非对称均衡的例子。

**解答**：

1. **存在对称均衡的证明**：

   - **对称博弈定义**：$N = \{1,2\}$，$A_1 = A_2$，且偏好关系满足 $(a_1, a_2) \succsim_1 (b_1, b_2) \iff (a_2, a_1) \succsim_2 (b_2, b_1)$。
   - **构造映射**：定义对称策略空间 $A_1 = A_2 = S$，令 $B: S \times S \to S \times S$ 为最佳响应函数。由对称性，若 $(a_1^*, a_2^*)$ 是均衡，则 $(a_2^*, a_1^*)$ 也是均衡。
   - **应用角谷定理**：考虑对称策略子空间 $S_{\text{sym}} = \{(a, a) | a \in S\}$，在此子空间上定义 $f(a) = B_1(a) \cap B_2(a)$。由命题 20.3 条件，$f$ 存在固定点 $(a^*, a^*)$，即对称均衡。

2. **有限对称博弈仅有非对称均衡的例子**：
   - **博弈矩阵**：
     $$
     \begin{array}{c|cc}
         & C & D \\
         \hline
         C & (0,0) & (2,1) \\
         D & (1,2) & (0,0) \\
     \end{array}
     $$
     - **对称性**：$A_1 = A_2 = \{C, D\}$，且 $(C, D) \succsim_1 (D, C) \iff (D, C) \succsim_2 (C, D)$。
     - **均衡分析**：
       - **纯策略均衡**：$(C, D)$ 和 $(D, C)$ 是仅有的纳什均衡，均为非对称。
       - **对称混合策略均衡**：若存在，需满足 $p = q$，但计算显示无解。
     - **结论**：该博弈是有限对称的，但仅有非对称均衡。

#### 2.5 Strictly Competitive Games 21

We can say little about the set of Nash equilibria of an arbitrary strategic game; only in limited classes of games can we say something about the qualitative character of the equilibria. One such class of games is that in which there are two players, whose preferences are diametrically opposed. We assume for convenience in this section that the names of the players are “1” and “2”(i.e. $N=\{1,2\})$.

[DEFINITION 21.1] A strategic game $\langle\{1,2\},(A_i),(\succsim_i)\rangle$ is strictly competitive if for any $a\in A$ and $b\in A$ we have $a\succsim_1b$ if and only if $b\succsim_2a$.

A strictly competitive game is sometimes called zerosum because if player 1's preference relation $\succsim_1$ is represented by the payoff function $u_1$ then player $2’$s preference relation is represented by $u_2$ with $u_1+u_2=0$.

We say that player $i$ maxminimizes if he chooses an action that is best for him on the assumption that whatever he does, player $j$ will choose her action to hurt him as much as possible. We now show that for a strictly competitive game that possesses a Nash equilibrium, a pair of actions is a Nash equilibrium if and only if the action of each player is a maxminimizer. This result is striking because it provides a link between individual decision-making and the reasoning behind the notion of Nash equilibrium. In establishing the result we also prove the strong result that for strictly competitive games that possess Nash equilibria all equilibria yield the same payoffs. This property of Nash equilibria is rarely satisfied in games that are not strictly competitive.

DEFINITION 21.2 Let $\langle\{1,2\},(A_i),(u_i)\rangle$ be a strictly competitive strategic game. The action $x^*\in A_1$ is a maxminimizer for player 1 if

$$
\min_{y\in A_2}u_1(x^*,y)\geq\min_{y\in A_2}u_1(x,y)\text{ for all }x\in A_1.
$$

Similarly, the action $y^*\in A_2$ is a maxminimizer for player 2 if

$$
\min_{x\in A_1}u_2(x,y^*)\geq\min_{x\in A_1}u_2(x,y)\text{ for all }y\in A_2.
$$

In words, a maxminimizer for player $i$ is an action that maximizes the payoff that player $i$ can $guarantee$. A maxminimizer for player 1 solves the problem $\max_x\min_yu_1(x,y)$ and a maxminimizer for player 2 solves the problem $\max_y\min_xu_2(x,y)$.

In the sequel we assume for convenience that player 1's preference relation is represented by a payoff function $u_1$ and, without loss of generality, that $u_2=-u_1$. The following result shows that the maxminimization of player $2’$s payoff is equivalent to the minmaximization of player 1's payoff.

[LEMMA 22.1] Let $\langle \{ 1, 2\} , ( A_i) , ( u_i) \rangle$ be a strictly competitive strategic game. Then

$$
\max _{y\in A_2}\min _{x\in A_1}u_2( x, y) = - \min _{y\in A_2}\max _{x\in A_1}u_1( x, y)
$$

Further, $y \in A_2$ solves the problem $\max _{y\in A_2}$ $\min _{x\in A_1}$ $u_2( x, y)$ if and only if it solves the problem $\min _{y\in A_2}\max _{x\in A_1}u_1( x, y) $.

[Proof]. For any function $f$ we have min$_z(-f(z))=-\max_zf(z)$ and $\arg \min _{z}( - f( z) )$ = $\arg \max _{z}f( z) $. It follows that for every $y$ $\in$ $A_2$ we have$-\min_x\in A_1u_2(x,y)=\max_{x\in A_1}(-u_2(x,y))=\max_{x\in A_1}u_1(x,y)$. $\begin{array}{rcl}\text{Hence }\max_{y\in A_2}\min_{x\in A_1}u_2(x,y)&=&-\min_{y\in A_2}[-\min_{x\in A_1}u_2(x,y)]\end{array}=$ $-\min_{y\in A_2}\max_{x\in A_1}u_1(x,y);$ in addition $y\in A_2$ is a solution of the $\mathrm{problem~max}_{y\in A_2}\min_{x\in A_1}u_2(x,y)$ if and only if it is a solution of the $\mathop{\text{problem }}\operatorname*{min}_{y\in A_2}\operatorname*{max}_{x\in A_1}u_1(x,y)$.

The following result gives the connection between the Nash equilibria of a strictly competitive game and the set of pairs of maxminimizers.

[PROPOSITION 22.2] $Let G= \langle \{ 1, 2\} , ( A_i) , ( u_i) \rangle be a strictly competitive$
strategic $game.$

1. If $( x^* , y^*)$ is a Nash equilibrium of G then $x^*$ is a maxminimizer
   for player 1 and $y*$ is a maxminimizer for player 2.
1. If $(x^*,y^*)$ is a Nash equilibrium of G then $max_x\min_yu_1(x,y)=\min_y\max_xu_1(x,y)=u_1(x^*,y^*)$ and thus all Nash equilibria of G. yield the same payoff.
1. If $\max_x\min_yu_1(x,y)=\min_y\max_xu_1(x,y)$ and thus, in particular, if $G$ has $a$ Nash equilibrium(see part b), $x^*$ is a maxminimizer for player 1, and $y^{* }$ is a maxminimizer for player 2, then $( x^{* }, y^{* })$ is a Nash equilibrium of $G$.

[Proof]. We frst prove parts (a) and (b). Let $(x^{*},y^{*})$ be a Nash equilibrium of $G.$ Then $u_2(x^*,y^*)\geq u_2(x^*,y)$ for all $y\in A_2$ or, since $u_2= - u_1$, $u_1( x^* , y^* )$ $\leq$ $u_1( x^* , y)$ for all $y\in A_2.$ Hence $u_1( x^* , y^* )$ = $\min _yu_1( x^* , y)$ $\leq \max _x\min _yu_1( x, y) .$ Similarly, $u_1( x^* , y^* )$ $\geq u_1( x, y^* )$

#### 2.6 Bayesian Games: Strategic Games with Imperfect Information 24

We frequently wish to model situations in which some of the parties are not certain of the characteristics of some of the other parties. The model of a Bayesian game, which is closely related to that of a strategic game, is designed for this purpose.

As for a strategic game, two primitives of a Bayesian game are a set $N$ of players and a profile $(A_i)$ of sets of actions. We model the players' uncertainty about each other by introducing a set $\Omega$ of possible “states of nature”, each of which is a description of all the players relevant characteristics. For convenience we assume that $\Omega$ is finite Each player $i$ has a prior belief about the state of nature given by a probability measure $p_i$ on $\Omega.$ In any given play of the game some state of nature $\omega\in\Omega$ is realized. We model the players' information about the state of nature by introducing a profile $(\tau_i)$ of signal functions, $\tau _i( \omega )$ being the signal that player $i$ observes, before choosing his action, when the state of nature is $\omega.$ Let $T_i$ be the set of all possible values of $\tau_i;$we refer to $T_i$ as the set of types of player $i.$ We assume that $p_i(\tau_i^{-1}(t_i))>0$ for all $t_i\in T_i$ (player $i$ assigns positive prior probability to every member of$T_i).$ If player $i$ receives the signal $t_i\in T_i$ then he deduces that the state is in the set $\tau_i^{-1}(t_i);$ his posterior belief about the state that has been realized assigns to each state $\omega\in\Omega$ the probability $p_i(\omega)/p_i(\tau_i^{-1}(t_i))$ if $\omega\in\tau_i^{-1}(t_i)$ and the probability zero otherwise (i.e. the probability of $\omega$ conditional on $\tau_i^{-1}(t_i)).$ As an example, if $\tau_i(\omega)=\omega$ for all $\omega\in\Omega$ then player $i$ has full information about the state of nature. Alternatively, if $\Omega=\times_{i\in N}T_{i}$ and for each player $i$ the probability measure $p_i$ is a product measure on $\Omega$ and $\tau_i(\omega)=\omega_i$ then the players' signals are independent and player $i$ does not learn from his signal anything about the other players’ information.

As in a strategic game, each player cares about the action profile; in addition he may care about the state of nature. Now, even if he knows the action taken by every other player in every state of nature, a player may be uncertain about the pair $(a,\omega)$ that will be realized given any action that he takes, since he has imperfect information about the state of nature. Therefore we include in the model a profile $(\succsim_i)$ of preference relations over lotteries on $A\times\Omega$ (where, as before, $A=\times_j\in NA_j$). To summarize, we make the following definition.

[DEFINITION 25.1] A Bayesian game consists of

- a finite set $N$ (the set of players)
- a finite set $\Omega$ (the set of states)

and for each player $i \in N$

- a set $A_i$ (the set of actions available to player $i$)
- a finite set $T_i$ (the set of signals that may be observed by player $i$) and a function $\tau_i: \Omega \rightarrow T_i$ (the signal function) of player $i$)
- a probability measure $p_i$ on $\Omega$ (the prior belief) of player $i$) for which $p_i(\tau_i^{-1}(t_i)) > 0$ for all $t_i \in T_i$
- a preference relation $\sim_i$ on the set of probability measures over $A \times \Omega$ (the preference relation) of player $i$), where $A = \times_{j \in N} A_j$.

Note that this defnition allows the players to have different prior beliefs. These beliefs may be related; commonly they are identical, coincident with an“objective”measure. Frequently the model is used in situations in which a state of nature is a profile of parameters of the players' preferences (for example, profiles of their valuations of an object). However, the model is much more general; in Section 2.6.3 we consider its use to capture situations in which each player is uncertain about what the others know.

Note also that sometimes a Bayesian game is described not in terms of an underlying state space $\Omega$, but as a“reduced form" in which the basic primitive that relates to the players' information is the profile of
the sets of possible types.

We now turn to a definition of equilibrium for a Bayesian game. In any given play of a game each player knows his type and does not need to plan what to do in the hypothetical event that he is of some other type. Consequently, one might think that an equilibrium should be defined for each state of nature in isolation. However, in any given state a player who wishes to determine his best action may need to hold a belief about what the other players would do in other states, since he may be imperfectly informed about the state. Further, the formation of such a belief may depend on the action that the player himself would choose in other states, since the other players may also be imperfectly informed.

Thus we are led to define a Nash equilibrium of a Bayesian game $\langle N,\Omega$, $(A_i),(T_i),(\tau_i),(p_i),(\succsim_i)\rangle$ to be a Nash equilibrium of the strategic game $G^*$ in which for each $i\in N$ and each possible signal $t_i\in T_i$ there is a player, whom we refer to as $(i,t_i)$ (“type $t_i$ of player $i$”). The set of actions of each such player $(i,t_i)$ is $A_i;$ thus the set of action profiles in $G^*$ is $\times_{j\in N}(\times_{t_j\in T_j}A_j).$ The preferences of each player $(i,t_i)$ are defined as follows. The posterior belief of player $i$,together with an action profile $a^*$ $\operatorname{in}G^*$, generates a lottery $L_i(a^*,t_i)$ over $A\times\Omega{:\text{ the probability assigned}}$ by $L_i(a^*,t_i)$ to $((a^*(j,\tau_j(\omega)))_{j\in N},\omega)$ is player $i$'s posterior belief that the state is $\omega$ when he receives the signal $t_i$ $(a^*(j,\tau_j(\omega))$ being the action of player $(j,\tau_j(\omega))$ in the profile $a^*).$ Player $(i,t_i)$ in $G^*$ prefers the action profile $a^*$ to the action profile $b^*$ if and only if player $i$ in the Bayesian game prefers the lottery $L_i(a^*,t_i)$ to the lottery $L_i(b^*,t_i).$ To summarize, we have the following.

[DEFINITION 26.1] A Nash equilibrium of a Bayesian game $\langle N,$ $\Omega, (A_{i}), (T_{i}), (\tau_{i}), (p_{i}), (\succsim_{i})\rangle$ is a Nash equilibrium of the strategic game defined as follows.

- The set of players is the set of all pairs $(i,t_{i})$ for $i \in N$ and $t_{i} \in T_{i}$.
- The set of actions of each player $(i,t_{i})$ is $A_{i}$.
- The preference ordering $\succsim^{*}_{(i,t_{i})}$ of each player $(i,t_{i})$ is defined by

$a^{*} \succsim^{*}_{(i,t_{i})} b^{*}$ if and only if $L_{i}(a^{*}, t_{i}) \succsim_{i} L_{i}(b^{*}, t_{i})$,
where $L_{i}(a^{*}, t_{i})$ is the lottery over $A \times \Omega$ that assigns probability $p_{i}(\omega)/p_{i}(\tau^{-1}_{i}(t_{i}))$ to $((a^{*}(j,\tau_{j}(\omega)))_{j \in N}, \omega)$ if $\omega \in \tau^{-1}_{i}(t_{i})$, zero otherwise.

In brief, in a Nash equilibrium of a Bayesian game each player chooses the best action available to him given the signal that he receives and his belief about the state and the other players' actions that he deduces from this signal. Note that to determine whether an action profile is a Nash equilibrium of a Bayesian game we need to know only how each player in the Bayesian game compares lotteries over $A\times\Omega$ in which the distribution over $\Omega$ is the same: a player never needs to compare lotteries in which this distribution is different. Thus from the point of view of Nash equilibrium the specification of the players' preferences in a Bayesian game contains more information than is necessary. (This redundancy) has an analog in a strategic game: to define a Nash equilibrium of a strategic game we need to know only how any player $i$ compares any outcome $(a_-i,a_i)$ with any other outcome $(a_-i,b_i)$.

**1. 贝叶斯博弈的背景**

- **目的**：贝叶斯博弈（Bayesian Game）用于建模某些参与人对其他参与人特征不确定的情况。它与战略博弈密切相关，但增加了对信息不对称的处理。
- **核心要素**：
  - **参与人集合** $N$ 和**行动集合** $(A_i)$；
  - **状态空间** $\Omega$：描述所有参与人相关特征的集合；
  - **先验信念** $p_i$：参与人 $i$ 对状态空间 $\Omega$ 的概率分布；
  - **信号函数** $\tau_i$：参与人 $i$ 在状态 $\omega$ 下观察到的信号；
  - **类型集合** $T_i$：参与人 $i$ 可能观察到的信号值集合。

---

**2. 贝叶斯博弈的定义**

[**定义 25.1**] 贝叶斯博弈由以下要素组成：

1. **参与人集合**：有限的集合 $N$；
2. **状态空间**：有限的集合 $\Omega$；
3. **每个参与人 $i \in N$ 的要素**：
   - **行动集合** $A_i$；
   - **信号集合** $T_i$ 和信号函数 $\tau_i: \Omega \rightarrow T_i$；
   - **先验信念** $p_i$：满足 $p_i(\tau_i^{-1}(t_i)) > 0$ 对所有 $t_i \in T_i$；
   - **偏好关系** $\succsim_i$：定义在 $A \times \Omega$ 上的概率分布的偏好关系。

**注意**：

- 参与人可能有不同的先验信念，通常假设这些信念是相同的或基于“客观”概率。
- 状态空间 $\Omega$ 可以表示参与人偏好的参数（例如对某物品的估值），但模型更具一般性。

---

**3. 贝叶斯博弈的均衡**

- **均衡定义**：贝叶斯博弈的纳什均衡是战略博弈 $G^*$ 的纳什均衡，其中：
  - **参与人**：所有 $(i, t_i)$（参与人 $i$ 的类型 $t_i$）；
  - **行动集合**：每个 $(i, t_i)$ 的行动集合为 $A_i$；
  - **偏好关系**：$(i, t_i)$ 的偏好关系由参与人 $i$ 的后验信念和行动组合生成的彩票决定。

[**定义 26.1**] 贝叶斯博弈的纳什均衡是战略博弈的纳什均衡，其中：

1. **参与人集合**：所有 $(i, t_i)$（$i \in N$，$t_i \in T_i$）；
2. **行动集合**：每个 $(i, t_i)$ 的行动集合为 $A_i$；
3. **偏好关系**：$(i, t_i)$ 的偏好关系定义为：
   $$
   a^* \succsim^*_{(i,t_i)} b^* \iff L_i(a^*, t_i) \succsim_i L_i(b^*, t_i),
   $$
   其中 $L_i(a^*, t_i)$ 是由后验信念和行动组合生成的彩票。

**核心思想**：

- 在贝叶斯博弈的纳什均衡中，每个参与人根据其接收到的信号和由此推断的状态及其他参与人行动，选择最优行动。
- 确定均衡时，只需知道参与人如何比较在相同状态分布下的彩票。

贝叶斯博弈用于建模信息不对称的情况，其核心要素包括参与人集合、状态空间、信号函数和先验信念。贝叶斯博弈的纳什均衡是战略博弈的纳什均衡，其中每个参与人根据其类型（信号）选择最优行动。均衡的定义依赖于参与人的后验信念和行动组合生成的彩票。这一模型为分析不完全信息下的策略互动提供了重要工具。

简而言之，在贝叶斯博弈的纳什均衡中，每个参与人会根据其接收到的信号以及从该信号中推断出的状态和其他参与人的行动，选择对自己最有利的行动。需要注意的是，判断一个行动组合是否是贝叶斯博弈的纳什均衡，我们只需知道每个参与人如何比较在 $\Omega$ 上分布相同的 $A \times \Omega$ 上的彩票：参与人无需比较在 $\Omega$ 上分布不同的彩票。因此，从纳什均衡的角度来看，贝叶斯博弈中对参与人偏好的描述包含了一些不必要的信息。（这种冗余性）在战略博弈中也有类似的情况：定义战略博弈的纳什均衡时，我们只需知道任何参与人 $i$ 如何比较结果 $(a_{-i}, a_i)$ 与结果 $(a_{-i}, b_i)$。

**补充说明**

- **核心思想**：贝叶斯博弈的纳什均衡关注的是参与人在给定信号和信念下的最优行动选择，而不需要比较不同状态分布下的彩票。
- **冗余性**：贝叶斯博弈中对偏好的描述可能包含超出纳什均衡需求的信息，这与战略博弈中仅需比较部分结果的情况类似。

Comments on the Model of a Bayesian Game

The idea that a situation in which the players are unsure about each other’s characteristics can be modeled as a Bayesian game, in which the players’ uncertainty is captured by a probability measure over some set of “states”, is due to Harsanyi (1967/68). Harsanyi assumes that the prior belief of every player is the same, arguing that all differences in the players’ knowledge should be derived from an objective mechanism that assigns information to each player, not from differences in the players' initial beliefs. In Section 5.3 we show that the assumption of a common prior belief has strong implications for the relationship between the players’ posterior beliefs. (For example, after a pair of players receive their signals it cannot be “common knowledge” between them that player 1 believes the probability that the state of nature is in some given set to

be $\alpha$ and that player 2 believes this probability to be $\beta\neq\alpha$, though it $is$ possible that player 1 believes the probability to be $\alpha$, player 2 believes it to be $\beta$, and one of them is unsure about the other's belief.)
A Bayesian game can be used to model not only situations in which each player is uncertain about the other players' payoffs, as in Example 27.1,but also situations in which each player is uncertain about the other players' $knowledge.$

Consider, for example, a Bayesian game in which the set of players is $N=\{1,2\}$, the set of states is $\Omega=\{\omega_1,\omega_2,\omega_3\}$, the prior belief of each player assigns probability $\frac13$ to each state, the signal functions are defined by $\tau_1(\omega_1)=\tau_1(\omega_2)=t_1^{\prime},\tau_1(\omega_3)=t_1^{\prime\prime}$, and $\tau_2(\omega_1)=t_2^{\prime}$, $\tau_2(\omega_2)=\tau_2(\omega_3)=t_2^{\prime\prime}$, and player 1's preferences satisfy $(b,\omega_j)\succ_1(c,\omega_j)$ for $j=1,2$ and $(c,\omega_3)\succ_1(b,\omega_3)$ for some action profiles $b$ and $c$,while player 2 is indifferent between all pairs $(a,\omega).$ In state $\omega_1$ in such a game player 2 knows that player 1 prefers $b$ to $c$,while in state $\omega_2$ he does not know whether player 1 prefers $b$ to $c$ or $c$ to $b.$ Since in state $\omega_1$ player 1 does not know whether the state is $\omega_1$ or $\omega_2$,she does not know in this case whether $(i)$ player 2 knows that she prefers $b$ to $c$,or $(ii)$ player 2 is not sure whether she prefers $b$ to $c$ or $c$ to $b.$
Can every situation in which the players are uncertain about each other's knowledge be modeled as a Bayesian game? Assume that the players' payoffs depend only on a parameter $\theta\in\Theta.$ Denote the set of possible beliefs of each player $i$ by $X_i.$ Then a belief of any player $j$ is a probability distribution over $\Theta\times X_-j.$ That is, the set of beliefs of any player has to be defined in terms of the sets of beliefs of all the other players. Thus the answer to the question we posed is not trivial and is equivalent to the question of whether we can find a collection $\{X_j\}_{j\in N}$ of sets with the property that for all $i\in N$ the set $X_i$ is isomorphic to the set of probability distributions over $\Theta\times X_-i.$ If so, we can let $\Omega=\Theta\times(\times_{i\in N}X_i)$ be the state space and use the model of a Bayesian game to capture any situation in which players are uncertain not only about each other's payoffs but also about each other's beliefs. A positive answer is given to the question by Mertens and Zamir (1985); we omit the argument.

**翻译**

**关于贝叶斯博弈模型的评论**

将玩家对彼此特征不确定的情境建模为贝叶斯博弈的想法源于 Harsanyi（1967/68）。在这种模型中，玩家的不确定性通过某个“状态”集合上的概率测度来刻画。Harsanyi 假设所有玩家的先验信念是相同的，并认为玩家之间知识的差异应源于某种客观机制分配信息给每个玩家，而不是源于玩家初始信念的差异。在第 5.3 节中，我们将展示共同先验信念假设对玩家后验信念之间关系的强约束性。（例如，当一对玩家接收到他们的信号后，玩家 1 认为自然状态属于某个给定集合的概率为$\alpha$，而玩家 2 认为该概率为$\beta\neq\alpha$，这种情况不能成为他们之间的“共同知识”。不过，玩家 1 认为概率为$\alpha$，玩家 2 认为概率为$\beta$，且其中一方不确定另一方的信念，这种情况是可能的。）

贝叶斯博弈不仅可以用于建模每个玩家对其他玩家收益不确定的情境（如例 27.1），还可以用于建模每个玩家对其他玩家**知识**不确定的情境。

例如，考虑一个贝叶斯博弈，其中玩家集合为$N=\{1,2\}$，状态集合为$\Omega=\{\omega_1,\omega_2,\omega_3\}$，每个玩家的先验信念为每个状态分配概率$\frac13$，信号函数定义为$\tau_1(\omega_1)=\tau_1(\omega_2)=t_1^{\prime}$，$\tau_1(\omega_3)=t_1^{\prime\prime}$，以及$\tau_2(\omega_1)=t_2^{\prime}$，$\tau_2(\omega_2)=\tau_2(\omega_3)=t_2^{\prime\prime}$。玩家 1 的偏好满足对某些行动组合$b$和$c$，$(b,\omega_j)\succ_1(c,\omega_j)$（$j=1,2$）且$(c,\omega_3)\succ_1(b,\omega_3)$，而玩家 2 对所有$(a,\omega)$无差异。在这种博弈的状态$\omega_1$中，玩家 2 知道玩家 1 偏好$b$胜过$c$；而在状态$\omega_2$中，玩家 2 不知道玩家 1 是偏好$b$胜过$c$还是相反。由于在状态$\omega_1$中，玩家 1 不知道状态是$\omega_1$还是$\omega_2$，因此她在这种情况下不知道：$(i)$ 玩家 2 是否知道她偏好$b$胜过$c$，或者$(ii)$ 玩家 2 是否不确定她偏好$b$胜过$c$还是相反。

是否所有玩家对彼此知识不确定的情境都可以建模为贝叶斯博弈？假设玩家的收益仅依赖于某个参数$\theta\in\Theta$。用$X_i$表示每个玩家$i$的可能信念集合。那么，任何玩家$j$的信念是$\Theta\times X_{-j}$上的概率分布。也就是说，任何玩家的信念集合必须根据所有其他玩家的信念集合来定义。因此，我们提出的问题的答案并不简单，它等价于是否能找到一组集合$\{X_j\}_{j\in N}$，使得对于所有$i\in N$，$X_i$与$\Theta\times X_{-i}$上的概率分布集合同构。如果可以，我们可以令$\Omega=\Theta\times(\times_{i\in N}X_i)$为状态空间，并使用贝叶斯博弈模型来刻画玩家不仅对其他玩家收益不确定，还对其信念不确定的任何情境。Mertens 和 Zamir（1985）给出了肯定的答案；我们在此省略其论证。

---

**补充说明**

- **核心思想**：贝叶斯博弈模型通过状态空间和概率测度刻画玩家之间的不确定性，适用于玩家对彼此收益或知识不确定的情境。
- **共同先验信念**：Harsanyi 的假设简化了模型，但对玩家后验信念的关系有强约束。
- **知识不确定性**：贝叶斯博弈可以进一步扩展，用于建模玩家对其他玩家信念的不确定性，但这需要复杂的数学构造。

### 3 Mixed, Correlated, and Evolutionary Equilibrium 31

- 3.1 Mixed Strategy Nash Equilibrium 31
- 3.2 Interpretations of Mixed Strategy Nash Equilibrium 37
- 3.3 Correlated Equilibrium 44
- 3.4 Evolutionary Equilibrium 48
- Notes 51

### 4 Rationalizability and Iterated Elimination of Dominated Actions 53

- 4.1 Rationalizability 53
- 4.2 Iterated Elimination of Strictly Dominated Actions 58
- 4.3 Iterated Elimination of Weakly Dominated Actions 62
- Notes 64

### 5 Knowledge and Equilibrium 67

- 5.1 A Model of Knowledge 67
- 5.2 Common Knowledge 73
- 5.3 Can People Agree to Disagree? 75
- 5.4 Knowledge and Solution Concepts 76
- 5.5 The Electronic Mail Game 81
- Notes 84

### IV Coalitional Games 9

The primitives of the models we study in Parts I, II, and III (often referred to as“noncooperative” games) are the players' sets of possible actions and their preferences over the possible outcomes, where an outcome is a profile of actions; each action is taken by a single playen autonomously. In this part we study the model of a coalitional game. One primitive of this model is the collection of sets of joint actions that each group of players (coalition) can take independently of the remaining players. An outcome of a coalitional game is a specification of the coalition that forms and the joint action it takes. (More general models, in which many coalitions may form simultaneously, are discussed in the literature.) The other primitive of the model of a coalitional game is the profile of the players' preferences over the set of all possible outcomes. Thus although actions are taken by coalitions, the theory is based (as are the theories in the other parts of the book) on the individuals preferences.

A solution concept for coalitional games assigns to each game a set of outcomes. As before, each solution concept we study captures the consequences of a natural line of reasoning for the participants in a game; it defines a set of arrangements that are stable in some sense. In general the stability requirement is that the outcome be immune to deviations of a certain sort by groups of players; by contrast, most (though not all) solutions for noncooperative games require immunity to deviations by individual players. Many variants of the solution concepts we study are analyzed in the literature; we consider a sample designed to illustrate the main ideas.

A coalitional model is distinguished from a noncooperative model primarily by its focus on what groups of players can achieve rather than on what individual players can do and by the fact that it does not consider the details of how groups of players function internally. If we wish to model the possibility of coalition formation in a noncooperative game then we must specify how coalitions form and how their members choose joint actions. These details are absent from a coalitional game, so that the outcome of such a game does not depend on them.

To illustrate the differences between the two modeling approaches, consider the following situation. Each of a group of individuals owns a bundle of inputs and has access to a technology for producing a valuable single output. Each individual's inputs are unproductive in his own technology but productive in some other individual's technology. A noncooperative model of this situation specifies precisely the set of actions that is available to each individual: perhaps each individual can announce a price vector at which he is willing to trade inputs, or perhaps he can propose a distribution of inputs for the whole of the society. A coalitional model, by contrast, starts from the sets of payoff vectors that each group of individuals can jointly achieve. A coalition may use contracts, threats, or promises to achieve a high level of production; these institutions are not modeled explicitly in a coalitional game.

We do not view either of the two approaches as superior or more basic. Each of them reflects different kinds of strategic considerations and contributes to our understanding of strategic reasoning. The study of the interconnections between noncooperative and cooperative models can also be illuminating.

在第一、二、三部分中，我们研究的模型（通常称为“非合作”博弈）的基本要素是玩家的可能行动集及其对可能结果的偏好，其中结果是行动的剖面；每个行动由单个玩家自主采取。在这一部分，我们将研究联盟博弈的模型。该模型的一个基本要素是每个玩家群体（联盟）可以独立于其他玩家采取的共同行动集。联盟博弈的结果是形成的联盟及其采取的共同行动的说明。（更一般的模型，其中多个联盟可能同时形成，在文献中有讨论。）联盟博弈模型的另一个基本要素是玩家对所有可能结果集的偏好剖面。因此，尽管行动是由联盟采取的，但该理论（如书中其他部分的理论一样）基于个人的偏好。

联盟博弈的解概念为每个博弈分配一组结果。与之前一样，我们研究的每个解概念都捕捉了博弈参与者自然推理的后果；它定义了一组在某种意义上是稳定的安排。一般来说，稳定性要求是结果对某种类型的玩家群体偏离具有免疫力；相比之下，大多数（尽管不是全部）非合作博弈的解要求对单个玩家的偏离具有免疫力。文献中分析了我们研究的解概念的许多变体；我们考虑了一些旨在说明主要思想的样本。

联盟模型与非合作模型的主要区别在于，它关注的是玩家群体可以实现的目标，而不是单个玩家可以做什么，并且它不考虑玩家群体内部如何运作的细节。如果我们希望在非合作博弈中建模联盟形成的可能性，那么我们必须指定联盟如何形成以及其成员如何选择共同行动。这些细节在联盟博弈中是不存在的，因此这种博弈的结果不依赖于它们。

为了说明这两种建模方法的区别，考虑以下情况。一组个体各自拥有一组投入，并且可以使用一种技术来生产一种有价值的产品。每个个体的投入在自己的技术中是无生产力的，但在其他个体的技术中是有生产力的。这种情况的非合作模型精确指定了每个个体可用的行动集：也许每个个体可以宣布他愿意交易投入的价格向量，或者他可以提出整个社会的投入分配。相比之下，联盟模型从每个个体群体可以共同实现的支付向量集开始。联盟可以使用合同、威胁或承诺来实现高水平的生产；这些制度在联盟博弈中没有明确建模。

我们并不认为这两种方法中的任何一种更优越或更基本。它们各自反映了不同类型的战略考虑，并有助于我们对战略推理的理解。非合作模型和合作模型之间相互关系的研究也可能具有启发性。

### 13 The Core 257

- 13.1 Coalitional Games with Transferable Payoff 257
- 13.2 The Core 258
- 13.3 Nonemptiness of the Core 262
- 13.4 Markets with Transferable Payoff 263
- 13.5 Coalitional Games without Transferable Payoff 268
- 13.6 Exchange Economies 269
- Notes 274

- 可转移支付联盟博弈
- 核
- 可转移支付市场
- 无可转移联盟博弈
- 交换经济

The core is a solution concept for coalitional games that requires that no set of players be able to break away and take a joint action that makes all of them better off. After defning the concept and giving conditions for its nonemptiness, we explore its connection with the concept of a competitive equilibrium in a model of a market.

#### 13.1 Coalitional Games with Transferable Payoff

We begin with a simple version of a coalitional game in which each group of players is associated with a single number, interpreted as the payoff that is available to the group; there are no restrictions on how this payoff may be divided among the members of the group.

DEFINITION 257.1

A coalitional game with transferable payoff cnsists of

- a finite set $N$ (the set of players)
- a function $v$ that associates with every nonempty subset $S$ of $N$ (acoalition) a real number $v(S)$ (the worth of $S$).

For each coalition $S$ the number $v(S)$ is the total payoff that is available for division among the members of $S.$ That is, the set of joint actions that the coalition $S$ can take consists of all possible divisions of $v(S)$ among the members of $S.$ (Later, in Section 13.5, we define a more general notion of a coalitional game in which each coalition is associated with a set of payoff vectors that is not necessarily the set of all possible divisions of some fixed amount.)

In many situations the payoff that a coalition can achieve depends on the actions taken by the other players. However, the interpretation of a

coalitional game that best fts our discussion is that it models a situation in which the actions of the players who are not part of $S$ do not influence $v(S).$ In the literature other interpretations are given to a coalitional game; for example, $v(S)$ is sometimes interpreted to be the most payoff that the coalition $S$ can guarantee independently of the behavior of the coalition $N\setminus S.$ These other interpretations alter the interpretation of the solutions concepts defined; we do not discuss them here.
Throughout this chapter and the next we assume that the coalitional games with transferable payoff that we study have the property that the worth of the coalition $N$ of all players is at least as large as the sum of the worths of the members of any partition of $N.$ This assumption ensures that it is optimal that the coalition $N$ of all players form, as is required by our interpretations of the solution concepts we study (though the formal analysis is meaningful without the assumption).

DEFINITION 258.1

A coalitional game $\langle N,v\rangle$ with transferable payoff is cohesive if

$$v(N)\geq\sum_{k=1}^Kv(S_k)\quad\text{for every partition }\{S_1,\ldots,S_K\}\text{of }N.$$

(This is a special case of the condition of $superadditivity$, which requires that $v(S\cup T)\geq v(S)+v(T)$ for all coalitions $S$ and $T$ with $S\cap T=\varnothing$.)

#### 13.2 The Core

The idea behind the core is analogous to that behind a Nash equilibrium of a noncooperative game: an outcome is stable if no deviation is profitable. In the case of the core, an outcome is stable if no coalition can deviate and obtain an outcome better for all its members. For a coalitional game with transferable payoff the stability condition is that no coalition can obtain a payoff that exceeds the sum of its members'current payoffs. Given our assumption that the game is cohesive we confine ourselves to outcomes in which the coalition $N$ of all players forms.

Let $\langle N,v\rangle$ be a coalitional game with transferable payoff. For any profile $(x_i)_{i\in N}$ of real numbers and any coalition $S$ we let $x(S)=\sum_i\in Sx_i.$ A vector $(x_i)_{i\in S}$ of real numbers is an $S$- feasible payoff vector if $x( S)$ = $v(S)$. We refer to an $N$-feasible payoff vector as a feasible payoff profile.

DEFINITION 258.2

The core of the coalitional game with transferable payoff $\langle N,v\rangle$ is the set of feasible payoff profiles $(x_i)_i\in N$ for which there is no coalition $S$ and $S$-feasible payoff vector $(y_{i})_{i\in S}$ for
which $y_{i}>x_{i}$ for all $i\in S.$

A definition that is obviously equivalent is that the core is the set of feasible payoff profiles $(x_i)_{i\in N}$ for which $v(S)\leq x(S)$ for every coalition $S.$ Thus the core is the set of payoff profiles satisfying a system of weak linear inequalities and hence is closed and convex.

The following examples indicate the wide range of situations that may be modeled as coalitional games and illustrate the notion of the core.

EXAMPLE 259. 1

(A three-player majority game) Suppose that three players can obtain one unit of payoff, any two of them can obtain $\alpha\in[0,1]$ independently of the actions of the third, and each player alone can obtain nothing, independently of the actions of the remaining two players. We can model this situation as the coalitional game $\langle N,v\rangle$ in which $N=\{1,2,3\},v(N)=1,v(S)=\alpha$ whenever $|S|=2$, and $v(\{i\})=0$ for all $i\in N.$ The core of this game is the set of all nonnegative payoff profles $(x_1,x_2,x_3)$ for which $x(N)=1$ and $x(S)\geq\alpha$ for every two-player coalition $S.$ Hence the core is nonempty if and only if $\alpha\leq2/3.$

EXAMPLE 259.2

An expedition of $n$ people has discovered treasure in the mountains; each pair of them can carry out one piece. A coalitional game that models this situation is $\langle N,v\rangle$, where

$$
\left.v(S)=\left\{\begin{array}{ll}|S|/2&\text{if }|S|\text{ is even}\\(|S|-1)/2&\text{if }|S|\text{ is odd.}\end{array}\right.\right.
$$

If $| N| \geq 4$ is even then the core consists of the single payoff profile $( \frac 12, \ldots , \frac 12) .$ If $| N| \geq 3$ is odd then the core is empty.

### 13 核心 257

核心是联盟博弈的一个解概念，要求没有任何玩家群体能够脱离并采取一个使他们都更好的共同行动。在定义了这一概念并给出了其非空性的条件后，我们探讨了其与市场模型中竞争均衡概念的联系。

#### 13.1 具有可转移支付的联盟博弈 Coalitional Games with Transferable Payof

我们从一个简单的联盟博弈版本开始，其中每个玩家群体与一个单一数字相关联，解释为该群体可用的支付；对于如何在群体成员之间分配这一支付没有任何限制。

**定义 257.1**

具有可转移支付的联盟博弈包括：

- 一个有限集 $N$（玩家集）
- 一个函数 $v$，它将每个非空子集 $S$（联盟）与一个实数 $v(S)$（$S$ 的价值）相关联。

对于每个联盟 $S$，数字 $v(S)$ 是可用于在 $S$ 成员之间分配的总支付。也就是说，联盟 $S$ 可以采取的共同行动集包括 $v(S)$ 在 $S$ 成员之间的所有可能分配。（稍后，在第 13.5 节中，我们定义了一个更一般的联盟博弈概念，其中每个联盟与一组支付向量相关联，这组支付向量不一定是某个固定金额的所有可能分配。）

在许多情况下，联盟可以实现的支付取决于其他玩家采取的行动。然而，最适合我们讨论的联盟博弈解释是，它模拟了那些不属于 $S$ 的玩家的行动不影响 $v(S)$ 的情况。在文献中，联盟博弈还有其他解释；例如，$v(S)$ 有时被解释为联盟 $S$ 可以独立于联盟 $N\setminus S$ 的行为保证的最大支付。这些其他解释改变了解概念的定义解释；我们在此不讨论它们。

在本章和下一章中，我们假设我们研究的具有可转移支付的联盟博弈具有以下性质：所有玩家的联盟 $N$ 的价值至少与 $N$ 的任何分区的成员价值之和一样大。这一假设确保了所有玩家的联盟 $N$ 形成是最优的，正如我们对所研究解概念的解释所要求的那样（尽管在没有这一假设的情况下，正式分析也是有意义的）。

**定义 258.1**

具有可转移支付的联盟博弈 $\langle N,v\rangle$ 是**凝聚的 (cohesive)**，如果

$$v(N)\geq\sum_{k=1}^Kv(S_k)\quad\text{对于 }N\text{ 的每个分区 }\{S_1,\ldots,S_K\}.$$

（这是 **超加性 (superadditivity)** 条件的一个特例，超加性要求对于所有不相交的联盟 $S$ 和 $T$，$v(S\cup T)\geq v(S)+v(T)$。）

#### 13.2 核心

核心背后的思想与非合作博弈中的纳什均衡类似：如果没有偏离是有利可图的，那么结果是稳定的。在核心的情况下，如果没有联盟可以偏离并获得对其所有成员更好的结果，那么结果是稳定的。对于具有可转移支付的联盟博弈，稳定性条件是没有任何联盟可以获得超过其成员当前支付总和的支付。鉴于我们假设博弈是凝聚的，我们将自己限制在所有玩家的联盟 $N$ 形成的结果中。

设 $\langle N,v\rangle$ 是一个具有可转移支付的联盟博弈。对于任何 **实数剖面 (profile of real numbers)** $(x_i)_{i\in N}$ 和任何联盟 $S$，我们令 $x(S)=\sum_{i\in S}x_i$。一个实数向量 $(x_i)_{i\in S}$ 是一个 $S$-**可行支付向量** (S-feasible payoff vector)，如果 $x(S) = v(S)$。我们将 $N$-可行支付向量称为**可行支付剖面 (feasible payoff profile)**。

**定义 258.2**

具有可转移支付的联盟博弈 $\langle N,v\rangle$ 的 **核心(Core)** 是可行支付剖面 $(x_i)_{i\in N}$ 的集合，对于这些剖面，不存在联盟 $S$ 和 $S$-可行支付向量 $(y_{i})_{i\in S}$，使得对于所有 $i\in S$，$y_{i}>x_{i}$。

一个显然等价的定义是，核心是可行支付剖面 $(x_i)_{i\in N}$ 的集合，对于这些剖面，$v(S)\leq x(S)$ 对于每个联盟 $S$。因此，核心是满足一组弱线性不等式的支付剖面的集合，因此是封闭且凸的。

以下示例说明了可以建模为联盟博弈的各种情况，并说明了核心的概念。

**示例 259.1**

（一个三人多数博弈）假设三个玩家可以获得一个单位的支付，任何两个玩家可以独立于第三个玩家的行动获得 $\alpha\in[0,1]$，而每个玩家单独无法获得任何支付，独立于其他两个玩家的行动。我们可以将这种情况建模为联盟博弈 $\langle N,v\rangle$，其中 $N=\{1,2,3\},v(N)=1,v(S)=\alpha$ 当 $|S|=2$ 时，$v(\{i\})=0$ 对于所有 $i\in N$。这个博弈的核心是所有非负支付剖面 $(x_1,x_2,x_3)$ 的集合，对于这些剖面，$x(N)=1$ 且 $x(S)\geq\alpha$ 对于每个两人联盟 $S$。因此，核心非空当且仅当 $\alpha\leq2/3$。

**示例 259.2**

一支由 $n$ 人组成的探险队在山区发现了宝藏；每两个人可以携带一件宝物。一个联盟博弈 $\langle N,v\rangle$ 可以模拟这种情况，其中

$$
\left.v(S)=\left\{\begin{array}{ll}|S|/2&\text{如果 }|S|\text{ 是偶数}\\(|S|-1)/2&\text{如果 }|S|\text{ 是奇数。}\end{array}\right.\right.
$$

如果 $| N| \geq 4$ 是偶数，则核心由单一支付剖面 $( \frac 12, \ldots , \frac 12)$ 组成。如果 $| N| \geq 3$ 是奇数，则核心为空。

- **可转移支付的联盟博弈**：描述玩家如何通过合作分配总收益的模型。
- **核心**：一种稳定的收益分配方案，没有任何玩家群体有动力去破坏现有的合作。
- **实数剖面**：表示每个玩家在分配方案中获得的收益的一组实数。

#### 13.3 Nonemptiness of the Core

We now derive a condition under which the core of a coalitional game is nonempty. Since the core is defined by a system of linear inequalities such a condition could be derived from the conditions for the existence of a solution to a general system of inequalities. However, since the system of inequalities that defines the core has a special structure we are able to derive a more specific condition.
Denote by $\mathcal{C}$ the set of all coalitions, for any coalition $S$ denote by $\mathbb{R}^S$ the $|S|$-dimensional Euclidian space in which the dimensions are indexed by the members of $S$, and denote by $1_S\in\mathbb{R}^N$ the characteristic vector of $S$ given by

$$(1_S)_i=\left\{\begin{matrix}1&\text{if }i\in S\\0&\text{otherwise.}\end{matrix}\right.$$

A collection $(\lambda_S)_{S\in\mathcal{C}}$ of numbers in [0,1] is a balanced collection of weights if for every player $i$ the sum of $\lambda_S$ over all the coalitions that contain $i$ is $1:\sum_{S\in\mathcal{C}}\lambda_S1_S=1_N.$ As an example, let $|N|=3.$ Then the collection $(\lambda_S)$ in which $\lambda_S=\frac12$ if $|S|=2$ and $\lambda_S=0$ otherwise is a balanced collection of weights; so too is the collection $(\lambda_S)$ in which $\lambda_{S}=1$ if $|S|=1$ and $\lambda_S=0$ otherwise. A game $\langle N,v\rangle$ is balanced if $\sum_{S\in\mathcal{C}}\lambda_Sv(S)\leq v(N)$ for every balanced collection of weights.
One interpretation of the notion of a balanced game is the following. Each player has one unit of time, which he must distribute among all the coalitions of which he is a member. In order for a coalition $S$ to be active for the fraction of time $\lambda_S$, all its members must be active in $S$ for this fraction of time, in which case the coalition yields the payoff $\lambda_Sv(S).$ In this interpretation the condition that the collection of weights be balanced is a feasibility condition on the players' allocation of time, and a game is balanced if there is no feasible allocation of time that yields the players more than $v(N).$
The following result is referred to as the Bondareva-Shapley theorem.

PROPOSITION $262. 1

A coalitional game with transferable payoff has a nonempty core if and only if it is balanced.

Proof. Let $\langle N,v\rangle$ be a coalitional game with transferable payoff. First let $x$ be a payoff profile in the core of $\langle N,v\rangle$ and let $(\lambda_S)_S\in\mathcal{C}$ be a balanced collection of weights. Then

$$\sum_{S\in\mathcal{C}}\lambda_Sv(S)\leq\sum_{S\in\mathcal{C}}\lambda_Sx(S)=\sum_{i\in N}x_i\sum_{S\ni i}\lambda_S=\sum_{i\in N}x_i=v(N),$$

so that $\langle N,v\rangle$ is balanced.

Now assume that $\langle N,v\rangle$ is balanced. Then there is no balanced collection $(\lambda_S)_{S\in\mathcal{C}}$ of weights for which$\sum_S\in\mathcal{C}\lambda_Sv(S)>v(N).$ Therefore the convex set $\{(1_N,v(N)+\epsilon)\in\mathbb{R}^{|N|+1}{:}\epsilon>0\}$ is disjoint from the convex cone

$$\{y\in\mathbb{R}^{|N|+1}\colon y=\sum_{S\in\mathcal{C}}\lambda_S(1_S,v(S))\text{ where }\lambda_S\geq0\text{ for all }S\in\mathcal{C}\},$$

since if not then $1_N=\underline{\sum}_{S\in\mathcal{C}}\lambda_S1_S$, so that $(\lambda_S)_{S\in\mathcal{C}}$ is a balanced collection of weights and $\sum_{S\in\mathcal{C}}\lambda_Sv(S)>v(N).$ Thus by the separating hyperplane theorem (see, for example, Rockafeller (1970,Theorem 11.3)) there is a nonzero vector $(\alpha_N,\alpha)\in\mathbb{R}^|N|\times\mathbb{R}$ such that

$$(\alpha_N,\alpha)\cdot y\geq0>(\alpha_N,\alpha)\cdot(1_N,v(N)+\epsilon)$$

for all $y$ in the cone and all $\epsilon>0.$ Since $(1_N,v(N))$ is in the cone, we have $\alpha<0.$

Now let $x=\alpha_N/(-\alpha).$ Since $(1_S,v(S))$ is in the cone for all $S\in\mathcal{C}$, we have $x(S)=x\cdot1_S\geq v(S)$ for all $S\in\mathcal{C}$ by the left-hand inequality in (263.1), and $v(N)\geq1_Nx=x(N)$ from the right-hand inequality. Thus $v(N)=x(N)$, so that the payoff profile $x$ is in the core of $\langle N, v\rangle$.

#### 13.3 核心的非空性

我们现在推导出一个条件，使得联盟博弈的核心非空。由于核心是由一组线性不等式定义的，因此可以从一般不等式系统存在解的条件中推导出这一条件。然而，由于定义核心的不等式系统具有特殊结构，我们能够推导出一个更具体的条件。

用 $\mathcal{C}$ 表示所有联盟的集合，对于任何联盟 $S$，用 $\mathbb{R}^S$ 表示 $|S|$ 维欧几里得空间，其中维度由 $S$ 的成员索引，并用 $1_S\in\mathbb{R}^N$ 表示 $S$ 的特征向量，定义为：

$$(1_S)_i=\left\{\begin{matrix}1&\text{如果 }i\in S\\0&\text{否则。}\end{matrix}\right.$$

如果对于每个玩家 $i$，包含 $i$ 的所有联盟的 $\lambda_S$ 之和为 $1$，即 $\sum_{S\in\mathcal{C}}\lambda_S1_S=1_N$，则 $(\lambda_S)_{S\in\mathcal{C}}$ 是一个平衡权重集合。例如，设 $|N|=3$，则当 $|S|=2$ 时 $\lambda_S=\frac12$，否则 $\lambda_S=0$ 的集合 $(\lambda_S)$ 是一个平衡权重集合；同样，当 $|S|=1$ 时 $\lambda_{S}=1$，否则 $\lambda_S=0$ 的集合 $(\lambda_S)$ 也是一个平衡权重集合。如果对于每个平衡权重集合，$\sum_{S\in\mathcal{C}}\lambda_Sv(S)\leq v(N)$，则博弈 $\langle N,v\rangle$ 是**平衡的**。

平衡博弈的概念可以解释为：每个玩家有一单位的时间，他必须将其分配给他所属的所有联盟。为了使联盟 $S$ 在时间比例 $\lambda_S$ 内活跃，其所有成员必须在这段时间内活跃于 $S$，此时联盟产生支付 $\lambda_Sv(S)$。在这种解释中，权重集合的平衡条件是玩家时间分配的可行性条件，而博弈是平衡的，如果没有可行的时间分配能够使玩家获得超过 $v(N)$ 的支付。

以下结果被称为**邦达列娃-沙普利定理**。

**命题 262.1**

一个具有可转移支付的联盟博弈具有非空核心，当且仅当它是平衡的。

**证明：**  
设 $\langle N,v\rangle$ 是一个具有可转移支付的联盟博弈。首先，设 $x$ 是 $\langle N,v\rangle$ 核心中的一个支付剖面，$(\lambda_S)_S\in\mathcal{C}$ 是一个平衡权重集合。那么

$$\sum_{S\in\mathcal{C}}\lambda_Sv(S)\leq\sum_{S\in\mathcal{C}}\lambda_Sx(S)=\sum_{i\in N}x_i\sum_{S\ni i}\lambda_S=\sum_{i\in N}x_i=v(N),$$

因此 $\langle N,v\rangle$ 是平衡的。

现在假设 $\langle N,v\rangle$ 是平衡的。那么不存在任何平衡权重集合 $(\lambda_S)_{S\in\mathcal{C}}$ 使得 $\sum_S\in\mathcal{C}\lambda_Sv(S)>v(N)$。因此，凸集 $\{(1_N,v(N)+\epsilon)\in\mathbb{R}^{|N|+1}{:}\epsilon>0\}$ 与凸锥

$$\{y\in\mathbb{R}^{|N|+1}\colon y=\sum_{S\in\mathcal{C}}\lambda_S(1_S,v(S))\text{ 其中 }\lambda_S\geq0\text{ 对所有 }S\in\mathcal{C}\}$$

不相交，因为如果不相交，则 $1_N=\underline{\sum}_{S\in\mathcal{C}}\lambda_S1_S$，因此 $(\lambda_S)_{S\in\mathcal{C}}$ 是一个平衡权重集合，且 $\sum_{S\in\mathcal{C}}\lambda_Sv(S)>v(N)$。因此，根据分离超平面定理（例如，参见 Rockafeller (1970, Theorem 11.3)），存在一个非零向量 $(\alpha_N,\alpha)\in\mathbb{R}^|N|\times\mathbb{R}$，使得

$$(\alpha_N,\alpha)\cdot y\geq0>(\alpha_N,\alpha)\cdot(1_N,v(N)+\epsilon)$$

对于锥中的所有 $y$ 和所有 $\epsilon>0$。由于 $(1_N,v(N))$ 在锥中，我们有 $\alpha<0$。

现在设 $x=\alpha_N/(-\alpha)$。由于 $(1_S,v(S))$ 对所有 $S\in\mathcal{C}$ 都在锥中，根据 (263.1) 左侧不等式，我们有 $x(S)=x\cdot1_S\geq v(S)$ 对所有 $S\in\mathcal{C}$，并且根据右侧不等式，$v(N)\geq1_Nx=x(N)$。因此 $v(N)=x(N)$，所以支付剖面 $x$ 在 $\langle N, v\rangle$ 的核心中。

#### 13.5 Coalitional Games without Transferable Payoff

In a coalitional game with transferable payoff each coalition $S$ is characterized by a single number $v(S)$, with the interpretation that $v(S)$ is a payoff that may be distributed in any way among the members of $S.$ We now study a more general concept, in which each coalition cannot necessarily achieve all distributions of some fıxed payoff; rather, each coalition $S$ is characterized by an arbitrary set $V(S)$ of consequences. DEFINITION 268.2 A coalitional game (without transferable payoff) consists of

- a finite set $N$ (the set of players)

- a set $X$ (the set of consequences)
- a function $V$ that assigns to every nonempty subset $S$ of $N$ a coalition a set $V( S) \subseteq X$.
- for each player $i\in N$ a preference relation $\succsim_i$ on $X.$

Any coalitional game with transferable payoff $\langle N,v\rangle$ (Definition 257.1) can be associated with a general coalitional game $\langle N,X,V,(\succsim_i)_{i\in N}\rangle$ as follows: $X=\mathbb{R}^N,V(S)=\{x\in\mathbb{R}^N:\sum_{i\in S}x_i=v(S)$ and $x_j=0$ if $j\in$ $N\setminus S\}$ for each coalition $S$, and $x\succsim_iy$ if and only if $x_i\geq y_i.$ Under this association the set of coalitional games with transferable payoff is a subset of the set of all coalitional games.

The definition of the core of a general coalitional game is a natural extension of our definition for the core of a game with transferable payoff (Definition 258.2).

DEFINITION 268.3

The core of the coalitional game $\langle N,V,X$, $(\succsim_{i})_{i\in N}\rangle$ is the set of all $x\in V(N)$ for which there is no coalition $S$ and $y\in V(S)$ for which $y\succ_ix$ for all $i\in S.$

Under conditions like that of balancedness for a coalitional game with transferable payoff (see Section 13.3) the core of a general coalitional game is nonempty (see Scarf (1967), Billera (1970), and Shapley (1973)).

We do not discuss these conditions here.

#### 13.5 无转移支付的联盟博弈

在具有可转移支付的联盟博弈中，每个联盟 $S$ 由一个单一数字 $v(S)$ 来表征，其含义是 $v(S)$ 是一种可以在 $S$ 成员之间以任意方式分配的支付。我们现在研究一个更一般的概念，其中每个联盟不一定能够实现某种固定支付的所有分配；相反，每个联盟 $S$ 由一个任意的结果集 $V(S)$ 来表征。

**定义 268.2**  

一个（无转移支付的）联盟博弈包括：

- 一个有限集 $N$（玩家集）
- 一个集合 $X$（结果集）
- 一个函数 $V$，它为 $N$ 的每个非空子集 $S$（联盟）分配一个集合 $V(S) \subseteq X$。
- 对于每个玩家 $i\in N$，$X$ 上的一个偏好关系 $\succsim_i$。

任何具有可转移支付的联盟博弈 $\langle N,v\rangle$（定义 257.1）都可以与一个一般联盟博弈 $\langle N,X,V,(\succsim_i)_{i\in N}\rangle$ 关联如下：$X=\mathbb{R}^N$，$V(S)=\{x\in\mathbb{R}^N:\sum_{i\in S}x_i=v(S)$ 且如果 $j\in N\setminus S$ 则 $x_j=0\}$ 对于每个联盟 $S$，并且 $x\succsim_iy$ 当且仅当 $x_i\geq y_i$。在这种关联下，具有可转移支付的联盟博弈集是所有联盟博弈集的一个子集。

一般联盟博弈的核心定义是对具有可转移支付的博弈核心定义（定义 258.2）的自然扩展。

**定义 268.3**  

联盟博弈 $\langle N,V,X,(\succsim_{i})_{i\in N}\rangle$ 的**核心**是所有 $x\in V(N)$ 的集合，其中不存在任何联盟 $S$ 和 $y\in V(S)$，使得对于所有 $i\in S$，$y\succ_ix$。

在类似于具有可转移支付的联盟博弈的平衡性条件下（参见第 13.3 节），一般联盟博弈的核心是非空的（参见 Scarf (1967)、Billera (1970) 和 Shapley (1973)）。

我们在此不讨论这些条件。

### 14 Stable Sets, the Bargaining Set, and the Shapley Value 277

- 14.1 Two Approaches 277
- 14.2 The Stable Sets of von Neumann and Morgenstern 278
- 14.3 The Bargaining Set, Kernel, and Nucleolus 281
- 14.4 The Shapley Value 289
- Notes 297

#### 14.1 Two Approaches 277

#### 14.2 The Stable Sets of von Neumann and Morgenstern 278

#### 14.3 The Bargaining Set, Kernel, and Nucleolus 281

#### 14.4 The Shapley Value 289

### 15 The Nash Solution 299

- 15.1 Bargaining Problems 299
- 15.3 The Nash Solution: Definition and Characterization 301
- 15.3 An Axiomatic Definition 305
- 15.4 The Nash Solution and the Bargaining Game of Alternating Offers 310
- 15.5 An Exact Implementation of the Nash Solution 311
- Notes 312
