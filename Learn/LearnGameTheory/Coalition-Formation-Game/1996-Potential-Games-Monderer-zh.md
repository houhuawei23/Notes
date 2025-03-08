# 势博弈

- 作者：Dov Monderer a\*, Lloyd S. Shapley b
- 期刊：Games and Economic Behavior

- Dov Monderer：以色列理工学院工业工程与管理学院，海法 32000，以色列
- Lloyd S. Shapley：加州大学洛杉矶分校经济系与数学系，洛杉矶，加利福尼亚 90024
- 收稿日期：1994 年 1 月 19 日
- 初稿：1988 年 12 月。第一作者感谢以色列理工学院研究促进基金的资助。电子邮件：dov@techunix.technion.ac.il

---

### 摘要：

我们定义并讨论了策略形式博弈中的几种势函数概念。我们刻画了具有势函数的博弈，并展示了多种应用。经济学文献分类号：C72, C73。© 1996 学术出版社。

### 1. 引言

考虑一个对称寡头垄断的 Cournot 竞争，其成本函数为线性函数 $c_{i}\left(q_{i}\right)=cq_{i}$ ，$1\leq i\leq n$。逆需求函数为 $F(Q)$ ，$Q>0$，是一个正函数（不需要对 $F$ 进行单调性、连续性或可微性假设）。定义在 $R_{++}^{n}$ 上的企业 $i$ 的利润函数为：

$$\Pi_i(q_1,q_2,\ldots,q_n)=F(Q)q_i-cq_i,$$

其中 $Q=\sum_{j=1}^{n}q_{j}$。

定义一个函数 $P$ ：$R_{++}^{n}\longrightarrow R$：

$$P(q_1,q_2,\ldots,q_n)=q_1q_2\cdots q_n(F(Q)-c).$$

对于每个企业 $i$ 和每个 $q_{-i}\in R_{++}^{n-1}$：

$$\begin{array}{rl}\Pi_i(q_i,q_{-i})-\Pi_i(x_i,q_{-i})>0,&\mathrm{iff}&P(q_i,q_{-i})-P(x_i,q_{-i})>0,\\&&\forall q_i,x_i\in R_{++}.\end{array}$$

满足 (1.1) 的函数 $P$ 被称为**序数势函数 (ordinal potential)**，而具有序数势函数的博弈被称为**序数势博弈 (ordinal potential game)**。显然，Cournot 博弈的纯策略均衡集与每个企业的利润由 $P$ 给出的博弈的纯策略均衡集一致。如果我们对混合策略感兴趣，则需要比 (1.1) 更强的条件。

考虑一个具有线性逆需求函数 $F(Q)=a-bQ,a,b>0$ 的准 Cournot 竞争，以及任意可微成本函数 $c_i(q_i)$，$1\leq i\leq n$。定义一个函数 $P^{*}((q_{1},q_{2},\ldots,q_{n}))$ 为：

$$\begin{aligned}P^{*}((q_{1},q_{2},\ldots,q_{n}))&=\:a\sum_{j=1}^{n}q_{j}-b\sum_{j=1}^{n}q_{j}^{2}-b\sum_{1\leq i<j\leq n}q_{i}q_{j}\\&-\sum_{j=1}^{n}c_{j}(q_{j}).\end{aligned}$$

可以验证，对于每个企业 $i$ 和每个 $q_{-i}\in R_{+}^{n-1}$：

$$\Pi_{i}(q_{i},q_{-i})-\Pi_{i}(x_{i},q_{-i})=P^{*}(q_{i},q_{-i})-P^{*}(x_{i},q_{-i}),\quad\forall q_{i},x_{i}\in R_{+}.$$

满足 (1.3) 的函数 $P^{*}$ 被称为势函数。等式 (1.3) 意味着准 Cournot 博弈的混合策略均衡集与将每个支付函数替换为 $P^{*}$ 的博弈的混合策略均衡集一致。特别是，试图联合最大化势函数 $P^{*}$（或序数势函数 $P$）的企业最终会达到均衡。我们将证明，最多存在一个势函数（相差一个常数）。这引发了一个自然的问题：$P^{*}$ 的经济内容（或解释）是什么？企业试图联合最大化什么？

> 1 在这个博弈中可能出现负价格，尽管在任何非退化均衡中价格都是正的。

> 2 在物理学中，$P^{*}$ 是 $(\Pi_{1},\Pi_{2},\ldots,\Pi_{n})$ 的势函数，如果

> $$\frac{\partial\Pi_{i}}{\partial q_{i}}=\frac{\partial P^{*}}{\partial q_{i}}\quad\mathrm{对于每个}\:1\leq i\leq n.$$

> 如果利润函数是连续可微的，则此条件等价于 (1.3)。

> 3 Slade (1993) 证明了准 Cournot 博弈存在满足 (1.3) 的函数 $P^{*}$，她称此函数为虚构目标函数。

> 4 每个最大化 $P$ 的 $q^*$ 都是纯策略均衡，但可能存在只是“局部”最大点的纯策略均衡配置，也可能存在混合策略均衡配置。因此，势函数的 argmax 集可以用作势博弈的细化工具（这个问题在第 5 节中讨论）。Neyman (1991) 证明，如果势函数是凹的且连续可微，则每个混合策略均衡配置都是纯策略的，并且必须最大化势函数。Neyman 的结果与 Shin 和 Williamson (1994) 的贝叶斯博弈中的“简单均衡结果”概念相关。

我们对此问题没有答案。然而，势函数的存在显然帮助我们（和玩家）更好地分析博弈。在本文中，我们将证明势博弈的各种性质，并提供简单的方法来检测它们并计算它们的势函数。

据我们所知，第一个在策略形式博弈中使用势函数的是 Rosenthal (1973)。Rosenthal 定义了拥堵博弈类，并通过显式构造势函数证明了该类中的每个博弈都具有纯策略均衡。拥堵博弈类一方面较窄，但另一方面对经济学非常重要。任何由同质代理从有限选择集中选择，且每个玩家的支付取决于选择每个选择的玩家数量的博弈都是拥堵博弈。我们将证明，拥堵博弈类与有限势博弈类（在同构意义上）一致。

最近，许多注意力集中在几种“短视”学习过程的概念上。我们证明，对于一般的有限博弈，序数势的存在等价于由单边更好响应动态定义的学习过程收敛到均衡。新的学习文献引发了人们对 Brown (1951) 定义的策略形式博弈中的虚构博弈过程的新兴趣。Robinson (1951) 研究了零和博弈中的虚构博弈，Miyasawa (1961)、Shapley (1964)、Deschamps (1973) 以及最近的 Krishna (1991)、Milgrom 和 Roberts (1991)、Sela (1992)、Fudenberg 和 Kreps (1993)、Jordan (1993)、Hofbauer (1994)、Krishna 和 Sjostrom (1994)、Fudenberg 和 Levine (1994)、Monderer 等人 (1994) 等研究了非零和博弈中的虚构博弈。在 Monderer 和 Shapley (1996) 中，我们证明了虚构博弈过程在一类包含有限（加权）势博弈的博弈中收敛到均衡集。Milchtaich (1996) 分析了与拥堵博弈相关的博弈类。他的工作以及 Blume (1993) 的工作表明，序数势博弈也与进化学习自然相关（参见 Crawford, 1991; Kandori 和 Rob, 1992; Young, 1993; Roth 和 Erev, 1995; 以及其中列出的参考文献）。

由于势函数在相差一个常数的情况下是唯一确定的，势函数的 argmax 集不依赖于特定的势函数。因此，对于势博弈，这个 argmax 集至少在技术上细化了均衡集。我们展示了这种细化概念准确预测了 Van Huyck 等人 (1990) 的实验结果。我们并不试图为这种预测能力提供任何解释（也许只是巧合）。Blume (1993) 提供了一种可能的解释方式。Blume 讨论了玩家仅与人口中的一小部分直接交互的各种随机策略修订过程。他证明了对于对数线性策略修订过程，对称势博弈中玩家的策略收敛到势函数的 argmax 集。

Hart 和 Mas-Colell (1989) 将势理论应用于合作博弈。除了我们都使用势理论外，我们的工作并无关联。然而，我们将在最后一节展示，将我们的工作与 Hart 和 Mas-Colell 的工作结合，可以得出一个令人惊讶的价值理论应用。

本文的结构如下：在第 2 节中，我们给出基本定义，并提供有限势博弈和有限序数势博弈的几个有用刻画。第 3 节给出了势博弈与拥堵博弈之间的等价定理。第 4 节讨论并刻画了无限势博弈。第 5 节讨论了 Van Huyck 等人的实验结果。第 6 节展示了我们的理论在合作博弈策略方法中的应用。

### 2. 势博弈

- 设 $\Gamma(u^{1},u^{2},\ldots,u^{n})$ 是一个具有有限数量玩家的策略形式博弈。
- 玩家集合为 $N=\{1,2,\ldots,n\}$，玩家 $i$ 的策略集为 $Y^{\mathrm{i}}$，玩家 $i$ 的支付函数为 $u^{i}$ ：$Y\rightarrow R$，其中 $Y=Y^{1}\times Y^{2}\times$ $\cdots\times Y^{n}$ 是策略配置集，$R$ 表示实数集。
- 当不会引起混淆时，我们将 $\Gamma(u^{1},u^{2},\ldots,u^{n})$ 简记为 $\Gamma$。
- 对于 $S\subseteq N$，$-S$ 表示 $S$ 的补集，$Y^{5}$ 表示笛卡尔积 $X_{i\in S}Y^i$。
- 对于单元素集 $\{i\}$，$Y^{-[i]}$ 记为 $Y^{-i}$。
- 函数 $P$ ：$Y\rightarrow R$ 是 $\Gamma$ 的序数势函数，如果对于每个 $i\in N$ 和每个 $y^{-i}\in Y^{-\bar{I}}$：

$$\begin{array}{rl}u^i(y^{-i},x)-u^i(y^{-i},z)>0&\mathrm{iff}\quad P(y^{-i},x)-P(y^{-i},z)>0\\&\mathrm{对于每个}\:x,z\in Y^i.\end{array}$$

$\Gamma$ 被称为序数势博弈，如果它允许一个序数势函数。

设 $w=(w^{i})_{i\in N}$ 是一个正数向量，称为权重。函数 $P$ ：$Y\rightarrow R$ 是 $\Gamma$ 的 $w$ -势函数，如果对于每个 $i\in N$ 和每个 $y^{-i}\in Y^{-i}$：

$$\begin{array}{rcl}u^i(y^{-i},x)-u^i(y^{-i},z)&=&w^i\left(P(y^{-i},x)-P(y^{-i},z)\right)\\&&\text{对于每个}\:x,z\in Y^i.\end{array}$$

$\Gamma$ 被称为 $w$ -势博弈，如果它允许一个 $w$ -势函数。

当我们不关心特定权重 $u$ 时，我们简单地说 $P$ 是一个加权势函数，$\Gamma$ 是一个加权势博弈。

函数 $P$ ：$Y\rightarrow R$ 是 $\Gamma$ 的精确势函数（简称势函数），如果它是 $\Gamma$ 的 $w$ -势函数，且对于每个 $i\in N$，$w^{i}=1$。

$\Gamma$ 被称为精确势博弈（简称势博弈），如果它允许一个势函数。例如，矩阵 $P$ 是下面描述的囚徒困境博弈 $G$ 的势函数：

$$G=\left(\begin{array}{cc}(1,1)&(9,0)\\(0,9)&(6,6)\end{array}\right),\quad P=\left(\begin{array}{cc}4&3\\3&0\end{array}\right).$$

下一个引理刻画了序数势博弈的均衡集。其显然的证明将被省略。

引理 2.1. 设 $P$ 是 $\Gamma(u^{1},u^{2},\ldots,u^{n})$ 的序数势函数。那么 $\Gamma(u^{1},u^{2},\ldots,u^{n})$ 的均衡集与 $\Gamma(P,P,\ldots,P)$ 的均衡集一致。即，$y\in Y$ 是 $\Gamma$ 的均衡点，当且仅当对于每个 $i\in N$：

$$P(y)\geq P(y^{-i},x)\quad \text{对于每个}\:x\in Y^{i}.$$

因此，如果 $P$ 在 $Y$ 中允许一个最大值，则 $\Gamma$ 具有一个（纯策略）均衡。

推论 2.2. 每个有限序数势博弈都具有一个纯策略均衡。

$Y$ 中的路径是一个序列 $\gamma=(y_{0},y_{1},\ldots)$，使得对于每个 $k\geq1$，存在唯一的玩家，比如玩家 $i$，使得 $y_{k}=(y_{k-1}^{-i},x)$，其中 $x\neq y_{k-1}^i$ 在 $Y^{i}$ 中。$y0$ 被称为 $\gamma$ 的初始点，如果 $\gamma$ 是有限的，则其最后一个元素被称为 $\gamma$ 的终点。$\gamma=(y_{0},y_{1},\ldots)$ 是关于 $\Gamma$ 的改进路径，如果对于所有 $k\geq 1$，$u^{i}( y_{k}) > u^{i}( y_{k- 1})$，其中 $i$ 是第 $k$ 步的唯一偏离者。因此，改进路径是由短视玩家生成的路径。$\Gamma$ 具有有限改进性质 $(FIP)$，如果每个改进路径都是有限的。

引理 2.3. 每个有限序数势博弈都具有 FIP。

证明. 对于每个改进路径 $\gamma=(y_{0},y_{1},y_{2},\ldots)$，我们有 (2.1)：

$$P(y_0)<P(y_1)<P(y_2)<\cdots.$$

由于 $Y$ 是有限集，序列 $\gamma$ 必须是有限的。

> 9 使用 Blume (1993) 的术语，我们可以给出一个等价定义：$\Gamma$ 是加权势博弈，当且仅当存在一个支付函数，它与每个玩家的支付函数强最佳响应等价。Sela (1992) 证明，如果两人博弈 $(A, B)$ 没有弱支配策略，则它具有加权势函数，当且仅当它在混合策略中与形式为 $(P,P)$ 的博弈更好响应等价（参见 Monderer 和 Shapley (1996) 的准确定义）。这个结果可以很容易地推广到 $n$ 人博弈。

---

显然，对于具有 FIP 的有限博弈，特别是有限序数势博弈，每个最大改进路径必须在均衡点终止。也就是说，基于单边更好响应动态的短视学习过程收敛到均衡集。然而，我们得到了一个更强的学习结果：

定理 2.4 (Monderer 和 Shapley, 1996). 每个有限加权势博弈都具有虚构博弈性质。

有趣的是，具有 FIP 并不等价于具有序数势函数。一个反例是下面描述的游戏 $G_{1}$。$G_{1}$ 中的行标记为 $d$ 和 $b$，列标记为 $t$ 和 $d$。

$$G_1=\left(\begin{array}{cc}(1,0)&(2,0)\\(2,0)&(0,1)\end{array}\right).$$

游戏 $G_{1}$ 具有 FIP，但任何序数势函数 $P$ 对于 $G_{1}$ 必须满足以下不可能的关系序列：

$$P(a,c)<P(b,c)<P(b,d)<P(a,d)=P(a,c).$$

函数 $P$ ：$Y\rightarrow R$ 是 $\Gamma$ 的广义序数势函数，如果对于每个 $i\in N$ 和每个 $y^{-i}\in Y^{-\bar{I}}$，以及每个 $x,z\in Y^{i}$：

$$^{-i},x)-u^{i}(y^{-i},z)>0\quad\mathrm{意味着}\quad P(y^{-i},x)-P(y^{-i},z)>0.$$

引理 2.5. 设 $\Gamma$ 是一个有限博弈。那么，$\Gamma$ 具有 FIP，当且仅当 $\Gamma$ 具有广义序数势函数。

证明. 设 $\Gamma$ 是一个具有 FIP 的博弈。定义一个二元关系 ‘>’ 在 $Y$ 上如下：$x>y$ 当且仅当 $x\neq y$ 且存在一个有限改进路径 $\gamma$，其初始点为 $y$，终点为 $\lambda$。有限改进性质意味着 ‘>’ 是一个传递关系。设 $Z\subseteq Y$。我们说 $Z$ 是可表示的，如果存在 $Q$ ：$Z\rightarrow R$，使得对于每个 $x,y\in Z$，$x>y$ 意味着 $Q(x) > Q(y)$。设 $Z$ 是 $Y$ 的最大可表示子集。我们继续证明 $Z=Y$。假设 $x\not\in Z$。如果 $x>z$ 对于每个 $z\in Z$，我们通过定义 $Q(x) = 1 + \max _{z\in \mathbb{Z} } Q(z)$ 将 $Q$ 扩展到 $Z\cup\{x\}$，从而与 $Z$ 的最大性矛盾。如果 $z>x$ 对于每个 $z\in\mathbb{Z}$，我们通过定义 $Q(x)=\operatorname*{min}_{z\in\mathbb{Z}}Q(z)-1$ 将 $Q$ 扩展到 $Z\cup\{x\}$，再次与 $Z$ 的最大性矛盾。否则，我们通过定义 $Q(x)=(a+b)/2$ 扩展 $Q$ 并与 $Z$ 的最大性矛盾，其中 $a=\operatorname*{max}\{Q(z):z\in Z,\quad x>z\},\operatorname{and}b=\operatorname*{min}\{Q(z):z\in Z,\quad z>.$ 因此，$Y$ 是可表示的。

推论 2.6. 设 $\Gamma$ 是一个具有 FIP 的有限博弈。假设此外，对于每个 $i\in N$ 和每个 $y^{-i}\in Y^{-i}$：

$$u^i(y^{-i},x)\neq u^i(y^{-i},z)\quad \text{对于每个}\:x\neq z\in Y^i.$$

那么 $\Gamma$ 具有序数势函数。

证明. 观察到 $\Gamma$ 的条件意味着每个广义序数势函数都是 $\Gamma$ 的序数势函数。因此，证明由引理 2.5 得出。

序数势博弈有许多序数势函数。对于精确势博弈，我们有：

引理 2.7. 设 $P_{1}$ 和 $P_{2}$ 是博弈 $\Gamma$ 的势函数。那么存在一个常数 $c$，使得：

$$P_1(y)-P_2(y)=c\quad \text{对于每个}\:y\in Y.$$

证明. 固定 $z\in Y$。对于所有 $y\in Y$，定义：

$$H(y)=\sum_{i=1}^n\left[u^i(a_{i-1})-u^i(a_i)\right],$$

其中 $u_{0}=y$，且对于每个 $1\leq i\leq n$，$a_{i}=(a_{i-1}^{-i},z^{i})$。

如果 $P$ 代表 $P_{1}$ 或 $P_{2}$，则由 (2.1)，$H(y)=P(y)-P(z)$ 对于每个 $y\in Y$。因此：

$$P_1(y)-P_2(y)=c\quad\mathrm{对于每个}\:y\in Y.$$

下一个结果以类似于物理学中势函数的标准方法刻画了精确势博弈。

对于有限路径 $\gamma=(y_{0},y_{1},\ldots,y_{N})$ 和函数向量 $v=(v^{1},v^{2},\ldots,v^{n})$，其中 $v^{\prime}$ ：$Y\rightarrow R$，我们定义：

$$I(\gamma,v)=\sum_{k=1}^{n}\left[v^{i_{k}}(y_{k})-v^{i_{k}}(y_{k-1}\right],$$

其中 $i_k$ 是第 $k$ 步的唯一偏离者（即 $y_{k}^{i_{k}}\neq y_{k-1}^{i_{k}}$）。

> 12 Milchtaich (1996) 给出了这个结果的构造性且更优雅的证明；他证明了将每个 $y\in Y$ 赋值为通过改进路径连接到 $y$ 的策略配置数的函数 $P$ 是 $\Gamma$ 的广义序数势函数。

---

路径 $\gamma=(y_{0},y_{1},\ldots,y_{N})$ 是闭合的，如果 $y_{0}=y_{N}$。如果此外 $y_{l}\neq y_{k}$ 对于每个 $0\leq I\neq k\leq N-1$，则它是简单闭合路径。简单闭合路径的长度定义为其中不同顶点的数量。即，$\gamma=(y_{0},y_{1},\ldots,y_{N})$ 的长度为 $N$。

定理 2.8. 设 $\Gamma$ 是一个策略形式博弈，如本节开头所述。那么以下命题等价：

- (1) $\Gamma$ 是势博弈。
- (2) $I\left(\gamma,u\right)=0$ 对于每个有限闭合路径 $\gamma$。
- (3) $I\left(\gamma,u\right)=0$ 对于每个有限简单闭合路径 $\gamma$。
- (4) $I\left(\gamma,u\right)=0$ 对于每个长度为 4 的有限简单闭合路径 $\gamma$。

定理 2.8 的证明在附录 A 中给出。

一个典型的长度为 4 的简单闭合路径 $\gamma$ 描述如下。在这个路径中，$i$ 和 $j$ 是活跃玩家，$a\in Y^{-\{i,j\}}$ 是其他玩家的固定策略配置，$x_{i},y_{i}\in Y^{\bar{I}}$，$x_{j},y_{j}\in Y^{j}$。

![](https://storage.simpletex.cn/view/fQgYhac0IDdu0XG7B7aIhWzEdN6uGSIh9)

其中 $A=(x_{i},x_{j},a)$，$B=(y_{i},x_{j},a)$，$C=(y_{i},y_{j},a)$，$D=(x_{i},y_{j},a)$。

推论 2.9. $\Gamma$ 是势博弈，当且仅当对于每个 $i, j\in N$，每个 $a\in Y^{-\{i,j\}}$，以及每个 $X_{i}$，$y_{i}\in Y^{i}$ 和 $X_{j}$，$y_{j}\in Y^{j}$：

$$u^{i}(B)-u^{i}(A)+u^{j}(C)-u^{j}(B)+u^{i}(D)-u^{i}(C)+u^{j}(A)-u^{j}(D)=0,$$

其中点 $A, B,C$ 和 $D$ 如上所述。

我们以关于有限博弈的混合扩展的重要评论结束本节。

引理 2.10. 设 $\Gamma$ 是一个有限博弈。那么 $\Gamma$ 是 $U$ -势博弈，当且仅当 $\Gamma$ 的混合扩展是 $u$ -势博弈。

证明. 对于 $i\in N$，设 $\Delta^{i}$ 是玩家 $i$ 的混合策略集，$U^{i}$ 是玩家 $i$ 在 $\Gamma$ 的混合扩展中的支付函数。即：

$$\begin{array}{ll}U^i(f)&=\:U^i(f^1,f^2,\dots,f^n)\\&=\:\sum_{y\in Y}u^i(y^1,y^2,\dots,y^n)f^1(y^1)f^2(y^2)\dots f^n(y^n),\quad\forall f\in\\\end{array}$$

其中 $\Delta=X_{\mathrm{~i~e}N}\Delta^{i}$。显然，如果 $\bar{P}$ ：$\Delta\rightarrow R$ 是 $\Gamma$ 的混合扩展的 $u$ -势函数，则其在 $Y$ 上的限制是 $\Gamma$ 的 $u$ -势函数。反之，假设 $P$ 是 $\Gamma$ 的 $W$ -势函数，则可以很容易地验证 $\bar{P}$ 是 $\Gamma$ 的混合扩展的势函数，其中：

$$\bar{P}(f^{1},f^{2},\ldots,f^{n})=\sum_{y\in Y}P(y^{1},y^{2},\ldots,y^{n})f^{1}(y^{1})f^{2}(y^{2})\ldots f^{n}(y^{n}).$$

Sela (1992) 给出了一个序数势博弈的例子，其混合扩展不是序数势博弈。

### 3. 拥堵博弈

拥堵博弈由 Rosenthal (1973) 定义。它们源自文献中广泛讨论的拥堵模型（参见 Garcia 和 Zangwill, 1981）。考虑一个说明性例子：

![](https://storage.simpletex.cn/view/fi9dDlatHWWwpbH65fQBKLPhZA3GHiSG5)

在上述拥堵模型中，司机 $Ur$ 必须从点 $A$ 到点 $C$，司机 $b$ 必须从点 $B$ 到点 $D$。$AB$ 称为路段 1，$BC$ 称为路段 $2,\ldots$ 等。$c_{j}(1)$ 表示路段 $j$ 的单个用户的支付（例如成本的负数）。$c_{j}(2)$ 表示如果两个司机都使用路段 $j$，则每个用户的支付。司机们因此参与了一个博弈（相关的拥堵博弈，$CG$），其策略形式如下（行标记为 $\{1,2\}$ 和 $\{3,4\}$，列标记为 $\{1,3\}$ 和 $\{2,4\}$）：

$$\left.CG=\left(\begin{array}{cc}{(c_{1}(2)+c_{2}(1),c_{1}(2)+c_{3}(1))}&{(c_{2}(2)+c_{1}(1),c_{2}(2)+c_{4}(1))}\\{(c_{3}(2)+c_{4}(1),c_{3}(2)+c_{1}(1))}&{(c_{4}(2)+c_{3}(1),c_{4}(2)+c_{2}(1))}\\\end{array}\right.\right.$$

根据推论 2.9，拥堵博弈 $CG$ 允许一个势函数。特别是（且对支付 $c_{j}(i)$ 没有任何限制），它具有一个（纯策略）均衡。为了完整性，我们在下面附上拥堵博弈的势函数 $P$。势函数由公式 (3.2) 计算：

$$P=\left(\begin{array}{cc}c_1(1)+c_1(2)+c_2(1)+c_3(1)&c_2(1)+c_2(2)+c_1(1)+c_4(1)\\c_3(1)+c_3(2)+c_4(1)+c_1(1)&c_4(1)+c_4(2)+c_3(1)+c_2(1)\end{array}\right)$$

拥堵模型 $C(N,M,(\Sigma^{i})_{i\in N},(c_{j})_{j\in M}$ 定义如下。$N$ 表示玩家集合 $\{1,2,\ldots,n\}$（例如司机）。$M$ 表示设施集合 $\{1,2,\ldots,m\}$（例如路段）。对于 $i\in N$，设 $\Sigma^{i}$ 是玩家 $l$ 的策略集，其中每个 $A^{i}\in\Sigma^{i}$ 是设施的非空子集（例如一条路线）。对于 $j\in M$，设 $c_{j}\in R^{|1,2,\ldots,n}$ 表示支付向量，其中 $c_{j}(k)$ 表示如果恰好有 $k$ 个用户使用设施 $j$，则每个用户的支付（例如成本）。与拥堵模型相关的拥堵博弈是策略形式博弈，其玩家集合为 $N$，策略集为 $(\Sigma^{i})_{i\in N}$，支付函数为 $(v^i)_{i\in N}$，定义如下：

设 $\Sigma=X_{i\in N}\Sigma^{i}$。对于所有 $A\in\Sigma$ 和每个 $j\in M$，设 $\sigma_{j}(A)$ 是设施 $j$ 的用户数量。即：

$$\sigma_j(A)=\#\{i\in N\colon j\in A^i\},$$

其中 $A=(A^{1},A^{2},\ldots,A^{n})$。定义 $v^{i}$ ：$\Sigma\rightarrow R$ 为：

$$v^i(A)=\sum_{j\in A^i}c_j(\sigma_j(A)).$$

以下定理可以从 Rosenthal (1973) 推导出。

定理 3.1. 每个拥堵博弈都是势博弈。

证明. 设 $\Gamma$ 是由参数 $N$，$M$，$(\Sigma^i)_{i\in N}$，$(c_j)_{j\in M}$ 定义的拥堵博弈。对于每个 $A\in\Sigma$，定义：

$$P(A)=\sum_{j\in\cup_{i=1}^{*}A^{i}}\left(\sum_{l=1}^{\sigma_{j}(A)}c_{j}(l)\right).$$

$P$ 是 $\Gamma$ 的势函数的证明可以从 Rosenthal (1973) 推导出，或直接使用推论 2.9。

设 $\Gamma_{1}$ 和 $\Gamma_{2}$ 是具有相同玩家集合 $N$ 的策略形式博弈。对于 $k$ = 1,2，设 $(Y_k^i)_{i\in N}$ 是 $\Gamma_{k}$ 中的策略集，$(u_k^i)_{i\in N}$ 是 $\Gamma_{k}$ 中的支付函数。我们说 $\Gamma_{1}$ 和 $\Gamma_{2}$ 是同构的，如果存在双射 $g^i$ ：$Y_{1}^{i}\rightarrow Y_{2}^{i}$，$i\in N$，使得对于每个 $i\in N$：

$$\begin{array}{rcl}u_1^i(y^1,y^2,\dots,y^n)&=&u_2^i(g^1(y^1),g^2(y^2),\dots,g^n(y^n))\\&&\text{对于每个}(y^1,y^2,\dots,y^n)\in Y_1,\end{array}$$

其中 $Y_{1}=X_{i\in N}Y_{1}^{i}$。

定理 3.2. 每个有限势博弈都同构于一个拥堵博弈。

证明以及相关讨论在附录 B 中给出。

---

### 4. 无限势博弈

设 $\Gamma$ 是一个策略形式博弈，如第 2 节所述。$\Gamma$ 被称为有界博弈，如果支付函数 $(u^i)_{i\in N}$ 是有界的。

引理 4.1. 每个有界势博弈对于每个 $\varepsilon>0$ 都具有一个 $\varepsilon$ -均衡点。

证明. 注意，由 (2.2)，每个势函数 $P$ 对于 $\Gamma$ 必须是有界的。设 $\varepsilon>0$。存在 $z\in Y$ 满足：

$$P(z)>\sup_{y\in Y}P(y)-\varepsilon.$$

显然 $z$ 是一个 $\varepsilon$ -均衡点。

回顾第 2 节中的路径概念。设 $\varepsilon>0$。路径 $\gamma=(y_{0},y_{1},\ldots)$ 是关于 $\Gamma$ 的 $\varepsilon$ -改进路径，如果对于所有 $k\geq 1$，$u^{i}( y_{k}) > u^{i}( y_{k- 1}) + \varepsilon$，其中 $i$ 是第 $k$ 步的唯一偏离者。博弈 $\Gamma$ 具有近似有限改进性质 (AFIP)，如果对于每个 $\varepsilon>0$，每个 $\varepsilon$ -改进路径都是有限的。下一个引理的证明是显然的，将被省略。

引理 4.2. 每个有界势博弈都具有 AFIP。

注意，对于具有 AFIP 的博弈，特别是有界势博弈，每个最大 $\varepsilon$ -改进路径在 $\varepsilon$ -均衡点终止。博弈 $\Gamma$ 被称为连续博弈，如果策略集是拓扑空间，且支付函数关于乘积拓扑是连续的。注意，由 (2.2)，连续势博弈的势函数是连续的。因此我们有：

引理 4.3. 设 $\Gamma$ 是一个具有紧策略集的连续势博弈。那么 $\Gamma$ 具有一个纯策略均衡点。

我们现在处理可微博弈。我们假设讨论的策略集是实数的区间。我们省略下一个引理的显然证明。

引理 4.4. 设 $\Gamma$ 是一个策略集为实数区间的博弈。假设支付函数 $\iota\iota^{i}$ ：$Y^{i}\to R$，$i\in N$，是连续可微的，且 $P$ ：$Y\rightarrow R$。那么 $P$ 是 $\Gamma$ 的势函数，当且仅当 $P$ 是连续可微的，且：

$$\frac{\partial u^i}{\partial y^i}=\frac{\partial P}{\partial y^i}\quad \text{对于每个}\:i\in N.$$

下一个定理是众所周知的（且非常有用）：

---

定理 4.5. 设 $\Gamma$ 是一个策略集为实数区间的博弈。假设支付函数是二次连续可微的。那么 $\Gamma$ 是势博弈，当且仅当：

$$\frac{\partial^2u^i}{\partial y^i\partial y^j}=\frac{\partial^2u^j}{\partial y^i\partial y^j}\quad \text{对于每个}\:i,j\in N.$$

此外，如果支付函数满足 (4.1)，且 $\zeta$ 是 $Y$ 中的任意（但固定）策略配置，则 $\Gamma$ 的势函数由下式给出：

$$\begin{aligned}P(y)=\sum_{i\in N}\int_0^1\frac{\partial u^i}{\partial y^i}(x(t))(x^i)'(t)\:dt,\end{aligned}$$

其中 $A$ $x\colon[0,1]\to Y$ 是连接 $z$ 到 $y$ 的分段连续可微路径（即 $x(0)=z$ 和 $x(1)=y$）。

考虑引言中描述的准 Cournot 博弈。可以很容易地验证 (4.1) 满足（因为 $\partial ^{2}u^{i}/ \partial y^{i}\partial y^{j}$ = $a$ 对于每个 $i\neq j\in N$），并且应用 (4.2) 得出 (1.2) 中给出的势函数。与（加权）势博弈不同，序数势博弈不容易刻画。我们不知道任何有用的刻画，类似于 (4.1) 中给出的可微序数势博弈。

### 5. 势函数作为均衡细化工具

设 $\Gamma$ 是一个势博弈，$P$ 是 $\Gamma$ 的势函数。所有最大化 $P$ 的策略配置集是均衡集的子集。由引理 2.7，这个集不依赖于特定的势函数。因此，至少在技术上，势函数定义了一个细化概念。考虑 Rouseau 的 Stag Hunt 博弈版本，如 Crawford (1991) 所述：有 $n$ 个玩家。玩家 $i$ 选择 $e_{i}\in\{1,2,\ldots,7\}$。玩家 $i$ 的支付 $\Pi_{i}$ 为：

$$\Pi_i(e_1,e_2,\ldots,e_n)=a\min(e_1,e_2,\ldots,e_n)-be_i+c,$$

其中 $a>b\geq0$，$C$ 是保证正支付的常数。定义势函数 $P$ 为：

$$P(e_1,e_2,\dots,e_n)=a\min(e_1,e_2,\dots,e_n)-b\sum_{j=1}^je_j.$$

> 13 还可以证明，对于加权势博弈，加权势函数的 argmax 集不依赖于特定的加权势函数选择（即使不同的加权势函数可能基于不同的权重集（即，一个权重向量不是另一个权重向量的标量倍数））。

---

注意，如果 $a<nb$，则 $P$ 在配置 $\ell$ 处最大化，其中 $e_i=1$ 对于每个 $1\leq i\leq n$。如果 $a>nb$，则 $P$ 在满足 $e_i=7$ 对于每个 $i$ 的策略配置处最大化。令人惊讶的是，势函数的 argmax 集预测的均衡选择与 Van Huyck 等人 (1990) 的实验结果一致。在实验 A（使用 Crawford 的符号）中，$a=0.2$，$b=0.1$，且 $14\leq n\leq16$。因此 $a<nb$。在实验 $B$ 中，$b$ 被切换为 0，因此 $a>nb$。在实验 $C_{d}$ 和 $C_{f}$ 中，$a=nb$。在这种情况下，每个均衡配置都最大化势函数，因此势函数不能用于预测。实际上，在 $C_{d}$ 中，玩家没有使用特定的均衡配置。在实验 $C_{f}$ 中，除了两个玩家是固定的（而不是随机匹配）外，其他与 $C_{d}$ 相同，玩家倾向于选择 $e_{1}=e_{2}=7$。我们认为，这反映了重复博弈中重复是合作的替代原则。我们并不试图解释势函数的 argmax 集在上述势博弈中预测行为的成功。这可能只是一个巧合。我们希望进一步的实验将进行以测试这个新的细化概念。

Van Huyck 等人 (1991) 进行了另一组关于平均意见博弈的实验。在这些实验中，玩家 $i$ 的支付函数为：

$$\Pi_i(e_1,e_2,\ldots,e_n)=\alpha M-\beta(M-e_i)^2+\gamma,$$

其中 $Cx$，$\beta$ 和 $\gamma$ 是正常数，$M=M(e_{1},e_{2},\ldots,e_{n})$ 是 $(e_{1},e_{2},\ldots,e_{n})$ 的中位数。

可以很容易地看出，这个博弈没有加权势函数，因此我们无法通过势函数方法分析他们的结果。然而，如果中位数函数 $M$ 被均值函数 $A(e_1,e_2,\ldots,e_n)=$ 1 $ln\sum_{i=1}^{m}e_{i}$ 替换，则由定理 4.5，这个博弈确实具有势函数。唯一最大化这个势函数的策略配置是 $e_{i}=7$ 对于每个 $i$。不幸的是，我们不知道任何使用均值函数 $A$ 进行的实验。

### 6. 势理论在价值理论策略方法中的应用

设 $N$ = $\{ 1, 2, \ldots , n\}$ 是玩家集合。对于每个非空联盟 $S\subseteq N$，我们用 $G(S)$ 表示在玩家集合 $S$ 上具有可转移效用的所有合作博弈的空间。即，$v\in G(S)$ 当且仅当 $u$ 是定义在 $2^{5}$ 上的实值函数，且 $v(\emptyset)=0$。解是一个函数 $\psi:\cup_{S\in2^N}G(S)\to\cup_{S\in2^N}R^{S}$，使得 $\psi(v)\in R^{S}$ 当 $v\in G(S)$。解 $\psi$ 是有效的，如果 $\sum_{i\in S}\psi v(i)=v(S)$ 对于每个 $S\in2^{N}$ 和每个 $v\in G(S)$。

对于每个解 $\psi$ 和每个 $c\in R^{N}$，我们将为每个 $v\in G(N)$ 定义一个策略形式博弈 $\Gamma(\psi,c,v)$，如下：

玩家集合为 $N$。玩家 $i$ 的策略集为 $Y^{i}=\{0\}$ 1}。玩家 $i$ 可以选择不加入博弈（选择 0）并获得支付 $c^{i}$，或参与博弈（选择 1）。设 $S$ 是选择 1 的所有玩家的集合。然后每个 $i\in S$ 获得支付 $\psi(v_{S})(i)$，其中 $v_{S}\in G(S)$ 是 $2^{5}$ 的限制。更准确地说，对于 $\varepsilon \in Y= \{ 0$, $1\} ^{N}$，设 $S(\varepsilon)=\{i\in N:\varepsilon^{i}=1\}$。那么玩家 $i$ 的支付函数 $u^{i}$ 为：

$$u^i(\varepsilon)=\left\{\begin{array}{ll}c^i,&\quad\text{如果}\varepsilon^i=0\\\psi(v_{S(\varepsilon)})(i),&\quad\text{如果}\varepsilon^i=1.\end{array}\right.$$

博弈 $\Gamma(\psi,c,v)$ 被称为参与博弈。我们现在提出两个刻画（局部刻画和全局刻画）Shapley 值在参与博弈的策略性质中的特征。

定理 6.1. 设 $\psi$ 是 $G= \cup _{S\in 2^{N}}G( S) , let$ $c\in R^{N}$ 和 $v\in G(N)$ 上的有效解。那么 $\psi$ 是 $\{v_S$ : $S\in2^{N}\}$ 上的 Shapley 值，当且仅当 $\Gamma=\Gamma(\psi,c,v)$ 是势博弈。

证明. 设 $i \in N$。那么：

$$u^i(\varepsilon^{-i},1)-u^i(\varepsilon^{-i},0)=\psi(v_{S\cup\{i\}})(i)-c^i\quad\mathrm{对于所有}\:\varepsilon\in Y,$$

其中 $S=\{j\neq i:\varepsilon^{j}=1\}$。

对于 $S\subseteq N$，设 $\varepsilon_{S}\in Y$ 定义如下：$\varepsilon_{S}^{i}=1$ 如果 $i\in S$，$\varepsilon_{S}^{i}=0$ 如果 $i\not\in S$。

从 (6.1) 我们推导出 $\Gamma$ 是势博弈，当且仅当存在 $Q$ ：$Y\rightarrow R$，使得：

$$Q(\varepsilon_{S})-Q(\varepsilon_{S\setminus\{i\}})=\psi(v_{S\cup\{i\}})(i)-c^{i}\quad\mathrm{对于每个}\:S\subseteq N\mathrm{~和~对于~e}$$

设 $P(\varepsilon_{S})=Q(\varepsilon_{S})+\sum_{i\in S}c^{i}$，则 $Q$ 满足 (6.2) 当且仅当 $P$ 满足：

(6.3)
$$P(\varepsilon_{S})-P(\varepsilon_{S\setminus\{i\}})=\psi(v_{S\cup\{i\}})(i)\quad\mathrm{对于所有}\:S\subseteq N\mathrm{~和~对于~每个}\:i$$

因此，证明由 Hart 和 Mas-Colell (1989) 的定理 A 得出。

定理 6.2. 设 $\psi$ 是 $G=\cup_{S\in2^{N}}G(S)$ 上的有效解，且 $c\in R^{N}$。那么 $\psi$ 是 $G$ 上的 Shapley 值，当且仅当 $\Gamma(\psi,c,v)$ 是势博弈对于每个 $v\in G(N)$。

证明. 证明由定理 6.1 得出。

根据 Hart 和 Mas-Colell (1989) 的定理 5.2，我们还可以证明以下加权 Shapley 值的刻画。


定理 6.3. 设 $\psi$ 是 $G= \cup _{S\in 2^{N}}G( S) , let$ $c\in R^{N}$，$v\in G(N)$，且 $w$ 是正权重向量。那么 $\psi$ 是 $\{v_{S}:S\in2^{N}\}$ 上的 $w$ -Shapley 值，当且仅当 $\Gamma=\Gamma(\psi,c,v)$ 是 $w$ -势博弈。

其他关于非合作势博弈与合作解的关系的结果在 Qin (1992) 中讨论。

### 附录 A

定理 2.8 的证明。显然 $(2)\Longrightarrow(3)\Longrightarrow(4)$。我们证明 (1) $\Longleftrightarrow(2)$ 和 $(4)\Longrightarrow(2)$。

$(1)\Longrightarrow(2)$ 假设 $P$ 是 $\Gamma$ 的势函数。设 $\gamma=(y_{0},y_{1},\ldots,y_{N})$ 是一个闭合路径。那么由 (2.2)：

$$I(\gamma,u)=I(\gamma,(P,P,\ldots,P))=P(y_N)-P(y_0)=0.$$

$(2)\Longrightarrow(1)$ 假设 $I\left(\gamma,u\right)=0$ 对于每个闭合路径 $\gamma$。固定 $z\in Y$。设 $y\in Y$。我们声称，对于每个连接 $Z$ 到 $y$ 的两个路径 $Y1$ 和 $Y_{2}$，$I\left(\gamma_{1},u\right)=$ $I\left(\gamma_{2},u\right)$。实际上，假设 $\gamma _{1}= ( z$, $y_{1}, \ldots , y_{N})$ 和 $\gamma_{2}=(z,z_{1},\ldots,z_{M})$，其中 $y_{N}=z_{M}=$y。设 $\mu$ 是闭合路径 $(\gamma_{1},\gamma_{2}^{-1})$。即：

$$\mu=(z,y_{1},\ldots,y_{N},z_{M-1},z_{M-2},\ldots,z).$$

那么 $I(\mu,u)=0$。因此 $I(\gamma_{1},u)=I(\gamma_{2},u)$。对于每个 $y\in Y$，选择一个路径，比如 $\gamma(y)$，连接 $Z$ 到 $y$。定义 $P(y)=I(\gamma(y),u)$ 对于所有 $y\in Y$。我们继续证明 $P$ 是 $\Gamma$ 的势函数。我们已经证明了：

$$P(y)=I(\gamma,u)\quad\mathrm{对于每个连接}\:z\:到\:y\:的路径\:\gamma.$$

设 $i\in N$，$y^{-i}\in Y^{-i}$，且 $a\neq b\in Y^{i}$。设 $\gamma=(z,y_{1},\ldots,(y^{-i},a))$ 是连接 $\angle L$ 到 $(y^{-i},a)$ 的路径。设 $\mu=(z,y_{1},\ldots,(y^{-i},a),(y^{-i},b))$。那么由 (A.1)：

$$(y^{-i},b)-P(y^{-i},a)=I(\mu,u)-I(\gamma,u)=u^{i}(y^{-i},b)-u^{i}(y^{-i},a).$$

因此 $P$ 是 $\Gamma$ 的势函数。

$(4)\Longrightarrow(2)$ 假设 $I(\gamma,u)=0$ 对于每个长度为 4 的简单闭合路径 $\gamma$。我们表示闭合路径 $\gamma$ = $( y_{0}, y_{1}, \ldots , y_{N})$ 的长度为 $l( \gamma )$ (= $N)$。假设对于某个闭合路径，比如 $\gamma$，$I\left(\gamma,u\right)\neq0$。显然 $N=l(\gamma)\geq$ 5。不失一般性，我们可以假设 $I\left(\mu,u\right)=0$，当 $l(\mu)<N$ 时。

假设 $\gamma=(y_{0},y_{1},y_{2},\ldots,y_{N})$。设 $i(j),0\leq j\leq N-1$ 是第 $j$ 步的唯一偏离者。即，$y_{j+1}=(y_{i}^{-i(j)},x(i(j)))$，其中 $x(i(j))\neq y_{i}^{i(j)}$。不失一般性，假设 $i(0)=1$。由于 $i(0)=1$，且 $y_{N}=y_{0}$，存在 $1\leq j\leq N-1$ 使得 $i(j)=1$。如果 $j=1$ 或 $j=N-1$，我们通过以下方式与 $\gamma$ 长度的最小性假设矛盾：假设 $i(1)=1$。定义 $\mu=(y_{0},y_{2},\ldots,y_{N})$。那么，$I(\mu,u)=I(\gamma,u)$ 且 $l(\mu)<N$。因此假设 $2\leq j\leq N-2$。我们证明存在 $z_{j}\in Y$ 使得路径 $\mu=(y_{0},y_{1},\ldots,y_{j-1},z_{j},y_{j+1},\ldots,y_{N})$ 满足：

$$I(\mu,u)=I(\gamma,u)\quad\mathrm{且}\quad i(j-1)=1.$$

实际上，定义：

$$z_j=(y_{j-1}^{-\{i(j-1),1\}},y_{j-1}^{i(j-1)},y_{j+1}^{1}).$$

那么，由我们对长度为 4 的闭合路径的假设：

$$I((y_{j-1},y_j,y_{j+1},z_j),u)=0.$$

这意味着 (A.2) 成立。

继续递归地，我们最终找到一个长度为 $N$ 的闭合路径 $\tau$，使得 $I\left(\tau,u\right)\neq0$，且 $i(0)=i(1)=1$，与最小性假设矛盾。我们得出结论，$I\left(\gamma,u\right)=0$ 对于每个闭合路径 $\gamma$。

### 附录 B

拥堵博弈中的支付函数在 (3.1) 中给出。我们需要一个等价公式来证明定理 3.2。对于 $A=(A^{1},A^{2},\ldots,A^{n})$ $\in\Sigma$ 和 $S\subseteq N$，我们表示 $A(S)=\cup_{i\in S}A^{i}$，$A(-S)=A(S^{c})$，其中 $S^{c}$ 是 $S$ 的补集。对于 $S=\{i\}$，$A(i)$ 和 $A(-i)$ 分别表示 $A(\{i\})$ 和 $A(-\{i\})$。对于 $x\in R^{M}$ 和 $B\subseteq M$，我们表示 $x(B)=\sum_{j\in B}x(j)$。

引理 B.1. 假设 $C$ 是如第 3 节所述的拥堵博弈。对于每个 $r\in N$，定义向量 $x^r\in R^M$ 为：

$$x^r(j)=c_j(m)\quad \text{对于每个}\:j\in M.$$

那么对于每个 $i\in N$ 和每个 $A\in\Sigma$：

$$
\begin{aligned}
v^{i}(A)\:=& x^{1}\left(A(i)\cap A(-i)^{c}\right)  \\
&+x^{2}\left(\cup_{k\neq i}[A(i)\cap A(k)\cap A(-\{i,k\})^{c}]\right) \\
&+\cdots+x^{n}\left(\cap_{k\in N}A(k)\right).
\end{aligned}
$$

证明. 证明由 (3.1) 得出。

---

定理 3.2 的证明. 设 $\Gamma$ 是如第 2 节所述的有限势博弈。玩家集合为 $N=\{1,2,\ldots,n\}$，策略集为 $(Y^i)_{i\in N}$，支付函数为 $(u^i)_{i\in N}$。设 $P$ 是 $\Gamma$ 的势函数。设 $k(i)=\#Y^{i}$ 是玩家 $i$ 的策略数，假设：

$$Y^i=\{a_1^i,a_2^i,\ldots,a_{k(i)}^i\}.$$

对于 $i\in N$，设 $K(i)=\{1,2,\ldots,k(i)\}$，设 $K=X_{i=1}^{n}K(i)$。

我们继续定义一个同构的拥堵博弈。设施集 $M$ 定义为所有 $\varepsilon=(\varepsilon^{1},\varepsilon^{2},\ldots,\varepsilon^{n})$ 的集合，其中对于每个 $i\in N$，$\varepsilon^{i}$ 是长度为 $k(i)$ 的 0 和 1 的向量。即，$\varepsilon ^{i}\in \{ 0$, $1\} ^{K( i) }$。换句话说：

$$M=\times_{i=1}^{n}\{0,1\}^{K(i)}.$$

拥堵博弈中的策略集 $(\Sigma^{i})_{i\in\mathcal{N}}$ 定义为：

$$\Sigma^{i}=\{A_{1}^{i},A_{2}^{i},\ldots,A_{k(i)}^{i}\}\quad\mathrm{对于每个}\:i\in N\:,$$

其中：

$$A_{l}^{i}=\{\varepsilon\in M:\varepsilon_{l}^{i}=1\}\quad\mathrm{对于每个}\:l\in K(i).$$

我们现在定义向量 $\left(x^{r}\right)_{r\in N}$ 在 $R^{M}$ 中，使得由引理 B.1 定义的支付函数 $(v^i)_{i\in N}$ 满足：

$$\begin{aligned}v^{i}(A_{m_{1}}^{1},A_{m_{2}}^{2},\ldots,A_{m_{s}}^{n})&=\:u^{i}(a_{m_{1}}^{1},\:a_{m_{2}}^{2},\ldots,a_{m_{n}}^{n}),\\&\forall i\in N\:\mathrm{且}\:\forall(m_{1},m_{2},\ldots,m_{n})\in K.\end{aligned}$$

对于 $1<r<n$，设 $x^{\prime}=0$。

对于 $r=n$，$x^{n}$ 定义为以下方程组的解：

$$\cap A_{m_{2}}^{2}\cap\cdots\cap A_{m_{\pi}}^{n})=P(a_{m_{1}}^{1},a_{m_{2}}^{2},\ldots,a_{m_{\pi}}^{n}),\quad(m_{1},m_{2},\ldots,m_{n})\in $$

我们必须证明 (B.2) 的解存在。对于每个 $m=(m_{1},m_{2},\ldots,m_{n})\in$ $K$，设 $\varepsilon(m)\in M$ 定义如下：$\varepsilon_{m_{i}}^{i}=1$ 对于每个 $i\in N$，且 $\varepsilon_{k}^{i}=0$ 对于每个 $i\in N$ 和每个 $k\neq m_{\mathrm{r}}$ 在 $K(i)$ 中。设：

$$M_1=\{\varepsilon(m)\colon m\in K\}.$$

注意，对于 $m\neq l\in K$，$\varepsilon(m)\neq\varepsilon(l)$。因此我们可以定义 $x^{n}$ 为：

$$x^n(\varepsilon)=\left\{\begin{array}{ll}P(a_{m_1}^1,a_{m_2}^2,\ldots,a_{m_n}^n),&\quad\text{如果}\varepsilon=\varepsilon(m)\in M_1\\0,&\quad\text{如果}\varepsilon\not\in M_1.\end{array}\right.$$

---

可以很容易地验证，对于每个 $m\in K$：

$$A_{m_{1}}^{1}\cap A_{m_{2}}^{2}\cap\cdots\cap A_{m_{n}}^{n}\cap M_{1}=\{\varepsilon(m)\}.$$

因此 $x^{n}$ 满足 (B.2)。

我们继续定义 $x^{1}$。注意，由 (2.2)，对于每个 $i\in N$ 和每个 $a^{-i}\in Y^{-i}$，表达式 $u^{i}(a^{-i},a^{i})-P(a^{-i},a^{i})$ 不依赖于 $a^{i}\in Y^{i}$。即：

$$u^{i}(a^{-i},a^{i})-P(a^{-i},a^{i})=u^{i}(a^{-i},b^{i})-P(a^{-i},b^{i})\quad\mathrm{对于每个}\:a^{i},$$

对于每个 $i\in N$，定义 $Q^{-i}$ ：$Y^{-i}\to R$ 为：

$$Q^i(a^{-i})=u^i(a^{-i},a^i)-P(a^{-i},a^i),$$

其中 $a^{\dot{r}}$ 是任意从 $Y^{i}$ 中选择的。

对于每个 $i\in N$ 和每个 $m^{i}=(m_{k}^{i})_{k\neq i}\in K^{-i}$，定义 $\varepsilon(m^{i})\in M$ 为：

$\varepsilon_{s}^{i}=1$ 对于每个 $s\in K(i)$，且对于每个 $k$，$k\neq i,\varepsilon_{s}^{k}=0$ 当且仅当 $s=m_k^i$。

设：

$$M_2=\{\varepsilon(m^i)\colon m^i\in K^{-i}\}.$$

定义 $x^{1}$ 为：

$$x^1(\varepsilon)=\left\{\begin{array}{ll}Q^i\left((a_{m_i}^k)_{k\neq i}\right),&\quad\text{如果}\varepsilon\in M_2\:\text{且}\varepsilon=\varepsilon(m^i)\\0,&\quad\text{如果}\varepsilon\not\in M_2.\end{array}\right.$$

$(A_{m_{1}}^{1},A_{m_{2}}^{2},\ldots,A_{m_{n}}^{n})\in\Sigma$ 对于每个 $m$ = $( m_{1}, m_{2}, \ldots , m_{n})$ $\in$ $K$ 和 $A=$

$$^{-1}\left(A_{m_{i}}^{i}\cap A(-i)\right)=x^{1}(\varepsilon(m^{i}))=u^{i}(a)-P^{i}(a)\quad\mathrm{对于每个}\:i\in N,$$

其中 $m^{i}=(m_{k})_{k\neq i}$ 和 $a=(a_{m_{1}}^{1},a_{m_{2}}^{2},\ldots,a_{m_{\pi}}^{n})$。结合 (B.6), (B.2), 和引理 B.1，我们得到对于每个 $i\in N$：

$$v^{i}(A_{m_{1}}^{1},A_{m_{2}}^{2},\ldots,A_{m_{n}}^{n})=u^{i}(a_{m_{1}}^{1},a_{m_{2}}^{2},\ldots,a_{m_{n}}^{n}),\quad\forall(m_{1},m_{2},\ldots,n$$

我们以关于表示势博弈所需的最少设施数量的评论结束本附录。

设玩家数量 $n$ 和策略集 $(Y^i)_{i=1}^n$ 固定。那么所有具有 $n$ 个玩家和策略集 $(Y^i)_{i=1}^n$ 的势博弈的线性空间的维度 $d$ 为：

$$d=\frac{k}{k(1)}+\frac{k}{k(2)}+\cdots+\frac{k}{k(n)}+k-1,$$

其中对于每个 $i\in N$，$k(i)=\#Y^{i}$ 且 $k=k(1)k(2)\cdots k(n)$。

假设我们正在寻找一个固定的设施集 $M$，具有 $m$ 个元素，以及固定的策略集 $(\Sigma^{i})_{i\in N}$，具有 $\#\Sigma^{i}=k(i)$ 对于每个 $i\in N$，使得每个势博弈将由具有 $n$ 个玩家、设施集 $M$ 和策略集 $(\Sigma^{i})_{i\in N}$ 的拥堵博弈表示。那么由引理 B.1，每个这样的拥堵博弈由 $n$ 个向量 $(x^i)_{i\in N}$ 在 $R^{M}$ 中唯一定义。假设我们还希望表示操作是线性的，那么我们必须有：

$$m\ge\frac{1}{n}\left(\frac{k}{k(1)}+\frac{k}{k(2)}+\cdots+\frac{k}{k(n)}+k-1\right).$$

在定理 3.2 的证明中，$m=2^{k(1)+k(2)+\cdots+k(n)}$。然而，我们可以将设施集定义为 $M_{1}$ 或 $M_{2}$（元素较多的那个）。因此，设施数量 $m$ 可以减少到：

$$m=\max\left(k,\frac{k}{k(1)}+\frac{k}{k(2)}+\cdots+\frac{k}{k(n)}\right).$$

比较 (B.7) 和 (B.8) 表明，我们可能可以改进我们的结果。

### 参考文献

Bergstrom, C., 和 Varian, H. R. (1985). 关于 Cournot 均衡的两个评论. 经济学快报. 19, 58 Blume, L. E. (1993). \*战略互动的统计力学. 博弈与经济行为. 5387424.

Brown, G. W. (195 1).“通过虚构博弈迭代求解博弈," 生产与分配的活动分析. 纽约: Wiley.

Crawford, V. P. (1991). “Van Huyck, Battalio, 和 Beil 实验结果的进化解释." 博弈与经济行为. 3, 2559

Deschamps, R. (1973). 博士论文. 鲁汶大学

Fudenberg, D., 和 Kreps, D. (1993). *学习, 混合均衡, 博弈与经济行为. 5, 320367 Fudenberg, D., 和 Levine, D. K. (1994). *一致性与谨慎虚构博弈,” 未发表.

Garcia, C. B., 和 Zangwill, W. I. (1981). “通往解, 固定点, 和均衡的路径," 纽约: Prentice Hall.

Hart, S., 和 Mas-Colell, A. (1989). 势, 价值, 与一致性, 计量经济学. 57, 589614 Hofbauer, J. (1994). “最佳响应动态的稳定性, 未发表

Jordan, J. S. (1993). “学习混合策略纳什均衡的三个问题," 博弈与经济行为.5.368386

---

Kandori, M., 和 Rob, R. (1992). *长期均衡的进化: 一般理论与应用," 未发表. Krishna, V. (1991). *具有战略互补性的博弈中的学习,* 未发表 Krishna, V., 和 Sjostrom. (1994). “虚构博弈的收敛速度,” 未发表 Milchtaich, I. (1996). 具有玩家特定支付函数的拥堵博弈, 博弈与经济行为. 13, 111124. Milgrom, P., 和 Roberts, J. (1991). “正常形式博弈中的自适应与复杂学习, 博弈与经济行为. 3, 82100 Miyasawa, K. (1961). “关于 $2\times2$ 非零和两人博弈中学习过程的收敛性, 普林斯顿大学经济研究计划, 研究备忘录第 33 号 Monderer, D., 和 Sela, A. (1992). “虚构博弈与无循环条件," 未发表 Monderer, D., Samet, D., 和 Sela, A. (1994). “学习过程中的信念确认,” 未发表. Monderer, D., 和 Shapley, L. S. (1996). “具有相同利益的博弈的虚构博弈性质, 经济理论杂志. 1, 258265 Neyman, A. (1991). “相关均衡与势博弈,” 未发表 Qin, C-Z. (1992). “关于合作结构内生形成的势博弈,” 未发表 Robinson, J. (1951). “求解博弈的迭代方法," 数学年刊. 54, 296301 Rosenthal, R. W. (1973). "一类具有纯策略纳什均衡的博弈," 国际博弈论杂志. 2, 6567. Roth, A. E., 和 Erev, I. (1995). *扩展形式博弈中的学习: 实验数据与简单动态模型的中期结果, 博弈与经济行为. 8, 164212 Sela, A. (1992). *博弈中的学习过程,” 硕士论文. 以色列理工学院, 海法, 以色列. [希伯来语] Shapley, L. S. (1964). *两人博弈中的一些主题," 博弈论进展 (M. Dresher L. S. Shapley, 和 A. W. Tucker, Eds.), pp. 128, 普林斯顿, NJ: 普林斯顿大学出版社 Shin, H. S., 和 Williamson, T. (1994). “一个约定需要多少共同信念, 未发表 Slade, M. E. (1993). "寡头垄断最大化什么? 工业经济学杂志, 即将出版 Van Huyck, J., Battalio, R., 和 Beil, R. (1990). “战术协调博弈, 战略不确定性, 与协调失败, 美国经济评论. 80, 234248 Van Huyck, JL., Battalio, R., 和 Beil, R. (1991). “战略不确定性, 均衡选择原则, 与平均意见博弈中的协调失败, 经济学季刊, 885910 Young, H. P. (1993). “约定的进化, 计量经济学, 61, 5784

# Deepseek 总结

### 文章主要内容与主要结论总结

#### 1. **引言与背景**

- **背景**：本文由 Dov Monderer 和 Lloyd S. Shapley 撰写，主要研究 **势博弈（Potential Games）** 的概念及其在博弈论中的应用。势博弈是一类具有特定结构的博弈，其均衡性质可以通过一个全局的势函数来描述。
- **动机**：势博弈的概念最早由 Rosenthal (1973) 提出，用于分析拥堵博弈（Congestion Games）。本文扩展了这一概念，提出了**序数势博弈（Ordinal Potential Games）**和**精确势博弈（Exact Potential Games）**，并探讨了它们在经济学中的应用。

#### 2. **势博弈的定义与性质**

- **序数势博弈**：一个博弈被称为序数势博弈，如果存在一个势函数 $P$，使得对于每个玩家 $i$，当且仅当 $P$ 增加时，玩家的支付函数 $u^i$ 也增加。序数势博弈的均衡集与势函数的均衡集一致。
- **精确势博弈**：精确势博弈是序数势博弈的加强版，要求势函数 $P$ 的变化与玩家支付函数的变化完全一致。精确势博弈的势函数在混合策略均衡中也保持一致。
- **主要结论**：
  - **引理 2.1**：序数势博弈的均衡集与势函数的均衡集一致。
  - **推论 2.2**：每个有限序数势博弈都具有纯策略均衡。
  - **引理 2.3**：每个有限序数势博弈都具有有限改进性质（FIP），即任何改进路径都是有限的。

#### 3. **势博弈与拥堵博弈的关系**

- **拥堵博弈**：拥堵博弈是一类特殊的博弈，玩家选择资源（如道路），支付函数取决于选择该资源的玩家数量。Rosenthal (1973) 证明了每个拥堵博弈都是势博弈。
- **主要结论**：
  - **定理 3.1**：每个拥堵博弈都是势博弈。
  - **定理 3.2**：每个有限势博弈都同构于一个拥堵博弈。这意味着势博弈与拥堵博弈在结构上是等价的。

#### 4. **无限势博弈**

- **无限势博弈**：本文还讨论了无限策略集的势博弈，特别是当策略集是连续的或可微的情况。
- **主要结论**：
  - **引理 4.1**：每个有界势博弈对于每个 $\varepsilon > 0$ 都具有 $\varepsilon$-均衡点。
  - **引理 4.3**：具有紧策略集的连续势博弈具有纯策略均衡点。
  - **定理 4.5**：对于策略集为实数区间的博弈，若支付函数是二次连续可微的，则博弈是势博弈当且仅当支付函数的交叉偏导数相等。

#### 5. **势函数作为均衡细化工具**

- **势函数的均衡细化**：势函数可以用来细化均衡集，特别是在实验经济学中，势函数的 argmax 集可以预测玩家的行为。
- **主要结论**：
  - **实验验证**：Van Huyck 等人 (1990) 的实验结果表明，势函数的 argmax 集能够准确预测玩家的均衡选择。例如，在 Stag Hunt 博弈中，势函数的最大化点与实验结果一致。

#### 6. **势博弈在合作博弈中的应用**

- **合作博弈中的势理论**：本文还探讨了势博弈在合作博弈中的应用，特别是与 Shapley 值的关系。
- **主要结论**：
  - **定理 6.1**：一个解 $\psi$ 是 Shapley 值，当且仅当参与博弈 $\Gamma(\psi,c,v)$ 是势博弈。
  - **定理 6.2**：解 $\psi$ 是 Shapley 值，当且仅当对于每个 $v \in G(N)$，参与博弈 $\Gamma(\psi,c,v)$ 是势博弈。

#### 7. **附录**

- **附录 A**：证明了定理 2.8，即势博弈的等价条件。
- **附录 B**：证明了定理 3.2，即每个有限势博弈都同构于一个拥堵博弈。

### 主要结论总结

1. **势博弈的定义与性质**：势博弈是一类具有全局势函数的博弈，其均衡性质可以通过势函数来描述。序数势博弈和精确势博弈分别对应不同的势函数条件。
2. **势博弈与拥堵博弈的等价性**：每个拥堵博弈都是势博弈，且每个有限势博弈都同构于一个拥堵博弈。
3. **无限势博弈的均衡存在性**：在无限策略集的情况下，有界势博弈和连续势博弈仍然具有均衡点。
4. **势函数作为均衡细化工具**：势函数的 argmax 集可以用来细化均衡集，并且在实验中能够准确预测玩家的行为。
5. **势博弈在合作博弈中的应用**：势博弈与 Shapley 值之间存在密切联系，参与博弈是势博弈当且仅当解是 Shapley 值。

### 总结

本文系统地研究了势博弈的概念及其在博弈论中的应用，提出了序数势博弈和精确势博弈的定义，并证明了势博弈与拥堵博弈的等价性。文章还探讨了势博弈在无限策略集、实验经济学和合作博弈中的应用，展示了势函数在均衡分析和预测中的强大工具性。
