# Molecular dynamics simulations

- [Molecular-Dynamics-Simulation: 樊哲勇](https://github.com/brucefan1983/Molecular-Dynamics-Simulation)

## 理论基础

### 牛顿力学

> TODO

### 分析力学

> TODO

### 热力学

- 体系温度与原子平均平动动能的关系
- 热力学第一定律
- 热力学第二定律
- 热力学函数和关系

#### 体系温度与原子平均平动动能的关系

> 温度是对大量原子（分子）热运动剧烈程度的度量，即原子平均平动动能的量度。

$$
\frac{3}{2}k_{\rm B}T = \frac{1}{2} m \langle\mathbf{v}^2\rangle
$$

#### 热力学第一定律

热力学第一定律表明，在一个过程中，系统内能的增加量 $\Delta E$ 等于环境对系统做的功 $W$ 和传给系统的热 $Q$ 的和：

$$
\Delta E = Q + W.
$$

如果系统对环境做功，则约定 $W<0$ ；如果系统传给环境热量，则约定 $Q<0$ 。对于无限小过程，热力学第一定律可写成

$$
\text{d} E = \delta Q + \delta W.
$$

内能是一个热力学体系所包含的能量，其数值无绝对意义，可被定义为将体系从某个标准状态变为当前状态所需能量。内能不包括与体系总体运动相关的动能及与体系总体位置相关的势能。功和热量都不是状态量，而是过程量，依赖于具体过程。从数学角度来说，状态量的微分是恰当微分，而过程量的微分则不是。所以，为了区分，用 $\text{d}E$ 表示内能的微分，用 $\delta W$ 和 $\delta Q$ 表示微小的功和热量。

系统在吸热时温度一般会升高。因为热量与过程有关，所以将一个系统的温度升高一定的值所需的热量依赖于系统所经历的过程。指定一个过程，可定义热容：

$$
C = \frac{\delta Q}{\text{d} T}.
$$

常见的两个过程是等容过程和等压过程，对应的热容分别为等容热容和等压热容。
如果体积固定，系统与外界互不做功，由热力学第一定律可知 $\delta Q = \text{d} E$，故等容热容可表达为

$$
C_V =\left(\frac{\delta Q}{\text{d}T}\right)_V
= \left(\frac{\partial E}{\partial T}\right)_V.
$$

如果压强固定（体积不固定），系统要对环境做功 $pdV$ ，由热力学第一定律可知 $\delta Q = \text{d} E + p \text{d} V = \text{d} (E + p V)$，故等压热容可表达为

$$
C_p = \left(\frac{\delta Q}{\text{d}T}\right)_p = \left(\frac{\partial H}{\partial T}\right)_p.
$$

其中，我们定义了一个类似内能的热力学函数，焓：

$$
H = E + p V.
$$

综上可知，等容过程中系统吸收的热量等于其内能的增加量；等压过程中系统吸收的热量等于其焓的增加量。

如果在一个过程中，系统与环境没有热交换，那么该过程被称为绝热过程。对理想气体来说，容易证明，绝热过程可由下式描述：

$$
p V^{\gamma} = \text{常数}.
$$

其中，

$$
\gamma \equiv \frac{C_p}{C_V}
$$

是等压热容和等容热容之比，称为绝热指数。

循环过程是指系统终态等于初态的过程。在 $p-V$ 图中，循环过程对应于一个闭合路径。若闭合路径为顺时针方向，则系统对环境做净功并从环境吸净热，对应于热机；反之，环境对系统做净功并从系统吸净热，对应于热泵或制冷机。

理论上最重要的循环过程为理想气体的卡诺 (Carnot) 循环，由等温膨胀（高温 $T_{1}$）、绝热膨胀、等温压缩（低温 $T_{2}$）、绝热压缩四个过程组成。高温和低温由环境中的热浴来保持。根据热力学第一定律，对于卡诺热机，系统在等温膨胀过程中从温度为 $T_{1}$ 的高温热源吸热，并在等温压缩过程中向温度为 $T_{2}$ 的低温热源放热。系统吸入和放出的热量分别记为 $Q_{1}$ 和 $|Q_{2}|$ ( $Q_{2} < 0$ )，并定义热机的效率 $\eta$ 为系统所做净功与从高温热源所吸热量之比：

$$
\eta = \frac{ Q_{1} - |Q_{2}| } { Q_{1} }.
$$

容易证明，该效率只与温度有关且总小于 100\%：

$$
\eta = \frac{ T_{1} - T_{2} } { T_{1} }
= 1- \frac{ T_{2} } { T_{1} }.
$$

为什么热机的效率总小于 100\%？或者说，为什么系统不能把吸收的热量皆转化为功？这是热力学第一定律无法回答的问题。要回答这个问题，我们需要学习热力学第二定律。

#### 热力学第二定律 (Second Law of Thermodynamics): 熵增原理

- 克劳修斯表述
  - 无法将热能从低温物体转移到高温物体而不对环境产生影响。
- 开尔文表述
  - 无法从单一热源取热使其完全转化为有用功而不对环境产生影响。
- 熵增原理表述
  - 孤立系统的熵永远不会自发减小，在可逆过程作用下熵保持不变，不可逆过程中熵一定会增加。

#### 热力学函数和关系

欧拉方程

$$
E = TS  - pV  + \mu N.
$$

### 经典统计力学

> TODO

#### 统计系综和统计分布函数

#### 微正则系综

#### 正则系综

#### 正则系综在理想气体体系中的应用

## Sim

基本要素

- 初始条件
- 边界条件
- 相互作用
- 运动方程的数值积分
- 物理量计算

在本章讨论的分子动力学模拟中，没有外界对系统的干扰，所有粒子的运动完全由粒子间的相互作用力决定。从经典力学的角度看，这样的体系对应哈密顿体系。而从经典统计力学的角度看，这样的体系则属于微正则系综，即粒子数 N、体积 V 和能量 E 保持恒定的 NVE 系综。

一个典型的简单分子动力学模拟流程如下：

1. 初始化：设置系统的初始条件，包括每个粒子的位置和速度。
2. 时间演化：根据粒子间的相互作用规律，确定所有粒子的运动方程（即二阶常微分方程组），并通过数值积分不断更新每个粒子的坐标和速度。最终，我们得到系统在不同时间点上的离散相空间位置，即一条离散的相轨迹。
3. 测量：利用统计力学的方法分析相轨迹中所包含的物理规律。

### 0 单位制

4 个基本单位

- 能量：电子伏特（记号为 eV），约为 $1.602177\times 10^{-19}$ J。
- 长度：埃（angstrom，记号为 $\AA$），即 $10^{-10}$ m。
- 质量：道尔顿（统一原子质量单位，记号为 Da），约为 $1.660539 \times 10^{-27}$ kg。
- 温度：开尔文（记号为 K）。

推导出其他相关物理量的单位

- 力。因为力乘以距离等于功（能量），故力的单位是能量单位除以长度单位，即 $\text{eV} \AA^{-1}$。
- 速度。因为动能正比于质量乘以速度的平方，故速度的单位是能量单位除以质量单位再开根号，即 $\text{eV}^{1/2}$ Da$^{-1/2}$。
- 时间。因为长度等于速度乘以时间，故时间的单位是长度单位除以速度单位，即$\AA \ \text{Da}^{1/2} \text{eV}^{-1/2}$，约为 $1.018051 \times 10^{1}$ fs（fs 指飞秒，即 $10^{-15}$ s）。
- 玻尔兹曼常数 $k _{\rm B}$。这是一个很重要的常数，它在国际单位制中约为 $1.380649\times 10^{-23}\text{J K}^{-1}$，对应于程序自然单位制的 $8.617343 \times 10^{-5}\text{eV} \text{K}^{-1}$。

### 1. 初始化

初始化是指确定初始的相空间点，包括各个粒子初始的坐标和速度。

坐标初始化是指为系统中每个粒子选定一个初始的位置坐标。分子动力学模拟中如何初始化位置主要取决于所要模拟的体系。例如，如要模拟固态氩，就得让各个氩原子的位置按面心立方结构排列。

最简单的速度初始化方法是产生 3N 个在某区间均匀分布的随机速度分量，再通过基本条件对其修正：

#### 系统的总动量应该为零

也就是说，我们不希望系统的质心在模拟的过程中跑动。分子间作用力是所谓的内力，不会改变系统的整体动量，即系统的整体动量守恒。只要初始的整体动量为零，在分子动力学模拟的时间演化过程中整体动量将保持为零。

如果整体动量明显偏离零（相对所用浮点数精度来说），则说明模拟出了问题。这正是判断模拟是否有误的标准之一。

#### 系统的总动能应该与所选定的初始温度对应

我们知道，在经典统计力学中，能量均分定理成立，即粒子的哈密顿量中每一个具有平方形式的能量项的统计平均值都等于 $k_BT /2$。

其中，$k_B$ 是玻尔兹曼常数，$T$ 是系统的绝对温度。所以，在将质心动量取为零后就可对每个粒子的速度进行一个标度变换，使得系统的初温与设定值一致。假设我们设置的目标温度是 $T_0$，那么对各个粒子的速度做如下变换即可让系统的温度从 $T$ 变成 $T_0$：

$$
v_{i} \rightarrow v_{i}^{\prime}=v_{i} \sqrt{\frac{T_{0}}{T}}
$$

容易验证，在做上式中的变换之前，如果系统的总动量已经为零，那么在做这个变换之后，系统的总动量也为零。

#### 系统的总角动量应该为零，但这是可选条件

这是因为，对于施加周期边界条件（见下面的讲解）的体系，系统的总角动量不守恒，故初始总角动量即使非零也无妨。

如果所模拟的体系为纳米粒子（三个方向都是非周期的）或纳米线（仅一个方向是周期的），则通常需要将初始角动量置零。

### 2. 边界条件

边界条件的选取对粒子间作用力的计算也有影响。常用的边界条件有好几种，但我们这里只先讨论其中的一种：**周期边界条件**。同时在本书的模拟中，总是采用**最小镜像约定**：在计算两粒子距离时，总是取最小的可能值。

在计算两粒子距离时，总是取最小的可能值。定义

$$
x_j-x_i \equiv x _{ij}
$$

### 3. 相互作用 LJ 势

考虑系统中的任意粒子对 $i$ 和 $j$，它们之间的 LJ 相互作用势能可写为

$$
U _{ij}(r _{ij})=4\epsilon
\left(
\frac{\sigma^{12}}{r _{ij}^{12}}-\frac{\sigma^{6}}{r _{ij}^{6}}
\right).
$$

其中， $\epsilon$ 和 $\sigma$ 是势函数中的参数，分别具有能量和长度的量纲； $r _{ij}=|\mathbf{r}_j-\mathbf{r}_i|$ 是两个粒子间的距离。

LJ 势是最早提出的两体势函数之一，较适合描述惰性元素组成的物质。所谓两体势，是指两个粒子 $i$ 和 $j$ 之间的相互作用势仅依赖于它们之间的距离 $r _{ij}$，不依赖于系统中其他粒子的存在与否及具体位置。本章只讨论两体势，后续章节会讨论多体势，即非两体势。对于两体势函数，我们可将整个系统的总势能 $U$ 写为

$$
U=\sum _{i=1}^N U_i;
$$

$$
U_i= \frac{1}{2} \sum _{j \neq i} U _{ij}(r _{ij}).
$$

将以上两式合起来，可写成

$$
U=\frac{1}{2}\sum _{i=1}^N  \sum _{j \neq i} U _{ij}(r _{ij}).
$$

上面的 $U_i$ 可称为粒子 $i$ 的势能。因为

$$
U_{ij}(r_{ij})=U_{ji}(r_{ji}),
$$

故也可将总势能写为如下形式：

$$
U=\sum _{i=1}^N \sum _{j > i} U _{ij}(r _{ij}).
$$

接下来推导 LJ 势中力的表达式：

$$
\mathbf{F}_{i} = \sum_{j \neq i} \mathbf{F}_{ij};
$$

$$
\mathbf{F}_{ij} =
\frac{\partial U_{ij}(r_{ij})}{\partial r_{ij}}
\frac{\mathbf{r}_{ij}}{r_{ij}}.
$$

此处的 $\mathbf{F}_{ij}$代表粒子$j$施加给粒子$i$的力。对于 LJ 势，其表达式可进一步推导为：

$$
\mathbf{F}_{ij} =
\left(
\frac{24 \epsilon \sigma^6} {r_{ij}^8} - \frac{48 \epsilon \sigma^{12}} {r_{ij}^{14}}
\right)\mathbf{r}_{ij}.
$$

显然，牛顿第三定律的强形式成立。

### 4. 运动方程数值积分 Verlet 积分算法

给定一个多粒子体系的初始状态（坐标和速度），根据各个粒子之间的相互作用力就可预测该体系的运动状态，即任意时刻各个粒子的坐标和速度。该预测过程本质上就是对运动方程的数值积分。

## Program

### main 主控函数

1. 从文件中读取模拟参数（步数、时间步长、温度）
2. 从单独的 XYZ 文件中读取原子数据（位置、质量）
3. 根据温度初始化原子速度
4. 启动计时器测量模拟时间
5. 打开一个输出文件，用于写入模拟数据 (thermo.out)
6. 按指定步数运行主模拟循环：
   - 对所有原子应用周期性边界条件 (PBC)
   - 使用 Verlet 算法对位置和速度进行积分
   - 计算作用在每个原子上的力
   - 再次积分更新位置和速度
   - 每 Ns 步输出数据（温度、动能、势能）
7. 停止计时器并打印模拟耗时。

TODO:

- 计算并输出温度/动能/势能
  - 可视化 X-Time 图，分析 X 量随时间的变化规律
- 输出轨迹: 即每一时刻各粒子的坐标和速度
  - 使用 ovito 可视化
  - 输出速度，验证速度是否以及何时满足麦克四位分布
- 验证体系动量守恒，角动量不守恒