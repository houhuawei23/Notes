
# Attention Is Turing Complete

Not yet complete.

[Attention Is Turing Complete](https://dl.acm.org/doi/abs/10.5555/3546258.3546333)

TLDR:

本文证明了 Attention 机制是图灵完备的，本文构造了一个有1层Encoder和3层Decoder的Transformer，用其模拟了图灵机的计算过程（即计算格局的变化）。

## Abstract

Alternatives to recurrent neural networks, in particular, architectures based on self-attention, are gaining momentum for processing input sequences. In spite of their relevance, the computational properties of such networks have not yet been fully explored. We study the computational power of the Transformer, one of the most paradigmatic architectures exemplifying self-attention. We show that the Transformer with hard-attention is Turing complete exclusively based on their capacity to compute and access internal dense representations of the data. Our study also reveals some minimal sets of elements needed to obtain this completeness result.

循环神经网络的替代方案，特别是基于自注意力的架构，在处理输入序列方面正在获得动力。尽管它们具有相关性，但此类网络的计算特性尚未得到充分探索。我们研究Transformer的计算能力，它是体现自我注意力的最具范式的架构之一。我们证明，具有硬注意力的Transformer 完全基于其计算和访问数据内部密集表示的能力，是图灵完备的。我们的研究还揭示了获得这种完整性结果所需的一些最小元素集。

## Introduction

序列到序列神经网络 我们对序列到序列 (seq-to-seq) 神经网络架构感兴趣,我们接下来将正式化。对于某些 d > 0,seq-to-seq 网络 N 接收向量 x∈ Q 的序列 X = (x, . . . . , x) 作为输入,并生成向量 y∈ Q 的序列 Y = (y, . . . . , y) 作为输出。大多数这种类型的架构都需要一个种子向量 s 和一些停止标准来确定输出的长度。后者通常基于特定输出向量的生成,称为序列结束标记。相反,在我们的形式化中,我们允许网络产生一个固定数量的 r ≥ 0 的输出向量。因此,为方便起见,我们将一般的 seq-to-seq 网络视为函数 N,使得值 N (X, s, r) 对应于 Y = (y, y, . . . . , y) 形式的输出序列。通过此定义,我们可以将 seq-to-seq 网络解释为字符串的语言识别器,如下所示。

定义 1 seq-to-seq 语言识别器是一个元组 A = (Σ, f, N, s, F),其中 Σ 是有限字母表,f : Σ → Q 是嵌入函数,N 是 seq-to-seq 网络,s ∈ Q 是种子向量,F ⊆ Q是一组最终向量。我们说 A 接受字符串 w ∈ Σ,如果存在一个整数 r ∈ N,使得 N (f (w), s, r) = (y, . . . , y) 和 y∈ F。


A 接受的语言(用 L(A) 表示)是 A 接受的所有字符串的集合。


我们对识别器施加了两个额外的限制。

- 嵌入函数 f : Σ → Q应该由图灵机以 Σ 大小的多项式时间计算。这涵盖了计算符号的 input 嵌入的两种最典型方法:one-hot 编码和由固定前馈网络计算的嵌入。
- 此外,集合 F 也应该在多项式时间内可识别;给定一个向量 f ,隶属度 F ∈ F 应该由在多项式时间内相对于 f 的大小(以位为单位)工作的图灵机决定。这涵盖了使用固定序列结束向量检查相等性的常用方法。

我们施加这些限制是为了防止通过在 input embedding 或 stopping 条件中编码任意计算来作弊的可能性,同时足够宽容地构建有意义的 embeddings 和 stopping criterions。

图灵机计算 让我们回顾一下,(确定性的)图灵机是 M = (Q, Σ, δ, q, F ) 形式的元组,其中:


seq-to-seq 神经网络架构的图灵完备性

seq-to-seq 神经网络架构的 N 类定义了类 L,该类由使用 N 中的网络的语言识别器接受的所有语言组成。从这些概念中,N 类的图灵完备性的形式化自然而然地随之而来。


定义 2 

如果 L 包含所有可判定语言(即图灵机可识别的所有语言),则 seq-to-seq 神经网络架构的 N 类为图灵完备。


3. The Transformer architecture


在本节中,我们提出了 Transformer 架构的形式化(Vaswani et al., 2017),从函数和参数的一些具体选择中抽象出来。我们的形式化并不是为了产生 Transformer 的有效实现,而是提供一个简单的设置,通过该设置可以以正式的方式建立其数学属性。

Transformer 在很大程度上基于接下来介绍的注意力机制。考虑评分函数 score : Q× Q→ Q 和归一化函数 ρ : Q→ Q,对于 d、n > 0。假设 q ∈ Q,并且 K = (k, . . . , k) 和 V = (v, . . . , v) 是 Q 中元素的元组。由 Att(q, K, V) 表示的 q-attention over (K, V) 是 Q ∈向量 a,定义如下。

$$
\left(s_{1}, \ldots, s_{n}\right)  =\rho\left(\operatorname{score}\left(\boldsymbol{q}, \boldsymbol{k}_{1}\right), \operatorname{score}\left(\boldsymbol{q}, \boldsymbol{k}_{2}\right), \ldots, \operatorname{score}\left(\boldsymbol{q}, \boldsymbol{k}_{n}\right)\right)
$$

$$
\boldsymbol{a} =s_{1} \boldsymbol{v}_{1}+s_{2} \boldsymbol{v}_{2}+\cdots+s_{n} \boldsymbol{v}_{n}
$$

通常,q 称为查询,K 称为键,V 称为值。我们对评分函数没有任何限制,但我们确实对归一化函数施加了一些限制,以确保它在持仓上产生概率分布。我们要求归一化函数满足以下条件:对于每个 x = (x, . . . , x) ∈ Q,有一个函数 f从 Q 到 Q,使得 ρ(x) 的第 i 个分量 ρ(x) 相等


j=1 f(x) 的 f(x) 。我们注意到,例如,可以通过简单地选择 f(x) 作为指数函数 e 来以这种方式定义 softmax 函数,但我们允许其他可能性,接下来我们将解释。

在证明可能性结果时,我们需要选择特定的评分和归一化函数。评分函数的通常选择是由具有输入 (q, k) 的前馈网络定义的非线性函数,有时称为加性注意力 (Bahdanau et al., 2014)。另一种可能性是使用点积 〈q, k〉,称为乘法注意力 (Vaswani et al., 2017)。

我们实际上使用了两者的组合:乘法注意力加上定义为 σ(g(·)) 形式的函数组成的前馈网络,其中 g 是仿射变换,σ 是方程 (1) 中定义的分段线性 S 形激活。对于归一化函数,softmax 是一个标准选项。尽管如此,在我们的证明中,我们使用 hardmax 函数,如果 x是 x 中的最大值,则通过设置 f(x) = 1 来获得,否则设置 f(x) = 0。因此,对于最大值出现 r 次的向量 x,如果 x是 x 的最大值,则 hardmax(x) = ,否则为 hardmax(x) = 0。每当 hardmax 用作归一化函数时,我们都称其为 hard attention。

让我们观察一下,hardmax 的选择对于我们的证明以当前形式工作至关重要,因为它允许模拟 “访问” 向量序列中特定位置的过程。Hard attention 以前专门用于处理图像(Xu et al., 2015;Elsayed et al., 2019),但是,据我们所知,它尚未在自我注意架构的上下文中用于处理序列。有关我们对正结果函数的选择的进一步讨论,请参见第 5 节。按照惯例,对于函数 F : Q→ Q 和序列 X = (x, x, . . . , x),其中 x∈ Q,我们写 F (X) 来表示序列 (F (x), . . . , F (x))。

Transformer 编码器和解码器 

Transformer 的单层编码器是一个参数函数 Enc(θ),其中 θ 是参数,它接收 Q 中向量的序列 X = (x, . . . , x) 作为输入,返回一个序列 Enc(X; θ) = (z, . . . . , z) Q中与 X 长度相同的向量。一般来说,我们认为 θ 中的参数是参数化函数 Q(·)、K(·)、V (·) 和 O(·),它们都从 Q到 Q。然后,单层编码器定义如下

a= Att(Q(x), K(X), V (X)) + x (4)
z= O(a) + a (5)

请注意,在等式 4 中,我们将函数 Q 和 V 分别应用于 X 中的每个条目。在实践中,Q(·)、K(·)、V (·) 通常是指定为维度 (d × d) 矩阵的线性变换,而 O(·) 是前馈网络。+ xand + asummands 通常称为残差连接(He et al., 2016;他等人)。当用作参数的特定函数不重要时,我们只需编写 Z = Enc(X)。

Transformer 编码器被简单地定义为单层编码器(具有独立参数)的重复应用,加上两个最终变换函数 K(·) 和 V (·) 应用于最终层输出序列中的每个向量。因此,L 层 Transformer 编码器由以下递归定义(1 ≤ ' ≤ L−1 且 X= X):

X= Enc(X; θ), K = K(X), V = V (X).(6)

V = V (X) 我们写 (K, V ) = TEnc(X) 来表示 (K, V ) 是 L 层变压器编码器在输入序列 X 上的结果。

解码器

单层解码器类似于单层编码器,但需要额外注意一对外部键值向量 (K, V )。单层解码器的输入是序列 Y = (y, . . . , y) 加上外部对 (K, V ),输出是与 Y 长度相同的序列 Z = (z, . . . . , z) 。在定义解码器层时,我们用 Y 表示序列 (y, . . . , y),为 1 ≤ i ≤ k。该层的输出 Z = (z, . . . . , z) 也被参数化,这次由四个函数 Q(·)、K(·)、V (·) 和 O(·) 从 Q to Q 组成,并且对于每个 1 ≤ i ≤ k 定义如下:

p= Att(Q(y), K(Y), V (Y)) + y (7) 
a= Att(p, K, V ) + p (8) 
z= O(a) + a (9)

请注意,对 (K(Y), V (Y)) 的第一个(自我)关注仅在索引 i 之前考虑 Y 的子序列,并用于生成查询 pto 关注外部对 (K, V)。我们将 Y 和 (K, V ) 上的单解码器层的输出表示为 Dec((K, V ), Y ; θ)。
Transformer 解码器是单层解码器的重复应用,加上一个转换函数 F : Q→ Q应用于解码序列的最终向量。因此,解码器的输出是 Q ∈单个向量 z。正式地,L 层 Transformer 解码器定义为

Y = Dec((K, V ), Y ; θ), z = F (y) (1 ≤ ' ≤ L − 1 和 Y = Y ).(10)


我们使用 z = TDec((K, V ), Y ) 来表示 z 是这个 L 层变压器解码器在输入 Y 和 (K, V) 上的输出。

Transformer 的一个重要限制是 Transformer 解码器的输出总是对应于某些有限字母Γ中的字母编码。从形式上讲,需要存在一个有限字母Γ和一个嵌入函数 g : Γ → Q,使得 Transformer 解码器的最终变换函数 F 将 Q 中的任何向量映射到 Γ 中字母嵌入的有限集 g(Γ) 中的向量。

完整的 Transformer 

Transformer 网络接收一个输入序列 X、一个种子向量 y 和一个值 r ∈ N。它的输出是一个序列 Y = (y, . . . , y),定义为

y= TDec(TEnc(X), (y, y, . . . , y)), 对于 0 ≤ t ≤ r − 1。(11)

$$
\boldsymbol{y}_{t+1}=\operatorname{TDec}\left(\operatorname{TEnc}(\boldsymbol{X}),\left(\boldsymbol{y}_{0}, \boldsymbol{y}_{1}, \ldots, \boldsymbol{y}_{t}\right)\right), \quad for ~ 0 \leq t \leq r-1.
$$
我们将变压器的输出序列表示为 Y = (y, y, . . . . , y) = Trans(X, y, r)。

3.1 比例下的不变性

如上所述,Transformer 网络在捕获语言的能力方面相当弱。这是因为 Transformer 是顺序不变的,即它们无法访问 input 中元素的相对顺序。更正式地说,两个相互排列的 Importing 序列会产生完全相同的输出。这是注意力函数的以下属性的结果:如果 K = (k, . . . . , k), V = (v, . . . , v) 且 π : {1, . . . . , n} → {1, . . . , n} 是排列,则对于每个查询 q,Att(q, K, V ) = Att(q, π(K), π(V ))。




$$
\boldsymbol{a}_{i} =\operatorname{Att}\left(Q\left(\boldsymbol{x}_{i}\right), K(\boldsymbol{X}), V(\boldsymbol{X})\right)+\boldsymbol{x}_{i} 
$$
$$
 \boldsymbol{z}_{i} =O\left(\boldsymbol{a}_{i}\right)+\boldsymbol{a}_{i}
$$


## 4. Transformer 的位置编码的图灵完备性

定理 6 

具有位置编码的 Transformer 网络类是图灵完备的。此外,图灵完备性即使在受限设置中也成立,其中 n ∈ N 的位置嵌入 pos（n） 中唯一的非恒定值是 n、1/n 和 1/n,并且 Transformer 网络具有单个编码器层和三个解码器层。

实际上,这个结果的证明显示了更有力的东西:不仅 Transformers 可以识别图灵机接受的所有语言,即所谓的可识别或可判定语言;它们可以识别所有递归可枚举或半可判定的语言,这些语言 L 存在枚举 L 中所有字符串的 TM。

我们现在提供定理 6 的完整证明。为了可读性,一些中间引理的证明被归入附录。
设 M = （Q, Σ, δ, q, F ） 是一个图灵机,其磁带向右是无限的,并假设特殊符号 # ∈ Σ 用于标记磁带中的空白位置。

我们对 M 在处理 Importing 字符串时的工作原理做出以下假设:

M 从状态 q 开始,指向磁带的第一个读取空白符号 #的单元格。输入将立即写入第一个单元格的右侧。

Q 具有特殊状态 qused 来读取完整的输入。

最初（步骤 0）,M 过渡到状态 q,并将其头部向右移动。

当处于状态时,qit 向右移动,直到读取符号 #。

不接受状态（F 中的状态）没有转换。

很容易证明,每个通用的图灵机都等同于满足上述假设的图灵机。我们证明了可以构建一个 transformer 网络 Transthat 能够在每个可能的输入字符串上模拟 M;或者,更正式地说,L（M ） = L（Trans）。
构造以某种方式参与其中,并使用了几个辅助定义和中间结果。为了便于阅读,我们将构造和证明分为三个部分。我们首先给出我们使用的策略的高级视图。然后,我们详细介绍了实现我们的策略所需的编码器和解码器的架构,最后我们正式证明我们架构的每个部分都可以实际实现。

在 Transwe 的编码器部分接收字符串 w = ss· · ·s.我们首先使用嵌入函数将每个 sas 表示为 one-hot 向量,并为每个索引添加位置编码。编码器产生输出 （K, V）,其中 K= （k, . . . , k） 和 V = （v, . . . , v） 是键和值的序列,使得 v包含沙子的信息 k包含第 i 个位置编码的信息。我们稍后会证明,这允许我们关注每个特定位置,并将每个输入符号从编码器复制到解码器（参见引理 7）。

在 Transwe 的解码器部分模拟 M 在 w = ss· · · ·s.为此,我们定义以下序列（对于 i ≥ 0）:

q:计算步骤 i 中 M 的状态 
s:步骤 i 中 M 头部读取的符号 
v:步骤 i 中 M 写入的符号 
m:步骤 i 中 M 头部在过渡中移动的方向

v = [   q1, s1, x1,
        q2, s2, x2, x3, x4, x5,
        s3, x6, s4, x7
        x8, x9, x10, x11 ],




