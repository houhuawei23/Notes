# Potential Games

- Dov Monderer a\*, Lloyd S. Shapley b
- Games and Economic Behavior

- Doy Monderer: Faculty of Industrial Engineering and Management, The Technion, Haifa 32000, Israei
- Lloyd S.Shapley: Department of Economics and Department of Mathematics, University of California. Los Angeles, California 90024
- Received January 19, 1994
- First version: December 1988. Financial support from the Fund for the Promotion of Research at the Technion is gratefully acknowledged by the first author. E-mail: dov@techunix.technion.ac.il

---

### Abstract:

We define and discuss several notions of potential functions for games in strategic form. We characterize games that have a potential function, and we present a variety of applica tions. Journal of Economic Literature Classification Numbers: C72, C73. @ 1996 Academic Press, Inc.

### 1.INTRODUCTION

Consider a symmetric oligopoly Cournot competition with linear cost func tions $c_{i}\left(q_{i}\right)=cq_{i}$ ， $1\leq i\leq n.$ The inverse demand function, $F(Q)$ ， $Q>0$ , is a positive function (no monotonicity, continuity, or differentiability assumptions on $F$ are needed). The profit function of Firm $i$ is defined on $R_{++}^n$ as

$$\Pi_i(q_1,q_2,\ldots,q_n)=F(Q)q_i-cq_i,$$

where $Q=\sum_{j=1}^{n}q_{j}$ Define a function $P$ ： $R_{++}^{n}\longrightarrow R$

$$P(q_1,q_2,\ldots,q_n)=q_1q_2\cdots q_n(F(Q)-c).$$

For every Firm $i$ and for every $q_{-i}\in R_{++}^{n-1}$ ·

$$\begin{array}{rl}\Pi_i(q_i,q_{-i})-\Pi_i(x_i,q_{-i})>0,&\mathrm{iff}&P(q_i,q_{-i})-P(x_i,q_{-i})>0,\\&&\forall q_i,x_i\in R_{++}.\end{array}$$

---

A function $P$ satisfying (1.1) is called an ordinal potential, and a game that pos sesses an ordinal potential is called an ordinal potential game. Clearly, the pure strategy equilibrium set of the Cournot game coincides with the pure-strategy equilibrium set of the game in which every firm's profit is given by $P$ .A condition stronger than (1.1) is required if we are interested in mixed strategies

Consider a quasi-Cournot competition1 with a linear inverse demand function $F(Q)=a-bQ,a,b>0$ a,b > 0 $a,b>0$ , and arbitrary differentiable cost functions $c_i(q_i)$ $1\leq i\leq n$ Define a function $P^{*}((q_{1},q_{2},\ldots,q_{n}))$ as

$$\begin{aligned}P^{*}((q_{1},q_{2},\ldots,q_{n}))&=\:a\sum_{j=1}^{n}q_{j}-b\sum_{j=1}^{n}q_{j}^{2}-b\sum_{1\leq i<j\leq n}q_{i}q_{j}\\&-\sum_{j=1}^{n}c_{j}(q_{j}).\end{aligned}$$

It can be verified that For every Firm i , and for every $q_{-i}\in R_{+}^{n-1}$

$$\Pi_{i}(q_{i},q_{-i})-\Pi_{i}(x_{i},q_{-i})=P^{*}(q_{i},q_{-i})-P^{*}(x_{i},q_{-i}),\quad\forall q_{i},x_{i}\in R_{+}.$$

A function $P^{*}$ satisfying (1.3) will be called a potential function. 2,3 The equal ities (1.3) imply that the mixed-strategy equilibrium set of the quasi-Cournot game coincides with the mixed-strategy equilibrium set of the game obtained by replacing every payoff function by $P^{*}.$ In particular, firms that are jointly trying to maximize the potential function $P^{*}$ (or the ordinal potential $P$ ) end up in an equilibrium.\_We will prove that there exists at most one potential function (up to an additive constant). This raises the natural question about the economic content (or interpretation) of $P^{*}$ : What do the firms try to jointly maximize?

> 1 Negative prices are possible in this game, though the prices in any nondegenerate equilibrium will be positive.

> 2 In physics, $P^{*}$ is a potential function for $(\Pi_{1},\Pi_{2},\ldots,\Pi_{n})$ if

> $$\frac{\partial\Pi_{i}}{\partial q_{i}}=\frac{\partial P^{*}}{\partial q_{i}}\quad\mathrm{for~every}\:1\leq i\leq n.$$

> If the profits functions are continuously differentiable then this condition is equivalent to (1.3)

> 3 Slade (1993) proved the existence of a function $P^{*}$ satisfying (1.3) for the quasi-Cournot game She called this function a fictitious objective function.

> 4 Every $q^*$ that maximizes $P$ is a pure-strategy equilibrium, but there may be pure-strategy equi librium profiles that are just “local" maximum points, and there may be mixed-strategy equilibriun profiles as well. Therefore, the argmax set of the potential can be used as a refinement tool for potentia games (this issue is discussed in Section 5). Neyman (1991) showed that if the potential function is concave and continuously differentiable, then every mixed-strategy equilibrium profile is pure and. must maximize the potential function. Neyman’s result is related by Shin and Williamson (1994) to the concept of “simple equilibrium outcome” in Bayesian games.

---

We do not have an answer to this question.However, it is clear that the mere existence of a potential function helps us (and the players) to better analyze the game.s In this paper we will prove various properties of potential games, and we will

provide simple methods for detecting them and for computing their potential functions.

To our knowledge, the first to use potential functions for games in strategic form was Rosenthal (1973). Rosenthal defined the class of congestion games and proved, by explicitly constructing a potential function, that every game in this class possesses a pure-strategy equilibrium. The class of congestion games is, on the one hand, narrow, but on the other hand, very important for economics. Any game where a collection of homogeneous agents have to choose from a finite set of alternatives, and where the payoff of a player depends on the number of players choosing each alternative,is a congestion game. We will show that the class of congestion games coincides (up to an isomorphism) with the class of finite potential games.

Recently, much attention has been devoted to several notions of “myopic" learning processes. We show that for generic finite games, the existence of an ordinal potential is equivalent to the convergence to equilibrium of the learning process defined by the one-sided better reply dynamic. The new learning literature raised a new interest in the Fictitious Play process in games in strategic form defined by Brown (1951). It was studied for zero-sum games by Robinson (1951) and for non-zero-sum games by Miyasawa (1961), Shapley (1964), Deschamps (1973), and lately by Krishna (1991), Milgrom and Roberts (1991), Sela (1992), Fudenberg and Kreps (1993), Jordan (1993), Hofbauer (1994), Krishna and Sjostrom (1994),Fudenberg and Levine (1994), Monderer et al. (1994), and others. In Monderer and Shapley (1996) we prove that the Fictitious Play process converges to the equilibrium set in a class of games that contains the finite (weighted) potential games. Milchtaich (1996) analyzed classes of games related to congestion games. His work, as well as that of Blume (1993), indicates that ordinal potential games are naturally related to the evolutionary learning as well (see e.g., Crawford, 1991; Kandori and Rob, 1992; Young, 1993; Roth and Erev,1995; and the references listed therein).

As the potential function is uniquely defined up to an additive constant, the argmax set of the potential function does not depend on a particular potential function. Thus, for potential games this argmax set refines the equilibrium set, at least technically.We show that this refinement concept accurately predicts the experimental results obtained by Van Huyck et al. (1990). We do not attempt to provide any explanation to this prediction power obtained (perhaps as a coincidence) in this case.A possible way of explaining this can be found in Blume (1993). Blume discusses various stochastic strategy revision processes for play. ers who have direct interaction only with small part of the population. He proves for the log-linear strategy revision process that the strategies of the players in a symmetric potential game converge to the argmax set of the potential.

Hart and Mas-Colell (1989) have applied potential theory to cooperative games. Except for the fact that we are all using potential theory our works are not connected. Nevertheless,we will show in the last section that combining our work with Hart and Mas-Colell's yields a surprising application to value theory.

The paper is organized as follows: 

- In Section 2 we give the basic definitions and provide several useful characterizations of finite potential and finite ordinal potential games. 
- An equivalence theorem between potential games and conges tion games is given in Section 3.
- In Section 4 we discuss and characterize infinite potential games. 
- Section 5 is devoted to a discussion of the experimental results of Van Huyck et al.
- In Section 6 we show an application of our theory to the strategic approach to cooperative games

### 2. POTENTIAL GAMES

Let $\Gamma(u^{1},u^{2},\ldots,u^{n})$ be a game in strategic form with a finite number of players. The set of players is $N=\{1,2,\ldots,n\}$ , the set of strategies of Player is $Y^{\mathrm{i}}$ ,and the payoff function of Player $i$ is $\iota\iota^{i}$ ： $Y\rightarrow R$ ,where $Y=Y^{1}\times Y^{2}\times$ $\cdots\times Y^{n}$ is the set of strategy profiles, and $R$ denotes the set of real numbers When no confusion may arise we denote $\Gamma(u^{1},u^{2},\ldots,u^{n})$ by $\Gamma.$ For $S\subseteq N$ $-S$ denotes the complementary set of $S$ , and $Y^{5}$ denotes the Cartesian product $X_{i\in S}Y^i.$ For singleton sets $\{i\}$ ， $Y^{-[i]}$ is denoted by $Y^{-i}$ .A function $P$ ： $Y\rightarrow R$ is an ordinal potential for $\Gamma$ ,if for every $i\in N$ and for every $y^{-i}\in Y^{-\bar{I}}$

$$\begin{array}{rl}u^i(y^{-i},x)-u^i(y^{-i},z)>0&\mathrm{iff}\quad P(y^{-i},x)-P(y^{-i},z)>0\\&\mathrm{for~every~}x,z\in Y^i.\end{array}$$

$\Gamma$ is called an ordinal potentialgame if it admits an ordinal potential

Let $w=(w^{i})_{i\in N}$ be a vector of positive numbers which will be called weights. A function $P$ ： $Y\rightarrow R$ is a $u$ -potential for T if for every $i\in N$ and for every $y^{-i}\in Y^{-i}$

$$\begin{array}{rcl}u^i(y^{-i},x)-u^i(y^{-i},z)&=&w^i\left(P(y^{-i},x)-P(y^{-i},z)\right)\\&&\text{for every}x,z\in Y^i.\end{array}$$

$\Gamma$ is called a $w$ -potential game if it admits a $w$ -potential

---

When we are not interested in particular weights $u$ ,we simply say that $P$ is a weighted potential and that $\Gamma$ is aweighted potential game.

A function $P$ ： $Y\rightarrow R$ is an exact potential (or, in short, a potential) for $\Gamma$ ifit is a $U$ -potential for $\Gamma$ with $w^{i}=1$ for every $i\in N.\Gamma$ 厂$\Gamma$ is called an exact potential game (or,in short, a potential game) if it admits a potential.For example,the matrix $P$ is a potential for the Prisoner's Dilemma game $G$ described below:

$$G=\left(\begin{array}{cc}(1,1)&(9,0)\\(0,9)&(6,6)\end{array}\right),\quad P=\left(\begin{array}{cc}4&3\\3&0\end{array}\right).$$

The next lemma characterizes the equilibrium set of ordinal potential games. Its obvious proof will be omitted.

LEMMA 2.1. Let $P$ be an ordinal potential function for $\Gamma(u^{1},u^{2},\ldots,u^{n})$ Then the equilibrium set of $\Gamma(u^{1},u^{2},\ldots,u^{n})$ coincides with the equilibrium set of $\Gamma(P,P,\ldots,P)$ . That is, $y\in Y$ is anequilibrium point for $\Gamma$ if and only if for every $i\in N$

$$P(y)\geq P(y^{-i},x)\quad for\:every\:x\in Y^{i}.$$

Consequently, If $P$ admits a maximal value 10 in $Y$ ,then T possesses a (pure strategy) equilibrium.

COROLLARY 2.2.Every finite ordinal potential game possesses a pure-strateg) equilibrium

A path in $Y$ is a sequence $\gamma=(y_{0},y_{1},\ldots)$ such that for every $k\geq1$ there exists a unique player, say Player $i$ , such that $y_{k}=(y_{k-1}^{-i},x)$ for some $x\neq y_{k-1}^i$ in $Y^{i}$ . $y0$ is called the initial point of $\gamma$ , and if $\gamma$ is finite,then its last element is called the terminal point of $\gamma$ $\gamma=(y_{0},y_{1},\ldots)$ is an improvement path with respect to $\Gamma$ if for all $k\geq 1$ $u^{i}( y_{k}) > u^{i}( y_{k- 1})$ ,where $i$ is the unique deviator at step $k$ .Hence, an improvement path is a path generated by myopic players. $\Gamma$ has the finite improvement property $(FIP)$ if every improvement path is finite.

LEMMA 2.3.Every finite ordinal potential game has the FIP

Proof.For every improvement path $\gamma=(y_{0},y_{1},y_{2},\ldots)$ we have by (2.1)

$$P(y_0)<P(y_1)<P(y_2)<\cdots.$$

As $Y$ is a finite set, the sequence $\gamma$ must be finite.

> 9 Using Blume's (1993) terminology we can give an equivalent definition: I is a weighted potential game if and only if there exists a payoff function which is strongly best-response equivalent to each of the players’ payoff functions. Sela (1992) proved that if the two-person game (A, B) does not have weakly dominated strategies, then it has a weighted potential if and only if it is better-response cquivalent in mixed strategies (see Monderer and Shapley (1996) for the precise definition) to a game of the form $(P,P)$ This result can be easily generalized to n-person games.

---

It is obvious that for finite games with the FIP,and in particular for finite ordinal potential games, every maximal improvement path must terminate in an equilibrium point. That is, the myopic learning process based on the one-sided better reply dynamic converges to the equilibrium set.However we have obtained a stronger learning resultl1:

THEOREM 2.4 (Monderer and Shapley, 1996).Every finite weighted potential game has the Fictitious Play property

It is interesting to note that having the FIP is not equivalent to having an ordinal potential.A counterexample is the game $G_{1}$ described below. The rows in $G_{1}$ are labeled by $d$ and $b$ , and the columns are labeled by $t$ and $d$

$$G_1=\left(\begin{array}{cc}(1,0)&(2,0)\\(2,0)&(0,1)\end{array}\right).$$

The game $G_{1}$ has the FIP, but any ordinal potential $P$ for $G_{1}$ must satisfy the following impossible sequence of relations

$$P(a,c)<P(b,c)<P(b,d)<P(a,d)=P(a,c).$$

A function $P$ ： $Y\rightarrow R$ is a generalized ordinal potential for $\Gamma$ if for every $i\in N$ and for every $y^{-i}\in Y^{-\bar{I}}$ , and for every $x,z\in Y^{i}$

$$^{-i},x)-u^{i}(y^{-i},z)>0\quad\mathrm{implies~that}\quad P(y^{-i},x)-P(y^{-i},z)>0.$$

LEMMA 2.5.Let $\Gamma$ be a finite game. Then, $\Gamma$ has theFIPifand only ifThas a generalized ordinal potential

Proof.Let T be a game with the FIP. Define a binary relation ‘ '> "on $Y$ as follows: $x>y$ iff $x\neq y$ and there exists a finite improvement path $\gamma$ with an initial point $y$ and a terminal point $\lambda$ The finite improvement property implies that $“>”$ is a transitive relation. Let $Z\subseteq Y$ .We say that $Z$ is represented if. there exists $Q$ ： $Z\rightarrow R$ such that for every $x,y\in Z,x>y$ x>y $x>y$ implies that $Q( x) >$ $Q( y)$ Let $Z$ be a maximal represented subset of $Y$ .We proceed to prove that $Z=Y$ . Suppose $x\not\in Z$ .If $x>z$ for every $z\in Z$ ，we extend $Q$ to $Z\cup\{x\}$ by defining $Q( x)$ = 1 $+ \max _{z\in \mathbb{Z} }$ $Q( z)$ , thus contradicting the maximality of Z. If $z>x$ for every $z\in\mathbb{Z}$ ,we extend $Q$ to $Z\cup\{x\}$ by defining $Q(x)=\operatorname*{min}_{z\in\mathbb{Z}}Q(z)-1$ ,contradicting again the maximality of $Z$ . Otherwise we extend $Q$ and contradict the maximality of Z by defining $Q(x)=(a+b)/2$, where $a=\operatorname*{max}\{Q(z):z\in Z,\quad x>z\},\operatorname{and}b=\operatorname*{min}\{Q(z):z\in Z,\quad z>.$
Hence $Y$ is represented.

COROLLARY 2.6.Let T be a finite game with the FIP. Suppose in additior thatfor every $i\in N$ and for every $y^{-i}\in Y^{-i}$

$$u^i(y^{-i},x)\neq u^i(y^{-i},z)\quad for\:every\:x\neq z\in Y^i.$$

Then $\Gamma$ has an ordinal potential

Proof.Observe that the condition on T implies that every generalized ordinal potential for $\Gamma$ is an ordinal potential for T .Hence,the proof follows from Lemma 2.5.

Ordinal potential games have many ordinal potentials. For exact potential games we have:

LEMMA 2.7. Let $P_{1}$ and $P_{2}$ be potentials for the game I. Then there exists a constant c such that

$$P_1(y)-P_2(y)=c\quad for\:every\:y\in Y.$$

Proof.Fix $z\in Y$ For all $y\in Y$ define

$$H(y)=\sum_{i=1}^n\left[u^i(a_{i-1})-u^i(a_i)\right],$$

where $u_{0}=y$ and for every $1\leq i\leq n$ ， $a_{i}=(a_{i-1}^{-i},z^{i})$

If $P$ stands for either $P_{1}$ or $P_{2}$ ,then by (2.1), $H(y)=P(y)-P(z)$ for every $y\in Y.$ Therefore

$$P_1(y)-P_2(y)=c\quad\mathrm{for~every~y\in Y.}$$

The next results characterize exact potential games in a way that resembles the standard approach to potential functions in physics

For a finite path $\gamma=(y_{0},y_{1},\ldots,y_{N})$ and for a vector $v=(v^{1},v^{2},\ldots,v^{n})$ of functions $v^{\prime}$ ： $Y\rightarrow R$ we define

$$I(\gamma,v)=\sum_{k=1}^{n}\left[v^{i_{k}}(y_{k})-v^{i_{k}}(y_{k-1}\right],$$

where $i_k$ is the unique deviator at step $k$ (i.e., $y_{k}^{i_{k}}\neq y_{k-1}^{i_{k}}$

> 12 A constructive and more elegant proof of this result is given in Milchtaich (1996); he showed that the function $P$ that assigns to each $y\in Y$ the number of strategy profiles that are connected to $y$ by an improvement path that terminates in $y$ is a generalized ordinal potential for $\Gamma$

---

The path y =(yo $\gamma=(y_{0})$ $\gamma=(y_{0},y_{1},\ldots,y_{N})$ is closed if $y_{0}=y_{N}.$ It is a simple closed path if in addition $y_{l}\neq y_{k}$ for every $0\leq I\neq k\leq N-1.$ The length of a simple closed path is defined to be the number of distinct vertices in it. That is, the length of $\gamma=(y_{0},y_{1},\ldots,y_{N})$ is $N$

THEOREM 2.8. Let $\Gamma$ be a game in strategic form, as described at the begin ning of this section. Then the following claims are equivalent

- (1) $\Gamma$ is a potential game
- (2) $I\left(\gamma,u\right)=0$ for every finite closed paths $\gamma$
- (3) $I\left(\gamma,u\right)=0$ for every finite simple closed paths y
- (4) $I\left(\gamma,u\right)=0$ for every finite simple closed paths y of length 4

The proof of Theorem 2.8 is given in Appendix A

A typical simple closed path, $\gamma$ , of length 4 is described below. In this path, $i$ and $j$ are the active players, $a\in Y^{-\{i,j\}}$ is a fixed strategy profile of the other players, $x_{i},y_{i}\in Y^{\bar{I}}$ , and $x_{j},y_{j}\in Y^{j}$

![](https://storage.simpletex.cn/view/fQgYhac0IDdu0XG7B7aIhWzEdN6uGSIh9)

where $A=(x_{i},x_{j},a)$ ， $B=(y_{i},x_{j},a)$ ， $C=(y_{i},y_{j},a)$ , and $D=(x_{i},y_{j},a)$

COROLLARY 2.9. $\Gamma$ is a potentialgame if and only if forevery i, $j\in N,for$ every $a\in Y^{-\{i,j\}}$ ,and for every $X_{i}$ $y_{i}\in Y^{i}$ and $X_{j}$ $y_{j}\in Y^{j}$

$$u^{i}(B)-u^{i}(A)+u^{j}(C)-u^{j}(B)+u^{i}(D)-u^{i}(C)+u^{j}(A)-u^{j}(D)=0,$$

where thepoints $A, B,C$, and $D$ are described above

We end this section with an important remark concerning the mixed extension of finite games.

LEMMA 2.10.Let $\Gamma$ be a finite game. Then T is a $U$ -potential game if and only if the mixedextension of $\Gamma$ isa $u$ -potential game

Proof. For $i\in N$ let $\Delta^{i}$ be the set of mixed strategies of Player i and let $U^{i}$ be thepayoff function of player $i$ in the mixed extension of $\Gamma.$ That is

$$\begin{array}{ll}U^i(f)&=\:U^i(f^1,f^2,\dots,f^n)\\&=\:\sum_{y\in Y}u^i(y^1,y^2,\dots,y^n)f^1(y^1)f^2(y^2)\dots f^n(y^n),\quad\forall f\in\\\end{array}$$

where $\Delta=X_{\mathrm{~i~e}N}\Delta^{i}.$ Obviously, if $\bar{P}$ ： $\Delta\rightarrow R$ is a $u$ -potential function for the mixed extension of $\Gamma$ ,then its restriction to $Y$ yields a $u$ -potential for $\Gamma$ .As for

---

the converse, suppose $P$ is a $W$ -potential for $\Gamma$ then it can be easily verified that $\bar{P}$ is a potential for the mixed extension of $\Gamma$ where

$$\bar{P}(f^{1},f^{2},\ldots,f^{n})=\sum_{y\in Y}P(y^{1},y^{2},\ldots,y^{n})f^{1}(y^{1})f^{2}(y^{2})\ldots f^{n}(y^{n}).$$

An example to an ordinal potential game whose mixed extension is not an ordinal potential game is given in Sela (1992)

### 3.CONGESTION GAMES

Congestion games were defined by Rosenthal (1973). They are derived from congestion models that have been extensively discussed in the literature (see e.g., Garcia and Zangwill, 1981). Consider an illustrative example

![](https://storage.simpletex.cn/view/fi9dDlatHWWwpbH65fQBKLPhZA3GHiSG5)

In the congestion model described above, Driver $Ur$ has to go from point A to point $C$ and Driver $b$ has to go from point $B$ topoint $D$ . $AB$ is called road segment 1, $BC$ is called road segment $2,\ldots$ etc. $c_{j}(1)$ denotes the payoff (e.g. the negative of the cost) for a single user of road segment $j$ . $c_{j}(2)$ denotes the payoff for each user of road segment $j$ if both drivers use road segment $j$ .The drivers are therefore engaged in a game (the associated congestion game, $CG$ whose strategic form is given below (The rows are labeled by $\{1,2\}$ and $\{3,4\}$ and the columns are labeled by $\{1,3\}$ and (2, 4}

$$\left.CG=\left(\begin{array}{cc}{(c_{1}(2)+c_{2}(1),c_{1}(2)+c_{3}(1))}&{(c_{2}(2)+c_{1}(1),c_{2}(2)+c_{4}(1))}\\{(c_{3}(2)+c_{4}(1),c_{3}(2)+c_{1}(1))}&{(c_{4}(2)+c_{3}(1),c_{4}(2)+c_{2}(1))}\\\end{array}\right.\right.$$

By Corollary 2.9 the congestion game $CG$ admits a potential. In particular (and with no restrictions on the payoff $c_{j}(i)$ ) it has a (pure-strategy) equilibrium For completeness we attach below a potential $P$ for the congestion game.The potential is computed by formula (3.2):

$$P=\left(\begin{array}{cc}c_1(1)+c_1(2)+c_2(1)+c_3(1)&c_2(1)+c_2(2)+c_1(1)+c_4(1)\\c_3(1)+c_3(2)+c_4(1)+c_1(1)&c_4(1)+c_4(2)+c_3(1)+c_2(1)\end{array}\right)$$

A congestion model $C(N,M,(\Sigma^{i})_{i\in N},(c_{j})_{j\in M}$ (cj)jeM $(c_{j})_{j\in M}$ is defined as follows. $N$ de notes the set of players $\{1,2,\ldots,n\}$ (e.g., drivers. $M$ denotes the set of facilities $\{1,2,\ldots,m\}$ (e.g, road segments). For $i\in N$ let $\Sigma^{i}$ be the set of strategies of player $l$ ,where each $A^{i}\in\Sigma^{i}$ is a nonempty subset of facilities (e.g., a route). For $j\in M$ let $c_{j}\in R^{|1,2,\ldots,n}$ denote the vector of payoffs,where $c_{j}(k)$ denotes the payoff (e.g., the cost) to each user of facility $j$ ,if there are exactly $k$ users. The congestion game associated with the congestion model is the game in strategic form with the set of players $N$ with the sets of strategies $(\Sigma^{i})_{i\in N}$ ,and with payoff functions $(v^i)_{i\in N}$ defined as follows:

Set $\Sigma=X_{i\in N}\Sigma^{i}$ For all $A\in\Sigma$ and for every $j\in M$ let $\sigma_{j}(A)$ be the number of users of facility $j$ . That is,

$$\sigma_j(A)=\#\{i\in N\colon j\in A^i\},$$

where $A=(A^{1},A^{2},\ldots,A^{n})$ Define $v^{i}$ ： $\Sigma\rightarrow R$ by

$$v^i(A)=\sum_{j\in A^i}c_j(\sigma_j(A)).$$

The following theorem can be deduced from Rosenthal (1973)

THEOREM 3.1.Every congestion game is a potential game

Proof.Let T be the congestion game defined by the parameters $N$ ， $M$ ()ieN $(\Sigma^i)_{i\in N}$ $(\Sigma^{i})_{i\in N},(c_{j})_{j\in M}$ (cj)jeM $(c_j)_{j\in M}$ For each $A\in\Sigma$ define

$$P(A)=\sum_{j\in\cup_{i=1}^{*}A^{i}}\left(\sum_{l=1}^{\sigma_{j}(A)}c_{j}(l)\right).$$

The proof that $P$ is a potential for $\Gamma$ can be deduced from Rosenthal (1973) or directly using Corollary 2.9.

Let $\Gamma_{1}$ and $\Gamma_{2}$ be games in strategic form with the same set of players $N.$ For $k$ = 1,2 let $(Y_k^i)_{i\in N}$ be the strategy sets in $\Gamma_{k}$ , and let $(u_k^i)_{i\in N}$ be the payoff functions in $\Gamma_{k}$ We say that $\Gamma_{1}$ and $\Gamma_{2}$ are isomorphic if there exist bijections $g^i$ g $Y_{1}^{i}\rightarrow Y_{2}^{i}$ Yi→Y $g^{i}\colon Y_{1}^{i}\to Y_{2}^{i},i\in N$ i e N $i\in N$ , such that for every $i\in N$

$$\begin{array}{rcl}u_1^i(y^1,y^2,\dots,y^n)&=&u_2^i(g^1(y^1),g^2(y^2),\dots,g^n(y^n))\\&&\text{for every}(y^1,y^2,\dots,y^n)\in Y_1,\end{array}$$

where $Y_{1}=X_{i\in N}Y_{1}^{i}$

THEOREM 3.2.Every finite potentialgame is isomorphic to a congestion game.

The proof, as well as several relevant discussions, is given in Appendix B.

---

### 4. INFINITE POTENTIAL GAMES

Let $\Gamma$ be a game in strategic form as described in Section 2. F is called a bounded game if the payoff functions $(u^i)_{i\in N}$ are bounded.

LEMMA 4.1.Every bounded potential game possesses an 8 -equilibrium point for every $\varepsilon>0$

Proof. Note that by (2.2) every potential $P$ for T must be bounded. Let $\varepsilon>0.$ There exists $z\in Y$ satisfying

$$P(z)>\sup_{y\in Y}P(y)-\varepsilon.$$

Obviously $z$ is an $\varepsilon$ -equilibrium point.

Recall the concept of a path from Section 2. Let $\varepsilon>0.$ A path $\gamma=(y_{0},y_{1},\ldots)$ is an $\varepsilon$ -improvement path with respect to $\Gamma$ if for all $k\geq 1$ $u^{i}( y_{k}) > u^{i}( y_{k- 1}) + \varepsilon$ where $i$ is the unique deviator at step $k$ . The game $\Gamma$ has the approximate finite improvement property (AFIP) if for every $\varepsilon>0$ every E -improvement path is finite. The proof of the next lemma is obvious and will be omitted.

LEMMA 4.2.Every bounded potential game has the AFIP.

Note that for games with the AFIP, and in particular for bounded potential games, every maximal $E$ -improvement path terminates in an 8 -equilibrium point. A game $\Gamma$ is called a continuous game if the strategy sets are topological spaces and the payoff functions are continuous with respect to the product topology Note that by (2.2), the potential of a continuous potential game is continuous Therefore we have:

LEMMA 4.3. Let T be a continuous potential game with compact strategy sets. Then I possesses a pure-strategy equilibrium point.

We now proceed to deal with differentiable games.We assume that the strategy sets under discussion are intervals of real numbers. We omit the obvious proof of the next lemma

LEMMA 4.4. Let $\Gamma$ beagameinwhich the strategy sets areintervals of real numbers. Suppose the payoff functions $\iota\iota^{i}$ : $Y^{i}\to R$ ， $i\in N$ , are continuously differentiable, and let $P$ ： $Y\rightarrow R$ .Then $P$ is a potential for IT if andonly if $\cdot P$ is continuously differentiable, and

$$\frac{\partial u^i}{\partial y^i}=\frac{\partial P}{\partial y^i}\quad for\:every\:i\in N.$$

The next theorem is well-known (and very useful)

---

THEOREM 4.5. Let IT be a game in which the strategy sets are intervals of. real numbers. Suppose the payoff functions are twice continuously differentiable Then IT is a potential game iff

$$\frac{\partial^2u^i}{\partial y^i\partial y^j}=\frac{\partial^2u^j}{\partial y^i\partial y^j}\quad for\:every\:i,j\in N.$$

Moreover,if the payoff functions satisfy (4.1) and $\zeta$ is an arbitrary (but fixed strategy profile in $Y$ , then a potential for T is given by

$$\begin{aligned}P(y)=\sum_{i\in N}\int_0^1\frac{\partial u^i}{\partial y^i}(x(t))(x^i)'(t)\:dt,\end{aligned}$$

where $A$ $x\colon[0,1]\to Y$ is a piecewise continuously differentiable path in. $Y$ that connects z to y (i.e., $x(0)=z$ and $x(1)=y$

Consider for example the quasi-Cournot game described in the Introduction It can be easily verified that (4.1) is satisfied (because $\partial ^{2}u^{i}/ \partial y^{i}\partial y^{j}$ = $a$ for every $i\neq j\in N$ ), and applying (4.2) yields the potential given in (1.2). Unlike (weighted) potential games, ordinal potential games are not easily characterized. We do not know of any useful characterization, analogous to the one given in (4.1), for differentiable ordinal potential games.

### 5. THE POTENTIAL AS AN EQUILIBRIUM REFINEMENT TOOL

Let $\Gamma$ be a potential game and let $P$ be a potential for $\Gamma.$ The set of all strategy profiles that maximize $P$ is a subset of the equilibria set. By Lemma 2.7, this set does not depend on a particular potential function.13 Thus, at least technically the potential defines a refinement concept Consider the version of the Stag Hunt game of Rouseau, as described in

Crawford (1991): There are TI players. Player i chooses $e_{i}\in\{1,2,\ldots,7\}$ .The payoff $\Pi_{i}$ of Playeri is

$$\Pi_i(e_1,e_2,\ldots,e_n)=a\min(e_1,e_2,\ldots,e_n)-be_i+c,$$

where $a>b\geq0$ ,and $C$ is a constant that guarantees positive payoffs. Define a potential function $P$ as

$$P(e_1,e_2,\dots,e_n)=a\min(e_1,e_2,\dots,e_n)-b\sum_{j=1}^je_j.$$

> 13 It can also be proved that for weighted potential games, the argmax set of a weighted potential does not depend on a particular choice of a weighted potential (even though distinct weighted potentials may be based on different sets of weights (i.e., neither vector of weights is a multiple by a scalar of the other vector))

---

Note that if $a<nb$ , then $P$ is maximized at the profile $\ell$ with $e_i=1$ for every $1\leq i\leq n$ .If $a>nb$ ,then $P$ is maximized at the strategy profile satisfying $e_i=7$ for every i. Surprisingly, the equilibrium selection predicted by the argmax set of the potential is the one that is supported by the experimental results of Van Huyck et al. (1990). In Experiment A (using Crawford’s notation), $a=0.2$ $b=0.1$ ,and $14\leq n\leq16$ Thus $a<nb$ In Experiment $B$ ， $b$ was switched to O, and therefore $a>nb$ In Experiments $C_{d}$ and $C_{f}$ ， $a=nb$ . In this case, every equilibrium profile maximizes the potential, and thus the potential cannot be used for a prediction. Indeed, in $C_{d}$ ,the players were not using a particular equilibrium profile. In Experiment $C_{f}$ , which was the same as $C_{d}$ except for the fact that the two players were fixed (and not randomly matched), players tended to choose $e_{1}=e_{2}=7$ This, to our opinion,reflects the principal that a repetition is a substitute to cooperation in repeated games. We do not attempt to explain the success of the argmax set of the potential

to predict behavior in the above potential game. It may be just a coincidence.14 We hope that further experiments will be conducted to test this new refinement concept

Van Huyck et al. (1991) conducted another set of experiments on average opinion games. In this experiments the payoff function of Player i is given by

$$\Pi_i(e_1,e_2,\ldots,e_n)=\alpha M-\beta(M-e_i)^2+\gamma,$$

where $Cx$ ， $\beta$ ,and $\gamma$ are positive constants, and $M=M(e_{1},e_{2},\ldots,e_{n})$ is the median of $(e_{1},e_{2},\ldots,e_{n})$

It can be seen easily that this game does not have a weighted potential, and thus we are unable to analyze their results via the potential approach.However if the median function $M$ is replaced by the mean function, $A(e_1,e_2,\ldots,e_n)=$ 1 $ln\sum_{i=1}^{m}e_{i}$ , then by Theorem 4.5 the game does have a potential. The unique strategy profile that maximizes this potential is $e_{i}=7$ for every i. Unfortunately we do not know of any experiment conducted with the mean function $A$

### 6. AN APPLICATION TO THE STRATEGIC APPROACH TO VALUE THEORY

Let $N$ = $\{ 1, 2, \ldots , n\}$ be the set of players. For each nonempty coalition $S\subseteq N$ we denote by $G(S)$ the space of all cooperative games with transferable utility on the set of players $S.$ That is, $v\in G(S)$ if and only if $u$ is a real-valued function defined on the set $2^{5}$ of subsets of $S$ with $v(\emptyset)=0$ .A solution is a function $\psi:\cup_{S\in2^N}G(S)\to\cup_{S\in2^N}R^{S}$ such that $\psi(v)\in R^{S}$ whenever $v\in G(S)$ A solution $\psi$ is efficient if $\sum_{i\in S}\psi v(i)=v(S)$ for every $S\in2^{N}$ and for every $v\in G(S)$

---

For each solution $\psi$ and for each $c\in R^{N}$ we will define a game in strategic form $\Gamma(\psi,c,v)$ for every $v\in G(N)$ as follows:

The set of players is $N.$ The set of strategies of player $i$ is $Y^{i}=\{0\}$ 1}.Player i can decide not to join the game (choosing O) and to get a payoff $c^{i}$ , or to participate in the game (choosing 1). Let $S$ be the set of all players that choose 1. Then each $i\in S$ receives the payoff $\psi(v_{S})(i)$ ,where $v_{S}\in G(S)$ is the restriction of 2 to $2^{5}.$ More precisely, for $\varepsilon \in Y= \{ 0$, $1\} ^{N}$ denote $S(\varepsilon)=\{i\in N:\varepsilon^{i}=1\}$ Then the payoff function $u^{i}$ of player i is

$$u^i(\varepsilon)=\left\{\begin{array}{ll}c^i,&\quad\text{if}\varepsilon^i=0\\\psi(v_{S(\varepsilon)})(i),&\quad\text{if}\varepsilon^i=1.\end{array}\right.$$

The games $\Gamma(\psi,c,v)$ will be called the participation games. We now present two characterizations (a local characterization and a global one) for the Shapley value in terms of the strategic properties of the participation games

THEOREM 6.1.Let be an efficient solution on $G= \cup _{S\in 2^{N}}G( S) , let$ $c\in R^{N}$ and let $v\in G(N)$ .Then $\psi$ is theShapleyvalue on $\{v_S$ : $S\in2^{N}\}$ if and only if $\Gamma=\Gamma(\psi,c,v)$ is a potential game

Proof.Leti ∈ N.Then

$$u^i(\varepsilon^{-i},1)-u^i(\varepsilon^{-i},0)=\psi(v_{S\cup\{i\}})(i)-c^i\quad\mathrm{for~all~}\varepsilon\in Y,$$

where $S=\{j\neq i:\varepsilon^{j}=1$

For $S\subseteq N$ let $\varepsilon_{S}\in Y$ be defined as follows: $\varepsilon_{S}^{i}=1$ if $i\in S$ , and $\varepsilon_{S}^{i}=0$ if $i\not\in S$

From (6.1) we deduce that $\Gamma$ is a potential game if and only if there exists $Q$ ： $Y\rightarrow R$ such that

$$Q(\varepsilon_{S})-Q(\varepsilon_{S\setminus\{i\}})=\psi(v_{S\cup\{i\}})(i)-c^{i}\quad\mathrm{for~every~}S\subseteq N\mathrm{~and~for~e}$$

Set $P(\varepsilon_{S})=Q(\varepsilon_{S})+\sum_{i\in S}c^{i}$ ,then $Q$ satisfies (6.2) iff $P$ satisfies

(6.3)
$$P(\varepsilon_{S})-P(\varepsilon_{S\setminus\{i\}})=\psi(v_{S\cup\{i\}})(i)\quad\mathrm{for~all~}S\subseteq N\mathrm{~and~for~every~}i$$

Thus, the proof follows from Theorem A in Hart and Mas-Colell (1989).

THEOREM 6.2.Let $\psi$ be an efficient solution on $G=\cup_{S\in2^{N}}G(S)$ ,andlet $c\in R^{N}$ Then is theShapley valueon $G$ if and onlyif $\Gamma(\psi,c,v)$ is a potential game for every $v\in G(N)$

Proof.The proof follows from Theorem 6.1.

By Theorem 5.2in Hart and Mas-Colell (1989) we can also prove the following characterization of weighted Shapley values.

---

THEOREM 6.3.Let be an efficient solution on $G= \cup _{S\in 2^{N}}G( S) , let$ $c\in R^{N}$ let $v\in G(N)$ ,and let w be a vector of positive weights.Then $\psi$ is the $w$ -Shapley value on $\{v_{S}:S\in2^{N}\}$ if and only if $\Gamma=\Gamma(\psi,c,v)$ is a w-potential game

Other results relating noncooperative potential games with cooperative solutions are discussed in Qin (1992)

### APPENDIX A

Proof of Theorem 2.8.Obviously $(2)\Longrightarrow(3)\Longrightarrow(4)$ .We prove that (1) $\Longleftrightarrow(2)$ and that $(4)\Longrightarrow(2)$

$(1)\Longrightarrow(2)$ Suppose $P$ is a potential for T . Let $\gamma=(y_{0},y_{1},\ldots,y_{N})$ be a closed path. Then by (2.2)

$$I(\gamma,u)=I(\gamma,(P,P,\ldots,P))=P(y_N)-P(y_0)=0.$$

$(2)\Longrightarrow(1)$ Suppose $I\left(\gamma,u\right)=0$ for every closed path $\gamma$ . Fix $z\in Y$ . Let $y\in Y.$ We claim that for every two paths $Y1$ and $Y_{2}$ that connect $Z$ to $y,I\left(\gamma_{1},u\right)=$ $I\left(\gamma_{2},u\right).$ Indeed, suppose $\gamma _{1}= ( z$, $y_{1}, \ldots , y_{N})$ and $\gamma_{2}=(z,z_{1},\ldots,z_{M})$ ,where $y_{N}=z_{M}=$y. Let $\mu$ μ $\mu$ be the closed path $(\gamma_{1},\gamma_{2}^{-1}).$ That is

$$\mu=(z,y_{1},\ldots,y_{N},z_{M-1},z_{M-2},\ldots,z).$$

Then $I(\mu,u)=0$ . Therefore $I(\gamma_{1},u)=I(\gamma_{2},u)$ . For every $y\in Y$ choose a path, say y $\gamma$ $\gamma(y)$ , connecting $Z$ to y. Define $P(y)=I(\gamma(y),u)$ for all $y\in Y$ .We proceed to prove that $P$ is a potential for $\Gamma$ .We have just proved tha

$$P(y)=I(\gamma,u)\quad\mathrm{for~every~\gamma~that~connects~z~to~y.}$$

Let $i\in N$ ,let $y^{-i}\in Y^{-i}$ , and let $a\neq b\in Y^{i}$ Let $\gamma=(z,y_{1},\ldots,(y^{-i},a))$ be a path connecting $\angle L$ to $(y^{-i},a)$ $\operatorname{Set}\mu=(z,y_{1},\ldots,(y^{-i},a),(y^{-i},b)).$ Then by (A.1)

$$(y^{-i},b)-P(y^{-i},a)=I(\mu,u)-I(\gamma,u)=u^{i}(y^{-i},b)-u^{i}(y^{-i},a).$$

Therefore $P$ is a potential for $\Gamma$

$(4)\Longrightarrow(2)$ Suppose $I(\gamma,u)=0$ for every simple closed path $Y$ of length 4. We denote the length of a closed path $\gamma$ = $( y_{0}, y_{1}, \ldots , y_{N})$ $l( \gamma )$ (= $N)$ （=N $(=N$ Suppose that for some closed path, say $Y$ ， $I\left(\gamma,u\right)\neq0.$ Obviously $N=l(\gamma)\geq$ 5. Without loss of generality we may assume that $I\left(\mu,u\right)=0$ , whenever $l(\mu)<N$

Suppose y =(yo $\gamma=(y_{0}$ YN $Y_{N}$ $\gamma=(y_{0},y_{1},y_{2},\ldots,Y_{N}).$Let$i(j),0\leq j\leq N-1$ be the unique de viator at step $j$ . That is, $y_{j+1}=(y_{i}^{-i(j)},x(i(j)))$ , where $x(i(j))\neq y_{i}^{i(j)}$ Without loss of generality assume that $i(0)=1.$ Since $i(0)=1$ , and $y_{N}=y_{0}$ there exists

---

$1\leq j\leq N-1$ such that $i(j)=1.$ If $j=1$ or $j=N-1$ ,we get a contradiction to the minimality assumption about the length of $\gamma$ in the following way: Assume w.l.o.g. that $i(1)=1.$ Define $\mu=(y_{0},y_{2},\ldots,y_{N})$ Then, $I(\mu,u)=I(\gamma,u)$ and $l(\mu)<N$ .Assume therefore that $2\leq j\leq N-2$ We show that there exists $z_{j}\in Y$ such that the path $\mu=(y_{0},y_{1},\ldots,y_{j-1},z_{j},y_{j+1},\ldots,y_{N})$ yj+1yN $y_{j+1},\ldots,y_N)$ satisfies

$$I(\mu,u)=I(\gamma,u)\quad\mathrm{and}\quad i(j-1)=1.$$

Indeed, define

$$z_j=(y_{j-1}^{-\{i(j-1),1\}},y_{j-1}^{i(j-1)},y_{j+1}^{1}).$$

Then, by our assumption on closed paths of length 4

$$I((y_{j-1},y_j,y_{j+1},z_j),u)=0.$$

This implies (A.2)

Continuing recursively,we finally find a closed path T of length $N$ such that $I\left(\tau,u\right)\neq0$ , and $i(0)=i(1)=1$ in contradiction to the minimality assumption We conclude that $I\left(\gamma,u\right)=0$ for every closed paths y.

### APPENDIX B

The payoff functions in the congestion game are given in (3.1). We need an equivalent formulation in order to proveTheorem 3.2. For $A=(A^{1},A^{2},\ldots,A^{n})$ $\in\Sigma$ and for $S\subseteq N$ we denote $A(S)=\cup_{i\in S}A^{i}$ ,and we denote $A(-S)=A(S^{c})$ where $S^{c}$ is the complementary set of $S$ .For $S=\{i\}$ ,A(i) and $A(-i)$ stand for $A(\{i\})$ and $A(-\{i\})$ respectively. For $x\in R^{M}$ and for $B\subseteq M$ we denote $x(B)=\sum_{j\in B}x(j)$

LEMMA B.1.Suppose $C$ is a congestion game as described in Section 3. For every $r\in N$ define the vector $x^r\in R^M$ as

$$x^r(j)=c_j(m)\quad for\:every\:j\in M.$$

Thenfor every $i\in N$ and for every $A\in\Sigma$

$$
\begin{aligned}
v^{i}(A)\:=& x^{1}\left(A(i)\cap A(-i)^{c}\right)  \\
&+x^{2}\left(\cup_{k\neq i}[A(i)\cap A(k)\cap A(-\{i,k\})^{c}]\right) \\
&+\cdots+x^{n}\left(\cap_{k\in N}A(k)\right).
\end{aligned}
$$

Proof.The proof follows from (3.1).

---

Proof of Theorem 3.2.Let $\Gamma$ be a finite potential game as described in Sec tion 2. The set of players is $N=\{1,2,\ldots,n\}$ , the strategy sets are $(Y^i)_{i\in N}$ ,and thepayoff functions are $(u^i)_{i\in N}$ Let $P$ be a potential for T Let $k(i)=\#Y^{i}$ be the number of strategies of player $i$ , and assume

$$Y^i=\{a_1^i,a_2^i,\ldots,a_{k(i)}^i\}.$$

For $i\in N$ , set $K(i)=\{1,2,\ldots,k(i)\}$ , and set $K=X_{i=1}^{n}K(i)$

We proceed to define an isomorphic congestion game.The facility set $M$ is defined to be the set of all $\varepsilon=(\varepsilon^{1},\varepsilon^{2},\ldots,\varepsilon^{n})$ ,where for every $i\in N\varepsilon^{i}$ is a vector of 0 's and 1 's of length $k(i)$ That is, $\varepsilon ^{i}\in \{ 0$, $1\} ^{K( i) }$ In other words

$$M=\times_{i=1}^{n}\{0,1\}^{K(i)}.$$

The strategy sets $(\Sigma^{i})_{i\in\mathcal{N}}$ in the congestion games are defined as

$$\Sigma^{i}=\{A_{1}^{i},A_{2}^{i},\ldots,A_{k(i)}^{i}\}\quad\mathrm{for~every~}i\in N\:,$$

where

$$A_{l}^{i}=\{\varepsilon\in M:\varepsilon_{l}^{i}=1\}\quad\mathrm{for~every}\:l\in K(i).$$

We now define vectors $\left(x^{r}\right)_{r\in N}$ in $R^{M}$ such that the payoffs $(v^i)_{i\in N}$ defined in Lemma B.1 satisfy

$$\begin{aligned}v^{i}(A_{m_{1}}^{1},A_{m_{2}}^{2},\ldots,A_{m_{s}}^{n})&=\:u^{i}(a_{m_{1}}^{1},\:a_{m_{2}}^{2},\ldots,a_{m_{n}}^{n}),\\&\forall i\in N\:\mathrm{and}\:\forall(m_{1},m_{2},\ldots,m_{n})\in K.\end{aligned}$$

For $1<r<n$ set $x^{\prime}=0$

For $r=n,x^{n}$ is defined to be a solution of the following system of equations:

$$\cap A_{m_{2}}^{2}\cap\cdots\cap A_{m_{\pi}}^{n})=P(a_{m_{1}}^{1},a_{m_{2}}^{2},\ldots,a_{m_{\pi}}^{n}),\quad(m_{1},m_{2},\ldots,m_{n})\in $$

We have to show that a solution to (B.2) exists.Foreach $m=(m_{1},m_{2},\ldots,m_{n})\in$ $K$ let $\varepsilon(m)\in M$ be defined as follows: $\varepsilon_{m_{i}}^{i}=1$ for every $i\in N$ , and $\varepsilon_{k}^{i}=0$ for every $i\in N$ and for every $k\neq m_{\mathrm{r}}$ in $K(i)$ .Set

$$M_1=\{\varepsilon(m)\colon m\in K\}.$$

Note that for $m\neq l\in K$ ， $\varepsilon(m)\neq\varepsilon(l).$ Therefore we can define $x^{n}$ as

$$x^n(\varepsilon)=\left\{\begin{array}{ll}P(a_{m_1}^1,a_{m_2}^2,\ldots,a_{m_n}^n),&\quad\text{if}\varepsilon=\varepsilon(m)\in M_1\\0,&\quad\text{if}\varepsilon\not\in M_1.\end{array}\right.$$

---

It can be verified easily that for every $m\in K$

$$A_{m_{1}}^{1}\cap A_{m_{2}}^{2}\cap\cdots\cap A_{m_{n}}^{n}\cap M_{1}=\{\varepsilon(m)\}.$$

Therefore $x^{n}$ satisfies (B.2)

Weproceed to define $x^{1}$ .Note that by (2.2) for every $i\in N$ and for every $a^{-i}\in Y^{-i}$ , the expression $u^{i}(a^{-i},a^{i})-P(a^{-i},a^{i})$ does not depend on $a^{i}\in Y^{i}$ That is,

$$u^{i}(a^{-i},a^{i})-P(a^{-i},a^{i})=u^{i}(a^{-i},b^{i})-P(a^{-i},b^{i})\quad\mathrm{for~every~}a^{i},$$

For every $i\in N$ define $Q^{-i}$ ： $Y^{-i}\to R$ by

$$Q^i(a^{-i})=u^i(a^{-i},a^i)-P(a^{-i},a^i),$$

where $a^{\dot{r}}$ is arbitrarily chosen from $Y^{i}$

For each $i\in N$ and for each $m^{i}=(m_{k}^{i})_{k\neq i}\in K^{-i}$ define $\varepsilon(m^{i})\in M$ as

$\varepsilon_{s}^{i}=1$ for every $s\in K(i)$ , and for every $k$ ， $k\neq i,\varepsilon_{s}^{k}=0$ iff $s=m_k^i$

Set

$$M_2=\{\varepsilon(m^i)\colon m^i\in K^{-i}\}.$$

Define $x^{1}$ as

$$x^1(\varepsilon)=\left\{\begin{array}{ll}Q^i\left((a_{m_i}^k)_{k\neq i}\right),&\quad\text{if}\varepsilon\in M_2\:\text{and}\varepsilon=\varepsilon(m^i)\\0,&\quad\text{if}\varepsilon\not\in M_2.\end{array}\right.$$

$(A_{m_{1}}^{1},A_{m_{2}}^{2},\ldots,A_{m_{n}}^{n})\in\Sigma$ evry $m$ = $( m_{1}, m_{2}, \ldots , m_{n})$ $\in$ $K$ and for $A=$

$$^{-1}\left(A_{m_{i}}^{i}\cap A(-i)\right)=x^{1}(\varepsilon(m^{i}))=u^{i}(a)-P^{i}(a)\quad\mathrm{for~every~}i\in N,$$

where $m^{i}=(m_{k})_{k\neq i}$ and $a=(a_{m_{1}}^{1},a_{m_{2}}^{2},\ldots,a_{m_{\pi}}^{n})$ Combine (B.6), (B.2), and Lemma B.1 to get that for every $i\in N$

$$v^{i}(A_{m_{1}}^{1},A_{m_{2}}^{2},\ldots,A_{m_{n}}^{n})=u^{i}(a_{m_{1}}^{1},a_{m_{2}}^{2},\ldots,a_{m_{n}}^{n}),\quad\forall(m_{1},m_{2},\ldots,n$$

We conclude this Appendix with a remark about the minimal number of fa cilities that are needed to represent potential games by congestion games

Let the number of players, 77 , and the strategy sets, $(Y^i)_{i=1}^n$ , be fixed. Then the dimension $d$ of the linear space of all potential games with Tr players and with

---

the strategy sets $(Y^i)_{i=1}^n$ is

$$d=\frac{k}{k(1)}+\frac{k}{k(2)}+\cdots+\frac{k}{k(n)}+k-1,$$

where for every $i\in N$ ， $k(i)=\#Y^{i}$ and $k=k(1)k(2)\cdots k(n)$

Suppose we are looking for a fixed set of facilities $M$ with m elements and for fixed strategy sets $(\Sigma^{i})_{i\in N}$ with $\#\Sigma^{i}=k(i)$ for every $i\in N$ ,such that each potential game will be represented by a congestion game with $IT$ players, with the facility set $M$ , and with the strategy sets $(\Sigma^{i})_{i\in N}.$ Then by Lemma B.1 each such congestion game is uniquely defined by 71 vectors $(x^i)_{i\in N}$ in $R^{M}$ . Suppose also that we wish the representation operation to be linear, then we must have

$$m\ge\frac{1}{n}\left(\frac{k}{k(1)}+\frac{k}{k(2)}+\cdots+\frac{k}{k(n)}+k-1\right).$$

In the proof of Theorem 3.2, $m=2^{k(1)+k(2)+\cdots+k(n)}$ However, instead of $M$ we could have defined our facility set to be either $M_{1}$ or $M_{2}$ (the one with the greater number of elements). Hence, the number of facilities 172 could be reduced to

$$m=\max\left(k,\frac{k}{k(1)}+\frac{k}{k(2)}+\cdots+\frac{k}{k(n)}\right).$$

Comparing (B.7) to (B.8) indicates that it may be possible to improve upon our result.

### REFERENCES

Bergstrom, C., and Varian, H. R. (1985). Two Remarks on Cournot Equilibria. Econ. Lett. 19, 58 Blume, L. E. (1993). \*The Statistical Mechanics of Strategic Interaction. Games Econ.Behav. 5387424.

Brown, G. W. (195 1).“Iterative Solution of Games by Fictitious Play," in Activity Analysis of Productior and Allocation. New York: Wiley.

Crawford, V. P. (1991). “An Evolutionary Interpretation of Van Huyck, Battalio, and Beil’s Experimental Results on Coordination." Games Econ. Behav. 3, 2559

Deschamps, R. (1973). Ph.D. Thesis. University of Louvain

Fudenberg, D., and Kreps, D. (1993). *Learning, Mixed Equilibria, Games Econ. Behav. 5, 320367 Fudenberg, D., and Levine, D. K. (1994). *Consistency and Cautious Fictitious Play,” mimeo.

Garcia, C. B., and Zangwill, W. I. (1981). “Pathways to Solutions, Fixed Points, and Equilibria," New York: Prentice Hall.

Hart, S., and Mas-Colell, A. (1989). Potential, Value, and Consistency, Econometrica 57, 589614 Hofbauer, J. (1994). “Stability for the Best Response Dynamics, mimeo

Jordan, J. S. (1993). “Three Problems in Learning Mixed-Strategy Nash Equilibria," Games Econ. Behay.5.368386

---

Kandori, M., and Rob, R. (1992). *Evolution of Equilibria in the Long Run: A General Theory and Applications," mimeo. Krishna, V. (1991). *Learning in Games with Strategic Complementarity,* mimeo Krishna, V., and Sjostrom. (1994). “On the Rate of Convergence of Fictitious Play,” mimeo Milchtaich, I. (1996). Congestion Games With PlayerSpecific Payoff Functions, Games Econ Behav. 13, 111124. Milgrom, P., and Roberts, J. (1991). “Adaptive and Sophisticated Learning in Normal Form Games, Games Econ. Behav. 3, 82100 Miyasawa, K. (1961). “On the Convergence of the Learning Process in a $2\times2$ Non-zero-sum Two Person Game, Economic Research Program, Princeton University, Research Memorandum No. 33 Monderer, D., and Sela, A. (1992). “Fictitious Play and No-Cycling Conditions," mimeo Monderer, D., Samet, D., and Sela, A. (1994). “Belief Affirming in Learning Processes,” mimeo. Monderer, D., and Shapley, L. S. (1996). “Fictitious Play Property for Games with Identical Interests, J. Econ. Theory 1, 258265 Neyman, A. (1991). “Correlated Equilibrium and Potential Games,” mimeo Qin, C-Z. (1992). “On a Potential Game for Endogenous Formation of Cooperation Structures,” mimeo Robinson, J. (1951). “An Iterative Method of Solving a Game," Ann. Math. 54, 296301 Rosenthal, R. W. (1973). "A Class of Games Possessing Pure-Strategy Nash Equilibria," Inr. J. Game Theory 2, 6567. Roth, A. E., and Erev, I. (1995). *Learning in Extensive-Form Games: Experimental Data and Simple Dynamic Models in the Intermediate Term, Games Econ. Behav. 8, 164212 Sela, A. (1992). *Learning Processes in Games,” M.Sc. Thesis. The Technion, Haifa, Israel. [In Hebrew] Shapley, L. S. (1964). *Some Topics in Two-Person Games," in Advances in Game Theory (M. Dresher L. S. Shapley, and A. W. Tucker, Eds.), pp. 128, Princeton, NJ: Princeton Univ. Press Shin, H. S., and Williamson, T. (1994). “How Much Common Belief Is Necessary for a Convention, mimeo Slade, M. E. (1993). "What Does an Oligopoly Maximize? J. Ind. Econ., forthcoming Van Huyck, J., Battalio, R., and Beil, R. (1990). “Tactic Coordination Games, Strategic Uncertainty and Coordination Failure, Amer. Econ. Rev. 80, 234248 Van Huyck, JL., Battalio, R., and Beil, R. (1991). “Strategic Uncertainty, Equilibrium Selection Princi ples, and Coordination Failure in Average Opinion Games, Quart. J. Econ., 885910 Young, H. P. (1993). “The Evolution of Conventions, Econometrica, 61, 5784
