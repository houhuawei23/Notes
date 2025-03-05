# A Course in Game Theory

## Contents

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

## Intro

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

### Temiology

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

## 2 Nash Equilibrium 11

- 2.1 Strategic Games 11
- 2.2 Nash Equilibrium 14
- 2.3 Examples 15
- 2.4 Existence of a Nash Equilibrium 19
- 2.5 Strictly Competitive Games 21
- 2.6 Bayesian Games: Strategic Games with Imperfect Information 24
- Notes 29

### 2.1 Strategic Games

[Definition] **Strategic Games**
A strategic game consists of

- a finite set N (the set of players)
- for each player $i \in N$ a nonempty set $A_i$ (the set of actions available to player i)
- for each player $i \in N$ a preference relation $\gtrsim_i$ on $A = \times_{j \in N} A_j$ (the preference relation of player i).

If the set $A_i$ of actions of every player i is finite then the game is **finite**.

In some situations the players' preferences are most naturally defined not over action profiles but over their consequences. When modeling an oligopoly, for example, we may take the set of players to be a set of firms and the set of actions of each fırm to be the set of prices; but we may wish to model the assumption that each fırm cares only about its profit not about the profile of prices that generates that proft. To do so we introduce a set $C$ of $consequences$, a function $g{:}A\to C$ that associates consequences with action profiles, and a profile $(\succsim_i^*)$ of preference relations over $C$. Then the preference relation $\succsim_i$ of each player $i$ in the strategic game is defined as follows: $a\succsim_ib$ if and only if $g(a)\succsim_i^*g(b)$.

Sometimes we wish to model a situation in which the consequence of an action profile is affected by an exogenous random variable whose realization is not known to the players before they take their actions. We can model such a situation as a strategic game by introducing a set $C$ of consequences, a probability space $\Omega$, and a function $g{:}A\times\Omega\to C$ with the interpretation that $g(a,\omega)$ is the consequence when the action profile is $a\in A$ and the realization of the random variable is $\omega\in\Omega$. A profile of actions induces a lottery on $C;$ for each player $i$ a preference relation $\succsim_i^*$ must be specified over the set of all such lotteries. Player $i’$s preference relation in the strategic game is defined as follows: $a\succsim_ib$ if and only if the lottery over $C$ induced by $g(a,\cdot)$ is at least as good according to $\succsim_i^*$ as the lottery induced by $g(b,\cdot)$.

Under a wide range of circumstances the preference relation $\succsim_i$ of player $i$ in a strategic game can be represented by a **payoff function** $u_i{:}A\to\mathbb{R}$ also called a **utility function** , in the sense that $u_i( a) \geq u_i( b)$ whenever $a\succsim_ib$. We refer to values of such a function as payoffs (or utilities). Frequently we specify a player's preference relation by giving a payoffunction that represents it. In such a case we denote the game by $\langle N,(A_{i}),(u_{i})\rangle$ rather than $\langle N,(A_{i}),(\succsim_{i})\rangle$.

### 2.2 Nash Equilibrium

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

### 2.4 Existence of a Nash Equilibrium

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

### 2.5 Strictly Competitive Games 21

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

### 2.6 Bayesian Games: Strategic Games with Imperfect Information 24

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

## 3 Mixed, Correlated, and Evolutionary Equilibrium 31

- 3.1 Mixed Strategy Nash Equilibrium 31
- 3.2 Interpretations of Mixed Strategy Nash Equilibrium 37
- 3.3 Correlated Equilibrium 44
- 3.4 Evolutionary Equilibrium 48
- Notes 51

## 4 Rationalizability and Iterated Elimination of Dominated Actions 53

- 4.1 Rationalizability 53
- 4.2 Iterated Elimination of Strictly Dominated Actions 58
- 4.3 Iterated Elimination of Weakly Dominated Actions 62
- Notes 64

## 5 Knowledge and Equilibrium 67

- 5.1 A Model of Knowledge 67
- 5.2 Common Knowledge 73
- 5.3 Can People Agree to Disagree? 75
- 5.4 Knowledge and Solution Concepts 76
- 5.5 The Electronic Mail Game 81
- Notes 84

## IV Coalitional Games 9

The primitives of the models we study in Parts I, II, and III (often referred to as“noncooperative” games) are the players' sets of possible actions and their preferences over the possible outcomes, where an outcome is a profile of actions; each action is taken by a single playen autonomously. In this part we study the model of a coalitional game. One primitive of this model is the collection of sets of joint actions that each group of players (coalition) can take independently of the remaining players. An outcome of a coalitional game is a specification of the coalition that forms and the joint action it takes. (More general models, in which many coalitions may form simultaneously, are discussed in the literature.) The other primitive of the model of a coalitional game is the profile of the players' preferences over the set of all possible outcomes. Thus although actions are taken by coalitions, the theory is based (as are the theories in the other parts of the book) on the individuals preferences.

A solution concept for coalitional games assigns to each game a set of outcomes. As before, each solution concept we study captures the consequences of a natural line of reasoning for the participants in a game; it defines a set of arrangements that are stable in some sense. In general the stability requirement is that the outcome be immune to deviations of a certain sort by groups of players; by contrast, most (though not all) solutions for noncooperative games require immunity to deviations by individual players. Many variants of the solution concepts we study are analyzed in the literature; we consider a sample designed to illustrate the main ideas.

A coalitional model is distinguished from a noncooperative model primarily by its focus on what groups of players can achieve rather than on what individual players can do and by the fact that it does not consider the details of how groups of players function internally. If we wish to model the possibility of coalition formation in a noncooperative game then we must specify how coalitions form and how their members choose joint actions. These details are absent from a coalitional game, so that the outcome of such a game does not depend on them.

To illustrate the differences between the two modeling approaches, consider the following situation. Each of a group of individuals owns a bundle of **inputs** and has access to a **technology** for **producing a valuable single output**. **Each individual's inputs are unproductive in his own technology but productive in some other individual's technology**.

- A **noncooperative model** of this situation specifies precisely the set of actions that is available to each individual: perhaps each individual can announce a price vector at which he is willing to trade inputs, or perhaps he can propose a distribution of inputs for the whole of the society.
- A **coalitional model**, by contrast, starts from the sets of payoff vectors that each group of individuals can jointly achieve. A coalition may use contracts, threats, or promises to achieve a high level of production; these institutions are not modeled explicitly in a coalitional game.

We do not view either of the two approaches as superior or more basic. Each of them reflects different kinds of strategic considerations and contributes to our understanding of strategic reasoning. The study of the interconnections between noncooperative and cooperative models can also be illuminating.

## 13 The Core 257

- 13.1 Coalitional Games with Transferable Payoff 257
- 13.2 The Core 258
- 13.3 Nonemptiness of the Core 262
- 13.4 Markets with Transferable Payoff 263
- 13.5 Coalitional Games without Transferable Payoff 268
- 13.6 Exchange Economies 269
- Notes 274

The core is a solution concept for coalitional games that requires that no set of players be able to break away and take a joint action that makes all of them better off. After defning the concept and giving conditions for its nonemptiness, we explore its connection with the concept of a competitive equilibrium in a model of a market.

### 13.1 Coalitional Games with Transferable Payoff

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

### 13.2 The Core

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

### 13.3 Nonemptiness of the Core

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

### 13.5 Coalitional Games without Transferable Payoff

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

## 14 Stable Sets, the Bargaining Set, and the Shapley Value

- 14.1 Two Approaches 277
- 14.2 The Stable Sets of von Neumann and Morgenstern 278
- 14.3 The Bargaining Set, Kernel, and Nucleolus 281
- 14.4 The Shapley Value 289
- Notes 297

In contrast to the core, the solution concepts we study in this chapter restrict the way that an objecting coalition may deviate, by requiring that each possible deviation either itself be a stable outcome or be balanced by a counterdeviation. These restrictions yield several solutions: stable sets, the bargaining set, the kernel, the nucleolus, and the Shapley value.

### 14.1 Two Approaches

The definition of the core does not restrict a coalition’s credible deviations, beyond imposing a feasibility constraint. In particular it assumes that any deviation is the end of the story and ignores the fact that a deviation may trigger a reaction that leads to a different final outcome. The solution concepts we study in this chapter consider various restrictions on deviations that are motivated by these considerations.

In the first approach we study (in Section 14.2), an objection by a coalition to an outcome consists of an alternative outcome that is itself constrained to be stable. The idea is that a deviation by a coalition will lead via some sequence of events to a stable outcome and that a coalition should choose to deviate on the basis of the **ultimate effect** of its action, not the **proximate effect**. This stability condition is self-referential: a stable outcome has the property that no coalition can achieve some other stable outcome that improves the lot of all its members.

In the second approach (studied in Sections 14.3 and 14.4) the chain of events that a deviation unleashes is cut short after two stages:the stability condition is that for every objection to an outcome there is a balancing counterobjection. Different notions of objection and counterobjection give rise to a number of different solution concepts.

The arguments captured by the solution concepts in this chapter are attractive. Nevertheless,it is our impression that there are few persuasive applications of the concepts. Consequently we simply describe the concepts, discuss their interpretations, and give simple examples. Throughout we restrict attention to coalitional games with transferable payoff.

核心的定义并不限制联盟的可信偏离，仅施加可行性约束。具体而言，该定义假定任何偏离都是博弈的最终结局，忽视了偏离可能引发反应从而导致不同最终结果的可能性。本章研究的解概念基于上述考虑，对偏离施加了不同类型的限制条件。

在第一种方法中（我们将在第 14.2 节研究），联盟对结果的异议体现为**提出另一个本身必须满足稳定性条件的替代结果**。其核心思想在于：联盟的偏离行为将通过一系列事件最终导向某个稳定结果，因此联盟的决策应当基于其行为产生的**终极效应**而非**即时效应**。这种稳定性条件具有自指性特征：所谓稳定结果必须满足这样的属性——任何联盟都无法通过偏离行为达成另一个对所有成员均有利的稳定结果。

在第二种方法（第 14.3 和 14.4 节的研究内容）中，**偏离所引发的事件链被限制在两个阶段内完成演化**：其稳定性条件表现为对于每个结果的异议都对应存在平衡性的反异议。通过构建不同形式的异议与反异议框架，可以推导出多种具有差异化解概念的系统解决方案。

本章所阐述的解概念蕴含的论点颇具吸引力。然而，我们的总体印象是这些概念目前尚缺乏具有说服力的实际应用案例。因此，本文仅对这些概念进行描述性阐释，探讨其理论内涵，并通过简单示例加以说明。需要特别说明的是，在全文讨论中，我们仅关注具有可转移支付特征的联盟博弈这一特定类型。

### 14.2 The Stable Sets of von Neumann and Morgenstern

The idea behind the first solution concept we study is that a coalition $S$ that is unsatisfied with the current division of $v(N)$ can **credibly object** by suggesting a **stable division** $x$ of $v(N)$ that is better for all the members of $S$ and is backed up by a threat to implement $(x_i)_{i\in S}$ on its own (by dividing the worth $v(S)$ among its members). The logic behind the requirement that an objection itself be stable is that otherwise the objection may unleash a process involving further objections by other coalitions, at the end of which some of members of the deviating coalition may be worse of.

This idea leads to a definition in which a set of stable outcomes satisfies two conditions:

- (i) for every outcome that is not stable some coalition has a credible objection and
- (ii) no coalition has a credible objection to any stable outcome.

Note that this definition is self-referential and admits the possibility that there be many stable sets.

We now turn to the formal definition.Let $\langle N,v\rangle$ be a coalitional game with transferable payoff. As in the previous chapter we assume that $\langle N,v\rangle$ is cohesive (see Definition 258.1). An imputation of $\langle N,v\rangle$ is a feasible payoff profile $y$ for which $x_i\geq v(\{i\})$ for all $i\in N$ ；let $X$ be the set of all imputations of $\langle N,v\rangle$ .We first define objections (which are not necessarily credible)

An imputation $x$ is an objection of the coalition $S$ to the imputation $y$ if $x_{i}>y_{i}$ for all $i\in S$ and $x(S)\leq v(S)$ , in which case we write $x>s$ y

(In the literature it is sometimes said that “ $x$ dominates $y$ via $S^{\prime\prime}$ if $x$ is an objection of S to $y$ .) Since $\langle N,v\rangle$ is cohesive we have $x>sy$ 3 $y$ if and only if there is an $S$ -feasible payoff vector $(x_i)_{i\in S}$ for which $\mathcal{T}_{i}>y_{i}$ for all $i\in S$ .The core of the game $\langle N,v\rangle$ is the set of all imputations to which there is no objection: $\{y\in X$ : there is no coalition S and imputation $L$ for which $x>sy\}$ .The solution concept we now study is defined as follows.

DEFINITION 279.1

A subset $Y$ of the set $X$ of imputations of a coalitional game with transferable payoff $\langle N,v\rangle$ is a stable set if it satisfies the following two conditions.

**Internal stability**: If $y\in Y$ then for no $z\in Y$ does there exist a coalition $S$ for which $2>sy$

**Eaternal stability**: If $z\in X\setminus Y$ then there exists $y\in Y$ such that $y>s$ for some coalition S

This definition can be written alternatively as follows. For any set $Y$ of imputations let ${\mathcal{D}}(Y)$ be the set of imputations $Z$ for which there is a coalition S and an imputation $y\in Y$ such that $y>s$ .Then internal and external stability are equivalent to the conditions $Y\subseteq X\setminus{\mathcal D}(Y)$ and $Y\supseteq X\setminus\mathcal{D}(Y)$ , so that a set $Y$ of imputations is a stable set if and only if $Y=X\setminus{\mathcal D}(Y)$ While the core is a single set of imputations, a game may have more

than one stable set (see the examples below) or none at all (as shown by the complex example in Lucas (1969)); each such set may contain many imputations. Von Neumann and Morgenstern (1944) interpret each stable set as corresponding to a standard of behavior, the idea being that all the imputations in any given stable set correspond to some mode of behavior while imputations in different stable sets correspond to different modes of behavior

Some simple properties of stable sets are given in the following result.

PROPOSITION 279.2 a.

The core is a subset of every stable set.b.No stable set is a proper subset of any other. c. If the core is a stable set then it is the only stable set.

Proof. a. Every member of the core is an imputation and no member is dominated by an imputation, so the result follows from external stability. $b$ . This follows from external stability. c. This follows from (a) and (b).

ExAMPLE 279.3 (The three-player majority game)

Consider the game $\langle\{1,2,3\},v\rangle$ in which $v(S)=1$ if $|S|\geq2$ and $v(S)=0$ otherwise. One stable set of this game is

$$Y=\{(1/2,1/2,0),(1/2,0,1/2),(0,1/2,1/2)\}.$$

This corresponds to the “standard of behavior" in which some pair of players shares equally the single unit of payoff that is available. The internal stability of $Y$ follows from the fact that for all $x$ and $y$ in $Y$ only one player prefers $x$ to $y$ .To check external stability, let $x$ be an imputation outside $Y$ Then there are two players $i$ and $j$ for whom

$z_{i}<\frac12$ and $z_{j}$ < $\frac 12$ ，so that there is an imputation in $Y$ that is an objection of $\{i,j\}$ to $x$ For any $c\in[0,\frac{1}{2})$ and any $i\in\{1,2,3\}$ the set

$$Y_{i,c}=\{x\in X\colon x_i=c\}$$

is also a stable set of the game.This corresponds to a “standard of behavior" in which one of the players is singled out and given a fixed payoff. The internal stability of $Y_{i,c}$ follows from the fact that for any $L$ and $y$ in the set there is only one player who prefers $x$ to $y$ . To show the external stability of $Y_{i,c}$ let $i=3$ and let $Z$ be an imputation outside $Y_{3,c}$ .If $z_{3}>c$ then $z_{1}+z_{2}<1-c$ and there exists $x\in Y_{3,c}$ such that $x_1\gg z_1$ and $x_{2}\gg z_{2}$ , so that ${\mathcal{T} }\succ \{ 1, 2\}$ ${\mathcal{Z} }$ If $23<c$ and, say, $z_1\leq z_2$ then $(1-c,0,c)\succ_{\{1,3\}}z$

Exercise 280.1 (Simple games)

Let $\langle N,v\rangle$ be a simple game (see Exercise 261.1). Let $T$ be a minimal winning coalition (a winning coalition that has no strict subset that is winning). Show that the set of imputations that assign 0 to all players not in $T$ is a stable set.

Exercise 280.2 (A market for an indivisible good)

For the market described in Example 260.1 with $|B|\geq|L|$ show that the set

$$Y=\{x\in X\colon x_i=x_j\text{if}\:i,j\in L\text{or}\:i,j\in B\},$$

is a stable set;interpret it.

Exercise 280.3 (Three-player games)

For a three-player game the set of imputations can be represented geometrically as an equilateral triangle with height $v(N)$ in which each point represents the imputation whose components are the distances to each edge. (Thus the corners correspond to the three imputations that assign $v(N)$ to a single player.) Use such a diagram to find the general form of a stable set of the three-player game in which $v(\{1,2\})=\beta<1$, $v(\{1,3\})=v(\{1,2,3\})=1$ , and $v(S)=0$ otherwise. We can interpret this game as a market in which player 1 is a seller and players 2 and 3 are buyers with reservation values $\beta$ and 1 respectively. Interpret the stable sets of the game in terms of this market

EXERCISE 280.4

Player $\dot{\boldsymbol{z}}$ is a dummy in $\langle N,v\rangle$ if $v(S\cup\{i\})-v(S)=$ $v(\{i\})$ for every coalition $S$ of which $i$ is not a member. Show that if player $i$ is a dummy in $\langle N,v\rangle$ then his payof in any imputation in any stable set is $v(\{i\})$

EXERCISE 280.5

Let $X$ be an arbitrary set (of outcomes) and let $D$ be a binary relation on $X$ ，with the interpretation that if $x$ $D$ $y$ then

$L$ is an objection of some coalition S to $y$ Generalize the definition of stable sets as follows. The set $Y\subseteq X$ of outcomes is stable if it satisfies the following two conditions.

Internal stability If $y\in Y$ then there exists no $z\in Y$ such that $zD$ y. External stability If $z\in X\setminus Y$ then there exists $y\in Y$ such that $y$ $D$ $z$ Consider an exchange economy (see Section 13.6) in which there are two goods and two agents. Let $X$ be the set of all allocations $x$ for which $x_i\succsim_i\omega_i$ for each agent $i$ .Define the relation $D$ by $x$ $D$ $y$ if both agents prefer 2 to $y$ .Show that the only (generalized) stable set is the core of the economy.

### 14.2 冯·诺依曼和摩根斯坦的稳定集

我们研究的第一个解概念的核心思想是，如果一个联盟 $S$ 对当前 $v(N)$ 的分配不满意，它可以通过提出一个对 $S$ 所有成员更有利的**稳定分配** $x$ 来**可信地反对**，并通过威胁自行实施 $(x_i)_{i\in S}$（即在联盟成员之间分配 $v(S)$）来支持这一反对。要求反对本身必须是稳定的逻辑在于，否则反对可能会引发其他联盟的进一步反对，最终导致偏离联盟的某些成员处境更糟。

这一思想引出了一个定义，其中一组稳定结果满足两个条件：

- **(i)** 对于每一个不稳定的结果，存在某个联盟有可信的反对；
- **(ii)** 对于任何稳定结果，没有任何联盟有可信的反对。

需要注意的是，这个定义是自指的，并且允许多个稳定集的存在。

现在我们转向正式的定义。设 $\langle N,v\rangle$ 是一个具有可转移支付的联盟博弈。与前一章一样，我们假设 $\langle N,v\rangle$ 是凝聚的（见定义 258.1）。$\langle N,v\rangle$ 的一个**分配**（imputation）是一个可行的支付剖面 $x$，满足 $x_i \geq v(\{i\})$ 对所有 $i\in N$ 成立；设 $X$ 为 $\langle N,v\rangle$ 的所有分配的集合。我们首先定义反对（不一定是可信的）：

一个分配 $x$ 是联盟 $S$ 对分配 $y$ 的**反对**，如果 $x_i > y_i$ 对所有 $i\in S$ 成立，并且 $x(S) \leq v(S)$，此时我们记作 $x >_S y$。

（在文献中，有时会说“$x$ 通过 $S$ 支配 $y$”，如果 $x$ 是 $S$ 对 $y$ 的反对。）由于 $\langle N,v\rangle$ 是凝聚的，$x >_S y$ 当且仅当存在一个 $S$ 可行的支付向量 $(x_i)_{i\in S}$，使得 $x_i > y_i$ 对所有 $i\in S$ 成立。博弈 $\langle N,v\rangle$ 的**核心**是所有没有反对的分配的集合：$\{y \in X$：不存在联盟 $S$ 和分配 $x$ 使得 $x >_S y\}$。我们现在研究的解概念定义如下。

**定义 279.1**  
具有可转移支付的联盟博弈 $\langle N,v\rangle$ 的分配集 $X$ 的一个子集 $Y$ 是一个**稳定集**，如果它满足以下两个条件：

- **内部稳定性**：如果 $y \in Y$，则不存在 $z \in Y$ 和联盟 $S$ 使得 $z >_S y$。
- **外部稳定性**：如果 $z \in X \setminus Y$，则存在 $y \in Y$ 和联盟 $S$ 使得 $y >_S z$。

这个定义可以改写如下。对于任何分配集 $Y$，设 ${\mathcal{D}}(Y)$ 为所有存在联盟 $S$ 和分配 $y \in Y$ 使得 $y >_S z$ 的分配 $z$ 的集合。那么内部稳定性和外部稳定性分别等价于条件 $Y \subseteq X \setminus {\mathcal{D}}(Y)$ 和 $Y \supseteq X \setminus {\mathcal{D}}(Y)$，因此分配集 $Y$ 是稳定集当且仅当 $Y = X \setminus {\mathcal{D}}(Y)$。虽然核心是一个单一的分配集，但一个博弈可能有多个稳定集（见下面的例子）或根本没有（如 Lucas (1969) 中的复杂例子所示）；每个这样的集可能包含许多分配。冯·诺依曼和摩根斯坦 (1944) 将每个稳定集解释为对应于一种行为标准，即任何给定稳定集中的所有分配对应于某种行为模式，而不同稳定集中的分配对应于不同的行为模式。

以下结果给出了一些稳定集的简单性质。

**命题 279.2**  
a. 核心是每个稳定集的子集。  
b. 任何稳定集都不是其他稳定集的真子集。  
c. 如果核心是一个稳定集，那么它是唯一的稳定集。

**证明**：  
a. 核心的每个成员都是一个分配，且没有任何成员被分配支配，因此结果由外部稳定性得出。  
b. 这由外部稳定性得出。  
c. 这由 (a) 和 (b) 得出。

**例 279.3（三人多数博弈）**  
考虑博弈 $\langle\{1,2,3\},v\rangle$，其中 $v(S)=1$ 如果 $|S|\geq2$，否则 $v(S)=0$。该博弈的一个稳定集是

$$Y=\{(1/2,1/2,0),(1/2,0,1/2),(0,1/2,1/2)\}.$$

这对应于一种“行为标准”，即某对玩家平分可用的单一单位支付。$Y$ 的内部稳定性由以下事实得出：对于 $Y$ 中的所有 $x$ 和 $y$，只有一个玩家更偏好 $x$ 而非 $y$。为了检查外部稳定性，设 $x$ 是 $Y$ 之外的一个分配。那么存在两个玩家 $i$ 和 $j$，使得 $x_i < \frac{1}{2}$ 且 $x_j < \frac{1}{2}$，因此存在 $Y$ 中的一个分配是 $\{i,j\}$ 对 $x$ 的反对。对于任何 $c\in[0,\frac{1}{2})$ 和任何 $i\in\{1,2,3\}$，集合

$$Y_{i,c}=\{x\in X\colon x_i=c\}$$

也是该博弈的一个稳定集。这对应于一种“行为标准”，即某个玩家被特别指定并获得固定支付。$Y_{i,c}$ 的内部稳定性由以下事实得出：对于集合中的任何 $x$ 和 $y$，只有一个玩家更偏好 $x$ 而非 $y$。为了展示 $Y_{i,c}$ 的外部稳定性，设 $i=3$，并设 $z$ 是 $Y_{3,c}$ 之外的一个分配。如果 $z_3 > c$，则 $z_1 + z_2 < 1 - c$，并且存在 $x \in Y_{3,c}$ 使得 $x_1 \gg z_1$ 且 $x_2 \gg z_2$，因此 $x >_{\{1,2\}} z$。如果 $z_3 < c$ 且，例如 $z_1 \leq z_2$，则 $(1-c,0,c) >_{\{1,3\}} z$。

**练习 280.1（简单博弈）**  
设 $\langle N,v\rangle$ 是一个简单博弈（见练习 261.1）。设 $T$ 是一个最小获胜联盟（一个没有严格子集是获胜的获胜联盟）。证明将所有不在 $T$ 中的玩家分配为 0 的分配集是一个稳定集。

**练习 280.2（不可分割商品的市场）**  
对于例 260.1 中描述的市场，假设 $|B|\geq|L|$，证明集合

$$Y=\{x\in X\colon x_i=x_j\text{如果}\:i,j\in L\text{或}\:i,j\in B\},$$

是一个稳定集；并解释其含义。

**练习 280.3（三人博弈）**  
对于三人博弈，分配集可以用一个高度为 $v(N)$ 的等边三角形几何表示，其中每个点表示其分量为到各边距离的分配。（因此，角对应于将 $v(N)$ 分配给单个玩家的三个分配。）使用这样的图来找到三人博弈的稳定集的一般形式，其中 $v(\{1,2\})=\beta<1$，$v(\{1,3\})=v(\{1,2,3\})=1$，否则 $v(S)=0$。我们可以将该博弈解释为一个市场，其中玩家 1 是卖家，玩家 2 和 3 是保留价值分别为 $\beta$ 和 1 的买家。解释该博弈的稳定集在这个市场中的含义。

**练习 280.4**  
玩家 $i$ 在 $\langle N,v\rangle$ 中是一个**傀儡**，如果对于任何不包含 $i$ 的联盟 $S$，$v(S\cup\{i\})-v(S)=v(\{i\})$。证明如果玩家 $i$ 在 $\langle N,v\rangle$ 中是一个傀儡，那么他在任何稳定集中的任何分配中的支付都是 $v(\{i\})$。

**练习 280.5**  
设 $X$ 是一个任意集合（结果集），$D$ 是 $X$ 上的一个二元关系，解释为如果 $x D y$，则 $x$ 是某个联盟 $S$ 对 $y$ 的反对。将稳定集的定义推广如下。结果集 $Y\subseteq X$ 是稳定的，如果它满足以下两个条件：

- **内部稳定性**：如果 $y \in Y$，则不存在 $z \in Y$ 使得 $z D y$。
- **外部稳定性**：如果 $z \in X \setminus Y$，则存在 $y \in Y$ 使得 $y D z$。

考虑一个交换经济（见第 13.6 节），其中有两种商品和两个代理。设 $X$ 为所有满足 $x_i \succsim_i \omega_i$ 的分配 $x$ 的集合。定义关系 $D$ 为 $x D y$ 如果两个代理都更偏好 $x$ 而非 $y$。证明唯一的（广义）稳定集是该经济的核心。

### 14.3 The Bargaining Set,Kernel, and Nucleolus

We now turn to the second approach that we described at the start of the chapter.That is,we regard an objection by a coalition to be convincing if no other coalition has a “balancing" counterobjection; we do not require the objection or counterobjection to be themselves stable in any sense. We study three solution concepts that differ in the nature of the objections and counterobjections

### 14.3.1 The Bargaining Set

Let $x$ be an imputation in a coalitional game with transferable payoff $\langle N,v\rangle$ . Define objections and counterobjections as follows.

- A pair $(y,S)$ ,where S is a coalition and $y$ is an S -feasible payoff vector, is an objection of $i$. against $j$ to $x$ if S includes $i$ but not $j$ and $y_{k}>x_{k}$ for all $k\in S$
- A pair $(z,T)$ ,where $T$ is a coalition and $Z$ is a $T$ -feasible payoff vector,is a counterobjection to the objection $(y,S)$ of $\dot{\tau}$ against $j$ if $T$ includes $j$ but not $i$, $z_k\geq x_k$ for all $k\in T\setminus S$ ,and $z_k\geq y_k$ for all $k\in T\cap S$

Such an objection is an argument by one player against another. An objection of $i$ against $j$ to $x$ specifies a coalition S that includes $i$ but not $j$ and a division $y$ of $v(S)$ that is preferred by all members of $S$ to $x$ .A counterobjection to $(y,S)$ by $j$ specifies an alternative coalition $T$ that contains $j$ but not $i$ and a division of $v(T)$ that is at least as good as $y$ for all the members of $T$ who are also in S and is at least as good as $y$ for the other members of $T$ .The solution concept that we study is defined as follows

DEFINITION 282.1

The bargaining set of a coalitional game with transferable payoff is the set of all imputations $x$ with the property that for every objection $(y,S)$ of any player $i$ against any other player $j$ to $x$ there is a counterobjection to $(y,S)$ by $j$.

The bargaining set models the stable arrangements in a society in which any argument that any player $i$ makes against an imputation $y$ takes the following form: “I get too little in the imputation $L$ and $j$ gets too much; I can form a coalition that excludes $j$ in which everybody is better off than in 2 ".Such an argument is ineffective as far as the bargaining set is concerned if player $j$ can respond as follows: “Your demand is not justified;I can form a coalition that excludes you in which everybody is at least as well off as they are in $x$ and the players who participate in your coalition obtain at least what you offer them." The bargaining set, like the other solution concepts in this section, assumes that the argument underlying an objection for which there is no counterobjection undermines the stability of an outcome. This fact is taken as given, and is not derived from more primitive assumptions about the players’behavior.The appropriateness of the solution in a particular situation thus depends on the extent to which the participants in that situation regard the existence of an objection for which there is no counterobjection as a reason to change the outcome.

Note that an imputation is in the core if and only if no player has an objection against any other player; hence the core is a subset of the bargaining set. We show later (in Corollary 288.3) that the bargaining set of every game is nonempty.

EXAMPLE 282.2 (The three-player majority game)

Consider the three player majority game.The core of this game is empty (see Example 259.1) and the game has many stable sets (see Example 279.3). The bargaining set of the game is the singleton $\{ ( \frac 13$, $\frac 13$, $\frac 13) \}$ , by the following argument. Let $x$ be an imputation and suppose that $(y,S)$ is an objec tion of 2 against $j$ to 2 .Then we must have $S=\{i,h\}$ ,where $h$ is the third player and $y_{h}<1-x_{i}$ (since $y_{i}\gg x_{i}$ and $y(S)=v(S)=1$ ).For $j$ to have a counterobjection to $(y,S)$ we need $y_{h}+x_{j}\leq1$ . Thus for $L$ to be in the bargaining set we require that for all players $i,j$ , and $h$ we have $y_{h}\leq1-x_{j}$ whenever $y_{h}<1-x_{i}$ , which implies that $1-x_{i}\leq1-x_{j}$ OT $x_j\leq x_i$ for all $i$ and $j$ , so that $x=(\frac{1}{3},\frac{1}{3},\frac{1}{3})$ . Obviously this imputation is in the bargaining set.

EXAMPLE 282.3 (My aunt and $I$)

Let $\langle\{1,2,3,4\},v\rangle$ be a simple game (see Exercise 261.1) in which $v(S)=1$ if and only if S contains one of

the coalitions $\{2,3,4\}$ or $\{1,i\}$ for $i\in\{2,3,4\}$ (Player 2 is \*I and player 1 is his aunt.) In this game, player 1 appears to be in a stronger position than the other players since she needs the cooperation of only one player to form a winning coalition. If 2 is an imputation for which $x_2<\boldsymbol{T}_3$ then player 2 has an objection against 3 (via $\{1,2\}$ )to 2 for which there is no counterobjection.Thus if $L$ is in the bargaining set then $x_{2}=x_{3}=x_{4}=0$ , say. Any objection of player 1 against player 2 to $x$ takes the form $(y,\{1,j\})$ where $j=3$ or 4 and $y_{j}<3\alpha$ ；;there is no counterobjection if and only if $\alpha+3\alpha+\alpha>1$ ，or $\alpha>\frac{1}{5}$ .An objection of player 2 against 1 to $x$ must use the coalition $\{2,3,4\}$ and give one of the players 3 or 4 less than $(1-\alpha)/2$ ; player 1 does not have a counterobjection if and only if $1-3\alpha+(1-\alpha)/2>1$ ，or $\alpha<\frac17$ Hence the bargaining set is $\{(1-3\alpha,\alpha,\alpha,\alpha):\frac{1}{7}\leq\alpha\leq\frac{1}{5}\}$ .

Note that by contrast the core is empty. We saw (Example 265.1) that the competition inherent in the core can drive to zero the payof of players holding goods that are in excess supply. The following exercise gives an example that shows how this intense competition is muted in the bargaining set.

Exercise 283.1 (A market) Consider the coalitional game derived from the market with transferable payoff in Exercise 265.2. Show that the bargaining set of this game is $\left\{(\alpha,\alpha,\beta,\beta,\beta);0\leq\alpha\leq\frac{3}{2}\right.$ and $2\alpha+3\beta=3\}$ . Contrast this set with the core and give an interpretation.

### 14.3.2 The Kernel

We now describe another solution that, like the bargaining set, is defined by the condition that to every objection there is a counterobjec tion; it differs from the bargaining set in the nature of objections and counterobjections that are considered effective

Let 2 be an imputation in a coalitional game with transferable payoff $\langle N,v\rangle$ ; for any coalition S call $e(S,x)=v(S)-x(S)$ the ezcess of $S$ If the excess of the coalition $S$ is positive then it measures the amount that $S$ has to forgo in order for the imputation $L$ to be implemented;it is the sacrifice that S makes to maintain the social order. If the excess of S is negative then its absolute value measures the amount over and above the worth of S that S obtains when the imputation $y$ is implemented it is S 's surplus in the social order. A player $i$ objects to an imputation $x$ by forming a coalition $S$ that

excludes some player $j$ for whom $x_{j}$ > $v( \{ j\} )$ and pointing out that

he is dissatisfied with the sacrifice or gain of this coalition. Player $j$ counterobjects by pointing to the existence of a coalition that contains $j$ but not $\dot{\boldsymbol{z}}$ and sacrifices more (if $e(S,x)>0$ ) or gains less (if $e(S,x)<0$ ) More precisely, define objections and counterobjections as follows.

- A coalition $S$ is an objection of $i$ against $j$ to $y$ if $S$ includes $i$ but not $j$ and $x_{j}>v(\{j\}$ . - A coalition $TI$ is counterobjection to the objection $S$ of $i$ against $j$ if $T$ includes $j$ but not 2 and $e(T,x)\geq e(S,x)$

DEFINITION 284.1

The kernel of a coalitional game with transferable payoff is the set of all imputations $x$ with the property that for every objection $S$ of any player $i$ against any other player $j$ to $x$ there is a counterobjection of $j$ to $S$.

For any two players $\dot{\boldsymbol{z}}$ and $j$ and any imputation 2 define $s_{ij}(x)$ to be the maximum excess of any coalition that contains $i$ but not $j$

$$s_{ij}(x)=\max_{S\in\mathcal{C}}\{e(S,x)\colon i\in S\mathrm{~and~}j\in N\setminus S\}.$$

Then we can alternatively define the kernel to be the set of imputations $x\in X$ such that for every pair $(i,j)$ of players either $s_{ji}(x)\geq s_{ij}(x)$ Or $x_{j}=v(\{j\})$ The kernel models the stable arrangements in a society in which a

player makes arguments of the following type against an imputation 2 “"Here is a coalition to which I belong that excludes player $j$ and sacrifices too much (or gains too little)". Such an argument is ineffective as far as the kernel is concerned if player $j$ can respond by saying “your demand is not justified; I can name a coalition to which I belong that excludes you and sacrifices even more (or gains even less) than the coalition that you name".

Note that the definitions of the core and the bargaining set do not require us to compare the payoffs of different players, while that of the kernel does. Thus the definitions of the former concepts can easily be extended to a general coalitional game $\langle N,X,V,(\succsim_{i})\rangle$ (see Defini tion 268.2). For example, as we saw in Section 13.5, the core is the set of all $x\in V(N)$ for which there is no coalition S and $y\in V(S)$ for which $y\succ_{i}x$ for all $i\in S$ .By contrast, the definition of the kernel cannot be So extended; it assumes that there is meaning to the statement that the excess of one coalition is larger than that of another. Thus the kernel is an appropriate solution concept only in situations in which the payoffs of different players can be meaningfully compared.

We show later that the kernel is nonempty (see Corollary 288.3).Its relation with the bargaining set is as follows.

LEMMA 285.1: The kernel of a coalitional game with transferable payoff is a subset of the bargaining set.

Proof. Let $\langle N,v\rangle$ be a coalitional game with transferable payoff, let 2 be an imputation in the kernel, and let $(y,S)$ be an objection in the sense of the bargaining set of player 2 against $j$ to 2 : $i\in S$, $j\in N\setminus S$ $y(S)=v(S)$ , and $y_{k}>x_{k}$ for all $k\in S$ .If $x_{j}=v(\{j\})$ then $(z,\{j\})$ with $z_{j}=v(\{j\})$ is a counterobjection to $(y,S)$ .If $x_{j}>v(\{j\})$ then since 2 is in the kernel we have $s_{ji}(x)\geq s_{ij}(x)\geq v(S)-x(S)=y(S)-x(S)$ . Let $T$ be a coalition that contains $j$ but not $i$ for which $s_{ji}(x)=v(T)-x(T)$ Then $v(T)-x(T)\geq y(S)-x(S)$ , so that $v(T)\geq y(S\cap T)+y(S\setminus T)+$ $x(T\setminus S)-x(S\setminus T)>y(S\cap T)+x(T\setminus S)$ , since $y(S\backslash T)>x(S\backslash T)$ .Thus there exists a $T$ -feasible payoff vector 2 with $z_k\geq x_k$ for all $k\in T\setminus S$ and $z_k\geq y_k$ for all $k\in T\cap S$ ，so that $(z,T)$ is a counterobjection to $(y,S)$.

EXAMPLE 285.2 (The three-player majority game)

It follows from our calculation of the bargaining set (Example 282.2),the previous lemma (285.1), and the nonemptiness of the kernel that the kernel of the threeplayer majority game is $\{(\frac{1}{3},\frac{1}{3},\frac{1}{3})\}$ .To see this directly, assume that $x_{1}\geq x_{2}\geq x_{3}$ ，with at least one strict inequality. Then $s_{31}(x)=1-$ $x_{2}-x_{3}>1-x_{2}-x_{1}=s_{13}(x)$ and $x_{1}>0=v(\{1\})$ , so that 2 : is not in the kernel

EXAMPLE 285.3 ( $My$ aunt and $I$ )

The kernel of the game in Exam ple 282.3 is $\{(\frac{2}{5},\frac{1}{5},\frac{1}{5},\frac{1}{5})\}$ , by the following argument. Let $x$ be in the kernel. By Lemma 285.1 and the calculation of the bargaining set of the game we have $x$ = $( 1- 3\alpha , \alpha , \alpha , \alpha )$ for some $\frac 17$ $\leq$ $\alpha$ $\leq$ $\frac 15$ ，S0 that $s_{12}( x)$ = $2\alpha$ and $s_{21}( x)$ = 1- $3\alpha$ Since $1-3\alpha>0$ we need $s_{12}(x)=2\alpha\geq s_{21}(x)=1-3\alpha$ :or $\alpha\geq\frac{1}{5}$ ; hence $\alpha=\frac{1}{5}$

### 14.3.3 The Nucleolus

A solution that is closely related to the kernel is the nucleolus.Let $y$ be an imputation in a coalitional game with transferable payoff. Define objections and counterobjections as follows.

- A pair $(S,y)$ consisting of a coalition $S$ and an imputation $y$ is an objection to $x$ if $e(S,x)>e(S,y)$ (i.e. $y(S)>x(S)$ )
- A coalition $T$ is a counterobjection to the objection $(S,y)$ if $e(T,y)>e(T,x)$ i.e. $x(T)>y(T)$ ）and $e(T,y)\geq e(S,x)$.

DEFINIrION 286.1 The nucleolus of a coalitional game with transferable payoff is the set of all imputations $x$ with the property that for every objection $(S,y)$ to $x$ there is a counterobjection to $(S,y)$

As for the kernel the idea is that the excess of S is a measure of S 's dissatisfaction with $x$ : it is the price that S pays to tolerate $x$ rather than secede from $N$ .In the definition of the kernel an objection is made by a single player, while here an objection is made by a coalition. An objection $(S,y)$ may be interpreted as a statement by $S$ of the form “our excess is too large in $x$ ; we suggest the alternative imputation $y$ in which it is smaller". The nucleolus models situations in which such objections cause outcomes to be unstable only if no coalition $T$ can respond by saying “your demand is not justified since our excess under $y$ is larger than it was under $y$ and furthermore exceeds under $y$ what yours was under $x$ ”.Put differently, an imputation fails to be stable according to the nucleolus if the excess of some coalition $S$ can be reduced without increasing the excess of some coalition to a level at least as large as that of the original excess of $S$ This definition of the nucleolus,which is not standard, facilitates a

comparison with the kernel and the bargaining set and is easier to interpret than the standard definition, to which we now show it is equivalent For any imputation $x$ let $S_{1},\ldots,S_{2|N|-1}$ be an ordering of the coali-

tions for which $e( S_{\ell }, x)$ $\geq$ $e( S_{\ell + 1}, x)$ for $\ell$ = $1, \ldots , 2^{| N| }- 2$ and let $E(x)$ be the vector of excesses defined by $E_{\ell }( x)$ = $e( S_{\ell }, x)$ for all $\ell=$ $1,\ldots,2^{|N|}-1$ .Let $B_{1}(x),\ldots,B_{K}(x)$ be the partition of the set of allcoalitions in which S and $S^{\prime}$ are in the same cell if and only if $e( S, x)$ = $e( S^{\prime }, x)$ .For any $S\in B_{k}(x)$ let $e( S, x)$ = $e_{k}( x)$ ，so that $e_{1}(x)>e_{2}(x)>\cdots>e_{K}(x)$ We say that $E(x)$ is lezicographically less than $E(y)$ if $E_{\ell}(x)<E_{\ell}(y)$

for the smallest $E$ for which $E_{\ell}(x)\neq E_{\ell}(y)$ ,or equivalently if there exists $k^{*}$ such that for all $k<k^{*}$ we have $|B_{k}(x)|=|B_{k}(y)|$ and $e_k(x)=e_k(y)$ and either (i) $e_{k^*}(x)<e_{k^*}(y)$ or (ii) $e_{k^*}(x)=e_{k^*}(y)$ and $|B_{k^{*}}(x)|<$ $|B_{k}\cdot(y)|$

LEMMA 286.2 The nucleolus ofa coalitional game with transferable pay off is the set of imputations z for which the vector $E(x)$ is lezicograph ically minimal

Proof. Let $\langle N,v\rangle$ be a coalitional game with transferable payoff and let $x$ be an imputation for which $E(x)$ is lexicographically minimal. To show that $x$ is in the nucleolus, suppose that $(S,y)$ is an objection to $x$ ，so that $e( S, y)$ < $e( S, x)$ .Let $k^{*}$ be the maximal value of $k$ such that $e_k( x)$ = $e_k( y)$ and $B_{k}( x)$ = $B_{k}( y)$ (not just $| B_{k}( x) |$ = $| B_{k}( y) | )$ for all $k<k^*$ .Since $E(y)$ is not lexicographically less than $E(x)$ we have either (i) $e_{k^*}(y)>e_{k^*}(x)$ or (i) $e_{k^*}(x)=e_{k^*}(y)$ and $|B_{k^{*}}(x)|\leq$ $|B_{k}\cdot(y)|$ .In either case there is a coalition $T\in B_{k^{*}}(y)$ with $e_{k^*}(y)=$ $e( T, y)$ > $e( T, x)$ .We now argue that $e( T, y)$ $\geq$ $e( S, x)$ ，so that $T$ is a counterobjection to $(S,y)$ .Since $e( S, y)$ < $e( S, x)$ we have $S\notin$ $\cup_{k=1}^{k^{*}-1}B_{k}(x)$ and hence $e_{k^*}(x)\geq e(S,x)$ : since $e_{k^*}(y)\geq e_{k^*}(x)$ we have $e(T,y)\geq e(S,x)$ Now assume that $x$ is in the nucleolus and that $E(y)$ is lexicographi-

cally less than $E(x)$ .Let $k^{*}$ be the smallest value of $k$ for which $B_{k}(x)=$ $B_{k}(y)$ for all $k<k^{*}$ and either (i) $e_{k^{*}}(y)<e_{k^{*}}(x)$ or (ii) $e_{k}\cdot(y)=$ $e_{k^{-}}(x)$ and $B_{k^{*}}(y)\neq B_{k^{*}}(x)$ (and hence $|B_{k^{*}}(y)|\neq|B_{k}.(x)|)$ .In either case there exists a coalition $S\in B_{k}.(x)$ for which $e( S, y)$ < $e( S, x)$ Let $\lambda\in(0,1)$ and let $z( \lambda )$ = $\lambda x+$ $( 1- \lambda ) y$ we have $e( R, z( \lambda ) )$ = $\lambda e(R,x)+(1-\lambda)e(R,y)$ for any coalition $R$ .We claim that the pair $(S,z(\lambda))$ is an objection to $x$ for which there is no counterobjection.It is an objection since $e(S,z(\lambda))<e(S,x)$ .For $T$ to be a counterobjec tion we need both $e(T,z(\lambda))>e(T,x)$ and $e(T,z(\lambda))\geq e(S,x)$ .How ever, if $e(T,z(\lambda))>e(T,x)$ then $e(T,y)>e(T,x)$ ,which implies that $T\notin\cup_{k=1}^{k^{*}}B_{k}(x)$ and hence $e(S,x)>e(T,x)$ .Also, since $T\notin\cup_{k=1}^{k^{*}-1}B_{k}(y)$ we have $e(S,x)=e_{k^{*}}(x)\geq e_{k^{*}}(y)\geq e(T,y)$ . Thus $e(S,x)>e(T,z(\lambda))$ We conclude that there is no counterobjection to $(S,z(\lambda))$

The nucleolus is related to the kernel as follows

LEMMA 287.1 The mucleolus of a coalitional game with transferable payoff is a subset of the kernel.

Proof. Let $\langle N,v\rangle$ be a coalitional game with transferable payoff and let $x$ be an imputation that is not in the kernel of $\langle N,v\rangle$ .We show that $L$ is not in the nucleolus of $\langle N,v\rangle$ .Since $y$ is not in the kernel there are players $i$ and $j$ for which $s_{ij}(x)>s_{ji}(x)$ and $x_{j}>v(\{j\}$ 0.Since $x_{j}>$ $v( \{ j\} )$ there exists $\epsilon>0$ such that $y$ = $x+ \epsilon$ $1_{\{ i\} }- \epsilon$ $\mathrm{l} _{\{ j\} }$ is an imputation (where $\mathbb{I}_{\{k\}}$ is the $k$ th unit vector); choose $t$ small enough that $s_{ij}(y)>s_{ji}(y)$ .Note that $e(S,x)<e(S,y)$ if and only if $S$ contains $i$ but not $j$ and $e(S,x)>e(S,y)$ if and only if S contains $j$ but not $x$ . Let $k^{*}$ be the minimal value of $k$ for which there is a coalition $S\in B_{k^{\cdot}}(x)$ with $e( S, x)$ $\neq$ $e( S, y)$ .Since $s_{ij}( x)$ > $s_{ji}( x)$ the set $B_{k^{*}}(x)$ contains

at least one coalition that contains $i$ but not $j$ and no coalition that contains $j$ but not $x$ .Further, for all $k<k^{*}$ we have $B_{k}(y)=B_{k}(x)$ and $e_{k}(y)=e_{k}(x)$ . Now, if $B_{k^{*}}(x)$ contains coalitions that contain both $i$ and $j$ or neither of them then $e_{k^*}(y)=e_{k^*}(x)$ and $B_{k}.(y)$ is a strict subset of $B_{k}\cdotp(x)$ .If not, then since $s_{ij}(y)>s_{ji}(y)$ we have $e_{k^*}(y)<e_{k^*}(x)$ . In both cases $E(y)$ is lexicographically less than $E(x)$ and hence $L$ is not in the nucleolus of $\langle N,v\rangle$

We now show that the nucleolus of any game is nonempty

PROPOsrTION 288.1 The nucleolus of any coalitional game with transferable payoff is nonempty.

Proof.First we argue that for each value of $k$ the function $E_{k}$ is continuous. This follows from the fact that for any $k$ we have

$$E_k(x)=\min_{T\in\mathcal C^{k-1}}\max_{S\in\mathcal C\setminus T}e(S,x),$$

where $\mathcal{C} ^{0}$ = $\{ \varnothing \}$ and $\mathcal{C}^{k}$ for $k\geq1$ is the set of all collections of $k$ coalitions. Since $E_{1}$ is continuous the set $X_{1}$ = $\arg \min _{x\in X}E_{1}( x)$ is nonempty and compact. Now, for each integer $k\geq1$ define $X_{k+1}=$ $\arg\operatorname*{min}_{x\in X_k}E_{k+1}(x)$ .By induction every such set is nonempty and compact; since $X_{2|N|-1}$ is the nucleolus the proof is complete.

This result immediately implies that the bargaining set and kernel of any game are nonempty

Corollary 288.3 The bargaining set and kernel of any coalitional game writh transferable payoff are nonempty

Proof. This follows from the nonemptiness of the nucleolus (Propo sition 288.1) and the facts that the nucleolus is a subset of the kernel (Lemma 287.1) and the kernel is a subset of the bargaining set (Lemma 285.1).

As we have seen above the bargaining set of a game may contain many imputations; the same is true of the kernel. However, the nucleolus is always a singleton, as the following result shows.

Proposition 288.4 The nucleolus of any coalitional game with transferable payoff is a singleton

Proof. Let $\langle N,v\rangle$ be a coalitional game with transferable payoff. Suppose that the imputations $x$ and $y$ are both in the nucleolus, so that $E(x)=$ $E(y)$ .We show that for any coalition S we have $e(S,x)=e(S,y)$ and hence, in particular, for any player $i$ we have $e(\{i\},x)=e(\{i\},y)$, so that ${:}E=y$ .Assume there is at least one coalition $S^{*}$ with $e(S^{*},x)\neq$ $e(S^{*},y)$ and consider the imputation $z=\frac{1}{2}(x+y)$ . Since $E_{k}(x)=E_{k}(y)$ for all $k$ we have $e_k( x)$ = $e_k( y)$ and $| B_{k}( x) |$ = $| B_{k}( y) |$ for all $k$ .But since $e(S^{*},x)\neq e(S^{*},y)$ there exists a minimal value $k^{*}$ of $k$ for which $B_{k^{*}}(x)\neq B_{k^{*}}(y)$ . Now, if $B_{k^{*}}(x)\cap B_{k^{*}}(y)\neq\varnothing$ then $B_{k^{*}}(z)=B_{k^{*}}(x)\cap$ $B_{k^{-}}(y)\subset B_{k^{+}}(x)$ ;if $B_{k}.(x)\cap B_{k^{*}}(y)=\varnothing$ then $e_{k^*}(z)<e_{k^*}(x)=e_{k^*}(y)$ In both cases $E(z)$ is lexicographically less than $E(x)$ , contradicting the fact that $x$ is in the nucleolus

Exercise 289.1(A production economy)

Show that the single impu tation in the nucleolus of the game in Exercise 259.3, which models a production economy with one capitalist and $U$ workers, gives each worker $\frac{1}{2}[f(w)-f(w-1)]$ .(Note that since the nucleolus is a singleton you need only to verify that the imputation is in the nucleolus).

Exercise 289.2 (Weighted majority games)

A weighted majority game is a simple game $\langle N,v\rangle$ in which

$$v(S)=\left\{\begin{array}{ll}1&\mathrm{if}\:w(S)\geq q\\0&\mathrm{otherwise},\end{array}\right.$$

for some $q\in\mathbb{R}$ and $w\in\mathbb{R}_+^N$ ,where $w(S)=\sum_{i\in S}w_{i}$ for any coalition $S$ An interpretation is that $Wi$ is the number of votes that player $\dot{\boldsymbol{r}}$ has and $q$ is the number of votes needed to win (the quota). A weighted majority game is homogeneous if $w(S)=q$ for any minimal winning coalition $S$ and is zerosum if for each coalition $S$ either $v(S)=1$ or $v(N\setminus S)=1$ but not both. Consider a zerosum homogeneous weighted majority game $\langle N,v\rangle$ in which $w_{i}=0$ for every player $\dot{\boldsymbol{z}}$ who does not belong to any minimal winning coalition. Show that the nucleolus of $\langle N,v\rangle$ consists of the imputation $x$ defined by $x_{i}=w_{i}/w(N)$ for all $i\in N$.

### 14.4 The Shapley Value

The last solution concept that we study in this chapter is the Shap ley value. Following our approach in the previous section we begin by characterizing this solution in terms of objections and counterobjections Then we turn to the standard (axiomatic) characterization

### 14.4.1 A Definition in Terms of Objections and Counterobjections

The solution concepts for coalitional games that we have studied so far are defined with reference to single games in isolation. By contrast, the Shapley value of a given game is defined with reference to other games. It is an example of a value — a function that assigns a unique feasible payoff profile to every coalitional game with transferable payoff, a payoff profile being feasible if the sum of its components is $v(N)$ .(The requirement that the payoff profile assigned by the value be feasible is sometimes called efficiency.)

Our first presentation of the Shapley value,like our presentations of the solutions studied in the previous section, is in terms of certain types of objections and counterobjections. To define these objections and counterobjections, let $\langle N,v\rangle$ be a coalitional game with transferable payoff and for each coalition $S$ define the subgame $\langle S,v^{S}\rangle$ of $\langle N,v\rangle$ tobe the coalitional game with transferable payoff in which $v^{\mathrm{S}}(T)=v(T)$ for any $T\subseteq S$ .Let $\psi$ be a value. An objection of player $i$ against player $j$ to the division $L$ of $v(N)$ may take one of the following two forms.

“Give me more since otherwise I will leave the game, causing you to obtain only $\psi_{j}(N\setminus\{i\},v^{N\backslash\{i\}})$ rather than the larger payoff iEj so that you will lose the positive amount $x_{j}-\psi_{j}(N\setminus\{i\},v^{N\setminus\{i\}})$ ” ·“Give me more since otherwise I will persuade the other players to exclude you from the game, causing me to obtain $\psi_{i}(N\backslash\{j\},v^{N\setminus\{j\}})$ rather than the smaller payoff $iL_{i}$ , so that I will gain the positive amount $\psi_{i}(N\setminus\{j\},v^{N\backslash\{j\}})-x_{i}$

A counterobjection by player $j$ to an objection of the first type is an assertion ·“It is true that if you leave then I will lose,but if $I$ leave then you will lose at least as much: $x_i-\psi_i(N\setminus\{j\},v^{N\backslash\{j\}})\geq x_j-$ $\psi_{j}(N\setminus\{i\},v^{N\backslash\{i\}})$ ” A counterobjection by player $j$ to an objection of the second type is an assertion ▪“It is true that if you exclude me then you will gain, but if $I$ exclude you then I will gain at least as much: $\psi_{j}(N\setminus\{i\},v^{N\backslash\{i\}})-x_{j}\geq$ $\psi_{i}(N\setminus\{j\},v^{\mathrm{N}\backslash\{j\}})-x_{i}$ ” The Shapley value is required to satisfy the property that for every objection of any player $i$ against any other player $j$ there is a counterobjection of player $j$

These objections and counterobjections differ from those used to de fine the bargaining set,kernel, and nucleolus in that they refer to the outcomes of smaller games. It is assumed that these outcomes are derived from the same logic as the payoff of the game itself: that is, the outcomes of the smaller games, like the outcome of the game itself, are given by the value. In this respect the definition of a value shares features with that of stable sets. The requirement that a value assign to every game a payoff profile with the property that every objection is balanced by a counterobjection is equivalent to the following condition.

DEFINITION 291.1

A value $\psi$ satisfies the balanced contributions property if for every coalitional game with transferable payoff $\langle N,v\rangle$ we have

$$\psi_{i}(N,v)-\psi_{i}(N\setminus\{j\},v^{N\setminus\{j\}})=\psi_{j}(N,v)-\psi_{j}(N\setminus\{i\},v^{N\setminus\{i\}})$$

whenever $i\in N$ and $j\in N$

We now show that the unique value that satisfies this property is the Shapley value, defined as follows. First define the marginal contribution of player 2 to any coalition S with $i\notin S$ in the game $\langle N,v\rangle$ to be

$$\Delta_i(S)=v(S\cup\{i\})-v(S).$$

DEFINITION 291.2 The Shapley value $\psi$ is defined by the condition

$$\varphi_{i}(N,v)=\frac{1}{|N|!}\sum_{R\in\mathcal{R}}\Delta_{i}(S_{i}(R))\:\mathrm{for~each}\:i\in N,$$

where $7R$ is the set of all $|N|$ orderings of $N$ and $S_{i}(R)$ is the set of players preceding 2 in the ordering $R$

We can interpret the Shapley value as follows. Suppose that all the players are arranged in some order, all orders being equally likely. Then $\varphi_{i}(N,v)$ is the expected marginal contribution over all orders of player $i$ to the set of players who precede him. Note that the sum of the marginal contributions of all players in any ordering is $v(N)$ , so that the Shapley value is indeed a value.

PROPOsITION 291.3 The unique value that satisfies the balanced contributions property is the Shapley value

Proof. First we show that there is at most one value that satisfies the property. Let $\psi$ and $\psi^{\prime}$ be any two values that satisfy the condition. We prove by induction on the number of players that $\psi$ and $\psi^{\prime}$ are identical. Suppose that they are identical for all games with less than $7t$ players and let $\langle N,v\rangle$ be a game with $7t$ players. Since $\psi_{i}(N\setminus\{j\},v^{N\setminus\{j\}})=$ $\psi_{i}^{\prime}(N\setminus\{j\},v^{N\setminus\{j\}})$ for any $i$, $j\in N$ ，we deduce from the balanced contributions property that $\psi_{i}(N,v)-\psi_{i}^{\prime}(N,v)=\psi_{j}(N,v)-\psi_{j}^{\prime}(N,v)$

for all $i$, $j\in N$ .Now fixing $i$ and summing over $j\in N$ ，using the fact that $\sum _{j\in N}\psi _{j}( N, v)$ = $\sum _{j\in N}\psi _{j}^{\prime }( N, v)$ = $v( N)$ ,we conclude that $\psi_{i}(N,v)=\psi_{i}^{\prime}(N,v)$ for all $i\in N$ We now verify that the Shapley value 4 satisfies the balanced contribu tions property. Fix a game $\langle N,v\rangle$ .We show that $\varphi_{i}(N,v)-\varphi_{j}(N,v)=$ $\varphi_{i}(N\setminus\{j\},v^{N\backslash\{j\}})-\varphi_{j}(N\setminus\{i\},v^{N\setminus\{i\}})$ for all 2, $j\in N$ . The left-hand side of this equation is

$$\sum_{S\subseteq N\setminus\{i,j\}}\alpha_{S}[\Delta_{i}(S)-\Delta_{j}(S)]+\beta_{S}[\Delta_{i}(S\cup\{j\})-\Delta_{j}(S\cup\{i\})],$$

where $\alpha _{S}$ = $| S| ! ( | N| - | S| - 1) ! / | N|$ and $\beta _{S}$ = $( | S|$ + $1) ! ( | N|$ - $| S|$ - $2)!/|N|$ ,while the right-hand side is

$$\sum_{S\subseteq N\setminus\{i,j\}}\gamma_{S}[\Delta_{i}(S)-\Delta_{j}(S)],$$

where $\gamma_{S}=|S|!(|N|-|S|-2)!/(|N|-1)!$ .The result follows from the facts that $\Delta_{i}(S)-\Delta_{j}(S)=\Delta_{i}(S\cup\{j\})-\Delta_{j}(S\cup\{i\})$ and $\alpha_{S}+\beta_{S}=\gamma_{S}.\square$

Note that the balanced contributions property links a game only with its subgames. Thus in the derivation of the Shapley value of a game $\langle N,v\rangle$ we could restrict attention to the subgames of $\langle N,v\rangle$ ,rather than work with the set of all possible games.

### 14.4.2 An Axiomatic Characterization

We now turn to an axiomatic characterization of the Shapley value.The derivation, unlike that in the previous section, restricts attention to the set of games with a given set of players. Throughout we fix this set to be $N$ and denote a game simply by its worth function 2

To state the axioms we need the following definitions.Player $i$ is a dummy in $U$ if $\Delta _{i}( S)$ = $v( \{ i\} )$ for every coalition S that excludes $i$ Players $i$ and $j$ are interchangeable in 2 if $\Delta _{i}( S)$ = $\Delta _{j}( S)$ for every coalition $S$ that contains neither $\dot{x}$ nor $j$ (or, equivalently, $v((S\setminus\{i\})\cup$ $\{j\})=v(S)$ for every coalition S that includes $i$ but not $j$ ).The axioms are the following

SYM (Symmetry) If $i$ and $j$ are interchangeable in 2 then $\psi_{i}(v)=\psi_{j}(v)$

DUM (Dummy player)If $i$ is a dummy in $U$ then $\psi_{i}(v)=v(\{i\})$

ADD (Additivity)For any two games $U$ and $u$ we have $\psi _{i}( v+ w)$ = $\psi_{i}(v)+\psi_{i}(w)$ for all $i\in N$ ,where $v+w$ is the game defined by $(v+w)(S)=v(S)+w(S)$ for every coalition $S$

Note that the first two axioms impose conditions on single games. while the last axiom links the outcomes of different games. This last axiom is mathematically convenient but hard to motivate: the structure of $v+w$ may induce behavior that is unrelated to that induced by 2 or $U$ separately. Luce and Raiffa (1957, p. 248) write that the axiom “"strikes us as a flaw in the concept of value”; for a less negative view see Myerson (1991, p. 437-438)

PROPOsrrION 293.1 The Shapley value is the only value that satisfies SYM, DUM, and ADD

Proof.We first verify that the Shapley value satisfies the axioms

SYM: Assume that $i$ and $j$ are interchangeable. For every ordering $R\in{\mathcal{R}}$ let $R^{\prime}\in\mathcal{R}$ differ from $R$ only in that the positions of $i$ and $j$ are interchanged. If $i$ precedes $j$ in $R$ then we have $\Delta_{i}(S_{i}(R))=\Delta_{j}(S_{j}(R^{\prime}))$ If $j$ precedes 2. then $\Delta_{i}(S_{i}(R))-\Delta_{j}(S_{j}(R^{\prime}))=v(S\cup\{i\})-v(S\cup\{j\})$ where $S$ = $S_{i}( R) \setminus \{ j\}$ .Since $\dot{\boldsymbol{z}}$ and $j$ are interchangeable we have $v(S\cup\{i\})=v(S\cup\{j\})$ ，so that $\Delta _{i}( S_{i}( R) )$ = $\Delta _{j}( S_{j}( R^{\prime }) )$ in this case too. It follows that 4 satisfies SYM

DUM: It is immediate that 4 satisfies this condition

ADD: This follows from the fact that if $u=v+u$ then

$$u(S\cup\{i\})-u(S)=v(S\cup\{i\})-v(S)+w(S\cup\{i\})-w(S).$$

We now show that the Shapley value is the only value that satisfies the axioms. Let $\psi$ be a value that satisfies the axioms. For any coalition S define the game $vS$ by

$$v_S(T)=\left\{\begin{array}{ll}1&\mathrm{if}\:T\supseteq S\\0&\mathrm{otherwise}.\end{array}\right.$$

Regard a game 2 as a collection of $2^{|N|-1}$ numbers $(v(S))_{S\in\mathcal{C}}$ .We begin by showing that for any game 2 there is a unique collection $(\alpha_{T})_{T\in\mathcal{C}}$ of real numbers such that $v=\sum_{T\in\mathcal{C}}\alpha_{T}v_{T}$ . That is, we show that $(v_T)_{T\in\mathcal{C}}$ is an algebraic basis for the space of games. Since the collection $(v_T)_{T\in\mathcal{C}}$ of games contains $2^{|N|}-1$ members it suffices to show that these games are linearly independent. Suppose that $\sum_{S\in\mathcal{C}}\beta_{S}v_{S}=0$ ;we need to show that $\beta_{S}=0$ for all $S$ .Suppose to the contrary that there exists some coalition $T$ with $\beta_{T}\neq0$ .Then we can choose such a coalition $T$ for which $\beta_{S}=0$ for all $S\subset T$ , in which case $\sum_{S\in\mathcal{C}}\beta_{S}v_{S}(T)=\beta_{T}\neq0$ a contradiction Now, by SYM and DUM the value of any game $\alpha vT$ for $\alpha\geq0$ is

given uniquely by $\psi_{i}(\alpha v_{T})=\alpha/|T|$ if $i\in T$ and $\psi_{i}(\alpha v_{T})=0$ otherwise We complete the proof by noting that if $v=\sum_{T\in\mathcal{C}}\alpha_{T}v_{T}$ then wehave

$v$ = $\sum _{\{ T\in \mathcal{C} : \alpha _{T}> 0\} }$ $\alpha _{T}v_{T}- \sum _{\{ T\in \mathcal{C} : \alpha _{T}< 0\} }( - \alpha _{T}v_{T})$ so that by ADD the value of 2 is determined uniquely

ExAMPLE 294.1 (Weighted majority games) Consider the weighted majority game $U$ see Exercise 289.2)with weights $w$ = (1,1,1,2) and quota $q=3$ .In all orderings in which player 4 is first or last his marginal contribution is. 0 ；;in all other orderings his marginal contribution is 1. Thus $\varphi ( v)$ = $( \frac 16, \frac 16, \frac 16, \frac 12)$ .Note that we have $v=$ $v_{\{1,4\}}+v_{\{2,4\}}+v_{\{3,4\}}+v_{\{1,2,3\}}-v_{\{1,2,4\}}-v_{\{1,3,4\}}-v_{\{2,3,4\}}$ , from which we can alternatively deduce $\varphi_{4}(v)=3\cdot\frac{1}{2}+0-3\cdot\frac{1}{3}=\frac{1}{2}$

ExERCIsE 294.2 Show the following results, which establish that if any one of the three axioms SYM, DUM, and ADD is dropped then there is a value different from the Shapley value that satisfies the remaining two.

a. For any game 2 and any $i\in N$ let $\psi_{i}(v)$ be the average marginal contribution of player $i$ over all the $(|N|-1)$ !orderings of $N$ in which player 1 is first. Then $\psi$ satisfies DUM and ADD but not SYM. b. For any game 2 let $\psi _{i}( v)$ = $v( N) / | N|$ .Then $\psi$ satisfies SYM and ADD but not DUM c. For any game U let $D(v)$ be the set of dummies in $U$ and let
$$\psi_i(v)=\begin{cases}\:\frac{1}{|N\setminus D(v)|}\:\Big(v(N)-\sum_{j\in D(v)}v(\{j\})\Big)&\text{if}i\in N\setminus D(v)\\\:v(\{i\})&\text{if}i\in D(v).\end{cases}$$
Then $\psi$ satisfies SYM and DUM but not ADD

ExAMPLE 294.3 Consider the game $\langle\{1,2,3\},v\rangle$ in which $v(1,2,3)=$ $v( 1, 2)$ = $v( 1, 3)$ = 1 and $v(S)=0$ otherwise. (This game can be interpreted as a model of a market in which there is a seller (player 1) who holds one unit of a good that she does not value and two potential buyers (players 2 and 3) who each value the good as worth one unit of payoff.) There are six possible orderings of the players. In the four in which player 1 is second or third her marginal contribution is 1 and the marginal contributions of the other two players are 0 ；; in the ordering (1,2,3) player 2's marginal contribution is 1, and in (1,3,2) player 3's marginal contribution is 1. Thus the Shapley value of the game is $\left(\frac{2}{3},\frac{1}{6},\frac{1}{6}\right)$ .By contrast, the core of the game consists of the single payoff profile (1,0,0)

ExAMPLE 294.4 (A market) Consider the market for an indivisible good in Example 260.1, in which there are $b$ buyers and $E$ sellers, with $\ell<b$

Consider replications of the market in which there are $kb$ buyers and $kE$ sellers for some positive integer $k$ .If $k$ is very large then in most random orderings of the players the fraction of buyers in the set of players who precede any given player 2 is close to $b/\ell>1$ .In any such ordering the marginal contribution of player $\dot{\tau}$ is 1 if she is a seller, so that the Shapley value payoff of a seller is close to 1 (and that of a buyer is close to 0 ). Precisely, it can be shown that the limit as $k\to\infty$ of the Shapley value payoff of a seller is 1. This is the simplest example of a more general result due to Aumann (1975) that the Shapley value converges to the profile of competitive payoffs as the size of the market increases.

ExERCIsE 295.1Find the core and the Shapley value of the game $\langle\{1,2,3,4\},v\rangle$ in which $v( \{ 1, 2, 3, 4\} )$ = 3, $v(S)=0$ if S includes at most one of the players in $\{1,2,3\}$ , and $v(S)=2$ otherwise. Explain the source of the difference between the two solutions

ExERCIsE 295.2 (A production economy）Find the Shapley value of the game in Exercise 259.3 and contrast it with the core and the nucleolus (see Exercise 289.1).

ExAMPLE 295.3 (A majority game）Consider a parliament in which there is one party with $m-1$ seats and $711L$ parties each with one seat, and a majority is decisive (a generalization of My aunt and $I$ ). This situation can be modeled as a weighted majority game (see Exercise 289.2） in which $N=\{1,\ldots,m+1\}$ $w_{1}=m-1$ $w_{i}=1$ for $i\neq1$ , and $y=m$ .The marginal contribution of the large party is 1 in all but the $2m!$ orderings in which it is first or last. Hence the Shapley value of the game assigns to the large party the payoff $[(m+1)!-2m!]/(m+1)!=(m-1)/(m+1)$

ExERCIsE 295.4 Consider a parliament in which there are 77 parties two of them have $\frac{1}{3}$ of the seats each and the other $n-2$ share the remaining seats equally.Model this situation as a weighted majority game (see Exercise 289.2).

α. Show that the limit as $7l\to0$ of the Shapley value payoff of each of the large parties is $\frac{1}{4}$ b. Is it desirable according to the Shapley value for the $n-2$ small parties to form a single united party?

ExERCIsE 295.5 Show that in a convex game (see Exercise 260.4) the Shapley value is a member of the core The result in the following exercise suggests an interpretation of the Shapley value that complements those discussed above.

ExERCIsE 296.1 Consider the following variant of the bargaining game of alternating offers studied in Chapter 7. Let $\langle N,v\rangle$ be a coalitional game with transferable payoff in which $v(S)\geq0$ and $v(S\cup\{i\})\geq v(S)+$ $v(\{i\})$ for every coalition S and player $i\in N\setminus S$ In each period there is a set $S\subseteq N$ of active players, initially $N$ , one of whom, say player $i$ is chosen randomly to propose an $S$ -feasible payoff vector $x^{S,i}$ .Then the remaining active players, in some fixed order, each either accepts or rejects $x^{S,i}$ .If every active player accepts $x^{S,i}$ then the game ends and each player $j\in S$ receives the payoff $x_{j}^{S,i}$ . If at least one active player rejects $x^{S,i}$ then we move to the next period, in which with probability $\rho\in(0,1)$ the set of active players remains S and with probability $1-\rho$ it becomes $S\setminus\{i\}$ (i.e. player $i$ is ejected from the game) and player $i$ receives the payoff $v(\{i\})$ .Players do not discount the future. Suppose that there is a collection $(x^{S,i})_{S\in\mathcal{C},i\in S}$ of $S$ -feasible payoff

vectors such that $x_{j}^{S,i}=\rho\overline{x}_{j}^{S}+(1-\rho)\overline{x}_{j}^{S\setminus\{i\}}$ for all $S$ all $i\in S$ , and all $j\in S\setminus\{i\}$ ,where $\overline{x}^{S}=\sum_{i\in S}x^{S,i}/|S|$ for all S Show that the game has a subgame perfect equilibrium in which each player $i\in S$ proposes $x^{S,i}$ whenever the set of active players is $S$ Show further that there is such a collection for which $\overline {x}^{S}$ = $\varphi ( S, v)$ for each $S\in C$ , thus showing that the game has a subgame perfect equilibrium in which the expected payoff of each player 2 is his Shapley value payoff $\varphi_{i}(N,v)$ .Note that if $\rho$ is close to 1 in this case then every proposal $x^{S,i}$ is close to the Shapley value of the game $\langle S,v\rangle$ .(Hart and Mas-Colell (1996) show that every subgame perfect equilibrium in which each player's strategy is independent of history has this property; Krishna and Serrano (1995) study non-stationary equilibria.

### 14.4.3 Cost-Sharing

Let $N$ be a set of players and for each coalition $S$ let $C(S)$ be the cost of providing some service to the members of $S$ .How should $C(N)$ be shared among the players? One possible answer is given by the Shapley value $\varphi(C)$ of the game $\langle N,C\rangle$ ,where $\varphi_{i}(C)$ is the payment requested from player $i$ .This method of cost-sharing is supported by the axioms presented above, which in the current context can be given the following interpretations. The feasibility requirement $\sum_{i\in\mathbb{N}}\varphi_{i}(C)=C(N)$ says that the total payments requested from the players should equal $C(N)$ the total cost of providing the service. The axioms DUM and SYM have interpretations as principles of “fairnes” when applied to the game. DUM says that a player for whom the marginal cost of providing the

service is the same, no matter which group is currently receiving the service, should pay that cost. SYM says that two players for whom the marginal cost is the same, no matter which group is currently receiving the service, should pay the same. ADD is somewhat more attractive here than it is in the context of strategic interaction. It says that the payment of any player for two different services should be the sum of the payments for the two services separately

### Notes

Stable sets were first studied by von Neumann and Morgenstern (1944) The idea of the bargaining set is due to Aumann and Maschler (1964); the formulation that we give is that of Davis and Maschler (1963). The kernel and nucleolus are due respectively to Davis and Maschler (1965) and Schmeidler (1969). Proofs of the nonemptiness of the bargaining set (using direct arguments) were first given by Davis, Maschler, and Peleg (see Davis and Maschler (1963, 1967) and Peleg (1963b, 1967)). Our definition of the nucleolus in terms of objections and counterob jections appears to be new. The results in Section 14.3.3 (other than Lemma 286.2) are due to Schmeidler (1969). The Shapley value is due to Shapley (1953),who proved Proposition 293.1. The balanced contributions property (Definition 291.1) is due to Myerson (1977,1980); see also Hart and Mas-Colell (1989).

The application of the Shapley value to the problem of cost-sharing was suggested by Shubik (1962); the theory has been developed by many authors, including Roth and Verrecchia (1979) and Billera, Heath, and Raanan (1978).

The game My aunt and $I$ in Examples 282.3 and 285.3 is studied by Davis and Maschler (1965, Section 6). The result in Exercise 283.1 is due to Maschler (1976). Exercise 289.1 is taken from Moulin (1988, pp. 126- 127; see also Exercise 5.3). Weighted majority games were first studied by von Neumann and Morgenstern (1944); the result in Exercise 289.2 is due to Peleg (1968). The game in Exercise 295.1 is due to Zamir, quoted in Aumann (1986, p. 986). Exercise 295.2 is taken from Moulin (1988. p. 111). The result in Exercise 295.4 is due to Milnor and Shapley (1978), that in Exercise 295.5 to Shapley (1971/72), and that in Exercise 296.1 to Hart and Mas-Colell (1996).

Much of the material in this chapter draws on Aumann’s (1989) lecture notes, though some of our interpretations of the solution concepts are different from his.

The definitions of stable sets and the bargaining set can be extended straightforwardly to coalitional games without transferable payoff (see, for example,Aumann and Peleg (1960) and Peleg (1963a)). For extensions of the Shapley value to such games see Harsanyi (1963), Shap ley (1969), Aumann (1985a) Hart (1985), and Maschler and Owen (1989, 1992).

Harsanyi (1974) studies an extensive game for which a class of sub-game perfect equilibria correspond to stable sets. Harsanyi (1981), Gul (1989), and Hart and Mas-Colell (1996) study extensive games that have equilibria corresponding to the Shapley value. The solution concepts that we study in this chapter can be interpreted as formalizing notions of “fairness"; for an analysis along these lines see Moulin (1988) Lucas (1992) and Maschler (1992) are surveys that cover the models in Sections 14.2 and 14.3.

---

## 15 The Nash Solution

- 15.1 Bargaining Problems 299
- 15.3 The Nash Solution: Definition and Characterization 301
- 15.3 An Axiomatic Definition 305
- 15.4 The Nash Solution and the Bargaining Game of Alternating Offers 310
- 15.5 An Exact Implementation of the Nash Solution 311
- Notes 312

In this chapter we study two-person bargaining problems from the perspective of coalitional game theory. We give a definition of the Nash solution in terms of objections and counterobjections and characterize the solution axiomatically.In addition we explore the connection between the Nash solution and the subgame perfect equilibrium outcome of a bargaining game of alternating offers

### 15.1 Bargaining Problems

In Chapter 7 we discuss two-person bargaining using the tools of the theory of extensive games. Here we do so using the approach of coalitional game theory.We define a bargaining problem to be a tuple $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ in which $X$ is a set of possible consequences that the two players can jointly achieve, $D\in X$ is the event that occurs if the players fail to agree, and $\succsim1$ and $\succsim_{2}$ are the players² preference relations over ${\mathcal{L}}(X)$ ,the set of lotteries over $X$ .We refer to $X$ as the set of possible agreements and to $D$ as the disagreement outcome. Note that such a tuple can be identifed with a coalitional game without transferable payoff $\langle\{1,2\},\mathcal{L}(X),V,(\succsim_i)\rangle$ in which $V(\{1,2\})=X$ and $V(\{i\})=\{D\}$ for $i=1$, 2 (see Definition 268.2) The members of $X$ should be thought of as deterministic.Note that

we require the players’ preference relations to be defined over the set of lotteries over $X$ ,rather than simply over $X$ itself. That is, each preference relation includes information not only about the player's preferences over the set of possible joint actions but also about his attitude

---

towards risk.We denote by $p\cdot x\oplus(1-p)\cdot y$ the lottery that gives $JL$ with probability $P$ and $y$ with probability $1-p$ and by $p\cdot x$ the lottery $p\cdot x\oplus(1-p)\cdot D$ Our basic definition of a bargaining problem contains some restric

tions, as follows

> DEFINITION 300.1 A bargaining problem is a tuple $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ where

· $X$ (the set of agreements) is a compact set (for example, in a Euclidian space, $D$ (the disagreement outcome)is a member of $X$ · \succsim1 and $\succsim_{2}$ are continuous preference relations on the set ${\mathcal{L}}(X)$ of lotteries over $X$ that satisfy the assumptions of von Neumann and Morgenstern · $x\succsim_{i}D$ for all $x\in X$ for $\dot{\boldsymbol{i}}=1$ ,2, and there exists $x\in X$ such that $x\succ_{1}D$ and $x>_{2}D$ ·(convexity) for any $x\in X$, $y\in X$ ,and $p\in[0,1]$ there exists $z\in X$ such that $z\sim_{i}p\cdot x\oplus(1-p)\cdot y$ for $i=1$ 2 non-redundancy) if $x\in X$ then there is no $x^{\prime}\in X$ with $x^{\prime}\neq x$ such that $x\sim_ix^{\prime}$ for $i=1$ 2 ·(unique best agreements) for each player $i$ there is a unique agree ment $B_{i}\in X$ with $B_{i}\succsim_{i}{\mathcal T}$ for all $x\in X$ · for each player $\dot{i}$ we have $B_{i}\sim_{j}D$ for $j\neq i$

The first three of these assumptions guarantee that each player's preference relation over ${\mathcal{L}}(X)$ can be represented by the expectation of some continuous function over $X$ (the players von NeumannMorgenstern utility function). The fourth assumption says that disagreement is the worst possible outcome and that the problem is non-degenerate in the sense that there exists an agreement that is more attractive to both players than disagreement. The assumption of convexity requires that the set of agreements be rich enough that every lottery is equivalent for both players to some (deterministic) agreement. The last three assump tions are made for convenience. The assumption of non-redundancy says that we identify any two agreements between which both players are indifferent. The assumption of unique best agreements implies that the best agreement for each player is strongly Pareto efficient (ie. there is no agreement that is better for one player and at least as good for the other). The last assumption says that each player is indifferent between

---

disagreement and the outcome in which the other player obtains his favorite agreement.

Given our assumptions on the players’ preferences we can associate with any bargaining problem $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ and any von NeumannMorgenstern utility functions $u_{1}$ and $u2$ that represent \succsim1 and $\succsim_{2}$ a pair $\langle U,d\rangle$ in which $U$ = $\{ ( u_{1}( x) , u_{2}( x) ) \nobreak {: } x\in X\}$ and $d$ = $( u_{1}( D) , u_{2}( D) )$ we can choose $u1$ and $u_{2}$ so that $d$ = (0,0) .Our assumptions imply that $U$ is compact and convex and contains a point $y$ for which $y_{i}>d_{i}$ for $i=1$, 2 .In the standard treatment of the Nash solution such a pair $\langle U,d\rangle$ , rather than a description like $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ of the physical agreements and the players’ preferences, is taken as the primitive; we find the language of agreements and preferences more natural.Note that bargaining problems with different agreement sets and preference relations can lead to the same pair $\langle U,d\rangle$ : a bargaining problem contains more information than such a pair. Our aim now is to construct reasonable systematic descriptions of the

way that bargaining problems may be resolved. The notion of a bargaining solution is a formal expression of such a systematic description

DEFINIrioN 301.1 A bargaining solution is a function that assigns to every bargaining problem $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ a unique member of $X$

A bargaining solution describes the way in which the agreement (or disagreement) depends upon the parameters of the bargaining problem The bargaining theory that we study focuses on the effect of the players risk attitudes on the bargaining outcome. Alternative theories focus on other relevant factors (for example the players’ time preferences or their ability to bargain), but such theories require that we change the primitives of the model.

### 15.2 The Nash Solution: Definition and Characterization

#### 15.2.1 Definition

We now define the solution concept that we study in this chapter.

DEFINIrION 301.2 The Nash solution is a bargaining solution that assigns to the bargaining problem $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ an agreement ${\mathcal{T}}^*\in X$ for which

if $p\cdot x\succ_{i}x^{*}$ for some $p\in[0,1]$ and $x\in X$ then $p\cdot x^{*}\gtrsim_{j}x$ for $j\neq i$

---

This definition is equivalent to one whose structure is similar to those of the bargaining set, kernel, and nucleolus given in the previous chapter. To see this,define an objection of player $\dot{\boldsymbol{z}}$ to the agreement $x^*\in X$ to be a pair $(x,p)$ with $x\in X$ and $p\in[0,1]$ for which $p\cdot x\succ_{i}x^{*}$ .The interpretation is that 2 is an alternative agreement that player $\dot{x}$ proposes and $1-p$ is the probability that the negotiations will break down if player $\dot{i}$ presses his objection. The agreement $iL$ and the probability $P$ are chosen by player $i$ ; the probability $P$ may be determined indirectly by the actions (like threats and intimidations) that player $i$ takes when he presses his demand that the agreement be $iL$ .Thus player $\dot{x}$ makes an argument of the form “I demand the outcome $iL$ rather than $x^{*}$ ：I back up this demand by threatening to take steps that will cause us to fail to agree with probability $1-p$ ,a threat that is credible since if I carry it out and the outcome is 2 then I will be better off than I am now". Player $j$ can counterobject to $(x,p)$ if $p\cdot x^{* }\succsim _{j}$ $x$ .The interpretation is that under the risky conditions that player $\dot{\tau}$ creates by his objection it is desirable for player $j$ to insist on the original agreement $x^*$ .Thus player $J$ 's argument is “If you take steps that will cause us to disagree with probability $1-p$ then it is still desirable for me to insist on $x^*$ rather than agreeing to $JL$ ".Given these definitions of objection and counterobjection the Nash solution is the set of all agreements ${\boldsymbol{L}}^{*}$ with the property that player $j$ can counterobject to every objection of player $\dot{\boldsymbol{z}}$ to $x^{*}$

#### 15.2.2 Characterization

We now show that the Nash solution is well-defined and has a simple characterization: the Nash solution of the bargaining problem $\langle X,D$ $\succsim_{1},\succsim_{2}\rangle$ is the agreement that maximizes the product $u_{1}(x)u_{2}(x)$ , where $u_{i}$ is a von Neumann-Morgenstern utility function that represents \succsim$i$ for $i=1$ 2.

PROPOSITION 302.1

a.The agreement $x^*\in X$ isa Nash solution of the bargaining problem $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ ifand only if

$$u_1(x^*)u_2(x^*)\geq u_1(x)u_2(x)\:for\:all\:x\in X,$$

where $u_{i}$ is a von Neumann-Morgenstern utility function that repre sents $\sum_{i=1}^{i}$ and satisfies $u_{i}(D)=0$ for $i=1$ ,2

b.The Nash solution is well-defined

---

Proof. We first prove (a). Suppose that $u_{1}(x^{*})u_{2}(x^{*})\geq u_{1}(x)u_{2}(x)$ for all $x\in X$ . Then $u_{i}(x^{*})>0$ for $\boldsymbol{v}=1$, 2 (since $X$ contains an agreement $y$ for which $u_{i}(y)>0$ for $i=1$ ,2).Now, if $pu_{i}(x)>u_{i}(x^{*})$ for some $p\in[0,1]$ then $pu_{i}( x) u_{j}( x^{* })$ > $u_{i}( x^{* }) u_{j}( x^{* })$ $\geq$ $u_{i}( x) u_{j}( x)$ and hence $pu_{j}(x^{*})>u_{j}(x)$ (since $u_{i}(x)>0$ ).or $p\cdot x^{*}\succ_{j}x$ Now suppose that $x^{*}$ satisfies (301.3): if $p\cdot x\succ_{i}x^{*}$ for some $p\in[0,1]$

and $x\in X$ then $p\cdot x^{* }\gtrsim _{j}$ $x$ .Let $x\in X$ be such that $u_{i}(x)>0$ for $i=1$, 2 and $u_{i}(x)>u_{i}(x^{*})$ for some $i$ .(For any other value of $iL$ we obviously have $u_{1}(x^{*})u_{2}(x^{*})\geq u_{1}(x)u_{2}(x)$ .) Then if $p>u_{i}(x^{*})/u_{i}(x)$ for some $p\in[0,1]$ we have $pu_{j}(x^{*})\geq u_{j}(x)$ , so that, since $u_{j}(x)>0$ ，we have $p\geq u_{j}(x)/u_{j}(x^{*})$ .Hence $u_{i}( x^{* }) / u_{i}( x)$ $\geq$ $u_{j}( x) / u_{j}( x^{* })$ and thus $u_{1}(x^{*})u_{2}(x^{*})\geq u_{1}(x)u_{2}(x)$ To prove (b), let $U$ = $\{ ( u_{1}( x) , u_{2}( x) ) \nobreak {: } x$ $\in$ $X\}$ .By (a), the agree

ment $x^*$ is a Nash solution of $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ if and only if $(v_1,v_2)=$ $(u_{1}(x^{*}),u_{2}(x^{*}))$ maximizes $v_{1}v_{2}$ over $U$ .Since $U$ is compact this problem has a solution; since the function $v_{1}v_{2}$ is strictly quasi-concave on the interior of $\mathbb{R}_+^2$ and $U$ is convex the solution is unique. Finally, by the assumption of non-redundancy there is a unique agreement $x^*\in X$ that yields the pair of maximizing utilities.

The simplicity of this characterization is attractive and accounts for the widespread application of the Nash solution. The characterization also allows us to illustrate the Nash solution geometrically, as in Figure 304.1. Although the maximization of a product of utilities is a simple mathematical operation it lacks a straightforward interpretation; we view it simply as a technical device. Originally Nash defined the solution in terms of this characterization; we find Definition 301.2 preferable since it has a natural interpretation.

#### 15.2.3 Comparative Statics of Risk Aversion

A main goal of Nash's theory is to provide a relationship between the players² attitudes towards risk and the outcome of the bargaining. Thus a first test of the plausibility of the theory is whether this relationship accords with our intuition. We compare two bargaining problems that differ only in that one player's preference relation in one of the problems is more risk-averse than it is in the other; we verify that the outcome of the former problem is worse for the player than that of the latter. Define the preference relation $\succsim_{1}^{\prime}$ to be at least as risk-averse as \succsim1

if 1 and $\succsim_{1}^{\prime}$ agree on $X$ and whenever $x\sim_{1}L$ for some ${\mathcal{T}}\in X$ and

---

![](https://storage.simpletex.cn/view/fuLWAL0CBst3GWEu81ZXUybeLEbLOX0ly)

Figure 304.1 A geometric characterization of the Nash solution $x^{*}$ of the bargaining problem $\langle X, D, \succsim_1,\succsim_2 \rangle$ . For $i=1,2$ the function $u_{2}$ is a von Neumann-Morgenstern utility function that represents \succsim$i$ and satisfies $u_{i}(D)=0$

$L\in\mathcal{L}(X)$ we have $x\succsim_{1}^{\prime}$ L. (This definition is equivalent to the standard definition that is given in terms of utility representations.)

■PROPOSITION 304.1 Let $iL$ and ${\boldsymbol{x}}^{\prime}$ be the Nashsolutions of the bargaining problem.s $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ and $\langle X,D,\succsim_{1}^{\prime},\succsim_{2}\rangle$ respectively, where \succsim$_{1}^{\prime}$ is at least as risk-awerse as \succsim1 .Then $x\succsim_1x^{\prime}$

Proof.Assume to the contrary that $x^{\prime }\succ _{1}$ $x$ .By the convexity of the bargaining problems there exists an agreement $z\in X$ such that $z$ $\sim _{i}$ $\frac 12\cdot$ $x^{\prime }\oplus$ $\frac 12$ $\cdot$ $x$ for $i=1$, 2 .Let $z^*$ be a Pareto efficient agreement for which $z^*\succsim_i$ for $i=1$ ，2. By the characterization of the Nash solution (Proposition 302.1a), the agreements $L$ and $x^{\prime}$ are Pareto efficient, so that $x$ $\prec _{1}$ $z^{* }$ $< _{1}$ $x^{\prime }$ and $x^{\prime }$ $< _2$ $z^*$ $< _2$ $x$ .Now, since 2 is the Nash solution of $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ we have $u_{1}( x) u_{2}( x)$ > $u_{1}(x^{\prime})u_{2}(x^{\prime})$ ,where $u_{i}$ is a von Neumann-Morgenstern utility function with $u_{i}(D)=0$ that represents $\sum_{i}$ for $i=1$ ,2. By the quasi-concavity of the function $H( v_{1}, v_{2})$ = $v_{1}v_{2}$ we have $u_{1}( z) u_{2}( z)$ > $u_{1}( x^{\prime }) u_{2}( x^{\prime })$ and hence $u_{1}(z^{*})u_{2}(z^{*})>u_{1}(x^{\prime})u_{2}(x^{\prime})$ .Since $x^{\prime}\succ_{1}z^{*}$ it follows that $1>u_{1}(z^{*})/u_{1}(x^{\prime})>u_{2}(x^{\prime})/u_{2}(z^{*})$ , so that there exists $p\in[0,1]$ such that $u_{1}( z^{* }) / u_{1}( x^{\prime })$ > $p$ > $u_{2}( x^{\prime }) / u_{2}( z^{* })$ and hence $p\cdot z^{*}\succ_{2}x^{\prime}$ and $z^*\succ_1p\cdot x^{\prime}$ Since the preference relation $\succsim_{1}^{\prime}$ is at least as risk-averse as \succsim1 we also have $z^{*}\succ_{1}^{\prime}p\cdot x^{\prime}$ , So that $(z^*,p)$ is an objection of player 2 to $x^{\prime}$ for which there is no counterobjection,contradicting the fact that $x^{\prime}$ is the Nash solution of $\langle X,D,\succsim_{1}^{\prime},\succsim_{2}\rangle$

---

### 15.3 An Axiomatic Definition

#### 15.3.1 Axioms

A beauty of the Nash solution is that it is uniquely characterized by three simple axioms (properties). In the following statements of these axioms $F$ denotes an arbitrary bargaining solution

PAR (Pareto efficiency）There is no agreement $x\in X$ such that $x\succsim_i$ $F(X,D,\succsim_{1},\succsim_{2})$ for $i=1$ ,2 with strict preference for at least one $i$

The standard justification of PAR is that an inefficient outcome is not likely since it leaves room for renegotiation that makes both players better off. The fact that the Nash solution satisfies PAR follows immediately from Proposition 302.1a.

To state the next axiom we need first to define a symmetric bargaining problem. Informally, a bargaining problem is symmetric if there is a relabeling of the set of agreements that interchanges the players’ pref erence relations: player 1’s preference relation in the relabeled problem coincides with player 2's preference relation in the original problem, and vice versa. To state this definition differently, consider the language that consists of the names of the preference relations and the name of the disagreement point, but not the names of the agreements. A problem is symmetric if any definition of an agreement by means of a formula in this language defines the same agreement if we interchange the names of the players.

> DEFINITION 305.1 A bargaining problem $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ is symmetric if there is a function $\phi{:}X\to X$ with $\phi(D)=D$ and $\phi(x)=y$ if and only if $\phi(y)=x$ , such that $L_{1}\succsim_{i}L_{2}$ if and only if $\phi ( L_{1}) \gtrsim _{j}$ $\phi ( L_{2})$ for $i\neq j$ and for any lotteries $L_{1}$ and $L_{2}$ in ${\mathcal{L}}(X)$ , where $\phi(L)$ is the lottery in which each prize 2 in the support of $L$ is replaced by the prize $\phi(x)$ We refer to the function $\phi{:}X\to X$ in this definition as the symme try function.An example of a symmetric bargaining problem is that in which two risk-neutral players split a pie, obtaining nothing if they disagree (consider the symmetry function given by $\phi(x_{1},x_{2})=(x_{2},x_{1})$ ) SYM (Symmetry) If $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ is symmetric with symmetry func tion $\phi$ then $\phi(F(X,D,\succsim_{1},\succsim_{2}))=F(X,D,\succsim_{1},\succsim_{2})$ The justification of this axiom is that we seek a solution in which all

asymmetries between the players are included in the description of the bargaining problem. Thus if players 1 and 2 are indistinguishable in a

---

certain problem then the agreement assigned to that problem should not discriminate between them

■LEMMA 306.1 The Nash solution satisfies SYM.

Proof. Let ${\mathcal{T}}^{*}$ be the Nash solution of the symmetric bargaining prob lem $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ with symmetry function $\phi$ .Suppose that $\phi(x^*)$ is not the Nash solution of the bargaining problem $\langle X,D,\succsim_{2},\succsim_{1}\rangle$ .Then some player $i$ has an objection $(x,p)$ to $\phi(x^*)$ for which there is no counterobjection by player $J$ ： $p\cdot x\succ_{i}\phi(x^{*})$ and $p\cdot\phi(x^{*})\prec_{j}x$ .But then $\phi ( p\cdot x) = p\cdot \phi ( x) \succ _{j}$ $\phi ( \phi ( x^{* }) ) = x^{* }$ and $\phi(p\cdot\phi(x^{*}))=p\cdot x^{*}\prec_{i}\phi(x)$ so that $(\phi(x),p)$ is an objection by player $j$ to $x^*$ for which there is no counterobjection by player $i$ , contradicting the fact that $x^*$ is the Nash solution.

The final axiom is the most problematic.

IIA (Independence of irrelevant alternatives) Let $x^{* }= F( X, D, \succsim _{1}, \succsim _{2})$ and let $\sum_{i}^{\prime}$ be a preference relation that agrees with $\sum_{i}$ on $X$ and satisfies ·if $x\succsim_ix^*$ and $p\cdot x\sim_{i}x^{*}$ for some $x\in X$ and $p\in[0,1]$ then $p\cdot x\lesssim_{i}^{\prime}x^{*}$ ·if $x$ $\lesssim _{i}$ $x^{* }$ and $x\sim_ip\cdot x^*$ for some $x\in X$ and $p\in[0,1]$ then $x\sim_{i}^{\prime}p\cdot x^{*}$ Then $F(X,D,\succsim_{i},\succsim_{j})=F(X,D,\succsim_{i},\succsim_{j}).$

A player whose preference relation is 六$i^{\prime}$ is more apprehensive than one whose preference relation is \succsim$i$ about the risk of demanding alternatives that are better than $x^{*}$ but has the same attitudes to alternatives that are worse than $x^*$ .The axiom requires that the outcome when player $i$ has the preference relation 六 is the same as that when player $i$ has the preference relation \succsim$i$ .The idea is that if ${\boldsymbol{L}}^{*}$ survives player 2 's objections originally then it should survive them also in a problem in which he is less eager to make them (i.e. fewer pairs $(x,p)$ are objections of player 2 ); it should continue also to survive player $j$ 's objections since player $i$ 's ability to counterobject has not been changed Note that despite its name, the axiom involves a comparison of two

problems in which the sets of alternatives are the same; it is the players preferences that are different.(The name derives from the fact that the axiom is analogous to an axiom presented by Nash that does involve a comparison of two problems with different sets of agreements.)Note also that the axiom differs from PAR and SYM in that it involves a com-

---

parison of bargaining problems, while PAR and SYM impose conditions on the solutions of single bargaining problems.

LEMMA 307.1 The Nash solution satisfies IIA.

Proof. Let ${\mathcal{T}}^*$ be the Nash solution of the bargaining problem $\langle X,D$ $\lesssim i,\succsim j\rangle$ and let 六$i^{\prime}$ be a preference relation that satisfies the hypotheses of IIA. Consider the bargaining problem $\langle X,D,\succsim_{i}^{\prime},\succsim_{j}\rangle$ .We show that for every objection of either $i$ or $j$ to $x^{*}$ in $\langle X,D,\succsim_{1}^{\prime},\succsim_{2}\rangle$ there is a counterobjection, so that $x^*$ is the Nash solution of $\langle X,D,\succsim_{1}^{\prime},\succsim_{2}\rangle$ First suppose that player $i$ has an objection to $x^{*}$ : $p\cdot x\succ_{i}^{\prime}x^{*}$ for some

$x\in X$ and $p\in[0,1]$ .Then $x\succ_{i}^{\prime}x^{*}$ and hence $x\succ_{i}x^{*}$ (since 六 i and $\sum_{i}^{\prime}$ agree on $X$ ). Thus from the first part of IIA we have $p\cdot x\succ_{i}x^{*}$ (if $p\cdotp x$ $\lesssim _{i}$ $x^{* }$ then there exists $q\geq p$ such that $q\cdotp x\sim_ix^*$ and thus $q\cdotp x$ $\lesssim _{i}^{\prime }$ $x^*$ so that $p\cdot x\lesssim_{i}^{\prime}x^{*}$ ).Since $x^*$ is the Nash solution of $\langle X,D,\succsim_{i},\succsim_{j}\rangle$ we thus have $p\cdot x^{*}\lesssim_{j}x$ Now suppose that player $j$ has an objection to $x^*$ : $p\cdot x\succsim_{j}x^{*}$ for some

$x\in X$ and $p\in[0,1]$ .Since ${\mathcal{T}}^{*}$ is Pareto effcient we have $x^{*}\succsim_{i}x$ and since $x^{*}$ is the Nash solution of $\langle X,D,\succsim_{i},\succsim_{j}\rangle$ we have $p\cdot x^{*}\succsim_{i}x$ . Thus D from the second part of IIA we have $p\cdot x^{*}\succsim_{i}^{\prime}x$

#### 15.3.2 Characterization

The following result completes the characterization of the Nash solution in terms of the axioms PAR, SYM, and IIA discussed above.

PROPOsITION 307.2 The Nash solution is the only bargaining solution that satisfies PAR, SYM, and IIA.

Proof. We have shown that the Nash solution satisfies the three axioms; we now show uniqueness

Step 1. Let $x^*$ be the Nash solution of the bargaining problem $\langle X,D$ $\lesssim_{1},\succsim_{2}\rangle$ .If $x\sim_{i}p\cdot x^{*}$ then $[ 1/ ( 2- p) ] \cdot x\text{ さ }_{j}$ $x^{* }$

Proof. For each player $\dot{\tau}$ choose the von NeumannMorgenstern utility function $u_{i}$ that represents $\succsim_{i}$ and satisfies $u_i( x^* )$ = 1 and $u_{i}(D)=$ $U$ .We first argue that for every agreement $y\in X$ we have $u_{1}(y)+$ $u_{2}(y)\leq2$ .To see this, suppose to the contrary that for some $y\in X$ we have $u_{1}( y) + u_{2}( y)$ = $2+ \epsilon$ with $\epsilon>0$ .By the convexity of the bargaining problem, for every $p\in[0,1]$ there is an agreement $z(p)\in X$ with $u_{i}(z(p))=pu_{i}(y)+(1-p)u_{i}(x^{*})=pu_{i}(y)+1$ - p for $i=1$, 2 , SO that u1( $z(p))u_{2}(z(p))=1+\epsilon p+p^{2}[u_{1}(y)u_{2}(y)-1-\epsilon]$ . Thus for $F$ close enough to (1) we have $u_{1}(z(p))u_{2}(z(p))>1=u_{1}(x^{*})u_{2}(x^{*})$ , contradicting

---

the fact that $x^*$ is the Nash solution of the problem. Now, if $x\sim_ip\cdot x^*$ we have $u_{i}(x)=p$ and hence $u_{j}(x)\leq2-p$ , so that $[ 1/ ( 2- p) ] \cdot x$ $\lesssim _{j}$ $x^{* }$

Step 2. Any bargaining solution that satisfies PAR, SYM, and IIA is the Nash solution

Proof. Let $x^*$ be the Nash solution of the bargaining problem $\langle X,D$ $\succsim_{1},\succsim_{2}\rangle$ and let $F$ be a bargaining solution that satisfies PAR, SYM, and IIA.Let $\succsim_{1}^{\prime}$ and $\succsim_{2}^{\prime}$ be preference relations that coincide with \succsim1 and \succsim 2 on $X$ and satisfy the following conditions. For any Pareto efficient agreement $x\in X$ we have

·if $x>_{1}x^{*}$ and $x\sim_2p\cdot x^*$ for some $p\in[0,1]$ then $x\sim_{2}^{\prime}p\cdot x^{*}$ and $x^{*}\sim_{1}^{\prime}\left[1/(2-p)\right]\cdot x$ ·if $x\prec_{1}x^{*}$ and $x\sim_{1}p\cdot x^{*}$ for some $p\in[0,1]$ then $x\sim_1^{\prime}p\cdot x^{*}$ and $x^{*}\sim_{2}^{\prime}\left[1/(2-p)\right]\cdot x$

(These conditions completely describe a pair of preference relations satisfying the assumptions of von Neumann and Morgenstern since for every $x\in X$ and each player $i$ there is some Pareto efficient agreement $x^{\prime}$ for which $x\sim_{i}x^{\prime}$ -)Let $u_{i}$ be the von Neumann-Morgenstern utility func tion that represents $\succsim_{i}^{\prime}$ and satisfies $u_{i}(D)=0$ and $u_{i}(x^{*})=1$ for $\boldsymbol{v}=1$ 2.Then $u_{1}(x)+u_{2}(x)=2$ for all Pareto efficient agreements $x\in X$ It is easy to verify that the problem $\langle X,D,\succsim_{1}^{\prime},\succsim_{2}^{\prime}\rangle$ is convex.(One

way to do so is to verify that the set of pairs of utilities is the triangle $\{(v_{1},v_{2}){:}v_{1}+v_{2}\leq2$ and $v_i\geq0$ for $i=1$, $2\}$ .To show this, use the fact that since $B_{i}$ is Pareto efficient and $B_{i}\sim_{j}D$ we have $u_{j}(B_{i})=0$ and $u_{i}(B_{i})=2$ .) To see that the problem is symmetric, define $\phi{:}X\to X$ by $\phi(D)=D$ and $\left[p\cdot B_{1}\sim_{1}^{\prime}x\right]$ and $q\cdot B_{2}\sim_{2}^{\prime}x$ if and only if $\left[p\cdot B_{2}\sim_{2}^{\prime}\phi(x)\right]$ and $q\cdot B_{1}\sim_{1}^{\prime}\phi(x)$ .This function $\phi$ assigns an agreement with the utilities $(v_1,v_2)$ to an agreement with utilities $(v_2,v_1)$ .Thus, an efficient agreement that is a fixed point of $\phi$ yields the pair of utilities (1,1) and hence by non-redundancy is $x^*$ .Thus by SYM and PAR we have $F(X,D,\succsim_{1}^{\prime},\succsim_{2}^{\prime})=x^{*}$ Now, the pair of problems $\langle X,D,\succsim_{1}^{\prime},\succsim_{2}^{\prime}\rangle$ and $\langle X,D,\succsim_{1},\succsim_{2}^{\prime}\rangle$ and the

pair of problems $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ and $\langle X,D,\succsim_{1},\succsim_{2}^{\prime}\rangle$ satisfy the hypothesis of IIA since by Step l we have $[1/(2-p)]\cdotp x\textit{ さ}_jx^*$ if $x\sim_ip\cdot x^*$ . Therefore $F(X,D,\succsim_{1},\succsim_{2})=F(X,D,\succsim_{1}^{\prime},\succsim_{2}^{\prime})=x^{*}$

As noted earlier, Nash defined a bargaining problem to be a pair $\langle U,d\rangle$ , where $U\subseteq\mathbb{R}^{2}$ is a compact convex set (the set of pairs of payoffs to agreements) and $d\in U$ (the pair of payoffs in the event of disagree ment). A bargaining solution in this context is a function that assigns

---

a point in $U$ to every bargaining problem $\langle U,d\rangle$ .Nash showed that there is a unique bargaining solution that satisfies axioms similar to those considered above and that this solution assigns to the bargaining problem $\langle U,d\rangle$ the pair $(v_1,v_2)$ of payoffs in $U$ for which the product $(v_{1}-d_{1})(v_{2}-d_{2})$ is highest. The following exercise asks you to prove this result

ExERcisE 309.1 Show, following the line of the proof of the previous result, that in the standard Nash bargaining model (as presented in the previous paragraph) there is a unique bargaining solution that satisfies analogs of PAR and SYM and the following two axioms, in which $f$ denotes a bargaining solution

(Covariance with positive affine transformations) Let $\langle U,d\rangle$ be a bargaining problem, let $\alpha_{i}>0$ and $\beta_{i}$ be real numbers, let

$U^{\prime}=\{(v_{1}^{\prime},v_{2}^{\prime}){:}v_{i}^{\prime}=\alpha_{i}v_{i}+\beta_{i}$ for $i=1$ , 2 for some $(v_{1},v_{2})\in U\}$

and let $d_{i}^{\prime}=\alpha_{i}d_{i}+\beta_{i}$ for $i=1$ ,2.Then $f_{i}(U^{\prime},d^{\prime})=\alpha_{i}f_{i}(U,d)+\beta_{i}$ for $i=1$ ,2.

(Independence of irrelevant alternatives) If $U\subseteq U^{\prime}$ and $f( U^{\prime }, d)$ $\in U$ then $f(U^{\prime},d)=f(U,d)$

#### 15.3.3 Is Any Aziom Superfluous?

We have shown that the axioms PAR, SYM, and IIA uniquely define the Nash solution; we now show that none of these axioms is superfuous. We do so by exhibiting, for each axiom, a bargaining solution that is different from Nash’s and satisfies the remaining two axioms.

PAR: Consider the solution defined by $F(X,D,\succsim_{1},\succsim_{2})=D$ .This satisfies SYM and IIA and differs from the Nash solution

ExERCIsE 309.2 Show that there is a solution $F$ different from the Nash solution that satisfies SYM, IIA, and $F(X,D,\succsim_{1},\succsim_{2})\succ_{i}D$ for $i=1$ ，2(strict individual rationality).Roth (1977) shows that in the standard Nash bargaining model (as presented in the previous exercise) the axioms SYM, IIA,and strict individual rationality are sufficient to characterize the Nash solution. Account for the difference

SYM: For each $\alpha\in(0,1)$ consider the solution (an asymmetric Nash solution) that assigns to $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ the agreement $x^*$ for which $( u_{1}( x^{* }) ) ^{\alpha }( u_{2}( x^{* }) ) ^{1- \alpha }$ $\geq$ $( u_{1}( x) ) ^{\alpha }( u_{2}( x) ) ^{1- \alpha }$ for all $x\in X$ ，where $u_{1}$ and $u_{2}$ represent \succsim1 and $\succsim_{2}$ and satisfy $u_{i}(D)=0$ for $i=1$ ,2

---

ExERCisE 310.1 Show that any asymmetric Nash solution is well defined (the agreement that it selects does not depend on the utility functions chosen to represent the preferences), satisfies PAR and IIA, and, for $\alpha\neq\frac{1}{2}$ , differs from the Nash solution.

IIA: Let $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ be a bargaining problem and let $u_{i}$ be a utility function that represents \succsim$i$ and satisfies $u_{i}( D)$ = 0 for $i=1$ ，2 The Kalai-Smorodinsky solution assigns to $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ the Pareto efficient agreement $JL$ for which $u_{1}(x)/u_{2}(x)=u_{1}(B_{1})/u_{2}(B_{2})$

ExERCisE 310.2Show that the Kalai-Smorodinsky solution is welldefined, satisfies SYM and PAR, and differs from the Nash solution

### 15.4 The Nash Solution and theBargaining Game of Alternating Offers

We now show that there is a close relationship between the Nash solution and the subgame perfect equilibrium outcome of the bargaining game of alternating offers studied in Chapter 7, despite the different methods that are used to derive them.

Fix a bargaining problem $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ and consider the version of the bargaining game of alternating offers described in Section 7.4.4, in which the set of agreements is $X$ ,the preference relations of the players are $\succsim_{1}$ and $\succsim2$ , and the outcome that results if negotiations break down at the end of a period, an event with probability $\alpha\in(0,1)$ ,is $D$ .Under assumptions analogous to A1A4 (Section 7.3.1) this game has a unique subgame perfect equilibrium outcome: player 1 proposes $x^*(\alpha)$ , which player 2 accepts, where $(x^{*}(\alpha),y^{*}(\alpha))$ is the pair of Pareto efficient agree ments that satisfies $(1-\alpha)\cdot x^{*}(\alpha)\sim_{1}y^{*}(\alpha)$ and $(1-\alpha)\cdot y^{*}(\alpha)\sim_{2}x^{*}(\alpha)$ (see Exercise 130.2) ■ PROPOSITION 310.3 Let $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ be a bargaining problem.The

agreements $x^*(\alpha)$ and $y^*(\alpha)$ proposed by the players in every subgame perfect equilibriurm of the variant of the bargaining game of alternating offers associated with $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ in which there is a probability $UX$ of breakdown after any rejection both converge to the Nash solution of $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ a.5 $\alpha\rightarrow0$ Proof. Let $u_{i}$ represent the preference relation $\succsim_i$ and satisfy $u_{i}(D)=0$ for $i=1$ ，2. From the conditions defining $x^{*}(\alpha)$ and $y^*(\alpha)$ we have $u_{1}( x^{* }( \alpha ) ) u_{2}( x^{* }( \alpha ) )$ = $u_{1}( y^{* }( \alpha ) ) u_{2}( y^{* }( \alpha ) )$ Since $x^{* }( \alpha )$ $\succ _{1}$ $y^{* }( \alpha )$ for all $\alpha\in[0,1)$ we have $x^{* }( \alpha )$ $\succsim _{1}$ $z^{* }$ $\succsim _{1}$ $y^{* }( \alpha )$ ，where $z^*$ is the Nash

---

![](https://storage.simpletex.cn/view/fNGNyTNvcPglDaaFcAtI52q9LNNF77ZRf)

Figure 311.1 An illustration of the proof of Proposition 310.3

solution of $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ (see Figure 311.1). For any sequence $\left(\alpha_{k}\right)_{k=1}^{\infty}$ converging to $U$ we have $u_{i}(x^{*}(\alpha_{k}))-u_{i}(y^{*}(\alpha_{k}))\to0$ for $i=1$ ,2 by the definition of $x^*(\alpha_k)$ and $y^{*}(\alpha_{k})$ , so that $u_{i}(x^{*}(\alpha_{k}))$ and $u_{i}(y^{*}(\alpha_{k}))$ converge to $u_i(z^*)$ for $i=1$, 2 and thus $x^*(\alpha_k)$ and $y^*(\alpha_k)$ converge to $z^*$ (using non-redundancy).

### 15.5 An Exact Implementation of the Nash Solution

We now return to the implementation approach described in Chapter 10. A byproduct of the result in the previous section is that the bargaining game of alternating offers with risk of breakdown approximately SPE implements the Nash solution. We now describe an extensive game with perfect information that ezactly implements it. From the point of view of a planner this game has the advantage that it is simpler, in the sense that it involves a small number of stages. However, it has the disadvantage of being more remote from familiar bargaining procedures. Fix a set $X$ and an event $D$ and assume the planner wants to imple

ment the Nash solution for all pairs (\succsim1,\succsim2) for which $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ is a bargaining problem. Consider the extensive game form (with perfect information and chance moves) consisting of the following stages.

·Player 1 chooses $y\in X$ ·Player 2 chooses $x\in X$ and $p\in[0,1]$ With probability $1-p$ the game ends, with the outcome $D$ , and with probability $P$ it continues

---

·Player 1 chooses either $L$ or the lottery $P\cdot y$ ；this choice is the outcome. -PROPOsirion 312.1 The game form described above SPE-implements the Nash solution. ExERCISE 312.2 Let ${\mathcal{T}}^*$ be the Nash solution of $\langle X,D,\succsim_{1},\succsim_{2}\rangle$ .Show that $x^*$ is the unique subgame perfect equilibrium outcome of the game form when the players’ preferences are $(\succsim_{1},\succsim_{2})$

### Notes

The seminal paper on the topic of this chapter is Nash (1950b)

Our presentation follows Rubinstein, Safra, and Thomson (1992) Zeuthen (1930, Ch. IV) contains an early model in which negotiators bear in mind the risk of a breakdown when making demands. The con- nection between the Nash solution and the subgame perfect equilibrium outcome of a bargaining game of alternating offers was first pointed out by Binmore (1987) and was further investigated by Binmore, Rubinstein, and Wolinsky (1986). The exact implementation of the Nash solution in Section 15.5 is due to Howard (1992) The comparative static result of Section 15.2.3 concerning the effect of

the players degree of risk aversion on the solution was first explored by Kihlstrom, Roth, and Schmeidler (1981). Harsanyi and Selten (1972) study the asymmetric Nash solutions described in Section 15.3.3 (axiomatizations appear in Kalai (1977) and Roth (1979, p. 16)) and Kalai and Smorodinsky (1975) axiomatize the Kalai-Smorodinsky solution. Exercise 309.2 is based on Roth (1977) Several other papers (e.g. Roemer (1988) study models in which the set of physical agreements, rather than the resulting set of utility pairs (as in Nash's model), is a primitive. Roth (1979) and Kalai (1985) are surveys of the field of axiomatic bargaining theory.

---

## List of Results

This is a list of the main results in the book, stated informally. It is designed to give an overview of the properties of the solutions that we study. Not all conditions are included in the statements; refer to the complete statements in the text for details.

### Strategic Games

Nash Equilibrium and Mized Strategy Equilibrium

(Nash equilibrium eristence) Every game in which the action set of each player is compact and convex and the preference relation of each player is continuous and quasi-concave has a Nash equilibriun

Proposition 20.3

■ A symmetric game has a symmetric Nash equilibrium Exercise 20.4 In a strictly competitive game that has a Nash equilibrium, a pair of actions is a Nash equilibrium if and only if each action is a maxminimizer Proposition 22.2 (Mized strategy equilibrium ezistence）Every finite game has a mixed strategy Nash equilibrium Proposition 33.1 A mixed strategy profile is a mixed strategy Nash equilibrium of a finite game if and only if every player is indifferent between all actions in the Lemma 33.2 support of his equilibrium strategy A strategy profile in a finite two-player strategic game is a trembling hand perfect equilibrium if and only if it is mixed strategy Nash equilibrium and the strategy of neither player is weakly dominated

---

Trembling hand perfect equilibrium existence) Every finite strategic Proposition 249.1 game has a trembling hand perfect equilibrium

Correlated Equilibrium

Every mixed strategy Nash equilibrium corresponds to a correlated equilibrium Proposition 45.3

Every convex combination of correlated equilibrium payoff profiles is a correlated equilibrium payoff profile Proposition 46.2

Every correlated equilibrium outcome is the outcome of a correlated equilibrium in which the set of states is the set of action profiles Proposition 47.1

Rationalizability

Every action used with positive probability in a correlated equilibrium is rationalizable Lemma 56.2

An action is a never-best response if and only if it is strictly dominated Lemma 60.1

An action that is not weakly dominated is a best response to a completely Exercise 64.2 mixed belief

Actions that survive iterated elimination of strictly dominated actions Proposition 61.2 are rationalizable

Knowledge

(Individuals cannot agree to disagree）If two individuals have the same prior and their posterior beliefs are common knowledge then these beliefs are the same Proposition 75.1

If each player is rational, knows the other players’ actions, and has a belief consistent with his knowledge, then the action profile is a Nash Proposition 77.1 equilibrium

If there are two players and each player knows that the other player is rational, knows the other player’s belief, and has a belief consistent with his knowledge, then the pair of beliefs is a mixed strategy Nash Proposition 78.1 equilibrium

---

If it is common knowledge that each player is rational and that each players’ belief is consistent with his knowledge then each player's action is rationalizable Proposition 80.1

If all players are rational in all states, every player's belief in every state is derived from a common prior, and each player's action is the same in all states in any given member of his information partition, then the information partitions and actions correspond to a correlated Exercise 81.1 equilibrium

### Extensive Games with Perfect Information

Basic Theory

A strategy profile is a subgame perfect equilibrium of a finite horizon game if and only if it has the one deviation property Lemma 98.2, Exercise 102.1, Exercise 103.3

(Subgame perfect equilibrium existence: Kuhn's theorem) Every finite game has a subgame perfect equilibriun Proposition 99.2, Exercise 102.1

All players are indifferent among all subgame perfect equilibria of a finite game that satisfies the no indifference condition, and all equilibria are Exercise 100.2 interchangeable

Bargaining Games

A bargaining game of alternating offers that satisfies A1A4 has a unique Proposition 122.1 subgame perfect equilibrium outcome

In a subgame perfect equilibrium of a bargaining game of alternating offers, a player is worse off the more impatient he is Proposition 126.1 Infinitely Repeated Games (Nash folk theoremn for limit of means）Every feasible enforceable payoff profile of the constituent game is a Nash equilibrium payoff profile of the limit of means infinitely repeated game Proposition 144.3 (Nash folk theorem for discounting) Every feasible strictly enforceable. payoff profile of the constituent game is close to a Nash equilibrium payoff profile of the discounting infinitely repeated game for a discount factor close enough to 1 Proposition 145.2

---

（Perfect folk theorem for limit of means）Every feasible strictly enforce able payoff profile of the constituent game is a subgame perfect equilib rium payoff profile of the limit of means infinitely repeated game Proposition 146.2

(Perfect folk theorem for overtaking) For every strictly enforceable outcome of the constituent game there is a subgame perfect equilibrium of the overtaking infinitely repeated game consisting of a repetition of the outcome Proposition 149.1

(Perfect folk theorem for discounting) For every feasible strictly enforce-. able outcome of a full-dimensional constituent game there is a discount factor close enough to 1 for which there is a subgame perfect equilibrium of the discounting infinitely repeated game consisting of a repetition of the outcome Proposition 151.1

A strategy profile is a subgame perfect equilibrium of a discounted infinitely repeated game if and only if it has the one deviation property Lemma 153.1

For any subgame perfect equilibrium outcome of a discounted infinitely repeated game there is a strategy profile that generates the same outcome in which the sequence of action profiles that follows a deviation depends only on the identity of the deviant (not on the history or on the nature of the deviation) Proposition 154.1

In every equilibrium of a machine game of a discounted infinitely repeated game there is a one-to-one correspondence between the actions Lemma 170.1 chosen by the two machines in the repeated game Every equilibrium of a machine game of a discounted infinitely repeated game consists of an introductory phase, in which all the states are distinct， followed by a cycling phase, in each cycle of which each state Proposition 171.1 appears at most once Finitely Repeated Games If the payoff profile in every Nash equilibrium of the constituent game is the profile of minmax payoffs then every Nash equilibrium of the finitely repeated game generates a sequence of Nash equilibria of the constituent Proposition 155.1 game (Nash folk theorem for finitely repeated games）If the constituent game has a Nash equilibrium in which every player's payoff exceeds his min max payoff then for any strictly enforceable outcome there is a Nash

---

equilibrium of the finitely repeated game in which each player’s payof is close to his payoff from the outcome Proposition 156.1

If the constituent game has a unique Nash equilibrium payof profle then every subgame perfect equilibrium of the finitely repeated game generates a sequence of Nash equilibria of the constituent game Proposition 157.2

(Perfect folk theorem for finitely repeated games）If the constituent game is full dimensional and for every player there are two Nash equilibria that yield different payoffs then for any strictly enforceable outcome a sufficiently long finitely repeated game has a subgame perfect equilibrium in which each player’s payoff is close to his payoff from the outcome Proposition 159.1

### Implementation Theory

(GibbardSatterthwaite theoren) In an environment in which there are at least three consequences and any preference ordering is possible, any choice rule that is DSE-implementable and satisfies the condition that for any consequence there is a preference profile for which the choice rule induces that consequence is dictatorial Proposition 181.2

(Revelation principle for DSE-implementation）If a choice rule is DSE implementable then it is truthfully DSE-implementable. Lemma 181.4

(Revelation principle for Nash-implementation) If a choice rule is Nashimplementable then it is truthfully Nash-implementable. Lemma 185.2

If a choice rule is Nash-implementable then it is monotonic Proposition 186.2

In an environment in which there are at least three players, a choice rule that is monotonic and has no veto power is Nash-implementable Proposition 187.2

In an environment in which there are at least three players, who can be required to pay monetary fines, every choice function is virtually SPEimplementable Proposition 193.1

### Extensive Games with Imperfect Information.

For any mixed strategy of a player in a finite extensive game with perfect recall there is an outcome-equivalent behavioral strategy Proposition 214.1

---

Every sequential equilibrium of the extensive game associated with a finite Bayesian game with observable actions induces a perfect Bayesian Proposition 234.1 equilibrium of the Bayesian game

Every trembling hand perfect equilibrium of a finite extensive game with perfect recall is associated with a sequential equilibrium Proposition 251.2

(Trembling hand perfect equilibrium and sequential equilibrium existence) Every finite extensive game with perfect recall has a trembling hand perCorollary 253.2 fect equilibrium and hence a sequential equilibrium

### Coalitional Games

Core

A coalitional game with transferable payoff has a nonempty core if and only if it is balanced Proposition 262.1

Every market with transferable payoff has a nonempty core

Proposition 264.2

Every profile of competitive payoffs in a market with transferable payoff is in the core of the market Proposition 267.1

Every competitive allocation in an exchange economy is in the core Proposition 272.1

If every agent's preference relation is increasing and strictly quasiconcave and every agent's endowment of every good is positive, the core Proposition 273.1 converges to the set of competitive allocations

Stable Sets

The core is a subset of every stable set; no stable set is a proper subset of any other; if the core is a stable set then it is the only stable set Proposition 279.2

Bargaining Set, Kernel, Nucleolus

In a coalitional game with transferable payoff the nucleolus is a member of the kernel, which is a subset of the bargaining set Lemmas 285.1 and 287.1

The nucleolus of any coalitional game with transferable payoff is a sinProposition 288.4 gleton

---

Shapley Value

The unique value that satisfies the balanced contributions property is Proposition 291.3 the Shapley value The Shapley value is the only value that satisfies axioms of symmetry, dummy, and additivity Proposition 293.1 Nash Solution The definition of the Nash solution of a bargaining problem in terms of objections and counterobjections is equivalent to the definition of it as the agreement that maximizes the product of the players² von NeumannProposition 302.1 Morgenstern utilities In the Nash solution a player is worse off the more risk-averse he is Proposition 304.1 The Nash solution is the only bargaining solution that satisfies axioms of Pareto efficiency, symmetry, and independence of irrelevant alternatives Proposition 307.2 The agreements proposed by the players in every subgame perfect equi librium outcome of the variant of a bargaining game of alternating offers in which there is a risk of breakdown converge to the Nash solution Proposition 310.3

---
