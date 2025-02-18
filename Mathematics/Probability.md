# Probability 概率论

## Monty Hall Problem

[Monty_Hall_problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)

蒙提霍尔问题（又称“三门问题”）

Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?

假设你参加一个游戏节目，面前有三扇门：一扇门后是汽车，另外两扇门后则是山羊。你选择了一扇门，比如 1 号门，而主持人知道门后的情况，他打开了另一扇门，比如 3 号门，展示给你看后面是一只山羊。然后他问你：“你想改选 2 号门吗？”改变选择对你有利吗？

分析:

- 策略 1: 第一次选择后，主持人打开一扇门后，不改选另一扇门
  - 策略 1 赢得汽车的概率为: 1/3
  - 因为第一次选择选到汽车的概率为 1/3, 此后主持人无论开多少门，不影响选对概率
- 策略 2: 第一次选择后，主持人打开一扇门后，改选另一扇门
  - 策略 2 赢得汽车的概率为: 2/3
  - 第一次没有选到汽车的概率为 2/3，主持人从剩下的 2 个门中排除一个，改选一定选到汽车

常见疑惑:

> 当主持人打开一扇门后，剩余 2 扇门，我相当于“随机选择”了一扇门，选中汽车的概率为 1/2，所以是否改选没有影响。

这样想的问题在于，这“门背后是汽车的概率” != “选中该门的概率”。你可以以 1/2 的概率选中一扇门，但是这扇门背后是汽车的概率并不是相等的，所以你“选中这扇门的概率” != “选中这扇门后中奖的概率”。当你坚持第一次选择的门时，该门背后是汽车的概率已经是 1/3 了。

可以通过条件概率（贝叶斯定理）计算出换门后获胜的概率。

宏观上看，主持人必然排除一个空门，是非随机的，该操作传递了额外的信息。“改选门” 利用了该信息，提升了中奖概率。

- 主持人并非随机排除门，而是基于参与者初始选择和奖品位置。
- 初始选择的 1/3 概率未改变，剩余门的概率继承自初始未选的 2/3，仅被浓缩至一扇门。

直观验证：

- 枚举所有情况：
  - 奖品在门 1：
    - 选门 1 -> 换必输。
    - 选门 2 -> 主持人开 3 -> 换必赢。
    - 选门 3 -> 主持人开 2 -> 换必赢。
  - 换门赢的概率：2/3。
