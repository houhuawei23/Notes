[PIR](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/paddle_ir_cn.html)


# PIR 基本概念和开发[¶](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/paddle_ir_cn.html#pir "永久链接至标题")

在 3.0 版本下，飞桨研发了基于 MLIR 范式的新一代中间表示技术，即 Paddle IR（下简称 PIR）。这项技术对底层的核心概念如 Operation、Attribute 等进行了系统性的抽象，为开发者提供了灵活的基础组件；同时，通过引入 Dialect 这一概念，飞桨能够全面、分层次管理框架各模块对中间表示的需求，并支持开发者根据需求定制化扩展 Dialect，显著提升了框架的扩展性。PIR 遵循 SSA（即 Static Single Assignment）原则，统一了顶层结构，实现“算子顺序性”和“计算图语义”的兼容表示。此外，PIR 还提供了更加简洁、低成本的 Pass 开发体系，并内置了一系列丰富且功能完备的 Pass 优化策略，为大模型的极致性能优化提供了强有力支撑。


在深度学习框架 IR 概念中，「顺序性」和「图语义」是两个非常高频常用的概念。旧的中间表示体系由「顺序性」ProgramDesc 和「图语义」Graph 两个核心类共同承载。用户在静态图 API 或者动转静模块下，产生的中间表示是 Op-by-Op 的 Program，如果要应用更高层面的优化策略（比如算子融合、inplace 策略、剪枝等），框架会将由 Program 构造出 Graph，其由数据节点、算子节点和彼此关联的边构成。 在新的 Paddle IR 中，飞桨在底层抽象了一套高度可扩展的基础组件，包括 Type、Attrbute、Op、Trait 和 Interface，并引入了 Dialect 的概念，支持开发者灵活扩展、自由定制，提供了完备鲁邦的语义表达能力；在模型表示层，通过多 Dialect 模块化管理，统一多端表示，实现了训推一体的全架构统一表示，无缝衔接组合算子和编译器，支持自动优化和多硬件适配；在图变换层，通过统一底层模块，简化基础概念，向用户提供了低成本开发、易用高性能、丰富可插拔的 Pass 优化机制。 飞桨的新一代的 IR 表示坚持 SSA（静态单赋值）原则，模型等价于一个有向无环图。并以 Value、Operation 对计算图进行抽象， Operation 为节点，Value 为边。

* Operation 表示计算图中的节点：一个 Operation 表示一个算子，它里面包含了零个或多个 Region；Region 表示一个闭包，它里面包含了零个或多个 Block；Block 表示一个符合 SSA 的基本块，里面包含了零个或多个 Operation；三者循环嵌套，可以实现任意复杂的语法结构
* Value 表示计算图中的有向边：用来将两个 Operaton 关联起来，描述了程序中的 UD 链（即 Use-Define 链）；OpResult 表示定义端，定义了一个 Value，OpOperand 表示使用端，描述了对一个 Value 的使用。

## 二、设计初衷[¶](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/paddle_ir_cn.html#ershejichuzhong "永久链接至标题")

计算图中间表示（Intermediate Representation，即 IR）是深度学习框架性能优化、推理部署、编译器等方向的重要基石。近些年来，越来越多的框架和研究者将编译器技术引入到深度学习的神经网络模型优化中，并在此基础上借助编译器的理念、技术和工具对神经网络进行自动优化和代码生成。飞桨历史上在架构层面并存着多套不同的中间表示体系，其表达能力各不相同、Pass 开发维护成本较高，代码复用性较差，缺乏统一规范，存在严重的框架稳定性问题。
