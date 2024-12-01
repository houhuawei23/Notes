# GPT-Academic Report
## # Title:



Polygeist: Raising C to Polyhedral MLIR

## # Abstract:



We present Polygeist, a new compilation flow that connects the MLIR compiler infrastructure to cutting edge polyhedral optimization tools. It consists of a C and C++ frontend capable of converting a broad range of existing codes into MLIR suitable for polyhedral transformation and a bi-directional conversion between MLIR and OpenScop exchange format. The Polygeist/MLIR intermediate representation featuring high-level (affine) loop constructs and n-D arrays embedded into a single static assignment (SSA) substrate enables an unprecedented combination of SSA-based and polyhedral optimizations. We illustrate this by proposing and implementing two extra transformations: statement splitting and reduction parallelization. Our evaluation demonstrates that Polygeist outperforms on average both an LLVM IR-level optimizer (Polly) and a source-to-source state-of-the-art polyhedral compiler (Pluto) when exercised on the Polybench/C benchmark suite in sequential (2.53x vs 1.41x, 2.34x) and parallel mode (9.47x vs 3.26x, 7.54x) thanks to the new representation and transformations.

## # Meta Translation

标题：Polygeist: 将C提升到多面体MLIR

摘要：我们提出了Polygeist，这是一种新的编译流程，旨在将MLIR编译基础设施与前沿的多面体优化工具连接起来。它包括一个C和C++前端，能够将广泛的现有代码转换为适合多面体变换的MLIR，并实现MLIR与OpenScop交换格式之间的双向转换。Polygeist/MLIR中间表示具备高层（仿射）循环构造和嵌入到单一静态赋值（SSA）基础结构中的n维数组，能够实现前所未有的SSA基础和多面体优化的结合。我们通过提出和实现两个额外变换：语句分割和归约并行化，来证明这一点。我们的评估表明，Polygeist在Polybench/C基准套件的顺序模式（2.53x对比1.41x，2.34x）和并行模式（9.47x对比3.26x，7.54x）上的表现平均优于LLVM IR级别优化器（Polly）和先进的源到源多面体编译器（Pluto），这得益于新的表示和变换。

## # I. INTRODUCTION

Improving the efficiency of computation has always been one of the prime goals of computing. Program performance can be improved significantly by reaping the benefits of parallelism, temporal and spatial locality, and other performance sources. Relevant program transformations are particularly tedious and challenging when targeting modern multicore CPUs and GPUs with deep memory hierarchies and parallelism, and are often performed automatically by optimizing compilers.
The polyhedral model enables precise analyses and a relatively easy specification of transformations (loop restructuring, automatic parallelization, etc.) that take advantage of hardware performance sources. As a result, there is growing evidence that the polyhedral model is one of the best frameworks for efficient transformation of compute-intensive programs [1], [2], [3], and for programming accelerator architectures [4], [5], [6]. Consequently, the compiler community has focused on building tools that identify and optimize parts of the program that can be represented within the polyhedral model (commonly referred to as static-control parts, or SCoP's). Such tools tend to fall into two categories.
Compiler-based tools like Polly [7] and Graphite [8] detect and transform SCoPs in compiler intermediate representations (IRs). While this offers seamless integration with rest of the compiler, the lack of high-level structure and information hinders the tools' ability to perform analyses and transformations. This structure needs to be recovered from optimized IR, often Fig. 1. The Polygeist compilation flow consists of 4 stages. The frontend traverses Clang AST to emit MLIR SCF dialect (Section III-A), which is raised to the Affine dialect and pre-optimized (Section III-B). The IR is then processed by a polyhedral scheduler (Sections III-C,III-D) before postoptimization and parallelization (Section III-E). Finally, it is translated to LLVM IR for further optimization and binary generation by LLVM. imperfectly or at a significant cost [9]. Moreover, common compiler optimizations such as LICM may interfere with the process [10]. Finally, low-level IRs often lack constructs for, e.g., parallelism or reductions, produced by the transformation, which makes the flow more complex.
Source-to-source compilers such as Pluto [11], POCC [12] and PPCG [5] operate directly on C or C++ code. While this can effectively leverage the high-level information from source code, the effectiveness of such tools is often reduced by the lack of enabling optimizations such as those converting hazardous memory loads into single-assignment virtual registers. Furthermore, the transformation results must be expressed in C, which is known to be complex [13], [14] and is also missing constructs for, e.g., reduction loops or register values not backed by memory storage.
This paper proposes and evaluates the benefits of a polyhedral compilation flow, Polygeist (Figure 1), that can leverage both the high-level structure available in source code and the fine-grained control of compiler optimization provided by lowlevel IRs. It builds on the recent MLIR compiler infrastructure that allows the interplay of multiple abstraction levels within the same representation, during the same transformations [15]. Intermixable MLIR abstractions, or dialects, include highlevel constructs such as loops, parallel and reduction patterns; low-level representations fully covering LLVM IR [16]; and a polyhedral-inspired representation featuring loops and memory accesses annotated with affine expressions. Moreover, by combining the best of source-level and IR-level tools in an end-to-end polyhedral flow, Polygeist preserves high-level information and leverages them to perform new or improved %result = "dialect.operation"(%operand, %operand) {attribute = #dialect<"value">} ({ // Inside a nested region. ^basic_block(%block_argument: !dialect.type):
"another.operation"() : () -> () }) : (!dialect.type) -> !dialect.result_type Fig. 2. Generic MLIR syntax for an operation with two operands, one result, one attribute and a single-block region.
optimizations, such as statement splitting and loop-carried value detection, on a lower-level abstraction as well as to influence downstream optimizations.
We make the following contributions:
• a C and C++ frontend for MLIR that preserves high-level loop structure from the original source code; • an end-to-end flow with raising to and lowering from the polyhedral model, leveraging our abstraction to perform more optimizations than both source-and IR-level tools, including reduction parallelization; • an exploration of new transformation opportunities created by Polygeist, in particular, statement splitting; • and an end-to-end comparison between Polygeist and state-of-the-art source-and IR-based tools (Pluto [11] and Polly [14]) along with optimization case studies.

I. 引言

提高计算效率一直是计算领域的主要目标之一。通过利用并行性、时间和空间局部性以及其他性能来源，可以显著提高程序性能。当针对现代多核 CPU 和 GPU 进行相关程序变换时，这些变换尤其繁琐且具有挑战性，因为现代架构具有深度内存层次和并行性，通常由优化编译器自动执行。

多面体模型能够实现精确分析和相对简单的变换规范（如循环重构、自动并行化等），以利用硬件性能来源。因此，有越来越多的证据表明，多面体模型是高效转化计算密集型程序 [1]、[2]、[3] 和编程加速器架构 [4]、[5]、[6] 的最佳框架之一。因此，编译器社区专注于构建工具，以识别和优化可以在多面体模型中表示的程序部分（通常称为静态控制部分或 SCoP）。这些工具往往分为两类。

基于编译器的工具，如 Polly [7] 和 Graphite [8]，在编译器中间表示（IR）中检测和转换 SCoP。尽管这与编译器的其余部分无缝集成，但缺乏高层次结构和信息限制了这些工具进行分析和变换的能力。这种结构通常需要通过优化后的 IR 恢复，往往是以不完全或高成本的方式实现 [9]。此外，诸如 LICM 的常见编译器优化可能会干扰这一过程 [10]。最后，低级 IR 通常缺乏由变换生成的并行性或归约的构造，从而使流程更加复杂。

源到源编译器，如 Pluto [11]、POCC [12] 和 PPCG [5]，直接在 C 或 C++ 代码上操作。虽然这可以有效地利用源代码中的高层信息，但这些工具的有效性往往因缺乏启用优化而降低，例如将危险的内存加载转换为单赋值虚拟寄存器。此外，变换结果必须用 C 表达，而 C 语言本身的复杂性众所周知 [13]、[14]，而且也缺乏例如归约循环或未被内存存储支持的寄存器值的构造。

本文提出并评估了一种多面体编译流程 Polygeist（图 1）的好处，该流程可以利用源代码中可用的高层结构以及由低级 IR 提供的编译器优化的细粒度控制。它构建在最近的 MLIR 编译器基础设施之上，该基础设施允许在同一表示中、在同一变换过程中混合多个抽象层次 [15]。可互换的 MLIR 抽象或方言包括高层构造，如循环、并行和归约模式；完整覆盖 LLVM IR 的低层表示 [16]；以及一种受到多面体启发的表示，具有带有仿射表达式注释的循环和内存访问。此外，通过将源级和 IR 级工具的优势结合在一个端到端的多面体流程中，Polygeist 保留了高层信息并利用这些信息在较低级别抽象上执行新的或改进的优化，如语句拆分和循环携带值检测，同时影响后续优化。

我们的贡献如下：
• 一个 C 和 C++ 的 MLIR 前端，保留原始源代码中的高层循环结构；
• 一个端到端的流程，实现向多面体模型的提升和降低，利用我们的抽象执行比源级和 IR 级工具更多的优化，包括归约并行化；
• 探索 Polygeist 创建的新变换机会，尤其是语句拆分；
• 以及 Polygeist 与最先进的基于源和 IR 的工具（Pluto [11] 和 Polly [14]）之间的端到端比较，以及优化案例研究。

## # A. Overview

MLIR is an optimizing compiler infrastructure inspired by LLVM [16] with a focus on extensibility and modularity [15]. Its main novelty is the IR supporting a fully extensible set of instructions (called operations) and types. Practically, MLIR combines SSA with nested regions, allowing one to express a range of concepts as first-class operations including machine instructions such as floating-point addition, structured control flow such as loops, hardware circuitry [17], and large machine learning graphs. Operations define the runtime semantics of a program and process immutable values. Compile-time information about values is expressed in types, and information about operations is expressed in attributes. Operations can have attached regions, which in turn contain (basic) blocks of further operations. The generic syntax, accepted by all operations, illustrates the structure of MLIR in Figure 2. Additionally, MLIR supports user-defined custom syntax.
Attributes, operations, and types are organized in dialects, which can be thought of as modular libraries. MLIR provides a handful of dialects that define common operations such as modules, functions, loops, memory or arithmetic instructions, and ubiquitous types such as integers and floats. We discuss the dialects relevant to Polygeist in the following sections.

A. 概述

MLIR是一种优化编译器基础设施，受到LLVM的启发，重点关注扩展性和模块化。其主要创新在于支持完全可扩展的一组指令（称为操作）和类型的中间表示（IR）。实际上，MLIR将静态单赋值（SSA）与嵌套区域相结合，使得能够将一系列概念表示为一类一流的操作，包括机器指令，例如浮点加法、结构化控制流，例如循环、硬件电路，以及大型机器学习图。操作定义了程序的运行时语义，并处理不可变值。有关值的编译时信息用类型表示，而有关操作的信息用属性表示。操作可以附带区域，而这些区域内部又包含其他操作的（基本）块。所有操作接受的通用语法在图2中展示了MLIR的结构。此外，MLIR支持用户自定义的自定义语法。

属性、操作和类型被组织在方言中，可以视为模块化库。MLIR提供了少量方言，以定义常见操作，如模块、函数、循环、内存或算术指令，以及像整数和浮点数这样的通用类型。我们将在接下来的章节中讨论与Polygeist相关的方言。

## # B. Affine and MemRef Dialects

The Affine dialect [18] aims at representing SCoP's with explicit polyhedral-friendly loop and conditional constructs. The core of its representation is the following classification of value categories: • Symbols-integer values that are known to be loopinvariant but unknown at compile-time, also referred to as program parameters in polyhedral literature, typically array dimensions or function arguments. In MLIR, symbols are values defined in the top-level region of an operation with "affine scope" semantics, e.g., functions; or array dimensions, constants, and affine map (see below) application results regardless of their definition point. • Dimensions-are an extension of symbols that also accepts induction variables of affine loops. • Non-affine-any other values. Symbols and dimensions have index type, which is a platform-specific integer that fits a pointer (intptr_t in C).
MLIR provides two attributes relevant for the Affine dialect:
• Affine maps are multi-dimensional (quasi-)linear functions that map a list of dimension and symbol arguments to a list of results. For example,
(d 0 , d 1 , d 2 , s 0 ) → (d 0 + d 1 , s 0 • d 2 )
is a two-dimensional quasi-affine map, which can be expressed in MLIR as affine_map<(d0,d1,d2)[s0] -> (d0+d1, s0 * d2)>. Dimensions (in parentheses on the left) and symbols (in brackets on the left) are separated to allow quasi-linear expressions: symbols are treated as constants, which can therefore be multiplied with dimensions, whereas a product of two dimensions is invalid. • Integer sets are collections of integer tuples constrained by conjunctions of (quasi-)linear expressions. For example, a "triangular" set {(d 0 , d 1 ) :
0 ≤ d 0 < s 0 ∧ 0 ≤ d 1 ≤ d 0 } is rep- resented as affine_set<(d0,d1)[s0]: (d0 >= 0, s0-d0-1 >= 0, d1 >= 0, d0-d1 >= 0)>.
The Affine dialect makes use of the concepts above to define a set of operations. An affine.for is a "for" loop with loop-invariant lower and upper bounds expressed as affine maps with a constant step. An affine.parallel is a "multifor" loop nest, iterations of which may be executed concurrently. Both kinds of loops support reductions via loopcarried values as well as max(min) expression lower(upper) bounds. An affine.if is a conditional construct, with an optional else region, and a condition defined as inclusion of the given values into an integer set. Finally, affine.load and affine.store express memory accesses where the address computation is expressed as an affine map. A core MLIR type-memref, which stands for memory reference-and the corresponding memref dialect are also featured in Figure 3. The memref type describes a structured multi-index pointer into memory, e.g., memref<?xf32> denotes a 1-d array of floating-point elements; and the memref dialect provides memory and type manipulation operations, e.g., memref.dim retrieves the dimensionality of a memref object. memref does not allow internal aliasing, i.e., different subscripts always point to different addresses. This effectively defines away the delinearization problem that hinders the application of polyhedral techniques at the LLVM IR level [9]. Throughout this paper, we only consider memrefs with the default layout that corresponds to contiguous row-major storage compatible with C ABI (Application Binary Interface). In practice, memrefs support arbitrary layouts expressible as affine maps, but these are not necessary in Polygeist context.

## B. 仿射和内存引用方言

仿射方言 [18] 旨在表示具有显式适合多面体的循环和条件构造的静态计算程序（SCoP）。其表示的核心是以下值类别的分类：
- **符号**——已知为循环不变但在编译时未知的整数值，在多面体文献中也称为程序参数，通常是数组维度或函数参数。在 MLIR 中，符号是在具有“仿射范围”语义的操作的顶级区域中定义的值，例如函数；或者数组维度、常量和仿射映射（见下文）应用结果，无论它们的定义点在哪里。
- **维度**——是符号的扩展，接受仿射循环的归纳变量。
- **非仿射**——任何其他值。符号和维度具有索引类型，即平台特定的整数，适合指针（在 C 中为 intptr_t）。

MLIR 提供了两个与仿射方言相关的属性：
- **仿射映射**是多维的（准）线性函数，它将一组维度和符号参数映射到一组结果。例如，

  \( (d_0, d_1, d_2, s_0) \rightarrow (d_0 + d_1, s_0 \cdot d_2) \)

  是一个二维准仿射映射，可以在 MLIR 中表示为 `affine_map<(d0,d1,d2)[s0] -> (d0+d1, s0 * d2)>`。维度（左侧的括号内）和符号（左侧的方括号内）被分隔开，以允许准线性表达式：符号被视为常量，因此可以与维度相乘，而两个维度的乘积是无效的。

- **整数集**是被（准）线性表达式的合取限制的整数元组集合。例如，“三角形”集合 \(\{(d_0, d_1) : 0 \leq d_0 < s_0 \land 0 \leq d_1 \leq d_0\}\) 表示为 `affine_set<(d0,d1)[s0]: (d0 >= 0, s0-d0-1 >= 0, d1 >= 0, d0-d1 >= 0)>`。

仿射方言利用上述概念定义了一组操作。`affine.for` 是一个“for”循环，其循环不变的下界和上界表示为带有常量步长的仿射映射。`affine.parallel` 是一个“多重for”循环嵌套，其迭代可以并发执行。这两种循环都支持通过循环传递值进行归约，以及 max(min) 表达式的下（上）界。`affine.if` 是一个条件构造，带有一个可选的 else 区域，条件被定义为给定值包含在一个整数集中。最后，`affine.load` 和 `affine.store` 表示内存访问，其中地址计算表示为仿射映射。核心的 MLIR 类型——内存引用（memref），以及相应的 memref 方言也在图 3 中展示。memref 类型描述了指向内存的结构化多索引指针，例如，`memref<?xf32>` 表示一个浮点元素的一维数组；而 memref 方言提供了内存和类型操作，例如 `memref.dim` 检索 memref 对象的维度。memref 不允许内部别名，即不同的下标始终指向不同的地址。这有效地解决了限制多面体技术在 LLVM IR 级别应用的去线性化问题 [9]。在本文中，我们只考虑与兼容 C ABI（应用程序二进制接口）的连续行主存储对应的默认布局的 memrefs。实际上，memrefs 支持可作为仿射映射表达的任意布局，但在 Polygeist 的上下文中，这些并不是必需的。

## # C. Other Relevant Core Dialects

MLIR provides several dozen dialects. Out of those, only a handful are relevant for our discussion:
• The Structured Control Flow (scf) dialect defines the control flow operations such as loops and conditionals that are not constrained by affine categorization rules. For example, the scf.for loop accepts any integer value as loop bounds, which are not necessarily affine expressions. • The Standard (std) dialect contains common operations such as integer and float arithmetic, which is used as a common lowering point from higher-level dialects before fanning out into multiple target dialects and can be seen as a generalization of LLVM IR [16]. • The LLVM dialect directly maps from LLVM IR instructions and types to MLIR, primarily to simplify the translation between them. • The OpenMP dialect provides a dialect-and platformagnostic representation of OpenMP directives such as "parallel" and "workshare loop", which can be used to transform OpenMP constructs or emit LLVM IR that interacts with the OpenMP runtime. • The Math dialect groups together mathematical operations on integer and floating type beyond simple arithmetic, e.g., math.pow or math.sqrt.

MLIR提供了几十种方言。在这些方言中，只有少数几种与我们的讨论相关：
- 结构化控制流（scf）方言定义了控制流操作，例如循环和条件语句，这些操作不受仿射分类规则的限制。例如，scf.for 循环可以接受任何整数值作为循环边界，这些边界不一定是仿射表达式。
- 标准（std）方言包含常见的操作，例如整数和浮点算术，这被用作从更高级别方言到多个目标方言的共同降低点，可以视为LLVM IR的概括[16]。
- LLVM方言直接将LLVM IR指令和类型映射到MLIR，主要是为了简化它们之间的转换。
- OpenMP方言提供了一种与方言和平台无关的OpenMP指令表示，例如“parallel”和“workshare loop”，可用于转换OpenMP构造或生成与OpenMP运行时交互的LLVM IR。
- 数学方言将整数和浮点类型上的数学操作（超出简单算术）归为一组，例如，math.pow或math.sqrt。

## # III. AN (AFFINE) MLIR COMPILATION PIPELINE

The Polygeist pipeline consists of 4 components (Figure 1):  4). This allows Polygeist to preserve more of the structure available within the original program (e.g., multi-dimensional arrays) and enables interaction with MLIR's high-level memory operations.
This diverges from the C ABI for any functions with pointer arguments and wouldn't interface correctly with C functions. Polygeist addresses this by providing an attribute for function arguments and allocations to use a C-compatible pointer type rather than memref, applied by default to external functions such as strcmp and scanf. When calling a pointer-ABI function with a memref-ABI argument, Polygeist generates wrapper code that recovers the C ABI-compatible pointer from memref and ensures the correct result. Figure 5 shows an example demonstrating how the Polygeist and C ABI may interact for a small program.
When allocating and deallocating memory, this difference in ABI becomes significant. This is because allocating several bytes of an array with malloc then casting to a memref will not result in legal code (as memref itself may not be implemented with a raw pointer). Thus, Polygeist identifies calls to allocation and deallocation functions and replaces them with legal equivalents for memref.
Functions and global variables are emitted using the same name used by the C or C++ ABI. This ensures that all external values are loaded correctly, and multi-versioned functions (such as those generated by C++ templates or overloading) have distinct names and definitions.
c) Instruction Generation: For most instructions, Polygeist directly emits an MLIR operation corresponding to the equivalent C operation (addi for integer add, call for function call, etc.). For some special instructions such as a call to pow, Polygeist chooses to emit a specific MLIR operation in the Math dialect, instead of a call to an external function (defined in libm). This permits such instructions to be better analyzed and optimized within MLIR.
Operations that involve memory or pointer arithmetic require additional handling. MLIR does not have a generic pointer arithmetic instruction; instead, it requires that load and store operations contain all of the indices being looked up. This presents issues for operations that perform pointer arithmetic. To remedy this, we introduce a temporary subindex operation for memref's keeps track of the additional address offsets. A subsequent optimization pass within Polygeist, forwards the offsets in a subindex to any load or store which uses them.
d) Local Variables: Local variables are handled by allocating a memref on stack at the top of a function. This permits the desired semantics of C or C++ to be implemented with relative ease. However, as many local variables and arguments contain memref types, this immediately results in a memref of a memref-a hindrance for most MLIR optimizations as it is illegal outside of Polygeist. As a remedy, we implement a heavyweight memory-to-register (mem2reg) transformation pass that eliminates unnecessary loads, stores, and allocations within MLIR constructs. Empirically this eliminates all memrefs of memref in the Polybench suite.

III. 一个（仿射）MLIR编译流程

Polygeist编译流程由4个组件组成（见图1）。这使得Polygeist能够保留原始程序中可用的更多结构（例如，多维数组），并能够与MLIR的高级内存操作进行交互。这与任何具有指针参数的C ABI不同，并且与C函数的接口不一致。Polygeist通过为函数参数和分配提供一个属性，使其使用与C兼容的指针类型，而不是memref，默认适用于外部函数，如strcmp和scanf。当使用memref ABI参数调用指针 ABI函数时，Polygeist生成包装代码，从memref中恢复与C ABI兼容的指针，并确保结果正确。图5展示了一个示例，演示了Polygeist与C ABI在一个小程序中的互动。

在分配和释放内存时，这种ABI的差异显得尤为重要。这是因为使用malloc分配数组的几个字节后，再将其转换为memref将不会产生合法代码（因为memref本身可能不是用原始指针实现的）。因此，Polygeist识别对分配和释放函数的调用，并将其替换为memref的合法等效函数。

函数和全局变量使用C或C++ ABI中使用的相同名称发出。这确保所有外部值都被正确加载，并且多版本函数（如C++模板或重载生成的函数）拥有不同的名称和定义。

c) 指令生成：对于大多数指令，Polygeist直接发出与等效C操作相对应的MLIR操作（如整数加法的addi，函数调用的call等）。对于某些特殊指令，如对pow的调用，Polygeist选择发出Math方言中特定的MLIR操作，而不是调用外部函数（在libm中定义）。这使得这些指令能够在MLIR中更好地进行分析和优化。

涉及内存或指针算术的操作需要额外处理。MLIR没有通用的指针算术指令；相反，它要求load和store操作包含所有被查找的索引。这为执行指针算术的操作带来了问题。为了解决这个问题，我们引入了一个临时子索引操作，用于memref，以跟踪附加的地址偏移。在Polygeist中的后续优化过程中，将子索引中的偏移量转发给任何使用它们的load或store。

d) 局部变量：局部变量通过在函数顶部分配一个memref在栈上进行处理。这使得C或C++的所需语义能够相对容易地实现。然而，由于许多局部变量和参数包含memref类型，这立即导致了memref的memref-这对大多数MLIR优化来说是一个障碍，因为在Polygeist之外这是不合法的。为了解决这个问题，我们实现了一个重量级的内存到寄存器（mem2reg）转换过程，该过程消除了MLIR构造中的不必要的加载、存储和分配。根据经验，这消除了Polybench套件中所有memref的memref。

## # B. Raising to Affine

The translation from C or C++ to MLIR directly preserves high-level information about loop structure and n-D arrays, but does not generate other Affine operations. Polygeist subsequently raises memory, conditional, and looping operations into their Affine dialect counterparts if it can prove them to be legal affine operations. If the corresponding frontend code was enclosed within #pragma scop, Polygeist assumes it is always legal to raise all operations within that region without additional checks. 1 Any operations which are not proven or assumed to be affine remain untouched. We perform simplifications on affine maps to remove loops with zero or one iteration and drop branches of a conditional with a condition known at compile time.

将C或C++直接翻译为MLIR能够保留关于循环结构和n维数组的高级信息，但不会生成其他的仿射操作。之后，Polygeist将内存、条件和循环操作提升至其仿射方言对应物，如果它能够证明这些操作是合法的仿射操作。如果相应的前端代码被包裹在`#pragma scop`中，Polygeist假设在该区域内提升所有操作是始终合法的，且无需额外检查。对于任何未被证明或假设为仿射的操作，则保持不变。我们对仿射映射进行简化，以去除具有零次或一次迭代的循环，并删除在编译时已知条件的条件语句的分支。

## # a) Memory operations and loop bounds:

To convert an operation, Polygeist replaces its bound and subscript operands with identity affine maps (affine_map<() [s0]->(s0)>[%bound]). It then folds the operations computing the map operands, e.g., addi, muli, into the map itself. Values that are transitively derived from loop induction variables become map dimensions and other values become symbols. For example, affine_map< ()[s0]->(s0)> [%bound] with %bound = addi %N, %i, where %i is an induction variable, is folded into
affine_map<(d0)[s0] ->(s0 + d0)>(%i)[%N].
The process terminates when no operations can be folded or when Affine value categorization rules are satisfied.
b) Conditionals: Conditional operations are emitted by the frontend for two input code patterns: if conditions and ternary expressions. The condition is transformed by introducing an integer set and by folding the operands into it similarly to the affine maps, with in addition and operations separating set constraints and not operations inverting them (affine.if only accepts ≥ 0 and = 0 constraints). Polygeist processes nested conditionals with C-style shortcircuit semantics, in which the subsequent conditions are checked within the body of the preceding conditionals, by hoisting conditions outside the outermost conditional when legal and replacing them with a boolean operation or a select. This is always legal within #pragma scop.
Conditionals emitted for ternary expressions often involve memory loads in their regions, which prevent hoisting due to side effects. We reuse our mem2reg pass to replace those to equivalent earlier loads when possible to enable hoisting. Empirically, this is sufficient to process all ternary expressions in the Polybench/C suite [19]. Otherwise, ternary expressions would need to be packed into a single statement by the downstream polyhedral pass.

a) 内存操作和循环边界：为了转换一个操作，Polygeist 用恒等仿射映射（affine_map<() [s0]->(s0)>[%bound]）替换其边界和下标操作数。它然后将计算映射操作数的操作，例如加法（addi）、乘法（muli），折叠到映射本身中。从循环归纳变量传递得出的值变成了映射维度，其他值则变成符号。例如，affine_map< ()[s0]->(s0)> [%bound] 在 %bound = addi %N, %i 的情况下，其中 %i 是一个归纳变量，被折叠为 affine_map<(d0)[s0] ->(s0 + d0)>(%i)[%N]。该过程在无法再折叠操作或满足仿射值分类规则时终止。

b) 条件语句：前端为两种输入代码模式（if 条件和三元表达式）生成条件操作。条件通过引入整数集合并将操作数折叠到其中进行转换，类似于仿射映射，此外，操作之间的加法和运算分隔集合约束，而非运算则对其进行反转（affine.if 仅接受 ≥ 0 和 = 0 的约束）。Polygeist 处理嵌套条件时遵循 C 风格的短路语义，其中后续条件在前面条件的主体内进行检查，当法律允许时，将条件提升到最外层条件之外，并用布尔操作或选择操作替换它们。在 #pragma scop 内，这总是合法的。

为三元表达式生成的条件语句通常涉及其区域内的内存加载，由于副作用，导致无法提升。我们重用 mem2reg 通道，在可能的情况下将其替换为等效的早期加载，以实现提升。从经验上看，这足以处理 Polybench/C 套件中的所有三元表达式 [19]。否则，三元表达式将需要被下游多面体通道打包成单个语句。

## # C. Connecting MLIR to Polyhedral Tools Part-1

Regions of the input program expressed using MLIR Affine dialect are amenable to the polyhedral model. Existing tools, however, cannot directly consume MLIR. We chose to implement a bi-directional conversion to and from OpenScop [20], an exchange format readily consumable by numerous polyhedral tools, including Pluto [11], and further convertible to isl [21] representation. This allows Polygeist to seamlessly connect with tools created in polyhedral compilation research without having to amend those tools to support MLIR.
Most polyhedral tools are designed to operate on C or FORTRAN inputs build around statements, which do not have a direct equivalent in MLIR. Therefore, we design a mechanism to create statement-like structure from chains of MLIR void setArray(int N, double val, double * array) {...} int main(int argc, char ** argv) { ... cmp = strcmp(str1, str2) ... double array [10]; setArray(10, 42.0, array) } ⇓ func @setArray(%N: i32, %val: f64, %array: memref<?xf64>) { %0 = index_cast %N : i32 to index affine.for %i = 0 to %0 { affine.store %val, %array[%i] : memref<?xf64> } return } func @main(%argc: i32, %argv: !llvm.ptr<ptr<i8>>) -> i32 { ... %cmp = llvm.call @strcmp(%str1, %str2) :
(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32 ... %array = memref.alloca() : memref<10xf64> %arraycst = memref.cast %array : memref<10xf64> to memref<?xf64> %val = constant 42.0 : f64 call @setArray(%N, %val, %arraycst) :
(i32, f64, memref<?xf64>) -> () } Fig. 5. Example demonstrating Polygeist ABI. For functions expected to be compiled with Polygeist such as setArray, pointer arguments are replaced with memref's. For functions that require external calling conventions (such as main/strcmp), Polygeist falls back to emitting llvm.ptr and generates conversion code.
operations. We further demonstrate that this gives Polygeist an ability to favorably affect the behavior of the polyhedral scheduler by controlling statement granularity (Section III-D).

输入程序中使用 MLIR Affine 方言表达的区域适用于多面体模型。然而，现有工具无法直接使用 MLIR。因此，我们选择实现与 OpenScop [20] 的双向转换，这是一种可被众多多面体工具直接使用的交换格式，包括 Pluto [11]，并且进一步可转换为 isl [21] 表示。这使得 Polygeist 能够与在多面体编译研究中创建的工具无缝连接，而无需修改这些工具以支持 MLIR。

大多数多面体工具设计用于处理以语句为基础的 C 或 FORTRAN 输入，而它们在 MLIR 中并没有直接对应的等价物。因此，我们设计了一种机制，从一系列 MLIR 操作创建类似语句的结构。例如： 

```
void setArray(int N, double val, double * array) {...} 
int main(int argc, char ** argv) { ... 
cmp = strcmp(str1, str2) ... 
double array [10]; 
setArray(10, 42.0, array) }
```

可转换为：

```
func @setArray(%N: i32, %val: f64, %array: memref<?xf64>) { 
    %0 = index_cast %N : i32 to index 
    affine.for %i = 0 to %0 { 
        affine.store %val, %array[%i] : memref<?xf64> 
    } 
    return 
} 

func @main(%argc: i32, %argv: !llvm.ptr<ptr<i8>>) -> i32 { 
    ... 
    %cmp = llvm.call @strcmp(%str1, %str2) : 
    (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.i32 
    ... 
    %array = memref.alloca() : memref<10xf64> 
    %arraycst = memref.cast %array : memref<10xf64> to memref<?xf64> 
    %val = constant 42.0 : f64 
    call @setArray(%N, %val, %arraycst) : 
    (i32, f64, memref<?xf64>) -> () 
}
```

图 5. 示例演示了 Polygeist 的 ABI。对于预期要由 Polygeist 编译的函数（如 setArray），指针参数被替换为 memref。对于需要外部调用约定（如 main/strcmp）的函数，Polygeist 退回到发出 llvm.ptr 并生成转换代码的方式。

我们进一步证明，这使得 Polygeist 能够通过控制语句粒度，积极影响多面体调度器的行为（详见第 III-D 节）。

## # C. Connecting MLIR to Polyhedral Tools Part-2

a) Simple Statement Formation: Observing that C statements amenable to the polyhedral model are (mostly) variable assignments, we can derive a mechanism to identify statements from chains of MLIR operations. A store into memory is the last operation of the statement. The backward slice of this operation, i.e., the operations transitively computing its operands, belong to the statement. The slice extension stops at operations producing a value categorized as affine dimension or symbol, directly usable in affine expressions. Such values are loop induction variables or loop-invariant constants.
Some operations may end up in multiple statements if the value is used more than once. However, we need the mapping between operations and statements to be bidirectional in order to emit MLIR after the scheduler has restructured the program without considering SSA value visibility rules. If an operation with multiple uses is side effect free, Polygeist simply duplicates it. For operations whose duplication is illegal, Polygeist stores their results in stack-allocated memref's and replaces all further uses with memory loads. Figure 6 illustrates the transformation for value %0 used in operation %20. This creates a new statement.
b) Region-Spanning Dependencies: In some cases, a statement may consist of MLIR operations across different (nested) loops, e.g., a load from memory into an SSA register happens in an outer loop while it is used in inner loops. The  location of such a statement in the loop hierarchy is unclear. More importantly, it cannot be communicated to the polyhedral scheduler. Polygeist resolves this by storing the value in a stack-allocated memref in the defining region and loading it back in the user regions. Figure 6 illustrates this transformation for value %0 used in operation %10. Similarly to the basic case, this creates a new statement in the outer loop that can be scheduled independently.
This approach can be seen as a reg2mem conversion, the inverse of mem2reg performed in the frontend. It only applies to a subset of values, and may be undone after polyhedral scheduling has completed. Furthermore, to decrease the number of dependencies and memory footprint, Polygeist performs a simple value analysis and avoids creating stack-allocated buffers if the same value is already available in another memory location and can be read from there.

a) 简单语句形成：观察到符合多面体模型的C语句主要是变量赋值，我们可以推导出一种机制，从MLIR操作链中识别语句。存储到内存中的操作是该语句的最后一个操作。该操作的逆向切片，即传递计算其操作数的操作，属于该语句。切片扩展在生成被分类为仿射维度或符号的值的操作处停止，这些值可以直接用于仿射表达式。此类值包括循环归纳变量或循环不变常量。一些操作可能出现在多个语句中，如果该值被多次使用。然而，我们需要操作与语句之间的映射是双向的，以便在调度器重新结构化程序后发出MLIR，而不考虑SSA值的可见性规则。如果一个多次使用的操作没有副作用，Polygeist会简单地复制它。对于重复操作的复制不合法的情况，Polygeist将其结果存储在堆栈分配的内存引用中，并将所有后续使用替换为内存加载。图6展示了在操作%20中使用的值%0的转换。这创建了一个新的语句。

b) 跨区域依赖关系：在某些情况下，一个语句可能由不同（嵌套）循环中的MLIR操作组成，例如，从内存加载到SSA寄存器的操作发生在外部循环中，而它在内部循环中被使用。该语句在循环层次结构中的位置并不明确。更重要的是，无法将其传达给多面体调度器。Polygeist通过在定义区域中将值存储在堆栈分配的内存引用中，并在使用区域中将其加载回去，从而解决了这一问题。图6展示了在操作%10中使用的值%0的这种转换。与基本情况类似，这在外部循环中创建了一个新的语句，可以独立调度。该方法可以视为reg2mem转换，这是在前端执行的mem2reg的反向操作。它仅适用于一部分值，并且在多面体调度完成后可以撤销。此外，为了减少依赖关系和内存占用，Polygeist执行简单的值分析，并在同一值在另一个内存位置可用并能够从那里读取时，避免创建堆栈分配的缓冲区。

## # C. Connecting MLIR to Polyhedral Tools Part-3

c) SCoP Formation: To define a SCoP, we outline individual statements into functions so that they can be represented as opaque calls with known memory footprints, similarly to Pencil [22]. This process also makes the inter-statement SSA dependencies clear. These dependencies exist between calls that use the same SSA value, but there are no values defined by these calls. We lift all local stack allocations and place them at the entry block of the surrounding function in order to keep them visible after loop restructuring. Figure 7 demonstrates the resulting IR.
The remaining components of the polyhedral representation are derived as follows: the domain of the statement is defined to be the iteration space of its enclosing loops, constrained by their respective lower and upper bounds, and intersected with any "if" conditions. This process leverages the fact that MLIR expresses bounds and conditions directly as affine constructs. The access relations for each statement are obtained  as unions of affine maps of the affine.load (read) and affine.store (must-write) operations, with RHS of the relation annotated by an "array" that corresponds to the SSA value of the accessed memref. Initial schedules are assigned using the (2d + 1) formalism, with odd dimensions representing the lexical order of loops in the input program and even dimensions being equal to loop induction variables. Affine constructs in OpenScop are represented as lists of linear equality (= 0) or inequality (≥ 0) coefficients, which matches exactly the internal representation in MLIR, making the conversion straightforward.
d) Code Generation Back to MLIR: The Pluto scheduler produces new schedules in OpenScop as a result. Generating loop structure back from affine schedules is a solved, albeit daunting, problem [13], [14]. Polygeist relies on CLooG [13] to generate an initial loop-level AST, which it then converts to Affine dialect loops and conditionals. There is no need to simplify affine expressions at code generation since MLIR accepts them directly and can simplify them at a later stage. Statements are introduced as function calls with rewritten operands and then inlined.

c) SCoP 形成：为了定义一个 SCoP，我们将单独的语句划分为函数，这样它们就可以被表示为具有已知内存占用的透明调用，类似于 Pencil [22]。这个过程也使得语句间的 SSA 依赖关系变得清晰。这些依赖关系存在于使用相同 SSA 值的调用之间，但这些调用没有定义任何值。我们提升所有局部栈分配，并将它们放置在周围函数的入口块中，以保持它们在循环重构后仍然可见。图 7 展示了得到的 IR。

多面体表示的其余组件如下派生：语句的域被定义为其所包含循环的迭代空间，由各自的下界和上界约束，并与任何“if”条件相交。这个过程利用了 MLIR 直接将边界和条件表示为仿射构造的特性。每个语句的访问关系作为 affine.load（读取）和 affine.store（必须写入）操作的仿射映射的并集获得，关系的右侧被注释为与被访问的 memref 对应的“数组”，这对应于 SSA 值。初始调度是使用 (2d + 1) 形式分配的，奇数维度表示输入程序中循环的词法顺序，而偶数维度等于循环引导变量。OpenScop 中的仿射构造表示为线性等式 (= 0) 或不等式 (≥ 0) 系数的列表，这正好与 MLIR 中的内部表示相匹配，使转换变得简单明了。

d) 代码生成回 MLIR：Pluto 调度器因此生成了 OpenScop 中的新调度。从仿射调度生成循环结构是一个已解决但相当复杂的问题 [13]，[14]。Polygeist 依赖于 CLooG [13] 来生成初始的循环级 AST，然后将其转换为仿射方言的循环和条件。在代码生成时无需简化仿射表达式，因为 MLIR 可以直接接受它们，并能够在后续阶段进行简化。语句作为具有重写操作数的函数调用引入，然后进行内联。

## # D. Controlling Statement Granularity

Recall that Polygeist reconstructs "statements" from sequences of primitive operations (Section III-C). We initially designed an approach that recovers the statement structure similar to that in the C input, but this is not a requirement. Instead, statements can be formed from any subsets of MLIR operations as long as they can be organized into loops and sorted topologically (i.e., there are no use-def cycles between statements). To expose the dependencies between such statements to the affine scheduler, we reuse the idea of going through scratchpad memory: each statement writes the values required by other statements to dedicated memory locations, and the following statements read from those. The scratchpads are subject to partial array expansion [23] to minimize their effect on the affine scheduler as single-element scratchpad arrays create artificial scalar dependencies. This change in statement granularity gives the affine scheduler unprecedented flexibility allowing it to chose different schedules for different parts of the same C statement. Consider, for example, the statement S in Figure 8(top) surrounded by three loops iterating over i, j and k. Such contraction patterns are common in computational programs (this particular example can be found in the correlation benchmark with B≡C, see Section V-E). The loop order that best exploits the locality is (k, i, j), which results in temporal locality for reads from B (the value is reused in all iterations of the now-innermost j loop) and in spatial locality for reads from C (consecutive values are read by consecutive iterations, increasing the likelihood of L1 cache hits). Yet, Pluto never proposes such an order because of a reduction dependency along the k dimension due to repeated read/write access to A[i][j] as Pluto tends to pick loops with fewer dependencies as outermost. While the dependency itself is inevitable, it can be moved into a separate statement T in Figure 8(bottom left). This approach provides scheduler with more freedom of choice for the first statement at a lesser memory cost than expanding the entire A array. It also factors out the reduction into a "canonical" statement that is easier to process for the downstream passes, e.g., vectorization.
Implementing this transformation at the C level would require manipulating C AST and reasoning about C (or even C++) semantics. This is typically out of reach for source-tosource polyhedral optimizers such as Pluto that treat statements as black boxes. While it is possible to implement this transformation at the LLVM IR level, e.g., in Polly, where statements are also reconstructed and injection of temporary allocations is easy, the heuristic driving the transformation is based on the loop structure and multi-dimensional access patterns which are difficult to recover at such a low level [9].
The space of potential splittings is huge-each MLIR operation can potentially become a statement. Therefore, we devise a heuristic to address the contraction cases similar to Figure 8. Reduction statement splitting applies to statements:
• surrounded by at least 3 loops;
• with LHS =RHS, and using all loops but the innermost;
• with two or more different access patterns on the RHS. This covers statements that could have locality improved by a different loop order and with low risk of undesired fission. This heuristic merely serves as an illustration of the kind of new transformations Polygeist can enable.

D. 控制语句粒度

回想一下，Polygeist 是通过原始操作的序列重建“语句”的（见第三节C）。我们最初设计了一种恢复与 C 输入中相似的语句结构的方法，但这并不是必需的。实际上，只要可以将语句组织成循环并进行拓扑排序（即语句之间没有使用-定义循环），语句可以由任何子集的 MLIR 操作形成。为了向仿射调度器暴露这些语句之间的依赖关系，我们重用了通过临时存储器的思想：每个语句将其他语句所需的值写入专用内存位置，随后语句从这些位置读取。临时存储器受到部分数组扩展的限制[23]，以最小化它们对仿射调度器的影响，因为单元素的临时存储器数组会产生人为的标量依赖关系。这种语句粒度的变化为仿射调度器提供了前所未有的灵活性，使其能够为同一 C 语句的不同部分选择不同的调度。以图8（顶部）中的语句 S 为例，该语句被三个针对 i、j 和 k 的循环包围。这种收缩模式在计算程序中很常见（此特定示例可以在相关基准中找到，B≡C，见第五节E）。最佳利用局部性的循环顺序是 (k, i, j)，这对从 B 的读取产生时效局部性（该值在现在最内层的 j 循环的所有迭代中被重用）并对从 C 的读取产生空间局部性（连续的值被连续的迭代读取，从而增加了 L1 缓存命中率的可能性）。然而，Pluto 从未提出这样的顺序，因为在 k 维度上由于对 A[i][j] 的重复读/写访问而存在减少依赖关系，Pluto 趋向于选择依赖关系较少的循环作为最外层。虽然这种依赖关系本身是不可避免的，但它可以转移到图8（底部左侧）的单独语句 T 中。这种方法为调度器提供了更多的选择自由度，内存成本相对较低，而不是扩展整个 A 数组。它还将减少问题分解为一个“规范”语句，使得下游处理（例如向量化）更容易进行。

在 C 级别实现这一转换将需要操作 C 抽象语法树（AST）并推理 C（甚至 C++）的语义。这通常超出了诸如 Pluto 这样的源到源多面体优化器的能力，因为这些优化器将语句视为黑箱。虽然在 LLVM IR 级别实现这一转换也是可能的，例如在 Polly 中，在那里语句也被重建且暂时分配的注入较为容易，但驱动转换的启发式是基于循环结构和多维访问模式的，而在如此低的级别恢复这些信息是困难的[9]。

潜在分割的空间是巨大的——每个 MLIR 操作都有可能成为一个语句。因此，我们设计了一种启发式方法来处理类似于图8的收缩情况。减少语句分割适用于以下语句：
• 至少被 3 个循环包围；
• 左侧 = 右侧，并且使用所有循环，除了最内层的；
• 右侧具有两个或更多不同的访问模式。这涵盖了那些可能通过不同循环顺序改善局部性的语句，并且风险较低，不会产生不希望的分裂。这个启发式方法仅作为 Polygeist 能够启用的新转换类型的示例。

## # E. Post-Transformations and Backend

Polygeist allows one to operate on both quasi-syntactic and SSA level, enabling analyses and optimizations that are extremely difficult, if not impossible, to perform at either level in isolation. In addition to statement splitting, we propose two techniques that demonstrate the potential of Polygeist.
a) Transforming Loops with Carried Values (Reductions): Polygeist leverages MLIR's first-class support for loopcarried values to detect, express and transform reduction-like loops. This support does not require source code annotations, unlike source-level tools [24] that use annotations to enable detection, nor complex modifications for parallel code emission, unlike Polly [25], which suffers from LLVM missing first-class parallel constructs. We do not modify the polyhedral scheduler either, relying on post-processing for reduction parallelization, including outermost parallel reduction loops.
The overall approach follows the definition proposed in [26] with adaptations to MLIR's region-based IR, and is illustrated in Figure 9. Polygeist identifies memory locations modified on each iteration, i.e. load/store pairs with loop-invariant subscripts and no interleaving aliasing stores, by scanning the single-block body of the loop. These are transformed into loop-carried values or secondary induction variables, with the load/store pair lifted out of the loop and repurposed for reading the initial and storing the final value. Loop-carried values may be updated by a chain of side effect-free operations in the loop body. If this chain is known to be associative and commutative, the loop is a reduction. Loop-carried values are detected even in absence of reduction-compatible operations. Loops with such values contribute to mem2reg, decreasing memory footprint, but are not subject to parallelization. b) Late Parallelization: Rather than relying on the dependence distance information obtained by the affine scheduler, Polygeist performs a separate polyhedral analysis to detect loop parallelism in the generated code. The analysis itself is a classical polyhedral dependence analysis [27], [28] implemented on top of MLIR region structure. Performing it after SSA-based optimizations, in particular mem2reg and reduction detection, allows parallelizing more loops. In particular, reduction loops and loops with variables whose value is only relevant within a single iteration similar to live-range reordering [29] but without expensive additional polyhedral analyses (live-range of an SSA value defined in a loop never extends beyond the loop).

E. 后变换及后端

Polygeist 使得在 quasi-syntactic 和 SSA 层面上操作成为可能，从而实现了在任何一个层面上单独进行分析和优化时极其困难（甚至不可能）的任务。除了语句拆分，我们提出了两种展示 Polygeist 潜力的技术。

a) 转换带有延续值的循环（归约）：Polygeist 利用 MLIR 对循环携带值的一级支持，来检测、表达和转换类似归约的循环。这种支持不需要源代码注释，而与使用注释以启用检测的源级工具 [24] 不同，也不需要复杂的修改来进行并行代码的生成，与 Polly [25] 不同，后者受到 LLVM 缺失一级并行构造的限制。我们也未对多面体调度器进行修改，而是依赖后处理来进行归约并行化，包括最外层的并行归约循环。

总体方法遵循了 [26] 中提出的定义，针对 MLIR 的基于区域的中间表示进行了调整，如图 9 所示。Polygeist 通过扫描循环的单块主体，识别每次迭代中被修改的内存位置，即具有循环不变下标且没有交错别名存储的加载/存储对。这些被转换为循环携带值或二次归纳变量，加载/存储对被提升到循环外并重新用于读取初始值和存储最终值。在循环主体中，循环携带值可以通过一系列无副作用操作进行更新。如果这个链被认为是结合的和交换的，那么该循环即为归约。即使在没有合适的归约操作的情况下，循环携带值也能被检测到。具有此类值的循环将对 mem2reg 产生贡献，降低内存占用，但不适合进行并行化。

b) 延迟并行化：Polygeist 并不依赖于通过仿射调度器获取的依赖距离信息，而是对生成的代码进行单独的多面体分析以检测循环并行性。该分析本身是一种经典的多面体依赖分析 [27]、[28]，在 MLIR 区域结构之上实现。在进行 SSA 基于优化之后执行这一步，特别是 mem2reg 和归约检测，能够并行化更多的循环。特别是，归约循环和那些变量值仅在单次迭代内相关的循环（类似于活跃范围重排序 [29]，但没有昂贵的额外多面体分析）都得到处理（在循环中定义的 SSA 值的活跃范围从不超出循环）。

## # IV. EVALUATION

Our evaluation has two goals. 1) We want to demonstrate that the code produced by Polygeist without additional op-  timization does not have any inexplicable performance differences than a state-of-the-art compiler like Clang. 2) We explore how Polygeist's internal representation can support a mix of affine and SSA-based transformation in the same compilation flow, and evaluate the potential benefits compared to existing source and compiler-based polyhedral tools.

我们的评估有两个目标。1）我们希望证明，Polygeist 生成的代码在没有额外优化的情况下，其性能与像 Clang 这样的现代编译器没有任何不可解释的差异。2）我们探讨 Polygeist 的内部表示如何在同一编译流程中支持结合使用仿射变换和基于 SSA 的变换，并评估与现有的源代码和编译器基础上的多面体工具相比的潜在优势。

## # A. Experimental Setup

We ran our experiments on an AWS c5.metal instance with hyper-threading and Turbo Boost disabled. The system is Ubuntu 20.04 running on a dual-socket Intel Xeon Platinum 8275CL CPU at 3.0 GHz with 24 cores each, with 0.75, 35, 35.75 MB L1, L2, L3 cache per socket, respectively, and 256 GB RAM. We ran all 30 benchmarks from Poly-Bench [19], using the "EXTRALARGE" dataset. Pluto is unable to extract SCoP from the adi benchmark. We ran a total of 5 trials for each benchmark, taking the execution time reported by PolyBench; the median result is taken unless stated otherwise. Every measurement or result reported in the following sections refers to double-precision data. All experiments were run on cores 1-8, which ensured that all threads were on the same socket and did not potentially conflict with processes scheduled on core 0.
In all cases, we use two-stage compilation: (i) using clang at -O3 excluding unrolling and vectorization; or Polygeist to emit LLVM IR from C; (ii) using clang at -O3 to emit the final binary. As several optimizations are not idempotent, a second round of optimization can potentially significantly boost (and rarely, hinder) performance. This is why we chose to only perform vectorization and unrolling at the last optimization stage. Since Polygeist applies some optimizations at the MLIR level (e.g., mem2reg), we compare against the twostage compilation pipeline as a more fair baseline (CLANG). We also evaluate a single-stage compilation to assess the effect of the two-stage flow (CLANGSING).

我们在一台 AWS c5.metal 实例上进行实验，该实例禁用了超线程和 Turbo Boost。系统为 Ubuntu 20.04，运行于双插槽 Intel Xeon Platinum 8275CL CPU，主频为 3.0 GHz，每个插槽有 24 个核心，L1、L2 和 L3 缓存分别为 0.75 MB、35 MB 和 35.75 MB，并配备 256 GB RAM。我们使用“EXTRALARGE”数据集运行 Poly-Bench [19] 中的所有 30 个基准测试。Pluto 无法从 adi 基准测试中提取 SCoP。我们对每个基准测试进行了总共 5 次试验，采用 PolyBench 报告的执行时间；除非另有说明，结果以中位数的形式给出。以下各节中报告的每个测量或结果均与双精度数据相关。所有实验均在核心 1-8 上运行，以确保所有线程位于同一插槽，并且不会与调度在核心 0 上的进程发生潜在冲突。

在所有情况下，我们采用两阶段编译：(i) 使用 clang 以 -O3 级别编译，不包括展开和向量化；或使用 Polygeist 从 C 发出 LLVM IR；(ii) 使用 clang 以 -O3 级别发出最终二进制文件。由于某些优化并非幂等的，因此第二轮优化可能会显著提升（而很少情况会抑制）性能。这就是我们选择仅在最后优化阶段进行向量化和展开的原因。由于 Polygeist 在 MLIR 级别应用了一些优化（例如，mem2reg），我们将其与两阶段编译流程（CLANG）进行比较，作为更公平的基线。我们还评估了单阶段编译，以评估两阶段流程的效果（CLANGSING）。

## # B. Baseline Performance

Polygeist must generate code with runtime as close as possible to that of existing compilation flows to establish a solid baseline. In other words, Polygeist should not introduce overhead nor speedup unless explicitly instructed otherwise, to allow for measuring the effects of additional optimizations. We evaluate this by comparing the runtime of programs produced by Polygeist with those produced by Clang at the same commit (Apr 2021) 2 . Figure 10 summarizes the results with the following flows:
• CLANG: A compilation of the program using Clang, when running two stages of optimization; • CLANGSING: A compilation of the program using Clang, when running one stage of optimization; • MLIR-CLANG: A compilation flow using the Polygeist frontend and preprocessing optimizations within MLIR, but not running polyhedral scheduling nor postprocessing.

B. 基线性能

Polygeist必须生成尽可能接近现有编译流程的运行时代码，以建立一个稳固的基线。换句话说，Polygeist不应该引入开销或加速，除非明确指示，以便能够测量额外优化的效果。我们通过比较Polygeist生成的程序与同一提交版本（2021年4月）下Clang生成的程序的运行时来评估这一点。图10总结了以下几种流程的结果：
• CLANG：使用Clang编译程序，运行两阶段优化的过程；
• CLANGSING：使用Clang编译程序，运行一阶段优化的过程；
• MLIR-CLANG：使用Polygeist前端的编译流程，在MLIR内进行预处理优化，但不执行多面体调度或后处理。

## # C. Compilation Flows

We compare Polygeist with a source-level and an IR-level optimizer (Pluto and Polly) in the following configurations:
• PLUTO: Pluto compiler auto-transformation [11] using polycc 3 with -noparallel and -tile flags; • PLUTOPAR: Same as above but with -parallel flag;
• POLLY: Polly [7] LLVM passes with affine scheduling and tiling, and no pattern-based optimizations [30]; • POLLYPAR: Same as above with auto-parallelization;
• POLYGEIST: Our flow with Pluto and extra transforms; • POLYGEISTPAR: Same as above but with -parallel Pluto schedule, Polygeist parallelization and reductions. Running between source and LLVM IR levels, we expect Polygeist to benefit from both worlds, thus getting code that is on par or better than competitors. When using Pluto, both standalone and within Polygeist, we disable the emission of vectorization hints and loop unrolling to make sure both transformations are fully controlled by the LLVM optimizer, which also runs in Polly flows. We run Polly in the latest stage of Clang compilation, using -mllvm -polly and additional flags to enable affine scheduling, tiling and parallelization as required. Polly is taken at the same LLVM commit as Clang. We disable pattern-based optimizations [30] that are not available elsewhere. Figures 11 and12 summarize the results for sequential and parallel flows, respectively.

我们将Polygeist与源代码级和中间表示级优化器（Pluto和Polly）进行比较，配置如下：
- PLUTO：使用-polycc 3及-noparallel和-tile标志的Pluto编译器自动转换；
- PLUTOPAR：与上述相同，但使用-parallel标志；
- POLLY：Polly [7]的LLVM通道，具有仿射调度和切片，没有基于模式的优化 [30]；
- POLLYPAR：与上述相同，但进行了自动并行化；
- POLYGEIST：我们的工作流，结合了Pluto和额外的变换；
- POLYGEISTPAR：与上述相同，但使用-parallel的Pluto调度，Polygeist并行化和归约。

在源代码和LLVM IR级别之间运行时，我们期望Polygeist能够从这两者的优点中受益，从而生成的代码能够与竞争对手相媲美或更优。在使用Pluto时，无论是独立地还是在Polygeist中，我们都会禁用向量化提示和循环展开的发出，以确保这两种变换都完全由LLVM优化器控制，该优化器也在Polly流程中运行。我们在Clang编译的最新阶段运行Polly，使用-mllvm -polly及其他标志以启用仿射调度、切片和并行化。Polly与Clang使用相同的LLVM提交版本。我们禁用在其他地方不可用的基于模式的优化 [30]。图11和图12分别总结了顺序和并行流程的结果。

## # A. Benchmarking

The transformation of reduction loops, in particular parallelization, may result in a different order of partial result accumulation. This is not allowed under IEEE 754 semantics, but is supported by compilers with -ffast-math option.
We found that Polybench allocation function hinders Clang/LLVM alias analysis, negatively affecting performance 2 LLVM commit 20d5c42e0ef5d252b434bcb610b04f1cb79fe771 3 Pluto commit dae26e77b94b2624a540c08ec7128f20cd7b7985 in, e.g., adi. Therefore, we modified all benchmarks to use malloc that is known to produce non-aliasing pointers.

降低循环的转换，特别是并行化，可能会导致部分结果累积的顺序发生变化。这在 IEEE 754 语义下是不允许的，但使用 -ffast-math 选项的编译器支持这种情况。我们发现，Polybench 的分配函数会妨碍 Clang/LLVM 的别名分析，从而对性能产生负面影响 2 LLVM 提交 20d5c42e0ef5d252b434bcb610b04f1cb79fe771 3 Pluto 提交 dae26e77b94b2624a540c08ec7128f20cd7b7985，举例来说，在 adi 中。因此，我们修改了所有基准测试，使用已知会产生非别名指针的 malloc。

## # B. Baseline Comparison

We did not observe a significant difference between the runtimes of CLANG and CLANGSING configurations, with a geometric mean of 0.43% symmetric difference 4 across benchmarks. Therefore, we only consider CLANG as baseline throughout the remainder of this paper. We did not observe a significant difference between the runtimes of CLANG and MLIR-CLANG configurations either, with a geometric mean of 0.24% symmetric difference.
We found a variation in runtimes of short-running benchmarks, in particular jacobi-1d. This can be attributed to the interaction with the data initialization and benchmarking code, and with other OS processes. Excluding the benchmarks running in under 0.05s (jacobi-1d, gesummv, atax, bicg) from the analysis, we obtain 0.32% and 0.17% geomean symmetric differences respectively for the two comparisons above. These results suggest that our flow has no unexplained (dis)advantages over the baseline.

我们没有观察到CLANG和CLANGSING配置之间的运行时间存在显著差异，在基准测试中，它们的几何平均对称差为0.43%。因此，在本文的其余部分中，我们仅考虑CLANG作为基线。同样，我们也没有观察到CLANG和MLIR-CLANG配置之间的运行时间存在显著差异，其几何平均对称差为0.24%。

我们发现短运行基准的运行时间存在变动，特别是在jacobi-1d上。这可以归因于数据初始化和基准测试代码之间的相互作用，以及其他操作系统进程的影响。从分析中排除运行时间少于0.05秒的基准（如jacobi-1d、gesummv、atax、bicg），我们分别获得上述两个比较的几何平均对称差为0.32%和0.17%。这些结果表明，我们的流程在基线之上没有未解释的（不）优势。

## # C. Performance Differences in Sequential Code

Overall, Polygeist leads to larger speedups, with 2.53× geometric mean, than both Pluto (2.34×) and Polly (1.41×), although improvements are not systematic. Some difference between Polygeist and Polly is due to the employed polyhedral schedulers, e.g., in lu and mvt. Polygeist produces code faster tha both Pluto and Polly in 2mm, 3mm and others thanks to statement splitting, see Section V-E.
Given identical statements and schedules, codegen-level optimization accounts for other performance difference. seidel-2d is the clearest example: Pluto executes 2.7•10 11 more integer instructions than Polygeist. Assuming these to be index/address computations, a mix of add (throughput 1/2 or 1/4) and imul/shl (thoughput 1), we can expect a ≈ 59s difference at 3GHz, consistent with experimental observations. Polygeist optimizes away a part of those in its post-optimization phase and emits homogeneous address computation from memref with proper machine size type, enabling more aggressive bound analysis and simplification in the downstream compiler. Conversely, jacobi-2d has poorer performance because Polygeist gives up on simplifying CLooG code, with up to 75 statement copies in 40 branches, for compiler performance reasons, as opposed to Clang that takes up to 5s to process it but results in better vectorization. Further work is necessary to address this issue by emitting vector instructions directly from Polygeist.

整体而言，Polygeist 相较于 Pluto（2.34×）和 Polly（1.41×）提供了更大的加速，几何平均值为 2.53×，尽管改进并不是系统性的。Polygeist 和 Polly 之间的一些差异是由于所采用的多面体调度器，例如在 lu 和 mvt 中。得益于语句分割，Polygeist 在 2mm、3mm 及其他某些情况下生成的代码速度超过了 Pluto 和 Polly，详见第 V-E 节。

在给定相同语句和调度的情况下，代码生成层面的优化是造成其他性能差异的原因。seidel-2d 是最明显的例子：Pluto 执行的整数指令比 Polygeist 多出 2.7•10^11 次。假设这些指令都是索引/地址计算，混合了 add（吞吐量为 1/2 或 1/4）和 imul/shl（吞吐量为 1），我们可以预计在 3GHz 下会有 ≈ 59 秒的差异，这与实验观察结果一致。Polygeist 在其后优化阶段优化掉了一部分指令，并从 memref 发出适当机器大小类型的均匀地址计算，使下游编译器能够进行更激进的界限分析和简化。相反，由于 Polygeist 为了编译器性能原因放弃了对 CLooG 代码的简化，jacobi-2d 的性能较差，在 40 个分支中有多达 75 个语句副本，而 Clang 处理这些代码需耗时多达 5 秒，但其结果却实现了更好的向量化。需要进一步的工作来解决这个问题，通过直接从 Polygeist 发出向量指令。

## # D. Performance Differences In Parallel Code

Similarly to sequential code, some performance differences are due to different schedulers. For example, in cholesky and lu, both Pluto and Polygeist outperform Polly, and the remaining gap can be attributed to codegen-level differences. Conversely, in gemver and mvt Polly has a benefit over both Fig. 10. Mean and 95% confidence intervals (log scale) of program run time across 5 runs of Polybench in CLANG, CLANGSING and MLIR-CLANG configurations, lower is better. The run times of code produced by Polygeist without optimization is comparable to that of Clang. No significant variation is observed between single and double optimization. Short-running jacobi-1d shows high intra-group variation. Fig. 11. Median speedup over CLANG for sequential configurations (log scale), higher is better. Polygeist outperforms (2.53× geomean speedup) both Pluto (2.34×) and Polly (1.41×) on average. Pluto can't process adi, which is therefore excluded from summary statistics. Fig. 12. Median speedup over CLANG for parallel configurations (log scale), higher is better. Polygeist outperforms (9.47× geomean speedup) both Pluto (7.54×) and Polly (3.26×) on average. Pluto can't process adi, which is therefore excluded from summary statistics.
Pluto and Polygeist. On ludcmp and syr(2)k, SSA-level optimizations let Polygeist produce code which is faster than Pluto and at least as fast as Polly. These results demonstrate that Polygeist indeed leverages the benefits of both the affine and SSA-based optimizations.
Polygeist is the only flow that obtains speedup on deriche (6.9×) and symm (7.7×). Examining the output code, we observe that only Polygeist manages to parallelize these two benchmarks. Considering the input code in Figure 13, one can observe that the i loop reuses the ym1 variable, which is in-terpreted as parallelism-preventing loop-carried dependency by polyhedral schedulers. Polygeist performs its own parallelism analysis after promoting ym1 to an SSA register (carried by the j loop) whose use-def range does not prevent parallelization.
Similarly, the Polygeist parallelizer identifies two benchmarks with parallel reduction loops that are not contained in other parallel loops: gramschmidt and durbin. gramschmidt benefits from a 56× speedup with Polygeist, compared to 34× with Polly and 54× with Pluto. durbin sees a 6× slowdown since the new parallel loop has relatively for (i=0; i<_PB_W; i++){ ym1 = SCALAR_VAL(0.0); // ... for (j=0; j<_PB_H; j++){ ym1 = y1[i][j]; / * ... * / } } %z = constant 0.0 : f64 affine.parallel %i = ... { affine.for %j = ... iter_args(%ym1=%z)->f64 { %0=affine.load %y1[%i,%j] // ... affine.yield %0 }} Fig. 13. Excerpt from the deriche benchmark. The outer loop reuses ym1 which makes it appear non-parallel to affine schedulers (left). Polygeist detects parallelism thanks to its mem2reg optimization, reduction-like loop-carried %ym1 value detection and late parallelization (right). few iterations and is nested inside a sequential loop, leading to synchronization costs that outweigh the parallelism benefit. Section V-F explores the durbin benchmark in more detail. Polybench is a collection of codes (mostly) known to be parallel and, as such, has little need for reduction parallelization on CPU where one degree of parallelism is sufficient. When targeting inherently target architectures as GPUs, however, exploiting reduction parallelism could be vital for achieving peak performance [31], [24].

同样，对于并行代码，某些性能差异是由于不同的调度器。例如，在 cholesky 和 lu 中，Pluto 和 Polygeist 的性能都优于 Polly，剩余的差距可以归因于代码生成级别的差异。相反，在 gemver 和 mvt 中，Polly 相较于两者皆有优势。

图 10 显示了在 CLANG、CLANGSING 和 MLIR-CLANG 配置下，Polybench 的五次运行中程序运行时间的均值和 95% 置信区间（对数尺度），值越低越好。由 Polygeist 产生的未优化代码的运行时间与 Clang 的运行时间相当。不同程度的单重和双重优化之间没有观察到显著差异。运行时间较短的 jacobi-1d 显示出高的组内变异性。

图 11 显示了在顺序配置下相对于 CLANG 的中位数加速比（对数尺度），值越高越好。Polygeist 在平均情况下的加速比（几何均值加速比为 2.53×）优于 Pluto（2.34×）和 Polly（1.41×）。Pluto 无法处理 adi，因此该案例被排除在总结统计之外。

图 12 显示了在并行配置下相对于 CLANG 的中位数加速比（对数尺度），值越高越好。Polygeist 在平均情况下的加速比（几何均值加速比为 9.47×）优于 Pluto（7.54×）和 Polly（3.26×）。Pluto 无法处理 adi，因此该案例也被排除在总结统计之外。

在 ludcmp 和 syr(2)k 中，SSA 级别的优化使得 Polygeist 生成的代码比 Pluto 更快，并且至少与 Polly 同样快。这些结果表明，Polygeist 确实利用了仿射优化和 SSA 基础优化的优点。

Polygeist 是唯一在 deriche（加速比 6.9×）和 symm（加速比 7.7×）上实现加速的编译流程。通过检查输出代码，我们观察到只有 Polygeist 成功地对这两个基准进行了并行化。考虑到图 13 中的输入代码，可以观察到 i 循环重用了 ym1 变量，这被仿射调度器解释为妨碍并行化的循环携带依赖。Polygeist 在将 ym1 提升为一个由 j 循环携带的 SSA 寄存器后，进行自己的并行性分析，其使用定义范围不妨碍并行化。

类似地，Polygeist 的并行化器识别出两个具有并行归约循环的基准，这些循环不包含在其他并行循环中：gramschmidt 和 durbin。相比于 Polly 的 34× 和 Pluto 的 54×，gramschmidt 在 Polygeist 的帮助下获得了 56× 的加速。durbin 则出现了 6× 的减速，因为新的并行循环迭代次数相对较少，并且嵌套在一个顺序循环内，导致同步成本超过了并行化带来的好处。第 V-F 节将更详细地探讨 durbin 基准。

Polybench 是一组（大多数已知）具有并行特性的代码，因此在 CPU 上进行归约并行化的需求不大，因为只有一度的并行性就已足够。然而，当针对固有的目标架构如 GPU 时，利用归约并行性可能对于实现峰值性能至关重要 [31], [24]。

## # E. Case Study: Statement Splitting

We identified 5 benchmarks where the statement splitting heuristic applied: 2mm, 3mm, correlation, covariance and trmm. To assess the effect of the transformation, we executed these benchmarks with statement splitting disabled, suffixed with -nosplit in Figure 14. In sequential versions, 2mm is 4.1% slower (3.13s vs 3.26s), but the other benchmarks see speedups of 25%, 50%, 51% and 27%, respectively. For parallel versions, the speedups are of 36%, 20%, 44%, 40% and -9% respectively.
Examination of polyhedral scheduler outputs demonstrates that it indeed produced the desired schedules. For example, in the correlation benchmark which had the statement
A[i][j] += B[k][i] * B[k][j]
Polygeist was able to find the (k, i, j) loop order after splitting. Using hardware performance counters on sequential code we confirm that the overall cache miss ratio has indeed decreased by 75%, 50%, 20%, 27%, and -26%, respectively. However, the memory traffic estimated by the number of bus cycles has increased by 9% for 2mm, and decreased by 18%, 32%, 32%, and 21% for the other benchmarks. This metric strongly correlates with the observed performance difference in the same run (r = 0.99, p = 3 • 10 -11 ). This behavior is likely due to the scheduler producing a different fusion structure, e.g., not fusing outermost loops in 2mm, which also affects locality. Similar results can be observed for parallel code. Further research is necessary to exploit the statement splitting opportunities, created by Polygeist, and interplay with fusion.

E. 案例研究：语句拆分

我们确定了 5 个基准测试，其中应用了语句拆分启发式方法：2mm、3mm、相关性、协方差和 trmm。为了评估变换的效果，我们执行了这些基准测试，并禁用了语句拆分，用 -nosplit 后缀标记，如图 14 所示。在顺序版本中，2mm 的速度慢了 4.1%（3.13 秒对比 3.26 秒），但其他基准测试分别看到了 25%、50%、51% 和 27% 的加速。对于并行版本，速度提升分别为 36%、20%、44%、40% 和 -9%。

对多面体调度器输出的检查表明，它确实生成了预期的调度。例如，在相关性基准中，语句
A[i][j] += B[k][i] * B[k][j]
Polygeist 能够在拆分后找到 (k, i, j) 的循环顺序。通过在顺序代码上使用硬件性能计数器，我们确认整体缓存缺失率确实分别降低了 75%、50%、20%、27% 和 -26%。然而，由总匣周期估算的内存流量在 2mm 中增加了 9%，而其他基准测试则分别减少了 18%、32%、32% 和 21%。该指标与同一运行中观察到的性能差异强相关（r = 0.99，p = 3 • 10 -11）。这种行为可能是由于调度器产生了不同的融合结构，例如，在 2mm 中没有融合最外层的循环，这也影响了局部性。并行代码中也可以观察到类似的结果。进一步的研究有必要利用 Polygeist 创造的语句拆分机会，并与融合进行相互作用。

## # F. Case Study: Reduction Parallelization in durbin

In this benchmark, Polygeist uses its reduction optimization to create a parallel loop that other tools cannot. For the relatively small input run by default, N = 4000 iterations inside another sequential loop with N iterations, the overall  performance decreases. We hypothesize that the cost of creating parallel threads and synchronizing them outweighs the benefit of the additional parallelism and test our hypothesis by increasing N . Considering the results in Figure 15, one observes that Polygeist starts yielding speedups (> 1) for N ≥ 16000 whereas Polly only does so at N ≥ 224000, and to a much lesser extent: 6.62× vs 1.01×. Without reduction parallelization, Polygeist follows the same trajectory as Polly. Pluto fails to parallelize any innermost loop and shows no speedup. This evidences in favor of our hypothesis and highlights the importance of being able to parallelize reductions.

F. 案例研究：Durbin中的归约并行化

在这个基准测试中，Polygeist利用其归约优化创建了一个其他工具无法实现的并行循环。对于默认运行的相对较小的输入，N = 4000的迭代嵌套在另一个具有N次迭代的顺序循环中，整体性能出现下降。我们假设创建并行线程和同步它们的成本超过了额外并行性带来的收益，并通过增加N来验证我们的假设。根据图15中的结果，可以观察到，Polygeist在N ≥ 16000时开始产生加速效果（> 1），而Polly仅在N ≥ 224000时才会如此，并且幅度要小得多：6.62×对比1.01×。在没有归约并行化的情况下，Polygeist的表现与Polly相同。Pluto未能对任何最内层循环进行并行化，并且显示没有加速。这支持了我们的假设，并强调了能够并行化归约的重要性。

## # VI. RELATED WORK

a) MLIR Frontends: Since the adoption of MLIR under the LLVM umbrella, several frontends have been created for generating MLIR from domain-specific languages. Teckyl [2] connects the productivity-oriented Tensor Comprehensions [1] notation to MLIR's Linalg dialect. Flang-the LLVM's Fortran frontend-models Fortran-specific constructs using the FIR dialect [32]. COMET, a domain-specific compiler for chemistry, introduces an MLIR-targeting domain-specific frontend from a tensor-based language [33]. NPComp aims at providing the necessary infrastructure to compile numerical Python and PyTorch programs taking advantage of the MLIR infrastructure [34]. PET-to-MLIR converts a subset of polyhedral C code to MLIR's Affine dialect by parsing pet's internal represen-tation. In addition to currently not handling specific constructs (ifs, symbolic bounds, and external function calls), parsing pet's representation limits the frontend's usability as it cannot interface with non-polyhedral code such as initialization, verification, or printing routines [35]. In contrast, Polygeist generates MLIR from non-polyhedral code (though not necessarily in the Affine dialect). CIRCT is a new project under the LLVM umbrella that aims to apply MLIR development methodology to the electronic design automation industry [17]. Stripe uses MLIR Affine dialect as a substrate for loop transformations in machine learning models, including tiling and vectorization, and accepts a custom DSL as input [36].

VI. 相关工作

a) MLIR前端：自从MLIR被纳入LLVM框架后，已经创建了多个前端用于从特定领域语言生成MLIR。Teckyl [2] 将面向生产力的张量理解 [1] 符号连接到MLIR的Linalg方言。Flang，LLVM的Fortran前端，使用FIR方言建模Fortran特定的构造 [32]。COMET，一个针对化学领域的特定领域编译器，引入了一个从基于张量的语言到MLIR的特定领域前端 [33]。NPComp旨在提供必要的基础设施以便编译数值Python和PyTorch程序，从而利用MLIR基础设施 [34]。PET-to-MLIR通过解析pet的内部表示，将一部分多面体C代码转换为MLIR的Affine方言。除了当前不处理特定构造（如if语句、符号边界和外部函数调用）外，解析pet的表示限制了前端的可用性，因为它无法与非多面体代码（如初始化、验证或打印例程）接口 [35]。相对而言，Polygeist能够从非多面体代码生成MLIR（尽管不一定在Affine方言中）。CIRCT是一个新的项目，旨在将MLIR开发方法应用于电子设计自动化行业，隶属于LLVM框架 [17]。Stripe使用MLIR的Affine方言作为机器学习模型中的循环变换的基础，包括拆分和向量化，并接受自定义DSL作为输入 [36]。

## # b) Compilers Leveraging Multiple Representations:

The SUIF compiler infrastructure pioneered a combined internal representation that supports higher-level transformations, including loop optimization and parallelization [37] and, in particular, reduction parallelization [38]. Polygeist leverages MLIR abstractions unavailable in SUIF: regular and affine for loops, OpenMP reduction constructs, etc. It also benefits from the SSA+regions form, which is only available as external extension in SUIF [39], for IR simplification. PIPS supports loop transformations and inter-procedural optimization when targeting OpenMP [40], [41]. Polygeist differs from both by emitting machine code rather than source code, which allows it to emit parallel runtime and other directives that have no representation in the source language such as C. c) Combining "Classical" and Polyhedral Flows: Few papers have focused on combining "classical", mostly ASTlevel, and polyhedral transformations. PolyAST pioneered the approach by combining an affine scheduler with ASTlevel heuristics for fusion and tiling [42], although similar results were demonstrated with only polyhedral transformations [43]. An analogous approach was experimented in CUDA-CHiLL [44]. Arguably, many automated polyhedral flows perform loop fusion and/or tiling as a separate step that can be assimilated to classical transformations. Pluto [11] uses several "syntactic" postprocessing passes to exploit spatial locality and parallelism in stencils [45]. Several tools have been proposed to drive polyhedral loop transformations with scripts using classical loop transformations such as fusion and permutation as operations, including URUK [46], CHiLL [47] and Clay [48]. Polygeist differs from all of these because it preserves the results of such transformations in its IR along with polyhedral constructs and enables interaction between different levels of abstraction. d) Additional (Post-)Polyhedral Transformations: Support for handling reduction loops was proposed in Polly [25], but the code generation is not implemented. At the syntactic level, reduction support was added to PET via manual annotation with PENCIL directives [24]. R-Stream reportedly uses a variant of statement splitting to affect scheduler's behavior and optimize memory consumption [49]. POLYSIMD uses variable renaming around PPCG polyhedral flow to improve vectorization [50]. Polygeist automates these leveraging both SSA and polyhedral information.

b) 利用多种表示的编译器：SUIF编译器基础设施开创了一种组合内部表示，支持更高级的转换，包括循环优化和并行化 [37]，尤其是归约并行化 [38]。Polygeist利用SUIF中不可用的MLIR抽象：常规和仿射循环，OpenMP归约构造等。它还受益于仅作为外部扩展存在于SUIF中的SSA+区域形式 [39]，可以用于中间表示的简化。PIPS在针对OpenMP时支持循环转换和过程间优化 [40]，[41]。Polygeist与这两者的不同之处在于它发出机器代码而不是源代码，从而能够发出并行运行时和其他在源语言（如C）中没有表示的指令。

c) 结合“经典”和多面体流程：少数论文关注于将“经典”的、大多数是AST级别的转化与多面体转化结合起来。PolyAST通过结合仿射调度器与AST级别的启发式方法进行融合和切片 [42]，开创了这种方法，尽管仅使用多面体转化也展示了类似的结果 [43]。CUDA-CHiLL [44] 也实验了一种类似的方法。可以说，许多自动化的多面体流程将循环融合和/或切片作为一个独立步骤来执行，这可以被认为是一种经典转化。Pluto [11] 使用几个“语法”后处理过程来利用模板中的空间局部性和并行性 [45]。已提出多个工具，以经典循环转化如融合和排列作为操作，驱动多面体循环转化，包括URUK [46]、CHiLL [47] 和Clay [48]。Polygeist与这些工具的不同之处在于它在其IR中保留了此类转化的结果，并且与多面体构造一起，允许不同抽象层次之间的交互。

d) 额外的（后）多面体转化：Polly [25] 提出了处理归约循环的支持，但并未实现代码生成。在语法层面，通过手动注释使用PENCIL指令向PET添加了归约支持 [24]。据报道，R-Stream使用了一种变体的语句拆分以影响调度器的行为并优化内存消耗 [49]。POLYSIMD在PPCG多面体流程周围使用变量重命名以改善向量化 [50]。Polygeist利用SSA和多面体信息自动化这些操作。

## # e) Integration of Polyhedral Optimizers into Compilers:

Polyhedral optimization passes are available in production (GCC [8], LLVM [7], IBM XL [51]) and research (R-Stream [49], ROSE [52]) compilers. In most cases, the polyhedral abstraction must be extracted from a lower-level representation before being transformed and lowered in a dedicated code generation step [13], [14]. This extraction process is not guaranteed and may fail to recover high-level information available at the source level [9]. Furthermore, common compiler optimizations such as LICM are known to interfere with it [10]. Polygeist maintains a sufficient amount of high-level information, in particular loop and n-D array structure, to circumvent these problems by design.
Source-to-source polyhedral compilers such as Pluto [11] and PPCG [5] operate on a C or C++ level. They lack interaction with other compiler optimizations and a global vision of the code, which prevents, e.g., constant propagation and inlining that could improve the results of polyhedral optimization. Being positioned between the AST and LLVM IR levels, Polygeist enables the interaction between higherand lower-level abstractions that is otherwise reduced to compiler pragmas, i.e. mere optimization hints. Furthermore, Polygeist can rely on MLIR's progressive raising [53] to target abstractions higher level than C code with less effort than polyhedral frameworks [54].

### e) 聚合优化器与编译器的集成:

聚合优化传递在生产（GCC [8]，LLVM [7]，IBM XL [51]）和研究（R-Stream [49]，ROSE [52]）编译器中可用。在大多数情况下，必须从低级表示中提取聚合抽象，然后在专门的代码生成步骤中进行转换和降级 [13]，[14]。这个提取过程并不是保证成功的，可能无法恢复源级别上可用的高级信息 [9]。此外，常见的编译器优化，例如LICM，已知会对此产生干扰 [10]。Polygeist 通过设计维护足够的高级信息，尤其是循环和多维数组结构，以规避这些问题。

源到源的聚合编译器，如Pluto [11] 和PPCG [5]，在C或C++级别上运行。它们缺乏与其他编译器优化的交互，以及对代码的全局视角，这阻碍了例如常量传播和内联等可能改善聚合优化结果的过程。Polygeist位于AST和LLVM IR级别之间，使得更高级别和低级别抽象之间的交互成为可能，而这种交互在其他情况下仅限于编译器指示，即仅仅是优化提示。此外，Polygeist可以依靠MLIR的逐步提升 [53]，以比聚合框架 [54] 更少的努力，针对更高于C代码的抽象。

## # VII. DISCUSSION

A. Limitations a) Frontend: While Polygeist could technically accept any valid C or C++ thanks to building off Clang, it has the following limitations. Only structs with values of the same type or are used within specific functions (such as FILE within fprintf) are supported due to the lack of a struct-type in high-level MLIR dialects. All functions that allocate memory must be compiled with Polygeist and not a C++ compiler to ensure that a memref is emitted rather than a pointer. b) Optimizer: The limitations of the optimizer are inherited from those of the tools involved. In particular, the MLIR affine value categorization results in all-or-nothing modeling, degrading any loop to non-affine if it contains even one nonaffine access or a negative step. Running Polygeist's backend on code not generated by Polygeist's frontend, which reverses loops with negative steps, is limited to loops with positive indices. Finally, MLIR does not yet provide extensive support for non-convex sets (typically expressed as unions). Work is ongoing within MLIR to address such issues.
c) Experiments: While our experiments clearly demonstrate the benefits of the techniques implemented in Polygeist-statement splitting and late (reduction) parallelization -non-negligible effects are due to scheduler difference: Pluto in Polygeist and isl in Polly. The version of Polly using Pluto 5 is not compatible with modern LLVM necessary to leverage MLIR. Connecting isl scheduler to Polygeist may have yielded results closer to Polly, but still not comparable more directly because of the interplay between SCoP detection, statement formation and affine scheduling.

VII. 讨论

A. 限制  
a) 前端：尽管Polygeist技术上可以接受任何有效的C或C++代码，因为它是基于Clang构建的，但仍然存在以下限制。由于高层次MLIR方言中缺乏结构类型，仅支持值类型相同的结构体，或在特定函数内（如fprintf中的FILE）使用的结构体。所有分配内存的函数必须使用Polygeist编译，而不是使用C++编译器，这样才能确保生成的是memref而非指针。  
b) 优化器：优化器的限制源自于所涉及工具的限制。特别是，MLIR的仿射值分类导致了全有或全无的建模，如果任何循环中包含一个非仿射访问或负步长，就会将该循环降级为非仿射。对不是由Polygeist前端生成的代码使用Polygeist的后端运行时，将限制在具有正索引的循环上。最后，MLIR目前尚未提供对非凸集合的广泛支持（通常以并集的形式表示）。MLIR内部正在进行相关工作以解决此类问题。  
c) 实验：尽管我们的实验清楚地展示了在Polygeist中实现的技术的好处——语句拆分和后期（归约）并行化——但仍有不容忽视的效果是由于调度器的差异：Polygeist中的Pluto和Polly中的isl。使用Pluto 5的Polly版本与现代LLVM不兼容，而后者又是利用MLIR所必需的。将isl调度器连接到Polygeist可能会使结果更接近Polly，但仍然无法更直接地进行比较，因为SCoP检测、语句形成和仿射调度之间的相互作用。

## # B. Opportunities and Future Work

Connecting MLIR to existing polyhedral flows opens numerous avenues for compiler optimization research, connecting polyhedral and conventional SSA-based compiler transformations. This gives polyhedral schedulers access to important analyses such as aliasing and useful information such as precise data layout and target machine description. Arguably, this information is already leveraged by Polly, but the representational mismatch between LLVM IR and affine loops makes it difficult to exploit them efficiently. MLIR exposes similar information at a sufficiently high level to make it usable in affine transformations.
By mixing abstractions in a single module, MLIR provides finer-grain control over the entire transformation process. An extension of Polygeist can, e.g., ensure loop vectorization by directly emitting vector instructions instead of relying on pragmas, which are often merely a recommendation for the compiler. The flow can also control lower-level mechanisms like prefetching or emit specialized hardware instructions. Conversely, polyhedral analyses can guarantee downstream passes that, e.g., address computation never produces out-ofbounds accesses and other information.
Future work is necessary on controlling statement granularity made possible by Polygeist. Beyond affecting affine schedules, this technique enables easy rematerialization and local transposition buffers, crucial on GPUs [55], as well as software pipelining; all without having to produce C source which is known to be complex [56]. On the other hand, this may have an effect on the compilation time as the number of statements is an important factor in the complexity bound of the dependence analysis and scheduling algorithms.

将MLIR与现有的多面体流程连接起来，为编译器优化研究开辟了许多新途径，连接了多面体和传统的基于SSA的编译器转换。这使得多面体调度器可以访问一些重要的分析信息，如别名分析以及精确的数据布局和目标机器描述等有用信息。可以说，这些信息已经被Polly利用，但LLVM IR和仿射循环之间的表示不匹配使得高效利用这些信息变得困难。MLIR以足够高的抽象层级暴露了类似的信息，使其可用于仿射变换。

通过在单个模块中混合不同的抽象，MLIR对整个转换过程提供了更细粒度的控制。Polygeist的扩展可以确保通过直接发出向量指令来实现循环向量化，而不是依赖于常常只是编译器建议的pragma指令。该流程还可以控制更低层次的机制，如预取或发出专用的硬件指令。反之，多面体分析可以确保下游的处理过程，例如，地址计算永远不会导致越界访问，以及其他信息。

在Polygeist的支持下，控制语句粒度的未来工作是必要的。除了影响仿射调度外，该技术还使得轻松的重新物化和局部转置缓冲成为可能，这在GPU上是至关重要的[55]，以及软件流水线化；所有这些都不需要生成已知复杂的C源代码[56]。另一方面，语句数量可能会对编译时间产生影响，因为语句数量是依赖分析和调度算法复杂性边界的重要因素。

## # C. Alternatives

Instead of allowing polyhedral tools to parse and generate MLIR, one could emit C (or C++) code from MLIR 6 and use C-based polyhedral tools on the C source, but this approach decreases the expressiveness of the flow. Some MLIR constructs, such as parallel reduction loops, can be directly expressed in the polyhedral model, whereas they would require a non-trivial and non-guaranteed raising step in C. Some other constructs, such as prevectorized affine memory operations, cannot be expressed in C at all. Polygeist enables transparent handling of such constructs in MLIR-to-MLIR flows, but we leave the details of such handling for future work.
The Polygeist flow can be similarly connected to other polyhedral formats, in particular isl. We choose OpenScop for this work because it is supported by a wider variety of tools. isl uses schedule trees [57] to represent the initial and transformed program schedule. Schedule trees are sufficiently close to the nested-operation IR model making the conversion straightforward: "for" loops correspond to band nodes (one loop per band dimension), "if" conditionals correspond to filter nodes, function-level constants can be included into the context node. The tree structure remains the same as that of MLIR regions. The inverse conversion can be obtained using isl's AST generation facility [14].

替代方案

与其让多面体工具解析并生成MLIR，不如直接从MLIR发出C（或C++）代码，并在C源代码上使用基于C的多面体工具，但这种方法降低了编译流程的表达能力。一些MLIR构造，例如并行归约循环，可以直接在多面体模型中表达，而在C中却需要一个复杂且不保证可行的提升步骤。另一些构造，例如预向量化的仿射内存操作，则根本无法在C中表达。Polygeist支持在MLIR到MLIR的流程中透明地处理这些构造，但我们将此类处理的细节留待未来研究。

Polygeist流程也可以与其他多面体格式相连接，特别是与isl相连。我们选择OpenScop作为本研究的对象，因为它得到更多工具的支持。isl使用调度树来表示初始和转化后的程序调度。调度树与嵌套操作的IR模型非常接近，使得转换变得简单：`for`循环对应于带节点（每个带维度一个循环），`if`条件对应于过滤节点，函数级常量可以包含在上下文节点中。树结构与MLIR区域保持一致。逆向转换可以通过isl的AST生成工具获得。

## # VIII. CONCLUSION Part-1

We present Polygeist, a compilation workflow for importing existing C or C++ code into MLIR and allows polyhedral tools, such as Pluto, to optimize MLIR programs. This enables MLIR to benefit from decades of research in polyhedral compilation. We demonstrate that the code generated by Polygeist has comparable performance with Clang, enabling unbiased comparisons between transformations built for MLIR and existing polyhedral frameworks. Finally, we demonstrate the optimization opportunities enabled by Polygeist considering two complementary transformations: statement splitting and reduction parallelization. In both cases, Polygeist achieves better performance than state-of-the-art polyhedral compiler and source-to-source optimizer. comments regarding how this may need to be modified to run on a system with hardware or software configuration that is distinct from what we used. As expected, the command description mirrors much of the content of the docker file. While a docker file is certainly more convenient and a good way of getting the compiler set up, similar changes to expectations of how many cores the system has in the evaluation will be required even with Docker.
To compile Polygeist, one must first compile several of its dependencies. We ran our experiments on an AWS c5.metal instance based on Ubuntu 20.04. We've tailored our build instructions to such a system. While many of the instructions are general and independent of machine, or OS, some steps may not be (and we describe what locations they may occur below).
$ sudo apt update $ sudo apt install apt-utils $ sudo apt install tzdata build-essential \ libtool autoconf pkg-config flex bison \ libgmp-dev clang-9 libclang-9-dev texinfo \ cmake ninja-build git texlive-full numactl # Change default compilers to make Pluto happy $ sudo update-alternatives --install \ /usr/bin/llvm-config llvm-config \ /usr/bin/llvm-config-9 100 $ sudo update-alternatives --install \ /usr/bin/FileCheck FileCheck-9 \ /usr/bin/FileCheck 100 $ sudo update-alternatives --install \ /usr/bin/clang clang \ /usr/bin/clang-9 100 $ sudo update-alternatives --install \ /usr/bin/clang++ clang++ \ /usr/bin/clang++-9 100
To begin, let us download a utility repository, which will contain several scripts and other files useful for compilation and benchmarking:
$ cd $ git clone \ https://github.com/wsmoses/Polygeist-Script\ scripts
One can now compile and build Pluto as shown below:

我们提出了Polygeist，一种将现有C或C++代码导入MLIR的编译工作流程，并允许多面体工具（如Pluto）对MLIR程序进行优化。这使得MLIR能够受益于数十年来在多面体编译方面的研究。我们证明了Polygeist生成的代码在性能上与Clang相当，从而实现了针对MLIR构建的变换与现有多面体框架之间的公正比较。最后，我们展示了Polygeist所带来的优化机会，考虑了两种互补的变换：语句拆分和归约并行化。在这两种情况下，Polygeist的性能均优于最先进的多面体编译器和源到源优化器。

关于如何在硬件或软件配置与我们所使用的不同的系统上运行，这可能需要修改的评论。如预期的那样，命令描述与docker文件的内容相似。虽然docker文件无疑更方便，是设置编译器的好方法，但即便使用Docker，对于评估系统的核心数量的期望也需要类似的更改。

要编译Polygeist，首先需要编译其若干依赖项。我们在基于Ubuntu 20.04的AWS c5.metal实例上运行了实验。我们根据该系统调整了构建说明。虽然许多说明是通用的，与机器或操作系统无关，但某些步骤可能并非如此（我们将在下文中描述它们可能出现的位置）。

```
$ sudo apt update 
$ sudo apt install apt-utils 
$ sudo apt install tzdata build-essential \ 
libtool autoconf pkg-config flex bison \ 
libgmp-dev clang-9 libclang-9-dev texinfo \ 
cmake ninja-build git texlive-full numactl 

# 更改默认编译器以使Pluto正常工作
$ sudo update-alternatives --install \ 
/usr/bin/llvm-config llvm-config \ 
/usr/bin/llvm-config-9 100 

$ sudo update-alternatives --install \ 
/usr/bin/FileCheck FileCheck-9 \ 
/usr/bin/FileCheck 100 

$ sudo update-alternatives --install \ 
/usr/bin/clang clang \ 
/usr/bin/clang-9 100 

$ sudo update-alternatives --install \ 
/usr/bin/clang++ clang++ \ 
/usr/bin/clang++-9 100 
```

首先，让我们下载一个实用工具库，其中将包含一些编译和基准测试所需的脚本和其他文件：
```
$ cd 
$ git clone \
https://github.com/wsmoses/Polygeist-Script\ scripts 
```

现在可以按照下面的步骤编译和构建Pluto：

## # VIII. CONCLUSION Part-2

$ cd $ git clone \ https://github.com/bondhugula/pluto $ cd pluto/ $ git checkout e5a039096547e0a3d34686295c $ git submodule init $ git submodule update $ ./autogen.sh $ ./configure $ make -j`nprocǸ ext one can build LLVM, MLIR, and the frontend by performing the following: From here, we need to modify omp.h by copying the version from the scripts repository and replacing the version we just built. 8$ cd $ export OMP_FILE=`find \ $HOME/mlir-clang/build -iname omp.h$ cp $HOME/scripts/omp.h $OMP_FILE
Let us now build the MLIR polyhedral analyses, along with the specific version of LLVM it requires. We shall begin by downloading the requisite code and building its dependencies.
$ cd $ git clone --recursive \ https://github.com/kumasento/polymer -b pact $ cd polymer/ $ cd llvm/ $ mkdir build $ cd build/ $ cmake ../llvm \ -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \ -DLLVM_TARGETS_TO_BUILD="host" \ -DLLVM_ENABLE_ASSERTIONS=ON \ -DCMAKE_BUILD_TYPE=Release \ -DLLVM_INSTALL_UTILS=ON \ -G Ninja $ ninja -j`nproc$ ninja check-mlir
We can now build the MLIR polyhedral analyses and export the corresponding build artifacts.
$ cd ~/polymer $ mkdir build $ cd build $ export BUILD=$PWD/../llvm/build $ cmake .. \ -DCMAKE_BUILD_TYPE=DEBUG \ -DMLIR_DIR=$BUILD/lib/cmake/mlir \ -DLLVM_DIR=$BUILD/lib/cmake/llvm \ -DLLVM_ENABLE_ASSERTIONS=ON \ -DLLVM_EXTERNAL_LIT=$BUILD/bin/llvm-lit \ -G Ninja $ ninja -j`nproc$ export LD_LIBRARY_PATH= \ `pwd`/pluto/lib:$LD_LIBRARY_PATH $ ninja check-polymer Finally, we are ready to begin benchmarking. We begin by running a script that disables turbo boost & hyperthreading and remaining nonessential services on the machine. The script is specific to both the number of cores on the AWS instance (all cores except the non hyperthreaded cores on the first socket were disabled), as well as the image used (all nonessential services still present on the image were disabled) and thus may require modification if intending to be used on a different machine.
$ cd ~/scripts/ $ sudo bash ./hyper.sh

$ cd $ git clone \ https://github.com/bondhugula/pluto $ cd pluto/ $ git checkout e5a039096547e0a3d34686295c $ git submodule init $ git submodule update $ ./autogen.sh $ ./configure $ make -j`nproc`

接下来，可以通过执行以下操作来构建LLVM、MLIR和前端：

从这里开始，我们需要通过复制脚本库中的版本并替换我们刚刚构建的版本来修改omp.h。 

$ cd $ export OMP_FILE=`find \ $HOME/mlir-clang/build -iname omp.h`

$ cp $HOME/scripts/omp.h $OMP_FILE

现在让我们构建MLIR多面体分析，以及它所需的特定版本的LLVM。我们将开始下载所需的代码并构建其依赖项。

$ cd $ git clone --recursive \ https://github.com/kumasento/polymer -b pact

$ cd polymer/ $ cd llvm/ $ mkdir build $ cd build/ 

$ cmake ../llvm \ -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \ -DLLVM_TARGETS_TO_BUILD="host" \ -DLLVM_ENABLE_ASSERTIONS=ON \ -DCMAKE_BUILD_TYPE=Release \ -DLLVM_INSTALL_UTILS=ON \ -G Ninja 

$ ninja -j`nproc`

$ ninja check-mlir

现在我们可以构建MLIR多面体分析并导出相应的构建产物。

$ cd ~/polymer $ mkdir build $ cd build $ export BUILD=$PWD/../llvm/build 

$ cmake .. \ -DCMAKE_BUILD_TYPE=DEBUG \ -DMLIR_DIR=$BUILD/lib/cmake/mlir \ -DLLVM_DIR=$BUILD/lib/cmake/llvm \ -DLLVM_ENABLE_ASSERTIONS=ON \ -DLLVM_EXTERNAL_LIT=$BUILD/bin/llvm-lit \ -G Ninja 

$ ninja -j`nproc` 

$ export LD_LIBRARY_PATH= \ `pwd`/pluto/lib:$LD_LIBRARY_PATH 

$ ninja check-polymer 

最后，我们准备开始基准测试。我们首先运行一个脚本，禁用涡轮提升和超线程，并关闭机器上仍然不必要的服务。该脚本特定于AWS实例上的核心数量（所有核心中除第一个插槽上的非超线程核心外均被禁用），以及使用的镜像（镜像上仍存在的所有不必要服务被禁用），因此如果打算在其他机器上使用，可能需要进行修改。

$ cd ~/scripts/ $ sudo bash ./hyper.sh

## # VIII. CONCLUSION Part-3

We can now run the benchmarking script. The script itself has assumptions about cores and layout (setting taskset -c 1-8 numactl -i all for example). If using a different machine, these settings may need to be tweaked as appropriate.
cd ~/scripts/ $ cd polybench-c-4.2.1-beta/ $ ./run.sh # Output comes through stdout
The output of this script will contain the runtime of each trial, describing what compilation setting was used, as well as which benchmark was run.

我们现在可以运行基准测试脚本。该脚本本身对核心和布局有一些假设（例如，设置 taskset -c 1-8 numactl -i all）。如果使用不同的机器，可能需要根据实际情况调整这些设置。
cd ~/scripts/
$ cd polybench-c-4.2.1-beta/
$ ./run.sh  # 输出通过标准输出生成
该脚本的输出将包含每次试验的运行时间，描述所使用的编译设置以及运行了哪个基准测试。

## # 

The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

本文件中包含的观点和结论是作者的观点，不应被解读为代表美国空军或美国政府的官方政策，无论是明示还是暗示。美国政府被授权为了政府目的复印和分发该文件的印刷本，尽管这里有任何版权说明。

## # APPENDIX

In this artifact appendix, we describe how to build Polygeist and evaluate its performance (as well as baseline compilers) on the Polybench benchmark suite. We provide two mechanisms for artifact evaluation: a Docker container 7 , and a commandby-command description of the installation process, along with

在本附录中，我们描述了如何构建Polygeist并评估其在Polybench基准测试套件上的性能（以及基准编译器）。我们提供了两种艺术品评估机制：一个Docker容器以及逐步命令描述的安装过程，连同

