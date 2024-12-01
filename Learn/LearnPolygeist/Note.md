[William S. Moses](https://wsmoses.com/academic/)


[Polygeist: Affine C in MLIR [MLIR Open Design Meeting 02/11/2021]](https://www.youtube.com/watch?v=GF45kitd3nY)

https://www.youtube.com/@billymoses7764

[getting_started/Use_Polygeist](https://polygeist.llvm.org/getting_started/Use_Polygeist/)

[Retargeting and Respecializing GPU Workloads for Performance Portability](https://ieeexplore.ieee.org/document/10444828)

重新定位和针对性能可移植性重新专业化的 GPU 工作负载

In order to come close to peak performance, accelerators like GPUs require significant architecture-specific tuning that understand the availability of shared memory, parallelism, tensor cores, etc. Unfortunately, the pursuit of higher performance and lower costs have led to a significant diversification of architecture designs, even from the same vendor. This creates the need for performance portability across different GPUs, especially important for programs in a particular programming model with a certain architecture in mind. Even when the program can be seamlessly executed on a different architecture, it may suffer a performance penalty due to it not being sized appropriately to the available hardware resources such as fast memory and registers, let alone not using newer advanced features of the architecture. We propose a new approach to improving performance of (legacy) CUDA programs for modern machines by automatically adjusting the amount of work each parallel thread does, and the amount of memory and register resources it requires. By operating within the MLIR compiler infrastructure, we are able to also target AMD GPUs by performing automatic translation from CUDA and simultaneously adjust the program granularity to fit the size of target GPUs. Combined with autotuning assisted by the platform-specific compiler, our approach demonstrates 27% geomean speedup on the Rodinia benchmark suite over baseline CUDA implementation as well as performance parity between similar NVIDIA and AMD GPUs executing the same CUDA program.

为了接近峰值性能，像 GPU 这样的加速器需要针对特定架构进行显著的调整，这些调整理解共享内存、并行性、张量核心等的可用性。不幸的是，追求更高的性能和更低的成本导致了架构设计的显著多样化，即使是来自同一供应商的产品也是如此。这产生了在不同 GPU 之间实现性能可移植性的需求，这对于特定编程模型和特定架构的程序尤为重要。即使程序可以在不同的架构上无缝执行，它也可能因为未适当调整以适应可用的硬件资源（如快速内存和寄存器）而遭受性能损失，更不用说没有使用架构的新先进特性。我们提出了一种新方法，通过自动调整每个并行线程执行的工作量以及它所需的内存和寄存器资源，来提高（遗留）CUDA 程序在现代机器上的性能。 通过在 MLIR 编译器基础设施中操作，我们能够通过自动从 CUDA 进行翻译来针对 AMD GPU 进行目标定位，同时调整程序粒度以适应目标 GPU 的大小。结合平台特定编译器辅助的自动调整，我们的方法在 Rodinia 基准测试套件上相对于基线 CUDA 实现实现了 27%的几何平均速度提升，以及执行相同 CUDA 程序时类似 NVIDIA 和 AMD GPU 之间的性能对等。

Frontend Performance Differences

- 8% performance boost on Floyd-Warshall occurs if Polygeist generates a single MLIR module for both benchmarking and timing code by default

- MLIR doesn't properly generate LLVM datalayout, preventing vectorization for MLIR-generated code (patched in our lowering)

- Different choice of allocation function can make a 30% impact on some tests (adi)

- LLVM strength-reduction is fragile and sometimes misses reversed loop induction variable (remaining gap in adi)

- 如果 Polygeist 默认为基准测试和计时代码生成单个 MLIR 模块，则 Floyd-Warshall 的性能将提升 8%

- MLIR 无法正确生成 LLVM 数据布局，从而阻止了 MLIR 生成的代码的矢量化（在我们的降低版本中进行了修补）

- 不同的分配函数选择可能会对某些测试 （adi） 产生 30% 的影响

- LLVM 强度降低很脆弱，有时会错过反向环感应变量（ADI 中的剩余间隙）

### Poligeist MLIR Compiler Frontend

Polygeist的核心功能
Polygeist的主要目标是bridging the gap between C/C++ and MLIR。它具有以下核心功能:

C/C++前端:能够解析和分析广泛的C和C++代码。
MLIR生成:将C/C++代码转换为适合多面体变换的MLIR表示。
多面体优化:利用MLIR的多面体优化能力进行高级循环优化。
并行优化:支持自动并行化和并行构造的优化。
GPU后端支持:包括CUDA和ROCm后端,实现GPU加速。
这些功能使Polygeist成为连接传统C/C++代码和现代MLIR编译架构的关键工具。

Polygeist的工作原理
Polygeist的工作流程可以简要概括为以下几个步骤:

解析C/C++代码:使用Clang的前端能力解析输入的C/C++代码。
AST分析:对抽象语法树(AST)进行深入分析,提取程序的结构和语义信息。
MLIR生成:基于AST分析结果,生成对应的MLIR表示。
多面体建模:将MLIR表示转换为多面体模型,为后续优化铺平道路。
优化应用:应用多面体优化、并行优化等高级优化技术。
代码生成:将优化后的MLIR转换回LLVM IR或直接生成目标代码。
通过这一系列步骤,Polygeist能够充分利用MLIR的强大功能,同时保持对原始C/C++代码的兼容性。

Polygeist的优势与应用
Polygeist为C/C++程序带来了诸多优势:

高级优化:通过多面体模型,可以进行更复杂和有效的循环优化。
并行化:自动检测和利用并行机会,提高程序性能。
可移植性:通过MLIR表示,可以更容易地将程序移植到不同的硬件平台。
GPU加速:内置的CUDA和ROCm后端支持,简化GPU编程。
与LLVM生态系统集成:作为LLVM项目的一部分,可以无缝集成到现有的LLVM工具链中。
这些优势使Polygeist在高性能计算、科学计算、机器学习等领域具有广泛的应用前景。

实际应用案例
以下是Polygeist在实际项目中的应用案例:

科学计算优化:在一个大规模数值模拟项目中,使用Polygeist对核心计算kernel进行优化,通过多面体变换和自动并行化,性能提升了30%。

机器学习框架:某开源机器学习框架使用Polygeist优化其C++后端,实现了更高效的张量运算,在某些模型上推理速度提升了20%。

图形渲染引擎:一个游戏引擎项目利用Polygeist的GPU后端支持,简化了CUDA代码的生成过程,大大提高了开发效率。

这些案例展示了Polygeist在提升程序性能和简化开发流程方面的巨大潜力。

```bash
cgeist input.c -S -emit-mlir | mlir-opt --canonicalize --cse > output.mlir

```
### 2022 LLVMHPC William S. Moses, Polygeist: C++ Frontend for MLIR

[text](https://www.youtube.com/watch?v=LIHxtR4Hop4)

#### The MLIR Framework

- MLIR is a recent compiler infrastructure designed for reuse and extensibility
- Rather than providing a predefined set of instructions and types, MLIR operates on collections of dialects that contain sets of interoperable user-defined operations, attributes and types
- Anyone can define their own optimizable dialect/operation, with a large set of existing dialects (structured control flow, affine, GPU, quantum, fully homomorphic encryption, circuits, LLVM, and more!)

#### The Polyhedral Model

- Represent programs as a collection of computations and constraints on a multi-dimensional grid (polyhedron)
- Makes it easy to analyze and specify program transformations best exploit the available hardware
- Loop restructuring for spatial/temporal locality, automatic parallelization, etc.
- One of the best frameworks for optimizing compute-intensive programs like machine learning kernels or scientific simulations as well as for programming accelerators.

#### Preserve the parallel structure

- Maintain GPU parallelism in a form understandable to the compiler
- Enables optimization between caller and kernel
- Enable parallelism-specific optimization

#### Synchronization via Memory

- Synchronization (sync_threads) ensures all threads within a block finish executing codeA before executing codeB
- The desired synchronization behavior can be reproduced by defining sync_threads to have the union of the memory semantics of the code before and after the sync. 
- This prevents code motion of instructions which require the synchronization for correctness, but permits other code motion (e.g. index computation).

- 同步 （sync_threads） 确保块中的所有线程在执行 CodeB 之前完成对 CodeA 的执行
- 可以通过定义 sync_threads 来重现所需的同步行为，以便在同步之前和之后具有代码的内存语义的并集。
- 这可以防止需要同步以确保正确性的指令的代码移动，但允许其他代码移动（例如索引计算）。

- High-level synchronization representation enables new optimizations, like sync elimination.
- A synchronize instruction is not needed if the set of read/writes before the sync don’t conflict with the read/writes after the sync.

- 高级同步表示支持新的优化，例如同步消除。
- 如果同步前的读/写集与同步后的读/写集不冲突，则不需要 synchronize 指令。

```c++
__global__ void bpnn_layerforward(...) {
  __shared__ float node[HEIGHT];
  __shared__ float weights[HEIGHT][WIDTH];

  if ( tx == 0 )
    node[ty] = input[index_in] ;

  // Unnecessary Barrier #1
  // None of the read/writes below the sync 
  //  (weights, hidden)
  // intersect with the read/writes above the sync
  //  (node, input)
  __syncthreads();
   
  // Unnecessary Store #1
  weights[ty][tx] = hidden[index];

  __syncthreads(); 
  // Unnecessary Load #1   
  weights[ty][tx] = weights[ty][tx] * node[ty];   
  // …
}

```

#### GPU Transpilation

- A unified representation of parallelism enables programs in one parallel architecture (e.g. CUDA) to be compiled to another (e.g. CPU/OpenMP)
- **Most CPU backends do not have an equivalent block synchronization** 
- Efficiently lower a top-level synchronization by distributing the parallel for loop around the sync, and interchanging control flow

```llvm
parallel_for %i = 0 to N {
  codeA(%i);   
  sync_threads;   
  codeB(%i);
}
; =>
parallel_for %i = 0 to N {
  codeA(%i);
} 

parallel_for %i = 0 to N {
  codeB(%i);
}
```

#### GPU Synchronization Lowering: Control Flow


Synchronization within control flow (for, if, while, etc) can be lowered by splitting around the control flop and interchanging the parallelism.

```c++
parallel_for %i = 0 to N {
  for %j = … {     
    codeB1(%i, %j);     
    sync_threads;      
    codeB2(%i, %j);
  }
}
; Interchange =>
for %j = … {   
    parallel_for %i = 0 to N {
        codeB1(%i, %j);     
        sync_threads;      
        codeB2(%i, %j);
  }
}
; Split =>
for %j = … {   
    parallel_for %i = 0 to N {
        codeB1(%i, %j);
    }   
    parallel_for %i = 0 to N { 
        codeB2(%i, %j);
    }
}

```

#### GPU Transpilation Performance

- CUDA programs transcompiled by Polygeist not only match the performance of handwritten OpenMP programs, but achieve a speedup!
  - 58% geomean speedup on Rodinia
  - 2.7x geomean speedup on PyTorch versus built-in CPU backend (also using our MocCUDA compatibility layer)

![GPU Transpilation Performance](GPU_Transpilation_Performance.png)

![GPU Memory Hierarchy](GPU_Memory_Hierarchy.png)