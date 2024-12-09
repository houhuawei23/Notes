# Linalg

Linalg is designed to solve the High-level Hierarchical Optimization (HHO box) in MLIR and to interoperate nicely within a *Mixture Of Expert Compilers* environment (i.e. the *CGSel* box).Linalg 旨在解决 MLIR 中的高级分层优化（HHO 框）问题，并在混合专家编译器环境中良好地交互（即 CGSel 框）。


## Set of Key Transformations一组关键变换

The following key transformations have been central to driving the design of Linalg. They are all implemented in terms of the properties of the `linalg.generic` OpInterface and avoid the pitfall of relying on hardcoded one-off op knowledge.以下关键转换是推动 Linalg 设计的关键。它们都基于 linalg.generic OpInterface 的性质实现，避免了依赖于硬编码的单次操作知识的陷阱。

The textual form description of these transformations is left for future work. Still, it is useful to list the key transformations that are performed on the Linalg IR and that have influenced its design:这些变换的文本形式描述留待以后的工作。尽管如此，列出在 Linalg IR 上执行的关键变换以及对其设计产生影响的变换是有用的：

1. Progressive Buffer Allocation.渐进式缓冲区分配。
2. Parametric Tiling. 参数铺砖。
3. Promotion to Temporary Buffer in Fast Memory.提升至快速内存中的临时缓冲区。
4. Tiled Producer-Consumer Fusion with Parametric Tile-And-Fuse.基于参数化分块融合的镶嵌式生产者-消费者融合
5. Map to Parallel and Reduction Loops and Hardware.并行和归约循环及硬件的映射
6. Vectorization: Rewrite in Vector Form.向量化：以向量形式重写。
7. Lower to Loops (Affine, Generic, and Parallel).下到循环（仿射、通用和并行）。
8. Lower to Library Calls or Special Instructions, Intrinsics or ISA.下至库函数调用或特殊指令，内建函数或指令集架构。
9. Partially Lower to Iterations Over a Finer-Grained Linalg Op.部分降低到更细粒度的 Linalg 操作迭代。
