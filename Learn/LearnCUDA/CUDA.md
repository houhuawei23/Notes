

# CUDA Learn

## NVIDIA CUDA
The NVIDIA® CUDA® Toolkit provides a comprehensive development environment for C and C++ developers building GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize, and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler, and a runtime library to deploy your application.

NVIDIA® CUDA® 工具包为构建 GPU 加速应用程序的 C 和 C++ 开发人员提供了一个全面的开发环境。借助 CUDA 工具包，您可以在 GPU 加速的嵌入式系统、桌面工作站、企业数据中心、基于云的平台和 HPC 超级计算机上开发、优化和部署您的应用程序。该工具包包括 GPU 加速库、调试和优化工具、C/C++ 编译器以及用于部署应用程序的运行时库。

Using built-in capabilities for distributing computations across multi-GPU configurations, scientists and researchers can develop applications that scale from single GPU workstations to cloud installations with thousands of GPUs.

使用内置功能在多 GPU 配置之间分配计算，科学家和研究人员可以开发从单个 GPU 工作站扩展到具有数千个 GPU 的云安装的应用程序。


Programming Massively Parallel Processors: A Hands-on Approach, Second Edition, teaches students how to program massively parallel processors. It offers a detailed discussion of various techniques for constructing parallel programs. Case studies are used to demonstrate the development process, which begins with computational thinking and ends with effective and efficient parallel programs. This guide shows both student and professional alike the basic concepts of parallel programming and GPU architecture. Topics of performance, floating-point format, parallel patterns, and dynamic parallelism are covered in depth. This revised edition contains more parallel programming examples, commonly-used libraries such as Thrust, and explanations of the latest tools. It also provides new coverage of CUDA 5.0, improved performance, enhanced development tools, increased hardware support, and more; increased coverage of related technology, OpenCL and new material on algorithm patterns, GPU clusters, host programming, and data parallelism; and two new case studies (on MRI reconstruction and molecular visualization) that explore the latest applications of CUDA and GPUs for scientific research and high-performance computing. This book should be a valuable resource for advanced students, software engineers, programmers, and hardware engineers.

Programming Massively Parallel Processors： A Hands-on Approach，第二版，教授学生如何对大规模并行处理器进行编程。它详细讨论了用于构建并行程序的各种技术。案例研究用于演示开发过程，该过程从计算思维开始，以有效和高效的并行程序结束。本指南向学生和专业人士展示了并行编程和 GPU 架构的基本概念。深入介绍了性能、浮点格式、并行模式和动态并行性等主题。此修订版包含更多并行编程示例、常用库（如 Thrust）以及最新工具的解释。它还提供了 CUDA 5.0 的新覆盖范围、改进的性能、增强的开发工具、增强的硬件支持等;增加了相关技术、OpenCL 和有关算法模式、GPU 集群、主机编程和数据并行性的新材料的覆盖范围;以及两个新的案例研究（关于 MRI 重建和分子可视化），探索 CUDA 和 GPU 在科学研究和高性能计算中的最新应用。这本书应该是高级学生、软件工程师、程序员和硬件工程师的宝贵资源。


[NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/doc/index.html)

[NVIDIA cuda-education](https://developer.nvidia.com/cuda-education)


[cuda-samples](https://github.com/houhuawei23/cuda-samples)

[CUDA Programming Course – High-Performance Computing with GPUs](https://www.youtube.com/watch?v=86FAWCzIe_4&t=13s)

[d_what_are_some_good_resources_to_learn_cuda](https://www.reddit.com/r/MachineLearning/comments/w52iev/d_what_are_some_good_resources_to_learn_cuda/?rdt=40526)

[What-are-some-of-the-best-resources-to-learn-CUDA-C](https://www.quora.com/What-are-some-of-the-best-resources-to-learn-CUDA-C)

Programming Massively Parallel Processors: A Hands-on Approach

[cuda-training-series](https://www.olcf.ornl.gov/cuda-training-series/)

https://developer.nvidia.com/blog/even-easier-introduction-cuda/

https://learnopencv.com/demystifying-gpu-architectures-for-deep-learning/

[numba](https://numba.readthedocs.io/en/stable/cuda/overview.html)

[GPU-Puzzles](https://github.com/houhuawei23/GPU-Puzzles)

[CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)

[GPU编程](https://lulaoshi.info/gpu/)

[nvvm-ir-spec](https://docs.nvidia.com/cuda/nvvm-ir-spec/)
## PTX Parallel Thread Execution

PTX: a low-level parallel thread execution virtual machine and instruction set architecture.

PTX 是一种低级并行线程执行虚拟机和指令集体系结构。

PTX exposes the GPU as data-parallel computing device.






# Numba

Overview 概述

Numba supports CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model. Kernels written in Numba appear to have direct access to NumPy arrays. NumPy arrays are transferred between the CPU and the GPU automatically.

Numba 通过将 Python 代码的受限子集直接编译为遵循 CUDA 执行模型的 CUDA 内核和设备函数来支持 CUDA GPU 编程。用 Numba 编写的内核似乎可以直接访问 NumPy 数组。 NumPy 数组在 CPU 和 GPU 之间自动传输。


## Install CUDA

[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)