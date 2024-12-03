# Kimi

这两个C文件（`passthrough.h.txt` 和 `passthrough.c.txt`）构成了一个名为 `cudawrap` 的库，它的作用是提供一个封装层，用于加载和访问 CUDA 相关的库（如 CUDA Runtime、cuDNN 和 cuBLAS）中的函数。这个封装层允许程序在运行时动态地加载这些库，并提供了一种机制来调用这些库中的函数。以下是对这两个文件的详细分析：

### `passthrough.h` 文件

这个头文件定义了 `cudawrap` 库的数据结构和宏，用于封装对 CUDA 相关库的调用。

1. **数据结构定义**：
   - `entry_t`：一个结构体，包含一个函数指针 `fn_ptr` 和一个函数名 `name`，用于存储库中函数的地址和名称。

2. **宏定义**：
   - `ENTER` 和 `LEAVE`：用于调试目的，记录函数的进入和离开时间，以及计算函数执行时间。
   - `TIME_GEMM_START` 和 `TIME_GEMM_STOP`：用于性能分析，记录和计算 GEMM（General Matrix Multiply）操作的开始和结束时间。
   - `CUDART_ENTRY_ENUM`、`CUDNN_ENTRY_ENUM` 和 `CUBLAS_ENTRY_ENUM`：宏，用于生成枚举值，代表不同的库函数。
   - `CUDART_FIND_ENTRY`、`CUDNN_FIND_ENTRY` 和 `CUBLAS_FIND_ENTRY`：宏，用于查找特定函数的入口。
   - `REAL_CUDA_RT_API_CALL`、`REAL_CUDNN_CALL` 和 `REAL_CUBLAS_CALL`：宏，用于实际调用库中的函数。

3. **枚举定义**：
   - `cudart_entry_enum_t`、`cudnn_entry_enum_t` 和 `cublas_entry_enum_t`：枚举类型，列出了所有需要封装的 CUDA Runtime、cuDNN 和 cuBLAS 函数。

### `passthrough.c` 文件

这个源文件实现了 `passthrough.h` 中定义的数据结构和函数。

1. **全局变量初始化**：
   - `_cudart_inited_`、`_cudnn_inited_` 和 `_cublas_inited_`：用于标记对应的库是否已经初始化。

2. **库函数入口数组**：
   - `cudart_lib_entry`、`cudnn_lib_entry` 和 `cublas_lib_entry`：数组，存储了对应库的所有函数入口。

3. **加载库函数**：
   - `load_cudart_lib`、`load_cudnn_lib` 和 `load_cublas_lib`：函数，用于动态加载 CUDA Runtime、cuDNN 和 cuBLAS 库，并查找所需的函数地址。

4. **初始化检查函数**：
   - `cudart_lib_inited`、`cudnn_lib_inited` 和 `cublas_lib_inited`：函数，用于检查对应的库是否已经初始化。

### 总结

`cudawrap` 库的主要作用是提供一个中间层，使得应用程序可以更加灵活地使用 CUDA 相关的库。通过动态加载和查找函数，它允许应用程序在不同的环境（如不同的 CUDA 版本或配置）下运行，而无需重新编译。此外，它还提供了性能分析和调试支持，帮助开发者优化和调试 CUDA 相关的代码。

## 使用方法

要使用 `cudawrap` 库，你需要按照以下步骤操作：

1. **包含头文件**：在你的 C 或 C++ 源文件中包含 `passthrough.h` 头文件。

2. **链接库文件**：在编译你的程序时，确保链接了 `cudawrap` 库以及它依赖的 CUDA 相关库（如 libcudart.so, libcudnn.so, libcublas.so）。

3. **初始化库**：在程序中调用 `load_cudart_lib()`, `load_cudnn_lib()`, 和 `load_cublas_lib()` 函数来加载对应的 CUDA 库，并初始化 `cudawrap` 库。

4. **调用封装函数**：使用 `cudawrap` 提供的宏来调用 CUDA 库中的函数。

下面是一个简单的例子，展示了如何使用 `cudawrap` 库来调用 CUDA Runtime API 中的 `cudaMalloc` 和 `cudaFree` 函数：

```c
#include "passthrough.h"

int main() {
    // 初始化 CUDA 运行时库
    if (load_cudart_lib() != 0) {
        fprintf(stderr, "Failed to load CUDA runtime library.\n");
        return -1;
    }

    // 分配 CUDA 设备内存
    void *devPtr;
    size_t size = 1024 * 1024; // 分配 1MB 内存
    REAL_CUDA_RT_API_CALL(cudart_lib_entry, , cudaMalloc, (void**)&devPtr, size);

    if (devPtr == NULL) {
        fprintf(stderr, "Failed to allocate device memory.\n");
        // 这里应该进行清理工作
        return -1;
    }

    // ... 在这里进行其他 CUDA 操作 ...

    // 释放 CUDA 设备内存
    REAL_CUDA_RT_API_CALL(cudart_lib_entry, , cudaFree, (devPtr));

    // 清理工作，关闭程序
    return 0;
}
```

在这个例子中，我们首先调用 `load_cudart_lib()` 来加载 CUDA 运行时库并初始化 `cudawrap` 库。然后，我们使用 `REAL_CUDA_RT_API_CALL` 宏来封装对 `cudaMalloc` 和 `cudaFree` 函数的调用。这个宏会查找对应的函数入口，并执行函数调用。

请注意，这个例子假设 `cudawrap` 库已经被正确编译并链接到你的程序中。在实际使用中，你可能需要根据你的环境和编译器来调整编译和链接选项。此外，错误处理和资源管理（如内存泄漏）也是实际编程中需要考虑的重要方面。

## Hijack CUBLAS

`hijack_cublas` 是一个库，它提供了对 cuBLAS 库函数的封装和可能的替代实现。这个库的目的是在不同的环境下，如模拟 CUDA 环境或在不支持 CUDA 的系统上，提供 cuBLAS 函数的接口。以下是对 `hijack_cublas` 的进一步分析：

### `hijack_cublas.h` 文件

这个头文件定义了 `hijack_cublas` 库的主要结构和枚举类型，以及函数原型。

1. **枚举类型**：
   - `cublasStatus_t`：定义了 cuBLAS API 可能返回的状态码。
   - `cublasOperation_t`：定义了矩阵操作类型，如 `CUBLAS_OP_N`（无转置）、`CUBLAS_OP_T`（转置）和 `CUBLAS_OP_C`（共轭转置）。

2. **结构体定义**：
   - `cublasContext`：定义了一个结构体，包含一个初始化标志和一个 CUDA 流。
   - `cublasHandle_t`：`cublasContext` 结构体的指针类型，用于作为 cuBLAS 库函数的上下文句柄。

3. **外部变量声明**：
   - `cublas_entry[]`：声明了一个 `entry_t` 类型的数组，用于存储 cuBLAS 库函数的入口。

4. **函数类型定义**：
   - `cublas_sym_t`：定义了一个函数指针类型，指向一个返回 `cublasStatus_t` 的函数。

### `hijack_cublas.c` 文件

这个源文件实现了 `hijack_cublas.h` 中声明的函数，提供了 cuBLAS 函数的具体实现。

1. **cuBLAS 函数实现**：
   - `cublasCreate_v2`：创建一个 cuBLAS 上下文（句柄）。
   - `cublasDestroy_v2`：销毁一个 cuBLAS 上下文（句柄）。
   - `cublasSetStream_v2`：设置 cuBLAS 上下文的 CUDA 流。

2. **异步 GEMM 实现**：
   - `cublasSgemm_async_fn`：一个异步执行 SGEMM 操作的函数，它使用 OpenMP 进行多线程计算。

3. **SGEMM 函数实现**：
   - `cublasSgemm_v2`：执行矩阵乘法操作（SGEMM）。这个函数根据编译时定义的条件（如 `USE_MocCUDA` 和 `USE_GCD`）来决定是调用真实的 cuBLAS 函数还是模拟的实现。

4. **条件编译**：
   - 根据定义的宏（如 `USE_MocCUDA`、`USE_GCD` 等），`hijack_cublas` 可以选择使用不同的实现策略。例如，如果定义了 `USE_MocCUDA`，则会使用模拟的 cuBLAS 实现，而不是调用真实的 CUDA 函数。

5. **调试和性能分析**：
   - 文件中包含了 `ENTER` 和 `LEAVE` 宏，用于调试和性能分析，记录函数的进入和离开时间。

### 总结

`hijack_cublas` 库的作用是在不具备完整 CUDA 支持的环境下，提供 cuBLAS 库的接口和可能的模拟实现。它允许开发者在编写代码时，不需要关心后端是真实的 CUDA 环境还是模拟环境，从而提高了代码的可移植性。此外，它还提供了异步执行和多线程支持，以优化性能。通过条件编译，`hijack_cublas` 可以灵活地适应不同的使用场景和需求。



# ChatGPT

The provided **C** and **header** files implement a dynamic loader for CUDA libraries (**libcudart**, **cudnn**, and **cublas**) and a wrapper for their functionality. Here's a detailed breakdown:

### Purpose
The main role of these files is to **dynamically load CUDA libraries at runtime** instead of linking them statically. This approach:
1. Enables the use of CUDA functionalities without requiring the application to depend on specific versions of CUDA libraries during compilation.
2. Provides a mechanism for runtime flexibility, allowing switching between different library versions.
3. Facilitates debugging, logging, and performance tracking of library function calls.

---

### **`passthrough.c` Analysis**

#### Key Features
1. **Dynamic Library Loading:**
   - Functions like `dlopen()` and `dlsym()` dynamically load the CUDA libraries and resolve symbols (function pointers).
   - If the library or a specific function is not found, it emits errors and gracefully exits.

2. **Function Resolution:**
   - Arrays (`cudart_lib_entry`, `cudnn_lib_entry`, and `cublas_lib_entry`) store the function names and their resolved function pointers (`fn_ptr`).
   - Functions like `load_cudart_lib()`, `load_cudnn_lib()`, and `load_cublas_lib()` populate these arrays.

3. **Initialization Flags:**
   - Static variables like `_cudart_inited_`, `_cudnn_inited_`, and `_cublas_inited_` prevent redundant re-initialization.

4. **Error Handling:**
   - Robust error checking for missing libraries and functions.

#### CUDA Libraries
- **`libcudart`**: Basic CUDA Runtime functionalities (e.g., `cudaMalloc`, `cudaMemcpy`).
- **`cudnn`**: Deep Neural Network primitives.
- **`cublas`**: Basic Linear Algebra operations.

---

### **`passthrough.h` Analysis**

#### Structure Definitions
- **`entry_t`**: A structure for maintaining a function pointer (`fn_ptr`) and the corresponding function name (`name`).

#### Wrapper Macros
1. **Function Wrapping:**
   - `REAL_CUDA_RT_API_CALL`, `REAL_CUDNN_CALL`, and `REAL_CUBLAS_CALL` resolve symbols and execute them.
   - Example: A call to `cudaMalloc()` gets resolved dynamically using the corresponding function pointer.

2. **Logging and Debugging:**
   - `ENTER` and `LEAVE` macros log function entry and exit with execution timings, but only if compiled with `DEBUG` or `FUNC_TIMINGS` flags.
   - `TIME_GEMM_START` and `TIME_GEMM_STOP` specifically measure performance of GEMM (General Matrix Multiplication) operations, crucial for BLAS and deep learning workloads.

#### Enumerations
- Enumerates all supported library functions (`CUDART_ENTRY_ENUM`, `CUDNN_ENTRY_ENUM`, and `CUBLAS_ENTRY_ENUM`) for easy indexing in the arrays.

#### Initialization Functions
- `load_cudart_lib()`, `load_cudnn_lib()`, and `load_cublas_lib()` are declared for library loading at runtime.

---

### Typical Workflow
1. **Initialization:**
   - A CUDA library (e.g., `libcudart`) is dynamically loaded when a function is called for the first time.
   - Initialization flags ensure one-time setup.

2. **Function Execution:**
   - For example, a wrapped call to `cudaMalloc()` would:
     - Resolve the function pointer via `dlsym()`.
     - Execute the function through the resolved pointer.

3. **Debugging and Logging (Optional):**
   - Execution times and other statistics are logged if debugging macros are enabled.

---

### Role in **CUDAWRAP**
These files act as the **core implementation for a dynamic wrapper** around CUDA libraries, suitable for environments where:
1. **Runtime Flexibility**: The application can dynamically switch between different versions of CUDA libraries by setting paths via environment variables (`SWITCHCUDART`, `SWITCHCUDNN`, `SWITCHCUBLAS`).
2. **Debugging Support**: Tracks performance and function calls without recompiling the CUDA libraries.
3. **Ease of Integration**: Reduces compilation dependencies, making applications portable across systems with varying CUDA versions.

Would you like to explore specific sections or use cases in more detail?

## How to use

To use the **CUDAWRAP** library defined in the provided files, follow these steps. Here's a detailed explanation with an example of how to integrate it into a project and use its features.

---

### **1. Setting Up the Environment**
The **CUDAWRAP** library uses environment variables to dynamically load CUDA libraries. You need to set these variables to point to the correct library paths. For example:

```bash
export SWITCHCUDART=/usr/local/cuda/lib64/libcudart.so
export SWITCHCUDNN=/usr/local/cuda/lib64/libcudnn.so
export SWITCHCUBLAS=/usr/local/cuda/lib64/libcublas.so
```

If the environment variables are not set, the library defaults to paths like `PATHTOCUDA_ROOT/lib64/libcudart.so`.

---

### **2. Compilation**
You need to compile the `passthrough.c` and link it into your project. Use the following steps:

1. **Compile the `passthrough.c`:**
   ```bash
   gcc -fPIC -c passthrough.c -o passthrough.o -ldl
   ```
   - `-fPIC`: Generates position-independent code (necessary for shared libraries).
   - `-ldl`: Links the dynamic loader library (`dlopen`, `dlsym`).

2. **Link the compiled object into your application:**
   ```bash
   gcc main.c passthrough.o -o myapp -ldl
   ```

---

### **3. Example Code**

**`main.c`** – Using CUDAWRAP to call CUDA functions.

```c
#include <stdio.h>
#include "passthrough.h" // Include the header

int main() {
    // Step 1: Load the CUDA Runtime Library (libcudart)
    if (load_cudart_lib() != 0) {
        fprintf(stderr, "Failed to load libcudart library!\n");
        return 1;
    }

    // Step 2: Initialize CUDA
    int deviceCount = 0;

    // Dynamically call the cudaGetDeviceCount function
    typedef cudaError_t (*cudaGetDeviceCount_t)(int *);
    cudaGetDeviceCount_t cudaGetDeviceCount =
        (cudaGetDeviceCount_t)CUDART_FIND_ENTRY(cudart_lib_entry, cudaGetDeviceCount);

    if (cudaGetDeviceCount == NULL) {
        fprintf(stderr, "Function cudaGetDeviceCount not found!\n");
        return 1;
    }

    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    // Step 3: Use another CUDA function
    void *devPtr;
    size_t size = 1024;

    typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
    cudaMalloc_t cudaMalloc =
        (cudaMalloc_t)CUDART_FIND_ENTRY(cudart_lib_entry, cudaMalloc);

    if (cudaMalloc == NULL) {
        fprintf(stderr, "Function cudaMalloc not found!\n");
        return 1;
    }

    err = cudaMalloc(&devPtr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Successfully allocated %zu bytes on the device\n", size);

    // Cleanup (if necessary)
    typedef cudaError_t (*cudaFree_t)(void *);
    cudaFree_t cudaFree =
        (cudaFree_t)CUDART_FIND_ENTRY(cudart_lib_entry, cudaFree);

    if (cudaFree != NULL) {
        cudaFree(devPtr);
    }

    return 0;
}
```

---

### **4. Explanation of Example**

1. **Loading the Library:**
   The `load_cudart_lib()` function dynamically loads the CUDA runtime library and populates `cudart_lib_entry` with function pointers.

2. **Finding a Function:**
   Use `CUDART_FIND_ENTRY` macro to retrieve the pointer for the desired function (e.g., `cudaGetDeviceCount`, `cudaMalloc`).

3. **Calling the Function:**
   Cast the function pointer to the appropriate type and call it like a regular function.

4. **Handling Errors:**
   Check for `NULL` function pointers and handle errors using CUDA's error functions (`cudaGetErrorString`).

---

### **5. Logging and Debugging**
Enable logging and performance tracking by compiling with the `DEBUG` or `FUNC_TIMINGS` flags. For example:

```bash
gcc -DDEBUG main.c passthrough.o -o myapp -ldl
```

This will log entry and exit times for CUDA functions, which is helpful for debugging and profiling.

---

### **6. Benefits of Using CUDAWRAP**
- **Runtime Flexibility:** Swap between CUDA library versions without recompilation.
- **Debugging:** Trace function calls and measure performance.
- **Compatibility:** Avoid hard-coding library paths, making your application portable across systems with different CUDA installations.

---

Would you like further clarification on any part or help setting up a specific CUDA example?

## Hijack CUBLAS

The `hijack_cublas.h` file builds upon the **passthrough** dynamic loading mechanism to provide wrappers for the cuBLAS library. Here’s how you can understand and use this extension effectively.

---

### **Purpose of `hijack_cublas.h`**
This header file:
1. Extends the **CUDAWRAP** framework to specifically work with the cuBLAS library.
2. Introduces cuBLAS-specific types and enumerations (e.g., `cublasStatus_t`, `cublasOperation_t`).
3. Implements mechanisms to hijack cuBLAS calls for logging, debugging, or dynamically replacing the cuBLAS library.

---

### **Key Components**

#### 1. **Enums for cuBLAS Status and Operations**
- **`cublasStatus_t`**:
  Represents return statuses for cuBLAS functions (e.g., success, allocation failure, or execution errors).

- **`cublasOperation_t`**:
  Represents matrix operations (`CUBLAS_OP_N` for normal, `CUBLAS_OP_T` for transpose, etc.).

#### 2. **`cublasContext`**
- Placeholder structure that mimics a cuBLAS handle (`cublasHandle_t`).
- Contains:
  - **`__inited__`**: Indicates if the handle has been initialized.
  - **`stream`**: CUDA stream associated with the handle.
  
> Note: This is a stub and may require proper initialization in a real-world scenario.

#### 3. **Function Wrapping**
- **`cublas_sym_t`**:
  A generic type for function pointers to cuBLAS functions.
  
- **`cublas_entry[]`**:
  Similar to `cudart_lib_entry` in `passthrough.h`, it stores cuBLAS function names and pointers, allowing dynamic resolution.

---

### **Integration into a Project**

To use `hijack_cublas.h` for cuBLAS calls:

#### **1. Include Headers**
```c
#include "passthrough.h"
#include "hijack_cublas.h"
```

#### **2. Initialize cuBLAS Library**
Call `load_cublas_lib()` to dynamically load the cuBLAS library and initialize the `cublas_entry` table:
```c
if (load_cublas_lib() != 0) {
    fprintf(stderr, "Failed to load cuBLAS library!\n");
    return -1;
}
```

#### **3. Use cuBLAS Functions Dynamically**
Retrieve and execute cuBLAS functions via `CUBLAS_FIND_ENTRY` macro:

```c
// Example: cublasCreate_v2
typedef cublasStatus_t (*cublasCreate_t)(cublasHandle_t *);
cublasCreate_t cublasCreate = (cublasCreate_t)CUBLAS_FIND_ENTRY(cublas_lib_entry, cublasCreate_v2);

cublasHandle_t handle;
cublasStatus_t status = cublasCreate(&handle);

if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed: %d\n", status);
    return -1;
}
```

---

### **Example Application**

Here’s a full example demonstrating how to use `hijack_cublas.h` with matrix multiplication (`cublasSgemm_v2`).

```c
#include <stdio.h>
#include "hijack_cublas.h"

// Example: Matrix multiplication using cuBLAS
int main() {
    // Step 1: Load cuBLAS library
    if (load_cublas_lib() != 0) {
        fprintf(stderr, "Failed to load cuBLAS library!\n");
        return -1;
    }

    // Step 2: Create cuBLAS handle
    typedef cublasStatus_t (*cublasCreate_t)(cublasHandle_t *);
    cublasCreate_t cublasCreate = (cublasCreate_t)CUBLAS_FIND_ENTRY(cublas_lib_entry, cublasCreate_v2);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create cuBLAS handle!\n");
        return -1;
    }

    // Step 3: Perform SGEMM (Single-precision General Matrix Multiply)
    float alpha = 1.0f, beta = 0.0f;
    int m = 3, n = 3, k = 3;

    float A[m * k] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[k * n] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float C[m * n] = {0};

    typedef cublasStatus_t (*cublasSgemm_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                            int, int, int, const float *, const float *, int,
                                            const float *, const float *, int, float *);
    cublasSgemm_t cublasSgemm = (cublasSgemm_t)CUBLAS_FIND_ENTRY(cublas_lib_entry, cublasSgemm_v2);

    if (cublasSgemm == NULL) {
        fprintf(stderr, "Function cublasSgemm_v2 not found!\n");
        return -1;
    }

    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        m, n, k, &alpha, A, m, B, k, &beta, C, m);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed: %d\n", status);
        return -1;
    }

    // Print result
    printf("Matrix C:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    // Cleanup
    typedef cublasStatus_t (*cublasDestroy_t)(cublasHandle_t);
    cublasDestroy_t cublasDestroy = (cublasDestroy_t)CUBLAS_FIND_ENTRY(cublas_lib_entry, cublasDestroy_v2);

    if (cublasDestroy) cublasDestroy(handle);

    return 0;
}
```

---

### **Key Steps in Example**
1. **Loading**: `load_cublas_lib` initializes the function pointer table.
2. **Retrieving Function Pointers**: Use `CUBLAS_FIND_ENTRY` to locate specific cuBLAS functions.
3. **Calling Functions**: Use resolved function pointers like normal cuBLAS functions.
4. **Matrix Multiplication**: Performs single-precision GEMM (`cublasSgemm_v2`).
5. **Cleanup**: Destroy cuBLAS handle using `cublasDestroy_v2`.

---

Would you like a deeper dive into any specific cuBLAS function or other areas of this framework?