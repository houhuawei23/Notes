__device__ float sum(float* data, int n){ … }

__global__ void normalize(float* out, float* in, int n) {
  int tid = blockIdx.x + blockDim.x * threadIdx.x;
  // Optimization: Compute the sum once per block.
  // __shared__ int val;
  // if (threadIdx.x == 0) val = sum(in, n);
  // __syncthreads;
  float val = sum(in, n);
  if (tid < n) out[tid] = in[tid] / val;
}

void launch(int* d_out, int* d_in, int n) {
  normalize<<<(n + 31) / 32, 32>>>(d_out, d_in, n);
}

/*

Figure 1:A sample CUDA program `normalize`, which normalizes a vector and the CPU function launch
which calls the kernel. Presently, the call to sum is called in each thread, leading to a total of
O​(N2) work. The work can be partially reduced to O​(N2/B) through the use of shared memory (in
the comments above), which enables the sum to be computed once per block, or completely reduced to
O​(N) by computing sum once before the kernel.

一个示例 CUDA 程序 `normalize` ，它规范化向量 bing you launch 调用 kernrl `normalize`。
目前，在每个线程中调用 sum，从而导致总共O​(N2) 的工作。
通过使用共享内存（在上面的评论中），可以部分减少O​(N2/B) 工作量，
这使得每个块可以计算一次总和，或者通过在内核之前计算一次来完全减少 O​(N) 。
 */