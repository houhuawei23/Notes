// Kernel launch is available within the calling
// function, enabling optimizations across the
// GPU/CPU boundary.
func @launch(%h_out : memref<?xf32>,
%h_in : memref<?xf32>, %n : i64) {
    // Parallel for across all blocks in a grid.
    parallel.for (%gx, %gy, %gz) = (0, 0, 0)
                to (grid.x, grid.y, grid.z) {
        // Shared memory = stack allocation in a block.
        %shared_val = memref.alloca : memref<f32>
        
        // Parallel for across all threads in a block.
        parallel.for (%tx, %ty, %tz) = (0, 0, 0)
                to (blk.x, blk.y, blk.z) {
            // Control-flow is directly preserved.
            if %tx == 0 {
                %sum = func.call @sum(%d_in, %n)
                memref.store %sum, %shared_val[] : memref<f32>
            }
            // Syncronization via explicit operation.
            polygeist.barrier(%tx, %ty, %tz)
            %tid = %gx + grid.x * %tx
            if %tid < %n {
                %res = ...
                store %res, %d_out[%tid] : memref<?xf32>
            }
        }
    }
}

Figure 3:Representation of the shared-memory variant of the CUDA launch/normalize code from Figure 1 in Polygeist/MLIR. 
The kernel call is made available directly in the host code which calls it. 
The parallelism is made explicit through the use of parallel for loops across the blocks and threads, 
and shared memory is placed within the block to signify it can be accessed from any thread in the same block, 
but not from a different block.

图 3： Polygeist/MLIR 中图1中 CUDA启动/标准化代码的共享内存变体的表示。
内核调用可以直接在调用它的主机代码中使用。
通过使用跨块和线程的并行 for 循环来明确并行性，并将共享内存放置在块内，
以表示可以从同一块中的任何线程访问它，但不能从不同的块访问它。