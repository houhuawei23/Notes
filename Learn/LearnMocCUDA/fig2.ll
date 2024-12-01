target triple = "x86_64-unknown-linux-gnu"

define @launch(float* %d_out, float* %d_in, i32 %n) {
    call @__cudaPushCallConfiguration(...)
    call @__cudaLaunchKernel(@normalize_stub, ...)
    ret
}


target triple = "nvptx64"

define @normalize(float* %out, float* %in, i32 %n) {
    %tid = call i32 @llvm.nvvm.ptx.tid.x()
    %sum = call i32 @sum(i32* %in, i32 %n)
    %cmp = icmp slt i32 %tid, %n
    br i1 %cmp, label %body, label %exit
body:
    %gep = getelementptr float* %in, i32 %tid
    %load = load float, float* %gep
    %nrm = fdiv float %load, %sum
    %ptr = getelementptr float* %out, i32 %tid
    store float %nrm, float* %ptr
    br label %exit
exit:
    ret
}