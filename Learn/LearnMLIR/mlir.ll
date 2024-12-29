module {
        memref.global "private" constant @string : memref<13xi8> = dense<[0x48,0x65,0x6c,0x6c,0x6f,0x2c,0x20,0x4d,0x4c,0x49,0x52,0x21,0]>
        llvm.func external @puts(!llvm.ptr<i8>) -> ()
        func.func @main() -> i64 {
                %c0_i64 = arith.constant 0 : i64
                %0 = memref.get_global @string : memref<13xi8>
                %1 = memref.extract_aligned_pointer_as_index %0 : memref<13xi8> -> index
                %2 = arith.index_cast %1 : index to i64
                %3 = llvm.inttoptr %2 : i64 to !llvm.ptr<i8>
                llvm.call @puts(%3) : (!llvm.ptr<i8>) -> ()
                return %c0_i64 : i64
        }
}

module attributes {llvm.data_layout = ""} {
  llvm.mlir.global private constant @string(dense<[72, 101, 108, 108, 111, 44, 32, 77, 76, 73, 82, 33, 0]> : tensor<13xi8>) {addr_space = 0 : i32} : !llvm.array<13 x i8>
  llvm.func @puts(!llvm.ptr<i8>)
  llvm.func @main() -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(13 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.null : !llvm.ptr
    %4 = llvm.getelementptr %3[13] : (!llvm.ptr) -> !llvm.ptr, i8
    %5 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %6 = llvm.mlir.addressof @string : !llvm.ptr
    %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<13 x i8>
    %8 = llvm.mlir.constant(3735928559 : index) : i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %12 = llvm.insertvalue %7, %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %1, %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %16 = llvm.insertvalue %2, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %17 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr<i8>
    llvm.call @puts(%18) : (!llvm.ptr<i8>) -> ()
    llvm.return %0 : i64
  }
}

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@string = private constant [13 x i8] c"Hello, MLIR!\00"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @puts(ptr)

define i64 @main() {
  call void @puts(ptr @string)
  ret i64 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

//定义函数forward，接受参数arg0，参数类型为tensor<1x16xf32>，函数的返回类型为tensor<1x10xf32>
func.func @forward(%arg0: tensor<1x16xf32>) -> tensor<1x10xf32> {
//定义常量，类型为tensor<1x16xf32>。常量是通过tosa.const操作创建，tosa.const操作接受属性value，其中value类型为tensor<1x16xf32>
    %0 = "tosa.const"() {value = dense<"0xC44B..."> : tensor<1x16xf32>} : () -> tensor<1x16xf32>
    //定义常量，类型为tensor<1x10xf32>。常量是通过tosa.const操作创建，tosa.const操作接受属性value，其中value类型为tensor<1x10xf32>
    %1 = "tosa.const"() {value = dense<"0xA270..."> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
    //对arg0进行类型进行变换，从类型tensor<1x16xf32>变成tensor<1x1x16xf32>
    %2 = "tosa.reshape"(%arg0) {new_shape = [1, 1, 16]} : (tensor<1x16xf32>) -> tensor<1x1x16xf32>
    //对%2和%0进行类matmul计算，输入类型为tensor<1x1x16xf32>, tensor<1x16x10xf32>，输出类型为tensor<1x1x10xf32>
    %3 = "tosa.matmul"(%2, %0) : (tensor<1x1x16xf32>, tensor<1x16x10xf32>) -> tensor<1x1x10xf32>
    //对%3进行类型进行变换，从类型tensor<1x1x10xf32>变成tensor<1x10xf32>
    %4 = "tosa.reshape"(%3) {new_shape = [1, 10]} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
    //对%4和%1进行张量加法，输入和输出类型都是tensor<1x10xf32>
    %5 = "tosa.add"(%4, %1) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    //返回%5，类型为tensor<1x10xf32>
    return %5 : tensor
} 

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)> 
func.func @forward(%arg0: tensor<1x16xf32>) -> tensor<1x10xf32> {
  %cst = arith.constant dense<"0xA270..."> : tensor<1x10xf32>
    %cst_0 = arith.constant dense<"0xC44B..."> : tensor<16x10xf32>
    %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %cst_0 : tensor<1x16xf32>, tensor<16x10xf32>) outs(%cst : tensor<1x10xf32>)
    {
        ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
            %1 = arith.mulf %arg1, %arg2 : f32
            %2 = arith.addf %arg3, %1 : f32
            linalg.yield %2 : f32
    } -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
} 