cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_USE_SANITIZER="Address;Undefined" \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON -DLLVM_USE_SPLIT_DWARF=ON

# Using clang and lld speeds up the build, we recommend adding:
#  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
# CCache can drastically speed up further rebuilds, try adding:
#  -DLLVM_CCACHE_BUILD=ON
# Optionally, using ASAN/UBSAN can find bugs early in development, enable with:
# -DLLVM_USE_SANITIZER="Address;Undefined"
# Optionally, enabling integration tests as well
# -DMLIR_INCLUDE_INTEGRATION_TESTS=ON

# -DCMAKE_BUILD_TYPE=Debug or -DCMAKE_BUILD_TYPE=RelWithDebInfo
