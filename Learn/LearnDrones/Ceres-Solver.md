# Ceres-Solver

[doc](http://ceres-solver.org/)

Install (on debian12)

```bash
# CMake
sudo apt-get install cmake

# google-glog + gflags
# libgoogle-glog-dev: library that implements application-level logging
# libgflags-dev: commandline flags module for C++
sudo apt-get install libgoogle-glog-dev libgflags-dev

# Use ATLAS for BLAS & LAPACK
# libatlas-base-dev: Automatically Tuned Linear Algebra Software, generic static
sudo apt-get install libatlas-base-dev

# Eigen3
# libeigen3-dev: lightweight C++ template library for linear algebra
sudo apt-get install libeigen3-dev

# SuiteSparse (optional)
# libsuitesparse-dev: libraries for sparse matrices computations (development files)
sudo apt-get install libsuitesparse-dev # to slow, skip it
```

```cmake
# FindTBB.cmake
file(STRINGS
    "${TBB_INCLUDE_DIR}/tbb/version.h" # tbb_stddef.h -> version.h
    # https://github.com/ceres-solver/ceres-solver/issues/1036
    TBB_VERSION_CONTENTS
    REGEX "VERSION")
```

```bash
tar zxf ceres-solver-2.2.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.2.0
make -j3
make test
# Optionally install Ceres, it can also be exported using CMake which
# allows Ceres to be used without requiring installation, see the documentation
# for the EXPORT_BUILD_DIR option for more information.
make install
```