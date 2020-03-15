cmake_minimum_required(VERSION 3.0)

project(cudatest)

# CUDA 6.5 supports compute capability 1.1 – 5.x. It needs gcc-4.8.
# To install gcc-4.8 on Ubuntu 19.04 temporary add line to /etc/apt/sources.list
# deb [url] bionic main universe
# and make "apt update && apt install g++-4.8"
set(CUDA_VERSION 6.5)

# https://en.wikipedia.org/wiki/CUDA#Supported_GPUs
# compute_11 - Tesla except G80
# compute_20 - Fermi
# compute_30 - Kepler
# compute_50 - Maxwell
set(COMPUTE_CAPABILIY compute_12)
set(CUDA_NVCC_FLAGS "-ccbin gcc-4.8 -arch=${COMPUTE_CAPABILIY}")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${CUDA_VERSION}")
find_package(CUDA 6.5 REQUIRED)

set(CXX_OPT "-fPIC -std=c++11")
set(CMAKE_CXX_FLAGS "${CXX_OPT}")
set(CMAKE_EXE_LINKER_FLAGS "${CXX_OPT}")

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

cuda_add_executable(test test.cu)
