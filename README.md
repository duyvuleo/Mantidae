# A development version of mantis (https://github.com/trevorcohn/mantis/)

### Dependencies

Before compiling dynet, you need:

 * [Eigen](https://bitbucket.org/eigen/eigen), e.g. 3.3.x

 * [cuda](https://developer.nvidia.com/cuda-toolkit) version 7.5 or higher

 * [cmake](https://cmake.org/), e.g., 3.5.1 using *cmake* ubuntu package

### Building

First, clone the repository

    git clone https://bitbucket.org/duyvuleo/mantis-dev

As mentioned above, you'll need the latest [development] version of eigen

    hg clone https://bitbucket.org/eigen/eigen/

A modified version of latest dynet (https://github.com/clab/dynet/tree/master/dynet) is already included (e.g., dynet folder).

#### CPU build

Compiling to execute on a CPU is as follows

    mkdir build_cpu
    cd build_cpu
    cmake .. -DEIGEN3_INCLUDE_DIR=eigen
    make -j 2 

MKL support. If you have Intel's MKL library installed on your machine, you can speed up the computation on the CPU by:

    cmake .. -DEIGEN3_INCLUDE_DIR=EIGEN -DMKL=TRUE -DMKL_ROOT=MKL -DENABLE_BOOST=TRUE

substituting in different paths to EIGEN and MKL if you have placed them in different directories. 

This will build the 3 binaries
    
    build_cpu/src/attentional
    build_cpu/src/biattentional
    build_cpu/src/relopt-decoder
    build_cpu/src/train-rnnlm
    build_cpu/src/dual-learning

#### GPU build

Building on the GPU uses the Nvidia CUDA library, currently tested against version 7.5.
The process is as follows

    mkdir build_gpu
    cd build_gpu
    cmake .. -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=EIGEN -DCUDA_TOOLKIT_ROOT_DIR=CUDA -DENABLE_BOOST=TRUE
    make -j 2

substituting in your Eigen and CUDA folders, as appropriate.

This will result in the 3 binaries

    build_gpu/src/attentional
    build_gpu/src/biattentional
    build_gpu/src/relopt-decoder
    build_gpu/src/train-rnnlm
    build_gpu/src/dual-learning

#### Using the model

See the script/RelOpt-README.txt

## Contacts

Hoang Cong Duy Vu, Trevor Cohn and Reza Haffari 

---
Updated June 2017
