#!/bin/bash

cd ../hcl-dialect

# Build LLVM 15.0.0
if [ ! -d "llvm-project" ]; then
    git clone https://github.com/llvm/llvm-project.git
    cd llvm-project
    git checkout tags/llvmorg-15.0.0
    python3 -m pip install -r mlir/python/requirements.txt
    mkdir -p build && cd build
    cmake -G "Unix Makefiles" ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=`which python3`
    make -j8
    export LLVM_BUILD_DIR=$(pwd)
    cd ../..
else
    export LLVM_BUILD_DIR=$(pwd)/llvm-project/build
fi

# Build HeteroCL dialect
mkdir -p build && cd build
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=ON \
   -DOPENSCOP=OFF \
   -DPython3_EXECUTABLE=`which python3`
make -j8

export PYTHONPATH=$(pwd)/tools/hcl/python_packages/hcl_core:${PYTHONPATH}