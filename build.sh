#! /bin/sh
mkdir -p build
cd build
cmake -DFIND_OPENMP=ON -DFIND_CUDA=ON -DUSE_PYQPANDA=ON ..
make -j4

