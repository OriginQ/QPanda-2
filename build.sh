#! /bin/sh
mkdir -p build
cd build
cmake -DFIND_CUDA=OFF -DUSE_PYQPANDA=OFF ..
make

