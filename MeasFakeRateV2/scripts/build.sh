#!/bin/bash
export WORKDIR=`pwd`
export ROOT_INCLUDE_PATH=$WORKDIR/include:$ROOT_INCLUDE_PATH
rm -rf build lib
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$WORKDIR ..
make && make install
cd -
