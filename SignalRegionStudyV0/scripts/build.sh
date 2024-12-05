#!/bin/bash
rm -rf build lib
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$WORKDIR/SignalRegionStudy $WORKDIR/SignalRegionStudy
make -j4 && make install
cd -
export LD_LIBRARY_PATH=$WORKDIR/SignalRegionStudy/lib:$LD_LIBRARY_PATH
