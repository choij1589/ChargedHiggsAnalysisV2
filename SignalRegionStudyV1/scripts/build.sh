#!/bin/bash
rm -rf build lib
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$WORKDIR/SignalRegionStudyV1 $WORKDIR/SignalRegionStudyV1
make -j4 && make install
cd -
export LD_LIBRARY_PATH=$WORKDIR/SignalRegionStudyV1/lib:$LD_LIBRARY_PATH
