#!/bin/bash

export CMAKE_VERSION=3.21.3

wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh
sudo sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --prefix="$HOME"/.local/ --exclude-subdir
