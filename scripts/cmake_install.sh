#!/bin/bash

wget https://cmake.org/files/v3.13/cmake-3.13.1-Linux-x86_64.sh
sudo sh cmake-3.13.1-Linux-x86_64.sh --prefix="$HOME"/.local/ --exclude-subdir
