#!/bin/bash
set -x

../exec/clMF -P 1 -d 0 -q 0 -k 10 -t 10 -l 0.05 -V 2 -c ../kcode/ -nThreadsPerBlock 32 -nBlocks 8192 ../../DATASETS/jester/

../exec/clMF -P 1 -d 0 -q 0 -k 10 -t 10 -l 0.05 -V 2 -c ../kcode/ -nThreadsPerBlock 32 -nBlocks 8192 ../../DATASETS/ml10M/
