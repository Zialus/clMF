#!/bin/bash
set -x

../exec/clMF -P 0 -d 1 -q 0 -k 10 -t 10 -l 0.05 -V 2 -c ../kcode/ -nThreadsPerBlock 1 -nBlocks 32 ../../DATASETS/jester/

../exec/clMF -P 0 -d 1 -q 0 -k 10 -t 10 -l 0.05 -V 2 -c ../kcode/ -nThreadsPerBlock 1 -nBlocks 32 ../../DATASETS/ml10M/
