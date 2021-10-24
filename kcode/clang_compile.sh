set -x

/usr/local/opt/llvm/bin/clang -Xclang -finclude-default-header -cl-std=CL3.0 ALS_rolled.cl -Weverything -c -emit-llvm -DVALUE_TYPE=float -DK_SIZE=10

