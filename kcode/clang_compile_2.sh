set -x

/usr/local/opt/llvm/bin/clang -Xclang -finclude-default-header -cl-std=CL3.0 ALS.cl -Weverything -c -emit-llvm -DVALUE_TYPE=float
