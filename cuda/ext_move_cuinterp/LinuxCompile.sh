nvcc -m64 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 --shared -Xcompiler -fPIC -o libinterpolationKernel.so interpolationKernel.cu
cp ./libinterpolationKernel.so ../../../../../../bin/libinterpolationKernel.so
