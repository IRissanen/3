nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_20 -o %1_20.ptx %1.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_30,code=sm_30 -o %1_30.ptx %1.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_35,code=sm_35 -o %1_35.ptx %1.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_50,code=sm_50 -o %1_50.ptx %1.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_52,code=sm_52 -o %1_52.ptx %1.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_53,code=sm_53 -o %1_53.ptx %1.cu
cuda2go %1.cu
