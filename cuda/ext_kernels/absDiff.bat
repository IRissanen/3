nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_20 -o absDiff_20.ptx absDiff.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_30,code=sm_30 -o absDiff_30.ptx absDiff.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_35,code=sm_35 -o absDiff_35.ptx absDiff.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_50,code=sm_50 -o absDiff_50.ptx absDiff.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_52,code=sm_52 -o absDiff_52.ptx absDiff.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_53,code=sm_53 -o absDiff_53.ptx absDiff.cu
cuda2go absDiff.cu