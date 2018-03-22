nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_20 -o spkernmulrsymm3d_20.ptx spkernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_30,code=sm_30 -o spkernmulrsymm3d_30.ptx spkernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_35,code=sm_35 -o spkernmulrsymm3d_35.ptx spkernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_50,code=sm_50 -o spkernmulrsymm3d_50.ptx spkernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_52,code=sm_52 -o spkernmulrsymm3d_52.ptx spkernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_53,code=sm_53 -o spkernmulrsymm3d_53.ptx spkernmulrsymm3d.cu
cuda2go spkernmulrsymm3d.cu