nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_20 -o eddykernmulrsymm3d_20.ptx eddykernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_30,code=sm_30 -o eddykernmulrsymm3d_30.ptx eddykernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_35,code=sm_35 -o eddykernmulrsymm3d_35.ptx eddykernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_50,code=sm_50 -o eddykernmulrsymm3d_50.ptx eddykernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_52,code=sm_52 -o eddykernmulrsymm3d_52.ptx eddykernmulrsymm3d.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_53 -o eddykernmulrsymm3d_53.ptx eddykernmulrsymm3d.cu
cuda2go eddykernmulrsymm3d.cu