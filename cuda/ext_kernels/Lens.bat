nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_20 -o Lens_20.ptx Lens.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_30,code=sm_30 -o Lens_30.ptx Lens.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_35,code=sm_35 -o Lens_35.ptx Lens.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_50,code=sm_50 -o Lens_50.ptx Lens.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_52,code=sm_52 -o Lens_52.ptx Lens.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_53,code=sm_53 -o Lens_53.ptx Lens.cu
cuda2go Lens.cu