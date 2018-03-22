nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_20 -o eddyLaplaceMove_20.ptx eddyLaplaceMove.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_30,code=sm_30 -o eddyLaplaceMove_30.ptx eddyLaplaceMove.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_35,code=sm_35 -o eddyLaplaceMove_35.ptx eddyLaplaceMove.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_50,code=sm_50 -o eddyLaplaceMove_50.ptx eddyLaplaceMove.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_52,code=sm_52 -o eddyLaplaceMove_52.ptx eddyLaplaceMove.cu
nvcc -ptx -ccbin="C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -gencode arch=compute_20,code=sm_53 -o eddyLaplaceMove_53.ptx eddyLaplaceMove.cu
cuda2go eddyLaplaceMove.cu