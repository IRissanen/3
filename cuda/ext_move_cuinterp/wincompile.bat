nvcc -shared -gencode arch=compute_20,code=compute_20 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 --machine 64 -Xcompiler /GS- -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC" -L"C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib\amd64" -lcudart -o interpolationKernel.dll interpolationKernel.cu
gendef interpolationKernel.dll
dlltool -dllname interpolationKernel.dll --def interpolationKernel.def --output-lib libinterpolationKernel.a
COPY .\interpolationKernel.dll ..\..\..\..\..\..\bin\interpolationKernel.dll
