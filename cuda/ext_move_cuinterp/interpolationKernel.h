__declspec(dllexport) void InterpolateField(float* newHX, int Nx, int Ny, int Nz, float mx, float my, float mz, int cellshift, float* geometry, int includeOrExclude);
__declspec(dllexport) void Precondition(float* pot, int Nx, int Ny, int Nz);
__declspec(dllexport) void NoPrecondition(float* pot, int Nx, int Ny, int Nz);
