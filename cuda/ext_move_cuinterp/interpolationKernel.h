void InterpolateField(float* newHX, int Nx, int Ny, int Nz, float mx, float my, float mz, int cellshift, float* geometry, int includeOrExclude);
void Precondition(float* pot, int Nx, int Ny, int Nz);
void NoPrecondition(float* pot, int Nx, int Ny, int Nz);
