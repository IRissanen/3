// 3D micromagnetic cross product with a distance kernel:
//
//        |Kx|   |(dB/dt)_x|
// |S | = |Ky| X |(dB/dt)_y|
//        |Kz|   |(dB/dt)_z|
//

extern "C" __global__ void
eddykernmulRSymm3D(float* __restrict__  fftdBx,  float* __restrict__  fftdBy,  float* __restrict__  fftdBz,
               float* __restrict__  fftKx, float* __restrict__  fftKy, float* __restrict__  fftKz,
               int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;


	if(ix>= Nx || iy>= Ny || iz>=Nz) {
		return;
	}

	// fetch (complex) FFT'ed dB/dt
	int I = (iz*Ny + iy)*Nx + ix;
	int e = 2 * I;
	float redBx = fftdBx[e  ];
	float imdBx = fftdBx[e+1];
	float redBy = fftdBy[e  ];
	float imdBy = fftdBy[e+1];
	float redBz = fftdBz[e  ];
	float imdBz = fftdBz[e+1];

    float signY = 1.0f;
    float signZ = 1.0f;

    // use symmetry to fetch from redundant parts:
    // mirror index into first quadrant and set signs.
    if (iy > Ny/2) {
        iy = Ny-iy;
        signY = -signY;
    }
    if (iz > Nz/2) {
        iz = Nz-iz;
        signZ = -signZ;
    }

    // fetch kernel element from non-redundant part
    // and apply minus signs for mirrored parts.
    I = (iz*(Ny/2+1) + iy)*Nx + ix; // Ny/2+1: only half is stored
    float Kxi = fftKx[I];
    float Kyi = signY*fftKy[I];
    float Kzi = signZ*fftKz[I];
	
	// dB/dt x v vector cross product, overwrite dB/dt with result.
	fftdBx[e  ] = (- imdBy*Kzi + imdBz*Kyi);
	fftdBx[e+1] = (redBy*Kzi - redBz*Kyi);
	fftdBy[e  ] = -(- imdBx*Kzi + imdBz*Kxi);
	fftdBy[e+1] = -(redBx*Kzi - redBz*Kxi);
	fftdBz[e  ] = (- imdBx*Kyi + imdBy*Kxi);
	fftdBz[e+1] = (redBx*Kyi - redBy*Kxi);
}

