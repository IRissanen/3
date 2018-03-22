// 3D micromagnetic scalar potential kernel multiplication:
//
//        |Kx|   |Mx|
// |S | = |Ky| * |My|
//        |Kz|   |Mz|
//

extern "C" __global__ void
spkernmulRSymm3D(float* __restrict__  fftMx,  float* __restrict__  fftMy,  float* __restrict__  fftMz,
               float* __restrict__  fftKx, float* __restrict__  fftKy, float* __restrict__  fftKz,
               int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;


	if(ix>= Nx || iy>= Ny || iz>=Nz) {
		return;
	}

	// fetch (complex) FFT'ed magnetization
	int I = (iz*Ny + iy)*Nx + ix;
	int e = 2 * I;
	float reMx = fftMx[e  ];
	float imMx = fftMx[e+1];
	float reMy = fftMy[e  ];
	float imMy = fftMy[e+1];
	float reMz = fftMz[e  ];
	float imMz = fftMz[e+1];

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
	
	// m * v vector dot product, overwrite m_x with result.
	fftMx[e  ] = - imMx * Kxi - imMy * Kyi - imMz * Kzi;
	fftMx[e+1] = reMx * Kxi + reMy * Kyi + reMz * Kzi;
}

