#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include <random>
#include <fstream>

// set a 3D volume
// To compile it with nvcc execute: nvcc -O2 -o set3d set3d.cu
//define the data set size (cubic volume)
#define DATAXSIZE 128
#define DATAYSIZE 128
#define DATAZSIZE 128
//define the chunk sizes that each threadblock will work on
#define BLKXSIZE 32
#define BLKYSIZE 4
#define BLKZSIZE 4

using namespace std;

// for cuda error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            return 1; \
        } \
    } while (0)

__device__ double Laplacian(double c[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  double cxx = (c[x+1][y][z] + c[x-1][y][z] - 2.0*c[x][y][z]) / (dx*dx);
  double cyy = (c[x][y+1][z] + c[x][y-1][z] - 2.0*c[x][y][z]) / (dy*dy);
  double czz = (c[x][y][z+1] + c[x][y][z-1] - 2.0*c[x][y][z]) / (dz*dz);

  double result = cxx + cyy + czz;

  return result;

}

__global__ void chemicalPotential(double c[][DATAYSIZE][DATAXSIZE], double mu[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, double gamma)
{

 unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
 unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
 unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

 if ((idx < (DATAXSIZE-1)) && (idy < (DATAYSIZE-1)) && (idz < (DATAZSIZE-1)) && (idx > (0)) && (idy > (0)) && (idz > (0))){

  mu[idx][idy][idz] = c[idx][idy][idz]*c[idx][idy][idz]*c[idx][idy][idz] - c[idx][idy][idz] - gamma*Laplacian(c,dx,dy,dz,idx,idy,idz);
 }
 else
 {
  mu[idx][idy][idz] = 0.0;
 }

}

// device function to set the 3D volume
__global__ void cahnHilliard(double cnew[][DATAYSIZE][DATAXSIZE], double cold[][DATAYSIZE][DATAXSIZE], double mu[][DATAYSIZE][DATAXSIZE], double D, double dt, double dx, double dy, double dz)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    if ((idx < (DATAXSIZE-1)) && (idy < (DATAYSIZE-1)) && (idz < (DATAZSIZE-1)) && (idx > (0)) && (idy > (0)) && (idz > (0))){
      cnew[idx][idy][idz] = cold[idx][idy][idz] + dt * D * Laplacian(mu,dx,dy,dz,idx,idy,idz);
      }
}

__global__ void boundaryConditions(double cnew[][DATAYSIZE][DATAXSIZE])
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx == 0){
      cnew[idx][idy][idz] = cnew[idx+1][idy][idz];
      }
    else if (idx == DATAXSIZE-1){
      cnew[idx][idy][idz] = cnew[idx-1][idy][idz];
      }
    else if (idy == 0){
      cnew[idx][idy][idz] = cnew[idx][idy+1][idz];
      }
    else if (idy == DATAYSIZE-1){
      cnew[idx][idy][idz] = cnew[idx][idy-1][idz];
      }
    else if (idz == 0){
      cnew[idx][idy][idz] = cnew[idx][idy][idz+1];
      }
    else if (idz == DATAZSIZE-1){
      cnew[idx][idy][idz] = cnew[idx][idy][idz-1];
      }

}

__global__ void Swap(double cnew[][DATAYSIZE][DATAXSIZE], double cold[][DATAYSIZE][DATAXSIZE])
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    double tmp;    

    if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE))) {
     tmp=cnew[idx][idy][idz];
     cnew[idx][idy][idz]=cold[idx][idy][idz];
     cold[idx][idy][idz]=tmp;
    }

}

void initialization(double c[][DATAYSIZE][DATAXSIZE])
{
    for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
     for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
      for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {
      double f = (double)rand() / RAND_MAX;
      c[idx][idy][idz] = -1.0 + 2.0*f;
    }
     }
      }
}

void write_output_vtk(double c[][DATAYSIZE][DATAXSIZE], int t, int nx, int ny, int nz)
{
    string name = "./out/output_" + to_string(t) + ".vtk";
    ofstream ofile (name);

    // vtk preamble
    ofile << "# vtk DataFile Version 2.0" << endl;
    ofile << "OUTPUT by LIBM\n";
    ofile << "ASCII" << endl;

    // write grid
    ofile << "DATASET RECTILINEAR_GRID" << endl;
    ofile << "DIMENSIONS " << nx << " " << ny << " " << nz << endl;
    ofile << "X_COORDINATES " << nx << " float" << endl;
    for(size_t i = 0; i < nx; i++)
        ofile << i << "\t";
    ofile << endl;
    ofile << "Y_COORDINATES " << ny << " float" << endl;
    for(size_t i = 0; i < ny; i++)
        ofile << i << "\t";
    ofile << endl;
    ofile << "Z_COORDINATES " << nz << " float" << endl;
    for(size_t i = 0; i < nz; i++)
        ofile << i << "\t";
    ofile << endl;

    // point data
    ofile << "POINT_DATA " << nx*ny*nz << endl;

    // write rho
    ofile << "SCALARS " << "c" << " double" << endl;
    ofile << "LOOKUP_TABLE default" << endl;
  for (int k = 0; k < nz; k++) 
    for(int j = 0; j < ny; j++)
        for(int i = 0; i < nx; i++)
            ofile << c[i][j][k] << endl;

}

int main(int argc, char *argv[])
{
    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double dt = 0.01;
    int t_f = 2500;
    int t_freq = 10;
    double gamma = 0.5;
    double D = 1.0;
    cudaSetDevice(0.0);
    typedef double nRarray[DATAYSIZE][DATAXSIZE];
    const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 gridSize(((DATAXSIZE+BLKXSIZE-1)/BLKXSIZE), ((DATAYSIZE+BLKYSIZE-1)/BLKYSIZE), ((DATAZSIZE+BLKZSIZE-1)/BLKZSIZE));
// overall data set sizes
    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;
// pointers for data set storage via malloc
    nRarray *c_host; // storage for result stored on host
    nRarray *d_cold;  // storage for result computed on device
    nRarray *d_cnew;
    nRarray *d_muold;
// allocate storage for data set
    if ((c_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
// allocate GPU device buffers
    cudaMalloc((void **) &d_cold, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_cnew, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_muold, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
// compute result

    initialization(c_host);

    write_output_vtk(c_host,0,nx,ny,nz);

    cudaMemcpy(d_cold, c_host, ((nx*ny*nz)*sizeof(double)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    double clock_d = double(clock()) / CLOCKS_PER_SEC;

    int t = 0;

    while (t < t_f) {

    printf("Timestep is: %d\n",t);

    chemicalPotential<<<gridSize,blockSize>>>(d_cold,d_muold,dx,dy,dz,gamma);
    cudaCheckErrors("Kernel launch failure");
    cahnHilliard<<<gridSize,blockSize>>>(d_cnew,d_cold,d_muold,D,dt,dx,dy,dz);
    cudaCheckErrors("Kernel launch failure");
    boundaryConditions<<<gridSize,blockSize>>>(d_cnew);
    cudaCheckErrors("Kernel launch failure");

    if (t % t_freq == 0 && t > 0) {

     cudaMemcpy(c_host, d_cnew, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
     cudaCheckErrors("CUDA memcpy failure");

     write_output_vtk(c_host,t,nx,ny,nz);

    }
    
    Swap<<<gridSize,blockSize>>>(d_cnew, d_cold);
    cudaCheckErrors("Kernel launch failure");

    t++;

    }

    cudaThreadSynchronize();
    clock_d = double(clock()) / CLOCKS_PER_SEC - clock_d; 
    printf("GPU time = %.3fms\n",clock_d*1e3);

    free(c_host);
    cudaFree(d_cold);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_cnew);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_muold);
    cudaCheckErrors("cudaFree fail");
    return 0;
}
