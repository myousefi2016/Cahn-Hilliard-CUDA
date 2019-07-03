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

  int xp, xn, yp, yn, zp, zn;

  int nx = (int)DATAXSIZE - 1;
  int ny = (int)DATAYSIZE - 1;
  int nz = (int)DATAZSIZE - 1;

  xp = x+1;
  xn = x-1;
  yp = y+1;
  yn = y-1;
  zp = z+1;
  zn = z-1;

  if (xp > nx)
  {
   xp = 0;
  }
  if (xn < 0)
  {
   xn = nx;
  }
  
  if (yp > ny)
  {
   yp = 0;
  }
  if (yn < 0)
  {
   yn = ny;
  }

  if (zp > nz)
  {
   zp = 0;
  }
  if (zn < 0)
  {
   zn = nz;
  }

  double cxx = (c[xp][y][z] + c[xn][y][z] - 2.0*c[x][y][z]) / (dx*dx);
  double cyy = (c[x][yp][z] + c[x][yn][z] - 2.0*c[x][y][z]) / (dy*dy);
  double czz = (c[x][y][zp] + c[x][y][zn] - 2.0*c[x][y][z]) / (dz*dz);

  double result = cxx + cyy + czz;

  return result;

}

__device__ double GradientX(double phi[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  int xp, xn;

  int nx = (int)DATAXSIZE - 1;

  xp = x+1;
  xn = x-1;

  if (xp > nx)
  {
   xp = 0;
  }
  if (xn < 0)
  {
   xn = nx;
  }

  double phix = (phi[xp][y][z] - phi[xn][y][z]) / (2.0*dx);

  return phix;

}

__device__ double GradientY(double phi[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  int yp, yn;

  int ny = (int)DATAYSIZE - 1;

  yp = y+1;
  yn = y-1;

  if (yp > ny)
  {
   yp = 0;
  }
  if (yn < 0)
  {
   yn = ny;
  }

  double phiy = (phi[x][yp][z] - phi[x][yn][z]) / (2.0*dy);

  return phiy;

}

__device__ double GradientZ(double phi[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  int zp, zn;

  int nz = (int)DATAZSIZE - 1;

  zp = z+1;
  zn = z-1;

  if (zp > nz)
  {
   zp = 0;
  }
  if (zn < 0)
  {
   zn = nz;
  }

  double phiz = (phi[x][y][zp] - phi[x][y][zn]) / (2.0*dz);

  return phiz;

}

__global__ void chemicalPotential(double c[][DATAYSIZE][DATAXSIZE], double mu[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, double gamma, double e_AA, double e_BB, double e_AB)
{

 unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
 unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
 unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

 if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE))){

  mu[idx][idy][idz] = ( 9.0 / 2.0 )*( ( c[idx][idy][idz] + 1.0 ) * e_AA + ( c[idx][idy][idz] - 1 ) * e_BB - 2.0 * c[idx][idy][idz] * e_AB ) + 3.0 * c[idx][idy][idz] + c[idx][idy][idz] * c[idx][idy][idz] * c[idx][idy][idz] - gamma * Laplacian(c,dx,dy,dz,idx,idy,idz);
 }

}

__device__ double freeEnergy(double c, double e_AA, double e_BB, double e_AB)
{

 return (((9.0 / 4.0) * ((c*c+2.0*c+1.0)*e_AA+(c*c-2.0*c+1.0)*e_BB+2.0*(1.0-c*c)*e_AB)) + ((3.0/2.0) * c * c) + ((3.0/12.0) * c * c * c * c));

}

__global__ void localFreeEnergyFunctional(double c[][DATAYSIZE][DATAXSIZE], double f[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, double gamma, double e_AA, double e_BB, double e_AB)
{

 unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
 unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
 unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

 if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE))){

  f[idx][idy][idz] = freeEnergy(c[idx][idy][idz],e_AA,e_BB,e_AB) + (gamma / 2.0) * (GradientX(c,dx,dy,dz,idx,idy,idz) * GradientX(c,dx,dy,dz,idx,idy,idz) + GradientY(c,dx,dy,dz,idx,idy,idz) * GradientY(c,dx,dy,dz,idx,idy,idz) + GradientZ(c,dx,dy,dz,idx,idy,idz) * GradientZ(c,dx,dy,dz,idx,idy,idz));
 }

}

// device function to set the 3D volume
__global__ void cahnHilliard(double cnew[][DATAYSIZE][DATAXSIZE], double cold[][DATAYSIZE][DATAXSIZE], double mu[][DATAYSIZE][DATAXSIZE], double D, double dt, double dx, double dy, double dz)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
    if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE))){
      cnew[idx][idy][idz] = cold[idx][idy][idz] + dt * D * Laplacian(mu,dx,dy,dz,idx,idy,idz);
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

void write_output_vtk(double c[][DATAYSIZE][DATAXSIZE], int t, int nx, int ny, int nz, string output, string variableName)
{
    string name = "./sweep_1/" + output + "_" + to_string(t) + ".vtk";
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
    ofile << "SCALARS " << variableName << " double" << endl;
    ofile << "LOOKUP_TABLE default" << endl;
  for (int k = 0; k < nz; k++) 
    for(int j = 0; j < ny; j++)
        for(int i = 0; i < nx; i++)
            ofile << c[i][j][k] << endl;

}

double integral(double c[][DATAYSIZE][DATAXSIZE], int nx, int ny, int nz)
{
  double summation = 0.0;  

  for (int k = 0; k < nz; k++)
    for(int j = 0; j < ny; j++)
        for(int i = 0; i < nx; i++)
            summation = summation + c[i][j][k];

  return summation;
}

int main(int argc, char *argv[])
{
    int iterator = 0;

    double e_AA = -(7.0/9.0); 

    double e_AA_f = (11.0/9.0);
    
    double deltaEAA = (1.0/9.0);

    while (e_AA < e_AA_f + deltaEAA)
    {

    printf("e_AA is: %f\n",e_AA);

    double dx = 1.0;
    double dy = 1.0;
    double dz = 1.0;
    double dt = 0.01;
    double e_BB = e_AA;
    double e_AB = (2.0/9.0);
    int t_f = 25000;
    int t_freq = 10;
    double gamma = 0.5;
    double D = 1.0;
    string output_c = "outputc" + to_string(iterator);
    string output_mu = "outputmu" + to_string(iterator);
    string output_f = "outputf" + to_string(iterator);
    string variableName_c = "c";
    string variableName_mu = "mu";
    string variableName_f = "f";
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
    nRarray *mu_host;
    nRarray *f_host;
    nRarray *d_cold;  // storage for result computed on device
    nRarray *d_cnew;
    nRarray *d_muold;
    nRarray *d_fold;
// allocate storage for data set
    if ((c_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
    if ((mu_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
    if ((f_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
// allocate GPU device buffers
    cudaMalloc((void **) &d_cold, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_cnew, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_muold, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_fold, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
// compute result

    initialization(c_host);

    write_output_vtk(c_host,0,nx,ny,nz,output_c,variableName_c);

    double integral_c = 0.0;
    double integral_mu = 0.0;
    double integral_f = 0.0;

    string name_c = "./sweep_1/integral_c_"+to_string(iterator)+".txt";
    ofstream ofile_c (name_c);

    string name_mu = "./sweep_1/integral_mu_"+to_string(iterator)+".txt";
    ofstream ofile_mu (name_mu);

    string name_f = "./sweep_1/integral_f_"+to_string(iterator)+".txt";
    ofstream ofile_f (name_f);

    integral_c = integral(c_host,nx,ny,nz);

    ofile_c << 0 << "," << integral_c << endl;

    cudaMemcpy(d_cold, c_host, ((nx*ny*nz)*sizeof(double)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    double clock_d = double(clock()) / CLOCKS_PER_SEC;

    int t = 0;

    while (t < t_f) {

    //printf("Timestep is: %d\n",t);

    chemicalPotential<<<gridSize,blockSize>>>(d_cold,d_muold,dx,dy,dz,gamma,e_AA,e_BB,e_AB);
    cudaCheckErrors("Kernel launch failure");
    localFreeEnergyFunctional<<<gridSize,blockSize>>>(d_cold,d_fold,dx,dy,dz,gamma,e_AA,e_BB,e_AB);
    cudaCheckErrors("Kernel launch failure");
    cahnHilliard<<<gridSize,blockSize>>>(d_cnew,d_cold,d_muold,D,dt,dx,dy,dz);
    cudaCheckErrors("Kernel launch failure");

    if (t == 0) {

     cudaMemcpy(mu_host, d_muold, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);   
     cudaCheckErrors("CUDA memcpy failure");

     cudaMemcpy(f_host, d_fold, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
     cudaCheckErrors("CUDA memcpy failure");

     write_output_vtk(mu_host,t,nx,ny,nz,output_mu,variableName_mu);

     write_output_vtk(f_host,t,nx,ny,nz,output_f,variableName_f);

     integral_mu = integral(mu_host,nx,ny,nz);

     ofile_mu << 0 << "," << integral_mu << endl;

     integral_f = integral(f_host,nx,ny,nz);

     ofile_f << 0 << "," << integral_f << endl;

    }

    if (t % t_freq == 0 && t > 0) {

     cudaMemcpy(c_host, d_cnew, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
     cudaCheckErrors("CUDA memcpy failure");

     cudaMemcpy(mu_host, d_muold, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
     cudaCheckErrors("CUDA memcpy failure");

     cudaMemcpy(f_host, d_fold, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
     cudaCheckErrors("CUDA memcpy failure");

     write_output_vtk(c_host,t,nx,ny,nz,output_c,variableName_c);

     write_output_vtk(mu_host,t,nx,ny,nz,output_mu,variableName_mu);

     write_output_vtk(f_host,t,nx,ny,nz,output_f,variableName_f);

     integral_c = integral(c_host,nx,ny,nz);

     ofile_c << t << "," << integral_c << endl;

     integral_mu = integral(mu_host,nx,ny,nz);

     ofile_mu << t << "," << integral_mu << endl;

     integral_f = integral(f_host,nx,ny,nz);

     ofile_f << t << "," << integral_f << endl;

    }
    
    Swap<<<gridSize,blockSize>>>(d_cnew, d_cold);
    cudaCheckErrors("Kernel launch failure");

    t++;

    }

    cudaThreadSynchronize();
    clock_d = double(clock()) / CLOCKS_PER_SEC - clock_d; 
    printf("GPU time = %.3fms\n",clock_d*1e3);

    free(c_host);
    free(mu_host);
    free(f_host);
    cudaFree(d_cold);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_cnew);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_muold);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_fold);
    cudaCheckErrors("cudaFree fail"); 

    iterator++;
    e_AA += deltaEAA;
    }
    return 0;
}
