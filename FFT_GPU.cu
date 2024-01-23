//**************************************************************
// Assignment 4
// Jose Soto and Nathan Van De Vyvere
// Parallel Programming Date: 11/03/2022
//**************************************************************
// Runs on maverick2
// sbatch a5Script
//
// Documentation: This program will compute FFT Radix-2
// with 8192 Samples. The host will create the sample
// table. The host calls the compute kernel where the device
// will divide up the discrete FFT to be computed.
//*************************************************************
//*************************************************************

#include <math.h>
#include <stdio.h>

const int SIZE = 8192;
const int BLOCKS = 4;
const int MAX_THREAD = 1024;
const double PI = 6.283185307179586;    // 2 * PI -- 15 digits after the decimal

//////////////////////////////////////////////////////////////////////////////////////////////////
/*
Function Name: compute
Parameters: double* k_vector_real, double* k_vector_imag, double* real_sample, double* imag_sample
Retunrn type: void
Description: This kernal will do the computations for each sample of the FFT. The kernal brings in
the sample data from the host. Each thread will calculate the even and odd parts for the k that 
matches its global index. The computed data is indexed into an array an passed back to the host.
*/
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void compute(double* k_vector_real, double *k_vector_imag, double* real_sample, double* imag_sample)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;          // global index
  double COS_PC, SIN_PC, PC;                              // variables for the pi constant
  double EVEN_REAL, EVEN_IMAG, ODD_REAL, ODD_IMAG;        // variables for calculations

  // where sample data = x + yi
  // where PC is the pi constant for the even/odd summations
  // e^(-PCi) = cos(PC) - isin(PC)
  // even/odd summation = (x + yi) * [cos(PC) - isin(PC)]
  //                    = xcos(PC) - xisin(PC) + yicos(PC) + ysin(PC)
  //               real = xcos(PC) + ysin(PC)
  //                odd = yisin(PC) - xisin(PC)
  // summation loop for both the even and odd parts
  for (int m = 0; m < SIZE/2; m++)
  {
    PC = (PI * m * k)/(SIZE/2);
    COS_PC = cos(PC);
    SIN_PC = sin(PC);
    EVEN_REAL += real_sample[m+m] * COS_PC + imag_sample[m+m] * SIN_PC;
    ODD_REAL += real_sample[m+m+1] * COS_PC + imag_sample[m+m+1] * SIN_PC;
    EVEN_IMAG += imag_sample[m+m] * COS_PC - real_sample[m+m] * SIN_PC;
    ODD_IMAG += imag_sample[m+m+1] * COS_PC - real_sample[m+m+1] * SIN_PC;
  } 
  PC = PI * k / SIZE;   //twiddle factor
  // twiddle factor multiplied by the odd summation
  COS_PC = cos(PC);
  SIN_PC = sin(PC);
  ODD_REAL = ODD_REAL * COS_PC + ODD_IMAG * SIN_PC;
  ODD_IMAG = ODD_IMAG * COS_PC - ODD_REAL * SIN_PC;

  // for each k -> SUM(even) + twiddle factor * SUM(odd)
  k_vector_real[k] = EVEN_REAL + ODD_REAL;
  k_vector_imag[k] = EVEN_IMAG + ODD_IMAG;

  // for each k+N/2 -> SUM(even) - twiddle factor * SUM(odd)
  k_vector_real[k+(SIZE/2)] = EVEN_REAL - ODD_REAL;
  k_vector_imag[k+(SIZE/2)] = EVEN_IMAG - ODD_IMAG;
}

int main(void) {
  double k_vector_real[SIZE], k_vector_imag[SIZE];
  double real_sample[SIZE] = {3.6, 2.9, 5.6, 4.8, 3.3, 5.9, 5.0, 4.3};
  double imag_sample[SIZE] = {2.6, 6.3, 4.0, 9.1, 0.4, 4.8, 2.6, 4.1};

  double *real_sample_d, *imag_sample_d, *k_vector_real_d, *k_vector_imag_d;
  int size_d = SIZE * sizeof(double);

  // allocates memory in the GPU
  cudaMalloc((void**) &real_sample_d, size_d);
  cudaMalloc((void**) &imag_sample_d, size_d);
  cudaMalloc((void**) &k_vector_real_d, size_d);
  cudaMalloc((void**) &k_vector_imag_d, size_d);

  // copies the sample data to the allocated memory in the GPU
  cudaMemcpy(real_sample_d, real_sample, size_d, cudaMemcpyHostToDevice);
  cudaMemcpy(imag_sample_d, imag_sample, size_d, cudaMemcpyHostToDevice);
  
  // allocates the blocks in the GPU on the x-axis
  dim3 dimGrid(BLOCKS,1);
  // allocates the number of threads per block in the GPU
  dim3 dimBlock(MAX_THREAD,1);
  // calls the kernal function for computation
  compute <<< dimGrid, dimBlock >>> (k_vector_real_d, k_vector_imag_d, real_sample_d, imag_sample_d);
  
  // copies the vectors with the results from the GPU back to the host
  cudaMemcpy(k_vector_real, k_vector_real_d, size_d, cudaMemcpyDeviceToHost);
  cudaMemcpy(k_vector_imag, k_vector_imag_d, size_d, cudaMemcpyDeviceToHost);
  // cudaMemcpy(real_sample, real_sample_d, size_d, cudaMemcpyDeviceToHost);

  printf("\nTOTAL PROCESSED SAMPLES : %d\n",SIZE);
  printf("==========================================\n");
  for (int i = 0; i < 8; i++)
  {
    printf("XR[%d]: %.6f XI[%d]: %.6f \n", i, k_vector_real[i], i, k_vector_imag[i]);
    printf("==========================================\n");
  }
  printf("==========================================\n");
  for (int i = 4096; i < (4096+8); i++)
  {
    printf("XR[%d]: %.6f XI[%d]: %.6f \n", i, k_vector_real[i], i, k_vector_imag[i]);
    printf("==========================================\n");
  }

  // deallocates memory from the GPU
  cudaFree(k_vector_imag_d);
  cudaFree(k_vector_real_d);
  cudaFree(real_sample_d);
  cudaFree(imag_sample_d);

  return 0;
}
