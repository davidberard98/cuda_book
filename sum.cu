#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <time.h>
#include <sys/time.h>

#define CHECK(call)                                                      \
{                                                                        \
  const cudaError_t error = call;                                        \
  if (error != cudaSuccess) {                                            \
    printf("Error: %s:%d", __FILE__, __LINE__);                          \
    printf("code:%d, reason %s\n", error, cudaGetErrorString(error));    \
    exit(1);                                                             \
  }                                                                      \
}

void sumOnHost(float* A, float* B, float *C, const int N) {
  for (int idx=0; idx<N; ++idx) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void sumOnDevice(float *A, float *B, float *C, int nElem) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nElem) {
    C[idx] = A[idx] + B[idx];
  }
}

void initialData(float *ip, int size) {
  time_t t;
  srand((unsigned int) time(&t));

  for (int i=0; i<size; ++i) {
    ip[i] = (float)( rand() & 0xFF )/10.0f;
  }
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

struct TimerGuard {
  double start_time;
  std::string label_;
  TimerGuard(std::string label = "") : label_(label) {
    start_time = cpuSecond();
  }
  ~TimerGuard() {
    double end_time = cpuSecond();
    std::cout << " time " << label_ << ": " << int((end_time - start_time)*1e6) << " us " << std::endl;
  }
};

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1e-8;
  for (int i = 0; i < N; ++i) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      printf("Arrays do not match!\n");
      printf("  host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      // std::cout << "Arrays do not match!\n";
      // std::cout << "  host " << hostRef[i] << " gpu " << %gpuRef[i] << " at current " << i << std::endl;
      return;
    }
  }
  std::cout << "Arrays match\n" << std::endl;
}

int main(int argc, char **argv) {
  int mb = std::stoi(argv[1]);
  int nElem = 1024 * 1024 * mb;
  size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *h_C, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  initialData(h_A, nElem);
  initialData(h_B, nElem);

  float *d_A, *d_B, *d_C;
  CHECK(cudaMalloc((float**)&d_A, nBytes));
  CHECK(cudaMalloc((float**)&d_B, nBytes));
  CHECK(cudaMalloc((float**)&d_C, nBytes));

  dim3 block = 256;
  dim3 grid = ((nElem + block.x - 1) / block.x);

  {
    TimerGuard guard("cpu");
    sumOnHost(h_A, h_B, h_C, nElem);
  }

  CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

  {
    TimerGuard guard("gpu");

    sumOnDevice <<<grid, block>>> (d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
  }

  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

  {
    TimerGuard guard("gpu");

  CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    sumOnDevice <<<grid, block>>> (d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
  }

  checkResult(h_C, gpuRef, nElem);

  free(h_A);
  free(h_B);
  free(h_C);

  return(0);
}
