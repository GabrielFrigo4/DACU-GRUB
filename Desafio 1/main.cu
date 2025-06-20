#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// === Simple error checking macro ===
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, const char *const func, const char *const file, int const line) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
				file, line, static_cast<unsigned int>(result),
				cudaGetErrorString(result), func);
		exit(EXIT_FAILURE);
	}
}

// === Basic argument handling ===
bool checkCmdLineFlag(int argc, char **argv, const char *flag) {
	for (int i = 1; i < argc; ++i)
		if (strstr(argv[i], flag)) return true;
	return false;
}

int getCmdLineArgumentInt(int argc, char **argv, const char *argName) {
	size_t nameLen = strlen(argName);
	for (int i = 1; i < argc; ++i)
		if (strncmp(argv[i], argName, nameLen) == 0 && argv[i][nameLen] == '=')
			return atoi(&argv[i][nameLen + 1]);
	return 0;
}

int findCudaDevice() {
	int deviceCount = 0;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "No CUDA-capable device found\n");
		exit(EXIT_FAILURE);
	}
	checkCudaErrors(cudaSetDevice(0));
	return 0;
}

// === Kernel ===
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin + wA - 1;
	int aStep = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;

	float Csub = 0;

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		__syncthreads();
	}

	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
	for (int i = 0; i < size; ++i) data[i] = val;
}

// === Host-side matrix multiplication driver ===
int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA, const dim3 &dimsB) {
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A;
	checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B;
	checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

	ConstantInit(h_A, size_A, 1.0f);
	ConstantInit(h_B, size_B, 0.01f);

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C;
	checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

	float *d_A, *d_B, *d_C;
	checkCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
	checkCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
	checkCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	printf("Computing result using CUDA kernel...\n");

	if (block_size == 16)
		MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	else
		MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

	checkCudaErrors(cudaStreamSynchronize(stream));
	printf("done\n");

	checkCudaErrors(cudaEventRecord(start, stream));

	for (int i = 0; i < 300; ++i) {
		if (block_size == 16)
			MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
		else
			MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
	}

	checkCudaErrors(cudaEventRecord(stop, stream));
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	float msecPerMatrixMul = msecTotal / 300.0f;

	double flops = 2.0 * dimsA.x * dimsA.y * dimsB.x;
	double gigaFlops = (flops * 1e-9f) / (msecPerMatrixMul / 1000.0f);

	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
		   gigaFlops, msecPerMatrixMul, flops, threads.x * threads.y);

	checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));

	printf("Checking result for correctness...\n");
	bool correct = true;
	double eps = 1.e-6;

	for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
		double expected = dimsA.x * 0.01f;
		double diff = fabs(h_C[i] - expected);
		double rel_err = diff / fabs(expected);

		if (rel_err > eps) {
			printf("Error at index %d: %.8f vs %.8f\n", i, h_C[i], expected);
			correct = false;
			break;
		}
	}

	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

// === Main ===
int main(int argc, char **argv) {
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	if (checkCmdLineFlag(argc, argv, "help") || checkCmdLineFlag(argc, argv, "?")) {
		printf("Usage: ./main [-wA=WidthA] [-hA=HeightA] [-wB=WidthB] [-hB=HeightB]\n");
		exit(EXIT_SUCCESS);
	}

	findCudaDevice();

	int block_size = 32;
	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

	if (checkCmdLineFlag(argc, argv, "wA")) dimsA.x = getCmdLineArgumentInt(argc, argv, "wA");
	if (checkCmdLineFlag(argc, argv, "hA")) dimsA.y = getCmdLineArgumentInt(argc, argv, "hA");
	if (checkCmdLineFlag(argc, argv, "wB")) dimsB.x = getCmdLineArgumentInt(argc, argv, "wB");
	if (checkCmdLineFlag(argc, argv, "hB")) dimsB.y = getCmdLineArgumentInt(argc, argv, "hB");

	if (dimsA.x != dimsB.y) {
		printf("Error: Matrix dimensions mismatch (%d != %d)\n", dimsA.x, dimsB.y);
		return EXIT_FAILURE;
	}

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

	checkCudaErrors(cudaProfilerStart());
	int result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
	checkCudaErrors(cudaProfilerStop());

	return result;
}
