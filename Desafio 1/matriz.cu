#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// Maneira complexa da mat_mul_device
template <int BLOCK_SIZE>
__global__ void mat_mul_device(float *C, const float *A, const float *B, const int N) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	float Csub = 0;

	for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		int aRow = row, aCol = m * BLOCK_SIZE + tx;
		int bRow = m * BLOCK_SIZE + ty, bCol = col;

		As[ty][tx] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;
		Bs[ty][tx] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];

		__syncthreads();
	}

	if (row < N && col < N)
		C[row * N + col] = Csub;
}


// Maneira complexa da mat_mul_host
template <int BLOCK_SIZE>
void mat_mul_host(float* h_C, const float* h_A, const float* h_B, const int N) {
	// Alocar matrizes no device (GPU)
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, N * N * sizeof(float));
	cudaMalloc(&d_B, N * N * sizeof(float));
	cudaMalloc(&d_C, N * N * sizeof(float));

	// Copiar dados para o device
	cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

	// Definir tamanho de bloco e grid
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// Lançar kernel
	mat_mul_device<BLOCK_SIZE><<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, N);
	cudaDeviceSynchronize();

	// Copiar resultado de volta
	cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	// Liberar memória
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main() {
	// Ler entrada N
	int N;
	std::cin >> N;

	// Alocar matrizes no host (CPU)
	float* h_A = new float[N * N];
	float* h_B = new float[N * N];
	float* h_C = new float[N * N];

	// Ler entrada matriz A e matriz B
	for (int i = 0; i < N * N; ++i)
		std::cin >> h_A[i];
	for (int i = 0; i < N * N; ++i)
		std::cin >> h_B[i];

	// Calcular a multiplicação
	mat_mul_host<32>(h_C, h_A, h_B, N);

	// Imprimir resultado
	std::cout << std::fixed << std::setprecision(2);
	for (int y = 0; y < N; ++y) {
		for (int x = 0; x < N; ++x) {
			std::cout << h_C[y * N + x] << " ";
		}
		std::cout << std::endl;
	}

	// Liberar memória
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
}