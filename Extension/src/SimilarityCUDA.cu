#include <torch/extension.h>

#include "../include/Similarity.h"

namespace ExtensionTest {

__global__ void Kernel_NormalisedCrossCorrelation_CUDA(long numel, const float *a, const float *b,
                                                       float *blockSumsArray) {
	extern __shared__ float buffer[];

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	const long index = threadIdx.x * 5;
	if (i >= numel) {
		buffer[index] = 0.f;
		buffer[index + 1] = 0.f;
		buffer[index + 2] = 0.f;
		buffer[index + 3] = 0.f;
		buffer[index + 4] = 0.f;
		return;
	}

	buffer[index] = a[i];
	buffer[index + 1] = b[i];
	buffer[index + 2] = a[i] * a[i];
	buffer[index + 3] = b[i] * b[i];
	buffer[index + 4] = a[i] * b[i];

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff) {
			const long sumWith = index + cutoff * 5;
			buffer[index] += buffer[sumWith];
			buffer[index + 1] += buffer[sumWith + 1];
			buffer[index + 2] += buffer[sumWith + 2];
			buffer[index + 3] += buffer[sumWith + 3];
			buffer[index + 4] += buffer[sumWith + 4];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		blockSumsArray[5 * blockIdx.x] = buffer[0];
		blockSumsArray[5 * blockIdx.x + 1] = buffer[1];
		blockSumsArray[5 * blockIdx.x + 2] = buffer[2];
		blockSumsArray[5 * blockIdx.x + 3] = buffer[3];
		blockSumsArray[5 * blockIdx.x + 4] = buffer[4];
	}
}

int blockSizeToDynamicSMemSize(int blockSize) {
	return 5 * blockSize * sizeof(float);
}

__host__ at::Tensor NormalisedCrossCorrelation_CUDA(const at::Tensor &a, const at::Tensor &b) {
	Similarity::Common(a, b, at::DeviceType::CUDA);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const at::Tensor bContiguous = b.contiguous();
	const float *bPtr = bContiguous.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_NormalisedCrossCorrelation_CUDA,
	                                               &blockSizeToDynamicSMemSize, 0);

	const size_t bufferSize = blockSizeToDynamicSMemSize(blockSize);
	const int gridSize = (static_cast<int>(a.numel()) + blockSize - 1) / blockSize;

	// stores the sums for each kernel block of a, b, a^2, b^2 and a*b (which is why the last dimension is 5)
	const at::Tensor blockSums = torch::zeros(at::IntArrayRef({gridSize, 5}), aContiguous.options());
	float *blockSumsPtr = blockSums.data_ptr<float>();

	Kernel_NormalisedCrossCorrelation_CUDA<<<gridSize, blockSize, bufferSize>>>(a.numel(), aPtr, bPtr, blockSumsPtr);

	const float nF = static_cast<float>(a.numel());

	const at::Tensor sums = blockSums.sum({0});

	return (nF * sums[4] - sums[0] * sums[1]) / ((nF * sums[2] - sums[0].square()).sqrt() * (
		                                             nF * sums[3] - sums[1].square()).sqrt() + 1e-10f);
}

} // namespace ExtensionTest