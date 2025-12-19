#include <torch/extension.h>

#include <reg23/Similarity.h>

namespace reg23 {

__global__ void Kernel_NormalisedCrossCorrelation_CUDA(long numel, const float *a, const float *b,
                                                       double *blockSumsArray) {
	extern __shared__ double buffer[];

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	const long index = threadIdx.x * 5;
	if (i >= numel) {
		buffer[index] = 0.0;
		buffer[index + 1] = 0.0;
		buffer[index + 2] = 0.0;
		buffer[index + 3] = 0.0;
		buffer[index + 4] = 0.0;
		return;
	}

	const double ai = static_cast<double>(a[i]);
	const double bi = static_cast<double>(b[i]);

	buffer[index] = ai;
	buffer[index + 1] = bi;
	buffer[index + 2] = ai * ai;
	buffer[index + 3] = bi * bi;
	buffer[index + 4] = ai * bi;

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

int blockSizeToDynamicSMemSize_NormalisedCrossCorrelation_CUDA(int blockSize) {
	return 5 * blockSize * static_cast<int>(sizeof(double));
}

__host__ std::tuple<at::Tensor, double, double, double, double, double> NormalisedCrossCorrelation_CUDA(
	const at::Tensor &a, const at::Tensor &b) {
	Similarity::Common(a, b, at::DeviceType::CUDA);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const at::Tensor bContiguous = b.contiguous();
	const float *bPtr = bContiguous.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_NormalisedCrossCorrelation_CUDA,
	                                               &blockSizeToDynamicSMemSize_NormalisedCrossCorrelation_CUDA, 0);

	const size_t bufferSize = blockSizeToDynamicSMemSize_NormalisedCrossCorrelation_CUDA(blockSize);
	const int gridSize = (static_cast<int>(a.numel()) + blockSize - 1) / blockSize;

	// stores the sums for each kernel block of a, b, a^2, b^2 and a*b (which is why the last dimension is 5)
	const at::Tensor blockSums = torch::zeros(at::IntArrayRef({gridSize, 5}),
											  torch::TensorOptions{}.dtype(torch::kDouble).device(a.device()));
	double *blockSumsPtr = blockSums.data_ptr<double>();

	Kernel_NormalisedCrossCorrelation_CUDA<<<gridSize, blockSize, bufferSize>>>(a.numel(), aPtr, bPtr, blockSumsPtr);

	const double nF = static_cast<double>(a.numel());

	const at::Tensor sums = blockSums.sum({0});

	const double numerator = nF * sums[4].item<double>() - sums[0].item<double>() * sums[1].item<double>();
	const double denominatorLeft = sqrt(nF * sums[2].item<double>() - sums[0].item<double>() * sums[0].item<double>());
	const double denominatorRight = sqrt(nF * sums[3].item<double>() - sums[1].item<double>() * sums[1].item<double>());

	const at::Tensor zncc = torch::tensor(numerator / (denominatorLeft * denominatorRight + 1e-10), a.options());

	return {zncc, sums[0].item<double>(), sums[1].item<double>(), numerator, denominatorLeft, denominatorRight};
}

} // namespace reg23