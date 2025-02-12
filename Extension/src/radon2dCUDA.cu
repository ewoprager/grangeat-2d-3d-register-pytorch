#include <torch/extension.h>

#include "../include/Texture2DCUDA.h"

namespace ExtensionTest {

__global__ void radon2d_kernel(Texture2DCUDA textureIn, long numelOut, float *arrayOut, Linear mappingIToOffset,
                               const float *phiValues, const float *rValues, long samplesPerLine, float scaleFactor) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(textureIn, phiValues[threadIndex],
	                                                                    rValues[threadIndex], mappingIToOffset);
	arrayOut[threadIndex] = scaleFactor * Radon2D<Texture2DCUDA>::IntegrateLooped(
		                        textureIn, indexMappings, samplesPerLine);
}

__host__ at::Tensor radon2d_cuda(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                                 const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCUDA texture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float lineLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const auto mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);

	constexpr int blockSize = 256;
	const int gridSize = (static_cast<int>(numelOut) + blockSize - 1) / blockSize;
	radon2d_kernel<<<gridSize, blockSize>>>(std::move(texture), numelOut, resultFlatPtr, mappingIToOffset, phiFlatPtr,
	                                        rFlatPtr, samplesPerLine, scaleFactor);
	return resultFlat.view(phiValues.sizes());
}

__global__ void dRadon2dDR_kernel(Texture2DCUDA textureIn, long numelOut, float *arrayOut, Linear mappingIToOffset,
                                  const float *phiValues, const float *rValues, long samplesPerLine,
                                  float scaleFactor) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(textureIn, phiValues[threadIndex],
	                                                                    rValues[threadIndex], mappingIToOffset);
	const auto derivativeWRTR = Radon2D<Texture2DCUDA>::GetDerivativeWRTR(
		textureIn, phiValues[threadIndex], rValues[threadIndex]);
	arrayOut[threadIndex] = scaleFactor * Radon2D<Texture2DCUDA>::DIntegrateLoopedDMappingParameter(
		                        textureIn, indexMappings, derivativeWRTR, samplesPerLine);
}

at::Tensor dRadon2dDR_cuda(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                           const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCUDA texture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const at::Tensor phiFlatContiguous = phiValues.contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float lineLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const auto mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);

	constexpr int blockSize = 256;
	const int gridSize = (static_cast<int>(numelOut) + blockSize - 1) / blockSize;
	dRadon2dDR_kernel<<<gridSize, blockSize>>>(std::move(texture), numelOut, resultFlatPtr, mappingIToOffset,
	                                           phiFlatPtr, rFlatPtr, samplesPerLine, scaleFactor);
	return resultFlat.view(phiValues.sizes());
}

struct Radon2DV2Consts {
	cudaTextureObject_t textureHandle{};
	long samplesPerLine{};
	float scaleFactor{};
	float *patchSumsArray{};
};

__device__ __constant__ Radon2DV2Consts radon2DV2Consts{};

__global__ void radon2d_v2_kernel(const Radon2D<Texture2DCUDA>::IndexMappings indexMappings) {
	extern __shared__ float buffer[];

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= radon2DV2Consts.samplesPerLine) {
		buffer[threadIdx.x] = 0.f;
		return;
	}

	const float iF = static_cast<float>(i);

	buffer[threadIdx.x] = tex2D<float>(radon2DV2Consts.textureHandle, indexMappings.mappingIToX(iF),
	                                   indexMappings.mappingIToY(iF));

	__syncthreads();

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff) {
			buffer[threadIdx.x] += buffer[threadIdx.x + cutoff];
		}

		__syncthreads();
	}
	if (threadIdx.x == 0) radon2DV2Consts.patchSumsArray[blockIdx.x] = radon2DV2Consts.scaleFactor * buffer[0];
}

__host__ at::Tensor radon2d_v2_cuda(const at::Tensor &image, double xSpacing, double ySpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCUDA texture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());

	constexpr int blockSize = 256;
	constexpr size_t bufferSize = blockSize * sizeof(float);
	const int gridSize = (static_cast<int>(samplesPerLine) + blockSize - 1) / blockSize;
	const at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize}), resultFlat.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	const float lineLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	const auto mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(lineLength, samplesPerLine);

	Radon2DV2Consts constants{texture.GetHandle(), samplesPerLine, scaleFactor, patchSumsPtr};
	CudaMemcpyToObjectSymbol(radon2DV2Consts, constants);

	for (long i = 0; i < numelOut; ++i) {
		const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(
			texture, phiFlat[i].item().toFloat(), rFlat[i].item().toFloat(), mappingIToOffset);

		radon2d_v2_kernel<<<gridSize, blockSize, bufferSize>>>(indexMappings);

		resultFlat.index_put_({i}, patchSums.sum());
	}

	return resultFlat.view(phiValues.sizes());
}

} // namespace ExtensionTest