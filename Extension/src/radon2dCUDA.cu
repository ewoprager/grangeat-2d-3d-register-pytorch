#include <torch/extension.h>

#include "../include/Texture2DCUDA.h"

namespace ExtensionTest {

__global__ void radon2d_kernel(Texture2DCUDA textureIn, long heightOut, long widthOut, float *arrayOut,
                               Linear mappingIToOffset, const float *phiValues, const float *rValues,
                               long samplesPerLine, float scaleFactor) {
	const long col = blockIdx.x * blockDim.x + threadIdx.x;
	const long row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= widthOut || row >= heightOut) return;
	const long index = row * widthOut + col;
	const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(textureIn, phiValues[row], rValues[col],
	                                                                    mappingIToOffset);
	arrayOut[index] = scaleFactor * Radon2D<Texture2DCUDA>::IntegrateLooped(textureIn, indexMappings, samplesPerLine);
}

__host__ at::Tensor radon2d_cuda(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                                 const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// phiValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// sValues should be a 1D array of floats on the GPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCUDA texture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const at::Tensor phisContiguous = phiValues.contiguous();
	const float *phisPtr = phisContiguous.data_ptr<float>();
	const at::Tensor rsContiguous = rValues.contiguous();
	const float *rsPtr = rsContiguous.data_ptr<float>();

	const long heightOut = phiValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float lineLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());

	const auto mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(lineLength, samplesPerLine);

	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);

	const dim3 blockSize{16, 16};
	const dim3 gridSize{(static_cast<unsigned>(widthOut) + blockSize.x - 1) / blockSize.x,
	                    (static_cast<unsigned>(heightOut) + blockSize.y - 1) / blockSize.y};
	radon2d_kernel<<<gridSize, blockSize>>>(std::move(texture), heightOut, widthOut, resultPtr, mappingIToOffset,
	                                        phisPtr, rsPtr, samplesPerLine, scaleFactor);

	return result;
}

__global__ void dRadon2dDR_kernel(Texture2DCUDA textureIn, long heightOut, long widthOut, float *arrayOut,
                                  Linear mappingIToOffset, const float *phiValues, const float *rValues,
                                  long samplesPerLine, float scaleFactor) {
	const long col = blockIdx.x * blockDim.x + threadIdx.x;
	const long row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= widthOut || row >= heightOut) return;
	const long index = row * widthOut + col;
	const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(textureIn, phiValues[row], rValues[col],
	                                                                    mappingIToOffset);
	const auto dIndexMappingsDR = Radon2D<Texture2DCUDA>::GetDIndexMappingsDR(textureIn, phiValues[row],
	                                                                          mappingIToOffset);
	arrayOut[index] = scaleFactor * Radon2D<Texture2DCUDA>::DIntegrateLoopedDMappingParameter(
		                  textureIn, indexMappings, dIndexMappingsDR, samplesPerLine);
}

at::Tensor dRadon2dDR_cuda(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                           const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// phiValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// sValues should be a 1D array of floats on the GPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCUDA texture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const at::Tensor phisContiguous = phiValues.contiguous();
	const float *phisPtr = phisContiguous.data_ptr<float>();
	const at::Tensor rsContiguous = rValues.contiguous();
	const float *rsPtr = rsContiguous.data_ptr<float>();

	const long heightOut = phiValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float lineLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const auto mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);

	const dim3 blockSize{16, 16};
	const dim3 gridSize{(static_cast<unsigned>(widthOut) + blockSize.x - 1) / blockSize.x,
	                    (static_cast<unsigned>(heightOut) + blockSize.y - 1) / blockSize.y};
	dRadon2dDR_kernel<<<gridSize, blockSize>>>(std::move(texture), heightOut, widthOut, resultPtr, mappingIToOffset,
	                                           phisPtr, rsPtr, samplesPerLine, scaleFactor);

	return result;
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
	// phiValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// sValues should be a 1D array of floats on the GPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCUDA texture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const long heightOut = phiValues.sizes()[0];
	const long widthOut = phiValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());

	constexpr unsigned blockSize = 256;
	constexpr size_t bufferSize = blockSize * sizeof(float);
	const unsigned gridSize = (samplesPerLine + blockSize - 1) / blockSize;
	at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize}), result.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	const float lineLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	const auto mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(lineLength, samplesPerLine);

	Radon2DV2Consts constants{texture.GetHandle(), samplesPerLine, scaleFactor, patchSumsPtr};
	cudaMemcpyToSymbol(radon2DV2Consts, &constants, sizeof(Radon2DV2Consts));

	for (long row = 0; row < heightOut; ++row) {
		for (long col = 0; col < widthOut; ++col) {
			const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(
				texture, phiValues[row].item().toFloat(), rValues[col].item().toFloat(), mappingIToOffset);

			radon2d_v2_kernel<<<gridSize, blockSize, bufferSize>>>(indexMappings);

			result.index_put_({row, col}, patchSums.sum());
		}
	}

	return result;
}

} // namespace ExtensionTest