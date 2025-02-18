#include <torch/extension.h>

#include "../include/Texture2DCUDA.h"
#include "../include/Radon2D.h"

namespace ExtensionTest {

__global__ void radon2d_kernel(Texture2DCUDA textureIn, long numelOut, float *arrayOut,
                               Linear<Vec<double, 2> > mappingIToOffset, const float *phiValues, const float *rValues,
                               long samplesPerLine, float scaleFactor) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCUDA>::GetMappingIndexToTexCoord(
		textureIn, phiValues[threadIndex], rValues[threadIndex], mappingIToOffset);
	arrayOut[threadIndex] = scaleFactor * Radon2D<Texture2DCUDA>::IntegrateLooped(
		                        textureIn, mappingIndexToTexCoord, samplesPerLine);
}

__host__ at::Tensor radon2d_cuda(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// image spacing should be a 1D tensor of 2 floats on the GPU
	TORCH_CHECK(imageSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(imageSpacing.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(imageSpacing.device().type() == at::DeviceType::CUDA);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	Texture2DCUDA texture = Texture2DCUDA::FromTensor(image, imageSpacing.cpu());

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), image.contiguous().options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float lineLength = sqrtf(texture.SizeWorld().Apply<double>(&Square<double>).Sum());
	const Linear<Vec<double, 2> > mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(
		lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);

	constexpr int blockSize = 256;
	const int gridSize = (static_cast<int>(numelOut) + blockSize - 1) / blockSize;
	radon2d_kernel<<<gridSize, blockSize>>>(std::move(texture), numelOut, resultFlatPtr, mappingIToOffset, phiFlatPtr,
	                                        rFlatPtr, samplesPerLine, scaleFactor);
	return resultFlat.view(phiValues.sizes());
}

__global__ void dRadon2dDR_kernel(Texture2DCUDA textureIn, long numelOut, float *arrayOut,
                                  Linear<Vec<double, 2> > mappingIToOffset, const float *phiValues,
                                  const float *rValues, long samplesPerLine, float scaleFactor) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCUDA>::GetMappingIndexToTexCoord(
		textureIn, phiValues[threadIndex], rValues[threadIndex], mappingIToOffset);
	const Vec<double, 2> dTexCoordDR = Radon2D<Texture2DCUDA>::GetDTexCoordDR(textureIn, phiValues[threadIndex],
	                                                                          rValues[threadIndex]);
	arrayOut[threadIndex] = scaleFactor * Radon2D<Texture2DCUDA>::DIntegrateLoopedDMappingParameter(
		                        textureIn, mappingIndexToTexCoord, dTexCoordDR, samplesPerLine);
}

at::Tensor dRadon2dDR_cuda(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                           const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// image spacing should be a 1D tensor of 2 floats on the GPU
	TORCH_CHECK(imageSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(imageSpacing.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(imageSpacing.device().type() == at::DeviceType::CUDA);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	Texture2DCUDA texture = Texture2DCUDA::FromTensor(image, imageSpacing.cpu());

	const at::Tensor phiFlatContiguous = phiValues.contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), image.contiguous().options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float lineLength = sqrtf(texture.SizeWorld().Apply<double>(&Square<double>).Sum());
	const Linear<Vec<double, 2> > mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(
		lineLength, samplesPerLine);
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

__global__ void radon2d_v2_kernel(const Linear<Vec<double, 2> > mappingIndexToTexCoord) {
	extern __shared__ float buffer[];

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= radon2DV2Consts.samplesPerLine) {
		buffer[threadIdx.x] = 0.f;
		return;
	}

	const Vec<double, 2> texCoord = mappingIndexToTexCoord(Vec<double, 2>::Full(static_cast<double>(i)));
	buffer[threadIdx.x] = tex2D<float>(radon2DV2Consts.textureHandle, texCoord.X(), texCoord.Y());

	__syncthreads();

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff) {
			buffer[threadIdx.x] += buffer[threadIdx.x + cutoff];
		}

		__syncthreads();
	}
	if (threadIdx.x == 0) radon2DV2Consts.patchSumsArray[blockIdx.x] = radon2DV2Consts.scaleFactor * buffer[0];
}

__host__ at::Tensor radon2d_v2_cuda(const at::Tensor &image, const at::Tensor &imageSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the GPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CUDA);
	// image spacing should be a 1D tensor of 2 floats on the GPU
	TORCH_CHECK(imageSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(imageSpacing.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(imageSpacing.device().type() == at::DeviceType::CUDA);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const Texture2DCUDA texture = Texture2DCUDA::FromTensor(image, imageSpacing.cpu());

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), image.contiguous().options());

	constexpr int blockSize = 256;
	constexpr size_t bufferSize = blockSize * sizeof(float);
	const int gridSize = (static_cast<int>(samplesPerLine) + blockSize - 1) / blockSize;
	const at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize}), resultFlat.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	const float lineLength = sqrtf(texture.SizeWorld().Apply<double>(&Square<double>).Sum());
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	const Linear<Vec<double, 2> > mappingIToOffset = Radon2D<Texture2DCUDA>::GetMappingIToOffset(
		lineLength, samplesPerLine);

	Radon2DV2Consts constants{texture.GetHandle(), samplesPerLine, scaleFactor, patchSumsPtr};
	CudaMemcpyToObjectSymbol(radon2DV2Consts, constants);

	for (long i = 0; i < numelOut; ++i) {
		const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCUDA>::GetMappingIndexToTexCoord(
			texture, phiFlat[i].item().toFloat(), rFlat[i].item().toFloat(), mappingIToOffset);

		radon2d_v2_kernel<<<gridSize, blockSize, bufferSize>>>(mappingIndexToTexCoord);

		resultFlat.index_put_({i}, patchSums.sum());
	}

	return resultFlat.view(phiValues.sizes());
}

} // namespace ExtensionTest