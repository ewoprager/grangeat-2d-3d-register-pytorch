#include <torch/extension.h>

#include "../include/Texture2DCUDA.h"
#include "../include/Radon2D.h"

namespace ExtensionTest {

using CommonData = Radon2D<Texture2DCUDA>::CommonData;

__global__ void Kernel_Radon2D_CUDA(Texture2DCUDA textureIn, int64_t numelOut, float *arrayOut,
                                    Linear<Vec<double, 2> > mappingIndexToOffset, const float *phiValues,
                                    const float *rValues, int64_t samplesPerLine, float scaleFactor) {
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCUDA>::GetMappingIndexToTexCoord(
		textureIn, phiValues[threadIndex], rValues[threadIndex], mappingIndexToOffset);
	arrayOut[threadIndex] = scaleFactor * Radon2D<Texture2DCUDA>::IntegrateLooped(
		                        textureIn, mappingIndexToTexCoord, samplesPerLine);
}

__host__ at::Tensor Radon2D_CUDA(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &rValues, int64_t samplesPerLine) {
	CommonData common = Radon2D<Texture2DCUDA>::Common(image, imageSpacing, phiValues, rValues, samplesPerLine,
	                                                   at::DeviceType::CUDA);

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_Radon2D_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(common.flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_Radon2D_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture), common.flatOutput.numel(),
	                                             resultFlatPtr, common.mappingIndexToOffset, phiFlatPtr, rFlatPtr,
	                                             samplesPerLine, common.scaleFactor);
	return common.flatOutput.view(phiValues.sizes());
}

__global__ void Kernel_DRadon2DDR_CUDA(Texture2DCUDA textureIn, int64_t numelOut, float *arrayOut,
                                       Linear<Vec<double, 2> > mappingIndexToOffset, const float *phiValues,
                                       const float *rValues, int64_t samplesPerLine, float scaleFactor) {
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCUDA>::GetMappingIndexToTexCoord(
		textureIn, phiValues[threadIndex], rValues[threadIndex], mappingIndexToOffset);
	const Vec<double, 2> dTexCoordDR = Radon2D<Texture2DCUDA>::GetDTexCoordDR(textureIn, phiValues[threadIndex],
	                                                                          rValues[threadIndex]);
	arrayOut[threadIndex] = scaleFactor * Radon2D<Texture2DCUDA>::DIntegrateLoopedDMappingParameter(
		                        textureIn, mappingIndexToTexCoord, dTexCoordDR, samplesPerLine);
}

at::Tensor DRadon2DDR_CUDA(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                           const at::Tensor &rValues, int64_t samplesPerLine) {
	CommonData common = Radon2D<Texture2DCUDA>::Common(image, imageSpacing, phiValues, rValues, samplesPerLine,
	                                                   at::DeviceType::CUDA);

	const at::Tensor phiFlatContiguous = phiValues.contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_DRadon2DDR_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(common.flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_DRadon2DDR_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture), common.flatOutput.numel(),
	                                                resultFlatPtr, common.mappingIndexToOffset, phiFlatPtr, rFlatPtr,
	                                                samplesPerLine, common.scaleFactor);
	return common.flatOutput.view(phiValues.sizes());
}

struct Radon2DV2Consts {
	cudaTextureObject_t textureHandle{};
	int64_t samplesPerLine{};
	double scaleFactor{};
	float *patchSumsArray{};
};

__device__ __constant__ Radon2DV2Consts radon2DV2Consts{};

__global__ void Kernel_Radon2D_CUDA_V2(const Linear<Vec<double, 2> > mappingIndexToTexCoord) {
	extern __shared__ float buffer[];

	const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= radon2DV2Consts.samplesPerLine) {
		buffer[threadIdx.x] = 0.f;
		return;
	}

	const Vec<double, 2> texCoord = mappingIndexToTexCoord(Vec<double, 2>::Full(static_cast<double>(i)));
	buffer[threadIdx.x] = tex2D<float>(radon2DV2Consts.textureHandle, texCoord.X(), texCoord.Y());

	__syncthreads();

	for (int64_t cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff) {
			buffer[threadIdx.x] += buffer[threadIdx.x + cutoff];
		}

		__syncthreads();
	}
	if (threadIdx.x == 0) radon2DV2Consts.patchSumsArray[blockIdx.x] = radon2DV2Consts.scaleFactor * buffer[0];
}

int blockSizeToDynamicSMemSize_Radon2D_CUDA_V2(int blockSize) {
	return blockSize * sizeof(float);
}

__host__ at::Tensor Radon2D_CUDA_V2(const at::Tensor &image, const at::Tensor &imageSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &rValues, int64_t samplesPerLine) {
	CommonData common = Radon2D<Texture2DCUDA>::Common(image, imageSpacing, phiValues, rValues, samplesPerLine,
	                                                   at::DeviceType::CUDA);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_Radon2D_CUDA_V2,
	                                               &blockSizeToDynamicSMemSize_Radon2D_CUDA_V2, 0);
	const size_t bufferSize = blockSizeToDynamicSMemSize_Radon2D_CUDA_V2(blockSize);
	const int gridSize = (static_cast<int>(samplesPerLine) + blockSize - 1) / blockSize;
	const at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize}), common.flatOutput.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	Radon2DV2Consts constants = {common.inputTexture.GetHandle(), samplesPerLine, common.scaleFactor, patchSumsPtr};
	CudaMemcpyToObjectSymbol(radon2DV2Consts, constants);

	for (int64_t i = 0; i < common.flatOutput.numel(); ++i) {
		const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCUDA>::GetMappingIndexToTexCoord(
			common.inputTexture, phiFlat[i].item().toFloat(), rFlat[i].item().toFloat(), common.mappingIndexToOffset);

		Kernel_Radon2D_CUDA_V2<<<gridSize, blockSize, bufferSize>>>(mappingIndexToTexCoord);

		common.flatOutput.index_put_({i}, patchSums.sum());
	}

	return common.flatOutput.view(phiValues.sizes());
}

} // namespace ExtensionTest