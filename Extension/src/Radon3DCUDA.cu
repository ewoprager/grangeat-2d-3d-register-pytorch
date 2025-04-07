#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"
#include "../include/Radon3D.h"

namespace ExtensionTest {

using CommonData = Radon3D<Texture3DCUDA>::CommonData;

__global__ void Kernel_Radon3D_CUDA(Texture3DCUDA textureIn, int64_t numelOut, float *arrayOut,
                                    Linear<Vec<double, 3> > mappingIToOffset, const float *phiValues,
                                    const float *thetaValues, const float *rValues, int64_t samplesPerDirection,
                                    float scaleFactor) {
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCUDA>::GetMappingIndexToTexCoord(
		textureIn, phiValues[threadIndex], thetaValues[threadIndex], rValues[threadIndex], mappingIToOffset);
	arrayOut[threadIndex] = scaleFactor * Radon3D<Texture3DCUDA>::IntegrateLooped(
		                        textureIn, mappingIndexToTexCoord, samplesPerDirection);
}

__host__ at::Tensor Radon3D_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                                 const at::Tensor &thetaValues, const at::Tensor &rValues,
                                 int64_t samplesPerDirection) {
	CommonData common = Radon3D<Texture3DCUDA>::Common(volume, volumeSpacing, phiValues, thetaValues, rValues,
	                                                   samplesPerDirection, at::DeviceType::CUDA);

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor thetaFlatContiguous = thetaValues.flatten().contiguous();
	const float *thetaFlatPtr = thetaFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_Radon3D_CUDA, 0, 0);
	const int gridSize = (static_cast<unsigned>(common.flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_Radon3D_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture), common.flatOutput.numel(),
	                                             resultFlatPtr, common.mappingIndexToOffset, phiFlatPtr, thetaFlatPtr,
	                                             rFlatPtr, samplesPerDirection, common.scaleFactor);
	return common.flatOutput.view(phiValues.sizes());
}

__global__ void Kernel_DRadon3DDR_CUDA(Texture3DCUDA textureIn, int64_t numelOut, float *arrayOut,
                                       Linear<Vec<double, 3> > mappingIToOffset, const float *phiValues,
                                       const float *thetaValues, const float *rValues, int64_t samplesPerDirection,
                                       float scaleFactor) {
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const float phi = phiValues[threadIndex];
	const float theta = thetaValues[threadIndex];
	const float r = rValues[threadIndex];
	const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCUDA>::GetMappingIndexToTexCoord(
		textureIn, phi, theta, r, mappingIToOffset);
	const Vec<double, 3> dTexCoordDR = Radon3D<Texture3DCUDA>::GetDTexCoordDR(textureIn, phi, theta, r);
	arrayOut[threadIndex] = scaleFactor * Radon3D<Texture3DCUDA>::DIntegrateLoopedDMappingParameter(
		                        textureIn, mappingIndexToTexCoord, dTexCoordDR, samplesPerDirection);
}

__host__ at::Tensor DRadon3DDR_CUDA(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, int64_t samplesPerDirection) {
	CommonData common = Radon3D<Texture3DCUDA>::Common(volume, volumeSpacing, phiValues, thetaValues, rValues,
	                                                   samplesPerDirection, at::DeviceType::CUDA);

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor thetaFlatContiguous = thetaValues.flatten().contiguous();
	const float *thetaFlatPtr = thetaFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_DRadon3DDR_CUDA, 0, 0);
	const int gridSize = (static_cast<unsigned>(common.flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_DRadon3DDR_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture), common.flatOutput.numel(),
	                                                resultFlatPtr, common.mappingIndexToOffset, phiFlatPtr,
	                                                thetaFlatPtr, rFlatPtr, samplesPerDirection, common.scaleFactor);
	return common.flatOutput.view(phiValues.sizes());
}

struct Radon3DV2Consts {
	cudaTextureObject_t textureHandle{};
	int64_t samplesPerDirection{};
	double scaleFactor{};
	float *patchSumsArray{};
};

__device__ __constant__ Radon3DV2Consts radon3DV2Consts{};

__global__ void Kernel_Radon3D_CUDA_V2(Linear2<Vec<double, 3> > mappingIndexToTexCoord) {
	extern __shared__ float buffer[];

	const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const int64_t j = blockIdx.y * blockDim.y + threadIdx.y;
	const int64_t localIndex = threadIdx.y * blockDim.x + threadIdx.x;
	if (i >= radon3DV2Consts.samplesPerDirection || j >= radon3DV2Consts.samplesPerDirection) {
		buffer[localIndex] = 0.f;
		return;
	}

	const Vec<double, 3> texCoord = mappingIndexToTexCoord(Vec<double, 3>::Full(static_cast<double>(i)),
	                                                       Vec<double, 3>::Full(static_cast<double>(j)));
	buffer[localIndex] = tex3D<float>(radon3DV2Consts.textureHandle, texCoord.X(), texCoord.Y(), texCoord.Z());

	__syncthreads();

	for (int64_t cutoff = (blockDim.x * blockDim.y) / 2; cutoff > 0; cutoff /= 2) {
		if (localIndex < cutoff) {
			buffer[localIndex] += buffer[localIndex + cutoff];
		}

		__syncthreads();
	}
	if (localIndex == 0) {
		radon3DV2Consts.patchSumsArray[blockIdx.y * gridDim.x + blockIdx.x] = radon3DV2Consts.scaleFactor * buffer[0];
	}
}

__global__ void Kernel_Radon3D_CUDA_V3(Linear2<Vec<double, 3> > mappingIndexToTexCoord) {
	extern __shared__ float buffer[];

	// REQUIRED: blockDim.x must be equal to blockDim.y

	const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const int64_t j = blockIdx.y * blockDim.y + threadIdx.y;
	// const int64_t localIndex = threadIdx.y * blockDim.x + threadIdx.x;
	if (i >= radon3DV2Consts.samplesPerDirection || j >= radon3DV2Consts.samplesPerDirection) {
		buffer[threadIdx.y * blockDim.x + threadIdx.x] = 0.f;
		return;
	}

	const Vec<double, 3> texCoord = mappingIndexToTexCoord(Vec<double, 3>::Full(static_cast<double>(i)),
	                                                       Vec<double, 3>::Full(static_cast<double>(j)));
	buffer[threadIdx.y * blockDim.x + threadIdx.x] = tex3D<float>(radon3DV2Consts.textureHandle, texCoord.X(),
	                                                              texCoord.Y(), texCoord.Z());

	__syncthreads();

	for (int64_t cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff && threadIdx.y < cutoff) {
			buffer[threadIdx.y * blockDim.x + threadIdx.x] += buffer[threadIdx.y * blockDim.x + threadIdx.x + cutoff] +
				buffer[(threadIdx.y + cutoff) * blockDim.x + threadIdx.x] + buffer[
					(threadIdx.y + cutoff) * blockDim.x + threadIdx.x + cutoff];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0 && threadIdx.y) {
		radon3DV2Consts.patchSumsArray[blockIdx.y * gridDim.x + blockIdx.x] = radon3DV2Consts.scaleFactor * buffer[0];
	}
}

int blockSizeToDynamicSMemSize_Radon3D_CUDA_V2(int blockSize) {
	return blockSize * sizeof(float);
}

__host__ at::Tensor Radon3D_CUDA_V2(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, int64_t samplesPerDirection) {
	CommonData common = Radon3D<Texture3DCUDA>::Common(volume, volumeSpacing, phiValues, thetaValues, rValues,
	                                                   samplesPerDirection, at::DeviceType::CUDA);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_Radon3D_CUDA_V3,
	                                               &blockSizeToDynamicSMemSize_Radon3D_CUDA_V2, 0);
	const unsigned blockWidth = static_cast<unsigned>(sqrtf(static_cast<float>(blockSize)));
	const dim3 blockDims = {blockWidth, blockSize / blockWidth};
	const size_t bufferSize = blockSizeToDynamicSMemSize_Radon3D_CUDA_V2(blockDims.x * blockDims.y);
	const dim3 gridSize = {(static_cast<unsigned>(samplesPerDirection) + blockDims.x - 1) / blockDims.x,
	                       (static_cast<unsigned>(samplesPerDirection) + blockDims.y - 1) / blockDims.y};
	const at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize.y, gridSize.x}), common.flatOutput.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	Radon3DV2Consts constants{common.inputTexture.GetHandle(), samplesPerDirection, common.scaleFactor, patchSumsPtr};
	CudaMemcpyToObjectSymbol(radon3DV2Consts, constants);

	for (int64_t i = 0; i < common.flatOutput.numel(); ++i) {
		const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCUDA>::GetMappingIndexToTexCoord(
			common.inputTexture, phiFlat[i].item().toFloat(), thetaFlat[i].item().toFloat(), rFlat[i].item().toFloat(),
			common.mappingIndexToOffset);

		Kernel_Radon3D_CUDA_V3<<<gridSize, blockDims, bufferSize>>>(mappingIndexToTexCoord);

		common.flatOutput.index_put_({i}, patchSums.sum());
	}
	return common.flatOutput.view(phiValues.sizes());
}

struct DRadon3DDRV2Consts {
	cudaTextureObject_t textureHandle{};
	int64_t samplesPerDirection{};
	double scaleFactor{};
	float *patchSumsArray{};
	int64_t volumeWidth{};
	int64_t volumeHeight{};
	int64_t volumeDepth{};
};

__device__ __constant__ DRadon3DDRV2Consts dRadon3DDRV2Consts{};

__global__ void Kernel_DRadon3DDR_CUDA_V2(Linear2<Vec<double, 3> > mappingIndexToTexCoord, Vec<double, 3> dTexCoordDR) {
	extern __shared__ float buffer[];

	// REQUIRED: blockDim.x must be equal to blockDim.y

	const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const int64_t j = blockIdx.y * blockDim.y + threadIdx.y;
	// const int64_t localIndex = threadIdx.y * blockDim.x + threadIdx.x;
	if (i >= dRadon3DDRV2Consts.samplesPerDirection || j >= dRadon3DDRV2Consts.samplesPerDirection) {
		buffer[threadIdx.y * blockDim.x + threadIdx.x] = 0.f;
		return;
	}

	const Vec<double, 3> texCoord = mappingIndexToTexCoord(Vec<double, 3>::Full(static_cast<double>(i)),
	                                                       Vec<double, 3>::Full(static_cast<double>(j)));
	buffer[threadIdx.y * blockDim.x + threadIdx.x] =
		Texture3DCUDA::DSampleDX(dRadon3DDRV2Consts.volumeWidth, dRadon3DDRV2Consts.textureHandle, texCoord) *
		dTexCoordDR.X() +
		Texture3DCUDA::DSampleDY(dRadon3DDRV2Consts.volumeHeight, dRadon3DDRV2Consts.textureHandle, texCoord) *
		dTexCoordDR.Y() + Texture3DCUDA::DSampleDZ(dRadon3DDRV2Consts.volumeDepth, dRadon3DDRV2Consts.textureHandle,
		                                           texCoord) * dTexCoordDR.Z();

	__syncthreads();

	for (int64_t cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff && threadIdx.y < cutoff) {
			buffer[threadIdx.y * blockDim.x + threadIdx.x] += buffer[threadIdx.y * blockDim.x + threadIdx.x + cutoff] +
				buffer[(threadIdx.y + cutoff) * blockDim.x + threadIdx.x] + buffer[
					(threadIdx.y + cutoff) * blockDim.x + threadIdx.x + cutoff];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0 && threadIdx.y) {
		dRadon3DDRV2Consts.patchSumsArray[blockIdx.y * gridDim.x + blockIdx.x] =
			dRadon3DDRV2Consts.scaleFactor * buffer[0];
	}
}

int blockSizeToDynamicSMemSize_DRadon3DDR_CUDA_V2(int blockSize) {
	return blockSize * sizeof(float);
}

__host__ at::Tensor DRadon3DDR_CUDA_V2(const at::Tensor &volume, const at::Tensor &volumeSpacing,
                                       const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                       const at::Tensor &rValues, int64_t samplesPerDirection) {
	CommonData common = Radon3D<Texture3DCUDA>::Common(volume, volumeSpacing, phiValues, thetaValues, rValues,
	                                                   samplesPerDirection, at::DeviceType::CUDA);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_DRadon3DDR_CUDA_V2,
	                                               &blockSizeToDynamicSMemSize_DRadon3DDR_CUDA_V2, 0);
	const unsigned blockWidth = static_cast<unsigned>(sqrtf(static_cast<float>(blockSize)));
	const dim3 blockDims = {blockWidth, blockSize / blockWidth};
	const size_t bufferSize = blockSizeToDynamicSMemSize_DRadon3DDR_CUDA_V2(blockDims.x * blockDims.y);
	const dim3 gridSize = {(static_cast<unsigned>(samplesPerDirection) + blockDims.x - 1) / blockDims.x,
	                       (static_cast<unsigned>(samplesPerDirection) + blockDims.y - 1) / blockDims.y};
	const at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize.y, gridSize.x}), common.flatOutput.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	DRadon3DDRV2Consts constants{common.inputTexture.GetHandle(), samplesPerDirection, common.scaleFactor, patchSumsPtr,
	                             volume.sizes()[2], volume.sizes()[1], volume.sizes()[0]};
	CudaMemcpyToObjectSymbol(dRadon3DDRV2Consts, constants);

	for (int64_t i = 0; i < common.flatOutput.numel(); ++i) {
		const float phi = phiFlat[i].item().toFloat();
		const float theta = thetaFlat[i].item().toFloat();
		const float r = rFlat[i].item().toFloat();
		const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCUDA>::GetMappingIndexToTexCoord(
			common.inputTexture, phi, theta, r, common.mappingIndexToOffset);
		const Vec<double, 3> dTexCoordDR = Radon3D<Texture3DCUDA>::GetDTexCoordDR(common.inputTexture, phi, theta, r);

		Kernel_DRadon3DDR_CUDA_V2<<<gridSize, blockDims, bufferSize>>>(mappingIndexToTexCoord, dTexCoordDR);

		common.flatOutput.index_put_({i}, patchSums.sum());
	}
	return common.flatOutput.view(phiValues.sizes());
}

} // namespace ExtensionTest