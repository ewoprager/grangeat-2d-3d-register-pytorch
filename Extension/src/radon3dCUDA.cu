#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"

namespace ExtensionTest {

__global__ void radon3d_kernel(Texture3DCUDA textureIn, long numelOut, float *arrayOut, Linear mappingIToOffset,
                               const float *phiValues, const float *thetaValues, const float *rValues,
                               long samplesPerDirection, float scaleFactor) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(textureIn, phiValues[threadIndex],
	                                                                    thetaValues[threadIndex], rValues[threadIndex],
	                                                                    mappingIToOffset);
	arrayOut[threadIndex] = scaleFactor * Radon3D<Texture3DCUDA>::IntegrateLooped(
		                        textureIn, indexMappings, samplesPerDirection);
}

__host__ at::Tensor radon3d_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                 const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                                 long samplesPerDirection) {
	// volume should be a 3D array of floats on the GPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing, zSpacing};

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor thetaFlatContiguous = thetaValues.flatten().contiguous();
	const float *thetaFlatPtr = thetaFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());
	const Linear mappingIToOffset = Radon3D<Texture3DCUDA>::GetMappingIToOffset(planeSize, samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr int blockSize = 512;
	const int gridSize = (static_cast<unsigned>(numelOut) + blockSize - 1) / blockSize;
	radon3d_kernel<<<gridSize, blockSize>>>(std::move(texture), numelOut, resultFlatPtr, mappingIToOffset, phiFlatPtr,
	                                        thetaFlatPtr, rFlatPtr, samplesPerDirection, scaleFactor);
	return resultFlat.view(phiValues.sizes());
}

__global__ void dRadon3dDR_kernel(Texture3DCUDA textureIn, long numelOut, float *arrayOut, Linear mappingIToOffset,
                                  const float *phiValues, const float *thetaValues, const float *rValues,
                                  long samplesPerDirection, float scaleFactor) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;
	const float phi = phiValues[threadIndex];
	const float theta = thetaValues[threadIndex];
	const float r = rValues[threadIndex];
	const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(textureIn, phi, theta, r, mappingIToOffset);
	const auto derivativeWRTR = Radon3D<Texture3DCUDA>::GetDerivativeWRTR(textureIn, phi, theta, r);
	arrayOut[threadIndex] = scaleFactor * Radon3D<Texture3DCUDA>::DIntegrateLoopedDMappingParameter(
		                        textureIn, indexMappings, derivativeWRTR, samplesPerDirection);
}

__host__ at::Tensor dRadon3dDR_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection) {
	// volume should be a 3D array of floats on the GPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing, zSpacing};

	const long numelOut = phiValues.numel();
	at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor thetaFlatContiguous = thetaValues.flatten().contiguous();
	const float *thetaFlatPtr = thetaFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());
	const Linear mappingIToOffset = Radon3D<Texture3DCUDA>::GetMappingIToOffset(planeSize, samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr int blockSize = 512;
	const int gridSize = (static_cast<unsigned>(numelOut) + blockSize - 1) / blockSize;
	dRadon3dDR_kernel<<<gridSize, blockSize>>>(std::move(texture), numelOut, resultFlatPtr, mappingIToOffset,
	                                           phiFlatPtr, thetaFlatPtr, rFlatPtr, samplesPerDirection, scaleFactor);
	return resultFlat.view(phiValues.sizes());
}

struct Radon3DV2Consts {
	cudaTextureObject_t textureHandle{};
	long samplesPerDirection{};
	float scaleFactor{};
	float *patchSumsArray{};
};

__device__ __constant__ Radon3DV2Consts radon3DV2Consts{};

__global__ void radon3d_v2_kernel(Radon3D<Texture3DCUDA>::IndexMappings indexMappings) {
	extern __shared__ float buffer[];

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	const long j = blockIdx.y * blockDim.y + threadIdx.y;
	const long localIndex = threadIdx.y * blockDim.x + threadIdx.x;
	if (i >= radon3DV2Consts.samplesPerDirection || j >= radon3DV2Consts.samplesPerDirection) {
		buffer[localIndex] = 0.f;
		return;
	}

	const float iF = static_cast<float>(i);
	const float jF = static_cast<float>(j);
	buffer[localIndex] = tex3D<float>(radon3DV2Consts.textureHandle, indexMappings.mappingIJToX(iF, jF),
	                                  indexMappings.mappingIJToY(iF, jF), indexMappings.mappingIJToZ(iF, jF));

	__syncthreads();

	for (long cutoff = (blockDim.x * blockDim.y) / 2; cutoff > 0; cutoff /= 2) {
		if (localIndex < cutoff) {
			buffer[localIndex] += buffer[localIndex + cutoff];
		}

		__syncthreads();
	}
	if (localIndex == 0) {
		radon3DV2Consts.patchSumsArray[blockIdx.y * gridDim.x + blockIdx.x] = radon3DV2Consts.scaleFactor * buffer[0];
	}
}

__global__ void radon3d_v3_kernel(Radon3D<Texture3DCUDA>::IndexMappings indexMappings) {
	extern __shared__ float buffer[];

	// REQUIRED: blockDim.x must be equal to blockDim.y

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	const long j = blockIdx.y * blockDim.y + threadIdx.y;
	// const long localIndex = threadIdx.y * blockDim.x + threadIdx.x;
	if (i >= radon3DV2Consts.samplesPerDirection || j >= radon3DV2Consts.samplesPerDirection) {
		buffer[threadIdx.y * blockDim.x + threadIdx.x] = 0.f;
		return;
	}

	const float iF = static_cast<float>(i);
	const float jF = static_cast<float>(j);
	buffer[threadIdx.y * blockDim.x + threadIdx.x] = tex3D<float>(radon3DV2Consts.textureHandle,
	                                                              indexMappings.mappingIJToX(iF, jF),
	                                                              indexMappings.mappingIJToY(iF, jF),
	                                                              indexMappings.mappingIJToZ(iF, jF));

	__syncthreads();

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
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

__host__ at::Tensor radon3d_v2_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection) {
	// volume should be a 3D array of floats on the GPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());
	const Linear mappingIToOffset = Radon3D<Texture3DCUDA>::GetMappingIToOffset(planeSize, samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr dim3 blockSize = {32, 32};
	constexpr size_t bufferSize = blockSize.x * blockSize.y * sizeof(float);
	const dim3 gridSize = {(static_cast<unsigned>(samplesPerDirection) + blockSize.x - 1) / blockSize.x,
	                       (static_cast<unsigned>(samplesPerDirection) + blockSize.y - 1) / blockSize.y};
	const at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize.y, gridSize.x}), resultFlat.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	Radon3DV2Consts constants{texture.GetHandle(), samplesPerDirection, scaleFactor, patchSumsPtr};
	CudaMemcpyToObjectSymbol(radon3DV2Consts, constants);

	for (long i = 0; i < numelOut; ++i) {
		const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(
			texture, phiFlat[i].item().toFloat(), thetaFlat[i].item().toFloat(), rFlat[i].item().toFloat(),
			mappingIToOffset);

		radon3d_v3_kernel<<<gridSize, blockSize, bufferSize>>>(indexMappings);

		resultFlat.index_put_({i}, patchSums.sum());
	}
	return resultFlat.view(phiValues.sizes());
}

struct DRadon3DDRV2Consts {
	cudaTextureObject_t textureHandle{};
	long samplesPerDirection{};
	float scaleFactor{};
	float *patchSumsArray{};
	long volumeWidth{};
	long volumeHeight{};
	long volumeDepth{};
};

__device__ __constant__ DRadon3DDRV2Consts dRadon3DDRV2Consts{};

__global__ void dRadon3dDR_v2_kernel(Radon3D<Texture3DCUDA>::IndexMappings indexMappings,
                                     Radon3D<Texture3DCUDA>::DerivativeWRTR derivativeWRTR) {
	extern __shared__ float buffer[];

	// REQUIRED: blockDim.x must be equal to blockDim.y

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	const long j = blockIdx.y * blockDim.y + threadIdx.y;
	// const long localIndex = threadIdx.y * blockDim.x + threadIdx.x;
	if (i >= dRadon3DDRV2Consts.samplesPerDirection || j >= dRadon3DDRV2Consts.samplesPerDirection) {
		buffer[threadIdx.y * blockDim.x + threadIdx.x] = 0.f;
		return;
	}

	const float iF = static_cast<float>(i);
	const float jF = static_cast<float>(j);
	const float x = indexMappings.mappingIJToX(iF, jF);
	const float y = indexMappings.mappingIJToY(iF, jF);
	const float z = indexMappings.mappingIJToZ(iF, jF);
	buffer[threadIdx.y * blockDim.x + threadIdx.x] =
		Texture3DCUDA::SampleXDerivative(dRadon3DDRV2Consts.volumeWidth, dRadon3DDRV2Consts.textureHandle, x, y, z) *
		derivativeWRTR.dXdR +
		Texture3DCUDA::SampleYDerivative(dRadon3DDRV2Consts.volumeHeight, dRadon3DDRV2Consts.textureHandle, x, y, z) *
		derivativeWRTR.dYdR + Texture3DCUDA::SampleZDerivative(dRadon3DDRV2Consts.volumeDepth,
		                                                       dRadon3DDRV2Consts.textureHandle, x, y,
		                                                       z) * derivativeWRTR.dZdR;

	__syncthreads();

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
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

__host__ at::Tensor dRadon3dDR_v2_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                       const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                       const at::Tensor &rValues, long samplesPerDirection) {
	// volume should be a 3D array of floats on the GPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the GPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());

	const Linear mappingIToOffset = Radon3D<Texture3DCUDA>::GetMappingIToOffset(planeSize, samplesPerDirection);

	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr dim3 blockSize = {32, 32};
	constexpr size_t bufferSize = blockSize.x * blockSize.y * sizeof(float);
	const dim3 gridSize = {(static_cast<unsigned>(samplesPerDirection) + blockSize.x - 1) / blockSize.x,
	                       (static_cast<unsigned>(samplesPerDirection) + blockSize.y - 1) / blockSize.y};
	const at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize.y, gridSize.x}), resultFlat.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	DRadon3DDRV2Consts constants{texture.GetHandle(), samplesPerDirection, scaleFactor, patchSumsPtr, volume.sizes()[2],
	                             volume.sizes()[1], volume.sizes()[0]};
	CudaMemcpyToObjectSymbol(dRadon3DDRV2Consts, constants);

	for (long i = 0; i < numelOut; ++i) {
		const float phi = phiFlat[i].item().toFloat();
		const float theta = thetaFlat[i].item().toFloat();
		const float r = rFlat[i].item().toFloat();
		const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(texture, phi, theta, r, mappingIToOffset);
		const auto derivativeWRTR = Radon3D<Texture3DCUDA>::GetDerivativeWRTR(texture, phi, theta, r);

		dRadon3dDR_v2_kernel<<<gridSize, blockSize, bufferSize>>>(indexMappings, derivativeWRTR);

		resultFlat.index_put_({i}, patchSums.sum());
	}
	return resultFlat.view(phiValues.sizes());
}

} // namespace ExtensionTest