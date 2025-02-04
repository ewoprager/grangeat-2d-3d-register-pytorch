#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"

namespace ExtensionTest {

__global__ void radon3d_kernel(Texture3DCUDA textureIn, long depthOut, long heightOut, long widthOut, float *arrayOut,
                               Linear mappingIToOffset, const float *phiValues, const float *thetaValues,
                               const float *rValues, long samplesPerDirection, float scaleFactor) {
	const long col = blockIdx.x * blockDim.x + threadIdx.x;
	const long row = blockIdx.y * blockDim.y + threadIdx.y;
	const long layer = blockIdx.z * blockDim.z + threadIdx.z;
	if (col >= widthOut || row >= heightOut || layer >= depthOut) return;

	const long index = layer * widthOut * heightOut + row * widthOut + col;
	const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(textureIn, phiValues[layer], thetaValues[row],
	                                                                    rValues[col], mappingIToOffset);
	arrayOut[index] = scaleFactor * Radon3D<Texture3DCUDA>::IntegrateLooped(
		                  textureIn, indexMappings, samplesPerDirection);
}

__host__ at::Tensor radon3d_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                 const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                                 long samplesPerDirection) {
	// volume should be a 3D array of floats on the GPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// phiValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// thetaValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// rValues should be a 1D array of floats on the GPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing, zSpacing};

	const long depthOut = phiValues.sizes()[0];
	const long heightOut = thetaValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const at::Tensor phisContiguous = phiValues.contiguous();
	const float *phisPtr = phisContiguous.data_ptr<float>();
	const at::Tensor thetasContiguous = thetaValues.contiguous();
	const float *thetasPtr = thetasContiguous.data_ptr<float>();
	const at::Tensor rsContiguous = rValues.contiguous();
	const float *rsPtr = rsContiguous.data_ptr<float>();

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());

	const Linear mappingIToOffset = Radon3D<Texture3DCUDA>::GetMappingIToOffset(planeSize, samplesPerDirection);

	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr dim3 blockSize{10, 10, 10};
	const dim3 gridSize{(static_cast<unsigned>(widthOut) + blockSize.x - 1) / blockSize.x,
	                    (static_cast<unsigned>(heightOut) + blockSize.y - 1) / blockSize.y,
	                    (static_cast<unsigned>(depthOut) + blockSize.z - 1) / blockSize.z};
	radon3d_kernel<<<gridSize, blockSize>>>(std::move(texture), depthOut, heightOut, widthOut, resultPtr,
	                                        mappingIToOffset, phisPtr, thetasPtr, rsPtr, samplesPerDirection,
	                                        scaleFactor);

	return result;
}

__global__ void dRadon3dDR_kernel(Texture3DCUDA textureIn, long depthOut, long heightOut, long widthOut,
                                  float *arrayOut, Linear mappingIToOffset, const float *phiValues,
                                  const float *thetaValues, const float *rValues, long samplesPerDirection,
                                  float scaleFactor) {
	const long col = blockIdx.x * blockDim.x + threadIdx.x;
	const long row = blockIdx.y * blockDim.y + threadIdx.y;
	const long layer = blockIdx.z * blockDim.z + threadIdx.z;
	if (col >= widthOut || row >= heightOut || layer >= depthOut) return;

	const long index = layer * widthOut * heightOut + row * widthOut + col;
	const float phi = phiValues[layer];
	const float theta = thetaValues[row];
	const float r = rValues[col];
	const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(textureIn, phi, theta, r, mappingIToOffset);
	const auto derivativeWRTR = Radon3D<Texture3DCUDA>::GetDerivativeWRTR(textureIn, phi, theta, r);
	arrayOut[index] = scaleFactor * Radon3D<Texture3DCUDA>::DIntegrateLoopedDMappingParameter(
		                  textureIn, indexMappings, derivativeWRTR, samplesPerDirection);
}

__host__ at::Tensor dRadon3dDR_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection) {
	// volume should be a 3D array of floats on the GPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// phiValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// thetaValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// rValues should be a 1D array of floats on the GPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing, zSpacing};

	const long depthOut = phiValues.sizes()[0];
	const long heightOut = thetaValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const at::Tensor phisContiguous = phiValues.contiguous();
	const float *phisPtr = phisContiguous.data_ptr<float>();
	const at::Tensor thetasContiguous = thetaValues.contiguous();
	const float *thetasPtr = thetasContiguous.data_ptr<float>();
	const at::Tensor rsContiguous = rValues.contiguous();
	const float *rsPtr = rsContiguous.data_ptr<float>();

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());

	const Linear mappingIToOffset = Radon3D<Texture3DCUDA>::GetMappingIToOffset(planeSize, samplesPerDirection);

	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr dim3 blockSize{10, 10, 10};
	const dim3 gridSize{(static_cast<unsigned>(widthOut) + blockSize.x - 1) / blockSize.x,
	                    (static_cast<unsigned>(heightOut) + blockSize.y - 1) / blockSize.y,
	                    (static_cast<unsigned>(depthOut) + blockSize.z - 1) / blockSize.z};
	dRadon3dDR_kernel<<<gridSize, blockSize>>>(std::move(texture), depthOut, heightOut, widthOut, resultPtr,
	                                           mappingIToOffset, phisPtr, thetasPtr, rsPtr, samplesPerDirection,
	                                           scaleFactor);
	return result;
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
	// phiValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// thetaValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// rValues should be a 1D array of floats on the GPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	const long depthOut = phiValues.sizes()[0];
	const long heightOut = thetaValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());

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
	at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize.y, gridSize.x}), result.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	Radon3DV2Consts constants{texture.GetHandle(), samplesPerDirection, scaleFactor, patchSumsPtr};
	CudaMemcpyToObjectSymbol(radon3DV2Consts, constants);

	for (long layer = 0; layer < depthOut; ++layer) {
		for (long row = 0; row < heightOut; ++row) {
			for (long col = 0; col < widthOut; ++col) {
				const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(
					texture, phiValues[layer].item().toFloat(), thetaValues[row].item().toFloat(),
					rValues[col].item().toFloat(), mappingIToOffset);

				radon3d_v3_kernel<<<gridSize, blockSize, bufferSize>>>(indexMappings);

				result.index_put_({layer, row, col}, patchSums.sum());
			}
		}
	}

	return result;
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
	// phiValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// thetaValues should be a 1D array of floats on the GPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CUDA);
	// rValues should be a 1D array of floats on the GPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture3DCUDA texture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing, zSpacing};

	const long depthOut = phiValues.sizes()[0];
	const long heightOut = thetaValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());

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
	at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize.y, gridSize.x}), result.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	DRadon3DDRV2Consts constants{texture.GetHandle(), samplesPerDirection, scaleFactor, patchSumsPtr, volume.sizes()[2],
	                             volume.sizes()[1], volume.sizes()[0]};
	CudaMemcpyToObjectSymbol(dRadon3DDRV2Consts, constants);

	for (long layer = 0; layer < depthOut; ++layer) {
		for (long row = 0; row < heightOut; ++row) {
			for (long col = 0; col < widthOut; ++col) {
				const float phi = phiValues[layer].item().toFloat();
				const float theta = thetaValues[row].item().toFloat();
				const float r = rValues[col].item().toFloat();
				const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(
					texture, phi, theta, r, mappingIToOffset);
				const auto derivativeWRTR = Radon3D<Texture3DCUDA>::GetDerivativeWRTR(texture, phi, theta, r);

				dRadon3dDR_v2_kernel<<<gridSize, blockSize, bufferSize>>>(indexMappings, derivativeWRTR);

				result.index_put_({layer, row, col}, patchSums.sum());
			}
		}
	}

	return result;
}

} // namespace ExtensionTest