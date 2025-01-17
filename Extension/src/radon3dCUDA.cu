#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"

namespace ExtensionTest {

__global__ void radon3d_kernel(Texture3DCUDA textureIn, long depthOut, long heightOut, long widthOut, float *arrayOut,
                               Radon3D<Texture3DCUDA>::ConstMappings constMappings, long samplesPerDirection,
                               float scaleFactor) {
	const long col = blockIdx.x * blockDim.x + threadIdx.x;
	const long row = blockIdx.y * blockDim.y + threadIdx.y;
	const long layer = blockIdx.z * blockDim.z + threadIdx.z;
	if (col >= widthOut || row >= heightOut || layer >= depthOut) return;

	const long index = layer * widthOut * heightOut + row * widthOut + col;
	const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(textureIn, col, row, layer, constMappings);
	arrayOut[index] = scaleFactor * Radon3D<Texture3DCUDA>::IntegrateLooped(
		                  textureIn, indexMappings, samplesPerDirection);
}

__host__ at::Tensor radon3d_cuda(const at::Tensor &a, double xSpacing, double ySpacing, double zSpacing, long depthOut,
                                 long heightOut, long widthOut, long samplesPerDirection) {
	// a should be a 3D array of floats on the GPU
	TORCH_CHECK(a.sizes().size() == 3);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture3DCUDA texture{aPtr, a.sizes()[2], a.sizes()[1], a.sizes()[0], xSpacing, ySpacing, zSpacing};

	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());

	const auto constMappings = Radon3D<Texture3DCUDA>::GetConstMappings(widthOut, heightOut, depthOut, planeSize,
	                                                                    samplesPerDirection);

	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr dim3 blockSize{10, 10, 10};
	const dim3 gridSize{(static_cast<unsigned>(widthOut) + blockSize.x - 1) / blockSize.x,
	                    (static_cast<unsigned>(heightOut) + blockSize.y - 1) / blockSize.y,
	                    (static_cast<unsigned>(depthOut) + blockSize.z - 1) / blockSize.z};
	radon3d_kernel<<<gridSize, blockSize>>>(std::move(texture), depthOut, heightOut, widthOut, resultPtr, constMappings,
	                                        samplesPerDirection, scaleFactor);

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

__host__ at::Tensor radon3d_v2_cuda(const at::Tensor &a, double xSpacing, double ySpacing, double zSpacing,
                                    long depthOut, long heightOut, long widthOut, long samplesPerDirection) {
	// a should be a 3D array of floats on the GPU
	TORCH_CHECK(a.sizes().size() == 3);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture3DCUDA texture{aPtr, a.sizes()[2], a.sizes()[1], a.sizes()[0], xSpacing, ySpacing, zSpacing};

	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());

	const float planeSize = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld() + texture.
		DepthWorld() * texture.DepthWorld());

	const auto constMappings = Radon3D<Texture3DCUDA>::GetConstMappings(widthOut, heightOut, depthOut, planeSize,
	                                                                    samplesPerDirection);

	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;

	constexpr dim3 blockSize = {32, 32};
	constexpr size_t bufferSize = blockSize.x * blockSize.y * sizeof(float);
	const dim3 gridSize = {(static_cast<unsigned>(samplesPerDirection) + blockSize.x - 1) / blockSize.x,
	                       (static_cast<unsigned>(samplesPerDirection) + blockSize.y - 1) / blockSize.y};
	at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize.y, gridSize.x}), result.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	Radon3DV2Consts constants{texture.GetHandle(), samplesPerDirection, scaleFactor, patchSumsPtr};
	cudaMemcpyToSymbol(radon3DV2Consts, &constants, sizeof(Radon3DV2Consts));

	for (long layer = 0; layer < depthOut; ++layer) {
		for (long row = 0; row < heightOut; ++row) {
			for (long col = 0; col < widthOut; ++col) {
				const auto indexMappings = Radon3D<Texture3DCUDA>::GetIndexMappings(
					texture, col, row, layer, constMappings);

				radon3d_v3_kernel<<<gridSize, blockSize, bufferSize>>>(indexMappings);

				result.index_put_({layer, row, col}, patchSums.sum());
			}
		}
	}

	return result;
}

} // namespace ExtensionTest