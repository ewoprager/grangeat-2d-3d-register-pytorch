#include <torch/extension.h>

#include "include/Texture2DCUDA.h"

namespace ExtensionTest {

__global__ void radon2d_kernel(Texture2DCUDA textureIn, long heightOut, long widthOut, float *arrayOut,
                               Radon2D<Texture2DCUDA>::ConstMappings constMappings, long samplesPerLine,
                               float scaleFactor) {
	const long col = blockIdx.x * blockDim.x + threadIdx.x;
	const long row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= widthOut || row >= heightOut) return;
	const long index = row * widthOut + col;
	const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(textureIn, col, row, constMappings);
	arrayOut[index] = scaleFactor * Radon2D<Texture2DCUDA>::IntegrateLooped(textureIn, indexMappings, samplesPerLine);
}

__host__ at::Tensor radon2d_cuda(const at::Tensor &a, long heightOut, long widthOut, long samplesPerLine) {
	// a should be a 2D array of floats on the GPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCUDA texture{aPtr, a.sizes()[0], a.sizes()[1], 1.f, 1.f};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float rayLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());

	const auto constMappings = Radon2D<Texture2DCUDA>::GetConstMappings(widthOut, heightOut, rayLength, samplesPerLine);

	const float scaleFactor = rayLength / static_cast<float>(samplesPerLine);

	const dim3 blockSize{16, 16};
	const dim3 gridSize{(static_cast<unsigned>(widthOut) + blockSize.x - 1) / blockSize.x,
	                    (static_cast<unsigned>(heightOut) + blockSize.y - 1) / blockSize.y};
	radon2d_kernel<<<gridSize, blockSize>>>(std::move(texture), heightOut, widthOut, resultPtr, constMappings,
	                                        samplesPerLine, scaleFactor);

	return result;
}

__global__ void radon2d_v2_kernel(const Texture2DCUDA *textureIn, long samplesPerLine,
                                  const Radon2D<Texture2DCUDA>::IndexMappings indexMappings, float scaleFactor,
                                  float *ret) {

	extern __shared__ float buffer[];

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= samplesPerLine) return;

	const float iF = static_cast<float>(i);

	buffer[i] = textureIn->Sample(indexMappings.mappingIToX(iF), indexMappings.mappingIToY(iF));

	__syncthreads();

	for (long cutoff = samplesPerLine / 2; cutoff > 0; cutoff /= 2) {
		if (i < cutoff) {
			buffer[i] += buffer[i + cutoff];
		}

		__syncthreads();
	}
	if (i == 0) *ret = scaleFactor * buffer[0];
}

__host__ at::Tensor radon2d_v2_cuda(const at::Tensor &a, long heightOut, long widthOut, long samplesPerLine) {
	// samplesPerLine should be no more than 1024
	TORCH_CHECK(samplesPerLine <= 1024);

	// a should be a 2D array of floats on the GPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCUDA texture{aPtr, a.sizes()[0], a.sizes()[1], 1.f, 1.f};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float rayLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const float scaleFactor = rayLength / static_cast<float>(samplesPerLine);
	const auto constMappings = Radon2D<Texture2DCUDA>::GetConstMappings(widthOut, heightOut, rayLength, samplesPerLine);
	for (unsigned row = 0; row < heightOut; ++row) {
		for (unsigned col = 0; col < widthOut; ++col) {
			const auto indexMappings = Radon2D<Texture2DCUDA>::GetIndexMappings(texture, col, row, constMappings);

			constexpr unsigned blockSize = 1024;
			radon2d_v2_kernel<<<1, blockSize, blockSize * sizeof(float)>>>(
				&texture, samplesPerLine, indexMappings, scaleFactor, &resultPtr[row * widthOut + col]);

		}
	}

	return result;
}

TORCH_LIBRARY_IMPL(ExtensionTest, CUDA, m) {
	m.impl("radon2d", &radon2d_cuda);
	m.impl("radon2d_v2", &radon2d_v2_cuda);
}

} // namespace ExtensionTest