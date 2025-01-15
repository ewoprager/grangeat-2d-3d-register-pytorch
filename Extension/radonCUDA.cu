#include <torch/extension.h>

#include "include/Texture2DCUDA.h"

namespace ExtensionTest {

__global__ void radon2d_kernel(const Texture2DCUDA textureIn, int heightOut, int widthOut, float *arrayOut,
                               float rayLength, long samplesPerLine) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= widthOut || row >= heightOut) return;
	const int index = row * widthOut + col;
	const Linear mappingItoOffset{-.5f * rayLength, rayLength / static_cast<float>(samplesPerLine - 1)};
	arrayOut[index] = Radon2D<Texture2DCUDA>::Integrate(
		textureIn, 3.1415926535f * (-.5f + static_cast<float>(row) / static_cast<float>(heightOut)),
		rayLength * (-.5f + static_cast<float>(col) / static_cast<float>(widthOut - 1)), mappingItoOffset,
		samplesPerLine);
}

__host__ at::Tensor radon2d_cuda(const at::Tensor &a, long heightOut, long widthOut, long samplesPerLine) {
	// a should be a 2D array of floats on the GPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCUDA texture{aPtr, a.sizes()[0], a.sizes()[1]};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float rayLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());

	const dim3 blockSize{16, 16};
	const dim3 gridSize{(static_cast<unsigned>(widthOut) + blockSize.x - 1) / blockSize.x,
	                    (static_cast<unsigned>(heightOut) + blockSize.y - 1) / blockSize.y};
	radon2d_kernel<<<gridSize, blockSize>>>(std::move(texture), heightOut, widthOut, resultPtr, rayLength,
	                                        samplesPerLine);

	return result;
}

__global__ void radon2d_v2_kernel(const Texture2DCUDA *textureIn, long samplesPerLine, const Linear mappingIToX,
                                  const Linear mappingIToY, float *ret) {

	extern __shared__ float buffer[];

	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= samplesPerLine) return;

	const float iF = static_cast<float>(i);

	buffer[i] = textureIn->Sample(mappingIToX(iF), mappingIToY(iF));

	__syncthreads();

	for (long cutoff = samplesPerLine / 2; cutoff > 0; cutoff /= 2) {
		if (i < cutoff) {
			buffer[i] += buffer[i + cutoff];
		}

		__syncthreads();
	}
	if (i == 0) *ret = buffer[0];
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
	Texture2DCUDA texture{aPtr, a.sizes()[0], a.sizes()[1]};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float rayLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());

	const Linear mappingIToOffset{-.5f * rayLength, rayLength / static_cast<float>(samplesPerLine - 1)};
	for (unsigned row = 0; row < heightOut; ++row) {
		for (unsigned col = 0; col < widthOut; ++col) {
			const float r = rayLength * (-.5f + static_cast<float>(col) / static_cast<float>(widthOut - 1));

			const float phi = 3.1415926535f * (-.5f + static_cast<float>(row) / static_cast<float>(heightOut));
			const float c = cosf(phi);
			const float s = sinf(phi);

			const Linear mappingOffsetToWorldX{r * c, -s};
			const Linear mappingOffsetToWorldY{r * s, c};
			const Linear mappingIToX = texture.MappingXWorldToNormalised()(mappingOffsetToWorldX(mappingIToOffset));
			const Linear mappingIToY = texture.MappingYWorldToNormalised()(mappingOffsetToWorldY(mappingIToOffset));

			constexpr unsigned blockSize = 1024;
			// const unsigned gridSize = (static_cast<unsigned>(samplesPerLine) + blockSize - 1) / blockSize;
			radon2d_v2_kernel<<</*gridSize*/1, blockSize, blockSize * sizeof(float)>>>(
				&texture, samplesPerLine, mappingIToX, mappingIToY, &resultPtr[row * widthOut + col]);
		}
	}

	return result;
}

TORCH_LIBRARY_IMPL(ExtensionTest, CUDA, m) {
	m.impl("radon2d", &radon2d_cuda);
	m.impl("radon2d_v2", &radon2d_v2_cuda);
}

} // namespace ExtensionTest