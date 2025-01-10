#include <torch/extension.h>

#include "Texture2D.h"

namespace ExtensionTest {

__global__ void muladd_kernel(int numel, const float *a, const float *b, float c, float *result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

__host__ at::Tensor mymuladd_cuda(const at::Tensor &a, const at::Tensor &b, double c) {
	TORCH_CHECK(a.sizes() == b.sizes());
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_CHECK(b.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
	at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
	const float *a_ptr = a_contig.data_ptr<float>();
	const float *b_ptr = b_contig.data_ptr<float>();
	float *result_ptr = result.data_ptr<float>();

	int numel = a_contig.numel();
	muladd_kernel<<<(numel + 255) / 256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
	return result;
}


__global__ void radon2d_kernel(const Texture2D textureIn, int heightOut, int widthOut, float *arrayOut,
                               float rayLength) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= widthOut || row >= heightOut) return;
	const int index = row * widthOut + col;
	arrayOut[index] = textureIn.IntegrateRay(
		3.1415926535f * (-.5f + static_cast<float>(row) / static_cast<float>(heightOut)),
		rayLength * (-.5f + static_cast<float>(col) / static_cast<float>(widthOut - 1)), .1f);
}

__host__ at::Tensor radon2d_cuda(const at::Tensor &a, const at::Tensor &outputDims) {
	// a should be a 2D array of floats on the GPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	// outputDims should be a 1D array of length 2 of ints on the CPU
	TORCH_CHECK(outputDims.sizes().size() == 1);
	TORCH_CHECK(outputDims.sizes()[0] == 2);
	TORCH_CHECK(outputDims.dtype() == at::kInt);
	TORCH_INTERNAL_ASSERT(outputDims.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = a.contiguous();
	const float *a_ptr = aContiguous.data_ptr<float>();
	const Texture2D texture{a_ptr, a.sizes()[0], a.sizes()[1]};

	at::Tensor outputDims_contig = outputDims.contiguous();
	const int *outputDims_ptr = outputDims_contig.data_ptr<int>();
	const int &height = outputDims_ptr[0];
	const int &width = outputDims_ptr[1];

	at::Tensor result = torch::zeros(at::IntArrayRef({height, width}), aContiguous.options());
	float *result_ptr = result.data_ptr<float>();

	const float rayLength = sqrt(
		texture.SizeXWorld() * texture.SizeXWorld() + texture.SizeYWorld() * texture.SizeYWorld());

	const dim3 blockSize{16, 16};
	const dim3 gridSize{(width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y};
	radon2d_kernel<<<gridSize, blockSize>>>(texture, height, width, result_ptr, rayLength);

	return result;
}


TORCH_LIBRARY_IMPL(ExtensionTest, CUDA, m) {
	m.impl("mymuladd", &mymuladd_cuda);
	m.impl("radon2d", &radon2d_cuda);
}

} // namespace ExtensionTest