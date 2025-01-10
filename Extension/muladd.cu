#include <cmath>
#include <numbers>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

namespace ExtensionTest {

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

__host__ at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
	TORCH_CHECK(a.sizes() == b.sizes());
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_CHECK(b.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
	at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
	const float* a_ptr = a_contig.data_ptr<float>();
	const float* b_ptr = b_contig.data_ptr<float>();
	float* result_ptr = result.data_ptr<float>();

	int numel = a_contig.numel();
	muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
	return result;
}

TORCH_LIBRARY_IMPL(ExtensionTest, CUDA, m) {
	m.impl("mymuladd", &mymuladd_cuda);
}

} // namespace ExtensionTest