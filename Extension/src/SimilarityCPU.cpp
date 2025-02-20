#include <torch/extension.h>

#include "../include/Similarity.h"

namespace ExtensionTest {

at::Tensor NormalisedCrossCorrelation(const at::Tensor &a, const at::Tensor &b) {
	const at::DeviceType device = at::DeviceType::CPU;
	// a and b should contain floats, match in size and be on the chosen device
	TORCH_CHECK(a.sizes() == b.sizes())
	TORCH_CHECK(a.dtype() == at::kFloat)
	TORCH_CHECK(b.dtype() == at::kFloat)
	TORCH_INTERNAL_ASSERT(a.device().type() == device)
	TORCH_INTERNAL_ASSERT(b.device().type() == device)

	const at::Tensor aFlat = a.flatten();
	const at::Tensor bFlat = b.flatten();

	float sumA = 0.f;
	float sumB = 0.f;
	float sumA2 = 0.f;
	float sumB2 = 0.f;
	float sumAB = 0.f;

	for (int i = 0; i < aFlat.numel(); ++i) {
		const float ai = aFlat[i].item().toFloat();
		const float bi = bFlat[i].item().toFloat();
		sumA += ai;
		sumB += bi;
		sumA2 += ai * ai;
		sumB2 += bi * bi;
		sumAB += ai * bi;
	}

	const float nF = static_cast<float>(aFlat.numel());

	return torch::tensor(
		{{(nF * sumAB - sumA * sumB) / (sqrtf(nF * sumA2 - sumA * sumA) * sqrtf(nF * sumB2 - sumB * sumB))}},
		a.options());
}

} // namespace ExtensionTest