#include <torch/extension.h>

#include "../include/Similarity.h"
#include "../include/Vec.h"

namespace reg23 {

at::Tensor NormalisedCrossCorrelation_CPU(const at::Tensor &a, const at::Tensor &b) {
	Similarity::Common(a, b, at::DeviceType::CPU);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const at::Tensor bContiguous = b.contiguous();
	const float *bPtr = bContiguous.data_ptr<float>();

	const int64_t n = aContiguous.numel();
	const double nF = static_cast<double>(n);

	const Vec<double, 5> sums = at::parallel_reduce( //
		0, n, 10000, Vec<double, 5>::Full(0.f),
		[aPtr, bPtr](int64_t begin, int64_t end, Vec<double, 5> threadSum) -> Vec<double, 5> {
			for (int64_t i = begin; i < end; ++i) {
				const double ai = static_cast<double>(aPtr[i]);
				const double bi = static_cast<double>(bPtr[i]);
				threadSum += Vec<double, 5>{ai, bi, ai * ai, bi * bi, ai * bi};
			}
			return threadSum;
		}, std::plus<Vec<double, 5> >{});
	return torch::tensor(
		(nF * sums[4] - sums[0] * sums[1]) / (sqrt(nF * sums[2] - sums[0] * sums[0]) * sqrt(
			                                      nF * sums[3] - sums[1] * sums[1]) + 1e-10), a.options());
}

std::tuple<at::Tensor, double, double, double, double, double> NormalisedCrossCorrelation_forward_CPU(
	const at::Tensor &a, const at::Tensor &b) {
	Similarity::Common(a, b, at::DeviceType::CPU);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const at::Tensor bContiguous = b.contiguous();
	const float *bPtr = bContiguous.data_ptr<float>();

	const int64_t n = aContiguous.numel();
	const double nF = static_cast<double>(n);

	const Vec<double, 5> sums = at::parallel_reduce( //
		0, n, 10000, Vec<double, 5>::Full(0.f),
		[aPtr, bPtr](int64_t begin, int64_t end, Vec<double, 5> threadSum) -> Vec<double, 5> {
			for (int64_t i = begin; i < end; ++i) {
				const double ai = static_cast<double>(aPtr[i]);
				const double bi = static_cast<double>(bPtr[i]);
				threadSum += Vec<double, 5>{ai, bi, ai * ai, bi * bi, ai * bi};
			}
			return threadSum;
		}, std::plus<Vec<double, 5> >{});

	const double numerator = nF * sums[4] - sums[0] * sums[1];
	const double denominatorLeft = sqrt(nF * sums[2] - sums[0] * sums[0]);
	const double denominatorRight = sqrt(nF * sums[3] - sums[1] * sums[1]);

	const at::Tensor zncc = torch::tensor(numerator / (denominatorLeft * denominatorRight + 1e-10), a.options());

	return {zncc, sums[0], sums[1], numerator, denominatorLeft, denominatorRight};
}

} // namespace reg23