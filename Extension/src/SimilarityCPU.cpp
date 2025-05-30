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
	const float nF = static_cast<float>(n);

	const Vec<float, 5> sums = at::parallel_reduce(0, n, 10000, Vec<float, 5>{},
	                                               [aPtr, bPtr](int64_t begin, int64_t end,
	                                                            Vec<float, 5> threadSum) -> Vec<float, 5> {
		                                               for (int64_t i = begin; i < end; i++) {
			                                               threadSum += Vec<float, 5>{
				                                               aPtr[i], bPtr[i], aPtr[i] * aPtr[i], bPtr[i] * bPtr[i],
				                                               aPtr[i] * bPtr[i]};
		                                               }
		                                               return threadSum;
	                                               }, std::plus<Vec<float, 5> >());
	// const float sumB = at::parallel_reduce(0, n, 10000, 0.f,
	//                                        [bPtr](int64_t begin, int64_t end, float threadSum) -> float {
	// 	                                       for (int64_t i = begin; i < end; i++) {
	// 		                                       threadSum += bPtr[i];
	// 	                                       }
	// 	                                       return threadSum;
	//                                        }, std::plus<float>());
	// const float sumA2 = at::parallel_reduce(0, n, 10000, 0.f,
	//                                         [aPtr](int64_t begin, int64_t end, float threadSum) -> float {
	// 	                                        for (int64_t i = begin; i < end; i++) {
	// 		                                        threadSum += aPtr[i] * aPtr[i];
	// 	                                        }
	// 	                                        return threadSum;
	//                                         }, std::plus<float>());
	// const float sumB2 = at::parallel_reduce(0, n, 10000, 0.f,
	//                                         [bPtr](int64_t begin, int64_t end, float threadSum) -> float {
	// 	                                        for (int64_t i = begin; i < end; i++) {
	// 		                                        threadSum += bPtr[i] * bPtr[i];
	// 	                                        }
	// 	                                        return threadSum;
	//                                         }, std::plus<float>());
	// const float sumAB = at::parallel_reduce(0, n, 10000, 0.f,
	//                                         [aPtr, bPtr](int64_t begin, int64_t end, float threadSum) -> float {
	// 	                                        for (int64_t i = begin; i < end; i++) {
	// 		                                        threadSum += aPtr[i] * bPtr[i];
	// 	                                        }
	// 	                                        return threadSum;
	//                                         }, std::plus<float>());

	// return torch::tensor(
	// 	(nF * sumAB - sumA * sumB) / (sqrtf(nF * sumA2 - sumA * sumA) * sqrtf(nF * sumB2 - sumB * sumB) + 1e-10f),
	// 	a.options());
	return torch::tensor(
		(nF * sums[4] - sums[0] * sums[1]) / (sqrtf(nF * sums[2] - sums[0] * sums[0]) * sqrtf(
			                                      nF * sums[3] - sums[1] * sums[1]) + 1e-10f), a.options());
}

} // namespace reg23