#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

namespace ExtensionTest {

struct Linear2;

struct Linear {
	float intercept;
	float gradient;

	__host__ __device__ [[nodiscard]] float operator()(const float x) const { return fmaf(gradient, x, intercept); }

	__host__ __device__ [[nodiscard]] Linear operator()(const Linear &other) const {
		return {fmaf(gradient, other.intercept, intercept), gradient * other.gradient};
	}

	__host__ __device__ [[nodiscard]] Linear2 operator()(const Linear2 &other) const;
};

struct Linear2 {
	float intercept;
	float gradient1;
	float gradient2;

	__host__ __device__ [[nodiscard]] float operator()(const float x, const float y) const {
		return fmaf(gradient1, x, fmaf(gradient2, y, intercept));
	}

	__host__ __device__ [[nodiscard]] Linear2 operator()(const Linear &other) const {
		return {fmaf(gradient1 + gradient2, other.intercept, intercept), gradient1 * other.gradient,
		        gradient2 * other.gradient};
	}

	__host__ __device__ [[nodiscard]] Linear2 operator()(const Linear2 &other) const {
		const float gradientSum = gradient1 + gradient2;
		return {fmaf(gradientSum, other.intercept, intercept), gradientSum * other.gradient1,
		        gradientSum * other.gradient2};
	}
};

__host__ __device__ [[nodiscard]] inline Linear2 Linear::operator()(const Linear2 &other) const {
	return {fmaf(gradient, other.intercept, intercept), gradient * other.gradient1, gradient * other.gradient2};
}


#ifdef __CUDACC__

template <typename T> __host__ cudaError_t CudaMemcpyToObjectSymbol(const T &symbol, T &src,
                                                                    cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
	return cudaMemcpyToSymbol(symbol, &src, sizeof(T), 0, kind);
}

#endif

} // namespace ExtensionTest