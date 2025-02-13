#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

#include "Vec.h"

namespace ExtensionTest {

template <typename T> struct Linear2;

template <typename T> struct Linear {
	T intercept;
	T gradient;

	__host__ __device__ [[nodiscard]] T operator()(const T x) const { return gradient * x + intercept; }

	__host__ __device__ [[nodiscard]] Linear operator()(const Linear &other) const {
		return {gradient * other.intercept + intercept, gradient * other.gradient};
	}

	__host__ __device__ [[nodiscard]] Linear2<T> operator()(const Linear2<T> &other) const;
};

template <typename T> struct Linear2 {
	T intercept;
	T gradient1;
	T gradient2;

	__host__ __device__ [[nodiscard]] T operator()(const T x, const T y) const {
		return gradient1 * x + gradient2 * y + intercept;
	}

	__host__ __device__ [[nodiscard]] Linear2 operator()(const Linear<T> &other) const {
		return {(gradient1 + gradient2) * other.intercept + intercept, gradient1 * other.gradient,
		        gradient2 * other.gradient};
	}

	__host__ __device__ [[nodiscard]] Linear2 operator()(const Linear2 &other) const {
		const T gradientSum = gradient1 + gradient2;
		return {gradientSum * other.intercept + intercept, gradientSum * other.gradient1,
		        gradientSum * other.gradient2};
	}
};

template <typename T> __host__ __device__ [[nodiscard]] Linear2<T> Linear<T>::operator(
)(const Linear2<T> &other) const {
	return {gradient * other.intercept + intercept, gradient * other.gradient1, gradient * other.gradient2};
}


#ifdef __CUDACC__

template <typename T> __host__ cudaError_t CudaMemcpyToObjectSymbol(const T &symbol, T &src,
                                                                    cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
	return cudaMemcpyToSymbol(symbol, &src, sizeof(T), 0, kind);
}

#endif

} // namespace ExtensionTest