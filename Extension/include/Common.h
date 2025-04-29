/**
 * @file
 * @brief Global functions
 */

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

/**
 * @brief Returns the square of the given value
 * @tparam T Type of parameter and result
 * @param x Input value
 * @return The input value squared using `operator*`
 */
template <typename T> __host__ __device__ T Square(const T &x) { return x * x; }

/**
 * @brief Modulo operation that respect the sign
 * @tparam T Type of the parameters and result
 * @param x
 * @param y
 * @return x modulo y, respecting the sign of x.
 */
template <typename T> __host__ __device__ T Modulo(const T &x, const T &y) { return (x % y + y) % y; }

template <typename T> struct Linear2;

/**
 * @brief A functor class that represents a linear transformation: intercept + gradient * x
 * @tparam T The type of the contained values
 *
 * Can be applied to other instances of `Linear` and `Linear2`, producing a single `Linear` or `Linear2` instance that
 * represents the linear transformation corresponding to the chaining of the two combined instances.
 */
template <typename T> struct Linear {
	T intercept;
	T gradient;

	/**
	 * @param x
	 * @return The linear transform of the input: intercept + gradient * x
	 */
	[[nodiscard]] __host__ __device__ T operator()(const T &x) const { return gradient * x + intercept; }

	/**
	 * Chain two `Linear` transformations
	 * @param other
	 * @return The `Linear` transformation corresponding to the application of `other`, followed by `this`
	 */
	[[nodiscard]] __host__ __device__ Linear operator()(const Linear &other) const {
		return {gradient * other.intercept + intercept, gradient * other.gradient};
	}

	/**
	 * Chain a `Linear` transformation with a `Linear2` transformation
	 * @param other
	 * @return The `Linear2` transformation corresponding to the application of `other`, followed by `this`
	 */
	[[nodiscard]] __host__ __device__ Linear2<T> operator()(const Linear2<T> &other) const;
};

/**
 *  @brief A functor class that represents a linear transformation of two variables: intercept + gradient1 * x +
 * gradient2 * y
 * @tparam T The type of the contained values
 */
template <typename T> struct Linear2 {
	T intercept;
	T gradient1;
	T gradient2;

	/**
	 * @param x
	 * @param y
	 * @return The linear transform of the input: intercept + gradient1 * x + gradient2 * y
	 */
	[[nodiscard]] __host__ __device__ T operator()(const T &x, const T &y) const {
		return gradient1 * x + gradient2 * y + intercept;
	}

	/**
	 * Chain a `Linear2` transformation with a `Linear` transformation
	 * @param other
	 * @return The `Linear2` transformation corresponding to the application of `other`, followed by `this`
	 */
	[[nodiscard]] __host__ __device__ Linear2 operator()(const Linear<T> &other) const {
		return {(gradient1 + gradient2) * other.intercept + intercept, gradient1 * other.gradient,
				gradient2 * other.gradient};
	}

	/**
	 * Chain two `Linear2` transformations
	 * @param other
	 * @return The `Linear2` transformation corresponding to the application of `other`, followed by `this`
	 */
	[[nodiscard]] __host__ __device__ Linear2 operator()(const Linear2 &other) const {
		const T gradientSum = gradient1 + gradient2;
		return {gradientSum * other.intercept + intercept, gradientSum * other.gradient1,
				gradientSum * other.gradient2};
	}
};

template <typename T>
[[nodiscard]] __host__ __device__ Linear2<T> Linear<T>::operator()(const Linear2<T> &other) const {
	return {gradient * other.intercept + intercept, gradient * other.gradient1, gradient * other.gradient2};
}

/**
 * @brief 'Unflips' the given spherical coordinates so that theta and phi both lie between -pi/2 and pi/2
 * @tparam T The type of the parameter and returned coordinate values
 * @param coordSph A 3-vector containing (r, theta, phi)
 * @return The unflipped 3-vector (r', theta', phi'),
 *
 * The spherical coordinates should be defined according to the convention stated in Extension/Conventions.md
 */
template <typename T> [[nodiscard]] __host__ __device__ Vec<T, 3> UnflipSphericalCoordinate(const Vec<T, 3> &coordSph) {
	constexpr T PI_T = static_cast<T>(M_PI);
	const T theta_div = std::floor((coordSph.Y() + .5f * PI_T) / PI_T);
	const bool theta_flip = static_cast<bool>(std::abs(static_cast<long>(theta_div)) % 2);
	const T phi_div = std::floor((coordSph.Z() + .5f * PI_T) / PI_T);
	const bool phi_flip = static_cast<bool>(std::abs(static_cast<long>(phi_div)) % 2);

	T r_ret = coordSph.X();
	T theta_ret = coordSph.Y() - PI_T * theta_div;
	T phi_ret = coordSph.Z() - PI_T * phi_div;

	if (phi_flip && !theta_flip) theta_ret *= -1;
	if (phi_flip != theta_flip) r_ret *= -1;

	return {r_ret, theta_ret, phi_ret};
}

#ifdef __CUDACC__

template <typename T>
__host__ cudaError_t CudaMemcpyToObjectSymbol(const T &symbol, T &src, cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
	return cudaMemcpyToSymbol(symbol, &src, sizeof(T), 0, kind);
}

#endif

} // namespace ExtensionTest