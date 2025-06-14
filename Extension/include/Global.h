/**
 * @file
 * @brief General tools
 */

#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#else
/**
 * Definition of CUDA macro as nothing when compiling with just C++
 */
#define __host__

/**
 * Definition of CUDA macro as nothing when compiling with just C++
 */
#define __device__
#endif

namespace reg23 {

/**
 * @defgroup general_tools General Tools
 * @brief Small functions used throughout the codebase.
 * @{
 */

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


template <typename T> __host__ __device__ T Sign(const T &x) { return static_cast<T>((x > 0) - (x < 0)); }

/**
 * @}
 */

} // namespace reg23