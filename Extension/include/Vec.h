#pragma once

#include "Global.h"

#include <array>

namespace reg23 {

/**
 * @ingroup data_structures
 * @brief A simple vector class derived from `std::array<T, N>`, providing overrides for all useful operators.
 * @tparam T The type of the vector's elements
 * @tparam N The size/dimensionality of the vector; must be greater than 0
 *
 * - CUDA-compatible
 * - Provides some conversions to and from PyTorch structures (at::Tensor and at::IntArrayRef)
 * - Methods and operator overloads are constexpr, excluding PyTorch-related functionality
 * - Can be used with very limited functionality for matrix manipulation using `Vec<Vec<T, C>, R>`; see
 * reg23::MatMul
 */
template <typename T, std::size_t N> class Vec : public std::array<T, N> {
public:
	static_assert(N > 0, "Vec size must be greater than 0");

	using ElementType = T;
	static constexpr std::size_t dimensionality = N;

	using Base = std::array<T, N>;

	__host__ __device__ /**/Vec() : Base() {
		for (T &e : *this) {
			e = T{};
		}
	}

	__host__ __device__ /**/Vec(Base array)
		: Base(
#ifdef __CUDACC__
			cuda::
#endif
			std::move(array)) {
	}

	__host__ __device__ /**/Vec(std::initializer_list<T> l) : Base() {
		int i = 0;
		for (const T &e : l) {
			(*this)[i] = e;
			++i;
		}
	}

	__host__ __device__ static Vec Full(const T &value) {
		Vec ret{};
		for (T &e : ret) {
			e = value;
		}
		return ret;
	}

	__host__ __device__ static constexpr Vec FromIntArrayRef(const at::IntArrayRef &v) {
		assert(v.size() == N);
		Vec ret{};
		int index = 0;
		for (int64_t e : v) {
			ret[index++] = static_cast<T>(e);
		}
		return
#ifdef __CUDACC__
			cuda::
#endif
			std::move(ret);
	}

	/**
	 * @brief Construct a vector from a 1D PyTorch tensor
	 * @param t a tensor of size (N,); the contained type must be consistent with the vector's type, i.e. for a
	 * `Vec<float32_t, N>`, `t` must contain values of type `torch.float32`.
	 * @return A `Vec` filled with the values copied from the given tensor
	 */
	__host__ static Vec FromTensor(const at::Tensor &t) {
		assert(t.sizes() == at::IntArrayRef({N}));
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
			return {{t[indices].item().template to<T>()...}};
		}(std::make_index_sequence<N>{});
	}

	/**
	 * @brief Construct a matrix from a 2D PyTorch tensor
	 * @param t a tensor of size (N, T::dimensionality). The contained type must be consistent with the vector's
	 * type, i.e. for a `Vec<Vec<float32_t, C>, R>`, `t` must contain values of type `torch.float32`.
	 * @return A `Vec` (of `Vec`s) filled with the values copied from the given tensor
	 *
	 * Only valid for `Vec` of contained type `T` that is a specialisation of `Vec`, e.g. `Vec<Vec<float, C>, R>`
	 */
	__host__ static Vec FromTensor2D(const at::Tensor &t) {
		static_assert(std::is_same_v<std::remove_const_t<decltype(T::dimensionality)>, std::size_t>);
		assert(t.sizes() == at::IntArrayRef({N, T::dimensionality}));
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
			return {{T::FromTensor(t.slice(1, indices, indices + 1).squeeze(1))...}};
		}(std::make_index_sequence<N>{});
	}

	__host__ __device__ [[nodiscard]] constexpr at::IntArrayRef ToIntArrayRef() const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> at::IntArrayRef {
			return at::IntArrayRef({(*this)[indices]...});
		}(std::make_index_sequence<N>{});
	}

	__host__ [[nodiscard]] at::Tensor ToTensor(at::TensorOptions options = {}) const {
		return torch::from_blob(this->data(), {this->size()}, options);
	}

	/**
	 * @return A copy of this vector, with the order of the elements reversed
	 */
	__host__ __device__ [[nodiscard]] constexpr Vec Flipped() const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
			return {{(*this)[N - 1 - indices]...}};
		}(std::make_index_sequence<N>{});
	}

	/**
	 * @return The sum of the elements of the vector
	 */
	__host__ __device__ [[nodiscard]] constexpr T Sum() const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> T {
			return ((*this)[indices] + ...);
		}(std::make_index_sequence<N>{});
	}

	/**
	 * @return The 3-vector corresponding to this homogeneous 4-vector, i.e. (x, y, z) / w
	 *
	 * For floating-point 4-vectors only.
	 */
	__host__ __device__ [[nodiscard]] constexpr Vec<T, 3> DeHomogenise() const {
		static_assert(std::is_floating_point_v<T>, "Can only de-homogenise floating-point vectors");
		static_assert(N == 4, "Can only de-homogenise 4-length vectors");
		return Vec<T, 3>{X(), Y(), Z()} / W();
	}

	/**
	 * @return A copy of this vector, with each element cast to the type `newT`
	 */
	template <typename newT> __host__ __device__ [[nodiscard]] constexpr Vec<newT, N> StaticCast() const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
			return {{static_cast<newT>((*this)[indices])...}};
		}(std::make_index_sequence<N>{});
	}

	__host__ __device__ [[nodiscard]] constexpr T Length() const {
		return sqrt(Apply(&Square<T>).Sum());
	}

	/**
	 * @return Whether all elements of the vector are `true`
	 *
	 * For boolean vectors only
	 */
	__host__ __device__ [[nodiscard]] constexpr bool BooleanAll() const {
		static_assert(std::is_integral_v<T>, "Only integral types are supported for boolean 'all' operation");
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> bool {
			return ((*this)[indices] && ...);
		}(std::make_index_sequence<N>{});
	}

	/**
	 * @return Whether any elements of the vector are `true`
	 *
	 * For boolean vectors only
	 */
	__host__ __device__ [[nodiscard]] constexpr bool BooleanAny() const {
		static_assert(std::is_integral_v<T>, "Only integral types are supported for boolean 'any' operation");
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> bool {
			return ((*this)[indices] || ...);
		}(std::make_index_sequence<N>{});
	}

	// Element-wise function:

	template <typename newT> __host__ __device__ [[nodiscard]] constexpr Vec<newT, N> Apply(
		const std::function<newT(T)> &f) const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
			return {{f((*this)[indices])...}};
		}(std::make_index_sequence<N>{});
	}

	template <typename newT> __host__ __device__ [[nodiscard]] constexpr Vec<newT, N> Apply(
		const std::function<newT(const T &)> &f) const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
			return {{f((*this)[indices])...}};
		}(std::make_index_sequence<N>{});
	}

	template <typename newT> __host__ __device__ [[nodiscard]] constexpr Vec<newT, N> Apply(newT (*f)(T)) const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
			return {{f((*this)[indices])...}};
		}(std::make_index_sequence<N>{});
	}

	template <typename newT> __host__ __device__ [[nodiscard]] constexpr Vec<newT, N>
	Apply(newT (*f)(const T &)) const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
			return {{f((*this)[indices])...}};
		}(std::make_index_sequence<N>{});
	}

	// Modification in place

	__host__ __device__ Vec &operator+=(const Vec &other) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] += other[indices]; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator+=(const T &scalar) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] += scalar; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator-=(const Vec &other) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] -= other[indices]; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator-=(const T &scalar) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] -= scalar; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator*=(const Vec &other) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] *= other[indices]; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator*=(const T &scalar) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] *= scalar; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator/=(const Vec &other) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] /= other[indices]; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator/=(const T &scalar) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() { (*this)[indices] /= scalar; }(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	// Named member accessors

	__host__ __device__ [[nodiscard]] constexpr const T &X() const { return (*this)[0]; }
	__host__ __device__ [[nodiscard]] constexpr T &X() { return (*this)[0]; }

	__host__ __device__ [[nodiscard]] constexpr const T &Y() const {
		static_assert(N > 1, "Vec size must be greater than 1 to access element Y");
		return (*this)[1];
	}

	__host__ __device__ [[nodiscard]] constexpr T &Y() {
		static_assert(N > 1, "Vec size must be greater than 1 to access element Y");
		return (*this)[1];
	}

	__host__ __device__ [[nodiscard]] constexpr const T &Z() const {
		static_assert(N > 2, "Vec size must be greater than 2 to access element Z");
		return (*this)[2];
	}

	__host__ __device__ [[nodiscard]] constexpr T &Z() {
		static_assert(N > 2, "Vec size must be greater than 2 to access element Z");
		return (*this)[2];
	}

	__host__ __device__ [[nodiscard]] constexpr const T &W() const {
		static_assert(N > 3, "Vec size must be greater than 3 to access element W");
		return (*this)[3];
	}

	__host__ __device__ [[nodiscard]] constexpr T &W() {
		static_assert(N > 3, "Vec size must be greater than 3 to access element W");
		return (*this)[3];
	}

	__host__ __device__ [[nodiscard]] constexpr Vec<T, 2> XY() const { return {X(), Y()}; }
	__host__ __device__ [[nodiscard]] constexpr Vec<T, 3> XYZ() const { return {X(), Y(), Z()}; }
};

// Element-wise addition: +

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise addition
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator+(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] + rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise addition
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator+(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] + rhs)...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise addition
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator+(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs + rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise subtraction: -

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise subtraction
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator-(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] - rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise subtraction
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator-(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] - rhs)...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise subtraction
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator-(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs - rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise multiplication: *

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise multiplication
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator*(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] * rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise multiplication
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator*(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] * rhs)...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise multiplication
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator*(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs * rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise division: /

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise division
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator/(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] / rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise division
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator/(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] / rhs)...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise division
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator/(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs / rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Dot product

/**
 * @ingroup data_structures
 * @brief reg23::Vec dot product
 */
template <typename T, std::size_t N> __host__ __device__ constexpr T
VecDot(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> T {
		return ((lhs[indices] * rhs[indices]) + ...);
	}(std::make_index_sequence<N>{});
}

// Element-wise greater-than: >
/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise greater-than
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] > rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise greater-than
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] > rhs)...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise greater-than
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs > rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise greater-or-equal-than: >=

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise greater-than-or-equal-to
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>=(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] >= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise greater-than-or-equal-to
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>=(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] >= rhs)...}};
	}(std::make_index_sequence<N>{});
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise greater-than-or-equal-to
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>=(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs >= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise less-than: <

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise less-than
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] < rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}


/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise less-than
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] < rhs)...}};
	}(std::make_index_sequence<N>{});
}


/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise less-than
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs < rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise less--or-equal-than: <=

/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise less-than-or-equal-to
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<=(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] <= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}


/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise less-than-or-equal-to
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<=(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] <= rhs)...}};
	}(std::make_index_sequence<N>{});
}


/**
 * @ingroup data_structures
 * @brief reg23::Vec element-wise less-than-or-equal-to
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<=(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs <= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Concatenation

/**
 * @ingroup data_structures
 * @brief reg23::Vec concatenation of two vectors
 */
template <typename T, std::size_t N1, std::size_t N2> __host__ __device__ constexpr Vec<T, N1 + N2> VecCat(
	const Vec<T, N1> &lhs, const Vec<T, N2> &rhs) {
	Vec<T, N1 + N2> ret;
	[&]<std::size_t... indices>(std::index_sequence<indices...>) {
		([&]() { ret[indices] = lhs[indices]; }(), ...);
	}(std::make_index_sequence<N1>{});
	[&]<std::size_t... indices>(std::index_sequence<indices...>) {
		([&]() { ret[N1 + indices] = rhs[indices]; }(), ...);
	}(std::make_index_sequence<N2>{});
	return ret;
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec concatenation of a vector and a scalar
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N + 1> VecCat(
	const Vec<T, N> &lhs, const T &rhs) {
	Vec<T, N + 1> ret;
	[&]<std::size_t... indices>(std::index_sequence<indices...>) {
		([&]() { ret[indices] = lhs[indices]; }(), ...);
	}(std::make_index_sequence<N>{});
	ret[N] = rhs;
	return ret;
}

/**
 * @ingroup data_structures
 * @brief reg23::Vec concatenation of a vector and a scalar
 */
template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N + 1> VecCat(
	const T &lhs, const Vec<T, N> &rhs) {
	Vec<T, N + 1> ret;
	ret[0] = lhs;
	[&]<std::size_t... indices>(std::index_sequence<indices...>) {
		([&]() { ret[1 + indices] = rhs[indices]; }(), ...);
	}(std::make_index_sequence<N>{});
	return ret;
}

/**
 * @ingroup data_structures
 * @brief Matrix-vector multiplication of the `Vec` struct
 * @tparam T Matrix / vector element type
 * @tparam R Number of rows in matrix, the size of the returned vector
 * @tparam C Number of columns in matrix, the size of the multiplied vector
 * @param lhs A column-major, `R` x `C` matrix
 * @param rhs A vector of length `C`
 * @return The vector of length `R` that is the result of multiplication of the given vector on the left by the given
 * matrix
 */
template <typename T, std::size_t R, std::size_t C> __host__ __device__ constexpr Vec<T, R> MatMul(
	const Vec<Vec<T, R>, C> &lhs, const Vec<T, C> &rhs) {
	constexpr auto rowDotProduct = []<std::size_t row>(const Vec<Vec<T, R>, C> &mat, const Vec<T, C> &vec) -> T {
		return [&]<std::size_t... cols>(std::index_sequence<cols...>) -> T {
			return ((mat[cols][row] * vec[cols]) + ...);
		}(std::make_index_sequence<C>{});
	};
	return [&]<std::size_t... rows>(std::index_sequence<rows...>) -> std::array<T, R> {
		return {{rowDotProduct.template operator()<rows>(lhs, rhs)...}};
	}(std::make_index_sequence<R>{});
}

} // namespace reg23