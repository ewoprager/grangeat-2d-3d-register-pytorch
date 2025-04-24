#pragma once

#include <array>

namespace ExtensionTest {
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

	__host__ __device__/**/Vec(Base array) : Base(
#ifdef __CUDACC__
		cuda::
#endif
		std::move(array)) {
	}

	__host__ __device__/**/Vec(std::initializer_list<T> l) : Base() {
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

	__host__ static Vec FromTensor(const at::Tensor &t) {
		assert(t.sizes() == at::IntArrayRef({N}));
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
			return {{t[indices].item().template to<T>()...}};
		}(std::make_index_sequence<N>{});
	}

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

	__host__ __device__ [[nodiscard]] constexpr Vec Flipped() const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
			return {{(*this)[N - 1 - indices]...}};
		}(std::make_index_sequence<N>{});
	}

	__host__ __device__ [[nodiscard]] constexpr T Sum() const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> T {
			return ((*this)[indices] + ...);
		}(std::make_index_sequence<N>{});
	}

	__host__ __device__ [[nodiscard]] constexpr Vec<T, 3> DeHomogenise() const {
		static_assert(std::is_floating_point_v<T>, "Can only de-homogenise floating-point vectors");
		static_assert(N == 4, "Can only de-homogenise 4-length vectors");
		return Vec<T, 3>{X(), Y(), Z()} / W();
	}

	template <typename newT> __host__ __device__ [[nodiscard]] constexpr Vec<newT, N> StaticCast() const {
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
			return {{static_cast<newT>((*this)[indices])...}};
		}(std::make_index_sequence<N>{});
	}

	__host__ __device__ [[nodiscard]] constexpr bool BooleanAll() const {
		static_assert(std::is_integral_v<T>, "Only integral types are supported for boolean 'all' operation");
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> bool {
			return ((*this)[indices] && ...);
		}(std::make_index_sequence<N>{});
	}

	__host__ __device__ [[nodiscard]] constexpr bool BooleanNone() const {
		static_assert(std::is_integral_v<T>, "Only integral types are supported for boolean 'none' operation");
		return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> bool {
			return (!(*this)[indices] && ...);
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
			([&]() {
				(*this)[indices] += other[indices];
			}(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator-=(const Vec &other) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() {
				(*this)[indices] -= other[indices];
			}(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator*=(const Vec &other) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() {
				(*this)[indices] *= other[indices];
			}(), ...);
		}(std::make_index_sequence<N>{});
		return *this;
	}

	__host__ __device__ Vec &operator/=(const Vec &other) {
		[&]<std::size_t... indices>(std::index_sequence<indices...>) {
			([&]() {
				(*this)[indices] /= other[indices];
			}(), ...);
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

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator+(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] + rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator+(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] + rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator+(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs + rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise subtraction: -

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator-(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] - rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator-(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] - rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator-(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs - rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise multiplication: *

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator*(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] * rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator*(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] * rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator*(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs * rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise division: /

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator/(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] / rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator/(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs[indices] / rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<T, N> operator/(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
		return {{(lhs / rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Dot product

template <typename T, std::size_t N> __host__ __device__ constexpr T
VecDot(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> T {
		return ((lhs[indices] * rhs[indices]) + ...);
	}(std::make_index_sequence<N>{});
}

// Element-wise greater-than: >

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] > rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] > rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs > rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise greater-or-equal-than: >=

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>=(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] >= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>=(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] >= rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator>=(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs >= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise less-than: <

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] < rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] < rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs < rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Element-wise less--or-equal-than: <=

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<=(
	const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] <= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<=(
	const Vec<T, N> &lhs, const T &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs[indices] <= rhs)...}};
	}(std::make_index_sequence<N>{});
}

template <typename T, std::size_t N> __host__ __device__ constexpr Vec<bool, N> operator<=(
	const T &lhs, const Vec<T, N> &rhs) {
	return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
		return {{(lhs <= rhs[indices])...}};
	}(std::make_index_sequence<N>{});
}

// Matrix-vector multiplication

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

} // namespace ExtensionTest