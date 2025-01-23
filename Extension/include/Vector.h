#pragma once

#include <cuda/std/array>

#include "Common.h"

namespace ExtensionTest {

template <uint8_t> struct vec_view {
	virtual ~vec_view() = default;

	virtual float &operator[](uint8_t i) = 0;

	virtual const float &operator[](uint8_t i) const = 0;
};

template <uint8_t N> struct vec : public vec_view<N> {
	float data[N]{};

	float &operator[](uint8_t i) override { return data[i]; }
	const float &operator[](uint8_t i) const override { return data[i]; }

	static vec Zero() {
		vec ret{};
		memset(ret.data, 0, sizeof(ret.data));
		return cuda::std::move(ret);
	}
};

template <uint8_t N, uint8_t M> struct mat {
	float data[N * M]{};

	float &operator()(uint8_t i, uint8_t j) { return data[i * M + j]; }
	const float &operator()(uint8_t i, uint8_t j) const { return data[i * M + j]; }

	static mat Zero() {
		mat ret;
		memset(ret.data, 0, sizeof(ret.data));
		return cuda::std::move(ret);
	}

	static mat Identity() requires (N == M) {
		mat ret = Zero();
		for (uint8_t i = 0; i < N; ++i) {
			ret(i, i) = 1.f;
		}
		return cuda::std::move(ret);
	}

	struct row_view : vec_view<M> {
		row_view(mat &_owner, uint8_t _row) : owner(_owner), row(_row) {
		}

		float &operator[](uint8_t i) override { return owner(row, i); }
		const float &operator[](uint8_t i) const override { return owner(row, i); }

	private:
		mat &owner;
		const uint8_t row;
	};

	struct col_view : vec_view<N> {
		col_view(mat &_owner, uint8_t _col) : owner(_owner), col(_col) {
		}

		float &operator[](uint8_t i) override { return owner(i, col); }
		const float &operator[](uint8_t i) const override { return owner(i, col); }

	private:
		mat &owner;
		const uint8_t col;
	};

	row_view Row(uint8_t row) { return row_view(*this, row); }
	col_view Col(uint8_t col) { return col_view(*this, col); }
};

template <uint8_t N> float Dot(const vec_view<N> &a, const vec_view<N> &b) {
	float ret = 0.f;
	for (uint8_t i = 0; i < N; ++i) {
		ret += a[i] * b[i];
	}
	return ret;
}

template <uint8_t N, uint8_t M> vec<N> operator*(const mat<N, M> &m, const vec_view<M> &v) {
	vec<N> ret{};
	for (uint8_t i = 0; i < N; ++i) {
		ret[i] = Dot(m.Row(i), v);
	}
	return cuda::std::move(ret);
}

template <uint8_t N, uint8_t M, uint8_t L> mat<N, L> operator*(const mat<N, M> &m1, const mat<M, L> &m2) {
	mat<N, L> ret{};
	for (uint8_t j = 0; j < N; ++j) {
		for (uint8_t i = 0; i < L; ++i) {
			ret(j, i) = Dot(m1.Row(j), m2.Col(i));
		}
	}
	return cuda::std::move(ret);
}

} // namespace ExtensionTest