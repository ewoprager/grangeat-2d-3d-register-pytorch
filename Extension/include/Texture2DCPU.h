#pragma once

#include "Radon2D.h"

namespace ExtensionTest {

class Texture2DCPU : public Texture2D {
public:
	Texture2DCPU() = default;

	Texture2DCPU(const float *_ptr, long _width, long _height, double _xSpacing,
	             double _ySpacing) : Texture2D(_width, _height, _xSpacing, _ySpacing), ptr(_ptr) {
	}

	// yes copy
	Texture2DCPU(const Texture2DCPU &) = default;

	Texture2DCPU &operator=(const Texture2DCPU &) = default;

	// yes move
	Texture2DCPU(Texture2DCPU &&) = default;

	Texture2DCPU &operator=(Texture2DCPU &&) = default;

	__host__ __device__ [[nodiscard]] float At(long row, long col) const {
		return In(row, col) ? ptr[row * Width() + col] : 0.0f;
	}

	__host__ __device__ [[nodiscard]] float Sample(float x, float y) const {
		x = -.5f + x * static_cast<float>(Width());
		y = -.5f + y * static_cast<float>(Height());
		const float xFloored = floorf(x);
		const float yFloored = floorf(y);
		const long col = static_cast<long>(xFloored);
		const long row = static_cast<long>(yFloored);
		const float fHorizontal = x - xFloored;
		const float fVertical = y - yFloored;
		const float r0 = (1.f - fHorizontal) * At(row, col) + fHorizontal * At(row, col + 1);
		const float r1 = (1.f - fHorizontal) * At(row + 1, col) + fHorizontal * At(row + 1, col + 1);
		return (1.f - fVertical) * r0 + fVertical * r1;
	}

	__host__ __device__ [[nodiscard]] float SampleXDerivative(float x, float y) const {
		x = -.5f + x * static_cast<float>(Width());
		y = -.5f + y * static_cast<float>(Height());
		const float xFloored = floorf(x);
		const float yFloored = floorf(y);
		const long col = static_cast<long>(xFloored);
		const long row = static_cast<long>(yFloored);
		const float fVertical = y - yFloored;
		return (1.f - fVertical) * (At(row, col + 1) - At(row, col)) + fVertical * (
			       At(row + 1, col + 1) - At(row + 1, col));
	}

	__host__ __device__ [[nodiscard]] float SampleYDerivative(float x, float y) const {
		x = -.5f + x * static_cast<float>(Width());
		y = -.5f + y * static_cast<float>(Height());
		const float xFloored = floorf(x);
		const float yFloored = floorf(y);
		const long col = static_cast<long>(xFloored);
		const long row = static_cast<long>(yFloored);
		const float fHorizontal = x - xFloored;
		return (1.f - fHorizontal) * (At(row + 1, col) - At(row, col)) + fHorizontal * (
			       At(row + 1, col + 1) - At(row, col + 1));
	}

private:
	const float *ptr{};
};

} // namespace ExtensionTest