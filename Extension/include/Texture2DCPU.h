#pragma once

#include "Common.h"

namespace ExtensionTest {

class Texture2DCPU : public Texture2D {
public:
	Texture2DCPU() = default;

	Texture2DCPU(const float *_ptr, long _width, long _height, float _xSpacing, float _ySpacing) : ptr(_ptr),
		Texture2D(_width, _height, _xSpacing, _ySpacing) {
	}

	// yes copy
	Texture2DCPU(const Texture2DCPU &) = default;

	Texture2DCPU &operator=(const Texture2DCPU &) = default;

	// yes move
	Texture2DCPU(Texture2DCPU &&) = default;

	Texture2DCPU &operator=(Texture2DCPU &&) = default;

	__host__ __device__ [[nodiscard]] float At(int row, int col) const {
		return In(row, col) ? ptr[row * Width() + col] : 0.0f;
	}

	__host__ __device__ [[nodiscard]] float SampleBilinear(float x, float y) const {
		const float xUnnormalised = x * static_cast<float>(Width() - 1);
		const float yUnnormalised = y * static_cast<float>(Height() - 1);
		const int col = static_cast<int>(floorf(xUnnormalised));
		const int row = static_cast<int>(floorf(yUnnormalised));
		const float fHorizontal = xUnnormalised - static_cast<float>(col);
		const float fVertical = yUnnormalised - static_cast<float>(row);
		const float r0 = (1.f - fHorizontal) * At(row, col) + fHorizontal * At(row, col + 1);
		const float r1 = (1.f - fHorizontal) * At(row + 1, col) + fHorizontal * At(row + 1, col + 1);
		return (1.f - fVertical) * r0 + fVertical * r1;
	}

	__host__ __device__ [[nodiscard]] float
	IntegrateRay(float phi, float r, const Linear &mappingIToOffset, long samplesPerLine) const {
		const float s = sinf(phi);
		const float c = cosf(phi);
		const Linear mappingOffsetToWorldX{r * c, -s};
		const Linear mappingOffsetToWorldY{r * s, c};
		const Linear mappingIToX = MappingXWorldToNormalised()(mappingOffsetToWorldX(mappingIToOffset));
		const Linear mappingIToY = MappingYWorldToNormalised()(mappingOffsetToWorldY(mappingIToOffset));
		float ret = 0.f;
		for (int i = 0; i < samplesPerLine; ++i) {
			const float iF = static_cast<float>(i);
			ret += SampleBilinear(mappingIToX(iF), mappingIToY(iF));
		}
		return ret;
	}

private:
	const float *ptr{};
};

} // namespace ExtensionTest