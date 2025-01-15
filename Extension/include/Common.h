#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace ExtensionTest {

struct Linear {
	float intercept;
	float gradient;

	__host__ __device__ [[nodiscard]] float operator()(const float x) const { return fmaf(gradient, x, intercept); }

	__host__ __device__ [[nodiscard]] Linear operator()(const Linear &other) const {
		return {fmaf(gradient, other.intercept, intercept), gradient * other.gradient};
	}
};

class Texture2D {
public:
	__host__ __device__ [[nodiscard]] long Width() const { return width; }
	__host__ __device__ [[nodiscard]] long Height() const { return height; }
	__host__ __device__ [[nodiscard]] float XSpacing() const { return xSpacing; }
	__host__ __device__ [[nodiscard]] float YSpacing() const { return ySpacing; }
	__host__ __device__ [[nodiscard]] float WidthWorld() const { return static_cast<float>(width) * xSpacing; }
	__host__ __device__ [[nodiscard]] float HeightWorld() const { return static_cast<float>(height) * ySpacing; }
	__host__ __device__ [[nodiscard]] static float In(const float x, const float y) {
		return x >= 0.f && x < 1.f && y >= 0.f && y < 1.f;
	}

	__host__ __device__ [[nodiscard]] Linear MappingXWorldToNormalised() const { return {.5f, 1.f / WidthWorld()}; }
	__host__ __device__ [[nodiscard]] Linear MappingYWorldToNormalised() const { return {.5f, -1.f / HeightWorld()}; }

protected:
	Texture2D() = default;

	Texture2D(long _width, long _height, float _xSpacing, float _ySpacing) : width(_width), height(_height),
	                                                                         xSpacing(_xSpacing), ySpacing(_ySpacing) {
	}

	// yes copy
	Texture2D(const Texture2D &) = default;

	Texture2D &operator=(const Texture2D &) = default;

	// yes move
	Texture2D(Texture2D &&) noexcept = default;

	Texture2D &operator=(Texture2D &&) noexcept = default;

private:
	long width{};
	long height{};
	float xSpacing{};
	float ySpacing{};
};

template <typename texture_t> struct Radon2D {
	__host__ __device__ static float Integrate(const texture_t &texture, const float phi, const float r,
	                                           const Linear &mappingIToOffset, const long samplesPerLine) {
		const float s = sinf(phi);
		const float c = cosf(phi);
		const Linear mappingOffsetToWorldX{r * c, -s};
		const Linear mappingOffsetToWorldY{r * s, c};
		const Linear mappingIToX = texture.MappingXWorldToNormalised()(mappingOffsetToWorldX(mappingIToOffset));
		const Linear mappingIToY = texture.MappingYWorldToNormalised()(mappingOffsetToWorldY(mappingIToOffset));
		float ret = 0.f;
		for (int i = 0; i < samplesPerLine; ++i) {
			const float iF = static_cast<float>(i);
			ret += texture.Sample(mappingIToX(iF), mappingIToY(iF));
		}
		return ret;
	}
};

} // namespace ExtensionTest