#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace ExtensionTest {
struct Linear {
	float intercept;
	float gradient;

	__host__ __device__ [[nodiscard]] float operator()(const float x) const { return intercept + gradient * x; }

	__host__ __device__ [[nodiscard]] Linear operator()(const Linear &other) const {
		return {intercept + gradient * other.intercept, gradient * other.gradient};
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

} // namespace ExtensionTest