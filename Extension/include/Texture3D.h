#pragma once

#include "Common.h"

namespace ExtensionTest {

class Texture3D {
public:
	__host__ __device__ [[nodiscard]] long Width() const { return width; }
	__host__ __device__ [[nodiscard]] long Height() const { return height; }
	__host__ __device__ [[nodiscard]] long Depth() const { return depth; }
	__host__ __device__ [[nodiscard]] double XSpacing() const { return xSpacing; }
	__host__ __device__ [[nodiscard]] double YSpacing() const { return ySpacing; }
	__host__ __device__ [[nodiscard]] double ZSpacing() const { return zSpacing; }
	__host__ __device__ [[nodiscard]] double CentreX() const { return centreX; }
	__host__ __device__ [[nodiscard]] double CentreY() const { return centreY; }
	__host__ __device__ [[nodiscard]] double CentreZ() const { return centreZ; }
	__host__ __device__ [[nodiscard]] float WidthWorld() const { return static_cast<float>(width) * xSpacing; }
	__host__ __device__ [[nodiscard]] float HeightWorld() const { return static_cast<float>(height) * ySpacing; }
	__host__ __device__ [[nodiscard]] float DepthWorld() const { return static_cast<float>(depth) * zSpacing; }
	__host__ __device__ [[nodiscard]] bool In(const long layer, const long row, const long col) const {
		return layer >= 0 && layer < depth && row >= 0 && row < height && col >= 0 && col < width;
	}

	__host__ __device__ [[nodiscard]] Linear MappingXWorldToNormalised() const {
		const float widthInverse = 1.f / WidthWorld();
		return {.5f - static_cast<float>(centreX) * widthInverse, widthInverse};
	}

	__host__ __device__ [[nodiscard]] Linear MappingYWorldToNormalised() const {
		const float heightInverse = 1.f / HeightWorld();
		return {.5f - static_cast<float>(centreY) * heightInverse, heightInverse};
	}

	__host__ __device__ [[nodiscard]] Linear MappingZWorldToNormalised() const {
		const float depthInverse = 1.f / DepthWorld();
		return {.5f - static_cast<float>(centreZ) * depthInverse, depthInverse};
	}

protected:
	Texture3D() = default;

	Texture3D(long _width, long _height, long _depth, double _xSpacing, double _ySpacing, double _zSpacing,
	          double _centreX = 0., double _centreY = 0., double _centreZ = 0.) : width(_width), height(_height),
		depth(_depth), xSpacing(_xSpacing), ySpacing(_ySpacing), zSpacing(_zSpacing), centreX(_centreX),
		centreY(_centreY), centreZ(_centreZ) {
	}

	// yes copy
	Texture3D(const Texture3D &) = default;

	Texture3D &operator=(const Texture3D &) = default;

	// yes move
	Texture3D(Texture3D &&) noexcept = default;

	Texture3D &operator=(Texture3D &&) noexcept = default;

private:
	long width{};
	long height{};
	long depth{};
	double xSpacing{};
	double ySpacing{};
	double zSpacing{};
	double centreX = 0.;
	double centreY = 0.;
	double centreZ = 0.;
};

} // namespace ExtensionTest