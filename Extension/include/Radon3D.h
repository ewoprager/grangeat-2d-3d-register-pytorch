#pragma once

#include "Common.h"

namespace ExtensionTest {

class Texture3D {
public:
	__host__ __device__ [[nodiscard]] long Width() const { return width; }
	__host__ __device__ [[nodiscard]] long Height() const { return height; }
	__host__ __device__ [[nodiscard]] long Depth() const { return depth; }
	__host__ __device__ [[nodiscard]] float XSpacing() const { return xSpacing; }
	__host__ __device__ [[nodiscard]] float YSpacing() const { return ySpacing; }
	__host__ __device__ [[nodiscard]] float ZSpacing() const { return zSpacing; }
	__host__ __device__ [[nodiscard]] float WidthWorld() const { return static_cast<float>(width) * xSpacing; }
	__host__ __device__ [[nodiscard]] float HeightWorld() const { return static_cast<float>(height) * ySpacing; }
	__host__ __device__ [[nodiscard]] float DepthWorld() const { return static_cast<float>(depth) * zSpacing; }
	__host__ __device__ [[nodiscard]] bool In(const long layer, const long row, const long col) const {
		return layer >= 0 && layer < depth && row >= 0 && row < height && col >= 0 && col < width;
	}

	__host__ __device__ [[nodiscard]] Linear MappingXWorldToNormalised() const { return {.5f, 1.f / WidthWorld()}; }
	__host__ __device__ [[nodiscard]] Linear MappingYWorldToNormalised() const { return {.5f, 1.f / HeightWorld()}; }
	__host__ __device__ [[nodiscard]] Linear MappingZWorldToNormalised() const { return {.5f, 1.f / DepthWorld()}; }

protected:
	Texture3D() = default;

	Texture3D(long _width, long _height, long _depth, float _xSpacing, float _ySpacing,
	          float _zSpacing) : width(_width), height(_height), depth(_depth), xSpacing(_xSpacing),
	                             ySpacing(_ySpacing), zSpacing(_zSpacing) {
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
	float xSpacing{};
	float ySpacing{};
	float zSpacing{};
};

template <typename texture_t> struct Radon3D {
	struct ConstMappings {
		Linear mappingIndexToOffset;
		Linear mappingColToR;
		Linear mappingRowToPhi;
		Linear mappingLayerToTheta;
	};

	__host__ __device__ [[nodiscard]] static ConstMappings GetConstMappings(
		long widthOut, long heightOut, long depthOut, float planeSize, long samplesPerDirection) {

		return {{-.5f * planeSize, planeSize / static_cast<float>(samplesPerDirection - 1)},
		        {-.5f * planeSize, planeSize / static_cast<float>(widthOut - 1)},
		        {-.5f * 3.1415926535f, 3.1415926535f / static_cast<float>(heightOut)},
		        {-.5f * 3.1415926535f, 3.1415926535f / static_cast<float>(depthOut)}};
	}

	struct IndexMappings {
		Linear2 mappingIJToX;
		Linear2 mappingIJToY;
		Linear2 mappingIJToZ;
	};

	__host__ __device__ [[nodiscard]] static IndexMappings GetIndexMappings(
		const texture_t &textureIn, long colOut, long rowOut, long layerOut, const ConstMappings &constMappings) {

		const float theta = constMappings.mappingLayerToTheta(layerOut);
		const float phi = constMappings.mappingRowToPhi(rowOut);
		const float r = constMappings.mappingColToR(colOut);
		const float sp = sinf(phi);
		const float cp = cosf(phi);
		const float st = sinf(theta);
		const float ct = cosf(theta);
		const Linear2 mappingOffsetToWorldX{r * ct * cp, -sp, -st * cp};
		const Linear2 mappingOffsetToWorldY{r * ct * sp, cp, -st * sp};
		const Linear2 mappingOffsetToWorldZ{r * st, 0.f, ct};
		return {textureIn.MappingXWorldToNormalised()(mappingOffsetToWorldX(constMappings.mappingIndexToOffset)),
		        textureIn.MappingYWorldToNormalised()(mappingOffsetToWorldY(constMappings.mappingIndexToOffset)),
		        textureIn.MappingZWorldToNormalised()(mappingOffsetToWorldZ(constMappings.mappingIndexToOffset))};
	}

	__host__ __device__ [[nodiscard]] static float IntegrateLooped(const texture_t &texture,
	                                                               const IndexMappings &indexMappings,
	                                                               long samplesPerDirection) {
		float ret = 0.f;
		for (long j = 0; j < samplesPerDirection; ++j) {
			for (long i = 0; i < samplesPerDirection; ++i) {
				const float iF = static_cast<float>(i);
				const float jF = static_cast<float>(j);
				ret += texture.Sample(indexMappings.mappingIJToX(iF, jF), indexMappings.mappingIJToY(iF, jF),
				                      indexMappings.mappingIJToZ(iF, jF));
			}
		}
		return ret;
	}

};

} // namespace ExtensionTest