#pragma once

#include "Common.h"

namespace ExtensionTest {

class Texture2D {
public:
	__host__ __device__ [[nodiscard]] long Width() const { return width; }
	__host__ __device__ [[nodiscard]] long Height() const { return height; }
	__host__ __device__ [[nodiscard]] double XSpacing() const { return xSpacing; }
	__host__ __device__ [[nodiscard]] double YSpacing() const { return ySpacing; }
	__host__ __device__ [[nodiscard]] float WidthWorld() const { return static_cast<float>(width) * xSpacing; }
	__host__ __device__ [[nodiscard]] float HeightWorld() const { return static_cast<float>(height) * ySpacing; }
	__host__ __device__ [[nodiscard]] bool In(const long row, const long col) const {
		return row >= 0 && row < height && col >= 0 && col < width;
	}

	__host__ __device__ [[nodiscard]] Linear MappingXWorldToNormalised() const { return {.5f, 1.f / WidthWorld()}; }
	__host__ __device__ [[nodiscard]] Linear MappingYWorldToNormalised() const { return {.5f, -1.f / HeightWorld()}; }

protected:
	Texture2D() = default;

	Texture2D(long _width, long _height, double _xSpacing, double _ySpacing) : width(_width), height(_height),
	                                                                           xSpacing(_xSpacing),
	                                                                           ySpacing(_ySpacing) {
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
	double xSpacing{};
	double ySpacing{};
};

template <typename texture_t> struct Radon2D {
	struct ConstMappings {
		Linear mappingIToOffset;
		Linear mappingColToR;
		Linear mappingRowToPhi;
	};

	__host__ __device__ [[nodiscard]] static ConstMappings GetConstMappings(
		long widthOut, long heightOut, float rayLength, long samplesPerLine) {

		return {{-.5f * rayLength, rayLength / static_cast<float>(samplesPerLine - 1)},
		        {-.5f * rayLength, rayLength / static_cast<float>(widthOut - 1)},
		        {-.5f * 3.1415926535f, 3.1415926535f / static_cast<float>(heightOut)}};
	}

	struct IndexMappings {
		Linear mappingIToX;
		Linear mappingIToY;
	};

	__host__ __device__ [[nodiscard]] static IndexMappings GetIndexMappings(
		const texture_t &textureIn, long colOut, long rowOut, const ConstMappings &constMappings) {
		const float phi = constMappings.mappingRowToPhi(rowOut);
		const float r = constMappings.mappingColToR(colOut);
		const float s = sinf(phi);
		const float c = cosf(phi);
		const Linear mappingOffsetToWorldX{r * c, -s};
		const Linear mappingOffsetToWorldY{r * s, c};
		return {textureIn.MappingXWorldToNormalised()(mappingOffsetToWorldX(constMappings.mappingIToOffset)),
		        textureIn.MappingYWorldToNormalised()(mappingOffsetToWorldY(constMappings.mappingIToOffset))};
	}

	__host__ __device__ [[nodiscard]] static IndexMappings GetDIndexMappingsDR(
		const texture_t &textureIn, long rowOut, const ConstMappings &constMappings) {
		const float phi = constMappings.mappingRowToPhi(rowOut);
		const float s = sinf(phi);
		const float c = cosf(phi);
		const Linear dMappingOffsetToWorldXDR{c, 0.f};
		const Linear dMappingOffsetToWorldYDR{s, 0.f};
		return {textureIn.MappingXWorldToNormalised()(dMappingOffsetToWorldXDR(constMappings.mappingIToOffset)),
		        textureIn.MappingYWorldToNormalised()(dMappingOffsetToWorldYDR(constMappings.mappingIToOffset))};
	}

	__host__ __device__ [[nodiscard]] static float IntegrateLooped(const texture_t &texture,
	                                                               const IndexMappings &indexMappings,
	                                                               long samplesPerLine) {
		float ret = 0.f;
		for (long i = 0; i < samplesPerLine; ++i) {
			const float iF = static_cast<float>(i);
			ret += texture.Sample(indexMappings.mappingIToX(iF), indexMappings.mappingIToY(iF));
		}
		return ret;
	}

	__host__ __device__ [[nodiscard]] static float DIntegrateLoopedDMappingParameter(const texture_t &texture,
		const IndexMappings &indexMappings, const IndexMappings &dIndexMappingsDParameter, long samplesPerLine) {
		float ret = 0.f;
		for (long i = 0; i < samplesPerLine; ++i) {
			const float iF = static_cast<float>(i);
			const float x = indexMappings.mappingIToX(iF);
			const float y = indexMappings.mappingIToY(iF);
			ret += texture.SampleXDerivative(x, y) * dIndexMappingsDParameter.mappingIToX(iF) + texture.
				SampleYDerivative(x, y) * dIndexMappingsDParameter.mappingIToY(iF);
		}
		return ret;
	}

};

at::Tensor radon2d_cpu(const at::Tensor &a, double xSpacing, double ySpacing, long heightOut, long widthOut,
                       long samplesPerLine);

at::Tensor radon2d_v2_cpu(const at::Tensor &a, double xSpacing, double ySpacing, long heightOut, long widthOut,
                          long samplesPerLine);

at::Tensor dRadon2dDR_cpu(const at::Tensor &a, double xSpacing, double ySpacing, long heightOut, long widthOut,
                          long samplesPerLine);

__host__ at::Tensor radon2d_cuda(const at::Tensor &a, double xSpacing, double ySpacing, long heightOut, long widthOut,
                                 long samplesPerLine);

__host__ at::Tensor radon2d_v2_cuda(const at::Tensor &a, double xSpacing, double ySpacing, long heightOut,
                                    long widthOut, long samplesPerLine);

} // namespace ExtensionTest