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

	Texture3D(long _width, long _height, long _depth, double _xSpacing, double _ySpacing,
	          double _zSpacing) : width(_width), height(_height), depth(_depth), xSpacing(_xSpacing),
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
	double xSpacing{};
	double ySpacing{};
	double zSpacing{};
};

/**
 * Cartesian coordinates:
 *	- Origin at centre of volume
 *	- x, y and z are right-handed
 *
 * Radial coordinates:
 *	- Origin at centre of image
 *	- r is distance from origin
 *	- theta is radians left-hand-rule around y-axis from the positive x-direction
 *	- phi is radians right-hand-rule around z-axis from the positive x-direction
 *
 * @tparam texture_t
 */
template <typename texture_t> struct Radon3D {

	__host__ __device__ [[nodiscard]] static Linear GetMappingIToOffset(float planeSize, long samplesPerDirection) {

		return {-.5f * planeSize, planeSize / static_cast<float>(samplesPerDirection - 1)};
	}

	struct IndexMappings {
		Linear2 mappingIJToX;
		Linear2 mappingIJToY;
		Linear2 mappingIJToZ;
	};

	struct DerivativeWRTR {
		float dXdR;
		float dYdR;
		float dZdR;
	};

	__host__ __device__ [[nodiscard]] static IndexMappings GetIndexMappings(
		const texture_t &textureIn, float phi, float theta, float r, const Linear &mappingIToOffset) {
		const float sp = sinf(phi);
		const float cp = cosf(phi);
		const float st = sinf(theta);
		const float ct = cosf(theta);
		const Linear2 mappingOffsetToWorldX{r * ct * cp, -sp, -st * cp};
		const Linear2 mappingOffsetToWorldY{r * ct * sp, cp, -st * sp};
		const Linear2 mappingOffsetToWorldZ{r * st, 0.f, ct};
		return {textureIn.MappingXWorldToNormalised()(mappingOffsetToWorldX(mappingIToOffset)),
		        textureIn.MappingYWorldToNormalised()(mappingOffsetToWorldY(mappingIToOffset)),
		        textureIn.MappingZWorldToNormalised()(mappingOffsetToWorldZ(mappingIToOffset))};
	}

	__host__ __device__ [[nodiscard]] static DerivativeWRTR GetDerivativeWRTR(
		const texture_t &textureIn, float phi, float theta, float r) {
		const float sign = static_cast<float>(r > 0.f) - static_cast<float>(r < 0.f);
		const float sp = sinf(phi);
		const float cp = cosf(phi);
		const float st = sinf(theta);
		const float ct = cosf(theta);
		return {textureIn.MappingXWorldToNormalised().gradient * sign * ct * cp,
		        textureIn.MappingYWorldToNormalised().gradient * sign * ct * sp,
		        textureIn.MappingZWorldToNormalised().gradient * sign * st};
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

	__host__ __device__ [[nodiscard]] static float DIntegrateLoopedDMappingParameter(const texture_t &texture,
		const IndexMappings &indexMappings, const DerivativeWRTR &derivativeWRTR, long samplesPerDirection) {
		float ret = 0.f;
		for (long j = 0; j < samplesPerDirection; ++j) {
			for (long i = 0; i < samplesPerDirection; ++i) {
				const float iF = static_cast<float>(i);
				const float jF = static_cast<float>(j);
				const float x = indexMappings.mappingIJToX(iF, jF);
				const float y = indexMappings.mappingIJToY(iF, jF);
				const float z = indexMappings.mappingIJToZ(iF, jF);
				ret += texture.SampleXDerivative(x, y, z) * derivativeWRTR.dXdR + texture.SampleYDerivative(x, y, z) *
					derivativeWRTR.dYdR + texture.SampleZDerivative(x, y, z) * derivativeWRTR.dZdR;
			}
		}
		return ret;
	}

};

at::Tensor radon3d_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                       const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                       long samplesPerDirection);

at::Tensor dRadon3dDR_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                          const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                          long samplesPerDirection);

__host__ at::Tensor radon3d_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                 const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                                 long samplesPerDirection);

__host__ at::Tensor dRadon3dDR_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection);

__host__ at::Tensor radon3d_v2_cuda(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                                    const at::Tensor &phiValues, const at::Tensor &thetaValues,
                                    const at::Tensor &rValues, long samplesPerDirection);

} // namespace ExtensionTest