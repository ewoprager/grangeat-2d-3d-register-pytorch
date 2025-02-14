#pragma once

#include "Texture3D.h"

namespace ExtensionTest {

at::Tensor ResampleRadonVolume_cpu(const at::Tensor &sinogram3d, double phiMinS, double phiMaxS, double thetaMinS,
                                   double thetaMaxS, double rMinS, double rMaxS, const at::Tensor &projectionMatrix,
                                   const at::Tensor &phiGrid, const at::Tensor &rGrid);

class Texture3DCPU : public Texture<3, int, float> {
public:
  	using Base = Texture<3, int, float>;

	Texture3DCPU() = default;

	Texture3DCPU(const float *_ptr,SizeType _size, VectorType _spacing,
				 VectorType _centrePosition = {}) : Base(_size, _spacing, _centrePosition), ptr(_ptr) {
	}

	// yes copy
	Texture3DCPU(const Texture3DCPU &) = default;

	Texture3DCPU &operator=(const Texture3DCPU &) = default;

	// yes move
	Texture3DCPU(Texture3DCPU &&) = default;

	Texture3DCPU &operator=(Texture3DCPU &&) = default;

	__host__ __device__ [[nodiscard]] float At(const SizeType &index) const {
		return In(index) ? ptr[index.Z() * Size().X() * Size().Y() + index.Y() * Size().X() + index.X()] : 0.0f;
	}

	/**
	 * @brief Sample the texture using linear interpolation.
	 * @param x sampling coordinate x in texture coordinates (0, 1)
	 * @param y sampling coordinate y in texture coordinates (0, 1)
	 * @param z sampling coordinate z in texture coordinates (0, 1)
	 * @return sample from texture at given coordinate
	 */
	__host__ __device__ [[nodiscard]] float Sample(VectorType texCoord) const {
		texCoord = texCoord * VecCast<float>(Size()) - .5f;
		const VectorType floored = VecApply(&floorf, texCoord);
		const SizeType index = VecCast<int>(floored);
		const VectorType fractions = texCoord - floored;
		const float l0r0 = (1.f - fHorizontal) * At(index) + fHorizontal * At({index.X() + 1, index.Y(), index.Z()});
		const float l0r1 = (1.f - fHorizontal) * At({index.X(), index.Y() + 1, index.Z()}) + fHorizontal * At({index.X() + 1, index.Y() + 1, index.Z()});
		const float l1r0 = (1.f - fHorizontal) * At({index.X(), index.Y(), index.Z() + 1}) + fHorizontal * At({index.X() + 1, index.Y(), index.Z() + 1});
		const float l1r1 = (1.f - fHorizontal) * At({index.X(), index.Y() + 1, index.Z() + 1}) + fHorizontal * At(
			                   {index.X() + 1, index.Y() + 1, index.Z() + 1});
		const float l0 = (1.f - fVertical) * l0r0 + fVertical * l0r1;
		const float l1 = (1.f - fVertical) * l1r0 + fVertical * l1r1;
		return (1.f - fInward) * l0 + fInward * l1;
	}

	__host__ __device__ [[nodiscard]] float SampleXDerivative(VectorType texCoord) const {
		const VectorType sizeF = VecCast<float>(Size());
		texCoord = texCoord * sizeF - .5f;
		const VectorType floored = VecApply(&floorf, texCoord);
		const SizeType index = VecCast<int>(floored);
		const float fVertical = texCoord.Y() - floored.Y();
		const float fInward = texCoord.Z() - floored.Z();
		const float l0 = (1.f - fVertical) * (At({index.X() + 1, index.Y(), index.Z()}) - At(index)) + fVertical * (
			                 At({index.X() + 1, index.Y() + 1, index.Z()}) - At({index.X(), index.Y() + 1, index.Z()}));
		const float l1 = (1.f - fVertical) * (At({index.X() + 1, index.Y(), index.Z() + 1}) - At({index.X(), index.Y(), index.Z() + 1})) + fVertical * (
			                 At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At({index.X(), index.Y() + 1, index.Z() + 1}));
		return widthF * ((1.f - fInward) * l0 + fInward * l1);
	}

	__host__ __device__ [[nodiscard]] float SampleYDerivative(VectorType texCoord) const {
		const VectorType sizeF = VecCast<float>(Size());
		texCoord = texCoord * sizeF - .5f;
		const VectorType floored = VecApply(&floorf, texCoord);
		const SizeType index = VecCast<int>(floored);
		const float fHorizontal = texCoord.X() - floored.X();
		const float fInward = texCoord.Z() - floored.Z();
		const float l0 = (1.f - fHorizontal) * (At({index.X(), index.Y() + 1, index.Z()}) - At(index)) + fHorizontal * (
			                 At({index.X() + 1, index.Y() + 1, index.Z()}) - At({index.X() + 1, index.Y(), index.Z()}));
		const float l1 = (1.f - fHorizontal) * (At({index.X(), index.Y() + 1, index.Z() + 1}) - At({index.X(), index.Y(), index.Z() + 1})) + fHorizontal * (
			                 At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At({index.X() + 1, index.Y(), index.Z() + 1}));
		return heightF * ((1.f - fInward) * l0 + fInward * l1);
	}

	__host__ __device__ [[nodiscard]] float SampleZDerivative(VectorType texCoord) const {
		const VectorType sizeF = VecCast<float>(Size());
		texCoord = texCoord * sizeF - .5f;
		const VectorType floored = VecApply(&floorf, texCoord);
		const SizeType index = VecCast<int>(floored);
		const float fHorizontal = texCoord.X() - floored.X();
		const float fVertical = texCoord.Y() - floored.Y();
		const float r0 = (1.f - fHorizontal) * (At({index.X(), index.Y(), index.Z() + 1}) - At(index)) + fHorizontal * (
			                 At({index.X() + 1, index.Y(), index.Z() + 1}) - At({index.X() + 1, index.Y(), index.Z()}));
		const float r1 = (1.f - fHorizontal) * (At({index.X(), index.Y() + 1, index.Z() + 1}) - At({index.X(), index.Y() + 1, index.Z()})) + fHorizontal * (
			                 At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At({index.X() + 1, index.Y() + 1, index.Z()}));
		return depthF * ((1.f - fVertical) * r0 + fVertical * r1);
	}

private:
	const float *ptr{};
};

} // namespace ExtensionTest