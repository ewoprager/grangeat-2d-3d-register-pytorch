#pragma once

#include "Texture3DCPU.h"

namespace reg23 {

/**
 * @ingroup textures
 * @brief A 3D texture stored for access by the CPU, structured for storing an even distribution of values over the
 * surface of S^2, according to the HEALPix mapping. ToDo: citation here
 *
 * Texture sampling is done using spherical coordinates $(r, \theta, \phi)$, but only the first dimension $r$ actually
 * corresponds to the texture coordinate in the volume data. Across two dimensions the values are stored according to a
 * rectangular tesselation of the HEALPix spherical projection.
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Both copy and move-constructable.
 */
class SinogramHEALPixCPU : Texture3DCPU {
public:
	using Base = Texture3DCPU;

	SinogramHEALPixCPU() = default;

	/**
	 * @brief Construct the texture with data.
	 * @param _ptr A pointer to the beginning of the data
	 * @param _nSide $N_\mathrm{side}$: The number of pixels along a side of a HEALPix base resolution pixel
	 * @param rCount The size of the volumes first dimension; the number of $r$ values
	 * @param rSpacing The spacing in world coordinates of the layers in the $r$ direction
	 */
	SinogramHEALPixCPU(const float *_ptr, IntType _nSide, IntType rCount, FloatType rSpacing)
		: Base(_ptr, {4 * _nSide, 3 * _nSide, rCount}, {1., 1., rSpacing}), nSide(_nSide) {
	}

	// yes copy
	SinogramHEALPixCPU(const SinogramHEALPixCPU &) = default;

	SinogramHEALPixCPU &operator=(const SinogramHEALPixCPU &) = default;

	// yes move
	SinogramHEALPixCPU(SinogramHEALPixCPU &&) = default;

	SinogramHEALPixCPU &operator=(SinogramHEALPixCPU &&) = default;

	/**
	 * @param tensor A 3D tensor of `torch.float32`s
	 * @param rSpacing The spacing between each HEALPix sphere
	 * @return An instance of this texture object that points to the data in the given image
	 */
	static SinogramHEALPixCPU FromTensor(const at::Tensor &tensor, FloatType rSpacing) {
		const SizeType tensorSize = SizeType::FromIntArrayRef(tensor.sizes()).Flipped();
		TORCH_CHECK(tensorSize.X() % 4 == 0)
		TORCH_CHECK(tensorSize.Y() % 3 == 0)
		TORCH_CHECK(tensorSize.X() / 4 == tensorSize.Y() / 3)
		return {tensor.contiguous().data_ptr<float>(), tensorSize.X() / 4, tensorSize.Z(), rSpacing};
	}

	/**
	 * @param rThetaPhi The location at which to sample in spherical coordinates
	 * @return The sample from this texture at the given coordinates using trilinear interpolation
	 */
	[[nodiscard]] __host__ __device__ float Sample(VectorType rThetaPhi) const {
		const FloatType z = sin(rThetaPhi.Y());
		const FloatType nSideF = static_cast<FloatType>(nSide);
		const FloatType i = nSideF * (2.0 - 1.5 * z);
		FloatType jEquiv;
		if (z > 0.666666666666666667) {
			// HEALPix top polar cap
			const FloatType j = 2.0 * i * rThetaPhi.Z() / M_PI + 0.5;
			jEquiv = nSideF * floor((j - 1.0) / i) + 1.0 + floor(0.5 * (nSideF - 1.0)) + fmod(j - 1.0, i);
		} else if (z > -0.66666666666667) {
			// HEALPix equatorial zone
			const FloatType j = 2.0 * nSideF * rThetaPhi.Z() / M_PI + 0.5 * fmod(i - nSideF + 1.0, 2.0);
			jEquiv = j;
		} else {
			// HEALPix bottom polar cap
			const FloatType j = 2.0 * (4.0 * nSideF - i) * rThetaPhi.Z() / M_PI + 0.5;
			jEquiv = nSideF * floor((j - 1.0) / (4.0 * nSideF - i)) + 1.0 + floor(0.5 * (nSideF - 1.0)) + fmod(
				         j - 1.0, 4.0 * nSideF - i);
		}
		const FloatType deltaV48 = jEquiv - floor(0.5 * i) + 1.0 < 0.0 ? nSideF : 0.0;
		// r,u,v is the order of the texture dimensions
		const VectorType ruv = {rThetaPhi.X(), fmod(jEquiv - floor(0.5 * i) + 1.0, 4.0 * nSideF),
		                        fmod(jEquiv + floor(0.5 * (i + 1.0)) - 3.0 + deltaV48, 3.0 * nSideF)};
		// texCoord is in the reverse order: (X, Y, Z)
		return Base::Sample(ruv.Flipped());
	}

private:
	IntType nSide{}; ///< The number of points per side of a HEALPix base resolution pixel
};

} // namespace reg23