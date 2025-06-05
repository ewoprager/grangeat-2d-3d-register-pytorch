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
		: Base(_ptr, {3 * _nSide, 2 * _nSide, rCount}, {1., 1., rSpacing}), nSide(_nSide) {}

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
		TORCH_CHECK(tensorSize.X() % 3 == 0)
		TORCH_CHECK(tensorSize.Y() % 2 == 0)
		TORCH_CHECK(tensorSize.X() / 3 == tensorSize.Y() / 2)
		return {tensor.contiguous().data_ptr<float>(), tensorSize.X() / 3, tensorSize.Z(), rSpacing};
	}

	/**
	 * @param rThetaPhi The location at which to sample in spherical coordinates
	 * @return The sample from this texture at the given coordinates using trilinear interpolation
	 */
	[[nodiscard]] __host__ __device__ float Sample(VectorType rThetaPhi) const {
		const FloatType nSideF = static_cast<FloatType>(nSide);

		// to x_s, y_s
		const FloatType z = sin(rThetaPhi.Y());
		const FloatType zAbs = abs(z);
		const FloatType sigma = Sign(z) * (2.0 - sqrt(3.0 * (1.0 - zAbs)));
		const FloatType xS = zAbs <= 2.0 / 3.0 ?			// equatorial zone
								 rThetaPhi.Z() + 0.5 * M_PI // with pi/2 adjustment
											   :			// polar cap
								 rThetaPhi.Z() - (abs(sigma) - 1.0) * (fmod(rThetaPhi.Z(), 0.5 * M_PI) - 0.25 * M_PI) +
									 0.5 * M_PI;					  // with pi/2 adjustment
		const FloatType yS = zAbs <= 2.0 / 3.0 ? 2.0 * M_PI * z / 8.0 // equatorial zone
											   : M_PI * sigma / 4.0;  // polar cap

		// to i, j
		const FloatType i = 2.0 * nSideF * (1.0 - 2.0 * yS / M_PI);
		const FloatType j = 2.0 * nSideF * xS / M_PI + 0.5 * fmod(i - nSideF + 1.0, 2.0);

		// to u, v
		FloatType u = j + 1.0 - floor(0.5 * i) + nSideF;
		FloatType v = j + 1.0 + floor(0.5 * (i + 1.0)) - nSideF;
		const bool vHigh = v >= 2.0 * nSideF;
		const bool uHigh = u >= 2.0 * nSideF;
		if (vHigh) {
			v -= 2.0 * nSideF;
			if (uHigh)
				u -= 2.0 * nSideF;
			else
				u += nSideF;
		}

		// u,v,r is the order of the texture dimensions
		const VectorType uvr = {u, v, rThetaPhi.X()};
		return Base::Sample(uvr / (Size() - IntType{1}).StaticCast<FloatType>().Flipped());
	}

  private:
	IntType nSide{}; ///< The number of points per side of a HEALPix base resolution pixel
};

} // namespace reg23