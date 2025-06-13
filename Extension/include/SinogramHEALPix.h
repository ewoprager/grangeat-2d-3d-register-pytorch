#pragma once

#include "Common.h"

namespace reg23 {

/**
 * @ingroup textures
 * @brief A 3D texture stored for access by the CPU, structured for storing an even distribution of values over the
 * surface of S^2, according to the HEALPix mapping. ToDo: citation here
 * @tparam texture_t
 *
 * Texture sampling is done using spherical coordinates $(r, \theta, \phi)$, but only the first dimension $r$ actually
 * corresponds to the texture coordinate in the volume data. Across two dimensions the values are stored according to a
 * rectangular tesselation of the HEALPix spherical projection.
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Both copy and move-constructable.
 */
template <typename texture_t> class SinogramHEALPix : texture_t {
public:
	using Base = texture_t;
	static_assert(Base::DIMENSIONALITY == 3, "The texture type of a `SinogramClassic3D` must have dimensionality 3.");
	using IntType = typename Base::IntType;
	using FloatType = typename Base::FloatType;
	using SizeType = typename Base::SizeType;
	using VectorType = typename Base::VectorType;
	using AddressModeType = typename Base::AddressModeType;

	SinogramHEALPix() = default;

	/**
	 * @brief Construct the texture with data.
	 * @param texture
	 */
	SinogramHEALPix(Base texture) : Base(std::move(texture)) {
	}

	// yes copy
	SinogramHEALPix(const SinogramHEALPix &) = default;

	SinogramHEALPix &operator=(const SinogramHEALPix &) = default;

	// yes move
	SinogramHEALPix(SinogramHEALPix &&) = default;

	SinogramHEALPix &operator=(SinogramHEALPix &&) = default;

	/**
	 * @param tensor A 3D tensor of `torch.float32`s
	 * @param rSpacing The spacing between each HEALPix sphere
	 * @return An instance of this texture object that points to the data in the given image
	 */
	static SinogramHEALPix FromTensor(const at::Tensor &tensor, FloatType rSpacing) {
		// tensor should be a 3D array of floats
		TORCH_CHECK(tensor.sizes().size() == 3)
		TORCH_CHECK(tensor.dtype() == at::kFloat)
		const SizeType tensorSize = SizeType::FromIntArrayRef(tensor.sizes()).Flipped();
		TORCH_CHECK((tensorSize.X() - 4) % 3 == 0)
		TORCH_CHECK((tensorSize.Y() - 4) % 2 == 0)
		TORCH_CHECK((tensorSize.X() - 4) / 3 == (tensorSize.Y() - 4) / 2)
		return SinogramHEALPix{Base::FromTensor(tensor, {1.0, 1.0, rSpacing})};
	}

	/**
	 * @param textureHandle
	 * @param sizeUVR
	 * @param rSpacing The spacing between each HEALPix sphere
	 * @return An instance of this texture object that points to the data in the given image
	 */
	static SinogramHEALPix FromCUDAHandle(int64_t textureHandle, const Vec<int64_t, 3> &sizeUVR, FloatType rSpacing) {
		TORCH_CHECK((sizeUVR.X() - 4) % 3 == 0)
		TORCH_CHECK((sizeUVR.Y() - 4) % 2 == 0)
		TORCH_CHECK((sizeUVR.X() - 4) / 3 == (sizeUVR.Y() - 4) / 2)
		return SinogramHEALPix{Base{textureHandle, sizeUVR, {1.0, 1.0, rSpacing}, VectorType::Full(0)}};
	}

	/**
	 * @param rThetaPhi The location at which to sample in spherical coordinates
	 * @return The sample from this texture at the given coordinates using trilinear interpolation
	 */
	[[nodiscard]] __host__ __device__ float Sample(const VectorType &rThetaPhi) const {
		const FloatType nSideF = static_cast<FloatType>((Base::Size().X() - 4) / 3);

		// to x_s, y_s
		const FloatType phiAdjusted = rThetaPhi[2] + 0.5 * M_PI; // with pi/2 adjustment
		const FloatType z = sin(rThetaPhi[1]); // sin instead of cos for adjustment
		const FloatType zAbs = abs(z);
		const bool equatorialZone = zAbs <= 2.0 / 3.0;
		const FloatType sigma = Sign(z) * (2.0 - sqrt(3.0 * (1.0 - zAbs)));
		const FloatType xS = equatorialZone
			                     ? phiAdjusted
			                     : phiAdjusted - (abs(sigma) - 1.0) * (fmod(phiAdjusted, 0.5 * M_PI) - 0.25 * M_PI);
		const FloatType yS = equatorialZone ? 3.0 * M_PI * z / 8.0 : M_PI * sigma / 4.0;

		// to x_p, y_p
		const FloatType xP = 2.0 * nSideF * xS / M_PI;
		const FloatType yP = nSideF * (1.0 - 2.0 * yS / M_PI);

		// to u, v, r
		FloatType r = rThetaPhi[0];
		FloatType u = xP - yP + 1.5 * nSideF;
		FloatType v = xP + yP - 0.5 * nSideF;
		const bool vHigh = v >= 2.0 * nSideF;
		const bool uHigh = u >= 2.0 * nSideF;
		if (vHigh) {
			// either in base pixel 6 or 9
			v -= 2.0 * nSideF;
			if (uHigh) {
				// in base pixel 6
				u -= 2.0 * nSideF;
#ifdef __CUDACC__
				cuda::
#endif
					std::swap(u, v); // theta is flipped for base pixel 6
				r *= -1.0; // r is flipped for base pixel 6
			} else {
				// in base pixel 9
				u += nSideF + 2.0; // the 2 adjusts for padding
				v -= 2.0; // this adjusts for padding
			}
		}

		// u,v,r is the order of the texture dimensions (X, Y, Z)
		const Vec<FloatType, 2> texCoordOrientation =
			Vec<FloatType, 2>{u + 1.0, v + 3.0} / Base::Size().XY().template StaticCast<FloatType>();
		// the 1 and 3 adjust for padding
		const FloatType texCoordR = .5 + r / (static_cast<FloatType>(Base::Size().Z()) * Base::Spacing().Z());
		return Base::Sample(VecCat(texCoordOrientation, texCoordR));
	}
};

} // namespace reg23