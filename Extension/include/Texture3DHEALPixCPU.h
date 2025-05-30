#pragma once

#include "Texture.h"

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
class Texture3DHEALPixCPU : public Texture<3, int64_t, double> {
public:
	using Base = Texture<3, int64_t, double>;

	Texture3DHEALPixCPU() = default;

	/**
	 * @brief Construct the texture with data.
	 * @param _ptr A pointer to the beginning of the data
	 * @param _nSide $N_\mathrm{side}$: The number of pixels along a side of a HEALPix base resolution pixel
	 * @param rCount The size of the volumes first dimension; the number of $r$ values
	 * @param rSpacing The spacing in world coordinates of the layers in the $r$ direction
	 * @param rRangeCentre The centre in world coordinates of the layers in the $r$ direction
	 * @param addressMode The address mode with which the texture should be sampled in the $r$-direction
	 */
	Texture3DHEALPixCPU(const float *_ptr, IntType _nSide, IntType rCount, FloatType rSpacing, FloatType rRangeCentre,
	                    TextureAddressMode addressMode = TextureAddressMode::ZERO)
		: Base(SizeType{rCount, 4 * _nSide, 3 * _nSide,}, VectorType{rSpacing, 0.0, 0.0},
		       VectorType{rRangeCentre, 0.0, 0.0}), ptr(_ptr), addressModes(AddressModeType::Full(addressMode)),
		  nSide(_nSide) {
	}

	// yes copy
	Texture3DHEALPixCPU(const Texture3DHEALPixCPU &) = default;

	Texture3DHEALPixCPU &operator=(const Texture3DHEALPixCPU &) = default;

	// yes move
	Texture3DHEALPixCPU(Texture3DHEALPixCPU &&) = default;

	Texture3DHEALPixCPU &operator=(Texture3DHEALPixCPU &&) = default;

	/**
	 * @param volume A 3D tensor of `torch.float32`s
	 * @param rSpacing The spacing between each HEALPix sphere
	 * @return An instance of this texture object that points to the data in the given image
	 */
	static Texture3DHEALPixCPU FromTensor(const at::Tensor &volume, FloatType rSpacing) {
		return {volume.contiguous().data_ptr<float>(), Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Flipped(),
		        rSpacing};
	}

	/**
	 * @param index The location in the texture at which to fetch the value
	 * @return The value in this texture at the given location, according to the texture's address mode
	 */
	[[nodiscard]] __host__ __device__ float At(const SizeType &index) const {
		if ((addressModes.X() == TextureAddressMode::ZERO && (index.X() < 0 || index.X() >= Size().X())) || (
			    addressModes.Y() == TextureAddressMode::ZERO && (index.Y() < 0 || index.Y() >= Size().Y())) || (
			    addressModes.Z() == TextureAddressMode::ZERO && (index.Z() < 0 || index.Z() >= Size().Z()))) {
			return 0.f;
		}
		// Uses wrapping for indices outside the texture.
		return ptr[Modulo(index.Z(), Size().Z()) * Size().X() * Size().Y() + Modulo(index.Y(), Size().Y()) * Size().X()
		           + Modulo(index.X(), Size().X())];
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
		const VectorType texCoord = ruv.Flipped() * Size().StaticCast<double>() - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const VectorType fractions = texCoord - floored;
		const float l0r0 = (1.f - fractions.X()) * At(index) + fractions.X() *
		                   At({index.X() + 1, index.Y(), index.Z()});
		const float l0r1 = (1.f - fractions.X()) * At({index.X(), index.Y() + 1, index.Z()}) + fractions.X() * At(
			                   {index.X() + 1, index.Y() + 1, index.Z()});
		const float l1r0 = (1.f - fractions.X()) * At({index.X(), index.Y(), index.Z() + 1}) + fractions.X() * At(
			                   {index.X() + 1, index.Y(), index.Z() + 1});
		const float l1r1 = (1.f - fractions.X()) * At({index.X(), index.Y() + 1, index.Z() + 1}) + fractions.X() * At(
			                   {index.X() + 1, index.Y() + 1, index.Z() + 1});
		const float l0 = (1.f - fractions.Y()) * l0r0 + fractions.Y() * l0r1;
		const float l1 = (1.f - fractions.Y()) * l1r0 + fractions.Y() * l1r1;
		return (1.f - fractions.Z()) * l0 + fractions.Z() * l1;
	}

	// /**
	//  * @param texCoord The texture coordinates at which to sample the texture
	//  * @return The derivative of the sample from this texture at the given texture coordinates using trilinear
	//  * interpolation, with respect to the X-component of the texture coordinate.
	//  */
	// [[nodiscard]] __host__ __device__ float DSampleDX(VectorType texCoord) const {
	// 	const VectorType sizeF = Size().StaticCast<double>();
	// 	texCoord = texCoord * sizeF - .5;
	// 	const VectorType floored = texCoord.Apply<double>(&floor);
	// 	const SizeType index = floored.StaticCast<int64_t>();
	// 	const float fVertical = texCoord.Y() - floored.Y();
	// 	const float fInward = texCoord.Z() - floored.Z();
	// 	const float l0 = (1.f - fVertical) * (At({index.X() + 1, index.Y(), index.Z()}) - At(index)) + fVertical * (
	// 		                 At({index.X() + 1, index.Y() + 1, index.Z()}) - At({index.X(), index.Y() + 1, index.Z()}));
	// 	const float l1 = (1.f - fVertical) * (At({index.X() + 1, index.Y(), index.Z() + 1}) -
	// 	                                      At({index.X(), index.Y(), index.Z() + 1})) + fVertical * (
	// 		                 At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At(
	// 			                 {index.X(), index.Y() + 1, index.Z() + 1}));
	// 	return sizeF.X() * ((1.f - fInward) * l0 + fInward * l1);
	// }
	//
	// /**
	//  * @param texCoord The texture coordinates at which to sample the texture
	//  * @return The derivative of the sample from this texture at the given texture coordinates using trilinear
	//  * interpolation, with respect to the Y-component of the texture coordinate.
	//  */
	// [[nodiscard]] __host__ __device__ float DSampleDY(VectorType texCoord) const {
	// 	const VectorType sizeF = Size().StaticCast<double>();
	// 	texCoord = texCoord * sizeF - .5;
	// 	const VectorType floored = texCoord.Apply<double>(&floor);
	// 	const SizeType index = floored.StaticCast<int64_t>();
	// 	const float fHorizontal = texCoord.X() - floored.X();
	// 	const float fInward = texCoord.Z() - floored.Z();
	// 	const float l0 = (1.f - fHorizontal) * (At({index.X(), index.Y() + 1, index.Z()}) - At(index)) + fHorizontal * (
	// 		                 At({index.X() + 1, index.Y() + 1, index.Z()}) - At({index.X() + 1, index.Y(), index.Z()}));
	// 	const float l1 = (1.f - fHorizontal) * (At({index.X(), index.Y() + 1, index.Z() + 1}) -
	// 	                                        At({index.X(), index.Y(), index.Z() + 1})) + fHorizontal * (
	// 		                 At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At(
	// 			                 {index.X() + 1, index.Y(), index.Z() + 1}));
	// 	return sizeF.Y() * ((1.f - fInward) * l0 + fInward * l1);
	// }
	//
	// /**
	//  * @param texCoord The texture coordinates at which to sample the texture
	//  * @return The derivative of the sample from this texture at the given texture coordinates using trilinear
	//  * interpolation, with respect to the Z-component of the texture coordinate.
	//  */
	// [[nodiscard]] __host__ __device__ float DSampleDZ(VectorType texCoord) const {
	// 	const VectorType sizeF = Size().StaticCast<double>();
	// 	texCoord = texCoord * sizeF - .5;
	// 	const VectorType floored = texCoord.Apply<double>(&floor);
	// 	const SizeType index = floored.StaticCast<int64_t>();
	// 	const float fHorizontal = texCoord.X() - floored.X();
	// 	const float fVertical = texCoord.Y() - floored.Y();
	// 	const float r0 = (1.f - fHorizontal) * (At({index.X(), index.Y(), index.Z() + 1}) - At(index)) + fHorizontal * (
	// 		                 At({index.X() + 1, index.Y(), index.Z() + 1}) - At({index.X() + 1, index.Y(), index.Z()}));
	// 	const float r1 = (1.f - fHorizontal) * (At({index.X(), index.Y() + 1, index.Z() + 1}) -
	// 	                                        At({index.X(), index.Y() + 1, index.Z()})) + fHorizontal * (
	// 		                 At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At(
	// 			                 {index.X() + 1, index.Y() + 1, index.Z()}));
	// 	return sizeF.Z() * ((1.f - fVertical) * r0 + fVertical * r1);
	// }

private:
	const float *ptr{}; ///< The pointer to the data this texture provides access to
	AddressModeType addressModes{}; ///< The address mode of the texture for each dimension
	IntType nSide{};
};

} // namespace reg23