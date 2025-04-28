#pragma once

#include "Texture.h"

namespace ExtensionTest {

/**
 * @brief A 3D texture stored for access by the CPU
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Both copy and move-constructable.
 */
class Texture3DCPU : public Texture<3, int64_t, double> {
  public:
	using Base = Texture<3, int64_t, double>;

	Texture3DCPU() = default;

	Texture3DCPU(const float *_ptr, SizeType _size, VectorType _spacing, VectorType _centrePosition = {},
				 AddressModeType _addressModes = AddressModeType::Full(TextureAddressMode::ZERO))
		: Base(std::move(_size), std::move(_spacing), std::move(_centrePosition)), ptr(_ptr),
		  addressModes(_addressModes) {}

	// yes copy
	Texture3DCPU(const Texture3DCPU &) = default;

	Texture3DCPU &operator=(const Texture3DCPU &) = default;

	// yes move
	Texture3DCPU(Texture3DCPU &&) = default;

	Texture3DCPU &operator=(Texture3DCPU &&) = default;

	/**
	 * @param volume A 3D tensor of `torch.float32`s
	 * @param spacing A tensor of `torch.float64`s of size (3,)
	 * @return An instance of this texture object that points to the data in the given image
	 */
	static Texture3DCPU FromTensor(const at::Tensor &volume, const at::Tensor &spacing) {
		return {volume.contiguous().data_ptr<float>(), Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Flipped(),
				Vec<double, 3>::FromTensor(spacing)};
	}

	/**
	 * @param index The location in the texture at which to fetch the value
	 * @return The value in this texture at the given location, according to the texture's address mode
	 */
	[[nodiscard]] __host__ __device__ float At(const SizeType &index) const {
		if ((addressModes.X() == TextureAddressMode::ZERO && (index.X() < 0 || index.X() >= Size().X())) ||
			(addressModes.Y() == TextureAddressMode::ZERO && (index.Y() < 0 || index.Y() >= Size().Y())) ||
			(addressModes.Z() == TextureAddressMode::ZERO && (index.Z() < 0 || index.Z() >= Size().Z()))) {
			return 0.f;
		}
		// Uses wrapping for indices outside the texture.
		return ptr[Modulo(index.Z(), Size().Z()) * Size().X() * Size().Y() +
				   Modulo(index.Y(), Size().Y()) * Size().X() + Modulo(index.X(), Size().X())];
	}

	/**
	 * @param texCoord The texture coordinates at which to sample the texture
	 * @return The sample from this texture at the given texture coordinates using trilinear interpolation
	 */
	[[nodiscard]] __host__ __device__ float Sample(VectorType texCoord) const {
		texCoord = texCoord * Size().StaticCast<double>() - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const VectorType fractions = texCoord - floored;
		const float l0r0 =
			(1.f - fractions.X()) * At(index) + fractions.X() * At({index.X() + 1, index.Y(), index.Z()});
		const float l0r1 = (1.f - fractions.X()) * At({index.X(), index.Y() + 1, index.Z()}) +
						   fractions.X() * At({index.X() + 1, index.Y() + 1, index.Z()});
		const float l1r0 = (1.f - fractions.X()) * At({index.X(), index.Y(), index.Z() + 1}) +
						   fractions.X() * At({index.X() + 1, index.Y(), index.Z() + 1});
		const float l1r1 = (1.f - fractions.X()) * At({index.X(), index.Y() + 1, index.Z() + 1}) +
						   fractions.X() * At({index.X() + 1, index.Y() + 1, index.Z() + 1});
		const float l0 = (1.f - fractions.Y()) * l0r0 + fractions.Y() * l0r1;
		const float l1 = (1.f - fractions.Y()) * l1r0 + fractions.Y() * l1r1;
		return (1.f - fractions.Z()) * l0 + fractions.Z() * l1;
	}

	/**
	 * @param texCoord The texture coordinates at which to sample the texture
	 * @return The derivative of the sample from this texture at the given texture coordinates using trilinear
	 * interpolation, with respect to the X-component of the texture coordinate.
	 */
	[[nodiscard]] __host__ __device__ float DSampleDX(VectorType texCoord) const {
		const VectorType sizeF = Size().StaticCast<double>();
		texCoord = texCoord * sizeF - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const float fVertical = texCoord.Y() - floored.Y();
		const float fInward = texCoord.Z() - floored.Z();
		const float l0 =
			(1.f - fVertical) * (At({index.X() + 1, index.Y(), index.Z()}) - At(index)) +
			fVertical * (At({index.X() + 1, index.Y() + 1, index.Z()}) - At({index.X(), index.Y() + 1, index.Z()}));
		const float l1 =
			(1.f - fVertical) *
				(At({index.X() + 1, index.Y(), index.Z() + 1}) - At({index.X(), index.Y(), index.Z() + 1})) +
			fVertical *
				(At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At({index.X(), index.Y() + 1, index.Z() + 1}));
		return sizeF.X() * ((1.f - fInward) * l0 + fInward * l1);
	}

	/**
	 * @param texCoord The texture coordinates at which to sample the texture
	 * @return The derivative of the sample from this texture at the given texture coordinates using trilinear
	 * interpolation, with respect to the Y-component of the texture coordinate.
	 */
	[[nodiscard]] __host__ __device__ float DSampleDY(VectorType texCoord) const {
		const VectorType sizeF = Size().StaticCast<double>();
		texCoord = texCoord * sizeF - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const float fHorizontal = texCoord.X() - floored.X();
		const float fInward = texCoord.Z() - floored.Z();
		const float l0 =
			(1.f - fHorizontal) * (At({index.X(), index.Y() + 1, index.Z()}) - At(index)) +
			fHorizontal * (At({index.X() + 1, index.Y() + 1, index.Z()}) - At({index.X() + 1, index.Y(), index.Z()}));
		const float l1 =
			(1.f - fHorizontal) *
				(At({index.X(), index.Y() + 1, index.Z() + 1}) - At({index.X(), index.Y(), index.Z() + 1})) +
			fHorizontal *
				(At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At({index.X() + 1, index.Y(), index.Z() + 1}));
		return sizeF.Y() * ((1.f - fInward) * l0 + fInward * l1);
	}

	/**
	 * @param texCoord The texture coordinates at which to sample the texture
	 * @return The derivative of the sample from this texture at the given texture coordinates using trilinear
	 * interpolation, with respect to the Z-component of the texture coordinate.
	 */
	[[nodiscard]] __host__ __device__ float DSampleDZ(VectorType texCoord) const {
		const VectorType sizeF = Size().StaticCast<double>();
		texCoord = texCoord * sizeF - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const float fHorizontal = texCoord.X() - floored.X();
		const float fVertical = texCoord.Y() - floored.Y();
		const float r0 =
			(1.f - fHorizontal) * (At({index.X(), index.Y(), index.Z() + 1}) - At(index)) +
			fHorizontal * (At({index.X() + 1, index.Y(), index.Z() + 1}) - At({index.X() + 1, index.Y(), index.Z()}));
		const float r1 =
			(1.f - fHorizontal) *
				(At({index.X(), index.Y() + 1, index.Z() + 1}) - At({index.X(), index.Y() + 1, index.Z()})) +
			fHorizontal *
				(At({index.X() + 1, index.Y() + 1, index.Z() + 1}) - At({index.X() + 1, index.Y() + 1, index.Z()}));
		return sizeF.Z() * ((1.f - fVertical) * r0 + fVertical * r1);
	}

  private:
	const float *ptr{};				///< The pointer to the data this texture provides access to
	AddressModeType addressModes{}; ///< The address mode of the texture for each dimension
};

} // namespace ExtensionTest