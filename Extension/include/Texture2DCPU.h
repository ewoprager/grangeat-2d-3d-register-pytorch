#pragma once

#include "Texture.h"

namespace ExtensionTest {

class Texture2DCPU : public Texture<2, int64_t, double> {
public:
	using Base = Texture<2, int64_t, double>;

	Texture2DCPU() = default;

	Texture2DCPU(const float *_ptr, SizeType _size, VectorType _spacing, VectorType _centrePosition = {},
	             const AddressModeType &_addressModes =
		             AddressModeType::Full(TextureAddressMode::ZERO)) : Base(std::move(_size), std::move(_spacing),
		                                                                     std::move(_centrePosition)), ptr(_ptr),
		                                                                addressModes(_addressModes) {
	}

	// yes copy
	Texture2DCPU(const Texture2DCPU &) = default;

	Texture2DCPU &operator=(const Texture2DCPU &) = default;

	// yes move
	Texture2DCPU(Texture2DCPU &&) = default;

	Texture2DCPU &operator=(Texture2DCPU &&) = default;

	static Texture2DCPU FromTensor(const at::Tensor &image, const at::Tensor &spacing) {
		return {image.contiguous().data_ptr<float>(), Vec<int64_t, 2>::FromIntArrayRef(image.sizes()).Flipped(),
		        Vec<double, 2>::FromTensor(spacing)};
	}

	[[nodiscard]] __host__ __device__ float At(const SizeType &index) const {
		if ((addressModes.X() == TextureAddressMode::ZERO && (index.X() < 0 || index.X() >= Size().X())) || (
			    addressModes.Y() == TextureAddressMode::ZERO && (index.Y() < 0 || index.Y() >= Size().Y()))) {
			return 0.f;
		}
		// Uses wrapping for indices outside the texture.
		return ptr[Modulo(index.Y(), Size().Y()) * Size().X() + Modulo(index.X(), Size().X())];
	}

	[[nodiscard]] __host__ __device__ float Sample(VectorType texCoord) const {
		texCoord = texCoord * Size().StaticCast<double>() - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const VectorType fractions = texCoord - floored;
		const float r0 = (1.f - fractions.X()) * At(index) + fractions.X() * At({index.X() + 1, index.Y()});
		const float r1 = (1.f - fractions.X()) * At({index.X(), index.Y() + 1}) + fractions.X() * At(
			                 {index.X() + 1, index.Y() + 1});
		return (1.f - fractions.Y()) * r0 + fractions.Y() * r1;
	}

	[[nodiscard]] __host__ __device__ float DSampleDX(VectorType texCoord) const {
		const VectorType sizeF = Size().StaticCast<double>();
		texCoord = texCoord * sizeF - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const float fVertical = texCoord.Y() - floored.Y();
		return sizeF.X() * ((1.f - fVertical) * (At({index.X() + 1, index.Y()}) - At(index)) + fVertical * (
			                    At({index.X() + 1, index.Y() + 1}) - At({index.X(), index.Y() + 1})));
	}

	[[nodiscard]] __host__ __device__ float DSampleDY(VectorType texCoord) const {
		const VectorType sizeF = Size().StaticCast<double>();
		texCoord = texCoord * sizeF - .5;
		const VectorType floored = texCoord.Apply<double>(&floor);
		const SizeType index = floored.StaticCast<int64_t>();
		const float fHorizontal = texCoord.X() - floored.X();
		return sizeF.Y() * ((1.f - fHorizontal) * (At({index.X(), index.Y() + 1}) - At(index)) + fHorizontal * (
			                    At({index.X() + 1, index.Y() + 1}) - At({index.X() + 1, index.Y()})));
	}

private:
	const float *ptr{};
	AddressModeType addressModes{};
};

} // namespace ExtensionTest