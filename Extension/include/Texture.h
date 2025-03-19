#pragma once

#include "Common.h"
#include "Vec.h"

namespace ExtensionTest {

enum class TextureAddressMode {
	ZERO,
	WRAP
};

#ifdef __CUDACC__
inline cudaTextureAddressMode TextureAddressModeToCuda(TextureAddressMode tam) {
	switch (tam) {
	case TextureAddressMode::ZERO:
		return cudaAddressModeBorder;
	case TextureAddressMode::WRAP:
		return cudaAddressModeWrap;
	default:
		return cudaAddressModeBorder;
	}
}
#endif

template <std::size_t dimensionality, typename intType=int, typename floatType=float> class Texture {
public:
	using SizeType = Vec<intType, dimensionality>;
	using VectorType = Vec<floatType, dimensionality>;
	using AddressModeType = Vec<TextureAddressMode, dimensionality>;

	[[nodiscard]] __host__ __device__ const SizeType &Size() const { return size; }
	[[nodiscard]] __host__ __device__ const VectorType &Spacing() const { return spacing; }
	[[nodiscard]] __host__ __device__ const VectorType &CentrePosition() const { return centrePosition; }

	[[nodiscard]] __host__ __device__ VectorType SizeWorld() const {
		return size.template StaticCast<floatType>() * spacing;
	}

	[[nodiscard]] __host__ __device__ bool In(const SizeType &index) const {
		return (index >= SizeType{}).BooleanAll() && (index < size).BooleanAll();
	}

	[[nodiscard]] __host__ __device__ Linear<VectorType> MappingWorldToTexCoord() const {
		const VectorType sizeWorldInverse = floatType{1.} / SizeWorld();
		return {VectorType::Full(floatType{.5}) - centrePosition * sizeWorldInverse, sizeWorldInverse};
	}

protected:
	Texture() = default;

	Texture(SizeType _size, VectorType _spacing, VectorType _centrePosition = {}) : size(_size), spacing(_spacing),
		centrePosition(_centrePosition) {
	}

	// yes copy
	Texture(const Texture &) = default;

	Texture &operator=(const Texture &) = default;

	// yes move
	Texture(Texture &&) noexcept = default;

	Texture &operator=(Texture &&) noexcept = default;

private:
	SizeType size{};
	VectorType spacing{};
	VectorType centrePosition{};
};

} // namespace ExtensionTest