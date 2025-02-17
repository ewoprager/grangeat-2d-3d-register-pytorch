#pragma once

#include "Common.h"
#include "Vec.h"

namespace ExtensionTest {

template <std::size_t dimensionality, typename intType=int, typename floatType=float> class Texture {
public:
	using SizeType = Vec<intType, dimensionality>;
	using VectorType = Vec<floatType, dimensionality>;

	__host__ __device__ [[nodiscard]] const SizeType &Size() const { return size; }
	__host__ __device__ [[nodiscard]] const VectorType &Spacing() const { return spacing; }
	__host__ __device__ [[nodiscard]] const VectorType &CentrePosition() const { return centrePosition; }
	__host__ __device__ [[nodiscard]] VectorType SizeWorld() const {
		return size.template StaticCast<floatType>() * spacing;
	}

	__host__ __device__ [[nodiscard]] bool In(const SizeType &index) const {
		return (index >= SizeType{}).BooleanAll() && (index < size).BooleanAll();
	}

	__host__ __device__ [[nodiscard]] Linear<VectorType> MappingWorldToTexCoord() const {
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