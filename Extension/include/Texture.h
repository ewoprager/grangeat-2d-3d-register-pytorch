#pragma once

#include "Common.h"

namespace ExtensionTest {

template <typename SizeType, typename VectorType>
class Texture {
public:
	__host__ __device__ [[nodiscard]] const SizeType &Size() const { return size; }
	__host__ __device__ [[nodiscard]] const VectorType &Spacing() const { return spacing; }
	__host__ __device__ [[nodiscard]] const VectorType &CentrePosition() const { return centrePosition; }
	__host__ __device__ [[nodiscard]] VectorType SizeWorld() const { return VecCast<float>(size) * spacing; }
	__host__ __device__ [[nodiscard]] bool In(const SizeType &index) const {
		return VecAll(index >= SizeType{}) && VecAll(index < size);
	}

	__host__ __device__ [[nodiscard]] Linear<VectorType> MappingWorldToTexCoord() const {
		const VectorType sizeWorldInverse = 1.f / SizeWorld();
		return {VectorType::Full(.5f) - centrePosition * sizeWorldInverse, sizeWorldInverse};
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