#pragma once

#include "Common.h"

namespace ExtensionTest {

class Texture2D {
public:
	using IntType = int;
	using SizeType = Vec<IntType, 2>;
	using VectorType = Vec<float, 2>;

	__host__ __device__ [[nodiscard]] const SizeType &Size() const { return size; }
	__host__ __device__ [[nodiscard]] const VectorType &Spacing() const { return spacing; }
	__host__ __device__ [[nodiscard]] VectorType SizeWorld() const { return VecCast<float>(size) * spacing; }
	__host__ __device__ [[nodiscard]] bool In(const SizeType &index) const {
		return VecAll(index >= SizeType{}) && VecAll(index < size);
	}

	__host__ __device__ [[nodiscard]] Linear<VectorType> MappingWorldToNormalised() const {
		return {VectorType{{.5f, .5f}}, 1.f / SizeWorld()};
	}

protected:
	Texture2D() = default;

	Texture2D(SizeType _size, VectorType _spacing, VectorType _centrePosition = {}) : size(_size), spacing(_spacing),
		centrePosition(_centrePosition) {
	}

	// yes copy
	Texture2D(const Texture2D &) = default;

	Texture2D &operator=(const Texture2D &) = default;

	// yes move
	Texture2D(Texture2D &&) noexcept = default;

	Texture2D &operator=(Texture2D &&) noexcept = default;

private:
	SizeType size{};
	VectorType spacing{};
	VectorType centrePosition{};
};

} // namespace ExtensionTest