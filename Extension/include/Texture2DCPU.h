#pragma once

#include "Texture2D.h"

namespace ExtensionTest {

class Texture2DCPU : public Texture2D {
public:
	Texture2DCPU() = default;

	Texture2DCPU(const float *_ptr, SizeType _size, VectorType _spacing,
	             VectorType _centrePosition = {}) : Texture2D(_size, _spacing, _centrePosition), ptr(_ptr) {
	}

	// yes copy
	Texture2DCPU(const Texture2DCPU &) = default;

	Texture2DCPU &operator=(const Texture2DCPU &) = default;

	// yes move
	Texture2DCPU(Texture2DCPU &&) = default;

	Texture2DCPU &operator=(Texture2DCPU &&) = default;

	__host__ __device__ [[nodiscard]] float At(const SizeType &index) const {
		return In(index) ? ptr[index.Y() * Size().X() + index.X()] : 0.0f;
	}

	__host__ __device__ [[nodiscard]] float Sample(VectorType texCoord) const {
		texCoord = texCoord * VecCast<float>(Size()) - .5f;
		const VectorType floored = VecApply(&floorf, texCoord);
		const SizeType index = VecCast<int>(floored);
		const VectorType fractions = texCoord - floored;
		const float r0 = (1.f - fractions.X()) * At(index) + fractions.X() * At({index.X() + 1, index.Y()});
		const float r1 = (1.f - fractions.X()) * At({index.X(), index.Y() + 1}) + fractions.X() * At(
			                 {index.X() + 1, index.Y() + 1});
		return (1.f - fractions.Y()) * r0 + fractions.Y() * r1;
	}

	__host__ __device__ [[nodiscard]] float SampleXDerivative(VectorType texCoord) const {
		const VectorType sizeF = VecCast<float>(Size());
		texCoord = texCoord * sizeF - .5f;
		const VectorType floored = VecApply(&floorf, texCoord);
		const SizeType index = VecCast<int>(floored);
		const float fVertical = texCoord.Y() - floored.Y();
		return sizeF.X() * ((1.f - fVertical) * (At({index.X() + 1, index.Y()}) - At(index)) + fVertical * (
			                    At({index.X() + 1, index.Y() + 1}) - At({index.X(), index.Y() + 1})));
	}

	__host__ __device__ [[nodiscard]] float SampleYDerivative(VectorType texCoord) const {
		const VectorType sizeF = VecCast<float>(Size());
		texCoord = texCoord * sizeF - .5f;
		const VectorType floored = VecApply(&floorf, texCoord);
		const SizeType index = VecCast<int>(floored);
		const float fHorizontal = texCoord.X() - floored.X();
		return sizeF.Y() * ((1.f - fHorizontal) * (At({index.X(), index.Y() + 1}) - At(index)) + fHorizontal * (
			                    At({index.X() + 1, index.Y() + 1}) - At({index.X() + 1, index.Y()})));
	}

private:
	const float *ptr{};
};

} // namespace ExtensionTest