#pragma once

#include "Radon3D.h"

namespace ExtensionTest {

class Texture3DCPU : public Texture3D {
public:
	Texture3DCPU() = default;

	Texture3DCPU(const float *_ptr, long _width, long _height, long _depth, double _xSpacing, double _ySpacing,
	             double _zSpacing) : Texture3D(_width, _height, _depth, _xSpacing, _ySpacing, _zSpacing), ptr(_ptr) {
	}

	// yes copy
	Texture3DCPU(const Texture3DCPU &) = default;

	Texture3DCPU &operator=(const Texture3DCPU &) = default;

	// yes move
	Texture3DCPU(Texture3DCPU &&) = default;

	Texture3DCPU &operator=(Texture3DCPU &&) = default;

	__host__ __device__ [[nodiscard]] float At(long layer, long row, long col) const {
		return In(layer, row, col) ? ptr[layer * Width() * Height() + row * Width() + col] : 0.0f;
	}

	__host__ __device__ [[nodiscard]] float Sample(float x, float y, float z) const {
		x = -.5f + x * static_cast<float>(Width());
		y = -.5f + y * static_cast<float>(Height());
		z = -.5f + z * static_cast<float>(Depth());
		const float xFloored = floorf(x);
		const float yFloored = floorf(y);
		const float zFloored = floorf(z);
		const long col = static_cast<long>(xFloored);
		const long row = static_cast<long>(yFloored);
		const long layer = static_cast<long>(zFloored);
		const float fHorizontal = x - xFloored;
		const float fVertical = y - yFloored;
		const float fInward = z - zFloored;
		const float l0r0 = (1.f - fHorizontal) * At(layer, row, col) + fHorizontal * At(layer, row, col + 1);
		const float l0r1 = (1.f - fHorizontal) * At(layer, row + 1, col) + fHorizontal * At(layer, row + 1, col + 1);
		const float l1r0 = (1.f - fHorizontal) * At(layer + 1, row, col) + fHorizontal * At(layer + 1, row, col + 1);
		const float l1r1 = (1.f - fHorizontal) * At(layer + 1, row + 1, col) + fHorizontal * At(
			                   layer + 1, row + 1, col + 1);
		const float l0 = (1.f - fVertical) * l0r0 + fVertical * l0r1;
		const float l1 = (1.f - fVertical) * l1r0 + fVertical * l1r1;
		return (1.f - fInward) * l0 + fInward * l1;
	}

	__host__ __device__ [[nodiscard]] float SampleXDerivative(float x, float y, float z) const {
		const float widthF = static_cast<float>(Width());
		x = -.5f + x * widthF;
		y = -.5f + y * static_cast<float>(Height());
		z = -.5f + z * static_cast<float>(Depth());
		const float xFloored = floorf(x);
		const float yFloored = floorf(y);
		const float zFloored = floorf(z);
		const long col = static_cast<long>(xFloored);
		const long row = static_cast<long>(yFloored);
		const long layer = static_cast<long>(zFloored);
		const float fVertical = y - yFloored;
		const float fInward = z - zFloored;
		const float l0 = (1.f - fVertical) * (At(layer, row, col + 1) - At(layer, row, col)) + fVertical * (
			                 At(layer, row + 1, col + 1) - At(layer, row + 1, col));
		const float l1 = (1.f - fVertical) * (At(layer + 1, row, col + 1) - At(layer + 1, row, col)) + fVertical * (
			                 At(layer + 1, row + 1, col + 1) - At(layer + 1, row + 1, col));
		return widthF * ((1.f - fInward) * l0 + fInward * l1);
	}

	__host__ __device__ [[nodiscard]] float SampleYDerivative(float x, float y, float z) const {
		const float heightF = static_cast<float>(Height());
		x = -.5f + x * static_cast<float>(Width());
		y = -.5f + y * heightF;
		z = -.5f + z * static_cast<float>(Depth());
		const float xFloored = floorf(x);
		const float yFloored = floorf(y);
		const float zFloored = floorf(z);
		const long col = static_cast<long>(xFloored);
		const long row = static_cast<long>(yFloored);
		const long layer = static_cast<long>(zFloored);
		const float fHorizontal = x - xFloored;
		const float fInward = z - zFloored;
		const float l0 = (1.f - fHorizontal) * (At(layer, row + 1, col) - At(layer, row, col)) + fHorizontal * (
			                 At(layer, row + 1, col + 1) - At(layer, row, col + 1));
		const float l1 = (1.f - fHorizontal) * (At(layer + 1, row + 1, col) - At(layer + 1, row, col)) + fHorizontal * (
			                 At(layer + 1, row + 1, col + 1) - At(layer + 1, row, col + 1));
		return heightF * ((1.f - fInward) * l0 + fInward * l1);
	}

	__host__ __device__ [[nodiscard]] float SampleZDerivative(float x, float y, float z) const {
		const float depthF = static_cast<float>(Depth());
		x = -.5f + x * static_cast<float>(Width());
		y = -.5f + y * static_cast<float>(Height());
		z = -.5f + z * depthF;
		const float xFloored = floorf(x);
		const float yFloored = floorf(y);
		const float zFloored = floorf(z);
		const long col = static_cast<long>(xFloored);
		const long row = static_cast<long>(yFloored);
		const long layer = static_cast<long>(zFloored);
		const float fHorizontal = x - xFloored;
		const float fVertical = y - yFloored;
		const float r0 = (1.f - fHorizontal) * (At(layer + 1, row, col) - At(layer, row, col)) + fHorizontal * (
			                 At(layer + 1, row, col + 1) - At(layer, row, col + 1));
		const float r1 = (1.f - fHorizontal) * (At(layer + 1, row + 1, col) - At(layer, row + 1, col)) + fHorizontal * (
			                 At(layer + 1, row + 1, col + 1) - At(layer, row + 1, col + 1));
		return depthF * ((1.f - fVertical) * r0 + fVertical * r1);
	}

private:
	const float *ptr{};
};

} // namespace ExtensionTest