#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace ExtensionTest {

struct Texture2D {
	const float *ptr{};
	int64_t height{};
	int64_t width{};
	float ySpacing = 1.f;
	float xSpacing = 1.f;

	__host__ __device__ [[nodiscard]] float SizeYWorld() const { return static_cast<float>(height) * ySpacing; }
	__host__ __device__ [[nodiscard]] float SizeXWorld() const { return static_cast<float>(width) * xSpacing; }
	__host__ __device__ [[nodiscard]] float WorldToImageY(float yWorld) const { return .5f - yWorld / SizeYWorld(); }
	__host__ __device__ [[nodiscard]] float WorldToImageX(float xWorld) const { return .5f + xWorld / SizeXWorld(); }

	__host__ __device__ [[nodiscard]] bool In(int row, int col) const {
		return row >= 0 && row < height && col >= 0 && col < width;
	}

	__host__ __device__ [[nodiscard]] float At(int row, int col) const {
		return In(row, col) ? ptr[row * width + col] : 0.0f;
	}

	__host__ __device__ [[nodiscard]] bool InWorld(float y, float x) const {
		return y >= -.5f * SizeYWorld() && y < .5f * SizeYWorld() && x >= -.5f * SizeXWorld() && x < .5f * SizeXWorld();
	}

	__host__ __device__ [[nodiscard]] float SampleBilinear(float y, float x) const {
		const float yUnnormalised = y * static_cast<float>(height - 1);
		const float xUnnormalised = x * static_cast<float>(width - 1);
		const int row = static_cast<int>(floorf(yUnnormalised));
		const int col = static_cast<int>(floorf(xUnnormalised));
		const float fVertical = yUnnormalised - static_cast<float>(row);
		const float fHorizontal = xUnnormalised - static_cast<float>(col);
		const float r0 = (1.f - fHorizontal) * At(row, col) + fHorizontal * At(row, col + 1);
		const float r1 = (1.f - fHorizontal) * At(row + 1, col) + fHorizontal * At(row + 1, col + 1);
		return (1.f - fVertical) * r0 + fVertical * r1;
	}

	__host__ __device__ [[nodiscard]] float IntegrateRay(float phi, float r, float spacing = 1.f) const {
		const float yCentreWorld = r * sinf(phi);
		const float xCentreWorld = r * cosf(phi);
		const float yDeltaWorld = spacing * cosf(phi);
		const float xDeltaWorld = -spacing * sinf(phi);
		float ret = 0.f;
		int i = 0;
		while (true) {
			const float y = yCentreWorld + static_cast<float>(i) * yDeltaWorld;
			const float x = xCentreWorld + static_cast<float>(i) * xDeltaWorld;
			if (!InWorld(y, x)) break;
			ret += SampleBilinear(WorldToImageY(y), WorldToImageX(x));
			++i;
		}
		i = 1;
		while (true) {
			const float y = yCentreWorld - static_cast<float>(i) * yDeltaWorld;
			const float x = xCentreWorld - static_cast<float>(i) * xDeltaWorld;
			if (!InWorld(y, x)) break;
			ret += SampleBilinear(WorldToImageY(y), WorldToImageX(x));
			++i;
		}
		return ret;
	}
};

} // namespace ExtensionTest