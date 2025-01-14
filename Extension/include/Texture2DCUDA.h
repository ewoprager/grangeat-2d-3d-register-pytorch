#pragma once

#include "Common.h"

namespace ExtensionTest {

class Texture2DCUDA : public Texture2D {
public:
	Texture2DCUDA() = default;

	Texture2DCUDA(const float *data, long _height, long _width, float _ySpacing = 1.f, float _xSpacing = 1.f)
		: Texture2D(_width, _height, _xSpacing, _ySpacing) {

		// Copy the given data into a CUDA array
		const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaMallocArray(&arrayHandle, &channelDesc, _width, _height);
		cudaMemcpy2DToArray(arrayHandle, 0, 0, data, _width * sizeof(float), _width * sizeof(float), _height,
							cudaMemcpyHostToDevice);

		// Create the texture object from the CUDA array
		const cudaResourceDesc resourceDescriptor = {.resType = cudaResourceTypeArray,
													 .res = {.array = {.array = arrayHandle}}};
		constexpr cudaTextureDesc textureDescriptor = {
			.addressMode = {cudaAddressModeBorder, cudaAddressModeBorder, cudaAddressModeBorder},
			.filterMode = cudaFilterModeLinear, .readMode = cudaReadModeElementType,
			.borderColor = {0.f, 0.f, 0.f, 0.f}, .normalizedCoords = true};
		cudaCreateTextureObject(&textureHandle, &resourceDescriptor, &textureDescriptor, nullptr);
	}

	// no copy
	Texture2DCUDA(const Texture2DCUDA &) = delete;

	void operator=(const Texture2DCUDA &) = delete;

	// yes move
	Texture2DCUDA(Texture2DCUDA &&other) noexcept : arrayHandle(other.arrayHandle), textureHandle(other.textureHandle),
													Texture2D(other) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}
	Texture2DCUDA &operator=(Texture2DCUDA &&other) noexcept {
		this->~Texture2DCUDA();
		arrayHandle = other.arrayHandle;
		textureHandle = other.textureHandle;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
		Texture2D::operator=(std::move(other));
		return *this;
	}

	~Texture2DCUDA() {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
	}

	__device__ [[nodiscard]] float Sample(const float x, const float y) const {
		return tex2D<float>(textureHandle, x, y);
	}

	__device__ [[nodiscard]] float Radon(const float phi, const float r, const float rayLength,
										 const long samplesPerLine) const {
		const float s = sinf(phi);
		const float c = cosf(phi);
		const Linear mappingOffsetToWorldX{r * c, -s};
		const Linear mappingOffsetToWorldY{r * s, c};
		const Linear mappingIToOffset{-.5f * rayLength, rayLength / static_cast<float>(samplesPerLine - 1)};
		const Linear mappingIToX = MappingXWorldToNormalised()(mappingOffsetToWorldX(mappingIToOffset));
		const Linear mappingIToY = MappingYWorldToNormalised()(mappingOffsetToWorldY(mappingIToOffset));
		float ret = 0.f;
		for (int i = 0; i < samplesPerLine; ++i) {
			const float iF = static_cast<float>(i);
			ret += Sample(mappingIToX(iF), mappingIToY(iF));
		}
		return ret;
	}

private:
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;
};

} // namespace ExtensionTest