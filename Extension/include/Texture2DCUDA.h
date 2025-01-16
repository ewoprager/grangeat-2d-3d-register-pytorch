#pragma once

#include "Radon2D.h"

namespace ExtensionTest {

class Texture2DCUDA : public Texture2D {
public:
	Texture2DCUDA() = default;

	Texture2DCUDA(const float *data, long _width, long _height, double _xSpacing, double _ySpacing)
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
	Texture2DCUDA(Texture2DCUDA &&other) noexcept : Texture2D(other), arrayHandle(other.arrayHandle),
	                                                textureHandle(other.textureHandle) {
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

private:
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;
};

} // namespace ExtensionTest