#pragma once

#include "Radon3D.h"

namespace ExtensionTest {

class Texture3DCUDA : public Texture3D {
public:
	Texture3DCUDA() = default;

	Texture3DCUDA(const float *data, long _width, long _height, long _depth, double _xSpacing, double _ySpacing,
	              double _zSpacing)
		: Texture3D(_width, _height, _depth, _xSpacing, _ySpacing, _zSpacing) {

		const cudaExtent extent = {.width = static_cast<size_t>(_width), .height = static_cast<size_t>(_height),
		                           .depth = static_cast<size_t>(_depth)};

		// Copy the given data into a CUDA array
		const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		// cudaMallocArray(&arrayHandle, &channelDesc, _width, _height);
		cudaMalloc3DArray(&arrayHandle, &channelDesc, extent);
		// cudaMemcpy2DToArray(arrayHandle, 0, 0, data, _width * sizeof(float), _width * sizeof(float), _height,
		//                     cudaMemcpyHostToDevice);

		const cudaMemcpy3DParms params = {
			.srcPtr = make_cudaPitchedPtr((void *)data, _width * sizeof(float), _width, _height),
			.dstArray = arrayHandle, .extent = extent, .kind = cudaMemcpyHostToDevice};
		cudaMemcpy3D(&params);

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
	Texture3DCUDA(const Texture3DCUDA &) = delete;

	void operator=(const Texture3DCUDA &) = delete;

	// yes move
	Texture3DCUDA(Texture3DCUDA &&other) noexcept : Texture3D(other), arrayHandle(other.arrayHandle),
	                                                textureHandle(other.textureHandle) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	Texture3DCUDA &operator=(Texture3DCUDA &&other) noexcept {
		this->~Texture3DCUDA();
		arrayHandle = other.arrayHandle;
		textureHandle = other.textureHandle;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
		Texture3D::operator=(std::move(other));
		return *this;
	}

	~Texture3DCUDA() {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
	}

	[[nodiscard]] cudaTextureObject_t GetHandle() const { return textureHandle; }

	__device__ [[nodiscard]] float Sample(const float x, const float y, const float z) const {
		return tex3D<float>(textureHandle, x, y, z);
	}

	__device__ [[nodiscard]] static float SampleXDerivative(long width, cudaTextureObject_t textureHandle, float x,
	                                                        const float y, const float z) {
		const float widthF = static_cast<float>(width);
		x = floorf(-.5f + x * widthF);
		const float x0 = (x + .5f) / widthF;
		const float x1 = (x + 1.5f) / widthF;
		return widthF * (tex3D<float>(textureHandle, x1, y, z) - tex3D<float>(textureHandle, x0, y, z));
	}

	__device__ [[nodiscard]] static float SampleYDerivative(long height, cudaTextureObject_t textureHandle,
	                                                        const float x, float y, const float z) {
		const float heightF = static_cast<float>(height);
		y = floorf(-.5f + y * heightF);
		const float y0 = (y + .5f) / heightF;
		const float y1 = (y + 1.5f) / heightF;
		return heightF * (tex3D<float>(textureHandle, x, y1, z) - tex3D<float>(textureHandle, x, y0, z));
	}

	__device__ [[nodiscard]] static float SampleZDerivative(long depth, cudaTextureObject_t textureHandle,
	                                                        const float x, const float y, float z) {
		const float depthF = static_cast<float>(depth);
		z = floorf(-.5f + z * depthF);
		const float z0 = (z + .5f) / depthF;
		const float z1 = (z + 1.5f) / depthF;
		return depthF * (tex3D<float>(textureHandle, x, y, z1) - tex3D<float>(textureHandle, x, y, z0));
	}

	__device__ [[nodiscard]] float SampleXDerivative(float x, const float y, const float z) const {
		return SampleXDerivative(Width(), textureHandle, x, y, z);
	}

	__device__ [[nodiscard]] float SampleYDerivative(float x, const float y, const float z) const {
		return SampleYDerivative(Height(), textureHandle, x, y, z);
	}

	__device__ [[nodiscard]] float SampleZDerivative(float x, const float y, const float z) const {
		return SampleZDerivative(Depth(), textureHandle, x, y, z);
	}

private:
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;
};

} // namespace ExtensionTest