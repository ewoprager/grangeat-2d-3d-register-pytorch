#pragma once

#include "Texture.h"

namespace ExtensionTest {

class Texture3DCUDA : public Texture<3, int64_t, double> {
public:
	using Base = Texture<3, int64_t, double>;

	Texture3DCUDA() = default;

	Texture3DCUDA(const float *data, SizeType _size, VectorType _spacing, VectorType _centrePosition = {})
		: Base(_size, _spacing, _centrePosition) {

		const cudaExtent extent = {.width = static_cast<size_t>(_size.X()), .height = static_cast<size_t>(_size.Y()),
		                           .depth = static_cast<size_t>(_size.Z())};

		// Copy the given data into a CUDA array
		const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		// cudaMallocArray(&arrayHandle, &channelDesc, _width, _height);
		cudaMalloc3DArray(&arrayHandle, &channelDesc, extent);
		// cudaMemcpy2DToArray(arrayHandle, 0, 0, data, _width * sizeof(float), _width * sizeof(float), _height,
		//                     cudaMemcpyHostToDevice);

		const cudaMemcpy3DParms params = {
			.srcPtr = make_cudaPitchedPtr((void *)data, _size.X() * sizeof(float), _size.X(), _size.Y()),
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
	Texture3DCUDA(Texture3DCUDA &&other) noexcept : Base(other), arrayHandle(other.arrayHandle),
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
		Base::operator=(std::move(other));
		return *this;
	}

	~Texture3DCUDA() {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
	}

	[[nodiscard]] cudaTextureObject_t GetHandle() const { return textureHandle; }

	__device__ [[nodiscard]] float Sample(const VectorType &texCoord) const {
		return tex3D<float>(textureHandle, texCoord.X(), texCoord.Y(), texCoord.Z());
	}

	__device__ [[nodiscard]] static float SampleXDerivative(long width, cudaTextureObject_t textureHandle,
	                                                        const VectorType &texCoord) {
		const float widthF = static_cast<float>(width);
		const float x = floorf(-.5f + texCoord.X() * widthF);
		const float x0 = (x + .5f) / widthF;
		const float x1 = (x + 1.5f) / widthF;
		return widthF * (tex3D<float>(textureHandle, x1, texCoord.Y(), texCoord.Z()) - tex3D<float>(
			                 textureHandle, x0, texCoord.Y(), texCoord.Z()));
	}

	__device__ [[nodiscard]] static float SampleYDerivative(long height, cudaTextureObject_t textureHandle,
	                                                        const VectorType &texCoord) {
		const float heightF = static_cast<float>(height);
		const float y = floorf(-.5f + texCoord.Y() * heightF);
		const float y0 = (y + .5f) / heightF;
		const float y1 = (y + 1.5f) / heightF;
		return heightF * (tex3D<float>(textureHandle, texCoord.X(), y1, texCoord.Z()) - tex3D<float>(
			                  textureHandle, texCoord.X(), y0, texCoord.Z()));
	}

	__device__ [[nodiscard]] static float SampleZDerivative(long depth, cudaTextureObject_t textureHandle,
	                                                        const VectorType &texCoord) {
		const float depthF = static_cast<float>(depth);
		const float z = floorf(-.5f + texCoord.Z() * depthF);
		const float z0 = (z + .5f) / depthF;
		const float z1 = (z + 1.5f) / depthF;
		return depthF * (tex3D<float>(textureHandle, texCoord.X(), texCoord.Y(), z1) - tex3D<float>(
			                 textureHandle, texCoord.X(), texCoord.Y(), z0));
	}

	__device__ [[nodiscard]] float SampleXDerivative(const VectorType &texCoord) const {
		return SampleXDerivative(Size().X(), textureHandle, texCoord);
	}

	__device__ [[nodiscard]] float SampleYDerivative(const VectorType &texCoord) const {
		return SampleYDerivative(Size().Y(), textureHandle, texCoord);
	}

	__device__ [[nodiscard]] float SampleZDerivative(const VectorType &texCoord) const {
		return SampleZDerivative(Size().Z(), textureHandle, texCoord);
	}

private:
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;
};

} // namespace ExtensionTest