#pragma once

#include "Common.h"

namespace reg23 {

class CUDATexture2D {
public:
	CUDATexture2D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY);

	CUDATexture2D(const at::Tensor &tensor, Vec<TextureAddressMode, 2> addressModes);

	[[nodiscard]] int64_t Handle() const;

#ifdef __CUDACC__

	[[nodiscard]] Vec<int64_t, 2> Size() const {
		return Vec<int64_t, 2>::FromIntArrayRef(backingTensor.sizes()).Flipped();
	}

	// no copy
	CUDATexture2D(const CUDATexture2D &) = delete;

	void operator=(const CUDATexture2D &) = delete;

	// yes move
	CUDATexture2D(CUDATexture2D &&other) noexcept
		: backingTensor(std::move(other.backingTensor)), arrayHandle(other.arrayHandle),
		  textureHandle(other.textureHandle) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	CUDATexture2D &operator=(CUDATexture2D &&other) noexcept {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
		backingTensor = std::move(other.backingTensor);
		arrayHandle = other.arrayHandle;
		textureHandle = other.textureHandle;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
		return *this;
	}

	~CUDATexture2D() {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
	}

private:
	at::Tensor backingTensor;
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;

#endif
};

class CUDATexture3D {
public:
	CUDATexture3D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY,
	              const std::string &addressModeZ);

	CUDATexture3D(const at::Tensor &tensor, Vec<TextureAddressMode, 3> addressModes);

	[[nodiscard]] int64_t Handle() const;

#ifdef __CUDACC__

	[[nodiscard]] Vec<int64_t, 3> Size() const {
		return Vec<int64_t, 3>::FromIntArrayRef(backingTensor.sizes()).Flipped();
	}

	// no copy
	CUDATexture3D(const CUDATexture3D &) = delete;

	void operator=(const CUDATexture3D &) = delete;

	// yes move
	CUDATexture3D(CUDATexture3D &&other) noexcept
		: backingTensor(std::move(other.backingTensor)), arrayHandle(other.arrayHandle),
		  textureHandle(other.textureHandle) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	CUDATexture3D &operator=(CUDATexture3D &&other) noexcept {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
		backingTensor = std::move(other.backingTensor);
		arrayHandle = other.arrayHandle;
		textureHandle = other.textureHandle;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
		return *this;
	}

	~CUDATexture3D() {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
	}

private:
	at::Tensor backingTensor{};
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;

#endif
};


} // namespace reg23