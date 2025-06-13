#pragma once

#include "Common.h"

namespace reg23 {

class CUDATexture2D {
public:
	explicit CUDATexture2D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY);

	[[nodiscard]] int64_t Handle() const;

#ifdef __CUDACC__

	[[nodiscard]] const Vec<int64_t, 2> &Size() const { return size; }

	// no copy
	CUDATexture2D(const CUDATexture2D &) = delete;

	void operator=(const CUDATexture2D &) = delete;

	// yes move
	CUDATexture2D(CUDATexture2D &&other) noexcept
		: arrayHandle(other.arrayHandle), textureHandle(other.textureHandle) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	CUDATexture2D &operator=(CUDATexture2D &&other) noexcept {
		this->~CUDATexture2D();
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
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;
	Vec<int64_t, 2> size;

#endif
};

class CUDATexture3D {
public:
	explicit CUDATexture3D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY,
	                       const std::string &addressModeZ);

	[[nodiscard]] int64_t Handle() const;

#ifdef __CUDACC__

	[[nodiscard]] const Vec<int64_t, 3> &Size() const { return size; }

	// no copy
	CUDATexture3D(const CUDATexture3D &) = delete;

	void operator=(const CUDATexture3D &) = delete;

	// yes move
	CUDATexture3D(CUDATexture3D &&other) noexcept
		: arrayHandle(other.arrayHandle), textureHandle(other.textureHandle) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	CUDATexture3D &operator=(CUDATexture3D &&other) noexcept {
		this->~CUDATexture3D();
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
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;
	Vec<int64_t, 3> size;

#endif
};


} // namespace reg23