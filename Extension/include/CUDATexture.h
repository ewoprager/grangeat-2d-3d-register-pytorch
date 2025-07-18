#pragma once

#include "Common.h"

namespace reg23 {

class CUDATexture2D {
public:
	CUDATexture2D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY);

	CUDATexture2D(const at::Tensor &tensor, Vec<TextureAddressMode, 2> addressModes);

	[[nodiscard]] int64_t Handle() const;

	[[nodiscard]] at::Tensor SizeTensor() const;

#ifdef __CUDACC__

	[[nodiscard]] Vec<int64_t, 2> Size() const {
		return size;
	}

	// no copy
	CUDATexture2D(const CUDATexture2D &) = delete;

	void operator=(const CUDATexture2D &) = delete;

	// yes move
	CUDATexture2D(CUDATexture2D &&other) noexcept
		: size(std::move(other.size)), arrayHandle(other.arrayHandle), textureHandle(other.textureHandle) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	// ToDo: Uncomment noexcept, and replace throws with `std::terminate()`
	CUDATexture2D &operator=(CUDATexture2D &&other) /*noexcept*/ {
		cudaError_t err;
		if (textureHandle) {
			err = cudaDestroyTextureObject(textureHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaDestroyTextureObject failed: " << cudaGetErrorString(err) << std::endl;
				throw std::runtime_error("cudaDestroyTextureObject failed");
			}
		}
		if (arrayHandle) {
			err = cudaFreeArray(arrayHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaFreeArray failed: " << cudaGetErrorString(err) << std::endl;
				throw std::runtime_error("cudaFreeArray failed");
			}
		}
		size = std::move(other.size);
		arrayHandle = other.arrayHandle;
		textureHandle = other.textureHandle;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
		return *this;
	}

	~CUDATexture2D() {
		cudaError_t err;
		if (textureHandle) {
			err = cudaDestroyTextureObject(textureHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaDestroyTextureObject failed: " << cudaGetErrorString(err) << std::endl;
				std::terminate();
			}
		}
		if (arrayHandle) {
			err = cudaFreeArray(arrayHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaFreeArray failed: " << cudaGetErrorString(err) << std::endl;
				std::terminate();
			}
		}
	}

private:
	Vec<int64_t, 2> size;
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

	[[nodiscard]] at::Tensor SizeTensor() const;

#ifdef __CUDACC__

	[[nodiscard]] Vec<int64_t, 3> Size() const {
		return size;
	}

	// no copy
	CUDATexture3D(const CUDATexture3D &) = delete;

	void operator=(const CUDATexture3D &) = delete;

	// yes move
	CUDATexture3D(CUDATexture3D &&other) noexcept
		: size(std::move(other.size)), arrayHandle(other.arrayHandle), textureHandle(other.textureHandle) {
		std::cout << "CUDATexture3D move constructing; texture handle: " << textureHandle << ", arrayHandle: " <<
			arrayHandle << std::endl;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	// ToDo: Uncomment noexcept, and replace throws with `std::terminate()`
	CUDATexture3D &operator=(CUDATexture3D &&other) /*noexcept*/ {
		std::cout << "CUDATexture3D move assigned; texture handle: " << other.textureHandle << ", arrayHandle: " <<
			other.arrayHandle << std::endl;
		cudaError_t err;
		if (textureHandle) {
			err = cudaDestroyTextureObject(textureHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaDestroyTextureObject failed: " << cudaGetErrorString(err) << std::endl;
				throw std::runtime_error("cudaDestroyTextureObject failed");
			}
		}
		if (arrayHandle) {
			err = cudaFreeArray(arrayHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaFreeArray failed: " << cudaGetErrorString(err) << std::endl;
				throw std::runtime_error("cudaFreeArray failed");
			}
		}
		size = std::move(other.size);
		arrayHandle = other.arrayHandle;
		textureHandle = other.textureHandle;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
		return *this;
	}

	~CUDATexture3D() {
		std::cout << "CUDATexture3D destructing; texture handle: " << textureHandle << ", arrayHandle: " << arrayHandle
			<< std::endl;
		cudaError_t err;
		if (textureHandle) {
			err = cudaDestroyTextureObject(textureHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaDestroyTextureObject failed: " << cudaGetErrorString(err) << std::endl;
				std::terminate();
			}
		}
		if (arrayHandle) {
			err = cudaFreeArray(arrayHandle);
			if (err != cudaSuccess) {
				std::cerr << "cudaFreeArray failed: " << cudaGetErrorString(err) << std::endl;
				std::terminate();
			}
		}
	}

private:
	Vec<int64_t, 3> size{};
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;

#endif
};


} // namespace reg23