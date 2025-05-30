#pragma once

#include "Texture.h"

namespace reg23 {

/**
 * @ingroup textures
 * @brief A 2D texture stored for access by an NVIDIA GPU
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Move constructable but not copy constructable.
 */
class Texture2DCUDA : public Texture<2, int64_t, double> {
  public:
	using Base = Texture<2, int64_t, double>;

	Texture2DCUDA() = default;

	Texture2DCUDA(const float *data, SizeType _size, VectorType _spacing, VectorType _centrePosition = {},
				  const AddressModeType &addressModes = AddressModeType::Full(TextureAddressMode::ZERO))
		: Base(std::move(_size), std::move(_spacing), std::move(_centrePosition)) {

		// Copy the given data into a CUDA array
		const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaMallocArray(&arrayHandle, &channelDesc, _size.X(), _size.Y());
		cudaMemcpy2DToArray(arrayHandle, 0, 0, data, _size.X() * sizeof(float), _size.X() * sizeof(float), _size.Y(),
							cudaMemcpyHostToDevice);

		// Create the texture object from the CUDA array
		const cudaResourceDesc resourceDescriptor = {.resType = cudaResourceTypeArray,
													 .res = {.array = {.array = arrayHandle}}};
		cudaTextureDesc textureDescriptor = {.filterMode = cudaFilterModeLinear,
											 .readMode = cudaReadModeElementType,
											 .borderColor = {0.f, 0.f, 0.f, 0.f},
											 .normalizedCoords = true};
		for (int i = 0; i < 2; ++i) {
			textureDescriptor.addressMode[i] = TextureAddressModeToCuda(addressModes[i]);
		}
		cudaCreateTextureObject(&textureHandle, &resourceDescriptor, &textureDescriptor, nullptr);
	}

	// no copy
	Texture2DCUDA(const Texture2DCUDA &) = delete;

	void operator=(const Texture2DCUDA &) = delete;

	// yes move
	Texture2DCUDA(Texture2DCUDA &&other) noexcept
		: Base(other), arrayHandle(other.arrayHandle), textureHandle(other.textureHandle) {
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
	}

	Texture2DCUDA &operator=(Texture2DCUDA &&other) noexcept {
		this->~Texture2DCUDA();
		arrayHandle = other.arrayHandle;
		textureHandle = other.textureHandle;
		other.arrayHandle = nullptr;
		other.textureHandle = 0;
		Base::operator=(std::move(other));
		return *this;
	}

	~Texture2DCUDA() {
		if (textureHandle) cudaDestroyTextureObject(textureHandle);
		if (arrayHandle) cudaFreeArray(arrayHandle);
	}

	static Texture2DCUDA FromTensor(const at::Tensor &image, const at::Tensor &spacing) {
		return {image.contiguous().data_ptr<float>(), Vec<int64_t, 2>::FromIntArrayRef(image.sizes()).Flipped(),
				Vec<double, 2>::FromTensor(spacing)};
	}

	[[nodiscard]] cudaTextureObject_t GetHandle() const { return textureHandle; }

	[[nodiscard]] __device__ float Sample(const VectorType &texCoord) const {
		return tex2D<float>(textureHandle, texCoord.X(), texCoord.Y());
	}

	[[nodiscard]] __device__ float DSampleDX(const VectorType &texCoord) const {
		const float widthF = static_cast<float>(Size().X());
		const float x = floorf(-.5f + texCoord.X() * widthF);
		const float x0 = (x + .5f) / widthF;
		const float x1 = (x + 1.5f) / widthF;
		return widthF * (tex2D<float>(textureHandle, x1, texCoord.Y()) - tex2D<float>(textureHandle, x0, texCoord.Y()));
	}

	[[nodiscard]] __device__ float DSampleDY(const VectorType &texCoord) const {
		const float heightF = static_cast<float>(Size().Y());
		const float y = floorf(-.5f + texCoord.Y() * heightF);
		const float y0 = (y + .5f) / heightF;
		const float y1 = (y + 1.5f) / heightF;
		return heightF *
			   (tex2D<float>(textureHandle, texCoord.X(), y1) - tex2D<float>(textureHandle, texCoord.X(), y0));
	}

  private:
	cudaArray_t arrayHandle = nullptr;
	cudaTextureObject_t textureHandle = 0;
};

} // namespace reg23