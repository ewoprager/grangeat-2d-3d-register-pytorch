#pragma once

#include "Texture.h"
#include "CUDATexture.h"

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

	Texture2DCUDA(int64_t _textureHandle, SizeType _size, VectorType _spacing, VectorType _centrePosition = {})
		: Base(std::move(_size), std::move(_spacing), std::move(_centrePosition)), textureHandle(_textureHandle) {
	}

	Texture2DCUDA(std::shared_ptr<CUDATexture2D> cudaTexture, VectorType _spacing, VectorType _centrePosition = {})
		: Base(cudaTexture->Size(), std::move(_spacing), std::move(_centrePosition)),
		  textureHandle(cudaTexture->Handle()), ownedTexture(std::move(cudaTexture)) {
	}

	// yes copy
	Texture2DCUDA(const Texture2DCUDA &) = default;

	Texture2DCUDA &operator=(const Texture2DCUDA &) = default;

	// yes move
	Texture2DCUDA(Texture2DCUDA &&) noexcept = default;

	Texture2DCUDA &operator=(Texture2DCUDA &&other) noexcept = default;

	/**
	 *
	 * @param image
	 * @param spacing
	 * @param centrePosition
	 * @param addressModes
	 * @return
	 */
	static Texture2DCUDA FromTensor(const at::Tensor &image, VectorType spacing,
	                                VectorType centrePosition = VectorType::Full(0),
	                                AddressModeType addressModes = AddressModeType::Full(TextureAddressMode::ZERO)) {
		return {std::make_shared<CUDATexture2D>(image, addressModes), std::move(spacing), std::move(centrePosition)};
	}

	__host__ __device__ [[nodiscard]] cudaTextureObject_t GetHandle() const { return textureHandle; }

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
		return heightF * (tex2D<float>(textureHandle, texCoord.X(), y1) - tex2D<
			                  float>(textureHandle, texCoord.X(), y0));
	}

	[[nodiscard]] __device__ VectorType DSampleDTexCoord(const VectorType &texCoord) const {
		return {DSampleDX(texCoord), DSampleDY(texCoord)};
	}

private:
	cudaTextureObject_t textureHandle = 0;
	std::shared_ptr<CUDATexture2D> ownedTexture = nullptr;
};

} // namespace reg23