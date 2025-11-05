#pragma once

#include "Texture.h"
#include "CUDATexture.h"

namespace reg23 {

/**
 * @ingroup textures
 * @brief A 3D texture stored for access by an NVIDIA GPU
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Move constructable but not copy constructable.
 */
class Texture3DCUDA : public Texture<3, int64_t, double> {
public:
	using Base = Texture<3, int64_t, double>;

	Texture3DCUDA() = default;

	Texture3DCUDA(int64_t _textureHandle, SizeType _size, VectorType _spacing, VectorType _centrePosition = {})
		: Base(std::move(_size), std::move(_spacing), std::move(_centrePosition)), textureHandle(_textureHandle),
		  ownedTexture(nullptr) {
	}

	Texture3DCUDA(std::shared_ptr<CUDATexture3D> cudaTexture, VectorType _spacing, VectorType _centrePosition = {})
		: Base(cudaTexture->Size(), std::move(_spacing), std::move(_centrePosition)),
		  textureHandle(cudaTexture->Handle()), ownedTexture(std::move(cudaTexture)) {
	}

	// yes copy
	Texture3DCUDA(const Texture3DCUDA &) = default;

	Texture3DCUDA &operator=(const Texture3DCUDA &) = default;

	// yes move
	Texture3DCUDA(Texture3DCUDA &&) noexcept = default;

	Texture3DCUDA &operator=(Texture3DCUDA &&) noexcept = default;

	/**
	 *
	 * @param volume
	 * @param spacing
	 * @param centrePosition
	 * @param addressModes
	 * @return
	 */
	static Texture3DCUDA FromTensor(const at::Tensor &volume, VectorType spacing,
	                                VectorType centrePosition = VectorType::Full(0),
	                                AddressModeType addressModes = AddressModeType::Full(TextureAddressMode::ZERO)) {
		return {std::make_shared<CUDATexture3D>(volume, addressModes), std::move(spacing), std::move(centrePosition)};
	}

	[[nodiscard]] cudaTextureObject_t GetHandle() const { return textureHandle; }

	[[nodiscard]] __device__ float Sample(const VectorType &texCoord) const {
		return tex3D<float>(textureHandle, texCoord.X(), texCoord.Y(), texCoord.Z());
	}

	[[nodiscard]] __device__ static float DSampleDX(int64_t width, cudaTextureObject_t textureHandle,
	                                                const VectorType &texCoord) {
		const float widthF = static_cast<float>(width);
		const float x = floorf(-.5f + texCoord.X() * widthF);
		const float x0 = (x + .5f) / widthF;
		const float x1 = (x + 1.5f) / widthF;
		return widthF * (tex3D<float>(textureHandle, x1, texCoord.Y(), texCoord.Z()) - tex3D<float>(
			                 textureHandle, x0, texCoord.Y(), texCoord.Z()));
	}

	[[nodiscard]] __device__ static float DSampleDY(int64_t height, cudaTextureObject_t textureHandle,
	                                                const VectorType &texCoord) {
		const float heightF = static_cast<float>(height);
		const float y = floorf(-.5f + texCoord.Y() * heightF);
		const float y0 = (y + .5f) / heightF;
		const float y1 = (y + 1.5f) / heightF;
		return heightF * (tex3D<float>(textureHandle, texCoord.X(), y1, texCoord.Z()) - tex3D<float>(
			                  textureHandle, texCoord.X(), y0, texCoord.Z()));
	}

	[[nodiscard]] __device__ static float DSampleDZ(int64_t depth, cudaTextureObject_t textureHandle,
	                                                const VectorType &texCoord) {
		const float depthF = static_cast<float>(depth);
		const float z = floorf(-.5f + texCoord.Z() * depthF);
		const float z0 = (z + .5f) / depthF;
		const float z1 = (z + 1.5f) / depthF;
		return depthF * (tex3D<float>(textureHandle, texCoord.X(), texCoord.Y(), z1) - tex3D<float>(
			                 textureHandle, texCoord.X(), texCoord.Y(), z0));
	}

	[[nodiscard]] __device__ static VectorType DSampleDTexCoord(const SizeType &volumeSize,
	                                                            cudaTextureObject_t textureHandle,
	                                                            const VectorType &texCoord) {
		return {DSampleDX(volumeSize.X(), textureHandle, texCoord), //
		        DSampleDY(volumeSize.Y(), textureHandle, texCoord), //
		        DSampleDZ(volumeSize.Z(), textureHandle, texCoord)};
	}

	[[nodiscard]] __device__ float DSampleDX(const VectorType &texCoord) const {
		return DSampleDX(Size().X(), textureHandle, texCoord);
	}

	[[nodiscard]] __device__ float DSampleDY(const VectorType &texCoord) const {
		return DSampleDY(Size().Y(), textureHandle, texCoord);
	}

	[[nodiscard]] __device__ float DSampleDZ(const VectorType &texCoord) const {
		return DSampleDZ(Size().Z(), textureHandle, texCoord);
	}

	[[nodiscard]] __device__ VectorType DSampleDTexCoord(const VectorType &texCoord) const {
		return DSampleDTexCoord(Size(), textureHandle, texCoord);
	}

private:
	cudaTextureObject_t textureHandle = 0;
	std::shared_ptr<CUDATexture3D> ownedTexture = nullptr;
};

} // namespace reg23