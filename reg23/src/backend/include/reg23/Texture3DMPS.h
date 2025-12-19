#pragma once

#include "CommonMPS.h"
#include "MPSTexture.h"
#include "Texture.h"

namespace reg23 {

/**
 * @ingroup textures
 * @brief A 3D texture stored for access by a Metal device
 *
 * **This object does not own the data it provides access to, it just holds a pointer to it**
 *
 * Move constructable but not copy constructable.
 */
class Texture3DMPS : public Texture<3, int32_t, float> {
  public:
	using Base = Texture<3, int32_t, float>;

	Texture3DMPS() = default;

	Texture3DMPS(std::shared_ptr<MPSTexture3D> mpsTexture, VectorType _spacing, VectorType _centrePosition = {})
		: Base(mpsTexture->Size(), std::move(_spacing), std::move(_centrePosition)),
		  textureHandle(mpsTexture->Handle()), samplerHandle(mpsTexture->SamplerHandle()),
		  ownedTexture(std::move(mpsTexture)) {}

	// yes copy
	Texture3DMPS(const Texture3DMPS &) = default;

	Texture3DMPS &operator=(const Texture3DMPS &) = default;

	// yes move
	Texture3DMPS(Texture3DMPS &&) noexcept = default;

	Texture3DMPS &operator=(Texture3DMPS &&) noexcept = default;

	/**
	 *
	 * @param volume
	 * @param spacing
	 * @param centrePosition
	 * @param addressModes
	 * @return
	 */
	static Texture3DMPS FromTensor(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer, const at::Tensor &volume,
								   VectorType spacing, VectorType centrePosition = VectorType::Full(0),
								   AddressModeType addressModes = AddressModeType::Full(TextureAddressMode::ZERO)) {
		return {std::make_shared<MPSTexture3D>(device, commandBuffer, volume, addressModes), std::move(spacing),
				centrePosition};
	}

	[[nodiscard]] id<MTLTexture> GetHandle() const { return textureHandle; }
	[[nodiscard]] id<MTLSamplerState> GetSamplerHandle() const { return samplerHandle; }

  private:
	id<MTLTexture> textureHandle = nullptr;
	id<MTLSamplerState> samplerHandle = nullptr;
	std::shared_ptr<MPSTexture3D> ownedTexture = nullptr;
};

} // namespace reg23