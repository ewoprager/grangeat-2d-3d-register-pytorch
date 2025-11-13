#pragma once

#include "Common.h"
#include "Texture.h"

namespace reg23 {

class MPSTexture3D {
  public:
	MPSTexture3D(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer, const at::Tensor &tensor,
				 const std::string &addressModeX, const std::string &addressModeY, const std::string &addressModeZ);

	MPSTexture3D(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer, const at::Tensor &tensor,
				 Vec<TextureAddressMode, 3> addressModes);

	[[nodiscard]] id<MTLTexture> Handle() const;

	[[nodiscard]] id<MTLSamplerState> SamplerHandle() const;

	[[nodiscard]] at::Tensor SizeTensor() const;

	/**
	 * @brief Cleans up the underlying PyTorch tensor.
	 *
	 * This is effectively the destructor, but it is defined separately so it can be called independently. This is
	 * because it doesn't seem to be possible to have the destructor of this class called automatically on destruction
	 * when used via Python bindings.
	 */
	void CleanUp() noexcept;

#ifdef USE_MPS

	[[nodiscard]] Vec<int32_t, 3> Size() const {
		return Vec<int32_t, 3>::FromIntArrayRef(backingTensor.sizes()).Flipped();
	}

	// no copy
	MPSTexture3D(const MPSTexture3D &) = delete;

	void operator=(const MPSTexture3D &) = delete;

	// yes move
	MPSTexture3D(MPSTexture3D &&other) noexcept
		: backingTensor(std::move(other.backingTensor)), textureHandle(other.textureHandle) {
		other.textureHandle = nullptr;
	}

	MPSTexture3D &operator=(MPSTexture3D &&other) noexcept {
		backingTensor = std::move(other.backingTensor);
		textureHandle = other.textureHandle;
		other.textureHandle = nullptr;
		return *this;
	}

	~MPSTexture3D() {
#ifdef DEBUG
		std::cout << "[C++] MPSTexture3D destructing." << std::endl << std::flush;
#endif
		CleanUp();
	}

  private:
	at::Tensor backingTensor{};
	id<MTLTexture> textureHandle = nullptr;
	id<MTLSamplerState> samplerHandle = nullptr;
#endif
};

} // namespace reg23