#pragma once

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <torch/extension.h>

namespace reg23 {

inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
	return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

id<MTLTexture> createTextureFromTensor(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
									   const at::Tensor &contigTensor);

} // namespace reg23
