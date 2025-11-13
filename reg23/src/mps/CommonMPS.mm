#include <reg23/CommonMPS.h>

namespace reg23 {

id<MTLTexture> createTextureFromTensor(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
									   const at::Tensor &contigTensor) {
	TORCH_CHECK(contigTensor.dim() == 3, "Expected 3D tensor (or 1xD,H,W)")

	const at::IntArrayRef sizes = contigTensor.sizes();
	id<MTLBuffer> buffer = getMTLBufferStorage(contigTensor);

	const NSUInteger bytesPerRow = sizes[2] * sizeof(float);
	const NSUInteger bytesPerImage = sizes[1] * bytesPerRow;

	MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType3D;
	desc.pixelFormat = MTLPixelFormatR32Float;
	desc.width = sizes[2];
	desc.height = sizes[1];
	desc.depth = sizes[0];
	desc.mipmapLevelCount = 1;
	desc.usage = MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModePrivate;

	id<MTLTexture> ret = [device newTextureWithDescriptor:desc];

	id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

	MTLOrigin origin = {0, 0, 0};
	MTLSize size = {static_cast<NSUInteger>(sizes[2]), static_cast<NSUInteger>(sizes[1]),
					static_cast<NSUInteger>(sizes[0])};
	[blitEncoder copyFromBuffer:buffer
				   sourceOffset:0
			  sourceBytesPerRow:bytesPerRow
			sourceBytesPerImage:bytesPerImage
					 sourceSize:size
					  toTexture:ret
			   destinationSlice:0
			   destinationLevel:0
			  destinationOrigin:origin];

	[blitEncoder endEncoding];

	return ret;
}

} // namespace reg23