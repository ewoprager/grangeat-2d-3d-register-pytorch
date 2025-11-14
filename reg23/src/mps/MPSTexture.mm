#include <torch/extension.h>

#include <reg23/CommonMPS.h>
#include <reg23/MPSTexture.h>
#include <reg23/Texture.h>

namespace reg23 {

id<MTLTexture> MPSTexture3D::Handle() const { return textureHandle; }
uintptr_t MPSTexture3D::HandleAsInt() const { return reinterpret_cast<uintptr_t>(textureHandle); }
id<MTLSamplerState> MPSTexture3D::SamplerHandle() const { return samplerHandle; }
uintptr_t MPSTexture3D::SamplerHandleAsInt() const { return reinterpret_cast<uintptr_t>(samplerHandle); }

at::Tensor MPSTexture3D::SizeTensor() const { return at::tensor(backingTensor.sizes(), at::dtype(at::kInt)).flip({0}); }

MPSTexture3D::MPSTexture3D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY,
						   const std::string &addressModeZ)
	: MPSTexture3D(MTLCreateSystemDefaultDevice(), torch::mps::get_command_buffer(), tensor,
				   StringsToAddressModes<3>({{addressModeX, addressModeY, addressModeZ}})) {}

MPSTexture3D::MPSTexture3D(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer, const at::Tensor &tensor,
						   Vec<TextureAddressMode, 3> addressModes) {
	// tensor should be a 3-dimensional array of floats on the GPU
	TORCH_CHECK(tensor.sizes().size() == 3);
	TORCH_CHECK(tensor.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(tensor.device().type() == at::DeviceType::MPS);

	backingTensor = tensor.contiguous();

	id<MTLBuffer> buffer = getMTLBufferStorage(backingTensor);
//	CFTypeRef bufRef = (__bridge CFTypeRef)buffer;
//	if (bufRef) CFRetain(bufRef);

	const Vec<int64_t, 3> size = Vec<int64_t, 3>::FromIntArrayRef(backingTensor.sizes()).Flipped();

	const NSUInteger bytesPerRow = size.X() * sizeof(float);
	const NSUInteger bytesPerImage = size.Y() * bytesPerRow;

	MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType3D;
	desc.pixelFormat = MTLPixelFormatR32Float;
	desc.width = size.X();
	desc.height = size.Y();
	desc.depth = size.Z();
	desc.mipmapLevelCount = 1;
	desc.usage = MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModePrivate;

	textureHandle = [device newTextureWithDescriptor:desc];

	id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

	MTLOrigin origin = {0, 0, 0};
	MTLSize mtlSize = {static_cast<NSUInteger>(size.X()), static_cast<NSUInteger>(size.Y()),
					   static_cast<NSUInteger>(size.Z())};
	[blitEncoder copyFromBuffer:buffer
				   sourceOffset:0
			  sourceBytesPerRow:bytesPerRow
			sourceBytesPerImage:bytesPerImage
					 sourceSize:mtlSize
					  toTexture:textureHandle
			   destinationSlice:0
			   destinationLevel:0
			  destinationOrigin:origin];

	[blitEncoder endEncoding];

	MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
	samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
	samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
	samplerDesc.sAddressMode = TextureAddressModeToMPS(addressModes.X());
	samplerDesc.tAddressMode = TextureAddressModeToMPS(addressModes.Y());
	samplerDesc.rAddressMode = TextureAddressModeToMPS(addressModes.Z());
	samplerHandle = [device newSamplerStateWithDescriptor:samplerDesc];
	if (!samplerHandle) {
		throw std::runtime_error("Error creating sampler.");
	}

//	[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
//	  if (bufRef) CFRelease(bufRef);
//	}];
}

void MPSTexture3D::CleanUp() noexcept {
	// Clean-up of texture and sampler is done automatically, as id<MTLTexture> is a reference-counting pointer?
	backingTensor.reset();
}

} // namespace reg23