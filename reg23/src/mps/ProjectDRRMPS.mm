#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <torch/extension.h>

#include <reg23/ProjectDRR.h>
#include <reg23/default_metallib.h>

namespace reg23 {

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
	return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

id<MTLTexture> createTextureFromTensor(id<MTLDevice> device, const at::Tensor &tensor) {
	TORCH_CHECK(tensor.dim() == 3, "Expected 3D tensor (or 1xD,H,W)")

	auto tensor_contig = tensor.contiguous();
	auto sizes = tensor_contig.sizes();

	std::cout << "ctft A" << std::endl;

	MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType3D;
	desc.pixelFormat = MTLPixelFormatR32Float;
	desc.width = sizes[2];
	desc.height = sizes[1];
	desc.depth = sizes[0];
	desc.mipmapLevelCount = 1;
	desc.usage = MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModeManaged;

	std::cout << "ctft B" << std::endl;

	id<MTLTexture> tex = [device newTextureWithDescriptor:desc];

	std::cout << "ctft C" << std::endl;

	NSUInteger bytesPerRow = sizes[2] * sizeof(float); // ((sizes[2] * sizeof(float) + 255) / 256) * 256;
	[tex replaceRegion:MTLRegionMake3D(0, 0, 0, sizes[2], sizes[1], sizes[0])
		   mipmapLevel:0
				 slice:0
			 withBytes:tensor_contig.data_ptr<float>()
		   bytesPerRow:bytesPerRow
		 bytesPerImage:bytesPerRow * sizes[1]];

	std::cout << "ctft D" << std::endl;

	return tex;
}

torch::Tensor sample_test(const torch::Tensor &a) {

	std::cout << "A" << std::endl;

	TORCH_INTERNAL_ASSERT(a.device().type() == torch::kMPS)

	const uint32_t width = 100;

	torch::Tensor resultFlat = torch::empty({width}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kMPS));

	std::cout << "B" << std::endl;

	@autoreleasepool {
		// Get the default Metal device
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();

		NSError *error = nil;

		// Load the shader binary
		dispatch_data_t shaderBinary = dispatch_data_create(src_mps_default_metallib, src_mps_default_metallib_len,
															NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		id<MTLLibrary> library = [device newLibraryWithData:shaderBinary error:&error];
		if (!library) {
			throw std::runtime_error("Error loading shader library: " +
									 std::string(error.localizedDescription.UTF8String));
		}

		id<MTLFunction> function = [library newFunctionWithName:@"sample_test"];
		if (!function) {
			throw std::runtime_error("Error: Metal function sample_test not found.");
		}

		std::cout << "C" << std::endl;

		// copy the tensor into a texture
		id<MTLTexture> texture = createTextureFromTensor(device, a);
		if (!texture) {
			throw std::runtime_error("Error creating texture.");
		}

		std::cout << "D" << std::endl;

		// create a sampler
		MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
		samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
		samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
		samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
		samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
		samplerDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
		id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
		if (!sampler) {
			throw std::runtime_error("Error creating sampler.");
		}

		std::cout << "E" << std::endl;

		// Create a Metal compute pipeline state
		id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
		if (!pipelineState) {
			NSLog(@"Pipeline error: %@", error);
			throw std::runtime_error([error.localizedDescription UTF8String]);
		}

		// Get PyTorch's command queue
		dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

		// Get PyTorch's Metal command buffer
		id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();

		std::cout << "F" << std::endl;

		dispatch_sync(serialQueue, ^() {
		  // Create a compute command encoder
		  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

		  // Set the compute pipeline state
		  [encoder setComputePipelineState:pipelineState];

		  // Set the buffers
		  [encoder setBuffer:getMTLBufferStorage(resultFlat)
					   offset:resultFlat.storage_offset()
			  attributeStride:resultFlat.element_size()
					  atIndex:0];
		  [encoder setTexture:texture atIndex:0];
		  [encoder setSamplerState:sampler atIndex:0];
		  [encoder setBytes:&width length:sizeof(width) atIndex:1];

		  // Dispatch the compute kernel
		  MTLSize gridSize = MTLSizeMake(width, 1, 1);
		  NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
		  if (threadGroupSize > width) {
			  threadGroupSize = width;
		  }
		  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
		  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
		  [encoder endEncoding];

		  // Tell torch to commit the command buffer
		  torch::mps::commit();
		});
	}

	return resultFlat;
}

} // namespace reg23
