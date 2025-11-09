#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include <reg23/default_metallib.h>
#include <reg23/ProjectDRR.h>

namespace reg23 {

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id <MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
	return __builtin_bit_cast(id < MTLBuffer > , tensor.storage().data());
}

id<MTLTexture> createTextureFromTensor(id<MTLDevice> device, const at::Tensor& tensor) {
	TORCH_CHECK(tensor.dim() == 3,
				"Expected 3D tensor (or 1xD,H,W)");
	
	auto tensor_contig = tensor.contiguous();
	auto sizes = tensor_contig.sizes();
	
	MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
	desc.textureType = MTLTextureType3D;
	desc.pixelFormat = MTLPixelFormatR32Float;
	desc.width  = sizes[2];
	desc.height = sizes[1];
	desc.depth  = sizes[0];
	desc.mipmapLevelCount = 1;
	desc.usage = MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModeManaged;
	
	id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
	[tex replaceRegion:MTLRegionMake3D(0, 0, 0, sizes[2], sizes[1], sizes[0])
		   mipmapLevel:0
				 slice:0
			 withBytes:tensor_contig.data_ptr<float>()
		   bytesPerRow:sizes[-1] * sizeof(float)
		 bytesPerImage:sizes[-1] * sizes[-2] * sizeof(float)];
	return tex;
}

torch::Tensor sample_test(const torch::Tensor &a) {
	
	TORCH_INTERNAL_ASSERT(a.device().type() == torch::kMPS)
	
	torch::Tensor resultFlat = torch::empty({1},
											torch::TensorOptions().dtype(torch::kFloat).device(torch::kMPS));
	
	@autoreleasepool {
		// Get the default Metal device
		id <MTLDevice> device = MTLCreateSystemDefaultDevice();
		
		NSError *error = nil;
		
		// Load the shader binary
		dispatch_data_t shaderBinary = dispatch_data_create(src_mps_default_metallib,
															src_mps_default_metallib_len, NULL,
															DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		id <MTLLibrary> library = [device newLibraryWithData:shaderBinary error:&error];
		if (!library) {
			throw std::runtime_error(
									 "Error compiling Metal shader: " + std::string(error.localizedDescription.UTF8String));
		}
		
		id <MTLFunction> function = [library newFunctionWithName:@"sample_test"];
		if (!function) {
			throw std::runtime_error("Error: Metal function addTensors not found.");
		}
		
		// copy the tensor into a texture
		id <MTLTexture> texture = createTextureFromTensor(device, a);
		// create a sampler
		MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
		samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
		samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
		samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
		samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
		samplerDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
		id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
		
		
		// Create a Metal compute pipeline state
		id <MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:nil];
		
		// Get PyTorch's command queue
		dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
		
		// Get PyTorch's Metal command buffer
		id <MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
		
		dispatch_sync(serialQueue, ^() {
			// Create a compute command encoder
			id <MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
			
			// Set the compute pipeline state
			[encoder setComputePipelineState:pipelineState];
			
			// Set the buffers
			[encoder setBuffer:getMTLBufferStorage(resultFlat)
						offset:resultFlat.storage_offset()
			   attributeStride:resultFlat.element_size()
					   atIndex:0];
			[encoder setTexture:texture atIndex:0];
			[encoder setSamplerState:sampler atIndex:0];
			
			
			// Dispatch the compute kernel
			MTLSize gridSize = MTLSizeMake(1, 1, 1);
			NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
			if (threadGroupSize > 1) {
				threadGroupSize = 1;
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
