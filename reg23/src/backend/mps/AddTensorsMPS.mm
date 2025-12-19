#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include <reg23/default_metallib.h>
#include <reg23/ProjectDRR.h>

namespace reg23 {

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
	return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Define a function to add tensors using Metal
torch::Tensor add_tensors_metal(const torch::Tensor &a, const torch::Tensor &b) {

	// Check that device is MPS
	TORCH_INTERNAL_ASSERT(a.device().type() == torch::kMPS)
	TORCH_INTERNAL_ASSERT(b.device().type() == torch::kMPS)
	TORCH_CHECK(a.sizes() == b.sizes())

	// Check that tensors are contiguous
	// Contiguous means that the memory is contiguous
	torch::Tensor aContiguous = a.contiguous();
	torch::Tensor bContiguous = b.contiguous();

	// Get the total number of elements in the tensors
	int numElements = aContiguous.numel();

	// Create an empty tensor on the MPS device to hold the result
	torch::Tensor resultFlat =
		torch::empty({numElements}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kMPS));

	@autoreleasepool {
		// Get the default Metal device
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();

		NSError *error = nil;

		// Load the shader binary
		dispatch_data_t shaderBinary = dispatch_data_create(src_backend_mps_default_metallib, src_backend_mps_default_metallib_len,
															NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		id<MTLLibrary> library = [device newLibraryWithData:shaderBinary error:&error];
		if (!library) {
			throw std::runtime_error("Error compiling Metal shader: " +
									 std::string(error.localizedDescription.UTF8String));
		}

		id<MTLFunction> function = [library newFunctionWithName:@"addTensors"];
		if (!function) {
			throw std::runtime_error("Error: Metal function addTensors not found.");
		}

		// Create a Metal compute pipeline state
		id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:nil];

		// Get PyTorch's command queue
		dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

		// Get PyTorch's Metal command buffer
		id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();

		dispatch_sync(serialQueue, ^() {
		  // Create a compute command encoder
		  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

		  // Set the compute pipeline state
		  [encoder setComputePipelineState:pipelineState];

		  // Set the buffers
		  [encoder setBuffer:getMTLBufferStorage(aContiguous)
					   offset:aContiguous.storage_offset()
			  attributeStride:aContiguous.element_size()
					  atIndex:0];
		  [encoder setBuffer:getMTLBufferStorage(bContiguous)
					   offset:aContiguous.storage_offset()
			  attributeStride:aContiguous.element_size()
					  atIndex:1];
		  [encoder setBuffer:getMTLBufferStorage(resultFlat)
					   offset:resultFlat.storage_offset()
			  attributeStride:resultFlat.element_size()
					  atIndex:2];

		  // Dispatch the compute kernel
		  MTLSize gridSize = MTLSizeMake(numElements, 1, 1);
		  NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
		  if (threadGroupSize > numElements) {
			  threadGroupSize = numElements;
		  }
		  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
		  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
		  [encoder endEncoding];

		  // Tell torch to commit the command buffer
		  torch::mps::commit();
		});
	}

	return resultFlat.view(a.sizes());
}

} // namespace reg23