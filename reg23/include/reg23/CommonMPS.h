#pragma once

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <torch/extension.h>

#include <reg23/default_metallib.h>

namespace reg23 {

inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
	return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

struct MPSComputeEnvironment {
	id<MTLDevice> device;
	id<MTLComputePipelineState> pipelineState;
	dispatch_queue_t serialQueue;
	id<MTLCommandBuffer> commandBuffer;

	/**
	 *
	 * @param shaderName
	 * @return
	 *
	 * Must be called inside an `@autoreleasepool`
	 */
	static MPSComputeEnvironment Create(NSString *shaderName) {
		MPSComputeEnvironment ret{};
		// Get the default Metal device
		ret.device = MTLCreateSystemDefaultDevice();

		NSError *error = nil;

		// Load the shader binary
		dispatch_data_t shaderBinary = dispatch_data_create(src_mps_default_metallib, src_mps_default_metallib_len,
															NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		id<MTLLibrary> library = [ret.device newLibraryWithData:shaderBinary error:&error];
		if (!library) {
			throw std::runtime_error("Error loading shader library: " +
									 std::string(error.localizedDescription.UTF8String));
		}

		id<MTLFunction> function = [library newFunctionWithName:shaderName];
		if (!function) {
			throw std::runtime_error("Error: Metal function sample_test not found.");
		}

		// Create a Metal compute pipeline state
		ret.pipelineState = [ret.device newComputePipelineStateWithFunction:function error:&error];
		if (!ret.pipelineState) {
			NSLog(@"Pipeline error: %@", error);
			throw std::runtime_error([error.localizedDescription UTF8String]);
		}

		// Get PyTorch's command queue
		ret.serialQueue = torch::mps::get_dispatch_queue();

		// Get PyTorch's Metal command buffer
		ret.commandBuffer = torch::mps::get_command_buffer();

		return ret;
	}
};

} // namespace reg23
