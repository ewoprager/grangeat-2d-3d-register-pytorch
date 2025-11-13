#include <reg23/CommonMPS.h>
#include <reg23/ProjectDRR.h>
#include <reg23/Texture3DMPS.h>
#include <reg23/default_metallib.h>

namespace reg23 {

using CommonData = ProjectDRR<Texture3DMPS>::CommonData;

at::Tensor ProjectDRR_MPS(const at::Tensor &volume, const at::Tensor &voxelSpacing,
						  const at::Tensor &homographyMatrixInverse, double sourceDistance, int64_t outputWidth,
						  int64_t outputHeight, const at::Tensor &outputOffset, const at::Tensor &detectorSpacing) {

	float sourceDistanceF = static_cast<float>(sourceDistance);
	const Vec<int32_t, 2> outputSizeVec = {static_cast<int32_t>(outputWidth), static_cast<int32_t>(outputHeight)};
	const Vec<float, 2> outputOffsetVec = Vec<float, 2>::FromTensor(outputOffset.to(at::dtype<float>()));
	const int32_t outputNumel = outputWidth * outputHeight;
	at::Tensor flatOutput = torch::zeros(at::IntArrayRef({outputNumel}), volume.contiguous().options());

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

		id<MTLFunction> function = [library newFunctionWithName:@"project_drr"];
		if (!function) {
			throw std::runtime_error("Error: Metal function sample_test not found.");
		}

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

		CommonData common =
			ProjectDRR<Texture3DMPS>::Common(volume, voxelSpacing, homographyMatrixInverse, sourceDistance,
											 outputOffset, detectorSpacing, at::DeviceType::MPS);

		dispatch_sync(serialQueue, ^() {
		  // Blit the tensor data into a texture
		  Texture3DMPS inputTexture = Texture3DMPS::FromTensor(device, commandBuffer, volume, common.spacing);

		  const Linear<Texture3DMPS::VectorType> texCoordMapping = inputTexture.MappingWorldToTexCoord();

		  // Create a compute command encoder
		  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

		  // Set the compute pipeline state
		  [encoder setComputePipelineState:pipelineState];

		  // Set the buffers
		  [encoder setTexture:inputTexture.GetHandle() atIndex:0];
		  [encoder setSamplerState:inputTexture.GetSamplerHandle() atIndex:0];
		  [encoder setBytes:&sourceDistanceF length:sizeof(sourceDistanceF) atIndex:0];
		  [encoder setBytes:&common.lambdaStart length:sizeof(common.lambdaStart) atIndex:1];
		  [encoder setBytes:&common.stepSize length:sizeof(common.stepSize) atIndex:2];
		  [encoder setBytes:&common.samplesPerRay length:sizeof(common.samplesPerRay) atIndex:3];
		  [encoder setBytes:&common.homographyMatrixInverse length:sizeof(common.homographyMatrixInverse) atIndex:4];
		  [encoder setBytes:&texCoordMapping.intercept length:sizeof(texCoordMapping.intercept) atIndex:5];
		  [encoder setBytes:&texCoordMapping.gradient length:sizeof(texCoordMapping.gradient) atIndex:6];
		  [encoder setBytes:&common.detectorSpacing length:sizeof(common.detectorSpacing) atIndex:7];
		  [encoder setBytes:&outputSizeVec length:sizeof(outputSizeVec) atIndex:8];
		  [encoder setBytes:&outputOffsetVec length:sizeof(outputOffsetVec) atIndex:9];
		  [encoder setBuffer:getMTLBufferStorage(flatOutput)
					   offset:flatOutput.storage_offset()
			  attributeStride:flatOutput.element_size()
					  atIndex:10];

		  // Dispatch the compute kernel
		  MTLSize gridSize = MTLSizeMake(outputNumel, 1, 1);
		  NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
		  if (threadGroupSize > outputNumel) {
			  threadGroupSize = outputNumel;
		  }
		  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
		  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
		  [encoder endEncoding];

		  // Tell torch to commit the command buffer
		  torch::mps::commit();
		});
	}

	return flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

} // namespace reg23
