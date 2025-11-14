#include <reg23/CommonMPS.h>
#include <reg23/ProjectDRR.h>
#include <reg23/Texture3DMPS.h>

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
		MPSComputeEnvironment computeEnv = MPSComputeEnvironment::Create(@"project_drr");

		CommonData common =
			ProjectDRR<Texture3DMPS>::Common(volume, voxelSpacing, homographyMatrixInverse, sourceDistance,
											 outputOffset, detectorSpacing, at::DeviceType::MPS);

		dispatch_sync(computeEnv.serialQueue, ^() {
		  // Blit the tensor data into a texture
		  Texture3DMPS inputTexture =
			  Texture3DMPS::FromTensor(computeEnv.device, computeEnv.commandBuffer, volume, common.spacing);

		  const Linear<Texture3DMPS::VectorType> texCoordMapping = inputTexture.MappingWorldToTexCoord();

		  // Create a compute command encoder
		  id<MTLComputeCommandEncoder> encoder = [computeEnv.commandBuffer computeCommandEncoder];

		  // Set the compute pipeline state
		  [encoder setComputePipelineState:computeEnv.pipelineState];

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
		  NSUInteger threadGroupSize = computeEnv.pipelineState.maxTotalThreadsPerThreadgroup;
		  if (threadGroupSize > outputNumel) {
			  threadGroupSize = outputNumel;
		  }
		  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
		  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
		  [encoder endEncoding];

		  //		  [computeEnv.commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
		  //			delete inputTexture; // destructor releases texture & sampler
		  //		  }];

		  // Tell torch to commit the command buffer
		  torch::mps::commit();
		});
	}

	return flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

} // namespace reg23
