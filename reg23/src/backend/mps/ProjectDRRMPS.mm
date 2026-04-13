#include <reg23/CommonMPS.h>
#include <reg23/ProjectDRR.h>
#include <reg23/Texture3DMPS.h>

namespace reg23 {

using CommonData = ProjectDRR<Texture3DMPS>::CommonData;

struct alignas(16) ProjectDRRArgs {
    float hmi[16];

    Vec<float, 3> mappingIntercept;
    float _pad0; // padding to 16

    Vec<float, 3> mappingGradient;
    float _pad1; // padding to 16

    Vec<float, 2> detectorSpacing;
    Vec<uint32_t, 2> outputSize;
    Vec<float, 2> outputOffset;

    float sourceDistance;
    float lambdaStart;
    float stepSize;
    int32_t samplesPerRay;
};

static_assert(sizeof(ProjectDRRArgs) % 16 == 0);

at::Tensor ProjectDRR_MPS(const at::Tensor &volume, const at::Tensor &voxelSpacing,
						  const at::Tensor &homographyMatrixInverse, double sourceDistance, int64_t outputWidth,
						  int64_t outputHeight, const at::Tensor &outputOffset, const at::Tensor &detectorSpacing) {

    std::cout << "Projecting on MPS!" << std::endl;

	float sourceDistanceF = static_cast<float>(sourceDistance);
	const Vec<uint32_t, 2> outputSizeVec = {static_cast<uint32_t>(outputWidth), static_cast<uint32_t>(outputHeight)};
	const Vec<float, 2> outputOffsetVec = Vec<float, 2>::FromTensor(outputOffset.to(at::dtype<float>()));
	const int64_t outputNumel = outputWidth * outputHeight;
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

	      id<MTLBuffer> argsBuffer = [computeEnv.device newBufferWithLength:sizeof(ProjectDRRArgs)
	                                                            options:MTLResourceStorageModeShared];
          CFRetain((__bridge CFTypeRef)argsBuffer);
          [computeEnv.commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              CFRelease((__bridge CFTypeRef)argsBuffer);
          }];
	      const Linear<Texture3DMPS::VectorType> texCoordMapping = inputTexture.MappingWorldToTexCoord();
          ProjectDRRArgs *args = reinterpret_cast<ProjectDRRArgs*>([argsBuffer contents]);
          std::memcpy(args->hmi, &common.homographyMatrixInverse, sizeof(args->hmi));
          args->mappingIntercept = texCoordMapping.intercept;
          args->mappingGradient  = texCoordMapping.gradient;
          args->detectorSpacing  = common.detectorSpacing;
          args->outputSize       = outputSizeVec;
          args->outputOffset     = outputOffsetVec;
          args->sourceDistance   = sourceDistanceF;
          args->lambdaStart      = common.lambdaStart;
          args->stepSize         = common.stepSize;
          args->samplesPerRay    = common.samplesPerRay;

	      // Create a compute command encoder
	      id<MTLComputeCommandEncoder> encoder = [computeEnv.commandBuffer computeCommandEncoder];

	      // Set the compute pipeline state
	      [encoder setComputePipelineState:computeEnv.pipelineState];

	      // Set the buffers
	      [encoder setTexture:inputTexture.GetHandle() atIndex:0];
	      [encoder setSamplerState:inputTexture.GetSamplerHandle() atIndex:0];
		  [encoder setBuffer:argsBuffer offset:0 atIndex:0];
	      [encoder setBuffer:getMTLBufferStorage(flatOutput)
				       offset:flatOutput.storage_offset() * flatOutput.element_size()
//		      attributeStride:flatOutput.element_size()
				      atIndex:1];

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
	      torch::mps::synchronize();
		});
	}

	return flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

} // namespace reg23
