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

//
// torch::Tensor sample_test(const torch::Tensor &a) {
//
//	TORCH_INTERNAL_ASSERT(a.device().type() == torch::kMPS)
//
//	const uint32_t width = 100;
//
//	torch::Tensor resultFlat = torch::empty({width}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kMPS));
//
//	@autoreleasepool {
//		// Get the default Metal device
//		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
//
//		NSError *error = nil;
//
//		// Load the shader binary
//		dispatch_data_t shaderBinary = dispatch_data_create(src_mps_default_metallib, src_mps_default_metallib_len,
//															NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
//		id<MTLLibrary> library = [device newLibraryWithData:shaderBinary error:&error];
//		if (!library) {
//			throw std::runtime_error("Error loading shader library: " +
//									 std::string(error.localizedDescription.UTF8String));
//		}
//
//		id<MTLFunction> function = [library newFunctionWithName:@"sample_test"];
//		if (!function) {
//			throw std::runtime_error("Error: Metal function sample_test not found.");
//		}
//
//		at::Tensor aContiguous = a.contiguous();
//
//		// create a sampler
//		MTLSamplerDescriptor *samplerDesc = [MTLSamplerDescriptor new];
//		samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
//		samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
//		samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
//		samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
//		samplerDesc.rAddressMode = MTLSamplerAddressModeClampToEdge;
//		id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:samplerDesc];
//		if (!sampler) {
//			throw std::runtime_error("Error creating sampler.");
//		}
//
//		// Create a Metal compute pipeline state
//		id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
//		if (!pipelineState) {
//			NSLog(@"Pipeline error: %@", error);
//			throw std::runtime_error([error.localizedDescription UTF8String]);
//		}
//
//		// Get PyTorch's command queue
//		dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
//
//		// Get PyTorch's Metal command buffer
//		id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
//
//		dispatch_sync(serialQueue, ^() {
//		  // Blit the tensor data into a texture
//		  id<MTLTexture> texture = createTextureFromTensor(device, commandBuffer, aContiguous);
//		  if (!texture) {
//			  throw std::runtime_error("Error creating texture.");
//		  }
//
//		  // Create a compute command encoder
//		  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
//
//		  // Set the compute pipeline state
//		  [encoder setComputePipelineState:pipelineState];
//
//		  // Set the buffers
//		  [encoder setBuffer:getMTLBufferStorage(resultFlat)
//					   offset:resultFlat.storage_offset()
//			  attributeStride:resultFlat.element_size()
//					  atIndex:0];
//		  [encoder setTexture:texture atIndex:0];
//		  [encoder setSamplerState:sampler atIndex:0];
//		  [encoder setBytes:&width length:sizeof(width) atIndex:1];
//
//		  // Dispatch the compute kernel
//		  MTLSize gridSize = MTLSizeMake(width, 1, 1);
//		  NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
//		  if (threadGroupSize > width) {
//			  threadGroupSize = width;
//		  }
//		  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
//		  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
//		  [encoder endEncoding];
//
//		  // Tell torch to commit the command buffer
//		  torch::mps::commit();
//		});
//	}
//
//	return resultFlat;
//}

at::Tensor ProjectDRR_MPS(const at::Tensor &volume, const at::Tensor &voxelSpacing,
						  const at::Tensor &homographyMatrixInverse, double sourceDistance, int64_t outputWidth,
						  int64_t outputHeight, const at::Tensor &outputOffset, const at::Tensor &detectorSpacing) {
	// volume should be a 3D tensor of floats on the chosen device
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::MPS);
	// voxelSpacing should be a 1D tensor of 3 doubles
	TORCH_CHECK(voxelSpacing.sizes() == at::IntArrayRef{3});
	//	TORCH_CHECK(voxelSpacing.dtype() == at::kDouble);
	// homographyMatrixInverse should be of size (4, 4), contain doubles and be on the chosen device
	TORCH_CHECK(homographyMatrixInverse.sizes() == at::IntArrayRef({4, 4}));
	//	TORCH_CHECK(homographyMatrixInverse.dtype() == at::kDouble);
	//	TORCH_INTERNAL_ASSERT(homographyMatrixInverse.device().type() == at::DeviceType::MPS);
	// outputOffset should be a 1D tensor of 2 doubles
	TORCH_CHECK(outputOffset.sizes() == at::IntArrayRef{2});
	//	TORCH_CHECK(outputOffset.dtype() == at::kDouble);
	// detectorSpacing should be a 1D tensor of 2 doubles
	TORCH_CHECK(detectorSpacing.sizes() == at::IntArrayRef{2});
	//	TORCH_CHECK(detectorSpacing.dtype() == at::kDouble);

	at::Tensor volumeContiguous = volume.contiguous();
	Vec<Vec<float, 4>, 4> homographyMatInverse =
		Vec<Vec<float, 4>, 4>::FromTensor2D(homographyMatrixInverse.to(at::dtype<float>()));
	const float sourceDistanceF = static_cast<float>(sourceDistance);
	const Vec<int64_t, 3> inputSize = Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Flipped();
	const Vec<float, 3> volumeDiagonal =
		inputSize.StaticCast<float>() * Vec<float, 3>::FromTensor(voxelSpacing.to(at::dtype<float>()));
	const float volumeDiagLength = volumeDiagonal.Length();
	const Vec<float, 3> sourcePosition = {0.f, 0.f, sourceDistanceF};
	float lambdaStart =
		MatMul(homographyMatInverse, VecCat(sourcePosition, 1.f)).XYZ().Length() - 0.5f * volumeDiagLength;
	float samplesPerRay = static_cast<float>(Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Max());
	float stepSize = volumeDiagLength / samplesPerRay;
	Vec<float, 2> outputOffsetVec = Vec<float, 2>::FromTensor(outputOffset.to(at::dtype<float>()));
	Vec<float, 2> detectorSpacingVec = Vec<float, 2>::FromTensor(detectorSpacing.to(at::dtype<float>()));
	const long long outputNumel = outputWidth * outputHeight;
	at::Tensor flatOutput = torch::empty(at::IntArrayRef({outputNumel}), volumeContiguous.options());
	Vec<uint32_t, 2> outputSizeVec = {static_cast<uint32_t>(outputWidth), static_cast<uint32_t>(outputHeight)};

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

		dispatch_sync(serialQueue, ^() {
		  // Blit the tensor data into a texture
		  id<MTLTexture> texture = createTextureFromTensor(device, commandBuffer, volumeContiguous);
		  if (!texture) {
			  throw std::runtime_error("Error creating texture.");
		  }

		  // Create a compute command encoder
		  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

		  // Set the compute pipeline state
		  [encoder setComputePipelineState:pipelineState];

		  // Set the buffers
		  [encoder setTexture:texture atIndex:0];
		  [encoder setSamplerState:sampler atIndex:0];
		  [encoder setBytes:&sourceDistanceF length:sizeof(sourceDistanceF) atIndex:0];
		  [encoder setBytes:&lambdaStart length:sizeof(lambdaStart) atIndex:1];
		  [encoder setBytes:&stepSize length:sizeof(stepSize) atIndex:2];
		  [encoder setBytes:&samplesPerRay length:sizeof(samplesPerRay) atIndex:3];
		  [encoder setBytes:&homographyMatInverse length:sizeof(homographyMatInverse) atIndex:4];
		  //		  [encoder setBytes:& length:sizeof() atIndex:5];
		  //		  [encoder setBytes:& length:sizeof() atIndex:6];
		  [encoder setBytes:&detectorSpacingVec length:sizeof(detectorSpacingVec) atIndex:7];
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
