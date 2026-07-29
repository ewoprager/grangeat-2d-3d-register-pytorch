#include <torch/extension.h>

#include <reg23_core/ObjectiveFunction.h>

namespace reg23 {

/*
 * 6 values per thread in the buffer:
 *	- w sum
 *	- wx sum
 *	- wy sum
 *	- wx^2 sum
 *	- wy^2 sum
 *	- wxy sum
 */
constexpr long sharedValueCount = 6;

__global__ void Kernel_ObjectiveFunction_CUDA(Texture3DCUDA texture, const float *fixedImage, DRRParams drrParams,
											  MaskGeometry maskGeometry, int64_t imageCount, const double *invHMatrices,
											  Vec<double, 2> detectorSpacing, Vec<int64_t, 2> imageSize,
											  int blocksPerImage, double sourceDistance, float weightAlpha,
											  Vec<double, 2> outputOffset, double *blockSumsArray) {
	extern __shared__ double buffer[];

	const uint64_t imageIndex = blockIdx.x / blocksPerImage;
	if (imageIndex >= imageCount) return;
	const uint64_t imageNumel = imageSize.X() * imageSize.Y();
	const uint64_t blockIndex = blockIdx.x % blocksPerImage;
	const uint64_t pixelIndex = blockIndex * blockDim.x + threadIdx.x;

	const long sharedBufferIndex = threadIdx.x * sharedValueCount;
	if (pixelIndex >= imageNumel) {
		for (long i = 0; i < sharedValueCount; ++i)
			buffer[sharedBufferIndex + i] = 0.0;
		return;
	}

	const uint64_t i = pixelIndex % imageSize.X();
	const uint64_t j = pixelIndex / imageSize.X();

	const Vec<double, 2> detectorPosition = EvaluateDetectorPosition(i, j, detectorSpacing, imageSize, outputOffset);

	Vec<Vec<double, 4>, 4> homographyMatrixInverse{};
	for (int k = 0; k < 16; ++k)
		homographyMatrixInverse[k % 4][k / 4] = invHMatrices[16 * imageIndex + k];

	const float drrIntensity = DRRRay(texture, drrParams, homographyMatrixInverse, detectorPosition, sourceDistance);
	const float maskValue = DRRCuboidMaskRay(detectorPosition, sourceDistance, maskGeometry, homographyMatrixInverse);
	const float maskedFixedImageValue = maskValue * fixedImage[pixelIndex];
	const float weightValue =
		weightAlpha > 1e-3 ? pow(3.f * maskValue * maskValue - 2.f * maskValue * maskValue * maskValue,
											1.f / (weightAlpha * weightAlpha))
						   : (maskValue > 1.0 - 1.0e-3 ? 1.0 : 0.0);

	buffer[sharedBufferIndex + 0] = weightValue;
	buffer[sharedBufferIndex + 1] = weightValue * drrIntensity;
	buffer[sharedBufferIndex + 2] = weightValue * maskedFixedImageValue;
	buffer[sharedBufferIndex + 3] = weightValue * drrIntensity * drrIntensity;
	buffer[sharedBufferIndex + 4] = weightValue * maskedFixedImageValue * maskedFixedImageValue;
	buffer[sharedBufferIndex + 5] = weightValue * drrIntensity * maskedFixedImageValue;

	__syncthreads();

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff) {
			const long sumWith = sharedBufferIndex + cutoff * sharedValueCount;
			for (long k = 0; k < sharedValueCount; ++k)
				buffer[sharedBufferIndex + k] += buffer[sumWith + k];
		}

		__syncthreads();
	}

	if (threadIdx.x < sharedValueCount) {
		blockSumsArray[										 //
			imageIndex * blocksPerImage * sharedValueCount + //
			blockIndex * sharedValueCount +					 //
			threadIdx.x] = buffer[threadIdx.x];
	}
}

int blockSizeToDynamicSMemSize_ObjectiveFunction_CUDA(int blockSize) {
	return sharedValueCount * blockSize * static_cast<int>(sizeof(double));
}

__host__ at::Tensor ObjectiveFunction_CUDA(const at::Tensor &volume, const at::Tensor &fixedImage,
										   const at::Tensor &voxelSpacing, const at::Tensor &invHMatrices,
										   double sourceDistance,
										   const at::Tensor &outputOffset, const at::Tensor &detectorSpacing,
										   double weightAlpha) {
	// volume should be a 3D tensor of floats on the chosen device
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// fixedImage should be a 2D tensor of floats on the chosen device
	TORCH_CHECK(fixedImage.sizes().size() == 2);
	TORCH_CHECK(fixedImage.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(fixedImage.device().type() == at::DeviceType::CUDA);
	// voxelSpacing should be a 1D tensor of 3 doubles
	TORCH_CHECK(voxelSpacing.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(voxelSpacing.dtype() == at::kDouble);
	// homographyMatrixInverse should be of size (..., 4, 4)
	TORCH_CHECK(invHMatrices.sizes().size() > 2);
	TORCH_CHECK(invHMatrices.sizes()[invHMatrices.sizes().size() - 2] == 4);
	TORCH_CHECK(invHMatrices.sizes()[invHMatrices.sizes().size() - 1] == 4);
	// outputOffset should be a 1D tensor of 2 doubles
	TORCH_CHECK(outputOffset.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(outputOffset.dtype() == at::kDouble);
	// detectorSpacing should be a 1D tensor of 2 doubles
	TORCH_CHECK(detectorSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(detectorSpacing.dtype() == at::kDouble);

	const Texture3DCUDA inputTexture =
		Texture3DCUDA::FromTensor(volume, Texture3DCUDA::VectorType::FromTensor(voxelSpacing));

	const int64_t outputWidth = fixedImage.sizes()[1];
	const int64_t outputHeight = fixedImage.sizes()[0];

	const at::Tensor invHMatricesContiguous = invHMatrices.to(at::kCUDA, at::kDouble).contiguous();
	const at::Tensor fixedImageContiguous = fixedImage.to(at::kCUDA, at::kFloat).contiguous();

	const DRRParams drrParams = DRRParams::Evaluate(inputTexture);
	const MaskGeometry maskGeometry = MaskGeometry::Evaluate(inputTexture);

	const at::IntArrayRef batchSizes = invHMatrices.sizes().slice(0, invHMatrices.sizes().size() - 2);
	long imageCount = 1;
	for (auto n : batchSizes)
		imageCount *= n;

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_ObjectiveFunction_CUDA,
												   &blockSizeToDynamicSMemSize_ObjectiveFunction_CUDA, 0);
	const size_t bufferSize = blockSizeToDynamicSMemSize_ObjectiveFunction_CUDA(blockSize);
	const int threadsPerImage = static_cast<int>(outputWidth) * static_cast<int>(outputHeight);
	const int blocksPerImage = (threadsPerImage + blockSize - 1) / blockSize;
	const int gridSize = imageCount * blocksPerImage;
	// stores the sums for each kernel block of w, wx, wy, wx^2, wy^2 and wxy, for each image
	at::Tensor blockSums = torch::zeros(at::IntArrayRef({imageCount, blocksPerImage, sharedValueCount}),
										torch::TensorOptions{}.dtype(torch::kDouble).device(at::kCUDA));
	double *blockSumsPtr = blockSums.data_ptr<double>();

	Kernel_ObjectiveFunction_CUDA<<<gridSize, blockSize, bufferSize>>>(
		inputTexture, fixedImageContiguous.data_ptr<float>(), drrParams, maskGeometry, imageCount,
		invHMatricesContiguous.data_ptr<double>(), Vec<double, 2>::FromTensor(detectorSpacing),
		Vec<int64_t, 2>{outputWidth, outputHeight}, blocksPerImage, sourceDistance, weightAlpha,
		Vec<double, 2>::FromTensor(outputOffset), blockSumsPtr);

	const at::Tensor sums = blockSums.sum({1});						  // size = (imageCount, sharedValueCount)
	const at::Tensor sums_w = sums.index({at::indexing::Slice(), 0}); // size = (imageCount,)
	const at::Tensor sums_wx = sums.index({at::indexing::Slice(), 1});
	const at::Tensor sums_wy = sums.index({at::indexing::Slice(), 2});
	const at::Tensor sums_wx2 = sums.index({at::indexing::Slice(), 3});
	const at::Tensor sums_wy2 = sums.index({at::indexing::Slice(), 4});
	const at::Tensor sums_wxy = sums.index({at::indexing::Slice(), 5});

	at::Tensor numerators = sums_w * sums_wxy - sums_wx * sums_wy;
	at::Tensor denominatorsLeft = sums_w * sums_wx2 - sums_wx.square();
	at::Tensor denominatorsRight = sums_w * sums_wy2 - sums_wy.square();
	auto invalid = torch::logical_or(denominatorsLeft < 1.0e-10, denominatorsRight < 1.0e-10);
	numerators.index_put_({invalid}, 0.0);
	denominatorsLeft.index_put_({invalid}, 1.0);
	denominatorsLeft.index_put_({invalid}, 1.0);
	return numerators / (denominatorsLeft.sqrt() * denominatorsRight.sqrt() + 1e-10);
}

} // namespace reg23