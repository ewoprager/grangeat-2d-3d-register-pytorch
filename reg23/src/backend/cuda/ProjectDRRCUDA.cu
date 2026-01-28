#include <torch/extension.h>

#include <reg23/Texture3DCUDA.h>
#include <reg23/ProjectDRR.h>

namespace reg23 {

using CommonData = ProjectDRR<Texture3DCUDA>::CommonData;

__global__ void Kernel_ProjectDRR_CUDA(Texture3DCUDA volume, double sourceDistance, double lambdaStart, double stepSize,
                                       int64_t samplesPerRay, Vec<Vec<double, 4>, 4> homographyMatrixInverse,
                                       Linear<Texture3DCUDA::VectorType> mappingWorldToTexCoord,
                                       Vec<double, 2> detectorSpacing, Vec<int64_t, 2> outputSize,
                                       Vec<double, 2> outputOffset, float *arrayOut) {
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= outputSize.X() * outputSize.Y()) return;
	const uint64_t i = threadIndex % outputSize.X();
	const uint64_t j = threadIndex / outputSize.X();
	const Vec<double, 2> detectorPosition = detectorSpacing * (Vec<uint64_t, 2>{i, j}.StaticCast<double>() - 0.5 *
	                                                           (outputSize - int64_t{1}).StaticCast<double>()) +
	                                        outputOffset;
	Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
	direction /= direction.Length();
	Vec<double, 3> delta = direction * stepSize;
	delta = MatMul(homographyMatrixInverse, VecCat(delta, 0.0)).XYZ();
	Vec<double, 3> start = Vec<double, 3>{0.0, 0.0, sourceDistance} + lambdaStart * direction;
	start = MatMul(homographyMatrixInverse, VecCat(start, 1.0)).XYZ();

	Vec<double, 3> samplePoint = start;
	float sum = 0.f;
	for (int k = 0; k < samplesPerRay; ++k) {
		sum += volume.Sample(mappingWorldToTexCoord(samplePoint));
		samplePoint += delta;
	}
	arrayOut[threadIndex] = static_cast<float>(stepSize) * sum;
}

__host__ at::Tensor ProjectDRR_CUDA(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                                    const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                    int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                    const at::Tensor &detectorSpacing) {
	const CommonData common = ProjectDRR<Texture3DCUDA>::Common(volume, voxelSpacing, homographyMatrixInverse, sourceDistance,
	                                                      outputOffset, detectorSpacing, at::DeviceType::CUDA);
	at::Tensor flatOutput = torch::zeros(at::IntArrayRef({outputWidth * outputHeight}), volume.contiguous().options());
	float *resultFlatPtr = flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_ProjectDRR_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_ProjectDRR_CUDA<<<gridSize, blockSize>>>(common.inputTexture, sourceDistance, common.lambdaStart,
	                                                common.stepSize, common.samplesPerRay,
	                                                common.homographyMatrixInverse,
	                                                common.inputTexture.MappingWorldToTexCoord(),
	                                                common.detectorSpacing, Vec<int64_t, 2>{outputWidth, outputHeight},
	                                                common.outputOffset, resultFlatPtr);

	return flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

__global__ void Kernel_ProjectDRRsBatched_CUDA(Texture3DCUDA volume, double sourceDistance, double stepSize,
										double volumeDiagLength,
									   int64_t samplesPerRay, int64_t batchCount, double *invHMatrices,
									   Linear<Texture3DCUDA::VectorType> mappingWorldToTexCoord,
									   Vec<double, 2> detectorSpacing, Vec<int64_t, 2> outputSize,
									   Vec<double, 2> outputOffset, float *arrayOut) {
	const uint64_t outputNumel = outputSize.X() * outputSize.Y();
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= outputNumel * batchCount) return;
	const uint64_t batchIndex = threadIndex / outputNumel;
	const uint64_t pixelIndex = threadIndex % outputNumel;
	const uint64_t i = pixelIndex % outputSize.X();
	const uint64_t j = pixelIndex / outputSize.X();
	const Vec<double, 2> detectorPosition = detectorSpacing * (Vec<uint64_t, 2>{i, j}.StaticCast<double>() - 0.5 *
															   (outputSize - int64_t{1}).StaticCast<double>()) +
											outputOffset;
	Vec<Vec<double, 4>, 4> homographyMatrixInverse{};
	for (int k=0; k<16; ++k) homographyMatrixInverse[k / 4][k % 4] = invHMatrices[16 * batchIndex + k]; // ToDo: Transpose??
	Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
	direction /= direction.Length();
	Vec<double, 3> delta = direction * stepSize;
	delta = MatMul(homographyMatrixInverse, VecCat(delta, 0.0)).XYZ();
	const Texture3DCUDA::VectorType sourcePosition = {0.0, 0.0, sourceDistance};
	const float lambdaStart = MatMul(homographyMatrixInverse, VecCat(sourcePosition, 1.0)).XYZ().Length() - 0.5 *
	                  volumeDiagLength;
	Vec<double, 3> start = Vec<double, 3>{0.0, 0.0, sourceDistance} + lambdaStart * direction;
	start = MatMul(homographyMatrixInverse, VecCat(start, 1.0)).XYZ();

	Vec<double, 3> samplePoint = start;
	float sum = 0.f;
	for (int k = 0; k < samplesPerRay; ++k) {
		sum += volume.Sample(mappingWorldToTexCoord(samplePoint));
		samplePoint += delta;
	}
	arrayOut[batchIndex * outputNumel + pixelIndex] = static_cast<float>(stepSize) * sum;
}

__host__ at::Tensor ProjectDRRsBatched_CUDA(const at::Tensor &volume, const at::Tensor &voxelSpacing,
									const at::Tensor &invHMatrices, double sourceDistance, int64_t outputWidth,
									int64_t outputHeight,  const at::Tensor &outputOffset,
									const at::Tensor &detectorSpacing) {
	// volume should be a 3D tensor of floats on the chosen device
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CUDA);
	// voxelSpacing should be a 1D tensor of 3 doubles
	TORCH_CHECK(voxelSpacing.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(voxelSpacing.dtype() == at::kDouble);
	// homographyMatrixInverse should be of size (4, 4), contain doubles and be on the chosen device
	TORCH_CHECK(invHMatrices.sizes().size() > 2);
	TORCH_CHECK(invHMatrices.sizes()[invHMatrices.sizes().size() - 2] == 4);
	TORCH_CHECK(invHMatrices.sizes()[invHMatrices.sizes().size() - 1] == 4);
	// outputOffset should be a 1D tensor of 2 doubles
	TORCH_CHECK(outputOffset.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(outputOffset.dtype() == at::kDouble);
	// detectorSpacing should be a 1D tensor of 2 doubles
	TORCH_CHECK(detectorSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(detectorSpacing.dtype() == at::kDouble);

	const int64_t samplesPerRay = Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Max();

	const Texture3DCUDA inputTexture = Texture3DCUDA::FromTensor(volume, Texture3DCUDA::VectorType::FromTensor(voxelSpacing));

	at::Tensor invHMatricesContiguous =	invHMatrices.to(at::kCUDA, at::kDouble).contiguous();
	const Vec<int64_t, 3> inputSize = Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Flipped();
	const Texture3DCUDA::VectorType volumeDiagonal = inputSize.StaticCast<Texture3DCUDA::FloatType>() * inputTexture.Spacing();
	const Texture3DCUDA::FloatType volumeDiagLength = volumeDiagonal.Length();
	const Texture3DCUDA::FloatType stepSize = volumeDiagLength / static_cast<Texture3DCUDA::FloatType>(samplesPerRay);
	const Vec<double, 2> outputOffsetVec = Vec<double, 2>::FromTensor(outputOffset);
	const Vec<double, 2> detectorSpacingVec = Vec<double, 2>::FromTensor(detectorSpacing);

	const at::IntArrayRef batchSizes = invHMatrices.sizes().slice(0, invHMatrices.sizes().size() - 2);
	long batchCount = 1;
	for (auto n : batchSizes) batchCount *= n;

	at::Tensor flatOutput = torch::zeros(at::IntArrayRef({batchCount * outputWidth * outputHeight}), volume.contiguous().options());
	float *resultFlatPtr = flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_ProjectDRRsBatched_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_ProjectDRRsBatched_CUDA<<<gridSize, blockSize>>>(inputTexture, sourceDistance, volumeDiagLength,
													stepSize, samplesPerRay,
													batchCount, invHMatricesContiguous.data_ptr<double>(),
													inputTexture.MappingWorldToTexCoord(),
													detectorSpacingVec, Vec<int64_t, 2>{outputWidth, outputHeight},
													outputOffsetVec, resultFlatPtr);
	std::vector<int64_t> outputSizesVector{};
	outputSizesVector.reserve(batchSizes.size() + 2);
	for (int k=0; k<batchSizes.size(); ++k) outputSizesVector.push_back(batchSizes[k]);
	outputSizesVector.push_back(outputHeight);
	outputSizesVector.push_back(outputWidth);
	const at::IntArrayRef outputSizes = {outputSizesVector};
	return flatOutput.view(outputSizes);
}

int blockSizeToDynamicSMemSize_ProjectDRR_backward_CUDA(int blockSize) {
	return 16 * blockSize * static_cast<int>(sizeof(double));
}

__global__ void Kernel_ProjectDRR_backward_CUDA(Texture3DCUDA volume, double sourceDistance, double lambdaStart,
                                                double stepSize, int64_t samplesPerRay,
                                                Vec<Vec<double, 4>, 4> homographyMatrixInverse,
                                                Linear<Texture3DCUDA::VectorType> mappingWorldToTexCoord,
                                                Vec<double, 2> detectorSpacing, Vec<int64_t, 2> outputSize,
                                                Vec<double, 2> outputOffset, const float *dLossDDRRFlatPtr,
                                                double *blockSumsArray) {
	extern __shared__ double buffer[];

	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const long bufferStart = threadIdx.x * 16;
	if (threadIndex >= outputSize.X() * outputSize.Y()) {
		for (int k = 0; k < 16; ++k) {
			buffer[bufferStart + k] = 0.0;
		}
		return;
	}

	const uint64_t i = threadIndex % outputSize.X();
	const uint64_t j = threadIndex / outputSize.X();
	const Vec<double, 2> detectorPosition = detectorSpacing * (Vec<uint64_t, 2>{i, j}.StaticCast<double>() - 0.5 *
	                                                           (outputSize - int64_t{1}).StaticCast<double>()) +
	                                        outputOffset;

	Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
	direction /= direction.Length();
	const Vec<double, 4> delta = VecCat(direction * stepSize, 0.0);
	const Vec<double, 4> start = VecCat(Vec<double, 3>{0.0, 0.0, sourceDistance} + lambdaStart * direction, 1.0);

	Vec<Vec<double, 4>, 4> dIntensityDHomographyMatrixInverse = Vec<Vec<double, 4>, 4>::Full(Vec<double, 4>::Full(0.f));
	for (int k = 0; k < samplesPerRay; ++k) {
		const Vec<double, 4> samplePointUntransformed = start + static_cast<double>(k) * delta;
		const Vec<double, 3> samplePoint = MatMul(homographyMatrixInverse, samplePointUntransformed).XYZ();
		const Vec<double, 3> samplePointMapped = mappingWorldToTexCoord(samplePoint);
		dIntensityDHomographyMatrixInverse += VecOuter(
			VecCat(mappingWorldToTexCoord.gradient * volume.DSampleDTexCoord(samplePointMapped).StaticCast<double>(),
			       0.0), samplePointUntransformed);
	}
	const Vec<Vec<double, 4>, 4> dLossDHomographyMatrixInverseThisKernelInstance =
		static_cast<double>(dLossDDRRFlatPtr[threadIndex]) * dIntensityDHomographyMatrixInverse * stepSize;

	for (int k = 0; k < 16; ++k) {
		buffer[bufferStart + k] = dLossDHomographyMatrixInverseThisKernelInstance[k / 4][k % 4];
	}

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff) {
			const long sumWith = bufferStart + cutoff * 16;
			for (int k = 0; k < 16; ++k) {
				buffer[bufferStart + k] += buffer[sumWith + k];
			}
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		for (int k = 0; k < 16; ++k) {
			blockSumsArray[16 * blockIdx.x + k] = buffer[k];
		}
	}
}

__host__ at::Tensor ProjectDRR_backward_CUDA(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                                             const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                             int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                             const at::Tensor &detectorSpacing, const at::Tensor &dLossDDRR) {
	// dLossDDRR should be of size (outputHeight, outputWidth), contain floats and be on the chosen device
	TORCH_CHECK(dLossDDRR.sizes() == at::IntArrayRef({outputHeight, outputWidth}));
	TORCH_CHECK(dLossDDRR.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(dLossDDRR.device().type() == at::DeviceType::CUDA);

	CommonData common = ProjectDRR<Texture3DCUDA>::Common(volume, voxelSpacing, homographyMatrixInverse, sourceDistance,
	                                                      outputOffset, detectorSpacing, at::DeviceType::CUDA);

	const float *dLossDDRRFlatPtr = dLossDDRR.contiguous().data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_ProjectDRR_backward_CUDA,
	                                               &blockSizeToDynamicSMemSize_ProjectDRR_backward_CUDA, 0);
	const size_t bufferSize = blockSizeToDynamicSMemSize_ProjectDRR_backward_CUDA(blockSize);
	const int gridSize = (static_cast<int>(outputHeight * outputWidth) + blockSize - 1) / blockSize;

	const at::Tensor blockSums = torch::zeros(at::IntArrayRef{gridSize, 16},
	                                          torch::TensorOptions{}.dtype(torch::kDouble).device(volume.device()));
	double *blockSumsPtr = blockSums.data_ptr<double>();

	Kernel_ProjectDRR_backward_CUDA<<<gridSize, blockSize, bufferSize>>>(std::move(common.inputTexture), sourceDistance,
	                                                                     common.lambdaStart, common.stepSize,
	                                                                     common.samplesPerRay,
	                                                                     common.homographyMatrixInverse,
	                                                                     common.inputTexture.MappingWorldToTexCoord(),
	                                                                     common.detectorSpacing,
	                                                                     Vec<int64_t, 2>{outputWidth, outputHeight},
	                                                                     common.outputOffset, dLossDDRRFlatPtr,
	                                                                     blockSumsPtr);

	return blockSums.sum({0}).view(at::IntArrayRef{4, 4});
}

} // namespace reg23