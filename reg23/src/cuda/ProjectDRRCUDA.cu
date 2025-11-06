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
	CommonData common = ProjectDRR<Texture3DCUDA>::Common(volume, voxelSpacing, homographyMatrixInverse, sourceDistance,
	                                                      outputOffset, detectorSpacing, at::DeviceType::CUDA);
	at::Tensor flatOutput = torch::zeros(at::IntArrayRef({outputWidth * outputHeight}), volume.contiguous().options());
	float *resultFlatPtr = flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_ProjectDRR_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_ProjectDRR_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture), sourceDistance, common.lambdaStart,
	                                                common.stepSize, common.samplesPerRay,
	                                                common.homographyMatrixInverse,
	                                                common.inputTexture.MappingWorldToTexCoord(),
	                                                common.detectorSpacing, Vec<int64_t, 2>{outputWidth, outputHeight},
	                                                common.outputOffset, resultFlatPtr);

	return flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
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