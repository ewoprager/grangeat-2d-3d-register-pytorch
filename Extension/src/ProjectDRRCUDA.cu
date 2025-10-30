#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"
#include "../include/ProjectDRR.h"

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
	return 16 * blockSize * static_cast<int>(sizeof(float));
}

__global__ void Kernel_ProjectDRR_backward_CUDA(Texture3DCUDA volume, double sourceDistance, double lambdaStart,
                                                double stepSize, int64_t samplesPerRay,
                                                Vec<Vec<double, 4>, 4> homographyMatrixInverse,
                                                Linear<Texture3DCUDA::VectorType> mappingWorldToTexCoord,
                                                Vec<double, 2> detectorSpacing, Vec<int64_t, 2> outputSize,
                                                Vec<double, 2> outputOffset, const float *dLossDDRRFlatPtr,
                                                float *blockSumsArray) {
	extern __shared__ float buffer[];

	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const long index = threadIdx.x * 16;
	if (threadIndex >= outputSize.X() * outputSize.Y()) {
		buffer[index] = 0.f;
		buffer[index + 1] = 0.f;
		buffer[index + 2] = 0.f;
		buffer[index + 3] = 0.f;
		buffer[index + 4] = 0.f;
		buffer[index + 5] = 0.f;
		buffer[index + 6] = 0.f;
		buffer[index + 7] = 0.f;
		buffer[index + 8] = 0.f;
		buffer[index + 9] = 0.f;
		buffer[index + 10] = 0.f;
		buffer[index + 11] = 0.f;
		buffer[index + 12] = 0.f;
		buffer[index + 13] = 0.f;
		buffer[index + 14] = 0.f;
		buffer[index + 15] = 0.f;
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

	buffer[index] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[0][0]);
	buffer[index + 1] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[0][1]);
	buffer[index + 2] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[0][2]);
	buffer[index + 3] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[0][3]);
	buffer[index + 4] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[1][0]);
	buffer[index + 5] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[1][1]);
	buffer[index + 6] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[1][2]);
	buffer[index + 7] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[1][3]);
	buffer[index + 8] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[2][0]);
	buffer[index + 9] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[2][1]);
	buffer[index + 10] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[2][2]);
	buffer[index + 11] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[2][3]);
	buffer[index + 12] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[3][0]);
	buffer[index + 13] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[3][1]);
	buffer[index + 14] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[3][2]);
	buffer[index + 15] = static_cast<float>(dLossDHomographyMatrixInverseThisKernelInstance[3][3]);

	for (long cutoff = blockDim.x / 2; cutoff > 0; cutoff /= 2) {
		if (threadIdx.x < cutoff) {
			const long sumWith = index + cutoff * 16;
			buffer[index] += buffer[sumWith];
			buffer[index + 1] += buffer[sumWith + 1];
			buffer[index + 2] += buffer[sumWith + 2];
			buffer[index + 3] += buffer[sumWith + 3];
			buffer[index + 4] += buffer[sumWith + 4];
			buffer[index + 5] += buffer[sumWith + 5];
			buffer[index + 6] += buffer[sumWith + 6];
			buffer[index + 7] += buffer[sumWith + 7];
			buffer[index + 8] += buffer[sumWith + 8];
			buffer[index + 9] += buffer[sumWith + 9];
			buffer[index + 10] += buffer[sumWith + 10];
			buffer[index + 11] += buffer[sumWith + 11];
			buffer[index + 12] += buffer[sumWith + 12];
			buffer[index + 13] += buffer[sumWith + 13];
			buffer[index + 14] += buffer[sumWith + 14];
			buffer[index + 15] += buffer[sumWith + 15];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		blockSumsArray[16 * blockIdx.x] = buffer[0];
		blockSumsArray[16 * blockIdx.x + 1] = buffer[1];
		blockSumsArray[16 * blockIdx.x + 2] = buffer[2];
		blockSumsArray[16 * blockIdx.x + 3] = buffer[3];
		blockSumsArray[16 * blockIdx.x + 4] = buffer[4];
		blockSumsArray[16 * blockIdx.x + 5] = buffer[5];
		blockSumsArray[16 * blockIdx.x + 6] = buffer[6];
		blockSumsArray[16 * blockIdx.x + 7] = buffer[7];
		blockSumsArray[16 * blockIdx.x + 8] = buffer[8];
		blockSumsArray[16 * blockIdx.x + 9] = buffer[9];
		blockSumsArray[16 * blockIdx.x + 10] = buffer[10];
		blockSumsArray[16 * blockIdx.x + 11] = buffer[11];
		blockSumsArray[16 * blockIdx.x + 12] = buffer[12];
		blockSumsArray[16 * blockIdx.x + 13] = buffer[13];
		blockSumsArray[16 * blockIdx.x + 14] = buffer[14];
		blockSumsArray[16 * blockIdx.x + 15] = buffer[15];
	}
}

__host__ at::Tensor ProjectDRR_backward_CUDA(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                                             const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                             int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                             const at::Tensor &detectorSpacing, const at::Tensor &dLossDDRR) {
	CommonData common = ProjectDRR<Texture3DCUDA>::Common(volume, voxelSpacing, homographyMatrixInverse, sourceDistance,
	                                                      outputOffset, detectorSpacing, at::DeviceType::CUDA);

	const float *dLossDDRRFlatPtr = dLossDDRR.contiguous().data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, &Kernel_ProjectDRR_backward_CUDA,
	                                               &blockSizeToDynamicSMemSize_ProjectDRR_backward_CUDA, 0);
	const size_t bufferSize = blockSizeToDynamicSMemSize_ProjectDRR_backward_CUDA(blockSize);
	const int gridSize = (static_cast<int>(outputHeight * outputWidth) + blockSize - 1) / blockSize;

	const at::Tensor blockSums = torch::zeros(at::IntArrayRef{gridSize, 16},
	                                          torch::TensorOptions{}.dtype(torch::kFloat).device(volume.device()));
	float *blockSumsPtr = blockSums.data_ptr<float>();

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