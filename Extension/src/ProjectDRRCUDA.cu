#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"
#include "../include/ProjectDRR.h"

namespace reg23 {

using CommonData = ProjectDRR<Texture3DCUDA>::CommonData;

__global__ void Kernel_ProjectDRR_CUDA(Texture3DCUDA volume, double sourceDistance, double lambdaStart, double stepSize,
                                       Vec<Vec<double, 4>, 4> homographyMatrixInverse,
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
	for (int k = 0; k < 500; ++k) {
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
	                                                      outputWidth, outputHeight, outputOffset, detectorSpacing,
	                                                      at::DeviceType::CUDA);
	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_ProjectDRR_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(common.flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_ProjectDRR_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture), sourceDistance, common.lambdaStart,
	                                                common.stepSize, common.homographyMatrixInverse,
	                                                common.inputTexture.MappingWorldToTexCoord(),
	                                                common.detectorSpacing, Vec<int64_t, 2>{outputWidth, outputHeight},
	                                                common.outputOffset, resultFlatPtr);

	return common.flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

} // namespace reg23