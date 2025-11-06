#include <torch/extension.h>

#include <reg23/ProjectDRRCuboidMaskCPU.h>

namespace reg23 {

using CommonData = ProjectDRRCuboidMask::CommonData;
using Cuboid = ProjectDRRCuboidMask::Cuboid;


__global__ void Kernel_ProjectDRRCuboidMask_CUDA(float sourceDistance, Cuboid cuboidIn, Cuboid cuboidAbove,
                                                 Cuboid cuboidBelow, Vec<Vec<float, 4>, 4> homographyMatrixInverse,
                                                 Vec<float, 3> sourcePositionTransformed, Vec<float, 2> detectorSpacing,
                                                 Vec<int64_t, 2> outputSize, Vec<float, 2> outputOffset,
                                                 float *arrayOut) {
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= outputSize.X() * outputSize.Y()) return;
	const uint64_t i = threadIndex % outputSize.X();
	const uint64_t j = threadIndex / outputSize.X();
	const Vec<float, 2> detectorPosition = detectorSpacing * (Vec<uint64_t, 2>{i, j}.StaticCast<float>() - 0.5f *
	                                                          (outputSize - int64_t{1}).StaticCast<float>()) +
	                                       outputOffset;
	Vec<float, 3> direction = VecCat(detectorPosition, -sourceDistance);
	direction /= direction.Length();
	direction = MatMul(homographyMatrixInverse, VecCat(direction, 0.f)).XYZ();

	const float distanceAboveBelow = //
		RayConvexPolyhedronDistance(cuboidAbove.facePoints, cuboidAbove.faceOutUnitNormals, sourcePositionTransformed,
		                            direction) + //
		RayConvexPolyhedronDistance(cuboidBelow.facePoints, cuboidBelow.faceOutUnitNormals, sourcePositionTransformed,
		                            direction);
	const float distanceIn = //
		RayConvexPolyhedronDistance(cuboidIn.facePoints, cuboidIn.faceOutUnitNormals, sourcePositionTransformed,
		                            direction);

	if (const float denominator = distanceIn + distanceAboveBelow; denominator > 1e-8f) {
		arrayOut[threadIndex] = distanceIn / denominator;
	} else {
		arrayOut[threadIndex] = 1.f;
	}
}

__host__ at::Tensor ProjectDRRCuboidMask_CUDA(const at::Tensor &volumeSize, const at::Tensor &voxelSpacing,
                                              const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                              int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                              const at::Tensor &detectorSpacing) {
	const CommonData common = ProjectDRRCuboidMask::Common(volumeSize, voxelSpacing, homographyMatrixInverse,
	                                                       sourceDistance, outputWidth, outputHeight, outputOffset,
	                                                       detectorSpacing, at::DeviceType::CUDA);
	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_ProjectDRRCuboidMask_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(common.flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_ProjectDRRCuboidMask_CUDA<<<gridSize, blockSize>>>( //
		static_cast<float>(sourceDistance), common.cuboidIn, common.cuboidAbove, common.cuboidBelow,
		common.homographyMatrixInverse, common.sourcePositionTransformed, common.detectorSpacing,
		Vec<int64_t, 2>{outputWidth, outputHeight}, common.outputOffset, resultFlatPtr);

	return common.flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

} // namespace reg23