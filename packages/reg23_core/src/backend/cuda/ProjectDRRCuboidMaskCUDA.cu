#include <torch/extension.h>

#include <reg23_core/ProjectDRRCuboidMaskCPU.h>

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
	if (outputSize.X() == 0) return;
	const uint64_t i = threadIndex % outputSize.X();
	const uint64_t j = threadIndex / outputSize.X();
	const Vec<float, 2> detectorPosition = detectorSpacing * (Vec<uint64_t, 2>{i, j}.StaticCast<float>() -
															  0.5f * (outputSize - int64_t{1}).StaticCast<float>()) +
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
	const CommonData common =
		ProjectDRRCuboidMask::Common(volumeSize, voxelSpacing, homographyMatrixInverse, sourceDistance, outputWidth,
									 outputHeight, outputOffset, detectorSpacing, at::DeviceType::CUDA);
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

__global__ void Kernel_ProjectDRRCuboidMaskBatched_CUDA(double sourceDistance, Cuboid cuboidIn, Cuboid cuboidAbove,
												 Cuboid cuboidBelow, int64_t batchCount, double *invHMatrices,
												 Vec<double, 2> detectorSpacing, Vec<int64_t, 2> outputSize,
												 Vec<double, 2> outputOffset, float *arrayOut) {
	const uint64_t outputNumel = outputSize.X() * outputSize.Y();
	const int64_t threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= outputNumel * batchCount) return;
	if (outputSize.X() == 0) return;
	const uint64_t batchIndex = threadIndex / outputNumel;
	const uint64_t pixelIndex = threadIndex % outputNumel;
	const uint64_t i = pixelIndex % outputSize.X();
	const uint64_t j = pixelIndex / outputSize.X();
	const Vec<double, 2> detectorPosition = detectorSpacing * (Vec<uint64_t, 2>{i, j}.StaticCast<double>() -
															  0.5f * (outputSize - int64_t{1}).StaticCast<double>()) +
															  	outputOffset;
	Vec<Vec<double, 4>, 4> homographyMatrixInverse{};
	for (int k = 0; k < 16; ++k)
		homographyMatrixInverse[k / 4][k % 4] = invHMatrices[16 * batchIndex + k]; // ToDo: Transpose??
	Vec<float, 3> sourcePositionTransformed =
		MatMul(homographyMatrixInverse, Vec<double, 4>{0.0, 0.0, sourceDistance, 1.0})
			.XYZ().StaticCast<float>();
	Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
	direction /= direction.Length();
	Vec<float, 3> directionF = MatMul(homographyMatrixInverse, VecCat(direction, 0.)).XYZ().StaticCast<float>();

	const float distanceAboveBelow = //
		RayConvexPolyhedronDistance(cuboidAbove.facePoints, cuboidAbove.faceOutUnitNormals, sourcePositionTransformed,
									directionF) + //
		RayConvexPolyhedronDistance(cuboidBelow.facePoints, cuboidBelow.faceOutUnitNormals, sourcePositionTransformed,
									directionF);
	const float distanceIn = //
		RayConvexPolyhedronDistance(cuboidIn.facePoints, cuboidIn.faceOutUnitNormals, sourcePositionTransformed,
									directionF);

	if (const float denominator = distanceIn + distanceAboveBelow; denominator > 1e-8) {
		arrayOut[threadIndex] = distanceIn / denominator;
	} else {
		arrayOut[threadIndex] = 1.f;
	}
}

__host__ at::Tensor ProjectDRRCuboidMaskBatched_CUDA(const at::Tensor &volumeSize, const at::Tensor &voxelSpacing,
											  const at::Tensor &invHMatrices, double sourceDistance,
											  int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
											  const at::Tensor &detectorSpacing) {
	// volumeSize should be a 1D tensor of 3 longs on the chosen device
	TORCH_CHECK(volumeSize.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(volumeSize.dtype() == at::kLong);
	TORCH_INTERNAL_ASSERT(volumeSize.device().type() == at::DeviceType::CUDA);
	// voxelSpacing should be a 1D tensor of 3 doubles on the chosen device
	TORCH_CHECK(voxelSpacing.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(voxelSpacing.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(voxelSpacing.device().type() == at::DeviceType::CUDA);
	// invHMatrices should be of size (..., 4, 4)
	TORCH_CHECK(invHMatrices.sizes().size() > 2);
	TORCH_CHECK(invHMatrices.sizes()[invHMatrices.sizes().size() - 2] == 4);
	TORCH_CHECK(invHMatrices.sizes()[invHMatrices.sizes().size() - 1] == 4);
	// outputOffset should be a 1D tensor of 2 doubles on the chosen device
	TORCH_CHECK(outputOffset.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(outputOffset.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(outputOffset.device().type() == at::DeviceType::CUDA);
	// detectorSpacing should be a 1D tensor of 2 doubles on the chosen device
	TORCH_CHECK(detectorSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(detectorSpacing.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(detectorSpacing.device().type() == at::DeviceType::CUDA);

	at::Tensor invHMatricesContiguous = invHMatrices.to(at::kCUDA, at::kDouble).contiguous();
	const Vec<float, 6> planeSigns = Vec<int, 6>{1, 1, 1, -1, -1, -1}.StaticCast<float>();
	const Vec<float, 3> cuboidHalfSize = 0.5f * Vec<float, 3>::FromTensor(voxelSpacing) *
										 Vec<int64_t, 3>::FromTensor(volumeSize).StaticCast<float>();
	const Cuboid cuboidIn = {
		VecOuter(cuboidHalfSize, planeSigns),
		VecCat(Vec<Vec<float, 3>, 3>::Identity(), -1.f * Vec<Vec<float, 3>, 3>::Identity())
	};
	const Vec<float, 3> aboveBelowHalfSize = Vec<float, 3>{1.f, 1.f, 4.f} * cuboidHalfSize;
	Cuboid cuboidAbove = {
		VecOuter(aboveBelowHalfSize, planeSigns),
		cuboidIn.faceOutUnitNormals
	};
	Cuboid cuboidBelow = {
		cuboidAbove.facePoints,
		cuboidIn.faceOutUnitNormals
	};
	const float zSum = aboveBelowHalfSize.Z() + cuboidHalfSize.Z();
	for (Vec<float, 3> &v : cuboidAbove.facePoints)
		v.Z() += zSum;
	for (Vec<float, 3> &v : cuboidBelow.facePoints)
		v.Z() -= zSum;

	const Vec<double, 2> vOutputOffset = Vec<double, 2>::FromTensor(outputOffset);
	const Vec<double, 2> vDetectorSpacing = Vec<double, 2>::FromTensor(detectorSpacing);

	const at::IntArrayRef batchSizes = invHMatrices.sizes().slice(0, invHMatrices.sizes().size() - 2);
	long batchCount = 1;
	for (auto n : batchSizes)
		batchCount *= n;

	at::Tensor flatOutput =
		torch::zeros(at::IntArrayRef({outputWidth * outputHeight}), at::TensorOptions{at::Device{at::DeviceType::CUDA}});
	float *resultFlatPtr = flatOutput.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_ProjectDRRCuboidMaskBatched_CUDA, 0, 0);
	const int gridSize = (static_cast<int>(flatOutput.numel()) + blockSize - 1) / blockSize;
	Kernel_ProjectDRRCuboidMaskBatched_CUDA<<<gridSize, blockSize>>>( //
		static_cast<float>(sourceDistance), cuboidIn, cuboidAbove, cuboidBelow,
		batchCount, invHMatricesContiguous.data_ptr<double>(), vDetectorSpacing,
		Vec<int64_t, 2>{outputWidth, outputHeight}, vOutputOffset, resultFlatPtr);
	std::vector<int64_t> outputSizesVector{};
	outputSizesVector.reserve(batchSizes.size() + 2);
	for (int k = 0; k < batchSizes.size(); ++k)
		outputSizesVector.push_back(batchSizes[k]);
	outputSizesVector.push_back(outputHeight);
	outputSizesVector.push_back(outputWidth);
	const at::IntArrayRef outputSizes = {outputSizesVector};
	return flatOutput.view(outputSizes);
}

} // namespace reg23