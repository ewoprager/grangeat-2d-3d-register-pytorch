#pragma once

#include "Common.h"

namespace reg23 {

at::Tensor ProjectDRRCuboidMask_CPU(const at::Tensor &volumeSize, const at::Tensor &voxelSpacing,
                                    const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                    int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                    const at::Tensor &detectorSpacing);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::ProjectDRRCuboidMask_CPU that uses CUDA parallelisation.
 */
__host__ at::Tensor ProjectDRRCuboidMask_CUDA(const at::Tensor &volumeSize, const at::Tensor &voxelSpacing,
                                              const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                              int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                              const at::Tensor &detectorSpacing);


template <typename T, std::size_t faceCount> __host__ __device__ T RayConvexPolyhedronDistance(
	const std::array<Vec<T, 3>, faceCount> &facePoints, const std::array<Vec<T, 3>, faceCount> &faceOutUnitNormals,
	const Vec<T, 3> &rayPoint, const Vec<T, 3> &rayUnitDirection) {

	constexpr T epsilon = 1e-8;

	T entryLambda = -std::numeric_limits<T>::infinity();
	T exitLambda = std::numeric_limits<T>::infinity();
	int entryIndex = 0;
	for (int i = 0; i < faceCount; ++i) {
		const T dot = VecDot(faceOutUnitNormals[i], rayUnitDirection);
		const T lambda = VecDot(faceOutUnitNormals[i], facePoints[i] - rayPoint) / dot;
		if (dot < -epsilon && lambda > entryLambda) {
			entryLambda = lambda;
			entryIndex = i;
		} else if (dot > epsilon && lambda < exitLambda) {
			exitLambda = lambda;
		}
	}

	const Vec<T, 3> entryPoint = rayPoint + entryLambda * rayUnitDirection;
	for (int i = 0; i < facePoints.size(); ++i) {
		if (i == entryIndex) continue;
		if (VecDot(entryPoint - facePoints[i], faceOutUnitNormals[i]) > epsilon) return 0.f;
	}

	return std::abs(exitLambda - entryLambda);
}


/**
 * @tparam texture_t Type of the texture object that input data will be converted to for sampling.
 *
 * This struct is used as a namespace for code that is shared between different implementations of `ProjectDRRCuboidMask_...`
 * functions
 */
struct ProjectDRRCuboidMask {

	struct CommonData {
		Vec<Vec<float, 4>, 4> homographyMatrixInverse{};
		Vec<float, 2> outputOffset{};
		Vec<float, 2> detectorSpacing{};
		Vec<float, 3> sourcePositionTransformed{};
		std::array<Vec<float, 3>, 6> cuboidFacePoints{};
		std::array<Vec<float, 3>, 6> cuboidFaceUnitNormals{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &volumeSize, const at::Tensor &voxelSpacing,
	                                  const at::Tensor &homographyMatrixInverse, double sourceDistance,
	                                  int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
	                                  const at::Tensor &detectorSpacing, at::DeviceType device) {
		// volumeSize should be a 1D tensor of 3 longs on the chosen device
		TORCH_CHECK(volumeSize.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(volumeSize.dtype() == at::kLong);
		TORCH_INTERNAL_ASSERT(volumeSize.device().type() == device);
		// voxelSpacing should be a 1D tensor of 3 doubles on the chosen device
		TORCH_CHECK(voxelSpacing.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(voxelSpacing.dtype() == at::kDouble);
		TORCH_INTERNAL_ASSERT(voxelSpacing.device().type() == device);
		// homographyMatrixInverse should be of size (4, 4), contain doubles and be on the chosen device
		TORCH_CHECK(homographyMatrixInverse.sizes() == at::IntArrayRef({4, 4}));
		TORCH_CHECK(homographyMatrixInverse.dtype() == at::kDouble);
		TORCH_INTERNAL_ASSERT(homographyMatrixInverse.device().type() == device);
		// outputOffset should be a 1D tensor of 2 doubles on the chosen device
		TORCH_CHECK(outputOffset.sizes() == at::IntArrayRef{2});
		TORCH_CHECK(outputOffset.dtype() == at::kDouble);
		TORCH_INTERNAL_ASSERT(outputOffset.device().type() == device);
		// detectorSpacing should be a 1D tensor of 2 doubles on the chosen device
		TORCH_CHECK(detectorSpacing.sizes() == at::IntArrayRef{2});
		TORCH_CHECK(detectorSpacing.dtype() == at::kDouble);
		TORCH_INTERNAL_ASSERT(detectorSpacing.device().type() == device);

		CommonData ret{};
		ret.homographyMatrixInverse = Vec<Vec<float, 4>, 4>::FromTensor2D(homographyMatrixInverse);
		const Vec<float, 6> planeSigns = ((Vec<int, 6>::Range() < 3).StaticCast<int>() * 2 - 1).StaticCast<float>();
		ret.cuboidFacePoints = VecOuter(
			0.5f * Vec<float, 3>::FromTensor(voxelSpacing) * Vec<int64_t, 3>::FromTensor(volumeSize).StaticCast<
				float>(), planeSigns);
		ret.cuboidFaceUnitNormals = VecCat(Vec<Vec<float, 3>, 3>::Identity(), -1.f * Vec<Vec<float, 3>, 3>::Identity());

		ret.sourcePositionTransformed = MatMul(ret.homographyMatrixInverse,
		                                       Vec<float, 4>{0.0f, 0.0f, static_cast<float>(sourceDistance), 1.0f}).
			XYZ();
		ret.outputOffset = Vec<float, 2>::FromTensor(outputOffset);
		ret.detectorSpacing = Vec<float, 2>::FromTensor(detectorSpacing);
		ret.flatOutput = torch::zeros(at::IntArrayRef({outputWidth * outputHeight}),
		                              at::TensorOptions{at::Device{device}});
		return ret;
	}
};

} // namespace reg23