/**
 * @file
 * @brief Implementations of DRR projection algorithms
 */

#pragma once

#include "Common.h"

namespace ExtensionTest {

/**
 * @ingroup pytorch_functions
 * @brief
 * @param homographyMatrixInverse Column-major
 */
at::Tensor ProjectDRR_CPU(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                          const at::Tensor &homographyMatrixInverse, double sourceDistance, int64_t outputWidth,
                          int64_t outputHeight, const at::Tensor &outputOffset, const at::Tensor &detectorSpacing);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of ExtensionTest::ProjectDRR_CPU that uses CUDA parallelisation.
 */
__host__ at::Tensor ProjectDRR_CUDA(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                                    const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                    int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                    const at::Tensor &detectorSpacing);

/**
 * @tparam texture_t Type of the texture object that input data will be converted to for sampling.
 *
 * This struct is used as a namespace for code that is shared between different implementations of `ProjectDRR_...`
 * functions
 */
template <typename texture_t> struct ProjectDRR {

	struct CommonData {
		texture_t inputTexture{};
		Vec<Vec<double, 4>, 4> homographyMatrixInverse;
		Vec<double, 2> outputOffset;
		Vec<double, 2> detectorSpacing;
		double lambdaStart;
		double stepSize;
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &volume, const at::Tensor &voxelSpacing,
	                                  const at::Tensor &homographyMatrixInverse, double sourceDistance,
	                                  int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
	                                  const at::Tensor &detectorSpacing, int64_t samplesPerRay, at::DeviceType device) {
		// volume should be a 3D tensor of floats on the chosen device
		TORCH_CHECK(volume.sizes().size() == 3);
		TORCH_CHECK(volume.dtype() == at::kFloat);
		TORCH_INTERNAL_ASSERT(volume.device().type() == device);
		// voxelSpacing should be a 1D tensor of 3 doubles
		TORCH_CHECK(voxelSpacing.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(voxelSpacing.dtype() == at::kDouble);
		// homographyMatrixInverse should be of size (4, 4), contain doubles and be on the chosen device
		TORCH_CHECK(homographyMatrixInverse.sizes() == at::IntArrayRef({4, 4}));
		TORCH_CHECK(homographyMatrixInverse.dtype() == at::kDouble);
		TORCH_INTERNAL_ASSERT(homographyMatrixInverse.device().type() == device);
		// outputOffset should be a 1D tensor of 2 doubles
		TORCH_CHECK(outputOffset.sizes() == at::IntArrayRef{2});
		TORCH_CHECK(outputOffset.dtype() == at::kDouble);
		// detectorSpacing should be a 1D tensor of 2 doubles
		TORCH_CHECK(detectorSpacing.sizes() == at::IntArrayRef{2});
		TORCH_CHECK(detectorSpacing.dtype() == at::kDouble);

		CommonData ret{};
		ret.inputTexture = texture_t::FromTensor(volume, voxelSpacing);
		ret.homographyMatrixInverse = Vec<Vec<double, 4>, 4>::FromTensor2D(homographyMatrixInverse);

		const Vec<int64_t, 3> inputSize = Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Flipped();
		const Vec<double, 3> volumeDiagonal = inputSize.StaticCast<double>() * ret.inputTexture.Spacing();
		const double volumeDiagLength = volumeDiagonal.Length();
		const Vec<double, 3> sourcePosition = {0.0, 0.0, sourceDistance};
		ret.lambdaStart = MatMul(ret.homographyMatrixInverse, VecCat(sourcePosition, 1.0)).XYZ().Length() - 0.5 *
		                  volumeDiagLength;
		ret.stepSize = volumeDiagLength / static_cast<double>(samplesPerRay);
		ret.outputOffset = Vec<double, 2>::FromTensor(detectorSpacing);
		ret.detectorSpacing = Vec<double, 2>::FromTensor(detectorSpacing);
		ret.flatOutput = torch::zeros(at::IntArrayRef({outputWidth * outputHeight}), volume.contiguous().options());
		return ret;
	}
};

} // namespace ExtensionTest