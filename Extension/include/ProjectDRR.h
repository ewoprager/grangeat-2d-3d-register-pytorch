/**
 * @file
 * @brief Implementations of DRR projection functions
 */

#pragma once

#include "Common.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Generate a DRR from the given volume at the given transformation.
 * @param volume a tensor of size (P,Q,R): The CT volume through which to project the DRR
 * @param voxelSpacing a tensor of size (3,): The spacing in [mm] between the volume layers in each cartesian direction
 * @param homographyMatrixInverse a tensor of size (4, 4): The **column-major** matrix representing the homography transformation of the volume.
 * @param sourceDistance The distance in [mm] of the source from the detector array
 * @param outputWidth The width in pixels of the DRR to generate.
 * @param outputHeight The height in pixels of the DRR to generate.
 * @param outputOffset a tensor of size (2,) containing `torch.float64`s: The offset in [mm] of the centre of the DRR from the central ray from the X-ray source perpendicularly onto the detector array.
 * @param detectorSpacing a tensor of size (2,): The spacing in [mm] between the columns and rows of the DRR image.
 * @return a tensor of size (outputHeight, outputWidth): The DRR projection through the given volume at the given transformation.
 *
 * The number of samples taken along each ray through the volume for approximation of each ray integral is equal to the largest dimension of the given volume.
 */
at::Tensor ProjectDRR_CPU(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                          const at::Tensor &homographyMatrixInverse, double sourceDistance, int64_t outputWidth,
                          int64_t outputHeight, const at::Tensor &outputOffset, const at::Tensor &detectorSpacing);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::ProjectDRR_CPU that uses CUDA parallelisation.
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

	static_assert(texture_t::DIMENSIONALITY == 3);

	using IntType = typename texture_t::IntType;
	using FloatType = typename texture_t::FloatType;
	using SizeType = typename texture_t::SizeType;
	using VectorType = typename texture_t::VectorType;
	using AddressModeType = typename texture_t::AddressModeType;

	struct CommonData {
		texture_t inputTexture{};
		Vec<Vec<double, 4>, 4> homographyMatrixInverse{};
		Vec<double, 2> outputOffset{};
		Vec<double, 2> detectorSpacing{};
		double lambdaStart{};
		FloatType stepSize{};
		int64_t samplesPerRay{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &volume, const at::Tensor &voxelSpacing,
	                                  const at::Tensor &homographyMatrixInverse, double sourceDistance,
	                                  int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
	                                  const at::Tensor &detectorSpacing, at::DeviceType device,
	                                  std::optional<int64_t> samplesPerRay = std::nullopt) {
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

		const int64_t samplesPerRayValue = samplesPerRay.value_or(
			Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Max());

		CommonData ret{};
		ret.inputTexture = texture_t::FromTensor(volume, VectorType::FromTensor(voxelSpacing));
		ret.homographyMatrixInverse = Vec<Vec<double, 4>, 4>::FromTensor2D(homographyMatrixInverse);

		const Vec<int64_t, 3> inputSize = Vec<int64_t, 3>::FromIntArrayRef(volume.sizes()).Flipped();
		const VectorType volumeDiagonal = inputSize.StaticCast<FloatType>() * ret.inputTexture.Spacing();
		const FloatType volumeDiagLength = volumeDiagonal.Length();
		const VectorType sourcePosition = {0.0, 0.0, sourceDistance};
		ret.lambdaStart = MatMul(ret.homographyMatrixInverse, VecCat(sourcePosition, 1.0)).XYZ().Length() - 0.5 *
		                  volumeDiagLength;
		ret.stepSize = volumeDiagLength / static_cast<FloatType>(samplesPerRayValue);
		ret.samplesPerRay = samplesPerRayValue;
		ret.outputOffset = Vec<double, 2>::FromTensor(outputOffset);
		ret.detectorSpacing = Vec<double, 2>::FromTensor(detectorSpacing);
		ret.flatOutput = torch::zeros(at::IntArrayRef({outputWidth * outputHeight}), volume.contiguous().options());
		return ret;
	}
};

/**
 *
 * @param volume
 * @param voxelSpacing
 * @param homographyMatrixInverse a tensor of size (4, 4): The **column-major** matrix representing the homography transformation of the volume.
 * @param sourceDistance
 * @param outputWidth
 * @param outputHeight
 * @param outputOffset
 * @param detectorSpacing
 * @return tensor of size (outputHeight, outputWidth, 4, 4): For every pixel in the DRR, a 4x4 **column-major** matrix equal to the derivative of its intensity value w.r.t. the inverse homography matrix
 *
 * Note: This function does not take the original value of the inverse homography matrix as a parameter, as the
 */
at::Tensor DProjectDRRDHMI_CPU(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                               const at::Tensor &homographyMatrixInverse, double sourceDistance, int64_t outputWidth,
                               int64_t outputHeight, const at::Tensor &outputOffset, const at::Tensor &detectorSpacing);

} // namespace reg23