#pragma once

#include "Common.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Calculate the distances of the intersections between the DRR-generation rays and the domain of the volume data.
 * @param volumeSize a tensor of size (3,): the sizes of the volume data tensor in the order (X, Y, Z) - this is the reverse order of tensor.size()
 * @param voxelSpacing a tensor of size (3,): the spacing in the (X, Y, Z) directions between adjactent voxels in the volume
 * @param homographyMatrixInverse a tensor of size (4, 4): the inverse homography matrix of the volume transformation
 * @param sourceDistance the distance of the X-ray source from the detector array
 * @param outputWidth the width of the detector array
 * @param outputHeight the height of the detector array
 * @param outputOffset a tensor of size (2,) containing `torch.float64`s: The offset in [mm] of the centre of the DRR from the central ray from the X-ray source perpendicularly onto the detector array.
 * @param detectorSpacing a tensor of size (2,): The spacing in [mm] between the columns and rows of the DRR image.
 * @return A tensor of size (outputHeight, outputWidth): The distance each DRR ray travels through the volume.
 */
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


/**
 * @ingroup general_tools
 * @brief Calculate the length of the intersection between a ray and a convex polyhedron
 * @tparam T Real type
 * @tparam faceCount Number of faces of the polyhedron
 * @param facePoints An array of points, one for each polyhedron face, where each point lies on its corresponding face
 * @param faceOutUnitNormals An array of unit vectors, one for each polyhedron face, where each vector is the outward-facing normal vector of its corresponding face
 * @param rayPoint A point that lies on the given ray
 * @param rayUnitDirection A unit vector parallel to the given ray's direction
 * @return The length of the section of the given ray that lies within the given polyhedron. If such a section doesn't
 * exist, 0 is returned.
 *
 * The polyhedron is defined by its faces. Each face is defined with an intersecting point and a unit normal vector.
 * The ray is defined by an intersecting point and a unit direction vector. The ray is considered to be infinite;
 * this function will return the same value if `rayUnitDirection` is multiplied by -1. The number of faces that the
 * polyhedron has must be known at compile time.
 */
template <typename T, std::size_t faceCount> __host__ __device__ T RayConvexPolyhedronDistance(
	const std::array<Vec<T, 3>, faceCount> &facePoints, const std::array<Vec<T, 3>, faceCount> &faceOutUnitNormals,
	const Vec<T, 3> &rayPoint, const Vec<T, 3> &rayUnitDirection) {

	constexpr T epsilon = 1e-8;

	T entryLambda = -std::numeric_limits<T>::infinity();
	T exitLambda = std::numeric_limits<T>::infinity();
	std::size_t entryIndex = 0;
	for (std::size_t i = 0; i < faceCount; ++i) {
		const T dot = VecDot(faceOutUnitNormals[i], rayUnitDirection);
		const T lambda = VecDot(faceOutUnitNormals[i], facePoints[i] - rayPoint) / dot;
		// If dot is 0, the ray is parallel to the face; if dot is negative, increasing lambda moves you along the ray
		// in the 'inward' direction relative to this face, and vice versa if dot is positive.
		if (dot < -epsilon && lambda > entryLambda) {
			// Moving in 'inward' direction, so could be entering the polyhedron through this face
			entryLambda = lambda;
			entryIndex = i;
		} else if (dot > epsilon && lambda < exitLambda) {
			// Moving in 'outward' direction, so could be exiting the polyhedron through this face
			exitLambda = lambda;
		}
	}
	// `entryLambda` is now the largest value for which ray is moving 'into' face, and `entryIndex` is index of
	// corresponding face. `exitLambda` is the smallest value for which ray is moving 'out of' face. This means that,
	// **if the ray intersects with the polyhedron**, it will enter at `entryLambda` and leave at `exitLambda`.

	const Vec<T, 3> entryPoint = rayPoint + entryLambda * rayUnitDirection;
	// Checking that the entry point lies 'within' all the other faces of the polyhedron. If it lies 'outside' of any,
	// it doesn't intersect the polyhedron. Technically, only the faces adjacent to the intersected face need to be
	// checked, but that would require expensive analysis of the face intersections.
	for (std::size_t i = 0; i < faceCount; ++i) {
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