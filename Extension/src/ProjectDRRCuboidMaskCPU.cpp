#include <torch/extension.h>

#include "../include/ProjectDRRCuboidMaskCPU.h"

namespace reg23 {

using CommonData = ProjectDRRCuboidMask::CommonData;

at::Tensor ProjectDRRCuboidMask_CPU(const at::Tensor &volumeSize, const at::Tensor &voxelSpacing,
                                    const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                    int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                    const at::Tensor &detectorSpacing) {
	const CommonData common = ProjectDRRCuboidMask::Common(volumeSize, voxelSpacing, homographyMatrixInverse,
	                                                       sourceDistance, outputWidth, outputHeight, outputOffset,
	                                                       detectorSpacing, at::DeviceType::CPU);
	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	for (int j = 0; j < outputHeight; ++j) {
		for (int i = 0; i < outputWidth; ++i) {
			const Vec<float, 2> detectorPosition = common.detectorSpacing * Vec<float, 2>{
				                                       static_cast<float>(i) - 0.5f * static_cast<float>(
					                                       outputWidth - 1),
				                                       static_cast<float>(j) - 0.5f * static_cast<float>(
					                                       outputHeight - 1)} + common.outputOffset;
			Vec<float, 3> direction = VecCat(detectorPosition, -static_cast<float>(sourceDistance));
			direction /= direction.Length();
			direction = MatMul(common.homographyMatrixInverse, VecCat(direction, 0.f)).XYZ();

			const float distanceAboveBelow = //
				RayConvexPolyhedronDistance(common.cuboidAbove.facePoints, common.cuboidAbove.faceOutUnitNormals,
				                            common.sourcePositionTransformed, direction) + //
				RayConvexPolyhedronDistance(common.cuboidBelow.facePoints, common.cuboidBelow.faceOutUnitNormals,
				                            common.sourcePositionTransformed, direction);
			const float distanceIn = //
				RayConvexPolyhedronDistance(common.cuboidIn.facePoints, common.cuboidIn.faceOutUnitNormals,
				                            common.sourcePositionTransformed, direction);
			if (const float denominator = distanceIn + distanceAboveBelow; denominator > 1e-8f) {
				resultFlatPtr[i + j * outputWidth] = distanceIn / denominator;
			} else {
				resultFlatPtr[i + j * outputWidth] = 1.f;
			}
		}
	}

	return common.flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

} // namespace reg23