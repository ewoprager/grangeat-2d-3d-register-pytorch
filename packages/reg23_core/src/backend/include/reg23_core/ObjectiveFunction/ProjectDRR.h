#pragma once

#include "reg23_core/CUDATexture.h"

#include <reg23_core/Global.h>
#include <reg23_core/Texture3DCUDA.h>
#include <reg23_core/Vec.h>

#include <reg23_core/ProjectDRRCuboidMaskCPU.h>

namespace reg23 {

using Cuboid = ProjectDRRCuboidMask::Cuboid;

__device__ inline Vec<double, 2> EvaluateDetectorPosition(uint64_t i, uint64_t j, const Vec<double, 2> &detectorSpacing,
														  const Vec<int64_t, 2> &outputSize,
														  const Vec<double, 2> &outputOffset) {
	return detectorSpacing *
			   (Vec<uint64_t, 2>{i, j}.StaticCast<double>() - 0.5f * (outputSize - int64_t{1}).StaticCast<double>()) +
		   outputOffset;
}

/**
 * Constant used globally for DRR projection
 */
struct DRRParams {
	double stepSize;
	double volumeDiagLength;
	int64_t samplesPerRay;

	__host__ static DRRParams Evaluate(const Texture3DCUDA &texture) {
		const Texture3DCUDA::SizeType textureSize = texture.Size();
		const Texture3DCUDA::IntType samplesPerRay = textureSize.Max();
		const Texture3DCUDA::FloatType volumeDiagLength =
			(textureSize.StaticCast<Texture3DCUDA::FloatType>() * texture.Spacing()).Length();
		return {volumeDiagLength / static_cast<Texture3DCUDA::FloatType>(samplesPerRay), volumeDiagLength,
				samplesPerRay};
	}
};

/**
 * Constants used globally for cuboid masking
 */
struct MaskGeometry {
	Cuboid cuboidIn{};
	Cuboid cuboidAbove{};
	Cuboid cuboidBelow{};

	__host__ static MaskGeometry Evaluate(const Texture3DCUDA &texture) {
		const Vec<float, 6> planeSigns = Vec<int, 6>{1, 1, 1, -1, -1, -1}.StaticCast<float>();
		const Vec<float, 3> cuboidHalfSize = 0.5f * texture.Spacing() * texture.Size().StaticCast<float>();
		const Cuboid cuboidIn = {VecOuter(cuboidHalfSize, planeSigns),
								 VecCat(Vec<Vec<float, 3>, 3>::Identity(), -1.f * Vec<Vec<float, 3>, 3>::Identity())};
		const Vec<float, 3> aboveBelowHalfSize = Vec<float, 3>{1.f, 1.f, 4.f} * cuboidHalfSize;
		Cuboid cuboidAbove = {VecOuter(aboveBelowHalfSize, planeSigns), cuboidIn.faceOutUnitNormals};
		Cuboid cuboidBelow = {cuboidAbove.facePoints, cuboidIn.faceOutUnitNormals};
		const float zSum = aboveBelowHalfSize.Z() + cuboidHalfSize.Z();
		for (Vec<float, 3> &v : cuboidAbove.facePoints)
			v.Z() += zSum;
		for (Vec<float, 3> &v : cuboidBelow.facePoints)
			v.Z() -= zSum;
		return {cuboidIn, cuboidAbove, cuboidBelow};
	}
};

__device__ float DRRRay(const Texture3DCUDA &volume, const DRRParams &params,
						const Vec<Vec<double, 4>, 4> &homographyMatrixInverse, const Vec<double, 2> &detectorPosition,
						double sourceDistance) {
	Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
	direction /= direction.Length();
	Vec<double, 3> delta = direction * params.stepSize;
	delta = MatMul(homographyMatrixInverse, VecCat(delta, 0.0)).XYZ();
	const Texture3DCUDA::VectorType sourcePosition = {0.0, 0.0, sourceDistance};
	const float lambdaStart =
		MatMul(homographyMatrixInverse, VecCat(sourcePosition, 1.0)).XYZ().Length() - 0.5 * params.volumeDiagLength;
	Vec<double, 3> start = Vec<double, 3>{0.0, 0.0, sourceDistance} + lambdaStart * direction;
	start = MatMul(homographyMatrixInverse, VecCat(start, 1.0)).XYZ();

	const Linear<Texture3DCUDA::VectorType> mappingWorldToTexCoord = volume.MappingWorldToTexCoord();

	Vec<double, 3> samplePoint = start;
	float sum = 0.f;
	for (int k = 0; k < params.samplesPerRay; ++k) {
		sum += volume.Sample(mappingWorldToTexCoord(samplePoint));
		samplePoint += delta;
	}
	return static_cast<float>(params.stepSize) * sum;
}

__device__ float DRRCuboidMaskRay(const Vec<double, 2> &detectorPosition, double sourceDistance,
								  const MaskGeometry &maskGeometry,
								  const Vec<Vec<double, 4>, 4> &homographyMatrixInverse) {
	Vec<float, 3> sourcePositionTransformed =
		MatMul(homographyMatrixInverse, Vec<double, 4>{0.0, 0.0, sourceDistance, 1.0}).XYZ().StaticCast<float>();
	Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
	direction /= direction.Length();
	Vec<float, 3> directionF = MatMul(homographyMatrixInverse, VecCat(direction, 0.)).XYZ().StaticCast<float>();

	const float distanceAboveBelow = //
		RayConvexPolyhedronDistance(maskGeometry.cuboidAbove.facePoints, maskGeometry.cuboidAbove.faceOutUnitNormals,
									sourcePositionTransformed,
									directionF) + //
		RayConvexPolyhedronDistance(maskGeometry.cuboidBelow.facePoints, maskGeometry.cuboidBelow.faceOutUnitNormals,
									sourcePositionTransformed, directionF);
	const float distanceIn = //
		RayConvexPolyhedronDistance(maskGeometry.cuboidIn.facePoints, maskGeometry.cuboidIn.faceOutUnitNormals,
									sourcePositionTransformed, directionF);

	if (const float denominator = distanceIn + distanceAboveBelow; denominator > 1e-8) {
		return distanceIn / denominator;
	} else {
		return 1.f;
	}
}

} // namespace reg23