#include <torch/extension.h>

#include "../include/Texture3DCPU.h"
#include "../include/ResampleSinogram3D.h"

namespace reg23 {

using CommonData = ResampleSinogram3D<Texture3DCPU>::CommonData;

/**
 * @brief
 *
 *	Note: Assumes that the projection matrix projects onto the x-y plane, and that the radial coordinates (phi, r)
 *	in that plane measure phi right-hand rule about the z-axis from the positive x-direction
 *
 * @param sinogram3d
 * @param sinogramSpacing
 * @param sinogramRangeCentres
 * @param projectionMatrix
 * @param phiValues
 * @param rValues
 * @return
 */
at::Tensor ResampleSinogram3D_CPU(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                  const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                  const at::Tensor &phiValues, const at::Tensor &rValues) {
	const CommonData common = ResampleSinogram3D<Texture3DCPU>::Common(sinogram3d, sinogramSpacing,
	                                                                   sinogramRangeCentres, projectionMatrix,
	                                                                   phiValues, rValues, at::DeviceType::CPU);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	const at::Tensor originProjectionHomogeneous = matmul(projectionMatrix, torch::tensor({{0.f, 0.f, 0.f, 1.f}}).t());
	const Vec<float, 2> originProjection = Vec<float, 2>{originProjectionHomogeneous[0].item().toFloat(),
	                                                     originProjectionHomogeneous[1].item().toFloat()} /
	                                       originProjectionHomogeneous[3].item().toFloat();
	const float squareRadius = .25f * originProjection.Apply<float>(&Square<float>).Sum();

	const Vec<Vec<float, 4>, 4> pht = Vec<Vec<float, 4>, 4>::FromTensor2D(projectionMatrix.t());

	for (int i = 0; i < common.flatOutput.numel(); ++i) {
		const float phi = phiFlat[i].item().toFloat();
		const float r = rFlat[i].item().toFloat();
		const float cp = cosf(phi);
		const float sp = sinf(phi);
		const Vec<float, 4> intermediate = MatMul(pht, Vec<float, 4>{cp, sp, 0.f, -r});
		const Vec<float, 3> intermediate3 = intermediate.XYZ();
		const Vec<float, 3> posCartesian = -intermediate.W() * intermediate3 / intermediate3.Apply<
			                                   float>(&Square<float>).Sum();

		Vec<double, 3> rThetaPhi{};
		rThetaPhi.Z() = atan2(posCartesian.Y(), posCartesian.X());
		const float magXY = posCartesian.X() * posCartesian.X() + posCartesian.Y() * posCartesian.Y();
		rThetaPhi.Y() = atan2(posCartesian.Z(), sqrt(magXY));
		rThetaPhi.X() = sqrt(magXY + posCartesian.Z() * posCartesian.Z());
		rThetaPhi = UnflipSphericalCoordinate(rThetaPhi);

		resultFlatPtr[i] = common.inputTexture.Sample(common.mappingRThetaPhiToTexCoord(rThetaPhi));

		if ((r * Vec<float, 2>{cp, sp} - .5f * originProjection).Apply<float>(&Square<float>).Sum() < squareRadius) {
			resultFlatPtr[i] *= -1.f;
		}
	}
	return common.flatOutput.view(phiValues.sizes());
}

} // namespace reg23