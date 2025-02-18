#include <torch/extension.h>

#include "../include/Texture3DCPU.h"
#include "../include/ResampleSinogram3D.h"

namespace ExtensionTest {

using CommonData = ResampleSinogram3D<Texture3DCPU>::CommonData;

at::Tensor ResampleSinogram3DCPU(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                   const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                   const at::Tensor &phiGrid, const at::Tensor &rGrid) {
	const CommonData common = ResampleSinogram3D<Texture3DCPU>::Common(sinogram3d, sinogramSpacing,
	                                                                   sinogramRangeCentres, projectionMatrix, phiGrid,
	                                                                   rGrid, at::DeviceType::CPU);

	const at::Tensor phiFlat = phiGrid.flatten();
	const at::Tensor rFlat = rGrid.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	const at::Tensor pht = projectionMatrix.t();

	for (int i = 0; i < common.flatOutput.numel(); ++i) {
		const float phi = phiFlat[i].item().toFloat();
		const at::Tensor intermediate = matmul(
			pht, torch::tensor({{cosf(phi), sinf(phi), 0.f, -rFlat[i].item().toFloat()}}).t());
		const float minusDOverSqMagN = -intermediate[3].item().toFloat() / (
			                               intermediate[0].item().toFloat() * intermediate[0].item().toFloat() +
			                               intermediate[1].item().toFloat() * intermediate[1].item().toFloat() +
			                               intermediate[2].item().toFloat() * intermediate[2].item().toFloat());
		const float x = minusDOverSqMagN * intermediate[0].item().toFloat();
		const float y = minusDOverSqMagN * intermediate[1].item().toFloat();
		const float z = minusDOverSqMagN * intermediate[2].item().toFloat();

		Vec<double, 3> rThetaPhi{};
		rThetaPhi.Z() = atan2(y, x);
		const bool over = rThetaPhi.Z() > .5 * M_PI;
		const bool under = rThetaPhi.Z() < -.5 * M_PI;
		if (over) rThetaPhi.Z() -= M_PI;
		else if (under) rThetaPhi.Z() += M_PI;
		const float magXY = x * x + y * y;
		rThetaPhi.Y() = atan2(z, sqrt(magXY));
		rThetaPhi.X() = static_cast<float>((static_cast<int>(!(over || under)) << 1) - 1) * sqrt(magXY + z * z);

		resultFlatPtr[i] = common.inputTexture.Sample(common.mappingRThetaPhiToTexCoord(rThetaPhi));
	}
	return common.flatOutput.view(phiGrid.sizes());
}

} // namespace ExtensionTest