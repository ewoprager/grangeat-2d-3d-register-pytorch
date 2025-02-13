#include <torch/extension.h>

#include "../include/Texture3DCPU.h"

namespace ExtensionTest {

at::Tensor ResampleRadonVolume_cpu(const at::Tensor &sinogram3d, double phiMinS, double phiMaxS, double thetaMinS,
                                   double thetaMaxS, double rMinS, double rMaxS, const at::Tensor &projectionMatrix,
                                   const at::Tensor &phiGrid, const at::Tensor &rGrid) {
	// sinogram3d should be a 3D tensor of floats on the CPU
	TORCH_CHECK(sinogram3d.sizes().size() == 3)
	TORCH_CHECK(sinogram3d.dtype() == at::kFloat)
	TORCH_INTERNAL_ASSERT(sinogram3d.device().type() == at::DeviceType::CPU)
	// projectionMatrix should be of size (4, 4), contain floats and be on the CPU
	TORCH_CHECK(projectionMatrix.sizes() == at::IntArrayRef({4, 4}))
	TORCH_CHECK(projectionMatrix.dtype() == at::kFloat)
	TORCH_INTERNAL_ASSERT(projectionMatrix.device().type() == at::DeviceType::CPU)
	// phiGrid and rGrid should be of the same size, contain floats and be on the CPU
	TORCH_CHECK(phiGrid.sizes() == rGrid.sizes())
	TORCH_CHECK(phiGrid.dtype() == at::kFloat)
	TORCH_CHECK(rGrid.dtype() == at::kFloat)
	TORCH_INTERNAL_ASSERT(phiGrid.device().type() == at::DeviceType::CPU)
	TORCH_INTERNAL_ASSERT(rGrid.device().type() == at::DeviceType::CPU)

	const double phiSpacing = (phiMaxS - phiMinS) / static_cast<float>(sinogram3d.sizes()[0] - 1);
	const double thetaSpacing = (thetaMaxS - thetaMinS) / static_cast<float>(sinogram3d.sizes()[1] - 1);
	const double rSpacing = (rMaxS - rMinS) / static_cast<float>(sinogram3d.sizes()[2] - 1);
	const double centrePhi = .5 * (phiMinS + phiMaxS);
	const double centreTheta = .5 * (thetaMinS + thetaMaxS);
	const double centreR = .5 * (rMinS + rMaxS);

	const at::Tensor sinogramContiguous = sinogram3d.contiguous();
	const float *sinogramPtr = sinogramContiguous.data_ptr<float>();
	const Texture3DCPU sinogramTexture{sinogramPtr, sinogram3d.sizes()[2], sinogram3d.sizes()[1], sinogram3d.sizes()[0],
	                                   rSpacing, thetaSpacing, phiSpacing, centreR, centreTheta, centrePhi};

	const at::Tensor phiFlat = phiGrid.flatten();
	const at::Tensor rFlat = rGrid.flatten();

	const long numelOut = phiGrid.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), sinogramContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const Linear mappingPhiToNormalised{static_cast<float>(-phiMinS / (phiMaxS - phiMinS)),
	                                    static_cast<float>(1. / (phiMaxS - phiMinS))};
	const Linear mappingThetaToNormalised{static_cast<float>(-thetaMinS / (thetaMaxS - thetaMinS)),
	                                      static_cast<float>(1. / (thetaMaxS - thetaMinS))};
	const Linear mappingRToNormalised{static_cast<float>(-rMinS / (rMaxS - rMinS)),
	                                  static_cast<float>(1. / (rMaxS - rMinS))};

	const at::Tensor pht = projectionMatrix.t();

	for (int i = 0; i < numelOut; ++i) {
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

		float phiS = atan2f(y, x);
		const bool over = phiS > .5 * M_PI;
		const bool under = phiS < -.5 * M_PI;
		if (over) phiS -= M_PI;
		else if (under) phiS += M_PI;
		const float magXY = x * x + y * y;
		const float thetaS = atan2f(z, sqrtf(magXY));
		const float rS = static_cast<float>((static_cast<int>(!(over || under)) << 1) - 1) * sqrtf(magXY + z * z);

		resultFlatPtr[i] = sinogramTexture.Sample(mappingRToNormalised(rS), mappingThetaToNormalised(thetaS),
		                                          mappingPhiToNormalised(phiS));
	}
	return resultFlat.view(phiGrid.sizes());
}

} // namespace ExtensionTest