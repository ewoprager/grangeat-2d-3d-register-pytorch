#include <torch/extension.h>

#include "../include/Texture3DCPU.h"

namespace ExtensionTest {

at::Tensor ResampleRadonVolume_cpu(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                   const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                   const at::Tensor &phiGrid, const at::Tensor &rGrid) {
	// sinogram3d should be a 3D tensor of floats on the CPU
	TORCH_CHECK(sinogram3d.sizes().size() == 3)
	TORCH_CHECK(sinogram3d.dtype() == at::kFloat)
	TORCH_INTERNAL_ASSERT(sinogram3d.device().type() == at::DeviceType::CPU)
	// sinogramSpacing should be a 1D tensor of 3 floats or doubles on the CPU
	TORCH_CHECK(sinogramSpacing.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(sinogramSpacing.dtype() == at::kFloat || sinogramSpacing.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(sinogramSpacing.device().type() == at::DeviceType::CPU);
	// sinogramRangeCentres should be a 1D tensor of 3 floats or doubles on the CPU
	TORCH_CHECK(sinogramRangeCentres.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(sinogramRangeCentres.dtype() == at::kFloat || sinogramRangeCentres.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(sinogramRangeCentres.device().type() == at::DeviceType::CPU);
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

	const at::Tensor sinogramContiguous = sinogram3d.contiguous();
	const float *sinogramPtr = sinogramContiguous.data_ptr<float>();
	const Texture3DCPU sinogramTexture{sinogramPtr, Vec<int64_t, 3>::FromIntArrayRef(sinogram3d.sizes()).Flipped(),
	                                   Vec<double, 3>::FromTensor(sinogramSpacing),
	                                   Vec<double, 3>::FromTensor(sinogramRangeCentres)};

	const at::Tensor phiFlat = phiGrid.flatten();
	const at::Tensor rFlat = rGrid.flatten();

	const long numelOut = phiGrid.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), sinogramContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const at::Tensor pht = projectionMatrix.t();
	const Linear<Vec<double, 3> > mappingRThetaPhiToTexCoord = sinogramTexture.MappingWorldToTexCoord();

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

		Vec<double, 3> rThetaPhi{};
		rThetaPhi.Z() = atan2(y, x);
		const bool over = rThetaPhi.Z() > .5 * M_PI;
		const bool under = rThetaPhi.Z() < -.5 * M_PI;
		if (over) rThetaPhi.Z() -= M_PI;
		else if (under) rThetaPhi.Z() += M_PI;
		const float magXY = x * x + y * y;
		rThetaPhi.Y() = atan2(z, sqrt(magXY));
		rThetaPhi.X() = static_cast<float>((static_cast<int>(!(over || under)) << 1) - 1) * sqrt(magXY + z * z);

		resultFlatPtr[i] = sinogramTexture.Sample(mappingRThetaPhiToTexCoord(rThetaPhi));
	}
	return resultFlat.view(phiGrid.sizes());
}

} // namespace ExtensionTest