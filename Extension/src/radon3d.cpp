#include <torch/extension.h>

#include "../include/Texture3DCPU.h"
#include "../include/Radon3D.h"

namespace ExtensionTest {


at::Tensor radon3d_cpu(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                       const at::Tensor &thetaValues, const at::Tensor &rValues, long samplesPerDirection) {
	// volume should be a 3D array of floats on the CPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CPU);
	// volumeSpacing should be a 1D tensor of 3 floats or doubles on the CPU
	TORCH_CHECK(volumeSpacing.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(volumeSpacing.dtype() == at::kFloat || volumeSpacing.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(volumeSpacing.device().type() == at::DeviceType::CPU);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const Texture3DCPU texture = Texture3DCPU::FromTensor(volume, volumeSpacing);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), volume.contiguous().options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const double planeSize = sqrtf(texture.SizeWorld().Apply<double>(&Square<double>).Sum());
	const Linear<Vec<double, 3> > mappingIToOffset = Radon3D<Texture3DCPU>::GetMappingIToOffset(
		planeSize, samplesPerDirection);
	const double rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const double scaleFactor = rootScaleFactor * rootScaleFactor;
	for (long i = 0; i < numelOut; ++i) {
		const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCPU>::GetMappingIndexToTexCoord(
			texture, phiFlat[i].item().toFloat(), thetaFlat[i].item().toFloat(), rFlat[i].item().toFloat(),
			mappingIToOffset);
		resultFlatPtr[i] = scaleFactor * Radon3D<Texture3DCPU>::IntegrateLooped(
			                   texture, mappingIndexToTexCoord, samplesPerDirection);
	}
	return resultFlat.view(phiValues.sizes());
}

at::Tensor dRadon3dDR_cpu(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                          const at::Tensor &thetaValues, const at::Tensor &rValues, long samplesPerDirection) {
	// volume should be a 3D array of floats on the CPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CPU);
	// volumeSpacing should be a 1D tensor of 3 floats or doubles on the CPU
	TORCH_CHECK(volumeSpacing.sizes() == at::IntArrayRef{3});
	TORCH_CHECK(volumeSpacing.dtype() == at::kFloat || volumeSpacing.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(volumeSpacing.device().type() == at::DeviceType::CPU);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const Texture3DCPU texture = Texture3DCPU::FromTensor(volume, volumeSpacing);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), volume.contiguous().options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const double planeSize = sqrtf(texture.SizeWorld().Apply<double>(&Square<double>).Sum());
	const Linear<Vec<double, 3> > mappingIToOffset = Radon3D<Texture3DCPU>::GetMappingIToOffset(
		planeSize, samplesPerDirection);
	const double rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const double scaleFactor = rootScaleFactor * rootScaleFactor;
	for (long i = 0; i < numelOut; ++i) {
		const double phi = phiFlat[i].item().toFloat();
		const double theta = thetaFlat[i].item().toFloat();
		const double r = rFlat[i].item().toFloat();
		const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCPU>::GetMappingIndexToTexCoord(
			texture, phi, theta, r, mappingIToOffset);
		const Vec<double, 3> dTexCoordDR = Radon3D<Texture3DCPU>::GetDTexCoordDR(texture, phi, theta, r);
		resultFlatPtr[i] = scaleFactor * Radon3D<Texture3DCPU>::DIntegrateLoopedDMappingParameter(
			                   texture, mappingIndexToTexCoord, dTexCoordDR, samplesPerDirection);

	}
	return resultFlat.view(phiValues.sizes());
}


} // namespace ExtensionTest