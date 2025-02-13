#include <torch/extension.h>

#include "../include/Texture3DCPU.h"
#include "../include/Radon3D.h"

namespace ExtensionTest {

at::Tensor radon3d_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                       const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                       long samplesPerDirection) {
	// volume should be a 3D array of floats on the CPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CPU);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCPU aTexture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float planeSize = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld() + aTexture.
		DepthWorld() * aTexture.DepthWorld());
	const Linear mappingIToOffset = Radon3D<Texture3DCPU>::GetMappingIToOffset(planeSize, samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;
	for (long i = 0; i < numelOut; ++i) {
		const auto indexMappings = Radon3D<Texture3DCPU>::GetIndexMappings(
			aTexture, phiFlat[i].item().toFloat(), thetaFlat[i].item().toFloat(), rFlat[i].item().toFloat(),
			mappingIToOffset);
		resultFlatPtr[i] = scaleFactor * Radon3D<Texture3DCPU>::IntegrateLooped(
			                   aTexture, indexMappings, samplesPerDirection);
	}
	return resultFlat.view(phiValues.sizes());
}

at::Tensor dRadon3dDR_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                          const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                          long samplesPerDirection) {
	// volume should be a 3D array of floats on the CPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CPU);
	// phiValues, thetaValues and rValues should have matching sizes, contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == thetaValues.sizes());
	TORCH_CHECK(thetaValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(thetaValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(thetaValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCPU aTexture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float planeSize = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld() + aTexture.
		DepthWorld() * aTexture.DepthWorld());
	const Linear mappingIToOffset = Radon3D<Texture3DCPU>::GetMappingIToOffset(planeSize, samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;
	for (long i = 0; i < numelOut; ++i) {
		const float phi = phiFlat[i].item().toFloat();
		const float theta = thetaFlat[i].item().toFloat();
		const float r = rFlat[i].item().toFloat();
		const auto indexMappings = Radon3D<Texture3DCPU>::GetIndexMappings(aTexture, phi, theta, r, mappingIToOffset);
		const auto derivativeWRTR = Radon3D<Texture3DCPU>::GetDerivativeWRTR(aTexture, phi, theta, r);
		resultFlatPtr[i] = scaleFactor * Radon3D<Texture3DCPU>::DIntegrateLoopedDMappingParameter(
			                   aTexture, indexMappings, derivativeWRTR, samplesPerDirection);

	}
	return resultFlat.view(phiValues.sizes());
}

} // namespace ExtensionTest