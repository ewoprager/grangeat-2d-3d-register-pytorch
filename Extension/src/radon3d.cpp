#include <torch/extension.h>

#include "../include/Texture3DCPU.h"

namespace ExtensionTest {

at::Tensor radon3d_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                       const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                       long samplesPerDirection) {
	// volume should be a 3D array of floats on the CPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CPU);
	// phiValues should be a 1D array of floats on the CPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	// thetaValues should be a 1D array of floats on the CPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	// rValues should be a 1D array of floats on the CPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCPU aTexture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	const long depthOut = phiValues.sizes()[0];
	const long heightOut = thetaValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float planeSize = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld() + aTexture.
		DepthWorld() * aTexture.DepthWorld());
	const Linear mappingIToOffset = Radon3D<Texture3DCPU>::GetMappingIToOffset(planeSize, samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;
	for (long layer = 0; layer < depthOut; ++layer) {
		for (long row = 0; row < heightOut; ++row) {
			for (long col = 0; col < widthOut; ++col) {
				const auto indexMappings = Radon3D<Texture3DCPU>::GetIndexMappings(
					aTexture, phiValues[layer].item().toFloat(), thetaValues[row].item().toFloat(),
					rValues[col].item().toFloat(), mappingIToOffset);
				resultPtr[layer * widthOut * heightOut + row * widthOut + col] =
					scaleFactor * Radon3D<Texture3DCPU>::IntegrateLooped(aTexture, indexMappings, samplesPerDirection);
			}
		}
	}

	return result;
}

at::Tensor dRadon3dDR_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing,
                          const at::Tensor &phiValues, const at::Tensor &thetaValues, const at::Tensor &rValues,
                          long samplesPerDirection) {
	// volume should be a 3D array of floats on the CPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CPU);
	// phiValues should be a 1D array of floats on the CPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	// thetaValues should be a 1D array of floats on the CPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	// rValues should be a 1D array of floats on the CPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCPU aTexture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	const long depthOut = phiValues.sizes()[0];
	const long heightOut = thetaValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float planeSize = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld() + aTexture.
		DepthWorld() * aTexture.DepthWorld());
	const Linear mappingIToOffset = Radon3D<Texture3DCPU>::GetMappingIToOffset(planeSize, samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;
	for (long layer = 0; layer < depthOut; ++layer) {
		for (long row = 0; row < heightOut; ++row) {
			for (long col = 0; col < widthOut; ++col) {
				const float phi = phiValues[layer].item().toFloat();
				const float theta = thetaValues[row].item().toFloat();
				const auto indexMappings = Radon3D<Texture3DCPU>::GetIndexMappings(
					aTexture, phi, theta, rValues[col].item().toFloat(), mappingIToOffset);
				const auto dIndexMappingsDR = Radon3D<Texture3DCPU>::GetDIndexMappingsDR(
					aTexture, phi, theta, mappingIToOffset);
				resultPtr[layer * widthOut * heightOut + row * widthOut + col] =
					scaleFactor * Radon3D<Texture3DCPU>::DIntegrateLoopedDMappingParameter(
						aTexture, indexMappings, dIndexMappingsDR, samplesPerDirection);
			}
		}
	}

	return result;
}

} // namespace ExtensionTest