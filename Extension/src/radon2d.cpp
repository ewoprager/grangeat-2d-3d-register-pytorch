#include <torch/extension.h>

#include "../include/Texture2DCPU.h"
#include "../include/Radon2D.h"

#include "../include/Vec.h"

namespace ExtensionTest {

at::Tensor radon2d_cpu(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                       const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D tensor of floats on the CPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CPU);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCPU aTexture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float lineLength = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld());
	const auto mappingIToOffset = Radon2D<Texture2DCPU>::GetMappingIToOffset(lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	for (long i = 0; i < numelOut; ++i) {
		const auto indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(
			aTexture, phiFlat[i].item().toFloat(), rFlat[i].item().toFloat(), mappingIToOffset);
		resultFlatPtr[i] = scaleFactor * Radon2D<Texture2DCPU>::IntegrateLooped(aTexture, indexMappings, samplesPerLine);
	}

	return resultFlat.view(phiValues.sizes());
}

at::Tensor dRadon2dDR_cpu(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                          const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the CPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CPU);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCPU aTexture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const float lineLength = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld());
	const auto constMappings = Radon2D<Texture2DCPU>::GetMappingIToOffset(lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	for (long i = 0; i < numelOut; ++i) {
		const float phi = phiFlat[i].item().toFloat();
		const float r = rFlat[i].item().toFloat();
		const auto indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(aTexture, phi, r, constMappings);
		const auto derivativeWRTR = Radon2D<Texture2DCPU>::GetDerivativeWRTR(aTexture, phi, r);
		resultFlatPtr[i] = scaleFactor * Radon2D<Texture2DCPU>::DIntegrateLoopedDMappingParameter(
			               aTexture, indexMappings, derivativeWRTR, samplesPerLine);
	}

	return resultFlat.view(phiValues.sizes());
}

} // namespace ExtensionTest