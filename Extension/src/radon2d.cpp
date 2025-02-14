#include <torch/extension.h>

#include "../include/Texture2DCPU.h"
#include "../include/Radon2D.h"

#include "../include/Vec.h"

namespace ExtensionTest {

at::Tensor radon2d_cpu(const at::Tensor &image, const Vec<double, 2> &imageSpacing, const at::Tensor &phiValues,
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
	const Texture2DCPU aTexture{aPtr, VecFlip(VecFromIntArrayRef<int64_t, 2>(image.sizes())), imageSpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const double lineLength = sqrt(VecSum(VecApply<double>(&Square<double>, aTexture.SizeWorld())));
	const Linear<Vec<double, 2> > mappingIToOffset = Radon2D<Texture2DCPU>::GetMappingIToOffset(
		lineLength, samplesPerLine);
	const double scaleFactor = lineLength / static_cast<double>(samplesPerLine);
	for (long i = 0; i < numelOut; ++i) {
		const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCPU>::GetMappingIndexToTexCoord(
			aTexture, phiFlat[i].item().toFloat(), rFlat[i].item().toFloat(), mappingIToOffset);
		resultFlatPtr[i] = scaleFactor * Radon2D<Texture2DCPU>::IntegrateLooped(
			                   aTexture, mappingIndexToTexCoord, samplesPerLine);
	}

	return resultFlat.view(phiValues.sizes());
}

at::Tensor dRadon2dDR_cpu(const at::Tensor &image, const Vec<double, 2> &imageSpacing, const at::Tensor &phiValues,
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
	const Texture2DCPU aTexture{aPtr, VecFlip(VecFromIntArrayRef<int64_t, 2>(image.sizes())), imageSpacing};

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), aContiguous.options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const double lineLength = sqrt(VecSum(VecApply<double>(&Square<double>, aTexture.SizeWorld())));
	const Linear<Vec<double, 2> > constMappings = Radon2D<
		Texture2DCPU>::GetMappingIToOffset(lineLength, samplesPerLine);
	const double scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	for (long i = 0; i < numelOut; ++i) {
		const double phi = phiFlat[i].item().toFloat();
		const double r = rFlat[i].item().toFloat();
		const Linear<Vec<double, 2> > indexMappings = Radon2D<Texture2DCPU>::GetMappingIndexToTexCoord(
			aTexture, phi, r, constMappings);
		const Vec<double, 2> dTexCoordDR = Radon2D<Texture2DCPU>::GetDTexCoordDR(aTexture, phi, r);
		resultFlatPtr[i] = scaleFactor * Radon2D<Texture2DCPU>::DIntegrateLoopedDMappingParameter(
			                   aTexture, indexMappings, dTexCoordDR, samplesPerLine);
	}

	return resultFlat.view(phiValues.sizes());
}

} // namespace ExtensionTest