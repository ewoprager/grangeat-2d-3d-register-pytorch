#include <torch/extension.h>

#include "../include/Texture2DCPU.h"
#include "../include/Radon2D.h"

#include "../include/Vec.h"

namespace ExtensionTest {


at::Tensor radon2d_cpu(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                       const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D tensor of floats on the CPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CPU);
	// imageSpacing should be a 1D tensor of 2 floats or doubles on the CPU
	TORCH_CHECK(imageSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(imageSpacing.dtype() == at::kFloat || imageSpacing.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(imageSpacing.device().type() == at::DeviceType::CPU);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const Texture2DCPU texture = Texture2DCPU::FromTensor(image, imageSpacing);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), image.contiguous().options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const double lineLength = sqrt(texture.SizeWorld().Apply<double>(&Square<double>).Sum());
	const Linear<Vec<double, 2> > mappingIToOffset = Radon2D<Texture2DCPU>::GetMappingIToOffset(
		lineLength, samplesPerLine);
	const double scaleFactor = lineLength / static_cast<double>(samplesPerLine);
	for (long i = 0; i < numelOut; ++i) {
		const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCPU>::GetMappingIndexToTexCoord(
			texture, phiFlat[i].item().toFloat(), rFlat[i].item().toFloat(), mappingIToOffset);
		resultFlatPtr[i] = scaleFactor * Radon2D<Texture2DCPU>::IntegrateLooped(
			                   texture, mappingIndexToTexCoord, samplesPerLine);
	}

	return resultFlat.view(phiValues.sizes());
}

at::Tensor dRadon2dDR_cpu(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
                          const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the CPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CPU);
	// imageSpacing should be a 1D tensor of 2 floats or doubles on the CPU
	TORCH_CHECK(imageSpacing.sizes() == at::IntArrayRef{2});
	TORCH_CHECK(imageSpacing.dtype() == at::kFloat || imageSpacing.dtype() == at::kDouble);
	TORCH_INTERNAL_ASSERT(imageSpacing.device().type() == at::DeviceType::CPU);
	// phiValues and rValues should have the same sizes, should contain floats, and be on the CPU
	TORCH_CHECK(phiValues.sizes() == rValues.sizes());
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	const Texture2DCPU texture = Texture2DCPU::FromTensor(image, imageSpacing);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	const long numelOut = phiValues.numel();
	const at::Tensor resultFlat = torch::zeros(at::IntArrayRef({numelOut}), image.contiguous().options());
	float *resultFlatPtr = resultFlat.data_ptr<float>();

	const double lineLength = sqrt(texture.SizeWorld().Apply<double>(&Square<double>).Sum());
	const Linear<Vec<double, 2> > constMappings = Radon2D<
		Texture2DCPU>::GetMappingIToOffset(lineLength, samplesPerLine);
	const double scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	for (long i = 0; i < numelOut; ++i) {
		const double phi = phiFlat[i].item().toFloat();
		const double r = rFlat[i].item().toFloat();
		const Linear<Vec<double, 2> > mappingIndexToTexCoord = Radon2D<Texture2DCPU>::GetMappingIndexToTexCoord(
			texture, phi, r, constMappings);
		const Vec<double, 2> dTexCoordDR = Radon2D<Texture2DCPU>::GetDTexCoordDR(texture, phi, r);
		resultFlatPtr[i] = scaleFactor * Radon2D<Texture2DCPU>::DIntegrateLoopedDMappingParameter(
			                   texture, mappingIndexToTexCoord, dTexCoordDR, samplesPerLine);
	}

	return resultFlat.view(phiValues.sizes());
}


} // namespace ExtensionTest