/**
 * @file
 * @brief CPU implementations of 2D Radon transforms
 */

#include <torch/extension.h>

#include "../include/Radon2D.h"
#include "../include/Texture2DCPU.h"

#include "../include/Vec.h"

namespace ExtensionTest {

using CommonData = Radon2D<Texture2DCPU>::CommonData;

at::Tensor Radon2D_CPU(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
					   const at::Tensor &rValues, int64_t samplesPerLine) {
	const CommonData common =
		Radon2D<Texture2DCPU>::Common(image, imageSpacing, phiValues, rValues, samplesPerLine, at::DeviceType::CPU);
	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	for (int64_t i = 0; i < common.flatOutput.numel(); ++i) {
		const Linear<Vec<double, 2>> mappingIndexToTexCoord = Radon2D<Texture2DCPU>::GetMappingIndexToTexCoord(
			common.inputTexture, phiFlat[i].item().toFloat(), rFlat[i].item().toFloat(), common.mappingIndexToOffset);
		resultFlatPtr[i] = common.scaleFactor * Radon2D<Texture2DCPU>::IntegrateLooped(
													common.inputTexture, mappingIndexToTexCoord, samplesPerLine);
	}
	return common.flatOutput.view(phiValues.sizes());
}

at::Tensor DRadon2DDR_CPU(const at::Tensor &image, const at::Tensor &imageSpacing, const at::Tensor &phiValues,
						  const at::Tensor &rValues, int64_t samplesPerLine) {
	const CommonData common =
		Radon2D<Texture2DCPU>::Common(image, imageSpacing, phiValues, rValues, samplesPerLine, at::DeviceType::CPU);
	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	for (int64_t i = 0; i < common.flatOutput.numel(); ++i) {
		const double phi = phiFlat[i].item().toFloat();
		const double r = rFlat[i].item().toFloat();
		const Linear<Vec<double, 2>> mappingIndexToTexCoord =
			Radon2D<Texture2DCPU>::GetMappingIndexToTexCoord(common.inputTexture, phi, r, common.mappingIndexToOffset);
		const Vec<double, 2> dTexCoordDR = Radon2D<Texture2DCPU>::GetDTexCoordDR(common.inputTexture, phi, r);
		resultFlatPtr[i] =
			common.scaleFactor * Radon2D<Texture2DCPU>::DIntegrateLoopedDMappingParameter(
									 common.inputTexture, mappingIndexToTexCoord, dTexCoordDR, samplesPerLine);
	}

	return common.flatOutput.view(phiValues.sizes());
}

} // namespace ExtensionTest