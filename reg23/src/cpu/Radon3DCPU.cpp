

#include <reg23/Texture3DCPU.h>
#include <reg23/Radon3D.h>

namespace reg23 {

using CommonData = Radon3D<Texture3DCPU>::CommonData;

at::Tensor Radon3D_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                       const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection) {
	const CommonData common = Radon3D<Texture3DCPU>::Common(volume, volumeSpacing, phiValues, thetaValues, rValues,
	                                                        samplesPerDirection, at::DeviceType::CPU);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	for (int64_t i = 0; i < common.flatOutput.numel(); ++i) {
		const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCPU>::GetMappingIndexToTexCoord(
			common.inputTexture, phiFlat[i].item().toFloat(), thetaFlat[i].item().toFloat(), rFlat[i].item().toFloat(),
			common.mappingIndexToOffset);
		resultFlatPtr[i] = common.scaleFactor * Radon3D<Texture3DCPU>::IntegrateLooped(
			                   common.inputTexture, mappingIndexToTexCoord, samplesPerDirection);
	}
	return common.flatOutput.view(phiValues.sizes());
}

at::Tensor DRadon3DDR_CPU(const at::Tensor &volume, const at::Tensor &volumeSpacing, const at::Tensor &phiValues,
                          const at::Tensor &thetaValues, const at::Tensor &rValues, int64_t samplesPerDirection) {
	const CommonData common = Radon3D<Texture3DCPU>::Common(volume, volumeSpacing, phiValues, thetaValues, rValues,
	                                                        samplesPerDirection, at::DeviceType::CPU);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor thetaFlat = thetaValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	for (int64_t i = 0; i < common.flatOutput.numel(); ++i) {
		const double phi = phiFlat[i].item().toFloat();
		const double theta = thetaFlat[i].item().toFloat();
		const double r = rFlat[i].item().toFloat();
		const Linear2<Vec<double, 3> > mappingIndexToTexCoord = Radon3D<Texture3DCPU>::GetMappingIndexToTexCoord(
			common.inputTexture, phi, theta, r, common.mappingIndexToOffset);
		const Vec<double, 3> dTexCoordDR = Radon3D<Texture3DCPU>::GetDTexCoordDR(common.inputTexture, phi, theta, r);
		resultFlatPtr[i] = common.scaleFactor * Radon3D<Texture3DCPU>::DIntegrateLoopedDMappingParameter(
			                   common.inputTexture, mappingIndexToTexCoord, dTexCoordDR, samplesPerDirection);

	}
	return common.flatOutput.view(phiValues.sizes());
}

} // namespace reg23
