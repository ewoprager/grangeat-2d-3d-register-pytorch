#include <torch/extension.h>

#include "../include/ResampleSinogram3D.h"
#include "../include/SinogramClassic3DCPU.h"
#include "../include/SinogramHEALPixCPU.h"

namespace reg23 {

using CommonData = ResampleSinogram3D::CommonData;

at::Tensor ResampleSinogram3D_CPU(const at::Tensor &sinogram3d, const std::string &sinogramType, const double rSpacing,
                                  const at::Tensor &projectionMatrix, const at::Tensor &phiValues,
                                  const at::Tensor &rValues) {
	const CommonData common = ResampleSinogram3D::Common(sinogram3d, sinogramType, projectionMatrix, phiValues, rValues,
	                                                     at::DeviceType::CPU);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	switch (common.sinogramType) {
	case ResampleSinogram3D::SinogramType::CLASSIC: {
		const SinogramClassic3DCPU sinogram = SinogramClassic3DCPU::FromTensor(sinogram3d, rSpacing);
		for (int i = 0; i < common.flatOutput.numel(); ++i) {
			const float phi = phiFlat[i].item().toFloat();
			const float r = rFlat[i].item().toFloat();
			resultFlatPtr[i] = ResampleSinogram3D::ResamplePlane(sinogram, common.geometry, phi, r);
		}
		break;
	}
	case ResampleSinogram3D::SinogramType::HEALPIX: {
		const SinogramHEALPixCPU sinogram = SinogramHEALPixCPU::FromTensor(sinogram3d, rSpacing);
		for (int i = 0; i < common.flatOutput.numel(); ++i) {
			const float phi = phiFlat[i].item().toFloat();
			const float r = rFlat[i].item().toFloat();
			resultFlatPtr[i] = ResampleSinogram3D::ResamplePlane(sinogram, common.geometry, phi, r);
		}
		break;
	}
	}

	return common.flatOutput.view(phiValues.sizes());
}

} // namespace reg23