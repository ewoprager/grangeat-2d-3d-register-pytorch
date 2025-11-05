#include <torch/extension.h>

#include "../include/ResampleSinogram3D.h"
#include "../include/SinogramClassic3D.h"
#include "../include/SinogramHEALPix.h"
#include "../include/Texture3DCPU.h"

namespace reg23 {

using CommonData = ResampleSinogram3D::CommonData;

at::Tensor ResampleSinogram3D_CPU(const at::Tensor &sinogram3d, const std::string &sinogramType, const double rSpacing,
                                  const at::Tensor &projectionMatrix, const at::Tensor &phiValues,
                                  const at::Tensor &rValues, c10::optional<at::Tensor> out) {
	const CommonData common = ResampleSinogram3D::Common(sinogramType, projectionMatrix, phiValues, rValues,
	                                                     at::DeviceType::CPU, out);

	const at::Tensor phiFlat = phiValues.flatten();
	const at::Tensor rFlat = rValues.flatten();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	switch (common.sinogramType) {
	case ResampleSinogram3D::SinogramType::CLASSIC: {
		const SinogramClassic3D<Texture3DCPU> sinogram = SinogramClassic3D<Texture3DCPU>::FromTensor(
			sinogram3d, rSpacing);
		for (int i = 0; i < common.flatOutput.numel(); ++i) {
			const float phi = phiFlat[i].item().toFloat();
			const float r = rFlat[i].item().toFloat();
			resultFlatPtr[i] = ResampleSinogram3D::ResamplePlane(sinogram, common.geometry, phi, r);
		}
		break;
	}
	case ResampleSinogram3D::SinogramType::HEALPIX: {
		const SinogramHEALPix<Texture3DCPU> sinogram = SinogramHEALPix<Texture3DCPU>::FromTensor(sinogram3d, rSpacing);
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