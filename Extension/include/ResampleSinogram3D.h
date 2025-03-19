#pragma once

#include "Common.h"
#include "Vec.h"
#include "Texture.h"

namespace ExtensionTest {

at::Tensor ResampleSinogram3D_CPU(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                  const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                  const at::Tensor &phiValues, const at::Tensor &rValues);

__host__ at::Tensor ResampleSinogram3D_CUDA(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                            const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                            const at::Tensor &phiValues, const at::Tensor &rValues);

template <typename texture_t> struct ResampleSinogram3D {

	struct CommonData {
		texture_t inputTexture{};
		Linear<Vec<double, 3> > mappingRThetaPhiToTexCoord;
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
	                                  const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
	                                  const at::Tensor &phiValues, const at::Tensor &rValues, at::DeviceType device) {
		// sinogram3d should be a 3D tensor of floats on the chosen device
		TORCH_CHECK(sinogram3d.sizes().size() == 3)
		TORCH_CHECK(sinogram3d.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(sinogram3d.device().type() == device)
		// sinogramSpacing should be a 1D tensor of 3 floats or doubles
		TORCH_CHECK(sinogramSpacing.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(sinogramSpacing.dtype() == at::kFloat || sinogramSpacing.dtype() == at::kDouble);
		// sinogramRangeCentres should be a 1D tensor of 3 floats or doubles
		TORCH_CHECK(sinogramRangeCentres.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(sinogramRangeCentres.dtype() == at::kFloat || sinogramRangeCentres.dtype() == at::kDouble);
		// projectionMatrix should be of size (4, 4), contain floats and be on the chosen device
		TORCH_CHECK(projectionMatrix.sizes() == at::IntArrayRef({4, 4}))
		TORCH_CHECK(projectionMatrix.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(projectionMatrix.device().type() == device)
		// phiValues and rValues should be of the same size, contain floats and be on the chosen device
		TORCH_CHECK(phiValues.sizes() == rValues.sizes())
		TORCH_CHECK(phiValues.dtype() == at::kFloat)
		TORCH_CHECK(rValues.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(phiValues.device().type() == device)
		TORCH_INTERNAL_ASSERT(rValues.device().type() == device)

		CommonData ret{};
		const at::Tensor sinogramContiguous = sinogram3d.contiguous();
		const float *sinogramPtr = sinogramContiguous.data_ptr<float>();
		ret.inputTexture = texture_t{sinogramPtr, Vec<int64_t, 3>::FromIntArrayRef(sinogram3d.sizes()).Flipped(),
		                             Vec<double, 3>::FromTensor(sinogramSpacing),
		                             Vec<double, 3>::FromTensor(sinogramRangeCentres),
		                             {TextureAddressMode::ZERO, TextureAddressMode::ZERO, TextureAddressMode::WRAP}};
		ret.mappingRThetaPhiToTexCoord = ret.inputTexture.MappingWorldToTexCoord();
		ret.flatOutput = torch::zeros(at::IntArrayRef({phiValues.numel()}), sinogramContiguous.options());
		return ret;
	}
};

} // namespace ExtensionTest