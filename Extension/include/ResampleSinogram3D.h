#pragma once

#include "Common.h"
#include "Vec.h"

namespace ExtensionTest {

at::Tensor ResampleSinogram3DCPU(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                 const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                 const at::Tensor &phiGrid, const at::Tensor &rGrid);

template <typename texture_t> struct ResampleSinogram3D {

	struct CommonData {
		texture_t inputTexture{};
		Linear<Vec<double, 3> > mappingRThetaPhiToTexCoord;
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
	                                  const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
	                                  const at::Tensor &phiGrid, const at::Tensor &rGrid, at::DeviceType device) {
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
		// phiGrid and rGrid should be of the same size, contain floats and be on the chosen device
		TORCH_CHECK(phiGrid.sizes() == rGrid.sizes())
		TORCH_CHECK(phiGrid.dtype() == at::kFloat)
		TORCH_CHECK(rGrid.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(phiGrid.device().type() == device)
		TORCH_INTERNAL_ASSERT(rGrid.device().type() == device)

		CommonData ret{};
		const at::Tensor sinogramContiguous = sinogram3d.contiguous();
		const float *sinogramPtr = sinogramContiguous.data_ptr<float>();
		ret.inputTexture = texture_t{sinogramPtr, Vec<int64_t, 3>::FromIntArrayRef(sinogram3d.sizes()).Flipped(),
		                             Vec<double, 3>::FromTensor(sinogramSpacing),
		                             Vec<double, 3>::FromTensor(sinogramRangeCentres)};
		ret.mappingRThetaPhiToTexCoord = ret.inputTexture.MappingWorldToTexCoord();
		ret.flatOutput = torch::zeros(at::IntArrayRef({phiGrid.numel()}), sinogramContiguous.options());
		return ret;
	}
};

} // namespace ExtensionTest