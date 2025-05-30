#pragma once

#include "Common.h"
#include "Vec.h"
#include "Texture.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Resample the given 3D sinogram at locations corresponding to the given 2D sinogram grid
 * (`phiValues`, `rValues`), according to the 2D-3D image registration method based on Grangeat's relation.
 * @param sinogram3d a tensor of size (P,Q,R) containing `torch.float32`s: The 3D sinogram to resample. It must contain
 * plane integrals evenly spaced along `phi`, `theta` and `r` over each dimension respectively. This is the
 * pre-calculated derivative of the Radon transform of the CT volume.
 * @param sinogramSpacing a tensor of size (3,) containing `torch.float64`s: The spacing of (`phi`, `theta`, `r`)
 * between values in the given sinogram.
 * @param sinogramRangeCentres a tensor of size (3,) containing `torch.float64`s: The central values of
 * (`phi`, `theta`, `r`) of the given sinogram.
 * @param projectionMatrix a tensor of size (4, 4) containing `torch.float32`s: The projection matrix `PH` at the
 * desired transformation.
 * @param phiValues a tensor of any size containing `torch.float32`s: The polar coordinate `phi` values of the fixed
 * image at which to resample corresponding values in the given sinogram
 * @param rValues a tensor of the same size as `phiValues` containing `torch.float32`s: The polar coordinate `r` values
 * of the fixed image at which to resample corresponding values in the given sinogram
 * @return a tensor of the same size as `phiValues` containing `torch.float32`s: The values sampled from the given
 * volume sinogram at the locations corresponding to the given polar coordinates (`phiValues`, `rValues`), according to
 * the 2D-3D image registration method based on Grangeat's relation:
 *
 * Frysch R, Pfeiffer T, Rose G. A novel approach to 2D/3D registration of X-ray images using Grangeat's relation. Med
 * Image Anal. 2021 Jan;67:101815. doi: 10.1016/j.media.2020.101815. Epub 2020 Sep 30. PMID: 33065470.
 */
at::Tensor ResampleSinogram3D_CPU(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                  const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                  const at::Tensor &phiValues, const at::Tensor &rValues);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::ResampleSinogram3D_CPU that uses CUDA parallelisation.
 */
__host__ at::Tensor ResampleSinogram3D_CUDA(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                            const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                            const at::Tensor &phiValues, const at::Tensor &rValues);

/**
 * @tparam texture_t Type of the texture object that input data will be converted to for sampling.
 *
 * This struct is used as a namespace for code that is shared between different implementations of
 * `ResampleSinogram3D_...` functions
 */
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
		// sinogramSpacing should be a 1D tensor of 3 doubles
		TORCH_CHECK(sinogramSpacing.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(sinogramSpacing.dtype() == at::kDouble);
		// sinogramRangeCentres should be a 1D tensor of 3 doubles
		TORCH_CHECK(sinogramRangeCentres.sizes() == at::IntArrayRef{3});
		TORCH_CHECK(sinogramRangeCentres.dtype() == at::kDouble);
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

} // namespace reg23