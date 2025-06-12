#pragma once

#include "Common.h"
#include "Texture.h"
#include "Vec.h"

namespace reg23 {

/**
 * @ingroup pytorch_functions
 * @brief Resample the given 3D sinogram at locations corresponding to the given 2D sinogram grid
 * (`phiValues`, `rValues`), according to the 2D-3D image registration method based on Grangeat's relation.
 * @param sinogram3d a tensor of size (P,Q,R) containing `torch.float32`s: The 3D sinogram to resample. It must contain
 * plane integrals evenly spaced along `phi`, `theta` and `r` over each dimension respectively. This is the
 * pre-calculated derivative of the Radon transform of the CT volume.
 * @param sinogramType a string; one of the following: "classic", "healpix". see documentation for `enum`
 * `reg23::ResampleSinogram3D::SinogramType`
 * @param rSpacing The spacing in `r` between the plane integrals in the sinogram.
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
 *
 * Note: Assumes that the projection matrix projects onto the x-y plane, and that the radial coordinates (phi, r)
 * in that plane measure phi right-hand rule about the z-axis from the positive x-direction
 */
at::Tensor ResampleSinogram3D_CPU(const at::Tensor &sinogram3d, const std::string &sinogramType, double rSpacing,
								  const at::Tensor &projectionMatrix, const at::Tensor &phiValues,
								  const at::Tensor &rValues);

/**
 * @ingroup pytorch_functions
 * @brief An implementation of reg23::ResampleSinogram3D_CPU that uses CUDA parallelisation.
 */
__host__ at::Tensor ResampleSinogram3D_CUDA(const at::Tensor &sinogram3d, const std::string &sinogramType,
											double rSpacing, const at::Tensor &projectionMatrix,
											const at::Tensor &phiValues, const at::Tensor &rValues);

/**
 * This struct is used as a namespace for code that is shared between different implementations of
 * `ResampleSinogram3D_...` functions
 */
struct ResampleSinogram3D {

	enum class SinogramType { CLASSIC, HEALPIX };

	struct ConstantGeometry {
		Vec<float, 2> originProjection{};
		float squareRadius{};
		Vec<Vec<float, 4>, 4> projectionMatrixTranspose{};
	};

	struct CommonData {
		SinogramType sinogramType{};
		ConstantGeometry geometry{};
		at::Tensor flatOutput{};
	};

	__host__ static CommonData Common(const at::Tensor &sinogram3d, const std::string &sinogramType,
									  const at::Tensor &projectionMatrix, const at::Tensor &phiValues,
									  const at::Tensor &rValues, at::DeviceType device) {
		// sinogram3d should be on the chosen device; other assertions will be made in sinogram_t::FromTensor
		TORCH_INTERNAL_ASSERT(sinogram3d.device().type() == device)
		// projectionMatrix should be of size (4, 4), contain floats and be on the chosen device
		TORCH_CHECK(projectionMatrix.sizes() == at::IntArrayRef({4, 4}))
		TORCH_CHECK(projectionMatrix.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(projectionMatrix.device().type() == device)
		// phiValues and rValues should be of the same size, contain floats and be on the chosen device
		TORCH_CHECK(phiValues.sizes() == rValues.sizes())
		TORCH_CHECK(phiValues.dtype() == at::kFloat)
		TORCH_CHECK(rValues.dtype() == at::kFloat)
		TORCH_INTERNAL_ASSERT(phiValues.device().type() == device)
		TORCH_INTERNAL_ASSERT(rValues.device().type() == device);

		SinogramType st = SinogramType::CLASSIC;
		if (sinogramType == "healpix") {
			st = SinogramType::HEALPIX;
		} else if (sinogramType != "classic") {
			TORCH_WARN("Invalid sinogram type string given. Valid values are: 'classic', 'healpix'. Assuming default "
					   "value: 'classic'.")
		}

		const at::Tensor originProjectionHomogeneous =
			matmul(projectionMatrix, torch::tensor({{0.f, 0.f, 0.f, 1.f}}, projectionMatrix.options()).t());

		CommonData ret{};
		ret.sinogramType = st;
		ret.geometry.originProjection = Vec<float, 2>{originProjectionHomogeneous[0].item().toFloat(),
													  originProjectionHomogeneous[1].item().toFloat()} /
										originProjectionHomogeneous[3].item().toFloat();
		ret.geometry.squareRadius = .25f * ret.geometry.originProjection.Apply<float>(&Square<float>).Sum();
		ret.geometry.projectionMatrixTranspose = Vec<Vec<float, 4>, 4>::FromTensor2D(projectionMatrix.t());
		ret.flatOutput = torch::zeros(at::IntArrayRef({phiValues.numel()}), sinogram3d.contiguous().options());
		return ret;
	}

	template <typename sinogram_t>
	__host__ __device__ static float ResamplePlane(const sinogram_t &sinogram, const ConstantGeometry &geometry,
												   float phi, float r) {
		const float cp = cosf(phi);
		const float sp = sinf(phi);
		const Vec<float, 4> intermediate = MatMul(geometry.projectionMatrixTranspose, Vec<float, 4>{cp, sp, 0.f, -r});
		const Vec<float, 3> intermediate3 = intermediate.XYZ();
		const Vec<float, 3> posCartesian =
			-intermediate.W() * intermediate3 / intermediate3.Apply<float>(&Square<float>).Sum();

		Vec<double, 3> rThetaPhi{};
		rThetaPhi.Z() = atan2(posCartesian.Y(), posCartesian.X());
		const float magXY = posCartesian.X() * posCartesian.X() + posCartesian.Y() * posCartesian.Y();
		rThetaPhi.Y() = atan2(posCartesian.Z(), sqrt(magXY));
		rThetaPhi.X() = sqrt(magXY + posCartesian.Z() * posCartesian.Z());
		rThetaPhi = UnflipSphericalCoordinate(rThetaPhi);

		float ret = sinogram.Sample(rThetaPhi);

		if ((r * Vec<float, 2>{cp, sp} - .5f * geometry.originProjection).Apply<float>(&Square<float>).Sum() <
			geometry.squareRadius) {
			ret *= -1.f;
		}
		return ret;
	}
};

} // namespace reg23