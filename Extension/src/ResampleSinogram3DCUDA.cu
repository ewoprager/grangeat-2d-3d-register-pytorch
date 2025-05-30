#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"
#include "../include/ResampleSinogram3D.h"

namespace reg23 {

using CommonData = ResampleSinogram3D<Texture3DCUDA>::CommonData;

__global__ void Kernel_ResampleSinogram3D_CUDA(Texture3DCUDA inputTexture,
                                               Linear<Vec<double, 3> > mappingRThetaPhiToTexCoord,
                                               const Vec<Vec<float, 4>, 4> projectionMatrixTranspose,
                                               const float *phiValues, const float *rValues, long numelOut,
                                               Vec<float, 2> originProjection, float squareRadius, float *resultPtr) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;

	const float phi = phiValues[threadIndex];
	const float r = rValues[threadIndex];
	const float cp = cosf(phi);
	const float sp = sinf(phi);
	const Vec<float, 4> intermediate = MatMul(projectionMatrixTranspose, Vec<float, 4>{cp, sp, 0.f, -r});
	const Vec<float, 3> intermediate3 = intermediate.XYZ();
	const Vec<float, 3> posCartesian = -intermediate.W() * intermediate3 / intermediate3.Apply<float>(&Square<float>).
	                                   Sum();

	Vec<double, 3> rThetaPhi{};
	rThetaPhi.Z() = atan2(posCartesian.Y(), posCartesian.X());
	const float magXY = posCartesian.X() * posCartesian.X() + posCartesian.Y() * posCartesian.Y();
	rThetaPhi.Y() = atan2(posCartesian.Z(), sqrt(magXY));
	rThetaPhi.X() = sqrt(magXY + posCartesian.Z() * posCartesian.Z());
	rThetaPhi = UnflipSphericalCoordinate(rThetaPhi);

	resultPtr[threadIndex] = inputTexture.Sample(mappingRThetaPhiToTexCoord(rThetaPhi));

	if ((r * Vec<float, 2>{cp, sp} - .5f * originProjection).Apply<float>(&Square<float>).Sum() < squareRadius) {
		resultPtr[threadIndex] *= -1.f;
	}
}

/**
 * @brief
 *
 *	Note: Assumes that the projection matrix projects onto the x-y plane, and that the radial coordinates (phi, r)
 *	in that plane measure phi right-hand rule about the z-axis from the positive x-direction
 *
 * @param sinogram3d
 * @param sinogramSpacing
 * @param sinogramRangeCentres
 * @param projectionMatrix
 * @param phiValues
 * @param rValues
 * @return
 */
__host__ at::Tensor ResampleSinogram3D_CUDA(const at::Tensor &sinogram3d, const at::Tensor &sinogramSpacing,
                                            const at::Tensor &sinogramRangeCentres, const at::Tensor &projectionMatrix,
                                            const at::Tensor &phiValues, const at::Tensor &rValues) {
	CommonData common = ResampleSinogram3D<Texture3DCUDA>::Common(sinogram3d, sinogramSpacing, sinogramRangeCentres,
	                                                              projectionMatrix, phiValues, rValues,
	                                                              at::DeviceType::CUDA);

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	const at::Tensor originProjectionHomogeneous = matmul(projectionMatrix,
	                                                      torch::tensor(
		                                                      {{0.f, 0.f, 0.f, 1.f}}, projectionMatrix.options()).t());
	const Vec<float, 2> originProjection = Vec<float, 2>{originProjectionHomogeneous[0].item().toFloat(),
	                                                     originProjectionHomogeneous[1].item().toFloat()} /
	                                       originProjectionHomogeneous[3].item().toFloat();
	const float squareRadius = .25f * originProjection.Apply<float>(&Square<float>).Sum();

	const at::Tensor pht = projectionMatrix.t();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_ResampleSinogram3D_CUDA, 0, 0);
	const int gridSize = (static_cast<unsigned>(common.flatOutput.numel()) + blockSize - 1) / blockSize;

	Kernel_ResampleSinogram3D_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture),
	                                                        common.mappingRThetaPhiToTexCoord,
	                                                        Vec<Vec<float, 4>, 4>::FromTensor2D(pht), phiFlatPtr,
	                                                        rFlatPtr, common.flatOutput.numel(), originProjection,
	                                                        squareRadius, resultFlatPtr);

	return common.flatOutput.view(phiValues.sizes());
}

} // namespace reg23