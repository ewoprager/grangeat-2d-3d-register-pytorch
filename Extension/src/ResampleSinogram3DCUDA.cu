#include <torch/extension.h>

#include "../include/ResampleSinogram3D.h"
#include "../include/SinogramClassic3DCPU.h"
#include "../include/SinogramHEALPixCPU.h"

namespace reg23 {

using CommonData = ResampleSinogram3D::CommonData;
using ConstantGeometry = ResampleSinogram3D::ConstantGeometry;

template <typename sinogram_t> __global__ void Kernel_ResampleSinogram3D_CUDA(
	sinogram_t inputSinogram, const ConstantGeometry geometry, const float *phiValues, const float *rValues,
	long numelOut, float *resultPtr) {

	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;

	const float phi = phiValues[threadIndex];
	const float r = rValues[threadIndex];
	resultPtr[threadIndex] = ResampleSinogram3D::ResamplePlane(inputSinogram, geometry, phi, r);
}

/**
 * @brief
 *
 *	Note: Assumes that the projection matrix projects onto the x-y plane, and that the radial coordinates (phi, r)
 *	in that plane measure phi right-hand rule about the z-axis from the positive x-direction
 *
 * @param sinogram3d
 * @param sinogramType
 * @param rSpacing
 * @param projectionMatrix
 * @param phiValues
 * @param rValues
 * @return
 */
__host__ at::Tensor ResampleSinogram3D_CUDA(const at::Tensor &sinogram3d, const std::string &sinogramType,
                                            double rSpacing, const at::Tensor &projectionMatrix,
                                            const at::Tensor &phiValues, const at::Tensor &rValues) {
	CommonData common = ResampleSinogram3D::Common(sinogram3d, sinogramType, projectionMatrix, phiValues, rValues,
	                                               at::DeviceType::CUDA);

	const at::Tensor phiFlatContiguous = phiValues.flatten().contiguous();
	const float *phiFlatPtr = phiFlatContiguous.data_ptr<float>();
	const at::Tensor rFlatContiguous = rValues.flatten().contiguous();
	const float *rFlatPtr = rFlatContiguous.data_ptr<float>();

	float *resultFlatPtr = common.flatOutput.data_ptr<float>();

	switch (common.sinogramType) {
	case ResampleSinogram3D::SinogramType::CLASSIC: {
		const SinogramClassic3DCPU sinogram = SinogramClassic3DCPU::FromTensor(sinogram3d, rSpacing);
		int minGridSize, blockSize;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		                                   &Kernel_ResampleSinogram3D_CUDA<SinogramClassic3DCPU>, 0, 0);
		const int gridSize = (static_cast<unsigned>(common.flatOutput.numel()) + blockSize - 1) / blockSize;

		Kernel_ResampleSinogram3D_CUDA<SinogramClassic3DCPU><<<gridSize, blockSize>>>(
			sinogram, common.geometry, phiFlatPtr, rFlatPtr, common.flatOutput.numel(), resultFlatPtr);
		break;
	}
	case ResampleSinogram3D::SinogramType::HEALPIX: {
		const SinogramHEALPixCPU sinogram = SinogramHEALPixCPU::FromTensor(sinogram3d, rSpacing);
		int minGridSize, blockSize;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
		                                   &Kernel_ResampleSinogram3D_CUDA<SinogramHEALPixCPU>, 0, 0);
		const int gridSize = (static_cast<unsigned>(common.flatOutput.numel()) + blockSize - 1) / blockSize;

		Kernel_ResampleSinogram3D_CUDA<SinogramHEALPixCPU><<<gridSize, blockSize>>>(
			sinogram, common.geometry, phiFlatPtr, rFlatPtr, common.flatOutput.numel(), resultFlatPtr);
		break;
	}
	}
	return common.flatOutput.view(phiValues.sizes());
}

} // namespace reg23