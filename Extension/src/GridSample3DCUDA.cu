#include <torch/extension.h>

#include "../include/Texture3DCUDA.h"
#include "../include/GridSample3D.h"

namespace ExtensionTest {

using CommonData = GridSample3D<Texture3DCUDA>::CommonData;

__global__ void Kernel_GridSample3D_CUDA(Texture3DCUDA inputTexture, Linear<Vec<double, 3> > mappingGridToTexCoord,
                                         float *const grid, long numelOut, float *resultPtr) {
	const long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIndex >= numelOut) return;

	const long arrayIndex = threadIndex * 3;
	const Vec<float, 3> pos = {grid[arrayIndex], grid[arrayIndex + 1], grid[arrayIndex + 2]};

	resultPtr[threadIndex] = inputTexture.Sample(mappingGridToTexCoord(pos.StaticCast<double>()));
}

at::Tensor GridSample3D_CUDA(const at::Tensor &input, const at::Tensor &grid, const std::string &addressMode) {
	CommonData common = GridSample3D<Texture3DCUDA>::Common(input, grid, addressMode, at::DeviceType::CUDA);
	float *resultFlatPtr = common.flatOutput.data_ptr<float>();
	const Linear<Texture3DCUDA::VectorType> mappingGridToTexCoord = common.inputTexture.MappingWorldToTexCoord();

	const at::Tensor gridFlat = grid.view({-1, 3});
	float *const gridFlatPtr = gridFlat.data_ptr<float>();

	int minGridSize, blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, &Kernel_GridSample3D_CUDA, 0, 0);
	const int gridSize = (static_cast<unsigned>(common.flatOutput.numel()) + blockSize - 1) / blockSize;

	Kernel_GridSample3D_CUDA<<<gridSize, blockSize>>>(std::move(common.inputTexture), mappingGridToTexCoord,
	                                                  gridFlatPtr, common.flatOutput.numel(), resultFlatPtr);

	return common.flatOutput.view(grid.sizes().slice(0, grid.sizes().size() - 1));
}

} // namespace ExtensionTest