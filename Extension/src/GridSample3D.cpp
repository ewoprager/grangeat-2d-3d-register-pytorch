#include <torch/extension.h>

#include "../include/Texture3DCPU.h"
#include "../include/GridSample3D.h"

#include "../include/Vec.h"

namespace ExtensionTest {

using CommonData = GridSample3D<Texture3DCPU>::CommonData;

at::Tensor GridSample3D_CPU(const at::Tensor &input, const at::Tensor &grid) {
	const CommonData common = GridSample3D<Texture3DCPU>::Common(input, grid, at::DeviceType::CPU);
	float *resultFlatPtr = common.flatOutput.data_ptr<float>();
	const Linear<Texture3DCPU::VectorType> mappingGridToTexCoord = common.inputTexture.MappingWorldToTexCoord();

	const at::Tensor gridFlat = grid.view({-1, 3});

	for (int i = 0; i < gridFlat.sizes()[0]; ++i) {
		const Texture3DCPU::VectorType pos = Texture3DCPU::VectorType::FromTensor(gridFlat[i]);
		resultFlatPtr[i] = common.inputTexture.Sample(mappingGridToTexCoord(pos));
	}

	return common.flatOutput.view(grid.sizes().slice(0, grid.sizes().size() - 1));
}

} // namespace ExtensionTest