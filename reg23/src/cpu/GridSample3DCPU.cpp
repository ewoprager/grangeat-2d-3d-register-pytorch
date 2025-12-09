#include <reg23/Texture3DCPU.h>
#include <reg23/GridSample3D.h>

#include <reg23/Vec.h>

namespace reg23 {

using CommonData = GridSample3D<Texture3DCPU>::CommonData;

at::Tensor GridSample3D_CPU(const at::Tensor &input, const at::Tensor &grid, const std::string &addressModeX,
                            const std::string &addressModeY, const std::string &addressModeZ,
                            c10::optional<at::Tensor> out) {
	const CommonData common = GridSample3D<Texture3DCPU>::Common(input, grid, addressModeX, addressModeY, addressModeZ,
	                                                             at::DeviceType::CPU, out);
	float *resultFlatPtr = common.flatOutput.data_ptr<float>();
	const Linear<Texture3DCPU::VectorType> mappingGridToTexCoord = common.inputTexture.MappingWorldToTexCoord();

	const at::Tensor gridFlat = grid.view({-1, 3});

	for (int i = 0; i < gridFlat.sizes()[0]; ++i) {
		const Texture3DCPU::VectorType pos = Texture3DCPU::VectorType::FromTensor(gridFlat[i]);
		resultFlatPtr[i] = common.inputTexture.Sample(mappingGridToTexCoord(pos));
	}

	return common.flatOutput.view(grid.sizes().slice(0, grid.sizes().size() - 1));
}

} // namespace reg23
