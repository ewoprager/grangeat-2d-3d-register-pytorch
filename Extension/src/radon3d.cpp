#include <torch/extension.h>

#include "../include/Texture3DCPU.h"

namespace ExtensionTest {

at::Tensor radon3d_cpu(const at::Tensor &volume, double xSpacing, double ySpacing, double zSpacing, long depthOut,
                       long heightOut, long widthOut, long samplesPerDirection) {
	// volume should be a 3D array of floats on the CPU
	TORCH_CHECK(volume.sizes().size() == 3);
	TORCH_CHECK(volume.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(volume.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = volume.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture3DCPU aTexture{aPtr, volume.sizes()[2], volume.sizes()[1], volume.sizes()[0], xSpacing, ySpacing,
	                            zSpacing};

	at::Tensor result = torch::zeros(at::IntArrayRef({depthOut, heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float planeSize = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld() + aTexture.
		DepthWorld() * aTexture.DepthWorld());
	const auto constMappings = Radon3D<Texture3DCPU>::GetConstMappings(widthOut, heightOut, depthOut, planeSize,
	                                                                   samplesPerDirection);
	const float rootScaleFactor = planeSize / static_cast<float>(samplesPerDirection);
	const float scaleFactor = rootScaleFactor * rootScaleFactor;
	for (long layer = 0; layer < depthOut; ++layer) {
		for (long row = 0; row < heightOut; ++row) {
			for (long col = 0; col < widthOut; ++col) {
				const auto indexMappings = Radon3D<Texture3DCPU>::GetIndexMappings(
					aTexture, col, row, layer, constMappings);
				resultPtr[layer * widthOut * heightOut + row * widthOut + col] =
					scaleFactor * Radon3D<Texture3DCPU>::IntegrateLooped(aTexture, indexMappings, samplesPerDirection);
			}
		}
	}

	return result;
}

} // namespace ExtensionTest