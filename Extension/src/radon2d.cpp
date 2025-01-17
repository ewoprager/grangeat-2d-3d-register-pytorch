#include <torch/extension.h>

#include "../include/Texture2DCPU.h"

namespace ExtensionTest {

at::Tensor radon2d_cpu(const at::Tensor &a, double xSpacing, double ySpacing, long heightOut, long widthOut,
                       long samplesPerLine) {
	// a should be a 2D array of floats on the CPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCPU aTexture{aPtr, a.sizes()[1], a.sizes()[0], xSpacing, ySpacing};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float rayLength = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld());
	const auto constMappings = Radon2D<Texture2DCPU>::GetConstMappings(widthOut, heightOut, rayLength, samplesPerLine);
	const float scaleFactor = rayLength / static_cast<float>(samplesPerLine);
	for (long row = 0; row < heightOut; ++row) {
		for (long col = 0; col < widthOut; ++col) {
			const Radon2D<Texture2DCPU>::IndexMappings indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(
				aTexture, col, row, constMappings);
			resultPtr[row * widthOut + col] = scaleFactor * Radon2D<Texture2DCPU>::IntegrateLooped(
				                                  aTexture, indexMappings, samplesPerLine);
		}
	}

	return result;
}

void radon_v2_kernel_synchronous(const Texture2DCPU &textureIn, size_t blockSize, size_t blockId, long samplesPerLine,
                                 const Radon2D<Texture2DCPU>::IndexMappings indexMappings, float scaleFactor,
                                 float *patchSumsPtr) {
	float *buffer = static_cast<float *>(malloc(blockSize * sizeof(float)));

	for (size_t j = 0; j < blockSize; ++j) {
		const int i = blockSize * blockId + j;
		if (i >= samplesPerLine) {
			buffer[j] = 0.f;
			break;
		}
		const float iF = static_cast<float>(i);
		buffer[j] = textureIn.Sample(indexMappings.mappingIToX(iF), indexMappings.mappingIToY(iF));
	}

	for (long cutoff = blockSize / 2; cutoff > 0; cutoff /= 2) {
		for (int i = 0; i < cutoff; ++i) {
			buffer[i] += buffer[i + cutoff];
		}
	}
	patchSumsPtr[blockId] = scaleFactor * buffer[0];
	free(buffer);
}

at::Tensor radon2d_v2_cpu(const at::Tensor &a, double xSpacing, double ySpacing, long heightOut, long widthOut,
                          long samplesPerLine) {
	// a should be a 2D array of floats on the CPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCPU texture{aPtr, a.sizes()[1], a.sizes()[0], xSpacing, ySpacing};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());

	constexpr unsigned blockSize = 1024;
	const unsigned gridSize = (samplesPerLine + blockSize - 1) / blockSize;
	at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize}), result.options());
	float *patchSumsPtr = patchSums.data_ptr<float>();

	const float rayLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const auto constMappings = Radon2D<Texture2DCPU>::GetConstMappings(widthOut, heightOut, rayLength, samplesPerLine);
	const float scaleFactor = rayLength / static_cast<float>(samplesPerLine);
	for (long row = 0; row < heightOut; ++row) {
		for (long col = 0; col < widthOut; ++col) {
			const auto indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(texture, col, row, constMappings);
			for (size_t i = 0; i < gridSize; ++i) {
				radon_v2_kernel_synchronous(texture, blockSize, i, samplesPerLine, indexMappings, scaleFactor,
				                            patchSumsPtr);
			}
			result.index_put_({row, col}, patchSums.sum());
		}
	}

	return result;
}

} // namespace ExtensionTest