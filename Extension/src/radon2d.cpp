#include <torch/extension.h>

#include "../include/Texture2DCPU.h"

namespace ExtensionTest {

at::Tensor radon2d_cpu(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                       const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the CPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CPU);
	// phiValues should be a 1D array of floats on the CPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	// rValues should be a 1D array of floats on the CPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCPU aTexture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const long heightOut = phiValues.sizes()[0];
	const long widthOut = rValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float lineLength = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld());
	const auto mappingIToOffset = Radon2D<Texture2DCPU>::GetMappingIToOffset(lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	for (long row = 0; row < heightOut; ++row) {
		for (long col = 0; col < widthOut; ++col) {
			const auto indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(
				aTexture, phiValues[row].item().toFloat(), rValues[col].item().toFloat(), mappingIToOffset);
			resultPtr[row * widthOut + col] = scaleFactor * Radon2D<Texture2DCPU>::IntegrateLooped(
				                                  aTexture, indexMappings, samplesPerLine);
		}
	}

	return result;
}

// void radon_v2_kernel_synchronous(const Texture2DCPU &textureIn, size_t blockSize, size_t blockId, long samplesPerLine,
//                                  const Radon2D<Texture2DCPU>::IndexMappings indexMappings, float scaleFactor,
//                                  float *patchSumsPtr) {
// 	float *buffer = static_cast<float *>(malloc(blockSize * sizeof(float)));
//
// 	for (size_t j = 0; j < blockSize; ++j) {
// 		const int i = blockSize * blockId + j;
// 		if (i >= samplesPerLine) {
// 			buffer[j] = 0.f;
// 			break;
// 		}
// 		const float iF = static_cast<float>(i);
// 		buffer[j] = textureIn.Sample(indexMappings.mappingIToX(iF), indexMappings.mappingIToY(iF));
// 	}
//
// 	for (long cutoff = blockSize / 2; cutoff > 0; cutoff /= 2) {
// 		for (int i = 0; i < cutoff; ++i) {
// 			buffer[i] += buffer[i + cutoff];
// 		}
// 	}
// 	patchSumsPtr[blockId] = scaleFactor * buffer[0];
// 	free(buffer);
// }
//
// at::Tensor radon2d_v2_cpu(const at::Tensor &image, double xSpacing, double ySpacing, long heightOut, long widthOut,
//                           long samplesPerLine) {
// 	// image should be a 2D array of floats on the CPU
// 	TORCH_CHECK(image.sizes().size() == 2);
// 	TORCH_CHECK(image.dtype() == at::kFloat);
// 	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CPU);
//
// 	const at::Tensor aContiguous = image.contiguous();
// 	const float *aPtr = aContiguous.data_ptr<float>();
// 	const Texture2DCPU texture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};
//
// 	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
//
// 	constexpr unsigned blockSize = 256;
// 	const unsigned gridSize = (samplesPerLine + blockSize - 1) / blockSize;
// 	at::Tensor patchSums = torch::zeros(at::IntArrayRef({gridSize}), result.options());
// 	float *patchSumsPtr = patchSums.data_ptr<float>();
//
// 	const float lineLength = sqrtf(
// 		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
// 	const auto constMappings = Radon2D<Texture2DCPU>::GetConstMappings(widthOut, heightOut, lineLength, samplesPerLine);
// 	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
// 	for (long row = 0; row < heightOut; ++row) {
// 		for (long col = 0; col < widthOut; ++col) {
// 			const auto indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(texture, col, row, constMappings);
// 			for (size_t i = 0; i < gridSize; ++i) {
// 				radon_v2_kernel_synchronous(texture, blockSize, i, samplesPerLine, indexMappings, scaleFactor,
// 				                            patchSumsPtr);
// 			}
// 			result.index_put_({row, col}, patchSums.sum());
// 		}
// 	}
//
// 	return result;
// }

at::Tensor dRadon2dDR_cpu(const at::Tensor &image, double xSpacing, double ySpacing, const at::Tensor &phiValues,
                          const at::Tensor &rValues, long samplesPerLine) {
	// image should be a 2D array of floats on the CPU
	TORCH_CHECK(image.sizes().size() == 2);
	TORCH_CHECK(image.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(image.device().type() == at::DeviceType::CPU);
	// phiValues should be a 1D array of floats on the CPU
	TORCH_CHECK(phiValues.sizes().size() == 1);
	TORCH_CHECK(phiValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(phiValues.device().type() == at::DeviceType::CPU);
	// rValues should be a 1D array of floats on the CPU
	TORCH_CHECK(rValues.sizes().size() == 1);
	TORCH_CHECK(rValues.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(rValues.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = image.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCPU aTexture{aPtr, image.sizes()[1], image.sizes()[0], xSpacing, ySpacing};

	const long widthOut = rValues.sizes()[0];
	const long heightOut = phiValues.sizes()[0];
	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float lineLength = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld());
	const auto constMappings = Radon2D<Texture2DCPU>::GetMappingIToOffset(lineLength, samplesPerLine);
	const float scaleFactor = lineLength / static_cast<float>(samplesPerLine);
	for (long row = 0; row < heightOut; ++row) {
		for (long col = 0; col < widthOut; ++col) {
			const float phi = phiValues[row].item().toFloat();
			const float r = rValues[col].item().toFloat();
			const auto indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(aTexture, phi, r, constMappings);
			const auto derivativeWRTR = Radon2D<Texture2DCPU>::GetDerivativeWRTR(aTexture, phi, r);
			resultPtr[row * widthOut + col] = scaleFactor * Radon2D<Texture2DCPU>::DIntegrateLoopedDMappingParameter(
				                                  aTexture, indexMappings, derivativeWRTR, samplesPerLine);
		}
	}

	return result;
}

} // namespace ExtensionTest