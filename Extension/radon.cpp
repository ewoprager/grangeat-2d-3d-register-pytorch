#include <torch/extension.h>

#include "include/Texture2DCPU.h"

namespace ExtensionTest {

at::Tensor radon2d_cpu(const at::Tensor &a, long heightOut, long widthOut, long samplesPerLine) {
	// a should be a 2D array of floats on the CPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	const Texture2DCPU aTexture{aPtr, a.sizes()[1], a.sizes()[0], 1.f, 1.f};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float rayLength = sqrtf(
		aTexture.WidthWorld() * aTexture.WidthWorld() + aTexture.HeightWorld() * aTexture.HeightWorld());
	const auto constMappings = Radon2D<Texture2DCPU>::GetConstMappings(widthOut, heightOut, rayLength, samplesPerLine);
	for (long row = 0; row < heightOut; ++row) {
		for (long col = 0; col < widthOut; ++col) {
			const Radon2D<Texture2DCPU>::IndexMappings indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(
				aTexture, col, row, constMappings);
			resultPtr[row * widthOut + col] = Radon2D<Texture2DCPU>::IntegrateLooped(
				aTexture, indexMappings, samplesPerLine);
		}
	}

	return result;
}

void radon_v2_kernel_synchronous(const Texture2DCPU &textureIn, long samplesPerLine,
                                 const Radon2D<Texture2DCPU>::IndexMappings indexMappings, float *ret) {
	float *buffer = static_cast<float *>(malloc(samplesPerLine * sizeof(float)));

	for (int i = 0; i < samplesPerLine; ++i) {
		const float iF = static_cast<float>(i);
		buffer[i] = textureIn.Sample(indexMappings.mappingIToX(iF), indexMappings.mappingIToY(iF));
	}

	for (long cutoff = samplesPerLine / 2; cutoff > 0; cutoff /= 2) {
		for (int i = 0; i < cutoff; ++i) {
			buffer[i] += buffer[i + cutoff];
		}
	}
	*ret = buffer[0];
	free(buffer);
}

at::Tensor radon2d_v2_cpu(const at::Tensor &a, long heightOut, long widthOut, long samplesPerLine) {
	// samplesPerLine should be no more than 1024
	TORCH_CHECK(samplesPerLine <= 1024);

	// a should be a 2D array of floats on the CPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);

	const at::Tensor aContiguous = a.contiguous();
	const float *aPtr = aContiguous.data_ptr<float>();
	Texture2DCPU texture{aPtr, a.sizes()[1], a.sizes()[0], 1.f, 1.f};

	at::Tensor result = torch::zeros(at::IntArrayRef({heightOut, widthOut}), aContiguous.options());
	float *resultPtr = result.data_ptr<float>();

	const float rayLength = sqrtf(
		texture.WidthWorld() * texture.WidthWorld() + texture.HeightWorld() * texture.HeightWorld());
	const auto constMappings = Radon2D<Texture2DCPU>::GetConstMappings(widthOut, heightOut, rayLength, samplesPerLine);
	for (long row = 0; row < heightOut; ++row) {
		for (long col = 0; col < widthOut; ++col) {
			const auto indexMappings = Radon2D<Texture2DCPU>::GetIndexMappings(texture, col, row, constMappings);
			radon_v2_kernel_synchronous(texture, samplesPerLine, indexMappings, &resultPtr[row * widthOut + col]);
		}
	}

	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}

TORCH_LIBRARY(ExtensionTest, m) {
	// Note that "float" in the schema corresponds to the C++ `double` type and the Python `int` type.
	// Note that "int" in the schema corresponds to the C++ `long` type and the Python `int` type.
	m.def("radon2d(Tensor a, int b, int c, int d) -> Tensor");
	m.def("radon2d_v2(Tensor a, int b, int c, int d) -> Tensor");
}

TORCH_LIBRARY_IMPL(ExtensionTest, CPU, m) {
	m.impl("radon2d", &radon2d_cpu);
	m.impl("radon2d_v2", &radon2d_v2_cpu);
}

} // namespace ExtensionTest