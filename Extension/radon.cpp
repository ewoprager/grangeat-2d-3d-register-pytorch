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
	const Linear mappingIToOffset{-.5f * rayLength, rayLength / static_cast<float>(samplesPerLine - 1)};
	for (int row = 0; row < heightOut; ++row) {
		for (int col = 0; col < widthOut; ++col) {
			resultPtr[row * widthOut + col] = aTexture.IntegrateRay(
				3.1415926535f * (-.5f + static_cast<float>(row) / static_cast<float>(heightOut)),
				rayLength * (-.5f + static_cast<float>(col) / static_cast<float>(widthOut - 1)), mappingIToOffset,
				samplesPerLine);
		}
	}

	return result;
}

void radon_v2_kernel_synchronous(const Texture2DCPU &textureIn, long samplesPerLine, const Linear mappingIToX,
                                 const Linear mappingIToY, float *ret) {
	float *buffer = static_cast<float *>(malloc(samplesPerLine * sizeof(float)));

	for (int i = 0; i < samplesPerLine; ++i) {
		const float iF = static_cast<float>(i);
		buffer[i] = textureIn.SampleBilinear(mappingIToX(iF), mappingIToY(iF));
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
	const Linear mappingIToOffset{-.5f * rayLength, rayLength / static_cast<float>(samplesPerLine - 1)};
	for (unsigned row = 0; row < heightOut; ++row) {
		for (unsigned col = 0; col < widthOut; ++col) {
			const float r = rayLength * (-.5f + static_cast<float>(col) / static_cast<float>(widthOut - 1));

			const float phi = 3.1415926535f * (-.5f + static_cast<float>(row) / static_cast<float>(heightOut));
			const float c = cosf(phi);
			const float s = sinf(phi);

			const Linear mappingOffsetToWorldX{r * c, -s};
			const Linear mappingOffsetToWorldY{r * s, c};
			const Linear mappingIToX = texture.MappingXWorldToNormalised()(mappingOffsetToWorldX(mappingIToOffset));
			const Linear mappingIToY = texture.MappingYWorldToNormalised()(mappingOffsetToWorldY(mappingIToOffset));

			radon_v2_kernel_synchronous(texture, samplesPerLine, mappingIToX, mappingIToY,
			                            &resultPtr[row * widthOut + col]);
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