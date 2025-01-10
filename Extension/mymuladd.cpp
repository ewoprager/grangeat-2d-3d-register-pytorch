#include <cmath>
#include <numbers>

#include <torch/extension.h>

namespace ExtensionTest {

at::Tensor mymuladd_cpu(const at::Tensor &a, const at::Tensor &b, double c) {
	TORCH_CHECK(a.sizes() == b.sizes());
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_CHECK(b.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
	const at::Tensor aContiguous = a.contiguous();
	const at::Tensor bContiguous = b.contiguous();
	const at::Tensor result = torch::empty(aContiguous.sizes(), aContiguous.options());
	const float *a_ptr = aContiguous.data_ptr<float>();
	const float *b_ptr = bContiguous.data_ptr<float>();
	float *result_ptr = result.data_ptr<float>();
	for (int64_t i = 0; i < result.numel(); i++) {
		result_ptr[i] = a_ptr[i] * b_ptr[i] + static_cast<float>(c);
	}
	return result;
}

struct Texture2D {
	const float *ptr{};
	int64_t height{};
	int64_t width{};
	float ySpacing = 1.f;
	float xSpacing = 1.f;

	[[nodiscard]] float SizeYWorld() const { return static_cast<float>(height) * ySpacing; }
	[[nodiscard]] float SizeXWorld() const { return static_cast<float>(width) * xSpacing; }
	[[nodiscard]] float WorldToImageY(float yWorld) const { return .5f - yWorld / SizeYWorld(); }
	[[nodiscard]] float WorldToImageX(float xWorld) const { return .5f + xWorld / SizeXWorld(); }

	[[nodiscard]] bool In(int row, int col) const {
		return row >= 0 && row < height && col >= 0 && col < width;
	}

	[[nodiscard]] float At(int row, int col) const {
		return In(row, col) ? ptr[row * width + col] : 0.0f;
	}

	[[nodiscard]] bool InWorld(float y, float x) const {
		return y >= -.5f * SizeYWorld() && y < .5f * SizeYWorld() && x >= -.5f * SizeXWorld() && x < .5f * SizeXWorld();
	}

	[[nodiscard]] float SampleBilinear(float y, float x) const {
		const float yUnnormalised = y * static_cast<float>(height - 1);
		const float xUnnormalised = x * static_cast<float>(width - 1);
		const int row = static_cast<int>(std::floor(yUnnormalised));
		const int col = static_cast<int>(std::floor(xUnnormalised));
		const float fVertical = yUnnormalised - static_cast<float>(row);
		const float fHorizontal = xUnnormalised - static_cast<float>(col);
		const float r0 = (1.f - fHorizontal) * At(row, col) + fHorizontal * At(row, col + 1);
		const float r1 = (1.f - fHorizontal) * At(row + 1, col) + fHorizontal * At(row + 1, col + 1);
		return (1.f - fVertical) * r0 + fVertical * r1;
	}

	[[nodiscard]] float IntegrateRay(float phi, float r, float spacing = 1.f) const {
		const float yCentreWorld = r * std::sin(phi);
		const float xCentreWorld = r * std::cos(phi);
		const float yDeltaWorld = spacing * std::cos(phi);
		const float xDeltaWorld = -spacing * std::sin(phi);
		float ret = 0.f;
		int i = 0;
		while (true) {
			const float y = yCentreWorld + static_cast<float>(i) * yDeltaWorld;
			const float x = xCentreWorld + static_cast<float>(i) * xDeltaWorld;
			if (!InWorld(y, x)) break;
			ret += SampleBilinear(WorldToImageY(y), WorldToImageX(x));
			++i;
		}
		i = 1;
		while (true) {
			const float y = yCentreWorld - static_cast<float>(i) * yDeltaWorld;
			const float x = xCentreWorld - static_cast<float>(i) * xDeltaWorld;
			if (!InWorld(y, x)) break;
			ret += SampleBilinear(WorldToImageY(y), WorldToImageX(x));
			++i;
		}
		return ret;
	}
};

at::Tensor radon2d_cpu(const at::Tensor &a, const at::Tensor &outputDims) {
	// a should be a 2D array of floats on the CPU
	TORCH_CHECK(a.sizes().size() == 2);
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
	// outputDims should be a 1D array of length 2 of ints on the CPU
	TORCH_CHECK(outputDims.sizes().size() == 1);
	TORCH_CHECK(outputDims.sizes()[0] == 2);
	TORCH_CHECK(outputDims.dtype() == at::kInt);
	TORCH_INTERNAL_ASSERT(outputDims.device().type() == at::DeviceType::CPU);

	at::Tensor aContiguous = a.contiguous();
	const float *a_ptr = aContiguous.data_ptr<float>();
	const Texture2D aTexture{a_ptr, a.sizes()[0], a.sizes()[1]};

	at::Tensor outputDims_contig = outputDims.contiguous();
	const int *outputDims_ptr = outputDims_contig.data_ptr<int>();

	at::Tensor result = torch::zeros(at::IntArrayRef({outputDims_ptr[0], outputDims_ptr[1]}), aContiguous.options());
	float *result_ptr = result.data_ptr<float>();

	const float rayLength = std::sqrt(
		aTexture.SizeXWorld() * aTexture.SizeXWorld() + aTexture.SizeYWorld() * aTexture.SizeYWorld());
	for (int row = 0; row < outputDims_ptr[0]; ++row) {
		for (int col = 0; col < outputDims_ptr[1]; ++col) {
			result_ptr[row * outputDims_ptr[1] + col] = aTexture.IntegrateRay(
				std::numbers::pi_v<float> * (-.5f + static_cast<float>(row) / static_cast<float>(outputDims_ptr[0])),
				rayLength * (-.5f + static_cast<float>(col) / static_cast<float>(outputDims_ptr[1] - 1)), .1f);
		}
	}

	return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}

// Define the operator
TORCH_LIBRARY(ExtensionTest, m) {
	// Note that "float" in the schema corresponds to the C++ double type
	// and the Python float type.
	m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
	m.def("radon2d(Tensor a, Tensor b) -> Tensor");
}

// Register the implementation
TORCH_LIBRARY_IMPL(ExtensionTest, CPU, m) {
	m.impl("mymuladd", &mymuladd_cpu);
	m.impl("radon2d", &radon2d_cpu);
}

} // namespace ExtensionTest