#include <torch/extension.h>

#include "../include/Radon3D.h"
#include "../include/Radon2D.h"

namespace ExtensionTest {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}

TORCH_LIBRARY(ExtensionTest, m) {
	// Note that "float" in the schema corresponds to the C++ `double` type and the Python `int` type.
	// Note that "int" in the schema corresponds to the C++ `long` type and the Python `int` type.
	m.def("radon2d(Tensor a, float xs, float ys, int b, int c, int d) -> Tensor");
	m.def("radon2d_v2(Tensor a, float xs, float ys, int b, int c, int d) -> Tensor");
	m.def("dRadon2dDR(Tensor a, float xs, float ys, int b, int c, int d) -> Tensor");
	m.def("radon3d(Tensor a, float xs, float ys, float zs, int b, int c, int d, int e) -> Tensor");
	m.def("radon3d_v2(Tensor a, float xs, float ys, float zs, int b, int c, int d, int e) -> Tensor");
}

TORCH_LIBRARY_IMPL(ExtensionTest, CPU, m) {
	m.impl("radon2d", &radon2d_cpu);
	m.impl("radon2d_v2", &radon2d_v2_cpu);
	m.impl("dRadon2dDR", &dRadon2dDR_cpu);
	m.impl("radon3d", &radon3d_cpu);
	m.impl("radon3d_v2", &radon3d_cpu);
}

TORCH_LIBRARY_IMPL(ExtensionTest, CUDA, m) {
	m.impl("radon2d", &radon2d_cuda);
	m.impl("radon2d_v2", &radon2d_v2_cuda);
	m.impl("dRadon2dDR", &dRadon2dDR_cuda);
	m.impl("radon3d", &radon3d_cuda);
	m.impl("radon3d_v2", &radon3d_v2_cuda);
}

} // namespace ExtensionTest