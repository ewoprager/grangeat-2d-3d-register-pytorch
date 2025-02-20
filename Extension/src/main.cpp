#include <torch/extension.h>

#include "../include/Radon3D.h"
#include "../include/Radon2D.h"
#include "../include/ResampleSinogram3D.h"
#include "../include/Similarity.h"

namespace ExtensionTest {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}

TORCH_LIBRARY(ExtensionTest, m) {
	// Note that "float" in the schema corresponds to the C++ `double` type and the Python `float` type.
	// Note that "int" in the schema corresponds to the C++ `long` type and the Python `int` type.
	m.def("radon2d(Tensor img, Tensor spacing, Tensor phis, Tensor rs, int sc) -> Tensor");
	m.def("radon2d_v2(Tensor img, Tensor spacing, Tensor phis, Tensor rs, int sc) -> Tensor");
	m.def("d_radon2d_dr(Tensor img, Tensor spacing, Tensor phis, Tensor rs, int sc) -> Tensor");
	m.def("radon3d(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def("radon3d_v2(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def("d_radon3d_dr(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def("d_radon3d_dr_v2(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def(
		"resample_sinogram3d(Tensor sinogram, Tensor spacing, Tensor centres, Tensor projMat, Tensor phis, Tensor rs) -> Tensor");
	m.def("normalised_cross_correlation(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(ExtensionTest, CPU, m) {
	m.impl("radon2d", &Radon2D_CPU);
	m.impl("radon2d_v2", &Radon2D_CPU); // doesn't have its own cpu version
	m.impl("d_radon2d_dr", &DRadon2DDR_CPU);
	m.impl("radon3d", &Radon3D_CPU);
	m.impl("radon3d_v2", &Radon3D_CPU); // doesn't have its own cpu version
	m.impl("d_radon3d_dr", &DRadon3DDR_CPU);
	m.impl("d_radon3d_dr_v2", &DRadon3DDR_CPU); // doesn't have its own cpu version
	m.impl("resample_sinogram3d", &ResampleSinogram3D_CPU);
	m.impl("normalised_cross_correlation", &NormalisedCrossCorrelation);
}

TORCH_LIBRARY_IMPL(ExtensionTest, CUDA, m) {
	m.impl("radon2d", &Radon2D_CUDA);
	m.impl("radon2d_v2", &Radon2D_CUDA_V2);
	m.impl("d_radon2d_dr", &DRadon2DDR_CUDA);
	m.impl("radon3d", &Radon3D_CUDA);
	m.impl("radon3d_v2", &Radon3D_CUDA_V2);
	m.impl("d_radon3d_dr", &DRadon3DDR_CUDA);
	m.impl("d_radon3d_dr_v2", &DRadon3DDR_CUDA_V2);
	m.impl("resample_sinogram3d", &ResampleSinogram3D_CUDA);
}

} // namespace ExtensionTest