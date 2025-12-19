/**
 * @file
 * @brief PyTorch binding generation
 */



#include <reg23/GridSample3D.h>
#include <reg23/ProjectDRR.h>
#include <reg23/ProjectDRRCuboidMaskCPU.h>
#include <reg23/Radon2D.h>
#include <reg23/Radon3D.h>
#include <reg23/ResampleSinogram3D.h>
#include <reg23/Similarity.h>

#ifdef USE_CUDA
#include <reg23/CUDATexture.h>
#endif

#ifdef USE_MPS
#include <reg23/MPSTexture.h>
#endif

namespace reg23 {

/**
 * @defgroup pytorch_functions PyTorch Functions
 * @brief Functions that have Python bindings generated and are accessible in the PyTorch extension.
 * @{
 * @}
 */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef USE_CUDA
	py::class_<CUDATexture2D>(m, "CUDATexture2DInternal")							   //
		.def(py::init<const at::Tensor &, const std::string &, const std::string &>()) //
		.def("handle", &CUDATexture2D::Handle)										   //
		.def("size", &CUDATexture2D::SizeTensor)									   //
		.def("clean_up", &CUDATexture2D::CleanUp);
	py::class_<CUDATexture3D>(m, "CUDATexture3DInternal")													//
		.def(py::init<const at::Tensor &, const std::string &, const std::string &, const std::string &>()) //
		.def("handle", &CUDATexture3D::Handle)																//
		.def("size", &CUDATexture3D::SizeTensor)															//
		.def("clean_up", &CUDATexture3D::CleanUp);
#endif
//#ifdef USE_MPS
//	py::class_<MPSTexture3D>(m, "MPSTexture3DInternal")														//
//		.def(py::init<const at::Tensor &, const std::string &, const std::string &, const std::string &>()) //
//		.def("handle", &MPSTexture3D::HandleAsInt)															//
//		.def("size", &MPSTexture3D::SizeTensor)																//
//		.def("clean_up", &MPSTexture3D::CleanUp);
//#endif
}

TORCH_LIBRARY(reg23, m) {
	// Note that "float" in the schema corresponds to the C++ `double` type and the Python `float` type.
	// Note that "int" in the schema corresponds to the C++ `int64_t` type and the Python `int` type.
	//	m.def("sample_test(Tensor a) -> Tensor");
	m.def("add_tensors_metal(Tensor a, Tensor b) -> Tensor");
	m.def("radon2d(Tensor img, Tensor spacing, Tensor phis, Tensor rs, int sc) -> Tensor");
	m.def("radon2d_v2(Tensor img, Tensor spacing, Tensor phis, Tensor rs, int sc) -> Tensor");
	m.def("d_radon2d_dr(Tensor img, Tensor spacing, Tensor phis, Tensor rs, int sc) -> Tensor");
	m.def("radon3d(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def("radon3d_v2(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def("d_radon3d_dr(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def("d_radon3d_dr_v2(Tensor vol, Tensor spacing, Tensor phis, Tensor thetas, Tensor rs, int sc) -> Tensor");
	m.def("resample_sinogram3d(Tensor sinogram, str type, float rSpacing, Tensor projMat, Tensor phis, Tensor "
		  "rs, Tensor? out=None) -> Tensor");
	m.def("resample_sinogram3d_cuda_texture(int handle, int w, int h, int d, str type, float rSpacing, Tensor projMat, "
		  "Tensor phis, Tensor rs, Tensor? out=None) -> Tensor");
	m.def("normalised_cross_correlation(Tensor a, Tensor b) -> (Tensor, float, float, float, float, float)");
	m.def("grid_sample3d(Tensor input, Tensor grid, str am_x, str am_y, str am_z, Tensor? out=None) -> Tensor");
	m.def("project_drr(Tensor volume, Tensor spacing, Tensor hi, float sourceDist, int outW, int outH, Tensor outOff, "
		  "Tensor outSpacing) -> Tensor");
	m.def("project_drr_backward(Tensor volume, Tensor spacing, Tensor hi, float sourceDist, int outW, int outH, Tensor "
		  "outOff, Tensor outSpacing, Tensor dLossDDRR) -> Tensor");
	m.def("project_drr_cuboid_mask(Tensor vSize, Tensor spacing, Tensor hi, float sourceDist, int outW, int outH, "
		  "Tensor outOff, Tensor outSpacing) -> Tensor");
}

TORCH_LIBRARY_IMPL(reg23, CPU, m) {
	m.impl("radon2d", &Radon2D_CPU);
	m.impl("radon2d_v2", &Radon2D_CPU); // doesn't have its own cpu version
	m.impl("d_radon2d_dr", &DRadon2DDR_CPU);
	m.impl("radon3d", &Radon3D_CPU);
	m.impl("radon3d_v2", &Radon3D_CPU); // doesn't have its own cpu version
	m.impl("d_radon3d_dr", &DRadon3DDR_CPU);
	m.impl("d_radon3d_dr_v2", &DRadon3DDR_CPU); // doesn't have its own cpu version
	m.impl("resample_sinogram3d", &ResampleSinogram3D_CPU);
	m.impl("normalised_cross_correlation", &NormalisedCrossCorrelation_CPU);
	m.impl("grid_sample3d", &GridSample3D_CPU);
	m.impl("project_drr", &ProjectDRR_CPU);
	m.impl("project_drr_backward", &ProjectDRR_backward_CPU);
	m.impl("project_drr_cuboid_mask", &ProjectDRRCuboidMask_CPU);
}

#ifdef USE_CUDA
TORCH_LIBRARY_IMPL(reg23, CUDA, m) {
	m.impl("radon2d", &Radon2D_CUDA);
	m.impl("radon2d_v2", &Radon2D_CUDA_V2);
	m.impl("d_radon2d_dr", &DRadon2DDR_CUDA);
	m.impl("radon3d", &Radon3D_CUDA);
	m.impl("radon3d_v2", &Radon3D_CUDA_V2);
	m.impl("d_radon3d_dr", &DRadon3DDR_CUDA);
	m.impl("d_radon3d_dr_v2", &DRadon3DDR_CUDA_V2);
	m.impl("resample_sinogram3d", &ResampleSinogram3D_CUDA);
	m.impl("resample_sinogram3d_cuda_texture", &ResampleSinogram3DCUDATexture);
	m.impl("normalised_cross_correlation", &NormalisedCrossCorrelation_CUDA);
	m.impl("grid_sample3d", &GridSample3D_CUDA);
	m.impl("project_drr", &ProjectDRR_CUDA);
	m.impl("project_drr_backward", &ProjectDRR_backward_CUDA);
	m.impl("project_drr_cuboid_mask", &ProjectDRRCuboidMask_CUDA);
}
#endif

#ifdef USE_MPS
TORCH_LIBRARY_IMPL(reg23, MPS, m) {
	m.impl("add_tensors_metal", &add_tensors_metal);
	m.impl("project_drr", &ProjectDRR_MPS);
}
#endif

} // namespace reg23
