#include <torch/extension.h>

#include "../include/Texture.h"
#include "../include/CUDATexture.h"

namespace reg23 {

int64_t CUDATexture2D::Handle() const {
	return textureHandle;
}

int64_t CUDATexture3D::Handle() const {
	return textureHandle;
}

CUDATexture2D::CUDATexture2D(const at::Tensor &tensor, const std::string &addressModeX,
                             const std::string &addressModeY) {
	// tensor should be a 2-dimensional array of floats on the GPU
	TORCH_CHECK(tensor.sizes().size() == 2);
	TORCH_CHECK(tensor.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(tensor.device().type() == at::DeviceType::CUDA);

	const float *const data = tensor.contiguous().data_ptr<float>();

	size = Vec<int64_t, 2>::FromIntArrayRef(tensor.sizes()).Flipped();

	// All addressMode<dim>s should be one of the valid values:
	Vec<TextureAddressMode, 2> addressModes = Vec<TextureAddressMode, 2>::Full(TextureAddressMode::ZERO);
	int dim = 0;
	for (const std::string_view &str : std::array<std::string_view, 2>{{addressModeX, addressModeY}}) {
		if (str == "wrap") {
			addressModes[dim] = TextureAddressMode::WRAP;
		} else if (str != "zero") {
			TORCH_WARN(
				"Invalid address mode string given. Valid values are: 'zero', 'wrap'. Using default value: 'zero'.")
		}
		++dim;
	}

	// Copy the given data into a CUDA array
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&arrayHandle, &channelDesc, size.X(), size.Y());
	cudaMemcpy2DToArray(arrayHandle, 0, 0, data, size.X() * sizeof(float), size.X() * sizeof(float), size.Y(),
	                    cudaMemcpyHostToDevice);

	// Create the texture object from the CUDA array
	const cudaResourceDesc resourceDescriptor = {.resType = cudaResourceTypeArray,
	                                             .res = {.array = {.array = arrayHandle}}};
	cudaTextureDesc textureDescriptor = {.filterMode = cudaFilterModeLinear, .readMode = cudaReadModeElementType,
	                                     .borderColor = {0.f, 0.f, 0.f, 0.f}, .normalizedCoords = true};
	for (int i = 0; i < 2; ++i) {
		textureDescriptor.addressMode[i] = TextureAddressModeToCuda(addressModes[i]);
	}
	cudaCreateTextureObject(&textureHandle, &resourceDescriptor, &textureDescriptor, nullptr);
}

CUDATexture3D::CUDATexture3D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY,
                             const std::string &addressModeZ) {
	// tensor should be a 3-dimensional array of floats on the GPU
	TORCH_CHECK(tensor.sizes().size() == 3);
	TORCH_CHECK(tensor.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(tensor.device().type() == at::DeviceType::CUDA);

	const float *const data = tensor.contiguous().data_ptr<float>();

	size = Vec<int64_t, 3>::FromIntArrayRef(tensor.sizes()).Flipped();

	// All addressMode<dim>s should be one of the valid values:
	Vec<TextureAddressMode, 3> addressModes = Vec<TextureAddressMode, 3>::Full(TextureAddressMode::ZERO);
	int dim = 0;
	for (const std::string_view &str : std::array<std::string_view, 3>{{addressModeX, addressModeY, addressModeZ}}) {
		if (str == "wrap") {
			addressModes[dim] = TextureAddressMode::WRAP;
		} else if (str != "zero") {
			TORCH_WARN(
				"Invalid address mode string given. Valid values are: 'zero', 'wrap'. Using default value: 'zero'.")
		}
		++dim;
	}

	const cudaExtent extent = {.width = static_cast<size_t>(size.X()), .height = static_cast<size_t>(size.Y()),
	                           .depth = static_cast<size_t>(size.Z())};

	// Copy the given data into a CUDA array
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	// cudaMallocArray(&arrayHandle, &channelDesc, _width, _height);
	cudaMalloc3DArray(&arrayHandle, &channelDesc, extent);
	// cudaMemcpy2DToArray(arrayHandle, 0, 0, data, _width * sizeof(float), _width * sizeof(float), _height,
	//                     cudaMemcpyHostToDevice);

	const cudaMemcpy3DParms params = {
		.srcPtr = make_cudaPitchedPtr((void *)data, size.X() * sizeof(float), size.X(), size.Y()),
		.dstArray = arrayHandle, .extent = extent, .kind = cudaMemcpyHostToDevice};
	cudaMemcpy3D(&params);

	// Create the texture object from the CUDA array
	const cudaResourceDesc resourceDescriptor = {.resType = cudaResourceTypeArray,
	                                             .res = {.array = {.array = arrayHandle}}};
	cudaTextureDesc textureDescriptor = {.filterMode = cudaFilterModeLinear, .readMode = cudaReadModeElementType,
	                                     .borderColor = {0.f, 0.f, 0.f, 0.f}, .normalizedCoords = true};
	for (int i = 0; i < 3; ++i) {
		textureDescriptor.addressMode[i] = TextureAddressModeToCuda(addressModes[i]);
	}
	cudaCreateTextureObject(&textureHandle, &resourceDescriptor, &textureDescriptor, nullptr);
}

} // namespace reg23