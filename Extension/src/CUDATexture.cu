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

at::Tensor CUDATexture2D::SizeTensor() const {
	return at::tensor(size, at::dtype(at::kInt));
}

at::Tensor CUDATexture3D::SizeTensor() const {
	return at::tensor(size, at::dtype(at::kInt));
}

CUDATexture2D::CUDATexture2D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY)
	: CUDATexture2D(tensor, StringsToAddressModes<2>({{addressModeX, addressModeY}})) {
}

CUDATexture2D::CUDATexture2D(const at::Tensor &tensor, Vec<TextureAddressMode, 2> addressModes) {
	cudaError_t err;

	// tensor should be a 2-dimensional array of floats on the GPU
	TORCH_CHECK(tensor.sizes().size() == 2);
	TORCH_CHECK(tensor.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(tensor.device().type() == at::DeviceType::CUDA);

	const at::Tensor tensorContiguous = tensor.contiguous();

	const float *const data = tensorContiguous.data_ptr<float>();

	size = Vec<int64_t, 2>::FromIntArrayRef(tensorContiguous.sizes()).Flipped();

	// Copy the given data into a CUDA array
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	err = cudaMallocArray(&arrayHandle, &channelDesc, size.X(), size.Y());
	if (err != cudaSuccess) {
		std::cerr << "cudaMallocArray failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaMallocArray failed");
	}

	err = cudaMemcpy2DToArray(arrayHandle, 0, 0, data, size.X() * sizeof(float), size.X() * sizeof(float), size.Y(),
	                          cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy2DToArray failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaMemcpy2DToArray failed");
	}

	cudaDeviceSynchronize(); // Ensure the copy is finished before tensorContiguous is destroyed

	// Create the texture object from the CUDA array
	const cudaResourceDesc resourceDescriptor = {.resType = cudaResourceTypeArray,
	                                             .res = {.array = {.array = arrayHandle}}};
	cudaTextureDesc textureDescriptor = {.filterMode = cudaFilterModeLinear, .readMode = cudaReadModeElementType,
	                                     .borderColor = {0.f, 0.f, 0.f, 0.f}, .normalizedCoords = true};
	for (int i = 0; i < 2; ++i) {
		textureDescriptor.addressMode[i] = TextureAddressModeToCuda(addressModes[i]);
	}
	err = cudaCreateTextureObject(&textureHandle, &resourceDescriptor, &textureDescriptor, nullptr);
	if (err != cudaSuccess) {
		std::cerr << "cudaCreateTextureObject failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaCreateTextureObject failed");
	}

	cudaDeviceSynchronize(); // Ensure the copy is finished before tensorContiguous is destroyed
}

CUDATexture3D::CUDATexture3D(const at::Tensor &tensor, const std::string &addressModeX, const std::string &addressModeY,
                             const std::string &addressModeZ) : CUDATexture3D(
	tensor, StringsToAddressModes<3>({{addressModeX, addressModeY, addressModeZ}})) {
}

CUDATexture3D::CUDATexture3D(const at::Tensor &tensor, Vec<TextureAddressMode, 3> addressModes) {
	cudaError_t err;

	// tensor should be a 3-dimensional array of floats on the GPU
	TORCH_CHECK(tensor.sizes().size() == 3);
	TORCH_CHECK(tensor.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(tensor.device().type() == at::DeviceType::CUDA);

	at::Tensor tensorContiguous = tensor.contiguous();

	const float *const data = tensorContiguous.data_ptr<float>();
	std::cout << "Hello, world0!\n";

	size = Vec<int64_t, 3>::FromIntArrayRef(tensorContiguous.sizes()).Flipped();
	std::cout << "Hello, world0.1!\n";

	cudaExtent extent;
	extent.width = static_cast<size_t>(size.X());
	extent.height = static_cast<size_t>(size.Y());
	extent.depth = static_cast<size_t>(size.Z());

	std::cout << "Hello, world0.2!\n";

	// Copy the given data into a CUDA array
	cudaChannelFormatDesc channelDesc = {};
	channelDesc.f = cudaChannelFormatKindFloat;
	channelDesc.x = (int)sizeof(float) * 8;
	channelDesc.y = 0;
	channelDesc.z = 0;
	channelDesc.w = 0;
	std::cout << "Hello, world0.3!\n";

	err = cudaMalloc3DArray(&arrayHandle, &channelDesc, extent);
	if (err != cudaSuccess) {
		std::cerr << "cudaMalloc3DArray failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaMalloc3DArray failed");
	}
	std::cout << "Hello, world0.4!\n";

	cudaDeviceSynchronize(); // Ensure the copy is finished before tensorContiguous is destroyed

	std::cout << "Hello, world1!\n";

	cudaMemcpy3DParms params;
	params.srcPtr = make_cudaPitchedPtr((void *)data, size.X() * sizeof(float), size.X(), size.Y());
	params.dstArray = arrayHandle;
	params.extent = extent;
	params.kind = cudaMemcpyDeviceToDevice;

	err = cudaMemcpy3D(&params);
	if (err != cudaSuccess) {
		std::cerr << "cudaMemcpy3D failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaMemcpy3D failed");
	}
	std::cout << "Hello, world2!\n";

	cudaDeviceSynchronize(); // Ensure the copy is finished before tensorContiguous is destroyed

	// Create the texture object from the CUDA array
	cudaResourceDesc resourceDescriptor;
	resourceDescriptor.resType = cudaResourceTypeArray;
	resourceDescriptor.res = {.array = {.array = arrayHandle}};

	cudaTextureDesc textureDescriptor;
	textureDescriptor.filterMode = cudaFilterModeLinear;
	textureDescriptor.readMode = cudaReadModeElementType;
	textureDescriptor.borderColor[0] = 0.f;
	textureDescriptor.borderColor[1] = 0.f;
	textureDescriptor.borderColor[2] = 0.f;
	textureDescriptor.borderColor[3] = 0.f;
	textureDescriptor.normalizedCoords = true;
	for (int i = 0; i < 3; ++i) {
		textureDescriptor.addressMode[i] = TextureAddressModeToCuda(addressModes[i]);
	}

	err = cudaCreateTextureObject(&textureHandle, &resourceDescriptor, &textureDescriptor, nullptr);
	if (err != cudaSuccess) {
		std::cerr << "cudaCreateTextureObject failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaCreateTextureObject failed");
	}
	std::cout << "Hello, world3!\n";

	cudaDeviceSynchronize(); // Ensure the copy is finished before tensorContiguous is destroyed

	std::cout << "Hello, world4!\n";
}

} // namespace reg23