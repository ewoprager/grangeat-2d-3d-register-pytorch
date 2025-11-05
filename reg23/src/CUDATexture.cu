#include <torch/extension.h>

#include "../include/Texture.h"
#include "../include/CUDATexture.h"

namespace reg23 {

unsigned long long CUDATexture2D::Handle() const {
	return static_cast<unsigned long long>(textureHandle);
}

unsigned long long CUDATexture3D::Handle() const {
	return static_cast<unsigned long long>(textureHandle);
}

at::Tensor CUDATexture2D::SizeTensor() const {
	return at::tensor(backingTensor.sizes(), at::dtype(at::kInt)).flip({0});
}

at::Tensor CUDATexture3D::SizeTensor() const {
	return at::tensor(backingTensor.sizes(), at::dtype(at::kInt)).flip({0});
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

	backingTensor = tensor.contiguous();

	const float *const data = backingTensor.data_ptr<float>();

	const Vec<int64_t, 2> size = Vec<int64_t, 2>::FromIntArrayRef(backingTensor.sizes()).Flipped();

	// Copy the given data into a CUDA array
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	err = cudaMallocArray(&arrayHandle, &channelDesc, size.X(), size.Y());
	if (err != cudaSuccess) {
		std::cout << "cudaMallocArray failed: " << cudaGetErrorString(err) << std::endl;
		throw std::bad_alloc();
	}
#ifdef DEBUG
	std::cout << "[C++] Array allocated." << std::endl;
#endif

	err = cudaMemcpy2DToArray(arrayHandle, 0, 0, data, size.X() * sizeof(float), size.X() * sizeof(float), size.Y(),
	                          cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		std::cout << "cudaMemcpy2DToArray failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaMemcpy2DToArray failed");
	}

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
		std::cout << "cudaCreateTextureObject failed: " << cudaGetErrorString(err) << std::endl;
		throw std::runtime_error("cudaCreateTextureObject failed");
	}
#ifdef DEBUG
	std::cout << "[C++] Texture " << static_cast<uint64_t>(textureHandle) << " created." << std::endl;
#endif

	// Ensure the tensor is no longer being used by the device before anything else can happen to it
	cudaDeviceSynchronize();
}

void CUDATexture2D::CleanUp() noexcept {
#ifdef DEBUG
	std::cout << "[C++] CUDATexture2D cleaning up." << std::endl;
#endif
	cudaError_t err;
	if (textureHandle) {
		err = cudaDestroyTextureObject(textureHandle);
		if (err != cudaSuccess) {
			std::cout << "cudaDestroyTextureObject failed: " << cudaGetErrorString(err) << std::endl;
			std::terminate();
		}
#ifdef DEBUG
		std::cout << "[C++] CUDA texture " << static_cast<uint64_t>(textureHandle) << " destroyed." << std::endl;
#endif
	}
	if (arrayHandle) {
		err = cudaFreeArray(arrayHandle);
		if (err != cudaSuccess) {
			std::cout << "cudaFreeArray failed: " << cudaGetErrorString(err) << std::endl;
			std::terminate();
		}
#ifdef DEBUG
		std::cout << "[C++] CUDA array freed." << std::endl;
#endif
	}
	backingTensor.reset();
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

	backingTensor = tensor.contiguous();

	const float *const data = backingTensor.data_ptr<float>();

	const Vec<int64_t, 3> size = Vec<int64_t, 3>::FromIntArrayRef(backingTensor.sizes()).Flipped();

	cudaExtent extent{};
	extent.width = static_cast<size_t>(size.X());
	extent.height = static_cast<size_t>(size.Y());
	extent.depth = static_cast<size_t>(size.Z());

	// Copy the given data into a CUDA array
	cudaChannelFormatDesc channelDesc{};
	channelDesc.f = cudaChannelFormatKindFloat;
	channelDesc.x = (int)sizeof(float) * 8;
	channelDesc.y = 0;
	channelDesc.z = 0;
	channelDesc.w = 0;

	err = cudaMalloc3DArray(&arrayHandle, &channelDesc, extent);
	if (err != cudaSuccess) {
		std::cout << "cudaMalloc3DArray failed: " << cudaGetErrorString(err) << std::endl << std::flush;
		throw std::bad_alloc();
	}
#ifdef DEBUG
	std::cout << "[C++] Array allocated." << std::endl << std::flush;
#endif

	cudaMemcpy3DParms params{};
	params.srcPtr = make_cudaPitchedPtr((void *)data, size.X() * sizeof(float), size.X(), size.Y());
	params.dstArray = arrayHandle;
	params.extent = extent;
	params.kind = cudaMemcpyDeviceToDevice;

	err = cudaMemcpy3D(&params);
	if (err != cudaSuccess) {
		std::cout << "cudaMemcpy3D failed: " << cudaGetErrorString(err) << std::endl << std::flush;
		throw std::runtime_error("cudaMemcpy3D failed");
	}

	// Create the texture object from the CUDA array
	cudaResourceDesc resourceDescriptor{};
	resourceDescriptor.resType = cudaResourceTypeArray;
	resourceDescriptor.res = {.array = {.array = arrayHandle}};

	cudaTextureDesc textureDescriptor{};
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
		std::cout << "cudaCreateTextureObject failed: " << cudaGetErrorString(err) << std::endl << std::flush;
		throw std::runtime_error("cudaCreateTextureObject failed");
	}
#ifdef DEBUG
	std::cout << "[C++] Texture " << static_cast<uint64_t>(textureHandle) << " created." << std::endl << std::flush;
#endif

	// Ensure the tensor is no longer being used by the device before anything else can happen to it
	cudaDeviceSynchronize();
}

void CUDATexture3D::CleanUp() noexcept {
#ifdef DEBUG
	std::cout << "[C++] CUDATexture3D cleaning up." << std::endl << std::flush;
#endif
	cudaError_t err;
	if (textureHandle) {
		err = cudaDestroyTextureObject(textureHandle);
		if (err != cudaSuccess) {
			std::cerr << "cudaDestroyTextureObject failed: " << cudaGetErrorString(err) << std::endl << std::flush;
			std::terminate();
		}
#ifdef DEBUG
		std::cout << "[C++] CUDA texture " << static_cast<uint64_t>(textureHandle) << " destroyed." << std::endl <<
			std::flush;
#endif
	}
	if (arrayHandle) {
		err = cudaFreeArray(arrayHandle);
		if (err != cudaSuccess) {
			std::cerr << "cudaFreeArray failed: " << cudaGetErrorString(err) << std::endl << std::flush;
			std::terminate();
		}
#ifdef DEBUG
		std::cout << "[C++] CUDA array freed." << std::endl << std::flush;
#endif
	}
	backingTensor.reset();
}

} // namespace reg23