#pragma once

#include "Common.h"
#include "Vec.h"

namespace reg23 {

/**
 * @defgroup textures Texture-Related Types
 * @brief Classes and enumerations related to accessing data as textures
 * @{
 */

/**
 * An enumeration of address modes for texture objects, similar to the argument `padding_mode` in
 * `torch.nn.functional.grid_sample`, or the CUDA enumeration `cudaTextureAddressMode`.
 */
enum class TextureAddressMode {
	ZERO, ///< Sampling locations outside texture coordinate range will be read as 0
	WRAP ///< Sampling locations outside texture coordinate range will be read wrapped back according to ((x - left) mod
		 ///< width + left, (y - bottom) mod height + bottom, etc...
};

#ifdef __CUDACC__
inline cudaTextureAddressMode TextureAddressModeToCuda(TextureAddressMode tam) {
	switch (tam) {
	case TextureAddressMode::ZERO:
		return cudaAddressModeBorder;
	case TextureAddressMode::WRAP:
		return cudaAddressModeWrap;
	default:
		return cudaAddressModeBorder;
	}
}
#endif

/**
 * @brief A parent texture class containing template data and functionality.
 * @tparam dimensionality The integer dimensionality of the texture
 * @tparam intType The type used to express integer values like the dimensions of the texture
 * @tparam floatType The type used to express floating-point values like the spacing of the texture's values in world
 * coordinates
 *
 * This class is not an interface, rather it is templated to store some values and implement some functionality common
 * to all texture objects. It is intended to be used as a fully template-specified base class for any texture object.
 */
template <std::size_t dimensionality, typename intType = int, typename floatType = float> class Texture {
  public:
	using SizeType = Vec<intType, dimensionality>;
	using VectorType = Vec<floatType, dimensionality>;
	using AddressModeType = Vec<TextureAddressMode, dimensionality>;

	/// Get the dimensions of the texture
	[[nodiscard]] __host__ __device__ const SizeType &Size() const { return size; }
	/// Get the spacing of the texture's values in world coordinates
	[[nodiscard]] __host__ __device__ const VectorType &Spacing() const { return spacing; }
	/// Get the position of the centre of the texture in world coordinates
	[[nodiscard]] __host__ __device__ const VectorType &CentrePosition() const { return centrePosition; }

	/**
	 * @return The size of the texture in world coordinates
	 */
	[[nodiscard]] __host__ __device__ VectorType SizeWorld() const {
		return size.template StaticCast<floatType>() * spacing;
	}

	/**
	 * @param index A location in the texture data
	 * @return Whether the location is within the bounds of the texture's data's size
	 */
	[[nodiscard]] __host__ __device__ bool In(const SizeType &index) const {
		return (index >= SizeType{}).BooleanAll() && (index < size).BooleanAll();
	}

	/**
	 * @return The `Linear` mapping from world coordinates to texture coordinates, according to the texture's world
	 * position and value spacing.
	 */
	[[nodiscard]] __host__ __device__ Linear<VectorType> MappingWorldToTexCoord() const {
		const VectorType sizeWorldInverse = floatType{1.} / SizeWorld();
		return {VectorType::Full(floatType{.5}) - centrePosition * sizeWorldInverse, sizeWorldInverse};
	}

  protected:
	Texture() = default;

	Texture(SizeType _size, VectorType _spacing, VectorType _centrePosition = {})
		: size(_size), spacing(_spacing), centrePosition(_centrePosition) {}

	// yes copy
	Texture(const Texture &) = default;

	Texture &operator=(const Texture &) = default;

	// yes move
	Texture(Texture &&) noexcept = default;

	Texture &operator=(Texture &&) noexcept = default;

  private:
	SizeType size{};			 ///< The dimensions of the texture
	VectorType spacing{};		 ///< The spacing between the values of the texture in world coordinates
	VectorType centrePosition{}; ///< The position of the centre of the texture in world coordinates
};

/**
 * @}
 */

} // namespace reg23