#include <torch/extension.h>

#include "../include/Texture3DCPU.h"
#include "../include/ProjectDRR.h"

namespace reg23 {

using CommonData = ProjectDRR<Texture3DCPU>::CommonData;

at::Tensor ProjectDRR_CPU(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                          const at::Tensor &homographyMatrixInverse, double sourceDistance, int64_t outputWidth,
                          int64_t outputHeight, const at::Tensor &outputOffset, const at::Tensor &detectorSpacing) {
	const CommonData common = ProjectDRR<Texture3DCPU>::Common(volume, voxelSpacing, homographyMatrixInverse,
	                                                           sourceDistance, outputOffset, detectorSpacing,
	                                                           at::DeviceType::CPU);
	const Linear<Texture3DCPU::VectorType> mappingWorldToTexCoord = common.inputTexture.MappingWorldToTexCoord();
	at::Tensor flatOutput = torch::zeros(at::IntArrayRef({outputWidth * outputHeight}), volume.contiguous().options());
	float *resultFlatPtr = flatOutput.data_ptr<float>();

	for (int j = 0; j < outputHeight; ++j) {
		for (int i = 0; i < outputWidth; ++i) {
			const Vec<double, 2> detectorPosition = common.detectorSpacing * Vec<double, 2>{
				                                        static_cast<double>(i) - 0.5 * static_cast<double>(
					                                        outputWidth - 1),
				                                        static_cast<double>(j) - 0.5 * static_cast<double>(
					                                        outputHeight - 1)} + common.outputOffset;
			Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
			direction /= direction.Length();
			Vec<double, 3> delta = direction * common.stepSize;
			delta = MatMul(common.homographyMatrixInverse, VecCat(delta, 0.0)).XYZ();
			Vec<double, 3> start = Vec<double, 3>{0.0, 0.0, sourceDistance} + common.lambdaStart * direction;
			start = MatMul(common.homographyMatrixInverse, VecCat(start, 1.0)).XYZ();

			Vec<double, 3> samplePoint = start;
			float sum = 0.f;
			for (int k = 0; k < common.samplesPerRay; ++k) {
				sum += common.inputTexture.Sample(mappingWorldToTexCoord(samplePoint));
				samplePoint += delta;
			}
			resultFlatPtr[i + j * outputWidth] = common.stepSize * sum;
		}
	}

	return flatOutput.view(at::IntArrayRef{outputHeight, outputWidth});
}

at::Tensor ProjectDRR_backward_CPU(const at::Tensor &volume, const at::Tensor &voxelSpacing,
                                   const at::Tensor &homographyMatrixInverse, double sourceDistance,
                                   int64_t outputWidth, int64_t outputHeight, const at::Tensor &outputOffset,
                                   const at::Tensor &detectorSpacing, const at::Tensor &dLossDDRR) {
	CommonData common = ProjectDRR<Texture3DCPU>::Common(volume, voxelSpacing, homographyMatrixInverse, sourceDistance,
	                                                     outputOffset, detectorSpacing, at::DeviceType::CPU);
	const Linear<Texture3DCPU::VectorType> mappingWorldToTexCoord = common.inputTexture.MappingWorldToTexCoord();

	Vec<Vec<float, 4>, 4> dLossDHomographyMatrixInverse = Vec<Vec<float, 4>, 4>::Full(Vec<float, 4>::Full(0.f));
	for (int j = 0; j < outputHeight; ++j) {
		for (int i = 0; i < outputWidth; ++i) {
			const Vec<double, 2> detectorPosition = common.detectorSpacing * Vec<double, 2>{
				                                        static_cast<double>(i) - 0.5 * static_cast<double>(
					                                        outputWidth - 1),
				                                        static_cast<double>(j) - 0.5 * static_cast<double>(
					                                        outputHeight - 1)} + common.outputOffset;
			Vec<double, 3> direction = VecCat(detectorPosition, -sourceDistance);
			direction /= direction.Length();
			const Vec<double, 4> delta = VecCat(direction * common.stepSize, 0.0);
			const Vec<double, 4> start = VecCat(
				Vec<double, 3>{0.0, 0.0, sourceDistance} + common.lambdaStart * direction, 1.0);

			Vec<Vec<float, 4>, 4> dIntensityDHomographyMatrixInverse = Vec<Vec<float, 4>, 4>::Full(
				Vec<float, 4>::Full(0.f));
			for (int k = 0; k < common.samplesPerRay; ++k) {
				const Vec<double, 4> samplePointUntransformed = start + static_cast<double>(k) * delta;
				const Vec<double, 3> samplePoint = MatMul(common.homographyMatrixInverse, samplePointUntransformed).
					XYZ();
				const Vec<double, 3> samplePointMapped = mappingWorldToTexCoord(samplePoint);
				const Vec<float, 3> dSampleDSamplePointMapped = {common.inputTexture.DSampleDX(samplePointMapped),
				                                                 common.inputTexture.DSampleDY(samplePointMapped),
				                                                 common.inputTexture.DSampleDZ(samplePointMapped)};
				dIntensityDHomographyMatrixInverse += VecOuter(
					VecCat(mappingWorldToTexCoord.gradient.StaticCast<float>() * dSampleDSamplePointMapped, 0.f),
					samplePointUntransformed.StaticCast<float>());
			}
			dIntensityDHomographyMatrixInverse *= common.stepSize;
			dLossDHomographyMatrixInverse += dLossDDRR[j][i].item<float>() * dIntensityDHomographyMatrixInverse;
		}
	}
	return torch::from_blob(dLossDHomographyMatrixInverse.data(), {4, 4},
	                        torch::TensorOptions{}.dtype(torch::kFloat32).device(homographyMatrixInverse.device()));
}

} // namespace reg23