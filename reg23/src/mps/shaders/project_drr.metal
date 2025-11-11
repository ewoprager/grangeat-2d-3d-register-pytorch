#include <metal_stdlib>
using namespace metal;

kernel void project_drr(texture3d<float, access::sample> volumeTex [[texture(0)]], sampler volumeSampler [[sampler(0)]],
						constant float &sourceDistance [[buffer(0)]], constant float &lambdaStart [[buffer(1)]],
						constant float &stepSize [[buffer(2)]], constant uint &samplesPerRay [[buffer(3)]],
						constant float16 &hmi [[buffer(4)]], constant float3 &mappingIntercept [[buffer(5)]],
						constant float3 &mappingGradient [[buffer(6)]], constant float2 &detectorSpacing [[buffer(7)]],
						constant uint2 &outputSize [[buffer(8)]], constant float2 &outputOffset [[buffer(9)]],
						device float *out [[buffer(10)]], uint3 id [[thread_position_in_grid]]) {
	if (id.x >= outputSize.x * outputSize.y) return;

	out[id.x] = sourceDistance;
}
