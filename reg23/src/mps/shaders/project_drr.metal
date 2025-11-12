#include <metal_stdlib>
using namespace metal;

kernel void project_drr(texture3d<float, access::sample> volumeTex [[texture(0)]], sampler volumeSampler [[sampler(0)]],
						constant float &sourceDistance [[buffer(0)]], constant float &lambdaStart [[buffer(1)]],
						constant float &stepSize [[buffer(2)]], constant uint &samplesPerRay [[buffer(3)]],
						constant float4x4 &hmi [[buffer(4)]], constant float3 &mappingIntercept [[buffer(5)]],
						constant float3 &mappingGradient [[buffer(6)]], constant float2 &detectorSpacing [[buffer(7)]],
						constant uint2 &outputSize [[buffer(8)]], constant float2 &outputOffset [[buffer(9)]],
						device float *out [[buffer(10)]], uint3 id [[thread_position_in_grid]]) {
	if (id.x >= outputSize.x * outputSize.y) return;

	uint i = id.x % outputSize.x;
	uint j = id.x / outputSize.x;
	float2 detectorPosition = detectorSpacing * (float2(uint2(i, j)) - 0.5 * float2(outputSize - 1)) + outputOffset;
	float3 direction = float3(detectorPosition, -sourceDistance);
	direction /= length(direction);
	float3 delta = direction * stepSize;
	delta = (hmi * float4(delta, 0.0)).xyz;
	float3 start = float3(0.0, 0.0, sourceDistance) + lambdaStart * direction;
	start = (hmi * float4(start, 1.0)).xyz;

	float3 samplePoint = start;
	float sum = 0.f;
	for (int k = 0; k < samplesPerRay; ++k) {
		sum += volumeTex.sample(volumeSampler, mappingGradient * samplePoint + mappingIntercept).r;
		samplePoint += delta;
	}
	out[id.x] = float(stepSize) * sum;
}
