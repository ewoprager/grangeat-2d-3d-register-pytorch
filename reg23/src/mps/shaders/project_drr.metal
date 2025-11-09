#include <metal_stdlib>
using namespace metal;

kernel void sample_test(texture3d<float, access::sample> volumeTex [[texture(0)]],
							 sampler volumeSampler [[sampler(0)]],
							 device float *out [[buffer(0)]],
							 uint3 gid [[thread_position_in_grid]])
{
	if (gid.x > 0) return;

	float4 value = volumeTex.sample(volumeSampler, float3(0.5, 0.5, 0.5));
	out[0] = value.x;
}
