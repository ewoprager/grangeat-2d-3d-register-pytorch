#include <metal_stdlib>
using namespace metal;

#include <metal_stdlib>
using namespace metal;

struct ProjectDRRArgs {
    float4x4   hmi;

    float3     mappingIntercept;
    float      _pad0;               // padding to 16

    float3     mappingGradient;
    float      _pad1;               // padding to 16

    float2     detectorSpacing;
    uint2      outputSize;
    float2     outputOffset;

    float      sourceDistance;
    float      lambdaStart;
    float      stepSize;
    int        samplesPerRay;
};

kernel void project_drr(texture3d<float, access::sample> volumeTex [[texture(0)]],
                        sampler volumeSampler [[sampler(0)]],
						constant ProjectDRRArgs &args [[buffer(0)]],
						device float *out [[buffer(1)]],
						uint3 id [[thread_position_in_grid]]) {
	if (id.x >= args.outputSize.x * args.outputSize.y) return;

	uint i = id.x % args.outputSize.x;
	uint j = id.x / args.outputSize.x;
	float2 detectorPosition = args.detectorSpacing * (float2(uint2(i, j)) - 0.5 * float2(args.outputSize - 1)) + args.outputOffset;
	float3 direction = float3(detectorPosition, -args.sourceDistance);
	direction /= length(direction);
	float3 delta = direction * args.stepSize;
	delta = (args.hmi * float4(delta, 0.0)).xyz;
	float3 start = float3(0.0, 0.0, args.sourceDistance) + args.lambdaStart * direction;
	start = (args.hmi * float4(start, 1.0)).xyz;

	float3 samplePoint = start;
	float sum = 0.0;
	for (int k = 0; k < args.samplesPerRay; ++k) {
		sum += volumeTex.sample(volumeSampler, args.mappingGradient * samplePoint + args.mappingIntercept).r;
		samplePoint += delta;
	}
	out[id.x] = args.stepSize * sum;
}
