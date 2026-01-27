#include <metal_stdlib>
using namespace metal;

struct FieldUniforms {
    uint count;
    float fieldScale;
    float maxTicks;
    float padding;
};

struct FieldVertexOut {
    float4 position [[position]];
};

struct MarkerVertex {
    float2 position;
    float4 color;
};

struct MarkerOut {
    float4 position [[position]];
    float4 color;
};

kernel void combine_fields(
    const device float* potential [[buffer(0)]],
    const device float* obstacle [[buffer(1)]],
    device float* combined [[buffer(2)]],
    const device FieldUniforms* uniforms [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= uniforms->count) {
        return;
    }
    combined[id] = potential[id] - obstacle[id];
}

vertex FieldVertexOut field_vertex(
    uint vid [[vertex_id]],
    const device float* combined [[buffer(0)]],
    const device FieldUniforms* uniforms [[buffer(1)]]
) {
    FieldVertexOut out;
    if (uniforms->count <= 1) {
        out.position = float4(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    float x = (float(vid) / float(uniforms->count - 1)) * 2.0 - 1.0;
    float y = clamp(combined[vid] * uniforms->fieldScale, -1.0, 1.0);
    out.position = float4(x, y, 0.0, 1.0);
    return out;
}

fragment float4 field_fragment() {
    return float4(0.2, 0.85, 1.0, 1.0);
}

vertex MarkerOut marker_vertex(
    const device MarkerVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    MarkerOut out;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    out.color = vertices[vid].color;
    return out;
}

fragment float4 marker_fragment(MarkerOut in [[stage_in]]) {
    return in.color;
}
