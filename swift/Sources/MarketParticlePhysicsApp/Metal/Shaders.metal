#include <metal_stdlib>
using namespace metal;

struct ColoredVertex {
    float2 position;
    float4 color;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut color_vertex(const device ColoredVertex *vertices [[buffer(0)]],
                              uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    out.color = vertices[vid].color;
    return out;
}

fragment float4 color_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
