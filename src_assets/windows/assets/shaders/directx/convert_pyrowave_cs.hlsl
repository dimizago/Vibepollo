// RGB -> Y'CbCr planes for the PyroWave encoder (src/platform/windows/pyrowave_encode.cpp).
//
// Writes three single-channel UNORM planes (R8 for 8-bit, R16 for 10-bit streams) that
// the D3D12 PyroWave encoder reads directly. Same color math as the pixel-shader
// converters used by the other encoders: CONVERT_FUNCTION picks the transfer function
// (defines PQ / LINEAR, else saturate) and the colour matrix cbuffer comes from
// video::color_vectors_from_colorspace(colorspace, true).
//
// SUBSAMPLE_420: one thread per 2x2 luma quad, whose chroma sample is the average of the
// quad (centre sited, like PyroWave's reference converter). Otherwise one thread per
// pixel with full-resolution chroma.
//
// The picture is letterboxed into the output: pixels outside it, and every pixel of a
// blank frame, are black.

#if defined(PQ)
  #include "include/convert_perceptual_quantizer_base.hlsl"
#elif defined(LINEAR)
  #include "include/convert_linear_base.hlsl"
#else
  #include "include/convert_base.hlsl"
#endif

Texture2D image : register(t0);
SamplerState def_sampler : register(s0);

RWTexture2D<unorm float> out_y : register(u0);
RWTexture2D<unorm float> out_u : register(u1);
RWTexture2D<unorm float> out_v : register(u2);

cbuffer color_matrix_cbuffer : register(b0) {
    float4 color_vec_y;
    float4 color_vec_u;
    float4 color_vec_v;
    float2 range_y;
    float2 range_uv;
};

cbuffer params_cbuffer : register(b1) {
    uint2 luma_size;        // output luma extent in pixels
    float2 content_offset;  // top-left of the picture in output pixels
    float2 content_scale;   // 1 / picture size in output pixels
    uint blank;             // nonzero: write black everywhere
};

// Unconverted source colour at an output pixel centre; black outside the picture.
float3 fetch(float2 out_pos)
{
    float2 uv = (out_pos - content_offset) * content_scale;
    if (blank != 0 || any(uv < 0.0) || any(uv > 1.0)) {
        return 0.0;
    }
    return image.SampleLevel(def_sampler, uv, 0).rgb;
}

float to_y(float3 rgb)
{
    return (dot(color_vec_y.xyz, rgb) + color_vec_y.w) * range_y.x + range_y.y;
}

float2 to_uv(float3 rgb)
{
    float u = dot(color_vec_u.xyz, rgb) + color_vec_u.w;
    float v = dot(color_vec_v.xyz, rgb) + color_vec_v.w;
    return float2(u, v) * range_uv.x + range_uv.y;
}

[numthreads(8, 8, 1)]
void main_cs(uint2 id : SV_DispatchThreadID)
{
#ifdef SUBSAMPLE_420
    uint2 base = id * 2;
    if (any(base >= luma_size)) {
        return;
    }

    float3 sum = 0.0;
    [unroll] for (uint dy = 0; dy < 2; dy++) {
        [unroll] for (uint dx = 0; dx < 2; dx++) {
            uint2 pos = base + uint2(dx, dy);
            float3 raw = fetch(float2(pos) + 0.5);
            sum += raw;
            if (all(pos < luma_size)) {
                out_y[pos] = to_y(CONVERT_FUNCTION(raw));
            }
        }
    }

    // Like the pixel-shader converters, average the source samples, then convert.
    float2 uv = to_uv(CONVERT_FUNCTION(sum * 0.25));
    out_u[id] = uv.x;
    out_v[id] = uv.y;
#else
    if (any(id >= luma_size)) {
        return;
    }

    float3 rgb = CONVERT_FUNCTION(fetch(float2(id) + 0.5));
    out_y[id] = to_y(rgb);
    float2 uv = to_uv(rgb);
    out_u[id] = uv.x;
    out_v[id] = uv.y;
#endif
}
