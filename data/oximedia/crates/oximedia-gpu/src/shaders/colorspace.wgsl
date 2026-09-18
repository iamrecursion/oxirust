// Color space conversion shaders for RGB <-> YUV transformations
// Using BT.601 coefficients for standard definition video

// RGB to YUV conversion kernel
@group(0) @binding(0) var<storage, read> input_rgb: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_yuv: array<u32>;
@group(0) @binding(2) var<uniform> params: ConversionParams;

struct ConversionParams {
    width: u32,
    height: u32,
    stride: u32,
    format: u32, // 0=BT.601, 1=BT.709, 2=BT.2020
}

// BT.601 coefficients
const BT601_KR: f32 = 0.299;
const BT601_KB: f32 = 0.114;
const BT601_KG: f32 = 0.587;

// BT.709 coefficients
const BT709_KR: f32 = 0.2126;
const BT709_KB: f32 = 0.0722;
const BT709_KG: f32 = 0.7152;

// BT.2020 coefficients
const BT2020_KR: f32 = 0.2627;
const BT2020_KB: f32 = 0.0593;
const BT2020_KG: f32 = 0.6780;

fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32((packed >> 24u) & 0xFFu) / 255.0;
    let g = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b = f32((packed >> 8u) & 0xFFu) / 255.0;
    let a = f32(packed & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn pack_yuva(y: f32, u: f32, v: f32, a: f32) -> u32 {
    let yi = u32(clamp(y * 255.0, 0.0, 255.0));
    let ui = u32(clamp(u * 255.0, 0.0, 255.0));
    let vi = u32(clamp(v * 255.0, 0.0, 255.0));
    let ai = u32(clamp(a * 255.0, 0.0, 255.0));
    return (yi << 24u) | (ui << 16u) | (vi << 8u) | ai;
}

fn rgb_to_yuv_bt601(rgb: vec3<f32>) -> vec3<f32> {
    let y = BT601_KR * rgb.r + BT601_KG * rgb.g + BT601_KB * rgb.b;
    let u = (rgb.b - y) / (2.0 * (1.0 - BT601_KB)) + 0.5;
    let v = (rgb.r - y) / (2.0 * (1.0 - BT601_KR)) + 0.5;
    return vec3<f32>(y, u, v);
}

fn rgb_to_yuv_bt709(rgb: vec3<f32>) -> vec3<f32> {
    let y = BT709_KR * rgb.r + BT709_KG * rgb.g + BT709_KB * rgb.b;
    let u = (rgb.b - y) / (2.0 * (1.0 - BT709_KB)) + 0.5;
    let v = (rgb.r - y) / (2.0 * (1.0 - BT709_KR)) + 0.5;
    return vec3<f32>(y, u, v);
}

fn rgb_to_yuv_bt2020(rgb: vec3<f32>) -> vec3<f32> {
    let y = BT2020_KR * rgb.r + BT2020_KG * rgb.g + BT2020_KB * rgb.b;
    let u = (rgb.b - y) / (2.0 * (1.0 - BT2020_KB)) + 0.5;
    let v = (rgb.r - y) / (2.0 * (1.0 - BT2020_KR)) + 0.5;
    return vec3<f32>(y, u, v);
}

@compute @workgroup_size(16, 16, 1)
fn rgb_to_yuv_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.stride + x;
    let rgba = unpack_rgba(input_rgb[idx]);

    var yuv: vec3<f32>;
    if (params.format == 0u) {
        yuv = rgb_to_yuv_bt601(rgba.rgb);
    } else if (params.format == 1u) {
        yuv = rgb_to_yuv_bt709(rgba.rgb);
    } else {
        yuv = rgb_to_yuv_bt2020(rgba.rgb);
    }

    output_yuv[idx] = pack_yuva(yuv.x, yuv.y, yuv.z, rgba.a);
}

// YUV to RGB conversion kernel
fn yuv_to_rgb_bt601(yuv: vec3<f32>) -> vec3<f32> {
    let y = yuv.x;
    let u = yuv.y - 0.5;
    let v = yuv.z - 0.5;

    let r = y + 2.0 * (1.0 - BT601_KR) * v;
    let b = y + 2.0 * (1.0 - BT601_KB) * u;
    let g = (y - BT601_KR * r - BT601_KB * b) / BT601_KG;

    return vec3<f32>(r, g, b);
}

fn yuv_to_rgb_bt709(yuv: vec3<f32>) -> vec3<f32> {
    let y = yuv.x;
    let u = yuv.y - 0.5;
    let v = yuv.z - 0.5;

    let r = y + 2.0 * (1.0 - BT709_KR) * v;
    let b = y + 2.0 * (1.0 - BT709_KB) * u;
    let g = (y - BT709_KR * r - BT709_KB * b) / BT709_KG;

    return vec3<f32>(r, g, b);
}

fn yuv_to_rgb_bt2020(yuv: vec3<f32>) -> vec3<f32> {
    let y = yuv.x;
    let u = yuv.y - 0.5;
    let v = yuv.z - 0.5;

    let r = y + 2.0 * (1.0 - BT2020_KR) * v;
    let b = y + 2.0 * (1.0 - BT2020_KB) * u;
    let g = (y - BT2020_KR * r - BT2020_KB * b) / BT2020_KG;

    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(16, 16, 1)
fn yuv_to_rgb_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.stride + x;
    let yuva = unpack_rgba(input_rgb[idx]); // Reusing input buffer name

    var rgb: vec3<f32>;
    if (params.format == 0u) {
        rgb = yuv_to_rgb_bt601(yuva.rgb);
    } else if (params.format == 1u) {
        rgb = yuv_to_rgb_bt709(yuva.rgb);
    } else {
        rgb = yuv_to_rgb_bt2020(yuva.rgb);
    }

    let r = u32(clamp(rgb.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(rgb.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(rgb.b * 255.0, 0.0, 255.0));
    let a = u32(clamp(yuva.a * 255.0, 0.0, 255.0));

    output_yuv[idx] = (r << 24u) | (g << 16u) | (b << 8u) | a;
}
