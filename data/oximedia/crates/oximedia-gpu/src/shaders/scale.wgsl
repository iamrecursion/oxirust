// Image scaling shaders with bilinear and bicubic interpolation

@group(0) @binding(0) var<storage, read> input_image: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<u32>;
@group(0) @binding(2) var<uniform> params: ScaleParams;

struct ScaleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    src_stride: u32,
    dst_stride: u32,
    filter_type: u32, // 0=nearest, 1=bilinear, 2=bicubic
    padding: u32,
}

fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32((packed >> 24u) & 0xFFu) / 255.0;
    let g = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b = f32((packed >> 8u) & 0xFFu) / 255.0;
    let a = f32(packed & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn pack_rgba(color: vec4<f32>) -> u32 {
    let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
    let a = u32(clamp(color.a * 255.0, 0.0, 255.0));
    return (r << 24u) | (g << 16u) | (b << 8u) | a;
}

fn sample_image(x: i32, y: i32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(params.src_width) - 1);
    let cy = clamp(y, 0, i32(params.src_height) - 1);
    let idx = u32(cy) * params.src_stride + u32(cx);
    return unpack_rgba(input_image[idx]);
}

// Nearest neighbor sampling
fn sample_nearest(u: f32, v: f32) -> vec4<f32> {
    let x = i32(u);
    let y = i32(v);
    return sample_image(x, y);
}

// Bilinear interpolation
fn sample_bilinear(u: f32, v: f32) -> vec4<f32> {
    let x = u - 0.5;
    let y = v - 0.5;

    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = fract(x);
    let fy = fract(y);

    let c00 = sample_image(x0, y0);
    let c10 = sample_image(x1, y0);
    let c01 = sample_image(x0, y1);
    let c11 = sample_image(x1, y1);

    let c0 = mix(c00, c10, fx);
    let c1 = mix(c01, c11, fx);

    return mix(c0, c1, fy);
}

// Cubic interpolation kernel (Catmull-Rom)
fn cubic_weight(t: f32) -> f32 {
    let a = -0.5;
    let t2 = t * t;
    let t3 = t2 * t;

    if (t < 1.0) {
        return (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0;
    } else if (t < 2.0) {
        return a * t3 - 5.0 * a * t2 + 8.0 * a * t - 4.0 * a;
    }
    return 0.0;
}

// Bicubic interpolation
fn sample_bicubic(u: f32, v: f32) -> vec4<f32> {
    let x = u - 0.5;
    let y = v - 0.5;

    let x0 = i32(floor(x));
    let y0 = i32(floor(y));

    let fx = fract(x);
    let fy = fract(y);

    var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum = 0.0;

    for (var dy = -1; dy <= 2; dy = dy + 1) {
        for (var dx = -1; dx <= 2; dx = dx + 1) {
            let sample_x = x0 + dx;
            let sample_y = y0 + dy;

            let wx = cubic_weight(abs(f32(dx) - fx));
            let wy = cubic_weight(abs(f32(dy) - fy));
            let w = wx * wy;

            result += sample_image(sample_x, sample_y) * w;
            weight_sum += w;
        }
    }

    if (weight_sum > 0.0) {
        result /= weight_sum;
    }

    return result;
}

@compute @workgroup_size(16, 16, 1)
fn scale_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.dst_width || y >= params.dst_height) {
        return;
    }

    // Calculate source coordinates
    let u = (f32(x) + 0.5) * f32(params.src_width) / f32(params.dst_width);
    let v = (f32(y) + 0.5) * f32(params.src_height) / f32(params.dst_height);

    var color: vec4<f32>;

    if (params.filter_type == 0u) {
        color = sample_nearest(u, v);
    } else if (params.filter_type == 1u) {
        color = sample_bilinear(u, v);
    } else {
        color = sample_bicubic(u, v);
    }

    let dst_idx = y * params.dst_stride + x;
    output_image[dst_idx] = pack_rgba(color);
}

// Downscale with area averaging (box filter)
@compute @workgroup_size(16, 16, 1)
fn downscale_area(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.dst_width || y >= params.dst_height) {
        return;
    }

    let scale_x = f32(params.src_width) / f32(params.dst_width);
    let scale_y = f32(params.src_height) / f32(params.dst_height);

    let src_x_start = f32(x) * scale_x;
    let src_y_start = f32(y) * scale_y;
    let src_x_end = src_x_start + scale_x;
    let src_y_end = src_y_start + scale_y;

    let x_start = i32(floor(src_x_start));
    let y_start = i32(floor(src_y_start));
    let x_end = i32(ceil(src_x_end));
    let y_end = i32(ceil(src_y_end));

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var count = 0.0;

    for (var py = y_start; py < y_end; py = py + 1) {
        for (var px = x_start; px < x_end; px = px + 1) {
            sum += sample_image(px, py);
            count += 1.0;
        }
    }

    let color = sum / count;
    let dst_idx = y * params.dst_stride + x;
    output_image[dst_idx] = pack_rgba(color);
}
