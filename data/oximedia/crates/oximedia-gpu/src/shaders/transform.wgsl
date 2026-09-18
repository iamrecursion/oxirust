// Transform shaders for DCT and FFT operations

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: TransformParams;

struct TransformParams {
    width: u32,
    height: u32,
    block_size: u32,    // 8 for 8x8 DCT, power of 2 for FFT
    transform_type: u32, // 0=DCT, 1=IDCT, 2=FFT, 3=IFFT
    stride: u32,
    is_inverse: u32,
    padding1: u32,
    padding2: u32,
}

const PI: f32 = 3.14159265359;

// 2D DCT (Type-II) for 8x8 blocks
@compute @workgroup_size(8, 8, 1)
fn dct_8x8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_x = global_id.x / 8u;
    let block_y = global_id.y / 8u;
    let u = global_id.x % 8u;
    let v = global_id.y % 8u;

    let blocks_x = params.width / 8u;
    let blocks_y = params.height / 8u;

    if (block_x >= blocks_x || block_y >= blocks_y) {
        return;
    }

    let base_x = block_x * 8u;
    let base_y = block_y * 8u;

    var sum = 0.0;

    for (var y = 0u; y < 8u; y = y + 1u) {
        for (var x = 0u; x < 8u; x = x + 1u) {
            let in_x = base_x + x;
            let in_y = base_y + y;
            let in_idx = in_y * params.stride + in_x;

            let fx = f32(x);
            let fy = f32(y);
            let fu = f32(u);
            let fv = f32(v);

            let cos_u = cos((2.0 * fx + 1.0) * fu * PI / 16.0);
            let cos_v = cos((2.0 * fy + 1.0) * fv * PI / 16.0);

            sum += input_data[in_idx] * cos_u * cos_v;
        }
    }

    // Apply normalization factors
    var alpha_u = 0.5;
    var alpha_v = 0.5;
    if (u == 0u) { alpha_u = 0.353553390593; } // 1/sqrt(2) / 2
    if (v == 0u) { alpha_v = 0.353553390593; }

    let out_x = base_x + u;
    let out_y = base_y + v;
    let out_idx = out_y * params.stride + out_x;

    output_data[out_idx] = alpha_u * alpha_v * sum;
}

// 2D Inverse DCT (Type-III) for 8x8 blocks
@compute @workgroup_size(8, 8, 1)
fn idct_8x8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_x = global_id.x / 8u;
    let block_y = global_id.y / 8u;
    let x = global_id.x % 8u;
    let y = global_id.y % 8u;

    let blocks_x = params.width / 8u;
    let blocks_y = params.height / 8u;

    if (block_x >= blocks_x || block_y >= blocks_y) {
        return;
    }

    let base_x = block_x * 8u;
    let base_y = block_y * 8u;

    var sum = 0.0;

    for (var v = 0u; v < 8u; v = v + 1u) {
        for (var u = 0u; u < 8u; u = u + 1u) {
            let in_x = base_x + u;
            let in_y = base_y + v;
            let in_idx = in_y * params.stride + in_x;

            let fx = f32(x);
            let fy = f32(y);
            let fu = f32(u);
            let fv = f32(v);

            var alpha_u = 0.5;
            var alpha_v = 0.5;
            if (u == 0u) { alpha_u = 0.353553390593; }
            if (v == 0u) { alpha_v = 0.353553390593; }

            let cos_u = cos((2.0 * fx + 1.0) * fu * PI / 16.0);
            let cos_v = cos((2.0 * fy + 1.0) * fv * PI / 16.0);

            sum += alpha_u * alpha_v * input_data[in_idx] * cos_u * cos_v;
        }
    }

    let out_x = base_x + x;
    let out_y = base_y + y;
    let out_idx = out_y * params.stride + out_x;

    output_data[out_idx] = sum;
}

// Cooley-Tukey FFT butterfly operation (radix-2)
fn fft_butterfly(
    a_real: f32, a_imag: f32,
    b_real: f32, b_imag: f32,
    twiddle_real: f32, twiddle_imag: f32,
    out_a: ptr<function, vec2<f32>>,
    out_b: ptr<function, vec2<f32>>
) {
    let t_real = b_real * twiddle_real - b_imag * twiddle_imag;
    let t_imag = b_real * twiddle_imag + b_imag * twiddle_real;

    (*out_a).x = a_real + t_real;
    (*out_a).y = a_imag + t_imag;
    (*out_b).x = a_real - t_real;
    (*out_b).y = a_imag - t_imag;
}

// 1D FFT horizontal pass (Stockham algorithm for simplicity)
@compute @workgroup_size(256, 1, 1)
fn fft_horizontal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let y = global_id.x;

    if (y >= params.height) {
        return;
    }

    let n = params.width;
    let log2n = u32(log2(f32(n)));

    // Bit-reverse permutation
    var real_buffer: array<f32, 1024>;
    var imag_buffer: array<f32, 1024>;

    for (var i = 0u; i < n; i = i + 1u) {
        var j = 0u;
        var temp = i;
        for (var b = 0u; b < log2n; b = b + 1u) {
            j = (j << 1u) | (temp & 1u);
            temp = temp >> 1u;
        }

        let idx = y * params.stride + i;
        real_buffer[j] = input_data[idx * 2u];
        imag_buffer[j] = input_data[idx * 2u + 1u];
    }

    // Cooley-Tukey FFT
    for (var s = 1u; s <= log2n; s = s + 1u) {
        let m = 1u << s;
        let m2 = m / 2u;

        for (var k = 0u; k < n; k = k + m) {
            for (var j = 0u; j < m2; j = j + 1u) {
                let t = -2.0 * PI * f32(j) / f32(m);
                let twiddle_real = cos(t);
                let twiddle_imag = sin(t);

                let idx1 = k + j;
                let idx2 = k + j + m2;

                var out_a: vec2<f32>;
                var out_b: vec2<f32>;

                fft_butterfly(
                    real_buffer[idx1], imag_buffer[idx1],
                    real_buffer[idx2], imag_buffer[idx2],
                    twiddle_real, twiddle_imag,
                    &out_a, &out_b
                );

                real_buffer[idx1] = out_a.x;
                imag_buffer[idx1] = out_a.y;
                real_buffer[idx2] = out_b.x;
                imag_buffer[idx2] = out_b.y;
            }
        }
    }

    // Write back results
    for (var i = 0u; i < n; i = i + 1u) {
        let idx = y * params.stride + i;
        output_data[idx * 2u] = real_buffer[i];
        output_data[idx * 2u + 1u] = imag_buffer[i];
    }
}

// 2D DCT using row-column decomposition (more general, any size)
@compute @workgroup_size(256, 1, 1)
fn dct_row(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let y = idx / params.width;
    let u = idx % params.width;

    if (y >= params.height || u >= params.width) {
        return;
    }

    let n = params.width;
    var sum = 0.0;

    for (var x = 0u; x < n; x = x + 1u) {
        let in_idx = y * params.stride + x;
        let fx = f32(x);
        let fu = f32(u);
        let cos_val = cos((2.0 * fx + 1.0) * fu * PI / (2.0 * f32(n)));
        sum += input_data[in_idx] * cos_val;
    }

    var alpha = sqrt(2.0 / f32(n));
    if (u == 0u) {
        alpha = sqrt(1.0 / f32(n));
    }

    let out_idx = y * params.stride + u;
    output_data[out_idx] = alpha * sum;
}

@compute @workgroup_size(256, 1, 1)
fn dct_col(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let x = idx / params.height;
    let v = idx % params.height;

    if (x >= params.width || v >= params.height) {
        return;
    }

    let n = params.height;
    var sum = 0.0;

    for (var y = 0u; y < n; y = y + 1u) {
        let in_idx = y * params.stride + x;
        let fy = f32(y);
        let fv = f32(v);
        let cos_val = cos((2.0 * fy + 1.0) * fv * PI / (2.0 * f32(n)));
        sum += input_data[in_idx] * cos_val;
    }

    var alpha = sqrt(2.0 / f32(n));
    if (v == 0u) {
        alpha = sqrt(1.0 / f32(n));
    }

    let out_idx = v * params.stride + x;
    output_data[out_idx] = alpha * sum;
}

// Magnitude computation for frequency domain visualization
@compute @workgroup_size(256, 1, 1)
fn compute_magnitude(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.width * params.height) {
        return;
    }

    let real = input_data[idx * 2u];
    let imag = input_data[idx * 2u + 1u];
    let magnitude = sqrt(real * real + imag * imag);

    output_data[idx] = magnitude;
}

// Phase computation for frequency domain analysis
@compute @workgroup_size(256, 1, 1)
fn compute_phase(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.width * params.height) {
        return;
    }

    let real = input_data[idx * 2u];
    let imag = input_data[idx * 2u + 1u];
    let phase = atan2(imag, real);

    output_data[idx] = phase;
}
