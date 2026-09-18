//! Image scaling compute shaders.

/// GLSL source for the bilinear scaling shader with shared-memory tiling.
///
/// Exposes the raw source string so that tests can verify structural properties
/// (e.g. presence of `shared` declarations and `barrier()` calls) without
/// requiring a live GPU.  The `bilinear` and `bilinear_tiled` macro
/// invocations below must embed identical GLSL so the two stay in sync.
pub const BILINEAR_GLSL: &str = r"
    #version 450

    layout(local_size_x = 16, local_size_y = 16) in;

    layout(set = 0, binding = 0) buffer InputBuffer {
        uint data[];
    } input_buf;

    layout(set = 0, binding = 1) buffer OutputBuffer {
        uint data[];
    } output_buf;

    layout(push_constant) uniform PushConstants {
        uint src_width;
        uint src_height;
        uint dst_width;
        uint dst_height;
        uint channels;
    } pc;

    // ---------------------------------------------------------------------------
    // Shared-memory tile (18 x 18 = 324 texels — 16-wide workgroup + 1 halo each
    // side).  The tile stores one uint per texel (packed RGBA8 or RGB8+pad).
    // This eliminates repeated global-buffer reads during the bilinear filter
    // when the scale ratio is ≤ 1 (upscale / unity).
    // ---------------------------------------------------------------------------
    shared uint tile[324]; // (16+2) * (16+2)

    // Read a clamped source texel from global input buffer.
    uint load_texel(int gx, int gy) {
        int clamped_x = clamp(gx, 0, int(pc.src_width)  - 1);
        int clamped_y = clamp(gy, 0, int(pc.src_height) - 1);
        uint idx = uint(clamped_y) * pc.src_width + uint(clamped_x);
        return input_buf.data[idx / 4];
    }

    // Unpack a uint texel into a normalised vec4 (RGBA8 layout).
    vec4 unpack_texel(uint t) {
        float r = float((t >>  0) & 0xFF) / 255.0;
        float g = float((t >>  8) & 0xFF) / 255.0;
        float b = float((t >> 16) & 0xFF) / 255.0;
        float a = float((t >> 24) & 0xFF) / 255.0;
        return vec4(r, g, b, a);
    }

    // ---------------------------------------------------------------------------
    // Cooperative tile fill.  Each invocation loads its own texel (tile centre)
    // plus halo neighbours so the 18×18 shared tile is fully populated before
    // workgroupBarrier.
    //
    // Tile layout (row-major, stride 18):
    //   tile[(ly + halo) * 18 + (lx + halo)]  is the centre of local (lx,ly)
    // where halo = 1.
    // ---------------------------------------------------------------------------
    void fill_tile(int tile_ox, int tile_oy,
                   int lx,     int ly) {
        const int HALO   = 1;
        const int STRIDE = 18; // 16 + 2*HALO

        // Centre
        tile[uint((ly + HALO) * STRIDE + (lx + HALO))] =
            load_texel(tile_ox + lx, tile_oy + ly);

        // Left halo (threads in first column)
        if (lx == 0) {
            tile[uint((ly + HALO) * STRIDE + 0)] =
                load_texel(tile_ox - HALO, tile_oy + ly);
        }
        // Right halo (threads in last column)
        if (lx == 15) {
            tile[uint((ly + HALO) * STRIDE + 17)] =
                load_texel(tile_ox + 16, tile_oy + ly);
        }
        // Top halo (threads in first row)
        if (ly == 0) {
            tile[uint(0 * STRIDE + (lx + HALO))] =
                load_texel(tile_ox + lx, tile_oy - HALO);
        }
        // Bottom halo (threads in last row)
        if (ly == 15) {
            tile[uint(17 * STRIDE + (lx + HALO))] =
                load_texel(tile_ox + lx, tile_oy + 16);
        }
        // Corners (only one thread per corner)
        if (lx == 0  && ly == 0)  { tile[uint(0 * STRIDE + 0)]   = load_texel(tile_ox - 1, tile_oy - 1); }
        if (lx == 15 && ly == 0)  { tile[uint(0 * STRIDE + 17)]  = load_texel(tile_ox + 16, tile_oy - 1); }
        if (lx == 0  && ly == 15) { tile[uint(17 * STRIDE + 0)]  = load_texel(tile_ox - 1, tile_oy + 16); }
        if (lx == 15 && ly == 15) { tile[uint(17 * STRIDE + 17)] = load_texel(tile_ox + 16, tile_oy + 16); }
    }

    // Bilinear sample from the shared tile using sub-texel fractional coordinates.
    // tile_lx / tile_ly are the local-space integer coordinates of the top-left
    // sample (0-based within the 16-wide workgroup region).
    vec4 sample_tile(int tile_lx, int tile_ly, float xf, float yf) {
        const int HALO   = 1;
        const int STRIDE = 18;
        int tx = tile_lx + HALO;
        int ty = tile_ly + HALO;
        vec4 p11 = unpack_texel(tile[uint(ty       * STRIDE + tx      )]);
        vec4 p21 = unpack_texel(tile[uint(ty       * STRIDE + tx + 1  )]);
        vec4 p12 = unpack_texel(tile[uint((ty + 1) * STRIDE + tx      )]);
        vec4 p22 = unpack_texel(tile[uint((ty + 1) * STRIDE + tx + 1  )]);
        vec4 row0 = mix(p11, p21, xf);
        vec4 row1 = mix(p12, p22, xf);
        return mix(row0, row1, yf);
    }

    // Bilinear sample directly from global buffer (fallback for downscale).
    vec4 sample_pixel(uint x, uint y) {
        uint idx = (y * pc.src_width + x) * pc.channels;
        if (pc.channels == 3) {
            uint packed = input_buf.data[idx / 4];
            uint shift = (idx % 4) * 8;
            float r = float((packed >> shift)         & 0xFF) / 255.0;
            float g = float((packed >> ((shift + 8)  % 32)) & 0xFF) / 255.0;
            float b = float((packed >> ((shift + 16) % 32)) & 0xFF) / 255.0;
            return vec4(r, g, b, 1.0);
        } else {
            uint packed = input_buf.data[idx / 4];
            float r = float((packed >>  0) & 0xFF) / 255.0;
            float g = float((packed >>  8) & 0xFF) / 255.0;
            float b = float((packed >> 16) & 0xFF) / 255.0;
            float a = float((packed >> 24) & 0xFF) / 255.0;
            return vec4(r, g, b, a);
        }
    }

    void main() {
        uint gid_x = gl_GlobalInvocationID.x;
        uint gid_y = gl_GlobalInvocationID.y;
        int  lx    = int(gl_LocalInvocationID.x);
        int  ly    = int(gl_LocalInvocationID.y);
        int  wg_x  = int(gl_WorkGroupID.x);
        int  wg_y  = int(gl_WorkGroupID.y);

        // Scale ratios (source pixels per destination pixel)
        float x_ratio = float(pc.src_width  - 1u) / float(pc.dst_width);
        float y_ratio = float(pc.src_height - 1u) / float(pc.dst_height);

        // Tile-origin in source coordinates (top-left source pixel of this
        // workgroup's destination tile).
        int tile_ox = int(float(wg_x * 16u) * x_ratio);
        int tile_oy = int(float(wg_y * 16u) * y_ratio);

        // ---------------------------------------------------------------------------
        // For upscale / unity (ratio ≤ 1.0) each of the 256 threads in the 16×16
        // workgroup maps to at most one source texel, so the 18×18 tile covers the
        // full neighbourhood needed for bilinear interpolation.  We fill the tile
        // cooperatively and read from shared memory.
        //
        // For downscale (ratio > 1.0) the source region is larger than 18 texels
        // wide/tall, so a fixed 18×18 tile cannot contain all required samples.
        // We fall back to direct global-buffer reads in that case.
        // ---------------------------------------------------------------------------
        bool use_tile = (x_ratio <= 1.0) && (y_ratio <= 1.0);

        if (use_tile) {
            fill_tile(tile_ox, tile_oy, lx, ly);
        }

        // All threads in the workgroup must synchronise before the filter reads.
        barrier();

        if (gid_x >= pc.dst_width || gid_y >= pc.dst_height) {
            return;
        }

        float src_x = float(gid_x) * x_ratio;
        float src_y = float(gid_y) * y_ratio;

        vec4 result;

        if (use_tile) {
            // Local tile coordinates of the top-left bilinear sample.
            int tile_lx = int(src_x) - tile_ox;
            int tile_ly = int(src_y) - tile_oy;
            float xf = fract(src_x);
            float yf = fract(src_y);

            // Guard: if tile coordinates go out of the 16×16 safe zone (can
            // happen at workgroup edges near image boundary), fall back to global.
            if (tile_lx >= 0 && tile_lx < 16 && tile_ly >= 0 && tile_ly < 16) {
                result = sample_tile(tile_lx, tile_ly, xf, yf);
            } else {
                // Boundary fallback
                uint x1 = uint(clamp(int(src_x),     0, int(pc.src_width  - 1u)));
                uint y1 = uint(clamp(int(src_y),     0, int(pc.src_height - 1u)));
                uint x2 = uint(clamp(int(src_x) + 1, 0, int(pc.src_width  - 1u)));
                uint y2 = uint(clamp(int(src_y) + 1, 0, int(pc.src_height - 1u)));
                float xf2 = fract(src_x);
                float yf2 = fract(src_y);
                vec4 p11 = sample_pixel(x1, y1);
                vec4 p12 = sample_pixel(x1, y2);
                vec4 p21 = sample_pixel(x2, y1);
                vec4 p22 = sample_pixel(x2, y2);
                result = mix(mix(p11, p21, xf2), mix(p12, p22, xf2), yf2);
            }
        } else {
            // Downscale path: direct global reads.
            uint x1 = uint(src_x);
            uint y1 = uint(src_y);
            uint x2 = min(x1 + 1u, pc.src_width  - 1u);
            uint y2 = min(y1 + 1u, pc.src_height - 1u);
            float xf = src_x - float(x1);
            float yf = src_y - float(y1);
            vec4 p11 = sample_pixel(x1, y1);
            vec4 p12 = sample_pixel(x1, y2);
            vec4 p21 = sample_pixel(x2, y1);
            vec4 p22 = sample_pixel(x2, y2);
            result = mix(mix(p11, p21, xf), mix(p12, p22, xf), yf);
        }

        uint out_idx = (gid_y * pc.dst_width + gid_x) * pc.channels;
        if (pc.channels == 3) {
            uint r = uint(clamp(result.r * 255.0, 0.0, 255.0));
            uint g = uint(clamp(result.g * 255.0, 0.0, 255.0));
            uint b = uint(clamp(result.b * 255.0, 0.0, 255.0));
            output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16);
        } else {
            uint r = uint(clamp(result.r * 255.0, 0.0, 255.0));
            uint g = uint(clamp(result.g * 255.0, 0.0, 255.0));
            uint b = uint(clamp(result.b * 255.0, 0.0, 255.0));
            uint a = uint(clamp(result.a * 255.0, 0.0, 255.0));
            output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16) | (a << 24);
        }
    }
";

/// Bilinear scaling compute shader with shared-memory tiling.
///
/// For upscale / unity ratios each workgroup cooperatively loads an 18×18
/// source tile (16-wide + 1-pixel halo) into `shared` memory.  After a
/// `barrier()` the bilinear filter reads exclusively from shared memory,
/// eliminating repeated global-buffer fetches.  For downscale ratios the
/// shader falls back to direct global-buffer reads because a fixed 18×18
/// tile cannot cover the larger source neighbourhood.
///
/// Only compiled with the `vulkan-backend` feature: the `vulkano_shaders::shader!`
/// proc-macro compiles this GLSL to SPIR-V at build time via `shaderc-sys`.
#[cfg(feature = "vulkan-backend")]
#[allow(missing_docs)]
pub mod bilinear {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(set = 0, binding = 0) buffer InputBuffer {
                uint data[];
            } input_buf;

            layout(set = 0, binding = 1) buffer OutputBuffer {
                uint data[];
            } output_buf;

            layout(push_constant) uniform PushConstants {
                uint src_width;
                uint src_height;
                uint dst_width;
                uint dst_height;
                uint channels;
            } pc;

            shared uint tile[324];

            uint load_texel(int gx, int gy) {
                int clamped_x = clamp(gx, 0, int(pc.src_width)  - 1);
                int clamped_y = clamp(gy, 0, int(pc.src_height) - 1);
                uint idx = uint(clamped_y) * pc.src_width + uint(clamped_x);
                return input_buf.data[idx / 4];
            }

            vec4 unpack_texel(uint t) {
                float r = float((t >>  0) & 0xFF) / 255.0;
                float g = float((t >>  8) & 0xFF) / 255.0;
                float b = float((t >> 16) & 0xFF) / 255.0;
                float a = float((t >> 24) & 0xFF) / 255.0;
                return vec4(r, g, b, a);
            }

            void fill_tile(int tile_ox, int tile_oy,
                           int lx,     int ly) {
                const int HALO   = 1;
                const int STRIDE = 18;

                tile[uint((ly + HALO) * STRIDE + (lx + HALO))] =
                    load_texel(tile_ox + lx, tile_oy + ly);

                if (lx == 0)  { tile[uint((ly + HALO) * STRIDE + 0)]  = load_texel(tile_ox - 1, tile_oy + ly); }
                if (lx == 15) { tile[uint((ly + HALO) * STRIDE + 17)] = load_texel(tile_ox + 16, tile_oy + ly); }
                if (ly == 0)  { tile[uint(0  * STRIDE + (lx + HALO))] = load_texel(tile_ox + lx, tile_oy - 1); }
                if (ly == 15) { tile[uint(17 * STRIDE + (lx + HALO))] = load_texel(tile_ox + lx, tile_oy + 16); }

                if (lx == 0  && ly == 0)  { tile[uint(0  * STRIDE + 0)]  = load_texel(tile_ox - 1, tile_oy - 1); }
                if (lx == 15 && ly == 0)  { tile[uint(0  * STRIDE + 17)] = load_texel(tile_ox + 16, tile_oy - 1); }
                if (lx == 0  && ly == 15) { tile[uint(17 * STRIDE + 0)]  = load_texel(tile_ox - 1, tile_oy + 16); }
                if (lx == 15 && ly == 15) { tile[uint(17 * STRIDE + 17)] = load_texel(tile_ox + 16, tile_oy + 16); }
            }

            vec4 sample_tile(int tile_lx, int tile_ly, float xf, float yf) {
                const int HALO   = 1;
                const int STRIDE = 18;
                int tx = tile_lx + HALO;
                int ty = tile_ly + HALO;
                vec4 p11 = unpack_texel(tile[uint(ty       * STRIDE + tx    )]);
                vec4 p21 = unpack_texel(tile[uint(ty       * STRIDE + tx + 1)]);
                vec4 p12 = unpack_texel(tile[uint((ty + 1) * STRIDE + tx    )]);
                vec4 p22 = unpack_texel(tile[uint((ty + 1) * STRIDE + tx + 1)]);
                return mix(mix(p11, p21, xf), mix(p12, p22, xf), yf);
            }

            vec4 sample_pixel(uint x, uint y) {
                uint idx = (y * pc.src_width + x) * pc.channels;
                if (pc.channels == 3) {
                    uint packed = input_buf.data[idx / 4];
                    uint shift = (idx % 4) * 8;
                    float r = float((packed >> shift)          & 0xFF) / 255.0;
                    float g = float((packed >> ((shift + 8)  % 32)) & 0xFF) / 255.0;
                    float b = float((packed >> ((shift + 16) % 32)) & 0xFF) / 255.0;
                    return vec4(r, g, b, 1.0);
                } else {
                    uint packed = input_buf.data[idx / 4];
                    float r = float((packed >>  0) & 0xFF) / 255.0;
                    float g = float((packed >>  8) & 0xFF) / 255.0;
                    float b = float((packed >> 16) & 0xFF) / 255.0;
                    float a = float((packed >> 24) & 0xFF) / 255.0;
                    return vec4(r, g, b, a);
                }
            }

            void main() {
                uint gid_x = gl_GlobalInvocationID.x;
                uint gid_y = gl_GlobalInvocationID.y;
                int  lx    = int(gl_LocalInvocationID.x);
                int  ly    = int(gl_LocalInvocationID.y);
                int  wg_x  = int(gl_WorkGroupID.x);
                int  wg_y  = int(gl_WorkGroupID.y);

                float x_ratio = float(pc.src_width  - 1u) / float(pc.dst_width);
                float y_ratio = float(pc.src_height - 1u) / float(pc.dst_height);

                int tile_ox = int(float(wg_x * 16) * x_ratio);
                int tile_oy = int(float(wg_y * 16) * y_ratio);

                bool use_tile = (x_ratio <= 1.0) && (y_ratio <= 1.0);

                if (use_tile) {
                    fill_tile(tile_ox, tile_oy, lx, ly);
                }

                barrier();

                if (gid_x >= pc.dst_width || gid_y >= pc.dst_height) {
                    return;
                }

                float src_x = float(gid_x) * x_ratio;
                float src_y = float(gid_y) * y_ratio;

                vec4 result;

                if (use_tile) {
                    int tile_lx = int(src_x) - tile_ox;
                    int tile_ly = int(src_y) - tile_oy;
                    float xf = fract(src_x);
                    float yf = fract(src_y);

                    if (tile_lx >= 0 && tile_lx < 16 && tile_ly >= 0 && tile_ly < 16) {
                        result = sample_tile(tile_lx, tile_ly, xf, yf);
                    } else {
                        uint x1 = uint(clamp(int(src_x),     0, int(pc.src_width  - 1u)));
                        uint y1 = uint(clamp(int(src_y),     0, int(pc.src_height - 1u)));
                        uint x2 = uint(clamp(int(src_x) + 1, 0, int(pc.src_width  - 1u)));
                        uint y2 = uint(clamp(int(src_y) + 1, 0, int(pc.src_height - 1u)));
                        float xf2 = fract(src_x);
                        float yf2 = fract(src_y);
                        result = mix(mix(sample_pixel(x1, y1), sample_pixel(x2, y1), xf2),
                                     mix(sample_pixel(x1, y2), sample_pixel(x2, y2), xf2), yf2);
                    }
                } else {
                    uint x1 = uint(src_x);
                    uint y1 = uint(src_y);
                    uint x2 = min(x1 + 1u, pc.src_width  - 1u);
                    uint y2 = min(y1 + 1u, pc.src_height - 1u);
                    float xf = src_x - float(x1);
                    float yf = src_y - float(y1);
                    result = mix(mix(sample_pixel(x1, y1), sample_pixel(x2, y1), xf),
                                 mix(sample_pixel(x1, y2), sample_pixel(x2, y2), xf), yf);
                }

                uint out_idx = (gid_y * pc.dst_width + gid_x) * pc.channels;
                if (pc.channels == 3) {
                    uint r = uint(clamp(result.r * 255.0, 0.0, 255.0));
                    uint g = uint(clamp(result.g * 255.0, 0.0, 255.0));
                    uint b = uint(clamp(result.b * 255.0, 0.0, 255.0));
                    output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16);
                } else {
                    uint r = uint(clamp(result.r * 255.0, 0.0, 255.0));
                    uint g = uint(clamp(result.g * 255.0, 0.0, 255.0));
                    uint b = uint(clamp(result.b * 255.0, 0.0, 255.0));
                    uint a = uint(clamp(result.a * 255.0, 0.0, 255.0));
                    output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16) | (a << 24);
                }
            }
        "
    }
}

/// Nearest neighbor scaling compute shader.
///
/// Only compiled with the `vulkan-backend` feature: the `vulkano_shaders::shader!`
/// proc-macro compiles this GLSL to SPIR-V at build time via `shaderc-sys`.
#[cfg(feature = "vulkan-backend")]
#[allow(missing_docs)]
pub mod nearest {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(set = 0, binding = 0) buffer InputBuffer {
                uint data[];
            } input_buf;

            layout(set = 0, binding = 1) buffer OutputBuffer {
                uint data[];
            } output_buf;

            layout(push_constant) uniform PushConstants {
                uint src_width;
                uint src_height;
                uint dst_width;
                uint dst_height;
                uint channels;
            } pc;

            void main() {
                uint gid_x = gl_GlobalInvocationID.x;
                uint gid_y = gl_GlobalInvocationID.y;

                if (gid_x >= pc.dst_width || gid_y >= pc.dst_height) {
                    return;
                }

                uint src_x = (gid_x * pc.src_width) / pc.dst_width;
                uint src_y = (gid_y * pc.src_height) / pc.dst_height;

                uint src_idx = (src_y * pc.src_width + src_x) * pc.channels;
                uint dst_idx = (gid_y * pc.dst_width + gid_x) * pc.channels;

                // Copy pixel data
                for (uint i = 0; i < pc.channels; i++) {
                    uint src_offset = (src_idx + i) / 4;
                    uint dst_offset = (dst_idx + i) / 4;
                    output_buf.data[dst_offset] = input_buf.data[src_offset];
                }
            }
        "
    }
}

#[cfg(test)]
mod tests {
    use super::BILINEAR_GLSL;

    #[test]
    fn bilinear_shader_has_shared_memory_tile() {
        assert!(
            BILINEAR_GLSL.contains("shared uint tile["),
            "bilinear scale shader must declare a shared-memory tile"
        );
    }

    #[test]
    fn bilinear_shader_has_workgroup_barrier() {
        assert!(
            BILINEAR_GLSL.contains("barrier()"),
            "bilinear scale shader must call barrier() after tile fill"
        );
    }

    #[test]
    fn bilinear_shader_tile_size_is_18x18() {
        // 18*18 = 324 — workgroup 16×16 + 1-pixel halo each side
        assert!(
            BILINEAR_GLSL.contains("shared uint tile[324]"),
            "shared tile must be 324 elements (18×18)"
        );
    }

    #[test]
    fn bilinear_shader_has_halo_fill_for_all_edges() {
        // Verify each edge / corner branch is present (left, right, top, bottom,
        // all four corners).
        assert!(
            BILINEAR_GLSL.contains("lx == 0"),
            "missing left halo branch"
        );
        assert!(
            BILINEAR_GLSL.contains("lx == 15"),
            "missing right halo branch"
        );
        assert!(BILINEAR_GLSL.contains("ly == 0"), "missing top halo branch");
        assert!(
            BILINEAR_GLSL.contains("ly == 15"),
            "missing bottom halo branch"
        );
    }

    #[test]
    fn bilinear_shader_has_downscale_fallback() {
        // For downscale the shader must fall back to direct global reads.
        assert!(
            BILINEAR_GLSL.contains("use_tile"),
            "bilinear scale shader must have use_tile branch for downscale fallback"
        );
    }

    #[test]
    fn bilinear_shader_uses_fract_for_sub_texel() {
        assert!(
            BILINEAR_GLSL.contains("fract("),
            "bilinear filter must use fract() for sub-texel fraction"
        );
    }
}
