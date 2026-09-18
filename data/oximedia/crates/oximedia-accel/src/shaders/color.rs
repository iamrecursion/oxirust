//! Color conversion compute shaders.

/// GLSL source for the RGB→YUV420p shader with shared-memory tiling.
///
/// Exposes the raw source so that tests can verify structural properties
/// (presence of `shared` declarations and `barrier()` calls) without a GPU.
pub const RGB_TO_YUV420P_GLSL: &str = r"
    #version 450

    layout(local_size_x = 16, local_size_y = 16) in;

    layout(set = 0, binding = 0) buffer InputBuffer {
        uint data[];
    } input_buf;

    layout(set = 0, binding = 1) buffer OutputYBuffer {
        uint data[];
    } output_y;

    layout(set = 0, binding = 2) buffer OutputUBuffer {
        uint data[];
    } output_u;

    layout(set = 0, binding = 3) buffer OutputVBuffer {
        uint data[];
    } output_v;

    layout(push_constant) uniform PushConstants {
        uint width;
        uint height;
        uint input_channels;
    } pc;

    // ---------------------------------------------------------------------------
    // Shared-memory tile: one uint per thread in the 16×16 workgroup.
    // Each thread loads its own packed-RGBA texel from the global buffer into
    // the tile, synchronises, then reads from the tile for the colour transform.
    //
    // For the YUV420p U/V subsampled planes the thread at (2i, 2j) also needs
    // its three neighbours (2i+1,2j), (2i,2j+1), (2i+1,2j+1).  All four are
    // in the same 16×16 tile so they are always available after the barrier.
    // ---------------------------------------------------------------------------
    shared uint tile[256]; // 16 * 16

    vec3 rgb_to_yuv(vec3 rgb) {
        float y =  0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
        float u = -0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b + 0.5;
        float v =  0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b + 0.5;
        return vec3(y, u, v);
    }

    vec3 unpack_rgb(uint gx, uint gy) {
        uint idx    = (gy * pc.width + gx) * pc.input_channels;
        uint packed = input_buf.data[idx / 4];
        uint shift  = (idx % 4) * 8;
        float r = float((packed >> shift)          & 0xFF) / 255.0;
        float g = float((packed >> ((shift + 8)  % 32)) & 0xFF) / 255.0;
        float b = float((packed >> ((shift + 16) % 32)) & 0xFF) / 255.0;
        return vec3(r, g, b);
    }

    void main() {
        uint gid_x = gl_GlobalInvocationID.x;
        uint gid_y = gl_GlobalInvocationID.y;
        uint lx    = gl_LocalInvocationID.x;
        uint ly    = gl_LocalInvocationID.y;

        // Cooperative load: every thread loads its own texel into the tile.
        if (gid_x < pc.width && gid_y < pc.height) {
            uint idx    = (gid_y * pc.width + gid_x) * pc.input_channels;
            uint packed = input_buf.data[idx / 4];
            tile[ly * 16u + lx] = packed;
        } else {
            tile[ly * 16u + lx] = 0u;
        }

        barrier();

        if (gid_x >= pc.width || gid_y >= pc.height) {
            return;
        }

        // Read from tile instead of global buffer.
        uint t      = tile[ly * 16u + lx];
        uint shift  = ((gid_x * pc.input_channels) % 4u) * 8u;
        float r = float((t >> shift)          & 0xFF) / 255.0;
        float g = float((t >> ((shift + 8u)  % 32u)) & 0xFF) / 255.0;
        float b = float((t >> ((shift + 16u) % 32u)) & 0xFF) / 255.0;

        vec3 yuv = rgb_to_yuv(vec3(r, g, b));

        // Write Y plane
        uint y_idx = gid_y * pc.width + gid_x;
        output_y.data[y_idx / 4] = uint(clamp(yuv.x * 255.0, 0.0, 255.0));

        // Write U and V planes (subsampled 2x2).
        // Only the even-column, even-row thread in each 2×2 quad writes U/V.
        // The three neighbours are in the same tile so we read from tile[].
        if ((gid_x % 2u) == 0u && (gid_y % 2u) == 0u) {
            // Average over the 2×2 quad sourced from shared memory.
            float r_sum = r, g_sum = g, b_sum = b;
            float count = 1.0;

            // Right neighbour (lx+1 may be in the same 16-wide tile)
            if (lx + 1u < 16u && gid_x + 1u < pc.width) {
                uint t1    = tile[ly * 16u + lx + 1u];
                uint sh1   = (((gid_x + 1u) * pc.input_channels) % 4u) * 8u;
                r_sum += float((t1 >> sh1)          & 0xFF) / 255.0;
                g_sum += float((t1 >> ((sh1 + 8u)  % 32u)) & 0xFF) / 255.0;
                b_sum += float((t1 >> ((sh1 + 16u) % 32u)) & 0xFF) / 255.0;
                count += 1.0;
            }
            // Bottom neighbour
            if (ly + 1u < 16u && gid_y + 1u < pc.height) {
                uint t2    = tile[(ly + 1u) * 16u + lx];
                uint sh2   = ((gid_x * pc.input_channels) % 4u) * 8u;
                r_sum += float((t2 >> sh2)          & 0xFF) / 255.0;
                g_sum += float((t2 >> ((sh2 + 8u)  % 32u)) & 0xFF) / 255.0;
                b_sum += float((t2 >> ((sh2 + 16u) % 32u)) & 0xFF) / 255.0;
                count += 1.0;
            }
            // Bottom-right neighbour
            if (lx + 1u < 16u && ly + 1u < 16u &&
                gid_x + 1u < pc.width && gid_y + 1u < pc.height) {
                uint t3    = tile[(ly + 1u) * 16u + lx + 1u];
                uint sh3   = (((gid_x + 1u) * pc.input_channels) % 4u) * 8u;
                r_sum += float((t3 >> sh3)          & 0xFF) / 255.0;
                g_sum += float((t3 >> ((sh3 + 8u)  % 32u)) & 0xFF) / 255.0;
                b_sum += float((t3 >> ((sh3 + 16u) % 32u)) & 0xFF) / 255.0;
                count += 1.0;
            }

            vec3 avg_yuv = rgb_to_yuv(vec3(r_sum / count, g_sum / count, b_sum / count));
            uint uv_idx = (gid_y / 2u) * (pc.width / 2u) + (gid_x / 2u);
            output_u.data[uv_idx / 4] = uint(clamp(avg_yuv.y * 255.0, 0.0, 255.0));
            output_v.data[uv_idx / 4] = uint(clamp(avg_yuv.z * 255.0, 0.0, 255.0));
        }
    }
";

/// GLSL source for the YUV420p→RGB shader with shared-memory tiling.
pub const YUV420P_TO_RGB_GLSL: &str = r"
    #version 450

    layout(local_size_x = 16, local_size_y = 16) in;

    layout(set = 0, binding = 0) buffer InputYBuffer {
        uint data[];
    } input_y;

    layout(set = 0, binding = 1) buffer InputUBuffer {
        uint data[];
    } input_u;

    layout(set = 0, binding = 2) buffer InputVBuffer {
        uint data[];
    } input_v;

    layout(set = 0, binding = 3) buffer OutputBuffer {
        uint data[];
    } output_buf;

    layout(push_constant) uniform PushConstants {
        uint width;
        uint height;
        uint output_channels;
    } pc;

    // Shared tile holds packed Y values for the 16×16 workgroup.
    // U/V are subsampled (one value per 2×2) so we read them directly from
    // global memory — the savings from tiling Y alone are significant because
    // the Y plane is fully dense and exhibits 2-D spatial reuse.
    shared uint tile_y[256]; // 16 * 16

    vec3 yuv_to_rgb(vec3 yuv) {
        float y = yuv.x;
        float u = yuv.y - 0.5;
        float v = yuv.z - 0.5;
        float r = y + 1.402 * v;
        float g = y - 0.344 * u - 0.714 * v;
        float b = y + 1.772 * u;
        return clamp(vec3(r, g, b), 0.0, 1.0);
    }

    void main() {
        uint gid_x = gl_GlobalInvocationID.x;
        uint gid_y = gl_GlobalInvocationID.y;
        uint lx    = gl_LocalInvocationID.x;
        uint ly    = gl_LocalInvocationID.y;

        // Cooperative load of Y plane into shared tile.
        if (gid_x < pc.width && gid_y < pc.height) {
            uint y_idx    = gid_y * pc.width + gid_x;
            tile_y[ly * 16u + lx] = input_y.data[y_idx / 4];
        } else {
            tile_y[ly * 16u + lx] = 0u;
        }

        barrier();

        if (gid_x >= pc.width || gid_y >= pc.height) {
            return;
        }

        // Read Y from tile; read U/V from global (subsampled, coalesced).
        float y = float(tile_y[ly * 16u + lx] & 0xFFu) / 255.0;

        uint uv_idx = (gid_y / 2u) * (pc.width / 2u) + (gid_x / 2u);
        float u = float(input_u.data[uv_idx / 4] & 0xFF) / 255.0;
        float v = float(input_v.data[uv_idx / 4] & 0xFF) / 255.0;

        vec3 rgb = yuv_to_rgb(vec3(y, u, v));

        uint out_idx = (gid_y * pc.width + gid_x) * pc.output_channels;
        uint r = uint(rgb.r * 255.0);
        uint g = uint(rgb.g * 255.0);
        uint b = uint(rgb.b * 255.0);

        if (pc.output_channels == 3u) {
            output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16);
        } else {
            output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16) | (255u << 24);
        }
    }
";

/// RGB to `YUV420p` conversion compute shader with shared-memory tiling.
///
/// Each invocation in the 16×16 workgroup loads its packed-RGB texel into a
/// 256-element `shared` tile.  After `barrier()` the colour transform reads
/// from the tile; for the U/V subsampled planes the three 2×2 quad neighbours
/// are also read from the tile, eliminating four global reads per 2×2 quad.
///
/// Only compiled with the `vulkan-backend` feature: the `vulkano_shaders::shader!`
/// proc-macro compiles this GLSL to SPIR-V at build time via `shaderc-sys`.
#[cfg(feature = "vulkan-backend")]
#[allow(missing_docs)]
pub mod rgb_to_yuv420p {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(set = 0, binding = 0) buffer InputBuffer {
                uint data[];
            } input_buf;

            layout(set = 0, binding = 1) buffer OutputYBuffer {
                uint data[];
            } output_y;

            layout(set = 0, binding = 2) buffer OutputUBuffer {
                uint data[];
            } output_u;

            layout(set = 0, binding = 3) buffer OutputVBuffer {
                uint data[];
            } output_v;

            layout(push_constant) uniform PushConstants {
                uint width;
                uint height;
                uint input_channels;
            } pc;

            shared uint tile[256];

            vec3 rgb_to_yuv(vec3 rgb) {
                float y =  0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
                float u = -0.169 * rgb.r - 0.331 * rgb.g + 0.500 * rgb.b + 0.5;
                float v =  0.500 * rgb.r - 0.419 * rgb.g - 0.081 * rgb.b + 0.5;
                return vec3(y, u, v);
            }

            void main() {
                uint gid_x = gl_GlobalInvocationID.x;
                uint gid_y = gl_GlobalInvocationID.y;
                uint lx    = gl_LocalInvocationID.x;
                uint ly    = gl_LocalInvocationID.y;

                if (gid_x < pc.width && gid_y < pc.height) {
                    uint idx    = (gid_y * pc.width + gid_x) * pc.input_channels;
                    tile[ly * 16u + lx] = input_buf.data[idx / 4];
                } else {
                    tile[ly * 16u + lx] = 0u;
                }

                barrier();

                if (gid_x >= pc.width || gid_y >= pc.height) {
                    return;
                }

                uint t     = tile[ly * 16u + lx];
                uint shift = ((gid_x * pc.input_channels) % 4u) * 8u;
                float r = float((t >> shift)          & 0xFF) / 255.0;
                float g = float((t >> ((shift + 8u)  % 32u)) & 0xFF) / 255.0;
                float b = float((t >> ((shift + 16u) % 32u)) & 0xFF) / 255.0;

                vec3 yuv = rgb_to_yuv(vec3(r, g, b));

                uint y_idx = gid_y * pc.width + gid_x;
                output_y.data[y_idx / 4] = uint(clamp(yuv.x * 255.0, 0.0, 255.0));

                if ((gid_x % 2u) == 0u && (gid_y % 2u) == 0u) {
                    float r_sum = r, g_sum = g, b_sum = b;
                    float count = 1.0;

                    if (lx + 1u < 16u && gid_x + 1u < pc.width) {
                        uint t1  = tile[ly * 16u + lx + 1u];
                        uint sh1 = (((gid_x + 1u) * pc.input_channels) % 4u) * 8u;
                        r_sum += float((t1 >> sh1)          & 0xFF) / 255.0;
                        g_sum += float((t1 >> ((sh1 + 8u)  % 32u)) & 0xFF) / 255.0;
                        b_sum += float((t1 >> ((sh1 + 16u) % 32u)) & 0xFF) / 255.0;
                        count += 1.0;
                    }
                    if (ly + 1u < 16u && gid_y + 1u < pc.height) {
                        uint t2  = tile[(ly + 1u) * 16u + lx];
                        uint sh2 = ((gid_x * pc.input_channels) % 4u) * 8u;
                        r_sum += float((t2 >> sh2)          & 0xFF) / 255.0;
                        g_sum += float((t2 >> ((sh2 + 8u)  % 32u)) & 0xFF) / 255.0;
                        b_sum += float((t2 >> ((sh2 + 16u) % 32u)) & 0xFF) / 255.0;
                        count += 1.0;
                    }
                    if (lx + 1u < 16u && ly + 1u < 16u &&
                        gid_x + 1u < pc.width && gid_y + 1u < pc.height) {
                        uint t3  = tile[(ly + 1u) * 16u + lx + 1u];
                        uint sh3 = (((gid_x + 1u) * pc.input_channels) % 4u) * 8u;
                        r_sum += float((t3 >> sh3)          & 0xFF) / 255.0;
                        g_sum += float((t3 >> ((sh3 + 8u)  % 32u)) & 0xFF) / 255.0;
                        b_sum += float((t3 >> ((sh3 + 16u) % 32u)) & 0xFF) / 255.0;
                        count += 1.0;
                    }

                    vec3 avg_yuv = rgb_to_yuv(vec3(r_sum / count, g_sum / count, b_sum / count));
                    uint uv_idx = (gid_y / 2u) * (pc.width / 2u) + (gid_x / 2u);
                    output_u.data[uv_idx / 4] = uint(clamp(avg_yuv.y * 255.0, 0.0, 255.0));
                    output_v.data[uv_idx / 4] = uint(clamp(avg_yuv.z * 255.0, 0.0, 255.0));
                }
            }
        "
    }
}

/// `YUV420p` to RGB conversion compute shader with shared-memory Y-tile.
///
/// Each invocation loads the Y plane texel into a 256-element `shared` tile
/// before `barrier()`.  The colour transform reads Y from the tile; U/V are
/// read from global memory because they are already subsampled (one value per
/// 2×2 quad, so four threads share each U/V fetch — already coalesced).
///
/// Only compiled with the `vulkan-backend` feature: the `vulkano_shaders::shader!`
/// proc-macro compiles this GLSL to SPIR-V at build time via `shaderc-sys`.
#[cfg(feature = "vulkan-backend")]
#[allow(missing_docs)]
pub mod yuv420p_to_rgb {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(set = 0, binding = 0) buffer InputYBuffer {
                uint data[];
            } input_y;

            layout(set = 0, binding = 1) buffer InputUBuffer {
                uint data[];
            } input_u;

            layout(set = 0, binding = 2) buffer InputVBuffer {
                uint data[];
            } input_v;

            layout(set = 0, binding = 3) buffer OutputBuffer {
                uint data[];
            } output_buf;

            layout(push_constant) uniform PushConstants {
                uint width;
                uint height;
                uint output_channels;
            } pc;

            shared uint tile_y[256];

            vec3 yuv_to_rgb(vec3 yuv) {
                float y = yuv.x;
                float u = yuv.y - 0.5;
                float v = yuv.z - 0.5;
                float r = y + 1.402 * v;
                float g = y - 0.344 * u - 0.714 * v;
                float b = y + 1.772 * u;
                return clamp(vec3(r, g, b), 0.0, 1.0);
            }

            void main() {
                uint gid_x = gl_GlobalInvocationID.x;
                uint gid_y = gl_GlobalInvocationID.y;
                uint lx    = gl_LocalInvocationID.x;
                uint ly    = gl_LocalInvocationID.y;

                if (gid_x < pc.width && gid_y < pc.height) {
                    uint y_idx = gid_y * pc.width + gid_x;
                    tile_y[ly * 16u + lx] = input_y.data[y_idx / 4];
                } else {
                    tile_y[ly * 16u + lx] = 0u;
                }

                barrier();

                if (gid_x >= pc.width || gid_y >= pc.height) {
                    return;
                }

                float y = float(tile_y[ly * 16u + lx] & 0xFFu) / 255.0;

                uint uv_idx = (gid_y / 2u) * (pc.width / 2u) + (gid_x / 2u);
                float u = float(input_u.data[uv_idx / 4] & 0xFF) / 255.0;
                float v = float(input_v.data[uv_idx / 4] & 0xFF) / 255.0;

                vec3 rgb = yuv_to_rgb(vec3(y, u, v));

                uint out_idx = (gid_y * pc.width + gid_x) * pc.output_channels;
                uint r = uint(rgb.r * 255.0);
                uint g = uint(rgb.g * 255.0);
                uint b = uint(rgb.b * 255.0);

                if (pc.output_channels == 3u) {
                    output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16);
                } else {
                    output_buf.data[out_idx / 4] = r | (g << 8) | (b << 16) | (255u << 24);
                }
            }
        "
    }
}

#[cfg(test)]
mod tests {
    use super::{RGB_TO_YUV420P_GLSL, YUV420P_TO_RGB_GLSL};

    // --- rgb_to_yuv420p structural tests ---

    #[test]
    fn rgb_to_yuv420p_shader_has_shared_memory_tile() {
        assert!(
            RGB_TO_YUV420P_GLSL.contains("shared uint tile["),
            "rgb_to_yuv420p shader must declare a shared-memory tile"
        );
    }

    #[test]
    fn rgb_to_yuv420p_shader_has_workgroup_barrier() {
        assert!(
            RGB_TO_YUV420P_GLSL.contains("barrier()"),
            "rgb_to_yuv420p shader must call barrier() after cooperative tile load"
        );
    }

    #[test]
    fn rgb_to_yuv420p_shader_tile_is_256_elements() {
        // 16 * 16 = 256
        assert!(
            RGB_TO_YUV420P_GLSL.contains("shared uint tile[256]"),
            "rgb_to_yuv420p shared tile must be 256 elements (16×16)"
        );
    }

    #[test]
    fn rgb_to_yuv420p_shader_reads_from_tile_not_global() {
        // After barrier the shader must read 't = tile[...]' for the main pixel.
        assert!(
            RGB_TO_YUV420P_GLSL.contains("tile[ly * 16u + lx]"),
            "rgb_to_yuv420p must read centre texel from tile after barrier"
        );
    }

    #[test]
    fn rgb_to_yuv420p_shader_has_uv_subsampling_from_tile() {
        // U/V plane also reads from tile (right/bottom/corner neighbours).
        assert!(
            RGB_TO_YUV420P_GLSL.contains("tile[ly * 16u + lx + 1u]"),
            "rgb_to_yuv420p must read right-neighbour U/V quad from tile"
        );
    }

    // --- yuv420p_to_rgb structural tests ---

    #[test]
    fn yuv420p_to_rgb_shader_has_shared_memory_y_tile() {
        assert!(
            YUV420P_TO_RGB_GLSL.contains("shared uint tile_y["),
            "yuv420p_to_rgb shader must declare a shared Y tile"
        );
    }

    #[test]
    fn yuv420p_to_rgb_shader_has_workgroup_barrier() {
        assert!(
            YUV420P_TO_RGB_GLSL.contains("barrier()"),
            "yuv420p_to_rgb shader must call barrier() after Y tile load"
        );
    }

    #[test]
    fn yuv420p_to_rgb_shader_tile_y_is_256_elements() {
        assert!(
            YUV420P_TO_RGB_GLSL.contains("shared uint tile_y[256]"),
            "yuv420p_to_rgb Y tile must be 256 elements (16×16)"
        );
    }

    #[test]
    fn yuv420p_to_rgb_shader_reads_y_from_tile() {
        assert!(
            YUV420P_TO_RGB_GLSL.contains("tile_y[ly * 16u + lx]"),
            "yuv420p_to_rgb must read Y from shared tile after barrier"
        );
    }

    // --- cross-shader tests ---

    #[test]
    fn both_color_shaders_have_shared_memory() {
        assert!(
            RGB_TO_YUV420P_GLSL.contains("shared"),
            "rgb_to_yuv420p must use shared memory"
        );
        assert!(
            YUV420P_TO_RGB_GLSL.contains("shared"),
            "yuv420p_to_rgb must use shared memory"
        );
    }

    #[test]
    fn both_color_shaders_have_barrier() {
        assert!(
            RGB_TO_YUV420P_GLSL.contains("barrier()"),
            "rgb_to_yuv420p must have barrier()"
        );
        assert!(
            YUV420P_TO_RGB_GLSL.contains("barrier()"),
            "yuv420p_to_rgb must have barrier()"
        );
    }
}
