//! Motion estimation compute shaders.

/// Block-based motion estimation using Sum of Absolute Differences (SAD).
///
/// Only compiled with the `vulkan-backend` feature: the `vulkano_shaders::shader!`
/// proc-macro compiles this GLSL to SPIR-V at build time via `shaderc-sys`.
#[cfg(feature = "vulkan-backend")]
#[allow(missing_docs)]
pub mod block_sad {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 8, local_size_y = 8) in;

            layout(set = 0, binding = 0) buffer ReferenceBuffer {
                uint data[];
            } ref_buf;

            layout(set = 0, binding = 1) buffer CurrentBuffer {
                uint data[];
            } cur_buf;

            layout(set = 0, binding = 2) buffer MotionVectorBuffer {
                ivec2 data[];
            } mv_buf;

            layout(push_constant) uniform PushConstants {
                uint width;
                uint height;
                uint block_size;
                uint search_range;
            } pc;

            uint calculate_sad(uint ref_x, uint ref_y, uint cur_x, uint cur_y) {
                uint sad = 0;

                for (uint dy = 0; dy < pc.block_size; dy++) {
                    for (uint dx = 0; dx < pc.block_size; dx++) {
                        uint ry = ref_y + dy;
                        uint rx = ref_x + dx;
                        uint cy = cur_y + dy;
                        uint cx = cur_x + dx;

                        if (rx >= pc.width || ry >= pc.height ||
                            cx >= pc.width || cy >= pc.height) {
                            continue;
                        }

                        uint ref_idx = ry * pc.width + rx;
                        uint cur_idx = cy * pc.width + cx;

                        uint ref_val = ref_buf.data[ref_idx / 4] & 0xFF;
                        uint cur_val = cur_buf.data[cur_idx / 4] & 0xFF;

                        sad += abs(int(ref_val) - int(cur_val));
                    }
                }

                return sad;
            }

            void main() {
                uint block_x = gl_GlobalInvocationID.x;
                uint block_y = gl_GlobalInvocationID.y;

                uint blocks_wide = (pc.width + pc.block_size - 1) / pc.block_size;
                uint blocks_high = (pc.height + pc.block_size - 1) / pc.block_size;

                if (block_x >= blocks_wide || block_y >= blocks_high) {
                    return;
                }

                uint cur_x = block_x * pc.block_size;
                uint cur_y = block_y * pc.block_size;

                int best_dx = 0;
                int best_dy = 0;
                uint best_sad = 0xFFFFFFFF;

                // Search within range
                int search = int(pc.search_range);
                for (int dy = -search; dy <= search; dy++) {
                    for (int dx = -search; dx <= search; dx++) {
                        int ref_x = int(cur_x) + dx;
                        int ref_y = int(cur_y) + dy;

                        if (ref_x < 0 || ref_y < 0 ||
                            uint(ref_x) + pc.block_size > pc.width ||
                            uint(ref_y) + pc.block_size > pc.height) {
                            continue;
                        }

                        uint sad = calculate_sad(
                            uint(ref_x), uint(ref_y),
                            cur_x, cur_y
                        );

                        if (sad < best_sad) {
                            best_sad = sad;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }

                uint mv_idx = block_y * blocks_wide + block_x;
                mv_buf.data[mv_idx] = ivec2(best_dx, best_dy);
            }
        "
    }
}
