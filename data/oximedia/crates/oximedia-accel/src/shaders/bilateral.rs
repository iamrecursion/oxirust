//! WGSL compute shader source for bilateral filter (spatial noise reduction).
//!
//! The shader implements an edge-preserving bilateral filter with cooperative
//! **shared-memory tiling**: each 16×16 workgroup cooperatively loads a 32×32
//! tile (16 + 2×8 halo) into a 1024-entry `workgroup`-scoped array before a
//! `workgroupBarrier()`.  The per-thread filter then reads only the tile,
//! eliminating repeated global-buffer fetches.
//!
//! The shader operates on a single-channel f32 buffer (one filter pass per
//! component).  Maximum supported kernel radius is **8 pixels** (set by the
//! 8-pixel halo); the Rust dispatcher [`crate::bilateral_gpu`] enforces this
//! at submission time.
//!
//! # Bindings
//! - `@binding(0)` — `input_buf: array<f32>` (read-only storage)
//! - `@binding(1)` — `output_buf: array<f32>` (read/write storage)
//! - `@binding(2)` — `params: BilateralParams` (uniform; 8 × u32/f32 = 32 B)

/// WGSL source for the bilateral filter compute shader (`bilateral` entry).
///
/// Tests verify presence of `workgroupBarrier()`, the 32×32 shared tile, and
/// the halo cooperative load — these are structural correctness markers.
pub const BILATERAL_WGSL: &str = r#"
struct BilateralParams {
    width: u32,
    height: u32,
    sigma_color: f32,
    sigma_space: f32,
    kernel_radius: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: BilateralParams;

const TILE_DIM: u32 = 16u;
const HALO_MAX: u32 = 8u;
const TILE_PADDED: u32 = 32u;  // TILE_DIM + 2*HALO_MAX

var<workgroup> tile: array<f32, 1024>;

fn sample_clamped(x: i32, y: i32) -> f32 {
    let xx = clamp(x, 0, i32(params.width) - 1);
    let yy = clamp(y, 0, i32(params.height) - 1);
    return input_buf[u32(yy) * params.width + u32(xx)];
}

@compute @workgroup_size(16, 16, 1)
fn bilateral(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wgid: vec3<u32>) {
    // Cooperative load of 32×32 tile (16×16 wg + 8-px halo on each side).
    // Each thread loads a 2×2 quad so all 1024 texels are populated.
    let tile_origin_x = i32(wgid.x * TILE_DIM) - i32(HALO_MAX);
    let tile_origin_y = i32(wgid.y * TILE_DIM) - i32(HALO_MAX);
    for (var dy: u32 = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
            let lx = lid.x * 2u + dx;
            let ly = lid.y * 2u + dy;
            if (lx < TILE_PADDED && ly < TILE_PADDED) {
                tile[ly * TILE_PADDED + lx] = sample_clamped(
                    tile_origin_x + i32(lx), tile_origin_y + i32(ly));
            }
        }
    }
    workgroupBarrier();

    // Per-thread bilateral filter using only the shared tile.
    let center_local_x = lid.x + HALO_MAX;
    let center_local_y = lid.y + HALO_MAX;
    let center = tile[center_local_y * TILE_PADDED + center_local_x];

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;
    let r = params.kernel_radius;
    let inv2sc2 = 1.0 / (2.0 * params.sigma_color * params.sigma_color);
    let inv2ss2 = 1.0 / (2.0 * params.sigma_space * params.sigma_space);

    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            let lx = i32(center_local_x) + dx;
            let ly = i32(center_local_y) + dy;
            let neighbor = tile[u32(ly) * TILE_PADDED + u32(lx)];
            let dc = center - neighbor;
            let color_w = exp(-(dc * dc) * inv2sc2);
            let space_w = exp(-f32(dx * dx + dy * dy) * inv2ss2);
            let w = color_w * space_w;
            sum = sum + neighbor * w;
            wsum = wsum + w;
        }
    }

    if (gid.x < params.width && gid.y < params.height) {
        output_buf[gid.y * params.width + gid.x] = sum / wsum;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::BILATERAL_WGSL;

    #[test]
    fn bilateral_wgsl_has_workgroup_barrier() {
        assert!(
            BILATERAL_WGSL.contains("workgroupBarrier()"),
            "bilateral WGSL must call workgroupBarrier() after cooperative tile load"
        );
    }

    #[test]
    fn bilateral_wgsl_has_workgroup_shared_tile() {
        assert!(
            BILATERAL_WGSL.contains("var<workgroup> tile: array<f32, 1024>"),
            "bilateral WGSL must declare a 1024-entry workgroup-scoped tile"
        );
    }

    #[test]
    fn bilateral_wgsl_tile_size_matches_halo() {
        // 32 = 16 (workgroup) + 2*8 (halo). 32*32 = 1024.
        assert!(BILATERAL_WGSL.contains("TILE_PADDED: u32 = 32u"));
        assert!(BILATERAL_WGSL.contains("HALO_MAX: u32 = 8u"));
        assert!(BILATERAL_WGSL.contains("TILE_DIM: u32 = 16u"));
    }

    #[test]
    fn bilateral_wgsl_has_cooperative_load() {
        // Each thread loads 2×2 = 4 texels so the 32×32 tile is fully populated.
        assert!(
            BILATERAL_WGSL.contains("dy < 2u") && BILATERAL_WGSL.contains("dx < 2u"),
            "bilateral WGSL must cooperatively load 2×2 quad per thread"
        );
    }

    #[test]
    fn bilateral_wgsl_has_color_and_space_weights() {
        assert!(
            BILATERAL_WGSL.contains("color_w") && BILATERAL_WGSL.contains("space_w"),
            "bilateral WGSL must compute both color and space weights"
        );
    }

    #[test]
    fn bilateral_wgsl_normalizes_output() {
        assert!(
            BILATERAL_WGSL.contains("sum / wsum"),
            "bilateral WGSL must normalize by weight sum"
        );
    }

    #[test]
    fn bilateral_wgsl_uses_exp_for_gaussian() {
        assert!(
            BILATERAL_WGSL.contains("exp(-"),
            "bilateral WGSL must use Gaussian (exp) weights"
        );
    }

    #[test]
    fn bilateral_wgsl_has_compute_entry_point() {
        assert!(
            BILATERAL_WGSL.contains("@compute") && BILATERAL_WGSL.contains("fn bilateral("),
            "bilateral WGSL must define `bilateral` compute entry point"
        );
    }

    #[test]
    fn bilateral_wgsl_workgroup_is_16x16() {
        assert!(
            BILATERAL_WGSL.contains("@workgroup_size(16, 16, 1)"),
            "bilateral WGSL must use 16×16×1 workgroup"
        );
    }

    #[test]
    fn bilateral_wgsl_params_struct_is_32_bytes() {
        // BilateralParams must be padded to 32 bytes (multiple of 16) for uniform.
        // Has _pad0/_pad1/_pad2 to round 5×4 → 8×4 = 32 bytes.
        assert!(BILATERAL_WGSL.contains("_pad0: i32"));
        assert!(BILATERAL_WGSL.contains("_pad1: i32"));
        assert!(BILATERAL_WGSL.contains("_pad2: i32"));
    }
}
