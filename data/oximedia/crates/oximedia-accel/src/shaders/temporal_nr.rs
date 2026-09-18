//! WGSL compute shader source for temporal noise reduction.
//!
//! Implements motion-adaptive IIR alpha-blending across up to N previous
//! frames.  For each pixel the absolute difference between current and prior
//! frame is normalised against `motion_threshold`; areas with little motion
//! get heavy temporal averaging, while moving areas are kept sharp.
//!
//! The shader uses a fixed three-buffer interface (cur + prev0 + prev1) and a
//! runtime `num_prev_frames` (0/1/2) field in the uniform to gate the
//! contribution of each prior buffer.  When `num_prev_frames < 2` the caller
//! must still bind a placeholder buffer for the unused slot — WGSL bindings
//! cannot be conditionally omitted.  The Rust dispatcher allocates a zero
//! buffer once and reuses it as the placeholder.

/// WGSL source for the temporal noise-reduction compute shader.
///
/// Entry point: `temporal_nr`.
pub const TEMPORAL_NR_WGSL: &str = r#"
struct TemporalParams {
    width: u32,
    height: u32,
    motion_threshold: f32,
    strength: f32,
    num_prev_frames: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> cur: array<f32>;
@group(0) @binding(1) var<storage, read> prev0: array<f32>;
@group(0) @binding(2) var<storage, read> prev1: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_buf: array<f32>;
@group(0) @binding(4) var<uniform> params: TemporalParams;

@compute @workgroup_size(16, 16, 1)
fn temporal_nr(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let c = cur[idx];
    var sum: f32 = c;
    var wsum: f32 = 1.0;
    if (params.num_prev_frames >= 1u) {
        let p = prev0[idx];
        let d = abs(c - p);
        let w = (1.0 - clamp(d / params.motion_threshold, 0.0, 1.0)) * params.strength;
        sum = sum + p * w;
        wsum = wsum + w;
    }
    if (params.num_prev_frames >= 2u) {
        let p = prev1[idx];
        let d = abs(c - p);
        let w = (1.0 - clamp(d / params.motion_threshold, 0.0, 1.0)) * params.strength;
        sum = sum + p * w;
        wsum = wsum + w;
    }
    output_buf[idx] = sum / wsum;
}
"#;

#[cfg(test)]
mod tests {
    use super::TEMPORAL_NR_WGSL;

    #[test]
    fn temporal_nr_wgsl_has_compute_entry() {
        assert!(
            TEMPORAL_NR_WGSL.contains("@compute") && TEMPORAL_NR_WGSL.contains("fn temporal_nr("),
            "temporal NR WGSL must define `temporal_nr` compute entry"
        );
    }

    #[test]
    fn temporal_nr_wgsl_uses_motion_threshold() {
        assert!(
            TEMPORAL_NR_WGSL.contains("motion_threshold"),
            "temporal NR WGSL must use motion_threshold"
        );
    }

    #[test]
    fn temporal_nr_wgsl_blends_prev_frames() {
        assert!(
            TEMPORAL_NR_WGSL.contains("prev0") && TEMPORAL_NR_WGSL.contains("prev1"),
            "temporal NR WGSL must blend two prev frames"
        );
    }

    #[test]
    fn temporal_nr_wgsl_clamps_motion_weight() {
        assert!(
            TEMPORAL_NR_WGSL.contains("clamp(d / params.motion_threshold, 0.0, 1.0)"),
            "temporal NR WGSL must clamp normalized motion weight to [0,1]"
        );
    }

    #[test]
    fn temporal_nr_wgsl_workgroup_is_16x16() {
        assert!(
            TEMPORAL_NR_WGSL.contains("@workgroup_size(16, 16, 1)"),
            "temporal NR WGSL must use 16×16×1 workgroup"
        );
    }

    #[test]
    fn temporal_nr_wgsl_normalizes_by_weight_sum() {
        assert!(
            TEMPORAL_NR_WGSL.contains("sum / wsum"),
            "temporal NR WGSL must normalize by weight sum"
        );
    }

    #[test]
    fn temporal_nr_wgsl_params_padded_to_32_bytes() {
        // 5 explicit fields + 3 pad = 32 bytes (multiple of 16) for uniform align.
        assert!(TEMPORAL_NR_WGSL.contains("_pad0: u32"));
        assert!(TEMPORAL_NR_WGSL.contains("_pad1: u32"));
        assert!(TEMPORAL_NR_WGSL.contains("_pad2: u32"));
    }

    #[test]
    fn temporal_nr_wgsl_has_output_storage_binding() {
        assert!(
            TEMPORAL_NR_WGSL.contains("output_buf: array<f32>"),
            "temporal NR WGSL must declare output_buf storage binding"
        );
    }

    #[test]
    fn temporal_nr_wgsl_uses_strength_scaling() {
        assert!(
            TEMPORAL_NR_WGSL.contains("* params.strength"),
            "temporal NR WGSL must scale weight by strength"
        );
    }
}
