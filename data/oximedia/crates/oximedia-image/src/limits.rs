//! Shared decode-safety limits (allocation / dimension ceilings).
//!
//! Image decoders size their output buffers from values taken directly out of
//! attacker-controlled headers. A handful of malformed header bytes can declare
//! enormous dimensions or element counts, turning a tiny input into either a
//! multi-gigabyte allocation (OOM / DoS) or — when the product overflows a
//! 32-bit intermediate — an under-allocated buffer that is then written out of
//! bounds. These helpers centralise the ceiling and checked-arithmetic guards.
//!
//! The functions return `Result<_, String>` (the `Err` payload being a
//! human-readable reason), so each caller adapts them to [`crate::error`] with a
//! single `.map_err(ImageError::invalid_format)?` (or another `String` variant).

/// Maximum accepted image dimension (width or height), in pixels.
///
/// Anything larger is treated as a malformed header.
pub const MAX_DIMENSION: usize = 16_384;

/// Maximum accepted size of a single decoded buffer, in bytes (4 GiB).
///
/// Stored as `u64` rather than `usize`: `1 << 32` overflows at const-eval time
/// on a 32-bit `usize` target such as `wasm32-unknown-unknown` (this crate
/// must build for wasm32 — it backs the browser demo surfaces), so the
/// ceiling is computed in a width that is always big enough and byte counts
/// are widened to `u64` at each comparison site instead (see
/// [`checked_dims`] and [`checked_alloc`]).
pub const MAX_ALLOC_BYTES: u64 = 1u64 << 32;

/// Validate a `width` × `height` image and return the element count
/// (`width * height * components`) computed with checked arithmetic.
///
/// Rejects (each defends a distinct malformed-header attack):
/// - a zero dimension (degenerate image that later divides by zero);
/// - either dimension above [`MAX_DIMENSION`] (allocation bomb);
/// - a zero component count (nothing to decode);
/// - an element count or byte size that overflows `usize` (integer-overflow
///   under-allocation → out-of-bounds write);
/// - a byte size above [`MAX_ALLOC_BYTES`] (allocation bomb).
///
/// `bytes_per` is the per-element byte size used only for the memory ceiling;
/// the returned value is the element count (not the byte size).
pub fn checked_dims(
    width: usize,
    height: usize,
    components: usize,
    bytes_per: usize,
) -> Result<usize, String> {
    // Defends against degenerate 0-sized images that later feed divisions.
    if width == 0 || height == 0 {
        return Err("zero image dimension".to_string());
    }
    // Defends against a header declaring an enormous frame (allocation bomb).
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(format!(
            "image dimension {width}x{height} exceeds ceiling {MAX_DIMENSION}"
        ));
    }
    // Defends against a header with a zero component count.
    if components == 0 {
        return Err("zero component count".to_string());
    }
    // Checked products defend against u32/usize wrap → under-allocation.
    let elements = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(components))
        .ok_or_else(|| "image element count overflow".to_string())?;
    let bytes = elements
        .checked_mul(bytes_per.max(1))
        .ok_or_else(|| "image byte size overflow".to_string())?;
    // Final memory ceiling. `bytes` is widened to `u64` for the comparison
    // since `MAX_ALLOC_BYTES` is `u64` (see its doc comment for why); the
    // widening is always lossless because `usize` never exceeds 64 bits on
    // any target this workspace builds for.
    if bytes as u64 > MAX_ALLOC_BYTES {
        return Err(format!(
            "declared image needs {bytes} bytes, exceeds ceiling {MAX_ALLOC_BYTES}"
        ));
    }
    Ok(elements)
}

/// Validate a `width` × `height` pair against the dimension ceiling only.
pub fn check_dimensions(width: usize, height: usize) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("zero image dimension".to_string());
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(format!(
            "image dimension {width}x{height} exceeds ceiling {MAX_DIMENSION}"
        ));
    }
    Ok(())
}

/// Validate a declared element `count` against the input bytes actually
/// remaining (`remaining`), where each element needs at least `min_elem_bytes`
/// of input.
///
/// Defends `Vec::with_capacity(count)` / `vec![_; count]` pre-allocations from a
/// tiny header that declares billions of entries the input cannot possibly
/// back: a count is rejected when `count * min_elem_bytes` overflows or exceeds
/// `remaining`. Returns the validated `count` for convenience.
pub fn checked_count(
    count: usize,
    min_elem_bytes: usize,
    remaining: usize,
) -> Result<usize, String> {
    let need = count
        .checked_mul(min_elem_bytes.max(1))
        .ok_or_else(|| "element count overflow".to_string())?;
    if need > remaining {
        return Err(format!(
            "declared {count} elements need {need} bytes but only {remaining} remain"
        ));
    }
    Ok(count)
}

/// Bound a `Vec::with_capacity` hint to what `remaining` input bytes can back.
///
/// Returns `min(count, remaining / min_elem_bytes)`. Use for pre-allocations
/// ahead of a read loop that already bounds each element access: a tiny header
/// declaring billions of entries then only reserves what the input could hold,
/// killing the allocation bomb without changing the values the loop produces
/// (the `Vec` still grows on demand). `min_elem_bytes` is the smallest number
/// of input bytes one element occupies.
#[must_use]
pub fn safe_capacity(count: usize, min_elem_bytes: usize, remaining: usize) -> usize {
    count.min(remaining / min_elem_bytes.max(1))
}

/// Validate a single-buffer byte size against [`MAX_ALLOC_BYTES`].
///
/// Used for raw sized reads (e.g. an EXR attribute payload) that are not laid
/// out as `width × height`.
pub fn checked_alloc(size: usize) -> Result<usize, String> {
    // See the comment in `checked_dims` for why `size` is widened to `u64`.
    if size as u64 > MAX_ALLOC_BYTES {
        return Err(format!(
            "declared allocation of {size} bytes exceeds ceiling {MAX_ALLOC_BYTES}"
        ));
    }
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_reasonable_dims() {
        assert_eq!(checked_dims(1920, 1080, 4, 1).unwrap(), 1920 * 1080 * 4);
        assert_eq!(checked_dims(64, 64, 1, 2).unwrap(), 64 * 64);
    }

    #[test]
    fn rejects_zero_and_oversized_dims() {
        assert!(checked_dims(0, 16, 3, 1).is_err());
        assert!(checked_dims(16, 0, 3, 1).is_err());
        assert!(checked_dims(MAX_DIMENSION + 1, 16, 3, 1).is_err());
        assert!(check_dimensions(MAX_DIMENSION + 1, 1).is_err());
    }

    #[test]
    fn rejects_zero_components() {
        assert!(checked_dims(16, 16, 0, 1).is_err());
    }

    #[test]
    fn rejects_alloc_bomb_within_dim_ceiling() {
        // 16384x16384x8 components x 4 bytes = 137 GiB > 4 GiB ceiling.
        assert!(checked_dims(MAX_DIMENSION, MAX_DIMENSION, 8, 4).is_err());
    }

    #[test]
    fn checked_count_rejects_unbacked_declaration() {
        assert!(checked_count(1_000_000_000, 8, 16).is_err());
        assert!(checked_count(usize::MAX, 8, usize::MAX).is_err());
        assert_eq!(checked_count(2, 8, 64).unwrap(), 2);
    }

    #[test]
    fn checked_alloc_enforces_ceiling() {
        // `MAX_ALLOC_BYTES + 1` only fits in a `usize` on a 64-bit host (it
        // exceeds `u32::MAX`, so no 32-bit `usize` could ever represent a
        // count this large in the first place — the cast is exact here
        // because `cargo test`/`nextest` run on the native 64-bit host, not
        // under wasm32).
        assert!(checked_alloc((MAX_ALLOC_BYTES + 1) as usize).is_err());
        assert_eq!(checked_alloc(1024).unwrap(), 1024);
    }

    #[test]
    fn max_alloc_bytes_is_exactly_four_gib() {
        // Locks in the documented ceiling: 4 GiB, and specifically the `u64`
        // representation that keeps `1u64 << 32` from overflowing at
        // const-eval time on a 32-bit `usize` target (wasm32-unknown-unknown).
        assert_eq!(MAX_ALLOC_BYTES, 4 * 1024 * 1024 * 1024);
        assert_eq!(MAX_ALLOC_BYTES, 1u64 << 32);
    }

    #[test]
    fn safe_capacity_bounds_to_input() {
        // A billion-entry declaration backed by only 64 bytes reserves at most
        // 64 / 8 = 8 slots, not a billion.
        assert_eq!(safe_capacity(1_000_000_000, 8, 64), 8);
        // A plausible count that the input can back is returned unchanged.
        assert_eq!(safe_capacity(4, 8, 64), 4);
        // Never divides by zero even when the element size is zero.
        assert_eq!(safe_capacity(5, 0, 10), 5);
    }
}
