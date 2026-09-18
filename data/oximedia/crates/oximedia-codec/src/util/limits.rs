//! Shared decode-safety limits (allocation / dimension ceilings).
//!
//! Image and video decoders size their output buffers from values taken
//! directly out of attacker-controlled headers. A handful of malformed header
//! bytes can therefore declare enormous dimensions or element counts, turning a
//! tiny input into either a multi-gigabyte allocation (OOM / DoS) or — when the
//! product overflows a 32-bit intermediate — an under-allocated buffer that is
//! then written out of bounds. These helpers centralise the ceiling and
//! checked-arithmetic guards so every decoder rejects such inputs uniformly.
//!
//! The functions return `Result<_, String>` (the `Err` payload being a
//! human-readable reason) rather than a concrete error type, so each codec can
//! adapt them to its own error enum with a single
//! `.map_err(MyError::Variant)?` — the JPEG-LS, JPEG 2000, JPEG XS, DNxHD and
//! generic `CodecError` paths all carry a `String` variant.

/// Maximum accepted image dimension (width or height), in pixels.
///
/// Matches the ceiling already enforced by the WebP and APV codecs in this
/// crate (`16_384`). Anything larger is treated as a malformed header.
pub const MAX_DIMENSION: usize = 16_384;

/// Maximum accepted size of a single decoded buffer, in bytes (4 GiB).
///
/// Rejects `pixels * bytes_per_sample` products that would exhaust memory even
/// when both dimensions individually sit under [`MAX_DIMENSION`].
///
/// Stored as `u64` rather than `usize`: `1 << 32` overflows at const-eval time
/// on a 32-bit `usize` target such as `wasm32-unknown-unknown` (this crate
/// must build for wasm32 — it backs the browser demo surfaces), so the
/// ceiling is computed in a width that is always big enough and byte counts
/// are widened to `u64` at the comparison site instead (see
/// [`checked_dims`]).
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
///
/// Use when the caller does its own element-count math but still needs to
/// reject a header that declares an impossibly large frame.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_reasonable_dims() {
        // A normal 1920x1080 RGBA frame is accepted and returns the pixel count.
        assert_eq!(checked_dims(1920, 1080, 4, 1).unwrap(), 1920 * 1080 * 4);
        assert_eq!(checked_dims(64, 64, 1, 2).unwrap(), 64 * 64);
    }

    #[test]
    fn rejects_zero_dims() {
        assert!(checked_dims(0, 16, 3, 1).is_err());
        assert!(checked_dims(16, 0, 3, 1).is_err());
        assert!(check_dimensions(0, 0).is_err());
    }

    #[test]
    fn rejects_oversized_dims() {
        assert!(checked_dims(MAX_DIMENSION + 1, 16, 3, 1).is_err());
        assert!(checked_dims(16, MAX_DIMENSION + 1, 3, 1).is_err());
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
        // 1 billion entries of >=8 bytes each cannot be backed by 16 input bytes.
        assert!(checked_count(1_000_000_000, 8, 16).is_err());
        // A count that overflows when multiplied by the element size.
        assert!(checked_count(usize::MAX, 8, usize::MAX).is_err());
        // A plausible count is accepted.
        assert_eq!(checked_count(2, 8, 64).unwrap(), 2);
    }
}
