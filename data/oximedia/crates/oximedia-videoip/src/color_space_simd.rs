//! SIMD-accelerated color space conversions for video transport hot paths.
//!
//! Provides:
//!
//! - `uyvy_to_planar_simd` — 8-bit 4:2:2 UYVY interleaved → separate Y, U, V planes.
//!   Uses `_mm_shuffle_epi8` (SSSE3) on x86_64 with runtime dispatch; scalar fallback
//!   on all other architectures.
//!
//! - `v210_to_planar` — 10-bit packed v210 format → 10-bit Y, Cb, Cr planes stored as
//!   `u16`.  A clean scalar implementation (SIMD for v210 requires complex repacking
//!   across 128-bit lanes and offers modest gains — left as a future optimisation).
//!
//! # UYVY format
//!
//! ```text
//! byte:  0    1    2    3    4    5    6    7   ...
//!       U0   Y0   V0   Y1   U1   Y2   V1   Y3  ...
//! ```
//! Two pixels share one U/V pair. Per line, `width` must be even.
//!
//! # v210 format
//!
//! Each 32-bit word packs three 10-bit components in little-endian bit order:
//!
//! ```text
//! word bits [9:0]   → Cb  (blue-difference chroma)
//! word bits [19:10] → Y   (luma)
//! word bits [29:20] → Cr  (red-difference chroma)
//! bits [31:30] are padding (always 0)
//! ```
//! Four words encode six pixels (twelve 10-bit components: 6×Y, 3×Cb, 3×Cr).

// ---------------------------------------------------------------------------
// UYVY → planar
// ---------------------------------------------------------------------------

/// Converts a UYVY (8-bit 4:2:2 interleaved) frame to separate Y, U, V planes.
///
/// On x86_64 with SSSE3 the inner loop processes 16 bytes (8 pixels) per
/// iteration using `_mm_shuffle_epi8`.  All other architectures use the scalar
/// fallback.
///
/// # Arguments
///
/// * `src`    — UYVY byte slice; length must equal `width * height * 2`.
/// * `width`  — Frame width in pixels (must be even).
/// * `height` — Frame height in lines.
///
/// # Returns
///
/// `(y_plane, u_plane, v_plane)` where each plane has `width * height / 2`
/// elements for U/V and `width * height` elements for Y.
///
/// # Panics
///
/// Panics if `src.len() != width * height * 2` or `width % 2 != 0`.
pub fn uyvy_to_planar_simd(src: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert_eq!(
        src.len(),
        width * height * 2,
        "UYVY src length mismatch: expected {}, got {}",
        width * height * 2,
        src.len()
    );
    assert_eq!(width % 2, 0, "width must be even for UYVY");

    let npix = width * height;
    let nchroma = npix / 2;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            // SAFETY: we just confirmed ssse3 is available via runtime detection.
            #[allow(unsafe_code)]
            return unsafe { uyvy_to_planar_ssse3(src, npix, nchroma) };
        }
    }

    uyvy_to_planar_scalar(src, npix, nchroma)
}

/// Scalar UYVY → planar fallback (all platforms).
fn uyvy_to_planar_scalar(src: &[u8], npix: usize, nchroma: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = Vec::with_capacity(npix);
    let mut u = Vec::with_capacity(nchroma);
    let mut v = Vec::with_capacity(nchroma);

    // UYVY order: U0 Y0 V0 Y1, U1 Y2 V1 Y3, …
    let mut i = 0usize;
    while i + 3 < src.len() {
        u.push(src[i]);
        y.push(src[i + 1]);
        v.push(src[i + 2]);
        y.push(src[i + 3]);
        i += 4;
    }

    (y, u, v)
}

/// SSSE3-accelerated UYVY → planar path.
///
/// Processes 16 UYVY bytes (8 pixels) per iteration using two
/// `_mm_shuffle_epi8` masks — one to gather Y bytes, one to gather
/// interleaved U/V bytes — then deinterleaves U and V with pshufb.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)]
#[target_feature(enable = "ssse3")]
unsafe fn uyvy_to_planar_ssse3(
    src: &[u8],
    npix: usize,
    nchroma: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use std::arch::x86_64::*;

    let mut y = Vec::with_capacity(npix);
    let mut u = Vec::with_capacity(nchroma);
    let mut v = Vec::with_capacity(nchroma);

    // Mask to extract Y bytes from a 16-byte UYVY group (4 pixels × UYVY):
    // bytes 1,3,5,7,9,11,13,15 → positions 0-7 of result, rest unused.
    #[rustfmt::skip]
    let y_mask = _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        15, 13, 11, 9, 7, 5, 3, 1,
    );
    // Mask to extract U bytes: bytes 0,4,8,12 → positions 0-3.
    #[rustfmt::skip]
    let u_mask = _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, 12, 8, 4, 0,
    );
    // Mask to extract V bytes: bytes 2,6,10,14 → positions 0-3.
    #[rustfmt::skip]
    let v_mask = _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, 14, 10, 6, 2,
    );

    let chunks = src.len() / 16; // each 16 UYVY bytes = 8 pixels
    let mut offset = 0usize;

    for _ in 0..chunks {
        // SAFETY: offset is within src.len() (chunks * 16 <= src.len()).
        let chunk = _mm_loadu_si128(src.as_ptr().add(offset).cast::<__m128i>());

        // Extract Y (8 bytes) into the low 64 bits.
        let y_vec = _mm_shuffle_epi8(chunk, y_mask);
        // Extract U (4 bytes) into the low 32 bits.
        let u_vec = _mm_shuffle_epi8(chunk, u_mask);
        // Extract V (4 bytes) into the low 32 bits.
        let v_vec = _mm_shuffle_epi8(chunk, v_mask);

        // Store via stack buffers to avoid unsafe pointer writes into Vec.
        let mut y_buf = [0u8; 16];
        let mut uv_buf = [0u8; 16];
        _mm_storeu_si128(y_buf.as_mut_ptr().cast::<__m128i>(), y_vec);
        _mm_storeu_si128(uv_buf.as_mut_ptr().cast::<__m128i>(), u_vec);
        y.extend_from_slice(&y_buf[..8]);
        u.extend_from_slice(&uv_buf[..4]);

        _mm_storeu_si128(uv_buf.as_mut_ptr().cast::<__m128i>(), v_vec);
        v.extend_from_slice(&uv_buf[..4]);

        offset += 16;
    }

    // Scalar tail for any remaining bytes (< 16).
    if offset < src.len() {
        let (y_tail, u_tail, v_tail) = uyvy_to_planar_scalar(
            &src[offset..],
            src.len() / 2 - y.len(),
            (src.len() / 4) - u.len(),
        );
        y.extend(y_tail);
        u.extend(u_tail);
        v.extend(v_tail);
    }

    (y, u, v)
}

// ---------------------------------------------------------------------------
// v210 → planar
// ---------------------------------------------------------------------------

/// Converts a v210 (10-bit packed 4:2:2) frame to 10-bit Y, Cb, Cr planes.
///
/// v210 stores three 10-bit samples per 32-bit word in a repeating pattern:
///
/// ```text
/// word 0: Cb0[9:0] | Y0[9:0] << 10 | Cr0[9:0] << 20
/// word 1: Cb1[9:0] | Y1[9:0] << 10 | Cr1[9:0] << 20
/// ...
/// ```
///
/// Every group of four 32-bit words encodes two luma + one chroma pair
/// (four words = 12 samples = 6Y + 3Cb + 3Cr before reordering — see the
/// Apple/AJA v210 spec for the exact pattern used here, which packs
/// Cb0 Y0 Cr0 | Cb1 Y1 Cr1 | ... with one sample per 10-bit triplet).
///
/// # Arguments
///
/// * `src`    — Packed v210 words; length must equal `width * height * 2 / 3` words
///              (each word encodes 30 usable bits across two pixels).
/// * `width`  — Frame width in pixels (must be a multiple of 6 for v210).
/// * `height` — Frame height in lines.
///
/// # Returns
///
/// `(y_plane, cb_plane, cr_plane)` as `Vec<u16>` with 10-bit values in [0, 1023].
///
/// # Panics
///
/// Panics if `width % 6 != 0` or if `src.len()` is inconsistent.
pub fn v210_to_planar(src: &[u32], width: usize, height: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    assert_eq!(width % 6, 0, "v210 width must be a multiple of 6");
    // v210 packs 3 samples per word: every 6-pixel group uses 4 words.
    // words_per_line = (width / 6) * 4
    let words_per_line = (width / 6) * 4;
    assert_eq!(
        src.len(),
        words_per_line * height,
        "v210 src length mismatch: expected {}, got {}",
        words_per_line * height,
        src.len()
    );

    let npix = width * height;
    let nchroma = npix / 2;

    let mut y = Vec::with_capacity(npix);
    let mut cb = Vec::with_capacity(nchroma);
    let mut cr = Vec::with_capacity(nchroma);

    // Each group of 4 words encodes 6 pixels (6 Y + 3 Cb + 3 Cr).
    // v210 word layout (bits):
    //   word[0]: [9:0]=Cb0  [19:10]=Y0  [29:20]=Cr0
    //   word[1]: [9:0]=Cb1  [19:10]=Y1  [29:20]=Cr1
    //   word[2]: [9:0]=Cb2  [19:10]=Y2  [29:20]=Cr2
    //   word[3]: [9:0]=Cb3  [19:10]=Y3  [29:20]=Cr3
    // Wait — that is the *simplified* 1-word-per-pixel picture; the actual
    // v210 specification has a more complex interleaving.  The canonical layout
    // (as used by Apple Final Cut and AJA hardware) is:
    //
    //   word[0]: Cb0[9:0] | Y0[9:0] << 10 | Cr0[9:0] << 20
    //   word[1]: Cb1[9:0] | Y1[9:0] << 10 | Cr1[9:0] << 20
    //   word[2]: Cb2[9:0] | Y2[9:0] << 10 | Cr2[9:0] << 20
    //   word[3]: Cb3[9:0] | Y3[9:0] << 10 | Cr3[9:0] << 20
    //
    // This matches the FFmpeg/libav v210 decoder that processes 6 pixels per 4
    // words, yielding 6 Y, 3 Cb, 3 Cr values:
    //   Y  = [Y0, Y1, Y2, Y3, Y4, Y5]
    //   Cb = [Cb0, Cb2, Cb4]  (sub-sampled 4:2:2)
    //   Cr = [Cr0, Cr2, Cr4]
    //
    // Actually the FFmpeg canonical mapping uses a 4-word / 6-pixel group:
    //   w0: Cb0 | Y0 << 10 | Cr0 << 20
    //   w1: Y1  | Cb1 << 10 | Y2 << 20
    //   w2: Cr1 | Y3 << 10 | Cb2 << 20
    //   w3: Y4  | Cr2 << 10 | Y5 << 20
    //
    // We implement that FFmpeg-compatible layout below.

    let mut i = 0usize;
    while i + 3 < src.len() {
        let w0 = src[i];
        let w1 = src[i + 1];
        let w2 = src[i + 2];
        let w3 = src[i + 3];

        // Extract 6 Y samples.
        let y0 = ((w0 >> 10) & 0x3FF) as u16;
        let y1 = (w1 & 0x3FF) as u16;
        let y2 = ((w1 >> 20) & 0x3FF) as u16;
        let y3 = ((w2 >> 10) & 0x3FF) as u16;
        let y4 = (w3 & 0x3FF) as u16;
        let y5 = ((w3 >> 20) & 0x3FF) as u16;

        // Extract 3 Cb samples (one per 2 Y samples → 4:2:2).
        let cb0 = (w0 & 0x3FF) as u16;
        let cb1 = ((w1 >> 10) & 0x3FF) as u16;
        let cb2 = ((w2 >> 20) & 0x3FF) as u16;

        // Extract 3 Cr samples.
        let cr0 = ((w0 >> 20) & 0x3FF) as u16;
        let cr1 = (w2 & 0x3FF) as u16;
        let cr2 = ((w3 >> 10) & 0x3FF) as u16;

        y.extend_from_slice(&[y0, y1, y2, y3, y4, y5]);
        cb.extend_from_slice(&[cb0, cb1, cb2]);
        cr.extend_from_slice(&[cr0, cr1, cr2]);

        i += 4;
    }

    (y, cb, cr)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Item 5 required tests ─────────────────────────────────────────────────

    /// Verify that the SIMD path produces the same output as the scalar path.
    #[test]
    fn test_uyvy_to_planar_simd_matches_scalar() {
        // Generate a synthetic 16×4 UYVY frame (width must be even, height any).
        let width = 16usize;
        let height = 4usize;
        let npix = width * height;

        // Fill with a pseudo-random pattern.
        let mut src = Vec::with_capacity(npix * 2);
        for i in 0..(npix / 2) {
            // UYVY pattern: vary all components to exercise all byte positions.
            let u = (i * 7 % 256) as u8;
            let y0 = (i * 13 % 256) as u8;
            let v = (i * 19 % 256) as u8;
            let y1 = (i * 23 % 256) as u8;
            src.push(u);
            src.push(y0);
            src.push(v);
            src.push(y1);
        }

        let (y_scalar, u_scalar, v_scalar) = uyvy_to_planar_scalar(&src, npix, npix / 2);

        let (y_simd, u_simd, v_simd) = uyvy_to_planar_simd(&src, width, height);

        assert_eq!(y_simd, y_scalar, "Y planes differ");
        assert_eq!(u_simd, u_scalar, "U planes differ");
        assert_eq!(v_simd, v_scalar, "V planes differ");
    }

    /// Verify v210 decoding against known byte values.
    #[test]
    fn test_v210_to_planar_known_values() {
        // Craft a single 4-word group (6 pixels) with known component values.
        // Layout (FFmpeg-compatible):
        //   w0: Cb0 | Y0 << 10 | Cr0 << 20
        //   w1: Y1  | Cb1 << 10 | Y2 << 20
        //   w2: Cr1 | Y3 << 10 | Cb2 << 20
        //   w3: Y4  | Cr2 << 10 | Y5 << 20

        let cb0: u32 = 0x100; // 256
        let y0: u32 = 0x200; // 512
        let cr0: u32 = 0x300; // 768

        let y1: u32 = 0x040; // 64
        let cb1: u32 = 0x080; // 128
        let y2: u32 = 0x0C0; // 192

        let cr1: u32 = 0x110; // 272
        let y3: u32 = 0x150; // 336
        let cb2: u32 = 0x190; // 400

        let y4: u32 = 0x020; // 32
        let cr2: u32 = 0x060; // 96
        let y5: u32 = 0x0A0; // 160

        let w0 = cb0 | (y0 << 10) | (cr0 << 20);
        let w1 = y1 | (cb1 << 10) | (y2 << 20);
        let w2 = cr1 | (y3 << 10) | (cb2 << 20);
        let w3 = y4 | (cr2 << 10) | (y5 << 20);

        let src = [w0, w1, w2, w3];
        let (y, cb, cr) = v210_to_planar(&src, 6, 1);

        // Y values.
        assert_eq!(y[0], 0x200, "Y0 mismatch");
        assert_eq!(y[1], 0x040, "Y1 mismatch");
        assert_eq!(y[2], 0x0C0, "Y2 mismatch");
        assert_eq!(y[3], 0x150, "Y3 mismatch");
        assert_eq!(y[4], 0x020, "Y4 mismatch");
        assert_eq!(y[5], 0x0A0, "Y5 mismatch");

        // Cb values.
        assert_eq!(cb[0], 0x100, "Cb0 mismatch");
        assert_eq!(cb[1], 0x080, "Cb1 mismatch");
        assert_eq!(cb[2], 0x190, "Cb2 mismatch");

        // Cr values.
        assert_eq!(cr[0], 0x300, "Cr0 mismatch");
        assert_eq!(cr[1], 0x110, "Cr1 mismatch");
        assert_eq!(cr[2], 0x060, "Cr2 mismatch");
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_uyvy_planar_output_sizes() {
        let width = 8usize;
        let height = 2usize;
        let src = vec![0u8; width * height * 2];
        let (y, u, v) = uyvy_to_planar_simd(&src, width, height);
        assert_eq!(y.len(), width * height);
        assert_eq!(u.len(), width * height / 2);
        assert_eq!(v.len(), width * height / 2);
    }

    #[test]
    fn test_v210_planar_output_sizes() {
        let width = 6usize;
        let height = 2usize;
        let words_per_line = (width / 6) * 4;
        let src = vec![0u32; words_per_line * height];
        let (y, cb, cr) = v210_to_planar(&src, width, height);
        assert_eq!(y.len(), width * height);
        assert_eq!(cb.len(), width * height / 2);
        assert_eq!(cr.len(), width * height / 2);
    }

    #[test]
    fn test_v210_10bit_values_in_range() {
        // All-ones in the 10-bit fields should yield 0x3FF.
        let w = 0x3FF | (0x3FF << 10) | (0x3FF << 20);
        let src = [w, w, w, w];
        let (y, cb, cr) = v210_to_planar(&src, 6, 1);
        for &val in y.iter().chain(cb.iter()).chain(cr.iter()) {
            assert!(val <= 0x3FF, "value {val} exceeds 10-bit range");
        }
    }
}
