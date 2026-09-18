//! Audio channel interleaving and de-interleaving.
//!
//! Provides fast conversion between planar (non-interleaved) and interleaved
//! multi-channel audio formats.  These operations are fundamental to feeding
//! audio hardware, encoding pipelines, and SIMD processing kernels.

#![allow(dead_code)]

/// Interleave two planar mono channels into a stereo interleaved buffer.
///
/// Given channel `a` (`[a0, a1, …]`) and channel `b` (`[b0, b1, …]`),
/// produces `[a0, b0, a1, b1, …]`.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn interleave_2ch(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "channel lengths must be equal");
    let mut out = Vec::with_capacity(a.len() * 2);
    for (&sa, &sb) in a.iter().zip(b.iter()) {
        out.push(sa);
        out.push(sb);
    }
    out
}

/// De-interleave a stereo interleaved buffer into two planar mono channels.
///
/// Given `[a0, b0, a1, b1, …]`, produces `([a0, a1, …], [b0, b1, …])`.
///
/// # Panics
///
/// Panics if the buffer length is not even.
#[must_use]
pub fn deinterleave_2ch(interleaved: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert!(
        interleaved.len().is_multiple_of(2),
        "interleaved buffer length must be even"
    );
    let n = interleaved.len() / 2;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in interleaved.chunks_exact(2) {
        a.push(chunk[0]);
        b.push(chunk[1]);
    }
    (a, b)
}

/// Interleave four planar mono channels into a quad interleaved buffer.
///
/// Produces `[a0, b0, c0, d0, a1, b1, c1, d1, …]`.
///
/// # Panics
///
/// Panics if the channels have different lengths.
#[must_use]
pub fn interleave_4ch(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "channel lengths must be equal");
    assert_eq!(b.len(), c.len(), "channel lengths must be equal");
    assert_eq!(c.len(), d.len(), "channel lengths must be equal");
    let mut out = Vec::with_capacity(a.len() * 4);
    for i in 0..a.len() {
        out.push(a[i]);
        out.push(b[i]);
        out.push(c[i]);
        out.push(d[i]);
    }
    out
}

/// De-interleave a quad interleaved buffer into four planar mono channels.
///
/// Given `[a0, b0, c0, d0, a1, b1, c1, d1, …]`,
/// produces `([a…], [b…], [c…], [d…])`.
///
/// # Panics
///
/// Panics if the buffer length is not a multiple of 4.
#[must_use]
pub fn deinterleave_4ch(interleaved: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(
        interleaved.len().is_multiple_of(4),
        "interleaved buffer length must be a multiple of 4"
    );
    let n = interleaved.len() / 4;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for chunk in interleaved.chunks_exact(4) {
        a.push(chunk[0]);
        b.push(chunk[1]);
        c.push(chunk[2]);
        d.push(chunk[3]);
    }
    (a, b, c, d)
}

/// Generic N-channel interleaver.
///
/// `channels` is a slice of planar buffers, all the same length.
/// Returns a single interleaved buffer of length `channels.len() * frame_count`.
///
/// # Panics
///
/// Panics if `channels` is empty or any channel has a different length.
#[must_use]
pub fn interleave_nch(channels: &[&[f32]]) -> Vec<f32> {
    assert!(!channels.is_empty(), "must have at least one channel");
    let frame_count = channels[0].len();
    for ch in channels.iter().skip(1) {
        assert_eq!(
            ch.len(),
            frame_count,
            "all channels must have the same length"
        );
    }
    let num_channels = channels.len();
    let mut out = Vec::with_capacity(frame_count * num_channels);
    for frame in 0..frame_count {
        for ch in channels {
            out.push(ch[frame]);
        }
    }
    out
}

/// Generic N-channel de-interleaver.
///
/// `num_channels` must divide `interleaved.len()` evenly.
/// Returns a `Vec` of `num_channels` planar buffers.
///
/// # Panics
///
/// Panics if `num_channels` is 0 or `interleaved.len()` is not divisible by `num_channels`.
#[must_use]
pub fn deinterleave_nch(interleaved: &[f32], num_channels: usize) -> Vec<Vec<f32>> {
    assert!(num_channels > 0, "num_channels must be > 0");
    assert!(
        interleaved.len().is_multiple_of(num_channels),
        "buffer length must be divisible by num_channels"
    );
    let frame_count = interleaved.len() / num_channels;
    let mut channels: Vec<Vec<f32>> = (0..num_channels)
        .map(|_| Vec::with_capacity(frame_count))
        .collect();
    for chunk in interleaved.chunks_exact(num_channels) {
        for (ch_idx, &sample) in chunk.iter().enumerate() {
            channels[ch_idx].push(sample);
        }
    }
    channels
}

/// Convert a stereo interleaved buffer to dual-mono by averaging L+R.
#[must_use]
pub fn stereo_to_mono(interleaved: &[f32]) -> Vec<f32> {
    interleaved
        .chunks_exact(2)
        .map(|c| (c[0] + c[1]) * 0.5)
        .collect()
}

/// Duplicate a mono buffer into a stereo interleaved buffer.
#[must_use]
pub fn mono_to_stereo(mono: &[f32]) -> Vec<f32> {
    interleave_2ch(mono, mono)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleave_2ch_basic() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let out = interleave_2ch(&a, &b);
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_deinterleave_2ch_basic() {
        let interleaved = [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let (a, b) = deinterleave_2ch(&interleaved);
        assert_eq!(a, [1.0, 2.0, 3.0]);
        assert_eq!(b, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_interleave_deinterleave_2ch_roundtrip() {
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (64..128).map(|i| i as f32).collect();
        let interleaved = interleave_2ch(&a, &b);
        let (ra, rb) = deinterleave_2ch(&interleaved);
        assert_eq!(ra, a);
        assert_eq!(rb, b);
    }

    #[test]
    fn test_interleave_4ch_basic() {
        let a = [1.0_f32];
        let b = [2.0_f32];
        let c = [3.0_f32];
        let d = [4.0_f32];
        let out = interleave_4ch(&a, &b, &c, &d);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_deinterleave_4ch_basic() {
        let interleaved = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (a, b, c, d) = deinterleave_4ch(&interleaved);
        assert_eq!(a, [1.0, 5.0]);
        assert_eq!(b, [2.0, 6.0]);
        assert_eq!(c, [3.0, 7.0]);
        assert_eq!(d, [4.0, 8.0]);
    }

    #[test]
    fn test_interleave_deinterleave_4ch_roundtrip() {
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let b: Vec<f32> = (32..64).map(|i| i as f32).collect();
        let c: Vec<f32> = (64..96).map(|i| i as f32).collect();
        let d: Vec<f32> = (96..128).map(|i| i as f32).collect();
        let interleaved = interleave_4ch(&a, &b, &c, &d);
        let (ra, rb, rc, rd) = deinterleave_4ch(&interleaved);
        assert_eq!(ra, a);
        assert_eq!(rb, b);
        assert_eq!(rc, c);
        assert_eq!(rd, d);
    }

    #[test]
    fn test_interleave_nch_single() {
        let ch = [1.0_f32, 2.0, 3.0];
        let out = interleave_nch(&[&ch]);
        assert_eq!(out, ch);
    }

    #[test]
    fn test_interleave_nch_three_channels() {
        let a = [1.0_f32, 4.0];
        let b = [2.0_f32, 5.0];
        let c = [3.0_f32, 6.0];
        let out = interleave_nch(&[&a, &b, &c]);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_deinterleave_nch_roundtrip() {
        let channels: Vec<Vec<f32>> = (0..6_usize)
            .map(|ch| (0..20_usize).map(|f| (ch * 20 + f) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = channels.iter().map(Vec::as_slice).collect();
        let interleaved = interleave_nch(&refs);
        let deint = deinterleave_nch(&interleaved, 6);
        assert_eq!(deint, channels);
    }

    #[test]
    fn test_stereo_to_mono() {
        let interleaved = [1.0_f32, 3.0, 2.0, 4.0]; // L=1,R=3 then L=2,R=4
        let mono = stereo_to_mono(&interleaved);
        assert_eq!(mono, [2.0, 3.0]);
    }

    #[test]
    fn test_mono_to_stereo() {
        let mono = [1.0_f32, 2.0];
        let stereo = mono_to_stereo(&mono);
        assert_eq!(stereo, [1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_empty_2ch() {
        let out = interleave_2ch(&[], &[]);
        assert!(out.is_empty());
        let (a, b) = deinterleave_2ch(&[]);
        assert!(a.is_empty());
        assert!(b.is_empty());
    }
}
