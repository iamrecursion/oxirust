//! Sample buffer management for audio processing.
//!
//! Provides typed sample formats, interleaved/planar buffer layouts, and a pool
//! for reusing allocations to reduce GC pressure.

#![allow(dead_code)]

/// Bit depth and data type for a single audio sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 24-bit integer (stored in 32-bit).
    I24,
    /// Signed 32-bit integer.
    I32,
    /// 32-bit IEEE 754 float.
    F32,
    /// 64-bit IEEE 754 float.
    F64,
}

impl SampleFormat {
    /// Returns the bit depth for this format.
    pub fn bit_depth(self) -> u8 {
        match self {
            SampleFormat::U8 => 8,
            SampleFormat::I16 => 16,
            SampleFormat::I24 => 24,
            SampleFormat::I32 => 32,
            SampleFormat::F32 => 32,
            SampleFormat::F64 => 64,
        }
    }

    /// Returns the byte size of one sample in this format.
    pub fn byte_size(self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::I16 => 2,
            SampleFormat::I24 => 4,
            SampleFormat::I32 => 4,
            SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
        }
    }

    /// Returns `true` for floating-point formats.
    pub fn is_float(self) -> bool {
        matches!(self, SampleFormat::F32 | SampleFormat::F64)
    }

    /// Returns `true` for integer formats.
    pub fn is_integer(self) -> bool {
        !self.is_float()
    }
}

/// A buffer of audio samples supporting both interleaved and planar layouts.
///
/// Internally always stores `f32` samples for processing convenience.
#[derive(Debug, Clone)]
pub struct SampleBuffer {
    /// Flat storage: interleaved `[L, R, L, R, ...]` or planar `[L, L, ..., R, R, ...]`.
    data: Vec<f32>,
    channels: usize,
    frames: usize,
    planar: bool,
}

impl SampleBuffer {
    /// Create a new zeroed interleaved buffer.
    pub fn new_interleaved(channels: usize, frames: usize) -> Self {
        Self {
            data: vec![0.0_f32; channels * frames],
            channels,
            frames,
            planar: false,
        }
    }

    /// Create a new zeroed planar buffer.
    pub fn new_planar(channels: usize, frames: usize) -> Self {
        Self {
            data: vec![0.0_f32; channels * frames],
            channels,
            frames,
            planar: true,
        }
    }

    /// Returns the number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels
    }

    /// Returns the number of frames (samples per channel).
    pub fn frame_count(&self) -> usize {
        self.frames
    }

    /// Returns `true` if data is stored interleaved.
    pub fn is_interleaved(&self) -> bool {
        !self.planar
    }

    /// Returns an interleaved view of the data.
    ///
    /// If the buffer is already interleaved this is a zero-copy slice.
    /// Otherwise it returns a newly converted `Vec<f32>`.
    pub fn interleaved(&self) -> Vec<f32> {
        if !self.planar {
            return self.data.clone();
        }
        // Convert planar -> interleaved
        let mut out = vec![0.0_f32; self.channels * self.frames];
        for ch in 0..self.channels {
            for fr in 0..self.frames {
                out[fr * self.channels + ch] = self.data[ch * self.frames + fr];
            }
        }
        out
    }

    /// Returns a planar view of the data as `Vec<Vec<f32>>` (one `Vec` per channel).
    pub fn deinterleaved(&self) -> Vec<Vec<f32>> {
        let mut planes: Vec<Vec<f32>> = (0..self.channels)
            .map(|_| vec![0.0_f32; self.frames])
            .collect();
        if self.planar {
            for ch in 0..self.channels {
                planes[ch].copy_from_slice(&self.data[ch * self.frames..(ch + 1) * self.frames]);
            }
        } else {
            for fr in 0..self.frames {
                for ch in 0..self.channels {
                    planes[ch][fr] = self.data[fr * self.channels + ch];
                }
            }
        }
        planes
    }

    /// Access the raw sample data slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Access the raw sample data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

/// A pool of [`SampleBuffer`] instances to reduce repeated allocations.
#[derive(Debug, Default)]
pub struct SampleBufferPool {
    free: Vec<SampleBuffer>,
}

impl SampleBufferPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self { free: Vec::new() }
    }

    /// Acquire a buffer with the given dimensions and layout.
    ///
    /// Returns a recycled buffer if one of the right size exists,
    /// otherwise allocates a new one.
    pub fn acquire(&mut self, channels: usize, frames: usize, planar: bool) -> SampleBuffer {
        let expected_len = channels * frames;
        if let Some(pos) = self
            .free
            .iter()
            .position(|b| b.data.len() == expected_len && b.planar == planar)
        {
            let mut buf = self.free.swap_remove(pos);
            buf.channels = channels;
            buf.frames = frames;
            // Zero the recycled buffer.
            for s in buf.data.iter_mut() {
                *s = 0.0;
            }
            return buf;
        }
        if planar {
            SampleBuffer::new_planar(channels, frames)
        } else {
            SampleBuffer::new_interleaved(channels, frames)
        }
    }

    /// Release a buffer back to the pool for future reuse.
    pub fn release(&mut self, buf: SampleBuffer) {
        self.free.push(buf);
    }

    /// Returns the number of free buffers in the pool.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format_bit_depth() {
        assert_eq!(SampleFormat::U8.bit_depth(), 8);
        assert_eq!(SampleFormat::I16.bit_depth(), 16);
        assert_eq!(SampleFormat::I24.bit_depth(), 24);
        assert_eq!(SampleFormat::I32.bit_depth(), 32);
        assert_eq!(SampleFormat::F32.bit_depth(), 32);
        assert_eq!(SampleFormat::F64.bit_depth(), 64);
    }

    #[test]
    fn test_sample_format_byte_size() {
        assert_eq!(SampleFormat::U8.byte_size(), 1);
        assert_eq!(SampleFormat::I16.byte_size(), 2);
        assert_eq!(SampleFormat::I24.byte_size(), 4);
        assert_eq!(SampleFormat::F32.byte_size(), 4);
        assert_eq!(SampleFormat::F64.byte_size(), 8);
    }

    #[test]
    fn test_sample_format_is_float() {
        assert!(SampleFormat::F32.is_float());
        assert!(SampleFormat::F64.is_float());
        assert!(!SampleFormat::I16.is_float());
    }

    #[test]
    fn test_sample_format_is_integer() {
        assert!(SampleFormat::U8.is_integer());
        assert!(SampleFormat::I32.is_integer());
        assert!(!SampleFormat::F32.is_integer());
    }

    #[test]
    fn test_interleaved_buffer_channel_count() {
        let buf = SampleBuffer::new_interleaved(2, 64);
        assert_eq!(buf.channel_count(), 2);
    }

    #[test]
    fn test_interleaved_buffer_frame_count() {
        let buf = SampleBuffer::new_interleaved(2, 64);
        assert_eq!(buf.frame_count(), 64);
    }

    #[test]
    fn test_interleaved_roundtrip() {
        let mut buf = SampleBuffer::new_interleaved(2, 4);
        let slice = buf.as_mut_slice();
        // L R L R L R L R
        for (i, s) in slice.iter_mut().enumerate() {
            *s = i as f32;
        }
        let interleaved = buf.interleaved();
        assert_eq!(interleaved.len(), 8);
        assert!((interleaved[0] - 0.0).abs() < 1e-6);
        assert!((interleaved[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_deinterleaved_from_interleaved() {
        let mut buf = SampleBuffer::new_interleaved(2, 3);
        // L0=1, R0=2, L1=3, R1=4, L2=5, R2=6
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        buf.as_mut_slice().copy_from_slice(&data);
        let planes = buf.deinterleaved();
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0], vec![1.0, 3.0, 5.0]);
        assert_eq!(planes[1], vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_planar_buffer_interleaved_conversion() {
        let mut buf = SampleBuffer::new_planar(2, 3);
        // plane 0: [1,3,5], plane 1: [2,4,6]
        let data = [1.0_f32, 3.0, 5.0, 2.0, 4.0, 6.0];
        buf.as_mut_slice().copy_from_slice(&data);
        let interleaved = buf.interleaved();
        assert_eq!(interleaved, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pool_acquire_new() {
        let mut pool = SampleBufferPool::new();
        let buf = pool.acquire(2, 128, false);
        assert_eq!(buf.channel_count(), 2);
        assert_eq!(buf.frame_count(), 128);
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn test_pool_release_and_reacquire() {
        let mut pool = SampleBufferPool::new();
        let buf = pool.acquire(2, 128, false);
        pool.release(buf);
        assert_eq!(pool.free_count(), 1);
        let buf2 = pool.acquire(2, 128, false);
        assert_eq!(pool.free_count(), 0);
        assert_eq!(buf2.channel_count(), 2);
    }

    #[test]
    fn test_pool_zeroes_recycled_buffer() {
        let mut pool = SampleBufferPool::new();
        let mut buf = pool.acquire(1, 4, false);
        buf.as_mut_slice().iter_mut().for_each(|s| *s = 99.0);
        pool.release(buf);
        let buf2 = pool.acquire(1, 4, false);
        for s in buf2.as_slice() {
            assert!((*s).abs() < 1e-6, "Expected zero, got {s}");
        }
    }

    #[test]
    fn test_pool_different_size_no_reuse() {
        let mut pool = SampleBufferPool::new();
        let buf = pool.acquire(2, 64, false);
        pool.release(buf);
        // Request different size — should not reuse
        let _buf2 = pool.acquire(2, 128, false);
        // The 64-frame buffer should still be in the pool
        assert_eq!(pool.free_count(), 1);
    }
}
