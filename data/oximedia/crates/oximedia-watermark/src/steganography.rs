//! LSB (Least Significant Bit) steganography for pixel data.
//!
//! Provides tools for embedding and extracting hidden payloads in the
//! least-significant bits of RGB image pixel data, plus anomaly detection.

/// LSB steganography encoder configuration.
#[derive(Debug, Clone)]
pub struct LsbEncoder {
    /// Number of LSBs to use per channel (1–4).
    pub bits_per_channel: u8,
    /// Which channels to use: index 0 = R, 1 = G, 2 = B.
    pub channel_mask: Vec<bool>,
}

impl Default for LsbEncoder {
    fn default() -> Self {
        Self {
            bits_per_channel: 1,
            channel_mask: vec![true, true, true],
        }
    }
}

impl LsbEncoder {
    /// Create a new encoder.
    #[must_use]
    pub fn new(bits_per_channel: u8, channel_mask: Vec<bool>) -> Self {
        let bpc = bits_per_channel.clamp(1, 4);
        Self {
            bits_per_channel: bpc,
            channel_mask,
        }
    }

    /// Calculate the capacity in bytes for the given pixel count.
    ///
    /// `pixels` is the number of RGB pixel triples.
    #[must_use]
    pub fn capacity_bytes(&self, pixel_count: usize) -> usize {
        let active_channels: usize = self.channel_mask.iter().filter(|&&b| b).count();
        let bits_per_pixel = active_channels * self.bits_per_channel as usize;
        (pixel_count * bits_per_pixel) / 8
    }
}

/// Embed `payload` bytes into the LSBs of an RGB pixel buffer.
///
/// `pixels` is a flat `width * height * 3` byte slice (R, G, B, R, G, B, …).
/// Returns `true` if the payload fit within the buffer, `false` if truncated.
#[must_use]
pub fn lsb_embed(pixels: &mut [u8], payload: &[u8]) -> bool {
    if payload.is_empty() {
        return true;
    }
    let total_bits = payload.len() * 8;
    // We only use the red channel LSB for simplicity (1 bit per 3 bytes).
    // Channel stride: every 3rd byte starting at offset 0 (R), 1 (G), 2 (B).
    let available_bits = pixels.len() / 3; // one bit per pixel in R channel
    if total_bits > available_bits {
        return false;
    }

    let mut bit_idx = 0usize;
    for pixel_idx in 0..available_bits {
        if bit_idx >= total_bits {
            break;
        }
        let byte_num = bit_idx / 8;
        let bit_num = 7 - (bit_idx % 8);
        let bit_val = (payload[byte_num] >> bit_num) & 1;

        // Embed into the R channel of this pixel
        let r_idx = pixel_idx * 3;
        pixels[r_idx] = (pixels[r_idx] & 0xFE) | bit_val;
        bit_idx += 1;
    }
    true
}

/// Extract `byte_count` bytes from the LSBs of an RGB pixel buffer.
///
/// Reads from the R channel LSB of each pixel, as written by [`lsb_embed`].
#[must_use]
pub fn lsb_extract(pixels: &[u8], byte_count: usize) -> Vec<u8> {
    let mut out = vec![0u8; byte_count];
    let pixel_count = pixels.len() / 3;
    let total_bits = byte_count * 8;

    for bit_idx in 0..total_bits.min(pixel_count) {
        let byte_num = bit_idx / 8;
        let bit_num = 7 - (bit_idx % 8);
        let r_idx = bit_idx * 3;
        if r_idx < pixels.len() {
            let bit_val = pixels[r_idx] & 1;
            out[byte_num] |= bit_val << bit_num;
        }
    }
    out
}

/// A steganographic payload with magic number and length prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StegoPayload {
    /// Magic identifier (`0x53544547` = "STEG" in ASCII).
    pub magic: u32,
    /// Length of the data in bytes.
    pub length: u32,
    /// Payload data.
    pub data: Vec<u8>,
}

impl StegoPayload {
    /// Magic constant for `StegoPayload`.
    pub const MAGIC: u32 = 0x5354_4547;

    /// Create a new payload.
    #[must_use]
    pub fn new(data: Vec<u8>) -> Self {
        let length = data.len() as u32;
        Self {
            magic: Self::MAGIC,
            length,
            data,
        }
    }

    /// Serialize to bytes: `[magic: 4][length: 4][data: N]`.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.data.len());
        out.extend_from_slice(&self.magic.to_le_bytes());
        out.extend_from_slice(&self.length.to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    /// Deserialize from bytes. Returns `None` if magic does not match or data is too short.
    #[must_use]
    pub fn deserialize(data: &[u8]) -> Option<StegoPayload> {
        if data.len() < 8 {
            return None;
        }
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != Self::MAGIC {
            return None;
        }
        let length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if data.len() < 8 + length {
            return None;
        }
        Some(StegoPayload {
            magic,
            length: length as u32,
            data: data[8..8 + length].to_vec(),
        })
    }
}

/// Detects potential LSB steganography by measuring the variance of LSBs.
pub struct StegoDetector;

impl StegoDetector {
    /// Measure the LSB distribution anomaly score for the R channel.
    ///
    /// Returns a value in `[0.0, 1.0]`. Values near `0.5` indicate uniform
    /// distribution (likely steganography), values near `0.0` or `1.0` indicate
    /// natural image bias.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_lsb_anomaly(pixels: &[u8]) -> f32 {
        if pixels.len() < 3 {
            return 0.0;
        }
        let pixel_count = pixels.len() / 3;
        let mut ones = 0usize;
        for i in 0..pixel_count {
            ones += (pixels[i * 3] & 1) as usize;
        }
        let ratio = ones as f32 / pixel_count as f32;
        // Distance from 0.5 inverted: 0 = highly biased (natural), 1 = perfectly uniform (suspicious)
        1.0 - (ratio - 0.5).abs() * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsb_encoder_default_bits_per_channel() {
        let enc = LsbEncoder::default();
        assert_eq!(enc.bits_per_channel, 1);
    }

    #[test]
    fn test_lsb_encoder_default_channels() {
        let enc = LsbEncoder::default();
        assert_eq!(enc.channel_mask, vec![true, true, true]);
    }

    #[test]
    fn test_capacity_bytes_basic() {
        let enc = LsbEncoder::default(); // 1 bpc, all 3 channels
                                         // 3 active × 1 bit = 3 bits per pixel → capacity = 3 * n / 8
        let cap = enc.capacity_bytes(100);
        assert_eq!(cap, (100 * 3) / 8);
    }

    #[test]
    fn test_capacity_bytes_single_channel() {
        let enc = LsbEncoder::new(1, vec![true, false, false]);
        let cap = enc.capacity_bytes(80);
        assert_eq!(cap, 10); // 80 bits / 8 = 10 bytes
    }

    #[test]
    fn test_lsb_embed_fits() {
        let mut pixels = vec![0xFFu8; 300]; // 100 pixels RGB
        let payload = b"hi"; // 2 bytes = 16 bits; 100 pixels avail
        let ok = lsb_embed(&mut pixels, payload);
        assert!(ok);
    }

    #[test]
    fn test_lsb_embed_too_large_returns_false() {
        let mut pixels = vec![0u8; 6]; // only 2 pixels → 2 bits
        let payload = vec![0u8; 10]; // 80 bits needed
        assert!(!lsb_embed(&mut pixels, &payload));
    }

    #[test]
    fn test_lsb_roundtrip() {
        let mut pixels = vec![0u8; 3 * 256]; // 256 pixels
        let payload = b"stego"; // 5 bytes = 40 bits
        let ok = lsb_embed(&mut pixels, payload);
        assert!(ok);
        let extracted = lsb_extract(&pixels, payload.len());
        assert_eq!(extracted, payload.as_slice());
    }

    #[test]
    fn test_lsb_extract_empty_payload() {
        let pixels = vec![0xAAu8; 30];
        let out = lsb_extract(&pixels, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_stego_payload_magic() {
        assert_eq!(StegoPayload::MAGIC, 0x5354_4547);
    }

    #[test]
    fn test_stego_payload_serialize_deserialize() {
        let data = b"hidden message".to_vec();
        let p = StegoPayload::new(data.clone());
        let bytes = p.serialize();
        let restored = StegoPayload::deserialize(&bytes).expect("should succeed in test");
        assert_eq!(restored.data, data);
    }

    #[test]
    fn test_stego_payload_deserialize_bad_magic() {
        let bytes = vec![0u8; 16];
        assert!(StegoPayload::deserialize(&bytes).is_none());
    }

    #[test]
    fn test_stego_payload_deserialize_too_short() {
        assert!(StegoPayload::deserialize(&[0, 1, 2]).is_none());
    }

    #[test]
    fn test_detect_lsb_anomaly_all_zero_lsb() {
        // All pixels have LSB 0 → ratio = 0 → very biased → low anomaly score
        let pixels = vec![0u8; 300]; // 100 pixels, all R=0
        let score = StegoDetector::detect_lsb_anomaly(&pixels);
        // ratio=0 → distance from 0.5 = 0.5 → score = 0
        assert!(score < 0.1, "score={score}");
    }

    #[test]
    fn test_detect_lsb_anomaly_uniform_returns_high() {
        // Alternating 0 and 1 in R → ratio ~0.5 → high anomaly score
        let mut pixels = vec![0u8; 300]; // 100 pixels
        for i in 0..100 {
            pixels[i * 3] = if i % 2 == 0 { 0 } else { 1 };
        }
        let score = StegoDetector::detect_lsb_anomaly(&pixels);
        assert!(score > 0.9, "score={score}");
    }

    #[test]
    fn test_detect_lsb_anomaly_tiny_input() {
        // Should not panic on very small input
        let score = StegoDetector::detect_lsb_anomaly(&[0, 1]);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
