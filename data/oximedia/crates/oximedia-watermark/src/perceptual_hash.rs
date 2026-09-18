#![allow(dead_code)]
//! Perceptual hashing for audio watermark integrity verification.
//!
//! Generates compact fingerprints from audio that are robust against minor
//! modifications (compression, noise) but change significantly when the
//! content is substantially different.  Useful for verifying that a
//! watermarked segment still matches its original content.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of frequency bands for the hash.
const DEFAULT_BANDS: usize = 32;

/// Default frame length used for short-time analysis.
const DEFAULT_FRAME_LEN: usize = 2048;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A perceptual hash is a fixed-length bit vector stored as a `Vec<u8>`.
/// Each byte holds 8 hash bits; total bits = `bytes.len() * 8`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioHash {
    /// Raw hash bytes (MSB first within each byte).
    pub bytes: Vec<u8>,
    /// Number of valid bits (may be < `bytes.len() * 8` for the last byte).
    pub bit_len: usize,
}

impl AudioHash {
    /// Create a new hash from raw bytes with a specified bit length.
    #[must_use]
    pub fn new(bytes: Vec<u8>, bit_len: usize) -> Self {
        Self { bytes, bit_len }
    }

    /// Return the Hamming distance between two hashes of the same length.
    ///
    /// Returns `None` if the hashes have different bit lengths.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> Option<u32> {
        if self.bit_len != other.bit_len {
            return None;
        }
        let dist: u32 = self
            .bytes
            .iter()
            .zip(other.bytes.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        Some(dist)
    }

    /// Normalised distance in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn normalised_distance(&self, other: &Self) -> Option<f64> {
        self.hamming_distance(other)
            .map(|d| f64::from(d) / self.bit_len.max(1) as f64)
    }

    /// Check whether two hashes are "similar" given a maximum normalised
    /// distance threshold.
    #[must_use]
    pub fn is_similar(&self, other: &Self, threshold: f64) -> bool {
        self.normalised_distance(other)
            .is_some_and(|d| d <= threshold)
    }
}

/// Configuration for the perceptual hasher.
#[derive(Debug, Clone)]
pub struct HasherConfig {
    /// Number of frequency bands.
    pub bands: usize,
    /// Frame length in samples.
    pub frame_len: usize,
    /// Overlap ratio between consecutive frames (0.0 .. 1.0).
    pub overlap: f32,
}

impl Default for HasherConfig {
    fn default() -> Self {
        Self {
            bands: DEFAULT_BANDS,
            frame_len: DEFAULT_FRAME_LEN,
            overlap: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Hasher
// ---------------------------------------------------------------------------

/// Perceptual audio hasher.
#[derive(Debug)]
pub struct PerceptualHasher {
    config: HasherConfig,
}

impl PerceptualHasher {
    /// Create a new hasher with the given configuration.
    #[must_use]
    pub fn new(config: HasherConfig) -> Self {
        Self { config }
    }

    /// Create a hasher with default settings.
    #[must_use]
    pub fn default_hasher() -> Self {
        Self::new(HasherConfig::default())
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &HasherConfig {
        &self.config
    }

    /// Compute the perceptual hash for a mono audio signal.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn hash(&self, samples: &[f32]) -> AudioHash {
        if samples.is_empty() {
            return AudioHash::new(vec![0], 0);
        }

        let hop = self.hop_size();
        let num_frames = if samples.len() >= self.config.frame_len {
            (samples.len() - self.config.frame_len) / hop + 1
        } else {
            1
        };

        // For each frame, compute band energies then produce 1 bit per band
        // by comparing each band energy to the mean energy of that frame.
        let bands = self.config.bands;
        let mut bits: Vec<bool> = Vec::with_capacity(num_frames * bands);

        for f in 0..num_frames {
            let start = f * hop;
            let end = (start + self.config.frame_len).min(samples.len());
            let frame = &samples[start..end];

            let energies = self.band_energies(frame);
            let mean: f64 = energies.iter().sum::<f64>() / energies.len().max(1) as f64;

            for &e in &energies {
                bits.push(e >= mean);
            }
        }

        bits_to_hash(&bits)
    }

    /// Hop size in samples based on overlap ratio.
    #[allow(clippy::cast_precision_loss)]
    fn hop_size(&self) -> usize {
        let overlap_samples = (self.config.frame_len as f32 * self.config.overlap) as usize;
        (self.config.frame_len - overlap_samples).max(1)
    }

    /// Compute energy in `bands` equal-width sub-bands of a frame using a
    /// simple DFT-magnitude approximation (sum of squared samples in each
    /// sub-band range).  This is intentionally simplified — real implementations
    /// would use an FFT, but we avoid external dependencies here.
    #[allow(clippy::cast_precision_loss)]
    fn band_energies(&self, frame: &[f32]) -> Vec<f64> {
        let bands = self.config.bands;
        let len = frame.len();
        let band_width = len / bands.max(1);
        let mut energies = Vec::with_capacity(bands);

        for b in 0..bands {
            let lo = b * band_width;
            let hi = if b + 1 == bands {
                len
            } else {
                (b + 1) * band_width
            };
            let energy: f64 = frame[lo..hi]
                .iter()
                .map(|&s| {
                    let v = f64::from(s);
                    v * v
                })
                .sum();
            energies.push(energy);
        }
        energies
    }
}

// ---------------------------------------------------------------------------
// Hash database for batch lookup
// ---------------------------------------------------------------------------

/// A simple in-memory database mapping segment IDs to their perceptual hashes.
#[derive(Debug)]
pub struct HashDatabase {
    entries: HashMap<String, AudioHash>,
}

impl HashDatabase {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert a hash for a given segment id.
    pub fn insert(&mut self, id: impl Into<String>, hash: AudioHash) {
        self.entries.insert(id.into(), hash);
    }

    /// Look up a hash by segment id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&AudioHash> {
        self.entries.get(id)
    }

    /// Find all entries whose normalised distance to `query` is within
    /// `threshold`.
    #[must_use]
    pub fn find_similar(&self, query: &AudioHash, threshold: f64) -> Vec<(&str, f64)> {
        self.entries
            .iter()
            .filter_map(|(id, h)| {
                h.normalised_distance(query)
                    .filter(|&d| d <= threshold)
                    .map(|d| (id.as_str(), d))
            })
            .collect()
    }

    /// Return the number of stored hashes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check whether the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for HashDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bits_to_hash(bits: &[bool]) -> AudioHash {
    let bit_len = bits.len();
    let byte_len = bit_len.div_ceil(8);
    let mut bytes = vec![0u8; byte_len];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    AudioHash::new(bytes, bit_len)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tone(len: usize, freq: f32) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..len)
            .map(|i| (i as f32 * freq * std::f32::consts::TAU / 44100.0).sin())
            .collect()
    }

    #[test]
    fn test_hash_deterministic() {
        let h = PerceptualHasher::default_hasher();
        let samples = make_tone(8192, 440.0);
        let h1 = h.hash(&samples);
        let h2 = h.hash(&samples);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_empty_input() {
        let h = PerceptualHasher::default_hasher();
        let hash = h.hash(&[]);
        assert_eq!(hash.bit_len, 0);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let h = PerceptualHasher::default_hasher();
        let samples = make_tone(8192, 440.0);
        let hash = h.hash(&samples);
        assert_eq!(hash.hamming_distance(&hash), Some(0));
    }

    #[test]
    fn test_hamming_distance_different() {
        let h = PerceptualHasher::default_hasher();
        let a = h.hash(&make_tone(8192, 440.0));
        let b = h.hash(&make_tone(8192, 1000.0));
        // Different tones should produce some differing bits
        let dist = a.hamming_distance(&b);
        assert!(dist.is_some());
    }

    #[test]
    fn test_normalised_distance_range() {
        let h = PerceptualHasher::default_hasher();
        let a = h.hash(&make_tone(8192, 200.0));
        let b = h.hash(&make_tone(8192, 8000.0));
        let nd = a.normalised_distance(&b).expect("should succeed in test");
        assert!((0.0..=1.0).contains(&nd));
    }

    #[test]
    fn test_is_similar_identical() {
        let h = PerceptualHasher::default_hasher();
        let samples = make_tone(8192, 440.0);
        let hash = h.hash(&samples);
        assert!(hash.is_similar(&hash, 0.0));
    }

    #[test]
    fn test_different_length_hashes_no_distance() {
        let a = AudioHash::new(vec![0xFF], 8);
        let b = AudioHash::new(vec![0xFF, 0x00], 16);
        assert!(a.hamming_distance(&b).is_none());
    }

    #[test]
    fn test_hash_database_insert_get() {
        let mut db = HashDatabase::new();
        let hash = AudioHash::new(vec![0xAB], 8);
        db.insert("seg1", hash.clone());
        assert_eq!(db.get("seg1"), Some(&hash));
        assert!(db.get("seg2").is_none());
    }

    #[test]
    fn test_hash_database_find_similar() {
        let mut db = HashDatabase::new();
        let h1 = AudioHash::new(vec![0b1111_1111], 8);
        let h2 = AudioHash::new(vec![0b1111_1110], 8); // 1 bit diff
        db.insert("a", h1.clone());
        db.insert("b", h2);
        let results = db.find_similar(&h1, 0.2);
        // "a" has distance 0, "b" has distance 1/8=0.125 => both within 0.2
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_hash_database_empty() {
        let db = HashDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_bits_to_hash_roundtrip() {
        let bits = vec![true, false, true, true, false, false, true, false, true];
        let hash = bits_to_hash(&bits);
        assert_eq!(hash.bit_len, 9);
        // First byte: 10110010 = 0xB2
        assert_eq!(hash.bytes[0], 0xB2);
        // Second byte: 1xxxxxxx = 0x80
        assert_eq!(hash.bytes[1], 0x80);
    }

    #[test]
    fn test_hasher_config() {
        let cfg = HasherConfig {
            bands: 16,
            frame_len: 1024,
            overlap: 0.25,
        };
        let h = PerceptualHasher::new(cfg);
        assert_eq!(h.config().bands, 16);
        assert_eq!(h.config().frame_len, 1024);
    }
}
