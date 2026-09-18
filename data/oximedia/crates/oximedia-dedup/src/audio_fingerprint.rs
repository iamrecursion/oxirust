//! Audio fingerprinting for deduplication.
//!
//! This module provides audio fingerprint generation and matching to detect
//! duplicate or near-duplicate audio content across media files.

/// An audio fingerprint representing the acoustic identity of a media clip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFingerprint {
    /// 64-bit compact fingerprint hash
    pub hash: u64,
    /// Duration of the analysed segment in milliseconds
    pub duration_ms: u32,
    /// Sample rate of the source audio in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u8,
}

impl AudioFingerprint {
    /// Create a new `AudioFingerprint`.
    #[must_use]
    pub fn new(hash: u64, duration_ms: u32, sample_rate: u32, channels: u8) -> Self {
        Self {
            hash,
            duration_ms,
            sample_rate,
            channels,
        }
    }

    /// Return `true` when `self` and `other` are considered matches.
    ///
    /// Matching is based on the Hamming distance between the two hashes being
    /// at most `tolerance` bits.
    #[must_use]
    pub fn matches(&self, other: &AudioFingerprint, tolerance: u64) -> bool {
        let dist = hamming_distance(self.hash, other.hash);
        u64::from(dist) <= tolerance
    }
}

/// Compute the Hamming distance between two 64-bit values.
///
/// Counts the number of bit positions where the two values differ.
#[must_use]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Compute a 12-element chroma vector from raw PCM samples.
///
/// The PCM data is split into overlapping windows and each window contributes
/// energy to one of 12 pitch-class bins (C, C#, D, … B).
///
/// # Arguments
/// * `pcm` – mono PCM samples normalised to \[-1.0, 1.0\]
/// * `sample_rate` – sample rate in Hz (e.g. 44100)
#[must_use]
pub fn chroma_vector(pcm: &[f32], sample_rate: u32) -> [f32; 12] {
    let mut chroma = [0.0f32; 12];
    if pcm.is_empty() || sample_rate == 0 {
        return chroma;
    }

    // Each "frame" is a 50 ms window
    let frame_len = (sample_rate / 20) as usize; // 50 ms
    let frame_len = frame_len.max(1);
    let a4_hz = 440.0_f32;
    let sr = sample_rate as f32;

    for frame in pcm.chunks(frame_len) {
        let n = frame.len();
        // For each frequency bin (very simplified DFT over 128 bins)
        for k in 1_usize..=128 {
            let freq = (k as f32) * sr / (n as f32);
            // Determine which chroma bin this frequency maps to
            if freq < 20.0 || freq > 20_000.0 {
                continue;
            }
            let semitones = 12.0 * (freq / a4_hz).log2();
            // Wrap to [0, 12)
            let bin = semitones.rem_euclid(12.0) as usize % 12;

            // Compute power at this frequency bin using DFT
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (i, &s) in frame.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (i as f32) / (n as f32);
                re += s * angle.cos();
                im += s * angle.sin();
            }
            chroma[bin] += re * re + im * im;
        }
    }

    // Normalise
    let total: f32 = chroma.iter().sum();
    if total > 0.0 {
        for c in &mut chroma {
            *c /= total;
        }
    }
    chroma
}

/// Fold a sequence of chroma frames into a 64-bit fingerprint.
///
/// Each frame contributes its dominant chroma class (top bit position) to the
/// resulting hash via FNV-1a accumulation of the quantised frame values.
#[must_use]
pub fn fold_chroma_to_fingerprint(chroma_frames: &[[f32; 12]]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for frame in chroma_frames {
        // Find the dominant chroma class
        let dominant = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx) as u8;

        // Quantise all 12 bins to 4-bit integers and mix into hash
        let mut packed = 0u64;
        for (i, &v) in frame.iter().enumerate() {
            let q = (v.clamp(0.0, 1.0) * 15.0) as u64;
            packed |= q << (i * 4);
        }
        hash ^= u64::from(dominant);
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= packed;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A collection of named audio fingerprints for similarity search.
pub struct AudioFingerprintMatcher {
    /// Maximum Hamming-distance threshold for considering two fingerprints similar
    pub threshold_bits: u32,
    /// Stored (id, fingerprint) pairs
    pub fingerprints: Vec<(String, AudioFingerprint)>,
}

impl AudioFingerprintMatcher {
    /// Create a new matcher with the given bit-distance threshold.
    #[must_use]
    pub fn new(threshold_bits: u32) -> Self {
        Self {
            threshold_bits,
            fingerprints: Vec::new(),
        }
    }

    /// Add a fingerprint under the given identifier.
    pub fn add(&mut self, id: impl Into<String>, fp: AudioFingerprint) {
        self.fingerprints.push((id.into(), fp));
    }

    /// Return the identifiers of all stored fingerprints that match `fp`.
    #[must_use]
    pub fn find_duplicates<'a>(&'a self, fp: &AudioFingerprint) -> Vec<&'a str> {
        self.fingerprints
            .iter()
            .filter(|(_, stored)| fp.matches(stored, u64::from(self.threshold_bits)))
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Group all stored fingerprints into clusters of similar items.
    ///
    /// Uses a simple union-find over pairwise comparisons.
    #[must_use]
    pub fn cluster_similar(&self) -> Vec<Vec<String>> {
        let n = self.fingerprints.len();
        if n == 0 {
            return Vec::new();
        }

        // Union-Find parent array
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let dist =
                    hamming_distance(self.fingerprints[i].1.hash, self.fingerprints[j].1.hash);
                if dist <= self.threshold_bits {
                    union(&mut parent, i, j);
                }
            }
        }

        // Collect clusters
        let mut clusters: std::collections::HashMap<usize, Vec<String>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters
                .entry(root)
                .or_default()
                .push(self.fingerprints[i].0.clone());
        }

        clusters.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance_identical() {
        assert_eq!(
            hamming_distance(0xDEAD_BEEF_CAFE_1234, 0xDEAD_BEEF_CAFE_1234),
            0
        );
    }

    #[test]
    fn test_hamming_distance_zero_one() {
        // 0 vs 1 differs in exactly one bit
        assert_eq!(hamming_distance(0, 1), 1);
    }

    #[test]
    fn test_hamming_distance_all_bits() {
        assert_eq!(hamming_distance(0, u64::MAX), 64);
    }

    #[test]
    fn test_audio_fingerprint_matches_identical() {
        let fp = AudioFingerprint::new(0xABCD_0000_1234_FFFF, 3000, 44100, 2);
        assert!(fp.matches(&fp.clone(), 0));
    }

    #[test]
    fn test_audio_fingerprint_matches_within_tolerance() {
        let fp1 = AudioFingerprint::new(0b1111_0000, 1000, 44100, 1);
        let fp2 = AudioFingerprint::new(0b1111_0001, 1000, 44100, 1); // 1 bit different
        assert!(fp1.matches(&fp2, 2));
    }

    #[test]
    fn test_audio_fingerprint_no_match_exceeds_tolerance() {
        let fp1 = AudioFingerprint::new(0x0000_0000_0000_0000, 1000, 44100, 1);
        let fp2 = AudioFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 1000, 44100, 1); // 64 bits differ
        assert!(!fp1.matches(&fp2, 10));
    }

    #[test]
    fn test_chroma_vector_empty_pcm() {
        let v = chroma_vector(&[], 44100);
        assert_eq!(v, [0.0f32; 12]);
    }

    #[test]
    fn test_chroma_vector_zero_sample_rate() {
        let pcm = vec![0.5f32; 1024];
        let v = chroma_vector(&pcm, 0);
        assert_eq!(v, [0.0f32; 12]);
    }

    #[test]
    fn test_chroma_vector_length() {
        let pcm: Vec<f32> = (0..4096).map(|i| (i as f32 / 4096.0).sin()).collect();
        let v = chroma_vector(&pcm, 44100);
        assert_eq!(v.len(), 12);
    }

    #[test]
    fn test_chroma_vector_normalised() {
        let pcm: Vec<f32> = (0..4096).map(|i| (i as f32 / 512.0).sin()).collect();
        let v = chroma_vector(&pcm, 44100);
        let sum: f32 = v.iter().sum();
        // Either all zeros (silent) or approximately normalised to 1.0
        assert!((sum - 0.0).abs() < 1e-5 || (sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_fold_chroma_empty() {
        let hash = fold_chroma_to_fingerprint(&[]);
        // Should return the FNV offset basis unchanged
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_fold_chroma_deterministic() {
        let frame = [
            0.1, 0.05, 0.3, 0.0, 0.0, 0.2, 0.1, 0.05, 0.1, 0.05, 0.05, 0.0,
        ];
        let h1 = fold_chroma_to_fingerprint(&[frame]);
        let h2 = fold_chroma_to_fingerprint(&[frame]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_matcher_add_and_find() {
        let mut matcher = AudioFingerprintMatcher::new(4);
        let fp1 = AudioFingerprint::new(0xAAAA_AAAA_AAAA_AAAA, 2000, 44100, 2);
        let fp2 = AudioFingerprint::new(0xAAAA_AAAA_AAAA_AAAB, 2000, 44100, 2); // 1 bit off
        matcher.add("track_a", fp1.clone());
        matcher.add("track_b", fp2.clone());
        let results = matcher.find_duplicates(&fp1);
        assert!(results.contains(&"track_a"));
        assert!(results.contains(&"track_b"));
    }

    #[test]
    fn test_matcher_no_match() {
        let mut matcher = AudioFingerprintMatcher::new(2);
        let fp1 = AudioFingerprint::new(0x0000_0000_0000_0000, 2000, 44100, 2);
        let fp2 = AudioFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 2000, 44100, 2);
        matcher.add("track_x", fp2);
        let results = matcher.find_duplicates(&fp1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cluster_single_item() {
        let mut matcher = AudioFingerprintMatcher::new(4);
        matcher.add(
            "solo",
            AudioFingerprint::new(0x1234_5678_9ABC_DEF0, 1000, 44100, 1),
        );
        let clusters = matcher.cluster_similar();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec!["solo".to_string()]);
    }

    #[test]
    fn test_cluster_two_groups() {
        let mut matcher = AudioFingerprintMatcher::new(4);
        // Group A: very similar hashes
        matcher.add(
            "a1",
            AudioFingerprint::new(0x0000_0000_0000_0000, 1000, 44100, 1),
        );
        matcher.add(
            "a2",
            AudioFingerprint::new(0x0000_0000_0000_0001, 1000, 44100, 1),
        );
        // Group B: far from group A
        matcher.add(
            "b1",
            AudioFingerprint::new(0xFFFF_FFFF_FFFF_FFFF, 1000, 44100, 1),
        );
        matcher.add(
            "b2",
            AudioFingerprint::new(0xFFFF_FFFF_FFFF_FFFE, 1000, 44100, 1),
        );
        let clusters = matcher.cluster_similar();
        assert_eq!(clusters.len(), 2);
    }
}
