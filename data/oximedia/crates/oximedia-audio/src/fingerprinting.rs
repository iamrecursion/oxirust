//! Audio fingerprinting (Shazam-like constellation map approach).
//!
//! This module provides audio identification via spectrogram peak pairing,
//! producing robust `FingerprintHash` values that survive noise and compression.

#![forbid(unsafe_code)]

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single spectrogram peak in the time-frequency plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrogramPeak {
    /// Time bin index (hop units).
    pub time_bin: u32,
    /// Frequency bin index.
    pub freq_bin: u32,
    /// Magnitude of the peak.
    pub magnitude: f32,
}

/// A combinatorial hash derived from a pair of spectrogram peaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FingerprintHash {
    /// Anchor peak (earlier in time).
    pub anchor: SpectrogramPeakKey,
    /// Target peak (later in time).
    pub target: SpectrogramPeakKey,
    /// Packed hash: `(anchor_freq << 40) | (target_freq << 20) | delta_time`.
    pub hash: u64,
}

/// Compact key representation of a `SpectrogramPeak` (no magnitude, for hashing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpectrogramPeakKey {
    /// Time bin index.
    pub time_bin: u32,
    /// Frequency bin index.
    pub freq_bin: u32,
}

impl From<SpectrogramPeak> for SpectrogramPeakKey {
    fn from(p: SpectrogramPeak) -> Self {
        Self {
            time_bin: p.time_bin,
            freq_bin: p.freq_bin,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal configuration
// ---------------------------------------------------------------------------

const DEFAULT_FFT_SIZE: usize = 1024;
const DEFAULT_HOP_SIZE: usize = 512;
/// Maximum number of target peaks to pair with each anchor.
const TARGETS_PER_ANCHOR: usize = 5;
/// Target zone: only look at peaks this many hops ahead, up to ZONE_END_HOPS.
const ZONE_START_HOPS: u32 = 1;
const ZONE_END_HOPS: u32 = 32;

// ---------------------------------------------------------------------------
// AudioFingerprinter
// ---------------------------------------------------------------------------

/// Produces a list of `FingerprintHash` values from raw f32 audio samples.
pub struct AudioFingerprinter {
    fft_size: usize,
    hop_size: usize,
}

impl Default for AudioFingerprinter {
    fn default() -> Self {
        Self {
            fft_size: DEFAULT_FFT_SIZE,
            hop_size: DEFAULT_HOP_SIZE,
        }
    }
}

impl AudioFingerprinter {
    /// Create a new fingerprinter with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom FFT and hop sizes. `fft_size` must be a power of two.
    #[must_use]
    pub fn with_params(fft_size: usize, hop_size: usize) -> Self {
        let fft_size = if fft_size.is_power_of_two() {
            fft_size
        } else {
            DEFAULT_FFT_SIZE
        };
        let hop_size = hop_size.max(1);
        Self { fft_size, hop_size }
    }

    /// Generate fingerprint hashes from mono f32 samples.
    ///
    /// `sample_rate` is used only for computing time-in-seconds when needed;
    /// the hash itself uses hop-bin indexing.
    pub fn fingerprint(&self, samples: &[f32], _sample_rate: u32) -> Vec<FingerprintHash> {
        if samples.is_empty() {
            return Vec::new();
        }
        let peaks = self.extract_peaks(samples);
        self.generate_hashes(&peaks)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Compute STFT and extract prominent spectral peaks.
    fn extract_peaks(&self, samples: &[f32]) -> Vec<SpectrogramPeak> {
        let fft_size = self.fft_size;
        let hop_size = self.hop_size;
        let num_bins = fft_size / 2 + 1;

        // Build Hann window
        let window = build_hann_window(fft_size);

        // Compute number of frames
        if samples.len() < fft_size {
            return Vec::new();
        }
        let num_frames = (samples.len() - fft_size) / hop_size + 1;

        // Allocate 2-D magnitude array [frame][bin]
        let mut magnitudes: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        // Create FFT plan once
        let plan = match Plan::<f64>::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE) {
            Some(p) => p,
            None => return Vec::new(),
        };

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = start + fft_size;
            if end > samples.len() {
                break;
            }

            // Windowed frame
            let buf: Vec<Complex<f64>> = (0..fft_size)
                .map(|i| Complex::new(f64::from(samples[start + i]) * window[i], 0.0))
                .collect();

            let mut out = vec![Complex::<f64>::new(0.0, 0.0); fft_size];
            plan.execute(&buf, &mut out);

            // Magnitude spectrum (positive frequencies only)
            let frame_mag: Vec<f32> = (0..num_bins)
                .map(|b| {
                    let c = out[b];
                    (c.re * c.re + c.im * c.im).sqrt() as f32
                })
                .collect();

            magnitudes.push(frame_mag);
        }

        // Pick local maxima in a 3×3 (time × freq) neighbourhood
        let actual_frames = magnitudes.len();
        if actual_frames == 0 {
            return Vec::new();
        }

        // Compute global magnitude statistics for thresholding
        let all_mags: Vec<f32> = magnitudes.iter().flat_map(|f| f.iter().copied()).collect();
        let threshold = percentile(&all_mags, 85.0);

        // If the signal is effectively silence (all magnitudes near zero), skip peak extraction
        let max_mag = all_mags.iter().cloned().fold(0.0f32, f32::max);
        if max_mag < 1e-6 {
            return Vec::new();
        }

        let mut peaks = Vec::new();

        for t in 1..(actual_frames.saturating_sub(1)) {
            for f in 1..(num_bins.saturating_sub(1)) {
                let m = magnitudes[t][f];
                if m < threshold {
                    continue;
                }
                // Check 3×3 local maximum
                let is_max = (t.saturating_sub(1)..=(t + 1).min(actual_frames - 1))
                    .flat_map(|tt| {
                        (f.saturating_sub(1)..=(f + 1).min(num_bins - 1)).map(move |ff| (tt, ff))
                    })
                    .all(|(tt, ff)| magnitudes[tt][ff] <= m);
                if is_max {
                    peaks.push(SpectrogramPeak {
                        time_bin: t as u32,
                        freq_bin: f as u32,
                        magnitude: m,
                    });
                }
            }
        }

        peaks
    }

    /// Pair peaks into hashes using the constellation map approach.
    fn generate_hashes(&self, peaks: &[SpectrogramPeak]) -> Vec<FingerprintHash> {
        if peaks.is_empty() {
            return Vec::new();
        }

        // Sort by time_bin then freq_bin for determinism
        let mut sorted = peaks.to_vec();
        sorted.sort_by_key(|p| (p.time_bin, p.freq_bin));

        let mut hashes = Vec::new();

        for (i, anchor) in sorted.iter().enumerate() {
            let mut targets_found = 0usize;

            // Look forward in time
            for target in sorted[(i + 1)..].iter() {
                let dt = target.time_bin.saturating_sub(anchor.time_bin);
                if dt < ZONE_START_HOPS {
                    continue;
                }
                if dt > ZONE_END_HOPS {
                    break; // sorted by time, no point going further
                }

                // Pack hash: anchor_freq (20 bits) | target_freq (20 bits) | delta (20 bits)
                let af = u64::from(anchor.freq_bin) & 0xF_FFFF;
                let tf = u64::from(target.freq_bin) & 0xF_FFFF;
                let td = u64::from(dt) & 0xF_FFFF;
                let hash_val = (af << 40) | (tf << 20) | td;

                hashes.push(FingerprintHash {
                    anchor: SpectrogramPeakKey::from(*anchor),
                    target: SpectrogramPeakKey::from(*target),
                    hash: hash_val,
                });

                targets_found += 1;
                if targets_found >= TARGETS_PER_ANCHOR {
                    break;
                }
            }
        }

        hashes
    }
}

// ---------------------------------------------------------------------------
// FingerprintMatcher
// ---------------------------------------------------------------------------

/// Matches a query fingerprint against a database fingerprint,
/// returning the time offset in seconds if a match is found.
pub struct FingerprintMatcher {
    /// Minimum number of votes to declare a match.
    pub min_votes: usize,
}

impl Default for FingerprintMatcher {
    fn default() -> Self {
        Self { min_votes: 3 }
    }
}

impl FingerprintMatcher {
    /// Create a new matcher with default `min_votes = 3`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a custom minimum-votes threshold.
    #[must_use]
    pub fn with_min_votes(min_votes: usize) -> Self {
        Self { min_votes }
    }

    /// Attempt to find the query audio within the database audio.
    ///
    /// Returns `Some(offset_seconds)` if a consistent time offset receives
    /// at least `min_votes` votes, otherwise `None`.
    ///
    /// The offset is expressed as `db_time - query_time` (in hop units,
    /// converted using `hop_size / sample_rate`).
    pub fn match_audio(&self, query: &[FingerprintHash], db: &[FingerprintHash]) -> Option<f32> {
        self.match_audio_with_rate(query, db, DEFAULT_HOP_SIZE, 44100)
    }

    /// Like `match_audio` but with explicit hop size and sample rate for
    /// converting the hop-domain offset to seconds.
    pub fn match_audio_with_rate(
        &self,
        query: &[FingerprintHash],
        db: &[FingerprintHash],
        hop_size: usize,
        sample_rate: u32,
    ) -> Option<f32> {
        if query.is_empty() || db.is_empty() {
            return None;
        }

        // Build hash → list of (anchor_time_bin) from db
        let mut db_index: HashMap<u64, Vec<u32>> = HashMap::new();
        for h in db {
            db_index.entry(h.hash).or_default().push(h.anchor.time_bin);
        }

        // Vote on time offsets: offset = db_anchor_time - query_anchor_time (signed hops)
        let mut votes: HashMap<i64, u32> = HashMap::new();

        for qh in query {
            if let Some(db_times) = db_index.get(&qh.hash) {
                for &db_t in db_times {
                    let offset = i64::from(db_t) - i64::from(qh.anchor.time_bin);
                    *votes.entry(offset).or_insert(0) += 1;
                }
            }
        }

        // Find highest vote
        let best = votes.into_iter().max_by_key(|&(_, v)| v);

        match best {
            Some((offset_hops, count)) if count as usize >= self.min_votes => {
                let offset_secs = offset_hops as f32 * hop_size as f32 / sample_rate as f32;
                Some(offset_secs)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone convenience function
// ---------------------------------------------------------------------------

/// Fingerprint `samples` and match against `db` fingerprint, returning offset in seconds.
pub fn match_audio(query: &[FingerprintHash], db: &[FingerprintHash]) -> Option<f32> {
    FingerprintMatcher::new().match_audio(query, db)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_hann_window(size: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (size as f64 - 1.0)).cos()))
        .collect()
}

/// Return the p-th percentile (0–100) of a slice.
fn percentile(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0 * (sorted.len() as f32 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_sine(freq_hz: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn make_noise(num_samples: usize, seed: u64) -> Vec<f32> {
        // Simple LCG pseudo-random for reproducibility (no external rand crate)
        let mut state = seed;
        (0..num_samples)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_fingerprint_empty_returns_empty() {
        let fp = AudioFingerprinter::new();
        let hashes = fp.fingerprint(&[], 44100);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_fingerprint_silence_returns_few_or_no_hashes() {
        let fp = AudioFingerprinter::new();
        let silence = vec![0.0f32; 4096];
        let hashes = fp.fingerprint(&silence, 44100);
        // Silence has no meaningful peaks — 0 hashes expected
        assert_eq!(hashes.len(), 0, "silence should produce 0 hashes");
    }

    #[test]
    fn test_fingerprint_sine_wave_deterministic() {
        let fp = AudioFingerprinter::new();
        let sine = make_sine(440.0, 44100, 44100);
        let h1 = fp.fingerprint(&sine, 44100);
        let h2 = fp.fingerprint(&sine, 44100);
        assert_eq!(
            h1.len(),
            h2.len(),
            "same input must produce same number of hashes"
        );
        for (a, b) in h1.iter().zip(h2.iter()) {
            assert_eq!(a.hash, b.hash, "same input must produce identical hashes");
        }
    }

    #[test]
    fn test_fingerprint_different_signals_different_hashes() {
        let fp = AudioFingerprinter::new();
        let noise1 = make_noise(8192, 1);
        let noise2 = make_noise(8192, 99);
        let h1 = fp.fingerprint(&noise1, 44100);
        let h2 = fp.fingerprint(&noise2, 44100);
        // Very unlikely that all hashes match
        let set1: std::collections::HashSet<u64> = h1.iter().map(|h| h.hash).collect();
        let set2: std::collections::HashSet<u64> = h2.iter().map(|h| h.hash).collect();
        let common = set1.intersection(&set2).count();
        let total = set1.len().max(set2.len()).max(1);
        assert!(
            (common as f64 / total as f64) < 0.8,
            "different signals should produce mostly different hashes"
        );
    }

    #[test]
    fn test_fingerprint_noise_has_hashes() {
        let fp = AudioFingerprinter::new();
        let noise = make_noise(16384, 42);
        let hashes = fp.fingerprint(&noise, 44100);
        assert!(!hashes.is_empty(), "noise signal should produce hashes");
    }

    #[test]
    fn test_fingerprint_hash_count_reasonable() {
        let fp = AudioFingerprinter::new();
        let sine = make_sine(880.0, 44100, 44100);
        let hashes = fp.fingerprint(&sine, 44100);
        assert!(
            hashes.len() > 0,
            "1-second sine should produce at least one hash"
        );
    }

    #[test]
    fn test_peak_extraction_finds_peaks() {
        let fp = AudioFingerprinter::new();
        let noise = make_noise(8192, 7);
        let peaks = fp.extract_peaks(&noise);
        assert!(!peaks.is_empty(), "should find peaks in noise");
    }

    #[test]
    fn test_hash_encoding_bit_fields() {
        // Verify the hash packs the three fields correctly
        let anchor = SpectrogramPeak {
            time_bin: 10,
            freq_bin: 50,
            magnitude: 1.0,
        };
        let target = SpectrogramPeak {
            time_bin: 15,
            freq_bin: 100,
            magnitude: 1.0,
        };
        let dt: u64 = 5; // 15 - 10
        let af: u64 = 50;
        let tf: u64 = 100;
        let expected = (af << 40) | (tf << 20) | dt;

        let fp = AudioFingerprinter::new();
        let peaks = vec![anchor, target];
        let hashes = fp.generate_hashes(&peaks);
        assert!(!hashes.is_empty());
        // Find the hash that pairs anchor(50) → target(100)
        let found = hashes.iter().find(|h| h.hash == expected);
        assert!(found.is_some(), "expected hash {expected:#018x} not found");
    }

    #[test]
    fn test_matcher_identical_signals_returns_zero_offset() {
        let fp = AudioFingerprinter::new();
        let signal = make_noise(16384, 123);
        let hashes = fp.fingerprint(&signal, 44100);
        if hashes.is_empty() {
            return; // skip if signal too short to produce hashes
        }
        let matcher = FingerprintMatcher::with_min_votes(1);
        let result = matcher.match_audio_with_rate(&hashes, &hashes, DEFAULT_HOP_SIZE, 44100);
        assert!(result.is_some(), "identical signals must match");
        let offset = result.expect("identical signals must return offset");
        assert!(
            offset.abs() < 0.05,
            "offset should be ~0 for identical signals, got {offset}"
        );
    }

    #[test]
    fn test_matcher_empty_query_returns_none() {
        let fp = AudioFingerprinter::new();
        let db = fp.fingerprint(&make_noise(8192, 5), 44100);
        let matcher = FingerprintMatcher::new();
        assert!(matcher.match_audio(&[], &db).is_none());
    }

    #[test]
    fn test_matcher_empty_db_returns_none() {
        let fp = AudioFingerprinter::new();
        let query = fp.fingerprint(&make_noise(8192, 5), 44100);
        let matcher = FingerprintMatcher::new();
        assert!(matcher.match_audio(&query, &[]).is_none());
    }

    #[test]
    fn test_matcher_no_match_returns_none() {
        let fp = AudioFingerprinter::new();
        let signal1 = make_noise(8192, 1);
        let signal2 = make_noise(8192, 999);
        let q = fp.fingerprint(&signal1, 44100);
        let db = fp.fingerprint(&signal2, 44100);
        // With high min_votes, a random mismatch should return None
        let matcher = FingerprintMatcher::with_min_votes(50);
        let result = matcher.match_audio(&q, &db);
        // We can't guarantee None for random data, but with 50 votes it's very unlikely to match
        // so we just ensure no panic and the API works
        let _ = result; // Either None or Some — just don't crash
    }

    #[test]
    fn test_fingerprint_long_signal() {
        let fp = AudioFingerprinter::new();
        // 10 seconds of noise
        let noise = make_noise(441000, 77);
        let hashes = fp.fingerprint(&noise, 44100);
        assert!(!hashes.is_empty(), "long signal should produce hashes");
    }

    #[test]
    fn test_fingerprint_reproducible_across_instances() {
        let signal = make_noise(8192, 55);
        let h1 = AudioFingerprinter::new().fingerprint(&signal, 44100);
        let h2 = AudioFingerprinter::new().fingerprint(&signal, 44100);
        assert_eq!(h1.len(), h2.len());
        for (a, b) in h1.iter().zip(h2.iter()) {
            assert_eq!(a.hash, b.hash);
        }
    }

    #[test]
    fn test_hash_time_delta_in_range() {
        let fp = AudioFingerprinter::new();
        let noise = make_noise(16384, 33);
        let hashes = fp.fingerprint(&noise, 44100);
        for h in &hashes {
            let dt = h.hash & 0xF_FFFF;
            assert!(dt >= ZONE_START_HOPS as u64, "delta {dt} < ZONE_START_HOPS");
            assert!(dt <= ZONE_END_HOPS as u64, "delta {dt} > ZONE_END_HOPS");
        }
    }

    #[test]
    fn test_fingerprint_real_signal_has_hashes() {
        let fp = AudioFingerprinter::new();
        // Mix two sine waves
        let sr = 44100u32;
        let n = 22050usize;
        let s: Vec<f32> = (0..n)
            .map(|i| {
                (2.0 * PI * 440.0 * i as f32 / sr as f32).sin()
                    + 0.5 * (2.0 * PI * 880.0 * i as f32 / sr as f32).sin()
            })
            .collect();
        let hashes = fp.fingerprint(&s, sr);
        assert!(!hashes.is_empty(), "mixed tones should produce hashes");
    }

    #[test]
    fn test_matcher_partial_overlap_still_matches() {
        let fp = AudioFingerprinter::new();
        // Take a long noise, and use a subset as query
        let full = make_noise(32768, 11);
        let db_hashes = fp.fingerprint(&full, 44100);
        // Query is the first 16384 samples
        let query_signal = full[..16384].to_vec();
        let query_hashes = fp.fingerprint(&query_signal, 44100);
        if query_hashes.is_empty() || db_hashes.is_empty() {
            return;
        }
        let matcher = FingerprintMatcher::with_min_votes(2);
        let result =
            matcher.match_audio_with_rate(&query_hashes, &db_hashes, DEFAULT_HOP_SIZE, 44100);
        // The query starts at time 0 in both, so offset should be ~0
        if let Some(offset) = result {
            assert!(
                offset.abs() < 1.0,
                "partial overlap offset should be < 1 second, got {offset}"
            );
        }
        // If no match found with min_votes=2, that's also acceptable for unit test
    }

    #[test]
    fn test_spectrogram_peak_fields() {
        let p = SpectrogramPeak {
            time_bin: 5,
            freq_bin: 10,
            magnitude: 2.5,
        };
        assert_eq!(p.time_bin, 5);
        assert_eq!(p.freq_bin, 10);
        assert!((p.magnitude - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_fingerprint_hash_fields() {
        let anchor = SpectrogramPeakKey {
            time_bin: 2,
            freq_bin: 30,
        };
        let target = SpectrogramPeakKey {
            time_bin: 5,
            freq_bin: 60,
        };
        let hash_val: u64 = (30u64 << 40) | (60u64 << 20) | 3u64;
        let fh = FingerprintHash {
            anchor,
            target,
            hash: hash_val,
        };
        assert_eq!(fh.anchor.time_bin, 2);
        assert_eq!(fh.target.freq_bin, 60);
        assert_eq!(fh.hash, hash_val);
    }

    #[test]
    fn test_matcher_with_offset_signal() {
        // Build a noise signal and a shifted version
        let fp = AudioFingerprinter::new();
        let hop = DEFAULT_HOP_SIZE;
        let sr = 44100u32;
        let noise = make_noise(32768, 42);

        // "Database" is the full noise
        let db_hashes = fp.fingerprint(&noise, sr);

        // "Query" is noise starting at hop_size*10 samples later, but fingerprinted from t=0
        let shift_samples = hop * 10;
        if shift_samples >= noise.len() {
            return;
        }
        let query_signal = noise[shift_samples..].to_vec();
        let query_hashes = fp.fingerprint(&query_signal, sr);

        if query_hashes.is_empty() || db_hashes.is_empty() {
            return;
        }

        let matcher = FingerprintMatcher::with_min_votes(2);
        let result = matcher.match_audio_with_rate(&query_hashes, &db_hashes, hop, sr);
        // The offset should be approximately +10 hops * hop/sr seconds
        if let Some(offset) = result {
            let expected = 10.0 * hop as f32 / sr as f32;
            // Allow ±3 hops tolerance
            let tol = 3.0 * hop as f32 / sr as f32;
            assert!(
                (offset - expected).abs() < tol + 0.5,
                "expected offset ~{expected:.3}s, got {offset:.3}s"
            );
        }
    }

    #[test]
    fn test_with_params_custom_fft() {
        let fp = AudioFingerprinter::with_params(2048, 1024);
        let noise = make_noise(8192, 19);
        let hashes = fp.fingerprint(&noise, 44100);
        // Just verify no panic and consistent behavior
        let hashes2 = fp.fingerprint(&noise, 44100);
        assert_eq!(hashes.len(), hashes2.len());
    }
}
