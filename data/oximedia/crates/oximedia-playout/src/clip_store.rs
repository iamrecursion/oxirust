//! Clip store for playout
//!
//! Manages a catalogue of `PlayoutClip` objects that can be searched by ID or
//! name and queried for aggregate duration.  Includes checksum-based integrity
//! verification on ingest (SHA-256 computed in pure Rust without any C deps).

#![allow(dead_code)]

use crate::{PlayoutError, Result};

// ---------------------------------------------------------------------------
// Pure-Rust SHA-256 (no external crate, no C/Fortran)
// ---------------------------------------------------------------------------

/// Compute a SHA-256 digest of the given bytes.
///
/// Implements FIPS 180-4 using 32-bit arithmetic only.  This is deliberately
/// straightforward; performance is sufficient for pre-ingest verification
/// (file hashing happens once, not per-frame).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    // Initial hash values (fractional parts of sqrt of first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    // Round constants (fractional parts of cube roots of first 64 primes).
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    // Pre-processing: padding
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded: Vec<u8> = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block.
    for block in padded.chunks(64) {
        // Prepare message schedule.
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        // Compression.
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    // Produce digest.
    let mut digest = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
    }
    digest
}

/// Encode a SHA-256 digest as a lowercase hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    digest.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    })
}

// ---------------------------------------------------------------------------
// Integrity check result
// ---------------------------------------------------------------------------

/// Result of an integrity verification check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityStatus {
    /// Checksum matches; clip is intact.
    Valid,
    /// Checksum mismatch; clip may be corrupted.
    Corrupted { expected: String, actual: String },
    /// No checksum was recorded at ingest; verification skipped.
    Unverified,
}

impl IntegrityStatus {
    pub fn is_valid(&self) -> bool {
        *self == Self::Valid
    }
}

/// A single clip available for playout
#[derive(Debug, Clone)]
pub struct PlayoutClip {
    /// Unique identifier
    pub id: u64,
    /// Human-readable name
    pub name: String,
    /// Absolute path to the media file
    pub path: String,
    /// Total length of the media in frames
    pub duration_frames: u64,
    /// In-point frame (first frame to use)
    pub in_point: u64,
    /// Out-point frame (last frame to use, inclusive)
    pub out_point: u64,
    /// Whether the clip has an audio track
    pub has_audio: bool,
    /// Whether the clip has a video track
    pub has_video: bool,
    /// SHA-256 checksum of the media file at ingest time (lowercase hex, or empty).
    pub checksum_sha256: String,
}

impl PlayoutClip {
    /// Create a new clip spanning the full duration (in=0, out=frames-1)
    ///
    /// `frames` must be at least 1.  If `frames` is 0, `out_point` is also
    /// set to 0 and `is_valid()` will return `false`.
    pub fn new(id: u64, name: &str, path: &str, frames: u64) -> Self {
        let out_point = if frames > 0 { frames - 1 } else { 0 };
        Self {
            id,
            name: name.to_string(),
            path: path.to_string(),
            duration_frames: frames,
            in_point: 0,
            out_point,
            has_audio: true,
            has_video: true,
            checksum_sha256: String::new(),
        }
    }

    /// Create a clip with a pre-computed SHA-256 checksum.
    pub fn new_with_checksum(id: u64, name: &str, path: &str, frames: u64, checksum: &str) -> Self {
        let mut clip = Self::new(id, name, path, frames);
        clip.checksum_sha256 = checksum.to_string();
        clip
    }

    /// Compute the SHA-256 checksum of the given byte slice and store it in
    /// this clip.  Call this at ingest time with the raw file bytes.
    pub fn ingest_with_data(&mut self, file_data: &[u8]) {
        self.checksum_sha256 = sha256_hex(file_data);
    }

    /// Verify the integrity of a byte slice against the stored checksum.
    ///
    /// - If no checksum was recorded (`checksum_sha256` is empty), returns
    ///   `IntegrityStatus::Unverified`.
    /// - If the computed checksum matches, returns `IntegrityStatus::Valid`.
    /// - Otherwise returns `IntegrityStatus::Corrupted` with both digests.
    pub fn verify_integrity(&self, file_data: &[u8]) -> IntegrityStatus {
        if self.checksum_sha256.is_empty() {
            return IntegrityStatus::Unverified;
        }
        let actual = sha256_hex(file_data);
        if actual == self.checksum_sha256 {
            IntegrityStatus::Valid
        } else {
            IntegrityStatus::Corrupted {
                expected: self.checksum_sha256.clone(),
                actual,
            }
        }
    }

    /// Effective duration: `out_point - in_point + 1` frames
    pub fn duration_frames(&self) -> u64 {
        if self.out_point >= self.in_point {
            self.out_point - self.in_point + 1
        } else {
            0
        }
    }

    /// A clip is valid when it has a non-empty path, non-zero duration, a valid
    /// in/out range, and carries at least one of audio or video.
    pub fn is_valid(&self) -> bool {
        !self.path.is_empty()
            && self.duration_frames > 0
            && self.out_point >= self.in_point
            && (self.has_audio || self.has_video)
    }
}

/// Catalogue of clips available for playout
#[derive(Debug, Default)]
pub struct ClipStore {
    clips: Vec<PlayoutClip>,
}

impl ClipStore {
    /// Create a new empty clip store
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a clip to the store
    pub fn add(&mut self, clip: PlayoutClip) {
        self.clips.push(clip);
    }

    /// Remove a clip by id; returns `true` if a clip was removed
    pub fn remove(&mut self, id: u64) -> bool {
        if let Some(pos) = self.clips.iter().position(|c| c.id == id) {
            self.clips.remove(pos);
            true
        } else {
            false
        }
    }

    /// Find a clip by id
    pub fn find(&self, id: u64) -> Option<&PlayoutClip> {
        self.clips.iter().find(|c| c.id == id)
    }

    /// Find the first clip whose name matches exactly
    pub fn find_by_name(&self, name: &str) -> Option<&PlayoutClip> {
        self.clips.iter().find(|c| c.name == name)
    }

    /// Sum of `duration_frames()` across all clips in the store
    pub fn total_duration_frames(&self) -> u64 {
        self.clips.iter().map(PlayoutClip::duration_frames).sum()
    }

    /// Number of clips in the store
    pub fn count(&self) -> usize {
        self.clips.len()
    }

    /// Iterate over all clips
    pub fn iter(&self) -> impl Iterator<Item = &PlayoutClip> {
        self.clips.iter()
    }

    /// Ingest a clip and compute its SHA-256 checksum from the provided file data.
    ///
    /// The checksum is stored in the clip for future integrity verification.
    pub fn ingest(&mut self, mut clip: PlayoutClip, file_data: &[u8]) {
        clip.ingest_with_data(file_data);
        self.clips.push(clip);
    }

    /// Verify a clip's stored checksum against a byte slice.
    ///
    /// Returns `Err` if the clip is not found or the integrity check fails.
    pub fn verify(&self, id: u64, file_data: &[u8]) -> Result<IntegrityStatus> {
        let clip = self
            .find(id)
            .ok_or_else(|| PlayoutError::NotFound(format!("clip {id} not found in store")))?;
        Ok(clip.verify_integrity(file_data))
    }

    /// Verify all clips with stored checksums against their expected checksums.
    ///
    /// `data_provider` is called with each clip's `path` to retrieve the current
    /// file bytes.  Clips without a checksum are skipped (reported as Unverified).
    pub fn verify_all(
        &self,
        data_provider: &dyn Fn(&str) -> Option<Vec<u8>>,
    ) -> Vec<(u64, IntegrityStatus)> {
        self.clips
            .iter()
            .map(|clip| {
                if clip.checksum_sha256.is_empty() {
                    (clip.id, IntegrityStatus::Unverified)
                } else if let Some(data) = data_provider(&clip.path) {
                    (clip.id, clip.verify_integrity(&data))
                } else {
                    (
                        clip.id,
                        IntegrityStatus::Corrupted {
                            expected: clip.checksum_sha256.clone(),
                            actual: "file not accessible".to_string(),
                        },
                    )
                }
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, frames: u64) -> PlayoutClip {
        PlayoutClip::new(id, &format!("clip_{id}"), "/media/clip.mxf", frames)
    }

    #[test]
    fn test_clip_new_full_range() {
        let c = make_clip(1, 100);
        assert_eq!(c.in_point, 0);
        assert_eq!(c.out_point, 99);
    }

    #[test]
    fn test_clip_duration_frames() {
        let c = make_clip(1, 100);
        assert_eq!(c.duration_frames(), 100);
    }

    #[test]
    fn test_clip_duration_custom_range() {
        let mut c = make_clip(1, 100);
        c.in_point = 10;
        c.out_point = 49;
        assert_eq!(c.duration_frames(), 40);
    }

    #[test]
    fn test_clip_is_valid() {
        let c = make_clip(1, 50);
        assert!(c.is_valid());
    }

    #[test]
    fn test_clip_zero_frames_invalid() {
        let c = make_clip(1, 0);
        assert!(!c.is_valid());
    }

    #[test]
    fn test_clip_no_tracks_invalid() {
        let mut c = make_clip(1, 50);
        c.has_audio = false;
        c.has_video = false;
        assert!(!c.is_valid());
    }

    #[test]
    fn test_store_add_and_count() {
        let mut store = ClipStore::new();
        store.add(make_clip(1, 100));
        store.add(make_clip(2, 200));
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_store_remove_existing() {
        let mut store = ClipStore::new();
        store.add(make_clip(1, 100));
        assert!(store.remove(1));
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_store_remove_nonexistent_returns_false() {
        let mut store = ClipStore::new();
        assert!(!store.remove(99));
    }

    #[test]
    fn test_store_find_by_id() {
        let mut store = ClipStore::new();
        store.add(make_clip(7, 300));
        assert!(store.find(7).is_some());
        assert!(store.find(8).is_none());
    }

    #[test]
    fn test_store_find_by_name() {
        let mut store = ClipStore::new();
        store.add(make_clip(3, 100));
        assert!(store.find_by_name("clip_3").is_some());
        assert!(store.find_by_name("missing").is_none());
    }

    #[test]
    fn test_store_total_duration_frames() {
        let mut store = ClipStore::new();
        store.add(make_clip(1, 100));
        store.add(make_clip(2, 200));
        store.add(make_clip(3, 50));
        assert_eq!(store.total_duration_frames(), 350);
    }

    // ── Integrity verification tests ─────────────────────────────────────────

    #[test]
    fn test_sha256_known_vector() {
        // FIPS 180-4 test vector: SHA-256("abc")
        // = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let digest = sha256_hex(b"abc");
        // Exact 64-char hex string:
        assert_eq!(digest.len(), 64);
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(digest.starts_with("ba78"));
    }

    #[test]
    fn test_sha256_empty_input() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256_hex(b"");
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_clip_ingest_stores_checksum() {
        let mut clip = make_clip(1, 100);
        let data = b"fake media bytes";
        clip.ingest_with_data(data);
        assert!(!clip.checksum_sha256.is_empty());
        assert_eq!(clip.checksum_sha256.len(), 64);
    }

    #[test]
    fn test_clip_verify_valid() {
        let mut clip = make_clip(1, 100);
        let data = b"media content here";
        clip.ingest_with_data(data);
        assert_eq!(clip.verify_integrity(data), IntegrityStatus::Valid);
    }

    #[test]
    fn test_clip_verify_corrupted() {
        let mut clip = make_clip(1, 100);
        clip.ingest_with_data(b"original data");
        let status = clip.verify_integrity(b"corrupted data");
        assert!(matches!(status, IntegrityStatus::Corrupted { .. }));
    }

    #[test]
    fn test_clip_verify_unverified() {
        let clip = make_clip(1, 100); // no checksum recorded
        let status = clip.verify_integrity(b"some bytes");
        assert_eq!(status, IntegrityStatus::Unverified);
    }

    #[test]
    fn test_store_ingest_method() {
        let mut store = ClipStore::new();
        let clip = make_clip(42, 100);
        let data = b"media file bytes";
        store.ingest(clip, data);
        let found = store.find(42).expect("clip should be in store");
        assert!(!found.checksum_sha256.is_empty());
    }

    #[test]
    fn test_store_verify() {
        let mut store = ClipStore::new();
        let clip = make_clip(10, 50);
        let data = b"test media";
        store.ingest(clip, data);
        let status = store.verify(10, data).expect("verify should succeed");
        assert_eq!(status, IntegrityStatus::Valid);
    }

    #[test]
    fn test_store_verify_not_found() {
        let store = ClipStore::new();
        assert!(store.verify(999, b"data").is_err());
    }

    #[test]
    fn test_store_verify_all() {
        let mut store = ClipStore::new();
        // Give each clip a unique path so the data_provider can identify them.
        let clip1 = PlayoutClip::new(1, "clip_1", "/media/clip_1.mxf", 10);
        let clip2 = PlayoutClip::new(2, "clip_2", "/media/clip_2.mxf", 10);
        store.ingest(clip1, b"data1");
        store.ingest(clip2, b"data2");

        let results = store.verify_all(&|path| {
            if path.contains("clip_1") {
                Some(b"data1".to_vec())
            } else if path.contains("clip_2") {
                Some(b"data2".to_vec())
            } else {
                None
            }
        });
        assert_eq!(results.len(), 2);
        for (_, status) in &results {
            assert!(
                status.is_valid(),
                "all clips should verify as valid: {status:?}"
            );
        }
    }
}
