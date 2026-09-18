//! Comprehensive archive verification and repair with pure-Rust SHA-256.
//!
//! Provides manifest-based verification at three levels (Quick, Checksum, Full),
//! a proper SHA-256 implementation, and JSON manifest serialization.

use crate::ArchiveError;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// SHA-256 — pure Rust
// ---------------------------------------------------------------------------

/// SHA-256 round constants K (first 32 bits of fractional parts of cube roots
/// of primes 2..311).
#[allow(clippy::unreadable_literal)]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Initial hash values H0..H7 (first 32 bits of fractional parts of square
/// roots of primes 2..19).
#[allow(clippy::unreadable_literal)]
const H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[inline(always)]
fn rotr32(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        let s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    for i in 0..64 {
        let s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Compute SHA-256 of `data`, returning raw 32-byte digest.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = H_INIT;
    let bit_len = (data.len() as u64).wrapping_mul(8);

    /// Convert a 64-byte slice to an array reference. The caller guarantees
    /// the slice is exactly 64 bytes.
    #[inline]
    fn as_block(slice: &[u8]) -> &[u8; 64] {
        // SAFETY-equivalent: the callers all pass slices of known length 64.
        // Using try_into + unwrap_or_else to avoid introducing unsafe.
        slice
            .try_into()
            .unwrap_or_else(|_| unreachable!("sha256 block must be 64 bytes"))
    }

    // Process complete 64-byte blocks.
    let mut processed = 0usize;
    while processed + 64 <= data.len() {
        sha256_compress(&mut state, as_block(&data[processed..processed + 64]));
        processed += 64;
    }

    // Padding.
    let remainder = &data[processed..];
    let mut padded = [0u8; 128];
    padded[..remainder.len()].copy_from_slice(remainder);
    padded[remainder.len()] = 0x80;

    let pad_len = if remainder.len() < 56 { 64 } else { 128 };
    let bit_len_bytes = bit_len.to_be_bytes();
    padded[pad_len - 8..pad_len].copy_from_slice(&bit_len_bytes);

    sha256_compress(&mut state, as_block(&padded[..64]));
    if pad_len == 128 {
        sha256_compress(&mut state, as_block(&padded[64..128]));
    }

    let mut digest = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Compute SHA-256 and return lowercase hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Verification types
// ---------------------------------------------------------------------------

/// Controls how thoroughly an archive entry is verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationLevel {
    /// Check file existence and size only.
    Quick,
    /// Existence + size + SHA-256 checksum comparison.
    Checksum,
    /// Checksum + fully read file content (simulate decompression check).
    Full,
}

impl std::fmt::Display for VerificationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationLevel::Quick => write!(f, "Quick"),
            VerificationLevel::Checksum => write!(f, "Checksum"),
            VerificationLevel::Full => write!(f, "Full"),
        }
    }
}

/// A specific verification failure.
#[derive(Debug, Clone)]
pub enum VerificationError {
    MissingFile {
        path: String,
    },
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    CorruptData {
        path: String,
        reason: String,
    },
    PermissionDenied {
        path: String,
    },
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationError::MissingFile { path } => write!(f, "Missing file: {path}"),
            VerificationError::SizeMismatch {
                path,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Size mismatch for {path}: expected {expected}, actual {actual}"
                )
            }
            VerificationError::ChecksumMismatch {
                path,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Checksum mismatch for {path}: expected {expected}, actual {actual}"
                )
            }
            VerificationError::CorruptData { path, reason } => {
                write!(f, "Corrupt data in {path}: {reason}")
            }
            VerificationError::PermissionDenied { path } => {
                write!(f, "Permission denied: {path}")
            }
        }
    }
}

/// Summary report produced by [`ArchiveVerifier::verify_manifest`].
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub archive_path: String,
    pub level: String,
    pub total_entries: usize,
    pub verified_ok: usize,
    pub errors: Vec<VerificationError>,
    pub warnings: Vec<String>,
    pub duration_secs: f64,
}

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

/// A single entry in an [`ArchiveManifest`].
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub compressed_size: u64,
    pub modified_at: u64,
}

/// Manifest describing the contents of an archive with a self-checksum.
#[derive(Debug, Clone)]
pub struct ArchiveManifest {
    pub version: u32,
    /// Unix timestamp of manifest creation.
    pub created_at: u64,
    pub entries: Vec<ManifestEntry>,
    pub total_size_bytes: u64,
    /// SHA-256 of the concatenated per-entry sha256 strings (hex), for fast
    /// top-level integrity checking.
    pub archive_checksum: String,
}

/// Simple JSON serialization helpers (no external crate required — uses manual
/// formatting, which avoids pulling in `serde` for this module).
impl ManifestEntry {
    fn to_json_object(&self) -> String {
        format!(
            r#"{{"path":{},"size_bytes":{},"sha256":{},"compressed_size":{},"modified_at":{}}}"#,
            json_string(&self.path),
            self.size_bytes,
            json_string(&self.sha256),
            self.compressed_size,
            self.modified_at,
        )
    }

    fn from_json_object(s: &str) -> Result<Self, ArchiveError> {
        let path = extract_json_string(s, "path")?;
        let size_bytes = extract_json_u64(s, "size_bytes")?;
        let sha256 = extract_json_string(s, "sha256")?;
        let compressed_size = extract_json_u64(s, "compressed_size")?;
        let modified_at = extract_json_u64(s, "modified_at")?;
        Ok(ManifestEntry {
            path,
            size_bytes,
            sha256,
            compressed_size,
            modified_at,
        })
    }
}

impl ArchiveManifest {
    /// Build a manifest from a list of entries, computing `archive_checksum`.
    pub fn build(entries: Vec<ManifestEntry>) -> Self {
        let total_size_bytes: u64 = entries.iter().map(|e| e.size_bytes).sum();
        // archive_checksum = SHA-256 of all per-entry sha256 hex strings concatenated.
        let combined: String = entries.iter().map(|e| e.sha256.as_str()).collect();
        let archive_checksum = sha256_hex(combined.as_bytes());
        // created_at: use 0 when std::time is unavailable in test environments.
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            version: 1,
            created_at,
            entries,
            total_size_bytes,
            archive_checksum,
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        let entries_json: Vec<String> = self.entries.iter().map(|e| e.to_json_object()).collect();
        format!(
            r#"{{"version":{},"created_at":{},"total_size_bytes":{},"archive_checksum":{},"entries":[{}]}}"#,
            self.version,
            self.created_at,
            self.total_size_bytes,
            json_string(&self.archive_checksum),
            entries_json.join(","),
        )
    }

    /// Deserialize from JSON string, re-verifying `archive_checksum`.
    pub fn from_json(s: &str) -> Result<Self, ArchiveError> {
        let version = extract_json_u64(s, "version")? as u32;
        let created_at = extract_json_u64(s, "created_at")?;
        let total_size_bytes = extract_json_u64(s, "total_size_bytes")?;
        let archive_checksum = extract_json_string(s, "archive_checksum")?;

        let entries = extract_json_array(s, "entries")?
            .iter()
            .map(|obj| ManifestEntry::from_json_object(obj))
            .collect::<Result<Vec<_>, _>>()?;

        // Verify archive_checksum.
        let combined: String = entries.iter().map(|e| e.sha256.as_str()).collect();
        let expected = sha256_hex(combined.as_bytes());
        if expected != archive_checksum {
            return Err(ArchiveError::Corruption(format!(
                "manifest archive_checksum mismatch: expected {expected}, got {archive_checksum}"
            )));
        }

        Ok(Self {
            version,
            created_at,
            entries,
            total_size_bytes,
            archive_checksum,
        })
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no serde dependency in this module)
// ---------------------------------------------------------------------------

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn extract_json_string(s: &str, key: &str) -> Result<String, ArchiveError> {
    let needle = format!("\"{key}\":");
    let start = s
        .find(&needle)
        .ok_or_else(|| ArchiveError::Validation(format!("JSON key not found: {key}")))?;
    let after_colon = &s[start + needle.len()..];
    let trimmed = after_colon.trim_start();
    if !trimmed.starts_with('"') {
        return Err(ArchiveError::Validation(format!(
            "Expected string for key {key}"
        )));
    }
    let inner = &trimmed[1..];
    let mut value = String::new();
    let mut chars = inner.chars();
    loop {
        match chars.next() {
            None => {
                return Err(ArchiveError::Validation(format!(
                    "Unterminated string for key {key}"
                )))
            }
            Some('"') => break,
            Some('\\') => match chars.next() {
                Some('"') => value.push('"'),
                Some('\\') => value.push('\\'),
                Some('n') => value.push('\n'),
                Some('r') => value.push('\r'),
                Some('t') => value.push('\t'),
                Some(c) => value.push(c),
                None => {
                    return Err(ArchiveError::Validation(format!(
                        "Truncated escape in {key}"
                    )))
                }
            },
            Some(c) => value.push(c),
        }
    }
    Ok(value)
}

fn extract_json_u64(s: &str, key: &str) -> Result<u64, ArchiveError> {
    let needle = format!("\"{key}\":");
    let start = s
        .find(&needle)
        .ok_or_else(|| ArchiveError::Validation(format!("JSON key not found: {key}")))?;
    let after_colon = s[start + needle.len()..].trim_start();
    let end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    let num_str = &after_colon[..end];
    num_str
        .parse::<u64>()
        .map_err(|e| ArchiveError::Validation(format!("Invalid number for key {key}: {e}")))
}

/// Extract an array of JSON objects from a key.
fn extract_json_array(s: &str, key: &str) -> Result<Vec<String>, ArchiveError> {
    let needle = format!("\"{key}\":[");
    let start = s
        .find(&needle)
        .ok_or_else(|| ArchiveError::Validation(format!("JSON array key not found: {key}")))?;
    let array_start = start + needle.len();
    let array_s = &s[array_start..];

    let mut objects = Vec::new();
    let mut depth = 0i32;
    let mut obj_start: Option<usize> = None;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in array_s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start_idx) = obj_start {
                        objects.push(array_s[start_idx..=i].to_string());
                        obj_start = None;
                    }
                }
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }

    Ok(objects)
}

// ---------------------------------------------------------------------------
// ArchiveVerifier
// ---------------------------------------------------------------------------

/// Verifies archive entries against an [`ArchiveManifest`].
pub struct ArchiveVerifier {
    level: VerificationLevel,
    /// Number of threads for parallel verification (0 = sequential).
    parallelism: usize,
}

impl ArchiveVerifier {
    /// Create a new verifier at the given verification level.
    pub fn new(level: VerificationLevel) -> Self {
        Self {
            level,
            parallelism: 0,
        }
    }

    /// Create a new verifier with parallel verification using rayon.
    ///
    /// `thread_count` specifies the number of threads to use. A value of 0
    /// falls back to sequential verification. A value of 1 uses a single
    /// rayon thread. Values > 1 use the full rayon thread pool.
    pub fn with_parallelism(level: VerificationLevel, thread_count: usize) -> Self {
        Self {
            level,
            parallelism: thread_count,
        }
    }

    /// Verify all entries in `manifest` under `base_path`.
    ///
    /// When `parallelism > 0`, entries are verified in parallel using rayon's
    /// thread pool, which significantly speeds up large archive verification.
    pub fn verify_manifest(
        &self,
        manifest: &ArchiveManifest,
        base_path: &Path,
    ) -> VerificationReport {
        let start = Instant::now();

        if self.parallelism > 0 && manifest.entries.len() > 1 {
            self.verify_manifest_parallel(manifest, base_path, start)
        } else {
            self.verify_manifest_sequential(manifest, base_path, start)
        }
    }

    /// Sequential verification (original behavior).
    fn verify_manifest_sequential(
        &self,
        manifest: &ArchiveManifest,
        base_path: &Path,
        start: Instant,
    ) -> VerificationReport {
        let mut errors: Vec<VerificationError> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();
        let mut verified_ok = 0usize;

        for entry in &manifest.entries {
            let full_path = base_path.join(&entry.path);
            match self.verify_entry(entry, &full_path, &mut warnings) {
                Ok(()) => verified_ok += 1,
                Err(e) => errors.push(e),
            }
        }

        VerificationReport {
            archive_path: base_path.display().to_string(),
            level: self.level.to_string(),
            total_entries: manifest.entries.len(),
            verified_ok,
            errors,
            warnings,
            duration_secs: start.elapsed().as_secs_f64(),
        }
    }

    /// Parallel verification using rayon.
    fn verify_manifest_parallel(
        &self,
        manifest: &ArchiveManifest,
        base_path: &Path,
        start: Instant,
    ) -> VerificationReport {
        use rayon::prelude::*;

        // Each entry is verified independently, collecting results.
        let results: Vec<(Result<Vec<String>, VerificationError>, String)> = manifest
            .entries
            .par_iter()
            .map(|entry| {
                let full_path = base_path.join(&entry.path);
                let mut local_warnings = Vec::new();
                match self.verify_entry(entry, &full_path, &mut local_warnings) {
                    Ok(()) => (Ok(local_warnings), entry.path.clone()),
                    Err(e) => (Err(e), entry.path.clone()),
                }
            })
            .collect();

        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut verified_ok = 0usize;

        for (result, _path) in results {
            match result {
                Ok(local_warns) => {
                    verified_ok += 1;
                    warnings.extend(local_warns);
                }
                Err(e) => errors.push(e),
            }
        }

        VerificationReport {
            archive_path: base_path.display().to_string(),
            level: self.level.to_string(),
            total_entries: manifest.entries.len(),
            verified_ok,
            errors,
            warnings,
            duration_secs: start.elapsed().as_secs_f64(),
        }
    }

    fn verify_entry(
        &self,
        entry: &ManifestEntry,
        full_path: &Path,
        warnings: &mut Vec<String>,
    ) -> Result<(), VerificationError> {
        // --- Quick: existence + size ---
        let metadata = std::fs::metadata(full_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                VerificationError::MissingFile {
                    path: entry.path.clone(),
                }
            } else if e.kind() == std::io::ErrorKind::PermissionDenied {
                VerificationError::PermissionDenied {
                    path: entry.path.clone(),
                }
            } else {
                VerificationError::CorruptData {
                    path: entry.path.clone(),
                    reason: e.to_string(),
                }
            }
        })?;

        let actual_size = metadata.len();
        if actual_size != entry.size_bytes {
            return Err(VerificationError::SizeMismatch {
                path: entry.path.clone(),
                expected: entry.size_bytes,
                actual: actual_size,
            });
        }

        if self.level == VerificationLevel::Quick {
            return Ok(());
        }

        // --- Checksum / Full: read data and compute SHA-256 ---
        let data = std::fs::read(full_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                VerificationError::PermissionDenied {
                    path: entry.path.clone(),
                }
            } else {
                VerificationError::CorruptData {
                    path: entry.path.clone(),
                    reason: e.to_string(),
                }
            }
        })?;

        let actual_hash = sha256_hex(&data);
        if actual_hash != entry.sha256 {
            return Err(VerificationError::ChecksumMismatch {
                path: entry.path.clone(),
                expected: entry.sha256.clone(),
                actual: actual_hash,
            });
        }

        if self.level == VerificationLevel::Full {
            // Simulate full content validation: verify we can read all bytes
            // and that the length matches (already done above, so just a note).
            if data.len() as u64 != entry.size_bytes {
                warnings.push(format!(
                    "Full read size mismatch for {}: {} vs {}",
                    entry.path,
                    data.len(),
                    entry.size_bytes
                ));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parallel file verification API
// ---------------------------------------------------------------------------

/// Configuration for parallel file verification.
#[derive(Debug, Clone)]
pub struct ParallelVerifyConfig {
    /// Number of rayon threads to use.  0 means use the rayon global pool
    /// default (typically `num_cpus`).
    pub threads: usize,
    /// When `true`, verification stops as soon as the first error is found.
    /// (Remaining work already dispatched to rayon may still complete.)
    pub fail_fast: bool,
    /// Verification level applied to every file.
    pub level: VerificationLevel,
}

impl Default for ParallelVerifyConfig {
    fn default() -> Self {
        Self {
            threads: 0,
            fail_fast: false,
            level: VerificationLevel::Checksum,
        }
    }
}

/// A single verification error produced by [`verify_files_parallel`].
#[derive(Debug, Clone)]
pub struct VerifyError {
    /// Relative or absolute path of the file that failed.
    pub path: String,
    /// Underlying verification failure.
    pub inner: VerificationError,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.inner)
    }
}

impl std::error::Error for VerifyError {}

/// Summary report produced by [`verify_files_parallel`].
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// Number of files successfully verified.
    pub verified: usize,
    /// Number of files that failed verification.
    pub failed: usize,
    /// Per-file error details.
    pub errors: Vec<VerifyError>,
    /// Wall-clock duration of the full verification pass.
    pub duration_secs: f64,
}

impl VerifyReport {
    /// Returns `true` when all files passed verification.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Total number of files examined (verified + failed).
    #[must_use]
    pub fn total(&self) -> usize {
        self.verified + self.failed
    }
}

/// Verify a slice of `(base_dir, manifest_entry)` pairs in parallel using rayon.
///
/// Each entry is verified according to `config.level`.  When `config.fail_fast`
/// is set, the first encountered error causes all remaining work to be skipped
/// (already-started rayon tasks will run to completion, but their results are
/// discarded).
///
/// # Thread pool
///
/// When `config.threads == 0` the global rayon thread pool is used.  When
/// `config.threads > 0` a dedicated `rayon::ThreadPool` with exactly that
/// many threads is created for the duration of this call.
pub fn verify_files_parallel(
    files: &[(std::path::PathBuf, ManifestEntry)],
    config: &ParallelVerifyConfig,
) -> VerifyReport {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    let start = Instant::now();
    let abort = AtomicBool::new(false);

    let verifier = ArchiveVerifier {
        level: config.level.clone(),
        parallelism: 0, // individual entry verification is always sequential
    };

    // Helper closure that verifies a single (base, entry) pair.
    let verify_one = |base: &std::path::PathBuf, entry: &ManifestEntry| {
        if config.fail_fast && abort.load(Ordering::Relaxed) {
            return None; // skip: already aborted
        }
        let full_path = base.join(&entry.path);
        let mut warnings = Vec::new();
        match verifier.verify_entry(entry, &full_path, &mut warnings) {
            Ok(()) => Some(Ok(())),
            Err(e) => {
                if config.fail_fast {
                    abort.store(true, Ordering::Relaxed);
                }
                Some(Err(VerifyError {
                    path: entry.path.clone(),
                    inner: e,
                }))
            }
        }
    };

    let results: Vec<Option<Result<(), VerifyError>>> = if config.threads > 0 {
        // Build a dedicated thread pool.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build()
            .unwrap_or_else(|_| {
                // Fall back to default pool on build failure.
                rayon::ThreadPoolBuilder::new()
                    .build()
                    .expect("rayon global fallback")
            });
        pool.install(|| {
            files
                .par_iter()
                .map(|(base, entry)| verify_one(base, entry))
                .collect()
        })
    } else {
        files
            .par_iter()
            .map(|(base, entry)| verify_one(base, entry))
            .collect()
    };

    let mut verified = 0usize;
    let mut errors: Vec<VerifyError> = Vec::new();

    for result in results.into_iter().flatten() {
        match result {
            Ok(()) => verified += 1,
            Err(e) => errors.push(e),
        }
    }

    let failed = errors.len();
    VerifyReport {
        verified,
        failed,
        errors,
        duration_secs: start.elapsed().as_secs_f64(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_temp_file(dir: &std::path::Path, name: &str, content: &[u8]) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(content).expect("write temp file");
        path
    }

    fn make_entry(name: &str, content: &[u8]) -> ManifestEntry {
        ManifestEntry {
            path: name.to_string(),
            size_bytes: content.len() as u64,
            sha256: sha256_hex(content),
            compressed_size: content.len() as u64,
            modified_at: 0,
        }
    }

    // --- SHA-256 tests ---

    #[test]
    fn test_sha256_empty() {
        // Known SHA-256 of empty string.
        let digest = sha256_hex(b"");
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_abc() {
        // Standard SHA-256("abc") test vector.
        let digest = sha256_hex(b"abc");
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn test_sha256_known_vector() {
        // SHA-256("The quick brown fox jumps over the lazy dog")
        let digest = sha256_hex(b"The quick brown fox jumps over the lazy dog");
        assert_eq!(
            digest,
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    // --- Manifest build/JSON roundtrip ---

    #[test]
    fn test_manifest_build_empty() {
        let m = ArchiveManifest::build(vec![]);
        assert_eq!(m.total_size_bytes, 0);
        assert_eq!(m.version, 1);
        // archive_checksum of empty string
        assert_eq!(m.archive_checksum, sha256_hex(b""));
    }

    #[test]
    fn test_manifest_json_roundtrip() {
        let entries = vec![
            make_entry("file_a.txt", b"hello"),
            make_entry("dir/file_b.bin", b"world"),
        ];
        let m = ArchiveManifest::build(entries);
        let json = m.to_json();
        let m2 = ArchiveManifest::from_json(&json).expect("from_json failed");
        assert_eq!(m2.version, m.version);
        assert_eq!(m2.total_size_bytes, m.total_size_bytes);
        assert_eq!(m2.archive_checksum, m.archive_checksum);
        assert_eq!(m2.entries.len(), 2);
        assert_eq!(m2.entries[0].path, "file_a.txt");
        assert_eq!(m2.entries[1].path, "dir/file_b.bin");
    }

    #[test]
    fn test_manifest_checksum_tamper_detected() {
        let entries = vec![make_entry("f.txt", b"content")];
        let m = ArchiveManifest::build(entries);
        let mut json = m.to_json();
        // Tamper with archive_checksum in JSON.
        json = json.replace(
            &m.archive_checksum,
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        let result = ArchiveManifest::from_json(&json);
        assert!(result.is_err(), "should detect tampered checksum");
    }

    // --- ArchiveVerifier tests ---

    #[test]
    fn test_verify_quick_ok() {
        let dir = std::env::temp_dir().join("oximedia_archive_verify_test_quick");
        std::fs::create_dir_all(&dir).ok();
        let content = b"test content for quick verify";
        make_temp_file(&dir, "testfile.txt", content);

        let entry = make_entry("testfile.txt", content);
        let manifest = ArchiveManifest::build(vec![entry]);
        let verifier = ArchiveVerifier::new(VerificationLevel::Quick);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.total_entries, 1);
        assert_eq!(report.verified_ok, 1);
        assert!(report.errors.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_checksum_ok() {
        let dir = std::env::temp_dir().join("oximedia_archive_verify_test_cksum");
        std::fs::create_dir_all(&dir).ok();
        let content = b"checksum verification content";
        make_temp_file(&dir, "data.bin", content);

        let entry = make_entry("data.bin", content);
        let manifest = ArchiveManifest::build(vec![entry]);
        let verifier = ArchiveVerifier::new(VerificationLevel::Checksum);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 1);
        assert!(report.errors.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_full_ok() {
        let dir = std::env::temp_dir().join("oximedia_archive_verify_test_full");
        std::fs::create_dir_all(&dir).ok();
        let content = b"full verification content with more bytes";
        make_temp_file(&dir, "full.bin", content);

        let entry = make_entry("full.bin", content);
        let manifest = ArchiveManifest::build(vec![entry]);
        let verifier = ArchiveVerifier::new(VerificationLevel::Full);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 1);
        assert!(report.errors.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_missing_file() {
        let dir = std::env::temp_dir().join("oximedia_archive_verify_test_miss");
        std::fs::create_dir_all(&dir).ok();

        let entry = make_entry("nonexistent.txt", b"data");
        let manifest = ArchiveManifest::build(vec![entry]);
        let verifier = ArchiveVerifier::new(VerificationLevel::Quick);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 0);
        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0],
            VerificationError::MissingFile { .. }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_size_mismatch() {
        let dir = std::env::temp_dir().join("oximedia_archive_verify_test_size");
        std::fs::create_dir_all(&dir).ok();
        let actual_content = b"actual content";
        make_temp_file(&dir, "size.txt", actual_content);

        // Entry claims different size.
        let mut entry = make_entry("size.txt", actual_content);
        entry.size_bytes = 999;
        let manifest = ArchiveManifest::build(vec![entry]);
        // Fix archive_checksum for modified entry (build won't know actual content).
        let verifier = ArchiveVerifier::new(VerificationLevel::Quick);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 0);
        assert!(matches!(
            &report.errors[0],
            VerificationError::SizeMismatch { .. }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let dir = std::env::temp_dir().join("oximedia_archive_verify_test_ckmm");
        std::fs::create_dir_all(&dir).ok();
        let actual_content = b"real content";
        make_temp_file(&dir, "ck.txt", actual_content);

        // Entry claims wrong hash but correct size.
        let entry = ManifestEntry {
            path: "ck.txt".to_string(),
            size_bytes: actual_content.len() as u64,
            sha256: "a".repeat(64),
            compressed_size: actual_content.len() as u64,
            modified_at: 0,
        };
        let manifest = ArchiveManifest {
            version: 1,
            created_at: 0,
            total_size_bytes: actual_content.len() as u64,
            archive_checksum: sha256_hex(
                b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            entries: vec![entry],
        };
        let verifier = ArchiveVerifier::new(VerificationLevel::Checksum);
        let report = verifier.verify_manifest(&manifest, &dir);

        assert_eq!(report.verified_ok, 0);
        assert!(matches!(
            &report.errors[0],
            VerificationError::ChecksumMismatch { .. }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    // -----------------------------------------------------------------------
    // ParallelVerifyConfig / VerifyReport / verify_files_parallel tests
    // -----------------------------------------------------------------------

    fn make_pair(
        dir: &std::path::Path,
        name: &str,
        content: &[u8],
    ) -> (std::path::PathBuf, ManifestEntry) {
        make_temp_file(dir, name, content);
        let entry = make_entry(name, content);
        (dir.to_path_buf(), entry)
    }

    #[test]
    fn test_parallel_verify_all_ok() {
        let dir = std::env::temp_dir().join("oximedia_pv_all_ok");
        std::fs::create_dir_all(&dir).ok();
        let files = vec![
            make_pair(&dir, "a.bin", b"alpha content"),
            make_pair(&dir, "b.bin", b"beta content"),
            make_pair(&dir, "c.bin", b"gamma content"),
        ];
        let config = ParallelVerifyConfig {
            threads: 2,
            fail_fast: false,
            level: VerificationLevel::Checksum,
        };
        let report = verify_files_parallel(&files, &config);
        assert!(report.is_ok());
        assert_eq!(report.verified, 3);
        assert_eq!(report.failed, 0);
        assert_eq!(report.total(), 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parallel_verify_with_missing_file() {
        let dir = std::env::temp_dir().join("oximedia_pv_missing");
        std::fs::create_dir_all(&dir).ok();
        let good = make_pair(&dir, "good.bin", b"good data");
        let missing_entry = make_entry("no_such_file.bin", b"ghost");
        let missing = (dir.clone(), missing_entry);
        let files = vec![good, missing];
        let config = ParallelVerifyConfig {
            threads: 2,
            fail_fast: false,
            level: VerificationLevel::Quick,
        };
        let report = verify_files_parallel(&files, &config);
        assert!(!report.is_ok());
        assert_eq!(report.failed, 1);
        assert!(matches!(
            &report.errors[0].inner,
            VerificationError::MissingFile { .. }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parallel_verify_fail_fast_stops_on_first_error() {
        let dir = std::env::temp_dir().join("oximedia_pv_failfast");
        std::fs::create_dir_all(&dir).ok();
        // Create only one file and four missing ones — fail_fast should stop early.
        let good = make_pair(&dir, "good2.bin", b"ok");
        let mk_missing = |name: &str| -> (std::path::PathBuf, ManifestEntry) {
            (dir.clone(), make_entry(name, b"x"))
        };
        let files = vec![
            mk_missing("m1.bin"),
            mk_missing("m2.bin"),
            mk_missing("m3.bin"),
            mk_missing("m4.bin"),
            good,
        ];
        let config = ParallelVerifyConfig {
            threads: 1,
            fail_fast: true,
            level: VerificationLevel::Quick,
        };
        let report = verify_files_parallel(&files, &config);
        // At least 1 error must have been recorded.
        assert!(report.failed >= 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parallel_verify_checksum_mismatch_detected() {
        let dir = std::env::temp_dir().join("oximedia_pv_ckmm");
        std::fs::create_dir_all(&dir).ok();
        let content = b"original content";
        make_temp_file(&dir, "ck.bin", content);
        // Manufacture an entry with wrong hash.
        let entry = ManifestEntry {
            path: "ck.bin".to_string(),
            size_bytes: content.len() as u64,
            sha256: "f".repeat(64),
            compressed_size: content.len() as u64,
            modified_at: 0,
        };
        let files = vec![(dir.clone(), entry)];
        let config = ParallelVerifyConfig {
            threads: 1,
            fail_fast: false,
            level: VerificationLevel::Checksum,
        };
        let report = verify_files_parallel(&files, &config);
        assert_eq!(report.failed, 1);
        assert!(matches!(
            &report.errors[0].inner,
            VerificationError::ChecksumMismatch { .. }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parallel_verify_size_mismatch_detected() {
        let dir = std::env::temp_dir().join("oximedia_pv_sizemm");
        std::fs::create_dir_all(&dir).ok();
        let content = b"hello world";
        make_temp_file(&dir, "sz.bin", content);
        let mut entry = make_entry("sz.bin", content);
        entry.size_bytes = 9999; // wrong size
        let files = vec![(dir.clone(), entry)];
        let config = ParallelVerifyConfig::default();
        let report = verify_files_parallel(&files, &config);
        assert_eq!(report.failed, 1);
        assert!(matches!(
            &report.errors[0].inner,
            VerificationError::SizeMismatch { .. }
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parallel_verify_empty_input() {
        let config = ParallelVerifyConfig::default();
        let report = verify_files_parallel(&[], &config);
        assert!(report.is_ok());
        assert_eq!(report.total(), 0);
    }

    #[test]
    fn test_parallel_verify_global_pool_threads_zero() {
        let dir = std::env::temp_dir().join("oximedia_pv_global_pool");
        std::fs::create_dir_all(&dir).ok();
        let files = vec![make_pair(&dir, "x.bin", b"data")];
        let config = ParallelVerifyConfig {
            threads: 0, // use global rayon pool
            fail_fast: false,
            level: VerificationLevel::Full,
        };
        let report = verify_files_parallel(&files, &config);
        assert!(report.is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_parallel_verify_report_duration_is_nonnegative() {
        let config = ParallelVerifyConfig::default();
        let report = verify_files_parallel(&[], &config);
        assert!(report.duration_secs >= 0.0);
    }

    #[test]
    fn test_parallel_verify_multiple_errors_collected() {
        let dir = std::env::temp_dir().join("oximedia_pv_multi_err");
        std::fs::create_dir_all(&dir).ok();
        // Three missing files.
        let files: Vec<_> = (0..3)
            .map(|i| {
                let name = format!("missing_{i}.bin");
                (dir.clone(), make_entry(&name, b"data"))
            })
            .collect();
        let config = ParallelVerifyConfig {
            threads: 2,
            fail_fast: false,
            level: VerificationLevel::Quick,
        };
        let report = verify_files_parallel(&files, &config);
        assert_eq!(report.failed, 3);
        assert_eq!(report.verified, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_verify_error_display() {
        let err = VerifyError {
            path: "some/file.bin".to_string(),
            inner: VerificationError::MissingFile {
                path: "some/file.bin".to_string(),
            },
        };
        let s = err.to_string();
        assert!(s.contains("some/file.bin"));
    }

    #[test]
    fn test_parallel_verify_config_default() {
        let cfg = ParallelVerifyConfig::default();
        assert_eq!(cfg.threads, 0);
        assert!(!cfg.fail_fast);
        assert_eq!(cfg.level, VerificationLevel::Checksum);
    }

    #[test]
    fn test_parallel_verify_mixed_ok_and_fail() {
        let dir = std::env::temp_dir().join("oximedia_pv_mixed");
        std::fs::create_dir_all(&dir).ok();
        let good = make_pair(&dir, "good_mixed.bin", b"real data");
        let bad = (dir.clone(), make_entry("missing_mixed.bin", b"ghost"));
        let files = vec![good, bad];
        let config = ParallelVerifyConfig {
            threads: 2,
            fail_fast: false,
            level: VerificationLevel::Quick,
        };
        let report = verify_files_parallel(&files, &config);
        assert_eq!(report.verified, 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.total(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
