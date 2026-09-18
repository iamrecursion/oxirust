//! Hash algorithm registry for media forensics.
//!
//! Tracks multiple hash algorithms, stores computed media hashes, and
//! detects potential collisions or discrepancies between hash values.
//!
//! In addition to exact-match collision detection, the registry supports
//! perceptual-hash near-duplicate search via Hamming distance over 64-bit
//! perceptual hashes ([`hamming_distance`] and
//! [`HashRegistry::nearest_perceptual`]).

use std::collections::HashMap;

// ── HashAlgorithm ─────────────────────────────────────────────────────────────

/// Supported hash algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    /// MD5 (legacy; 128-bit)
    Md5,
    /// SHA-256
    Sha256,
    /// SHA-512
    Sha512,
    /// BLAKE3
    Blake3,
    /// Perceptual hash (64-bit)
    Perceptual,
}

impl HashAlgorithm {
    /// Returns the output size in bits.
    #[must_use]
    pub fn output_bits(&self) -> usize {
        match self {
            Self::Md5 => 128,
            Self::Sha256 => 256,
            Self::Sha512 => 512,
            Self::Blake3 => 256,
            Self::Perceptual => 64,
        }
    }

    /// Returns the name of the algorithm as a string.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Md5 => "MD5",
            Self::Sha256 => "SHA-256",
            Self::Sha512 => "SHA-512",
            Self::Blake3 => "BLAKE3",
            Self::Perceptual => "Perceptual",
        }
    }

    /// Returns `true` if this algorithm is considered cryptographically strong.
    #[must_use]
    pub fn is_cryptographic(&self) -> bool {
        matches!(self, Self::Sha256 | Self::Sha512 | Self::Blake3)
    }
}

// ── MediaHash ─────────────────────────────────────────────────────────────────

/// A hash value computed for a piece of media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaHash {
    /// Algorithm used to produce this hash.
    pub algorithm: HashAlgorithm,
    /// Hex-encoded hash string.
    pub hex_value: String,
}

impl MediaHash {
    /// Create a new `MediaHash`.
    #[must_use]
    pub fn new(algorithm: HashAlgorithm, hex_value: impl Into<String>) -> Self {
        Self {
            algorithm,
            hex_value: hex_value.into(),
        }
    }

    /// Returns `true` if this hash matches another (same algorithm and value).
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.algorithm == other.algorithm && self.hex_value == other.hex_value
    }

    /// Returns the expected hex string length for this hash's algorithm.
    #[must_use]
    pub fn expected_hex_len(&self) -> usize {
        self.algorithm.output_bits() / 4
    }

    /// Returns `true` if the hex value length matches expectations.
    #[must_use]
    pub fn is_valid_length(&self) -> bool {
        self.hex_value.len() == self.expected_hex_len()
    }
}

// ── Perceptual near-duplicate search ──────────────────────────────────────────

/// Compute the Hamming distance between two 64-bit perceptual hashes.
///
/// Returns `None` when the hashes are not comparable as perceptual hashes:
/// - the two algorithms differ, or
/// - either algorithm is not [`HashAlgorithm::Perceptual`], or
/// - either hex value fails to parse as a 64-bit unsigned integer.
///
/// Otherwise returns `Some(distance)` where `distance` is the number of
/// differing bits between the two 64-bit values (i.e. `(a ^ b).count_ones()`),
/// a value in `0..=64`.
///
/// A small Hamming distance indicates that two assets are perceptually similar
/// (likely near-duplicates), even if their bytes differ.
#[must_use]
pub fn hamming_distance(a: &MediaHash, b: &MediaHash) -> Option<u32> {
    if a.algorithm != b.algorithm
        || a.algorithm != HashAlgorithm::Perceptual
        || b.algorithm != HashAlgorithm::Perceptual
    {
        return None;
    }

    let x = u64::from_str_radix(&a.hex_value, 16).ok()?;
    let y = u64::from_str_radix(&b.hex_value, 16).ok()?;

    Some((x ^ y).count_ones())
}

// ── HashRegistry ─────────────────────────────────────────────────────────────

/// A perceptual hash stored alongside the asset that produced it.
///
/// Maintained in [`HashRegistry::perceptual`] so that near-duplicate search can
/// scan only perceptual hashes (per asset) without affecting the exact-match
/// storage exposed by [`HashRegistry::lookup`].
#[derive(Debug, Clone)]
struct PerceptualEntry {
    /// Hex-encoded 64-bit perceptual hash.
    hex_value: String,
    /// Asset ID that produced the hash.
    asset_id: String,
}

/// Registry that stores hashes for media assets and detects collisions.
#[derive(Debug, Default)]
pub struct HashRegistry {
    /// Map: hex_value -> list of asset IDs that produced that hash.
    registry: HashMap<String, Vec<String>>,
    /// Per-asset index of perceptual hashes only.
    ///
    /// Tracked in parallel with `registry` so perceptual near-duplicate search
    /// can identify perceptual hashes per asset.  Keeping this separate from
    /// `registry` (rather than a hex->algorithm map) means a hex value reused
    /// across algorithms — e.g. the same string inserted once as SHA-256 and
    /// once as Perceptual — is disambiguated correctly: only the perceptual
    /// insert is recorded here.
    perceptual: Vec<PerceptualEntry>,
    /// Total insertions.
    total_inserts: usize,
}

impl HashRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a hash for the given `asset_id`.
    pub fn insert(&mut self, asset_id: impl Into<String>, hash: &MediaHash) {
        let asset_id = asset_id.into();
        self.registry
            .entry(hash.hex_value.clone())
            .or_default()
            .push(asset_id.clone());
        if hash.algorithm == HashAlgorithm::Perceptual {
            self.perceptual.push(PerceptualEntry {
                hex_value: hash.hex_value.clone(),
                asset_id,
            });
        }
        self.total_inserts += 1;
    }

    /// Look up all asset IDs that produced the given hash hex value.
    #[must_use]
    pub fn lookup(&self, hex_value: &str) -> &[String] {
        self.registry
            .get(hex_value)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns the number of hash values that map to more than one asset
    /// (potential collision or genuine duplicate).
    #[must_use]
    pub fn collision_count(&self) -> usize {
        self.registry.values().filter(|ids| ids.len() > 1).count()
    }

    /// Returns the total number of hash insertions.
    #[must_use]
    pub fn total_inserts(&self) -> usize {
        self.total_inserts
    }

    /// Returns the number of unique hash values stored.
    #[must_use]
    pub fn unique_hash_count(&self) -> usize {
        self.registry.len()
    }

    /// Returns all asset IDs involved in any collision.
    #[must_use]
    pub fn colliding_assets(&self) -> Vec<&str> {
        self.registry
            .values()
            .filter(|ids| ids.len() > 1)
            .flat_map(|ids| ids.iter().map(String::as_str))
            .collect()
    }

    /// Find stored perceptual-hash assets within `max_hamming` bits of `query`.
    ///
    /// Iterates over every stored hash that was inserted under
    /// [`HashAlgorithm::Perceptual`], computes the [`hamming_distance`] to
    /// `query`, and returns those whose distance is `<= max_hamming` as
    /// `(asset_id, distance)` pairs.
    ///
    /// Results are sorted ascending by distance, with a stable tie-break by
    /// asset ID so the ordering is deterministic.  When the same perceptual hex
    /// value was inserted for multiple assets (an exact duplicate), every
    /// matching asset ID is returned with distance `0`.
    ///
    /// Returns an empty vector when `query` is not a parseable perceptual hash
    /// or when no stored perceptual hash is within range.
    #[must_use]
    pub fn nearest_perceptual(&self, query: &MediaHash, max_hamming: u32) -> Vec<(&str, u32)> {
        if query.algorithm != HashAlgorithm::Perceptual {
            return Vec::new();
        }

        let mut matches: Vec<(&str, u32)> = Vec::new();

        for entry in &self.perceptual {
            let stored = MediaHash::new(HashAlgorithm::Perceptual, entry.hex_value.as_str());
            let Some(distance) = hamming_distance(query, &stored) else {
                continue;
            };
            if distance > max_hamming {
                continue;
            }
            matches.push((entry.asset_id.as_str(), distance));
        }

        // Sort ascending by distance, breaking ties by asset ID for a
        // deterministic ordering.
        matches.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        matches
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256_hash(hex: &str) -> MediaHash {
        MediaHash::new(HashAlgorithm::Sha256, hex)
    }

    #[test]
    fn test_algorithm_output_bits_md5() {
        assert_eq!(HashAlgorithm::Md5.output_bits(), 128);
    }

    #[test]
    fn test_algorithm_output_bits_sha256() {
        assert_eq!(HashAlgorithm::Sha256.output_bits(), 256);
    }

    #[test]
    fn test_algorithm_output_bits_blake3() {
        assert_eq!(HashAlgorithm::Blake3.output_bits(), 256);
    }

    #[test]
    fn test_algorithm_output_bits_perceptual() {
        assert_eq!(HashAlgorithm::Perceptual.output_bits(), 64);
    }

    #[test]
    fn test_algorithm_is_cryptographic() {
        assert!(HashAlgorithm::Sha256.is_cryptographic());
        assert!(HashAlgorithm::Blake3.is_cryptographic());
        assert!(!HashAlgorithm::Md5.is_cryptographic());
        assert!(!HashAlgorithm::Perceptual.is_cryptographic());
    }

    #[test]
    fn test_media_hash_matches_same() {
        let h1 = sha256_hash("abcdef");
        let h2 = sha256_hash("abcdef");
        assert!(h1.matches(&h2));
    }

    #[test]
    fn test_media_hash_no_match_different_value() {
        let h1 = sha256_hash("aaaa");
        let h2 = sha256_hash("bbbb");
        assert!(!h1.matches(&h2));
    }

    #[test]
    fn test_media_hash_no_match_different_algorithm() {
        let h1 = MediaHash::new(HashAlgorithm::Sha256, "abcd");
        let h2 = MediaHash::new(HashAlgorithm::Blake3, "abcd");
        assert!(!h1.matches(&h2));
    }

    #[test]
    fn test_registry_insert_and_lookup() {
        let mut reg = HashRegistry::new();
        let h = sha256_hash("deadbeef");
        reg.insert("asset_1", &h);
        let found = reg.lookup("deadbeef");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "asset_1");
    }

    #[test]
    fn test_registry_collision_count_zero() {
        let mut reg = HashRegistry::new();
        reg.insert("a", &sha256_hash("hash1"));
        reg.insert("b", &sha256_hash("hash2"));
        assert_eq!(reg.collision_count(), 0);
    }

    #[test]
    fn test_registry_collision_count_nonzero() {
        let mut reg = HashRegistry::new();
        reg.insert("a", &sha256_hash("samehash"));
        reg.insert("b", &sha256_hash("samehash"));
        assert_eq!(reg.collision_count(), 1);
    }

    #[test]
    fn test_registry_total_inserts() {
        let mut reg = HashRegistry::new();
        reg.insert("a", &sha256_hash("h1"));
        reg.insert("b", &sha256_hash("h2"));
        reg.insert("c", &sha256_hash("h1")); // collision
        assert_eq!(reg.total_inserts(), 3);
    }

    #[test]
    fn test_registry_unique_hash_count() {
        let mut reg = HashRegistry::new();
        reg.insert("a", &sha256_hash("h1"));
        reg.insert("b", &sha256_hash("h1")); // same hash
        reg.insert("c", &sha256_hash("h2"));
        assert_eq!(reg.unique_hash_count(), 2);
    }

    #[test]
    fn test_registry_colliding_assets() {
        let mut reg = HashRegistry::new();
        reg.insert("asset_x", &sha256_hash("collision_hash"));
        reg.insert("asset_y", &sha256_hash("collision_hash"));
        let colliders = reg.colliding_assets();
        assert!(colliders.contains(&"asset_x"));
        assert!(colliders.contains(&"asset_y"));
    }

    #[test]
    fn test_algorithm_name() {
        assert_eq!(HashAlgorithm::Sha256.name(), "SHA-256");
        assert_eq!(HashAlgorithm::Md5.name(), "MD5");
        assert_eq!(HashAlgorithm::Blake3.name(), "BLAKE3");
    }

    // ── Perceptual Hamming distance ───────────────────────────────────────────

    fn phash(hex: &str) -> MediaHash {
        MediaHash::new(HashAlgorithm::Perceptual, hex)
    }

    #[test]
    fn test_hamming_distance_single_bit() {
        let a = phash("0000000000000000");
        let b = phash("0000000000000001");
        assert_eq!(hamming_distance(&a, &b), Some(1));
    }

    #[test]
    fn test_hamming_distance_all_low_byte() {
        // 0x00..ff differs from 0 in the low 8 bits.
        let a = phash("00000000000000ff");
        let b = phash("0000000000000000");
        assert_eq!(hamming_distance(&a, &b), Some(8));
    }

    #[test]
    fn test_hamming_distance_full_64() {
        let a = phash("ffffffffffffffff");
        let b = phash("0000000000000000");
        assert_eq!(hamming_distance(&a, &b), Some(64));
    }

    #[test]
    fn test_hamming_distance_identical_is_zero() {
        let a = phash("0123456789abcdef");
        let b = phash("0123456789abcdef");
        assert_eq!(hamming_distance(&a, &b), Some(0));
    }

    #[test]
    fn test_hamming_distance_symmetric() {
        let a = phash("00000000000000ff");
        let b = phash("0000000000000000");
        assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
    }

    #[test]
    fn test_hamming_distance_different_algorithms_none() {
        let a = MediaHash::new(HashAlgorithm::Perceptual, "0000000000000000");
        let b = MediaHash::new(HashAlgorithm::Md5, "0000000000000000");
        assert_eq!(hamming_distance(&a, &b), None);
    }

    #[test]
    fn test_hamming_distance_non_perceptual_none() {
        // Both same algorithm, but not Perceptual.
        let a = MediaHash::new(HashAlgorithm::Sha256, "0000000000000000");
        let b = MediaHash::new(HashAlgorithm::Sha256, "0000000000000001");
        assert_eq!(hamming_distance(&a, &b), None);
    }

    #[test]
    fn test_hamming_distance_bad_hex_none() {
        let a = phash("zzzzzzzzzzzzzzzz");
        let b = phash("0000000000000000");
        assert_eq!(hamming_distance(&a, &b), None);
        // Also when the second operand is unparseable.
        let c = phash("0000000000000000");
        let d = phash("not_valid_hex!!!");
        assert_eq!(hamming_distance(&c, &d), None);
    }

    // ── nearest_perceptual ────────────────────────────────────────────────────

    #[test]
    fn test_nearest_perceptual_in_threshold_sorted() {
        let mut reg = HashRegistry::new();
        // distance 1 (low bit)
        reg.insert("near_1", &phash("0000000000000001"));
        // distance 2 (two low bits)
        reg.insert("near_2", &phash("0000000000000003"));
        // distance 8 (low byte)
        reg.insert("near_8", &phash("00000000000000ff"));
        // distance 64 — out of any small threshold
        reg.insert("far_64", &phash("ffffffffffffffff"));

        let query = phash("0000000000000000");
        let results = reg.nearest_perceptual(&query, 8);

        // far_64 (distance 64) must be excluded; the rest included, ascending.
        assert_eq!(results, vec![("near_1", 1), ("near_2", 2), ("near_8", 8)]);
    }

    #[test]
    fn test_nearest_perceptual_threshold_zero_exact_only() {
        let mut reg = HashRegistry::new();
        reg.insert("exact", &phash("0123456789abcdef"));
        reg.insert("off_by_one", &phash("0123456789abcdee"));

        let query = phash("0123456789abcdef");
        let results = reg.nearest_perceptual(&query, 0);
        assert_eq!(results, vec![("exact", 0)]);
    }

    #[test]
    fn test_nearest_perceptual_ignores_non_perceptual_entries() {
        let mut reg = HashRegistry::new();
        // A SHA-256 entry whose hex happens to start the same — must be ignored.
        reg.insert(
            "sha_asset",
            &MediaHash::new(HashAlgorithm::Sha256, "0000000000000001"),
        );
        reg.insert("phash_asset", &phash("0000000000000001"));

        let query = phash("0000000000000000");
        let results = reg.nearest_perceptual(&query, 4);
        assert_eq!(results, vec![("phash_asset", 1)]);
    }

    #[test]
    fn test_nearest_perceptual_duplicate_assets_distance_zero() {
        let mut reg = HashRegistry::new();
        // Same perceptual hex for two assets — both returned at distance 0.
        reg.insert("dup_a", &phash("00000000000000ff"));
        reg.insert("dup_b", &phash("00000000000000ff"));

        let query = phash("00000000000000ff");
        let results = reg.nearest_perceptual(&query, 0);
        // Stable tie-break by asset id.
        assert_eq!(results, vec![("dup_a", 0), ("dup_b", 0)]);
    }

    #[test]
    fn test_nearest_perceptual_non_perceptual_query_empty() {
        let mut reg = HashRegistry::new();
        reg.insert("phash_asset", &phash("0000000000000001"));
        let query = MediaHash::new(HashAlgorithm::Sha256, "0000000000000000");
        assert!(reg.nearest_perceptual(&query, 64).is_empty());
    }

    #[test]
    fn test_nearest_perceptual_empty_registry() {
        let reg = HashRegistry::new();
        let query = phash("0000000000000000");
        assert!(reg.nearest_perceptual(&query, 64).is_empty());
    }

    // ── Large-N collision / lookup correctness ────────────────────────────────

    #[test]
    fn test_large_n_distinct_perceptual_no_collisions() {
        let mut reg = HashRegistry::new();
        let n: u64 = 10_000;

        // Insert 10_000 distinct 64-bit perceptual hashes. Using the index as
        // the value guarantees uniqueness, hence zero collisions.
        for i in 0..n {
            let hex = format!("{i:016x}");
            reg.insert(format!("asset_{i}"), &phash(&hex));
        }

        assert_eq!(reg.total_inserts(), n as usize);
        assert_eq!(reg.unique_hash_count(), n as usize);
        assert_eq!(reg.collision_count(), 0);
        assert!(reg.colliding_assets().is_empty());

        // A handful of exact lookups resolve to the expected single asset.
        for &i in &[0u64, 1, 42, 1234, 9_999] {
            let hex = format!("{i:016x}");
            let found = reg.lookup(&hex);
            assert_eq!(found.len(), 1, "hex {hex} should map to one asset");
            assert_eq!(found[0], format!("asset_{i}"));
        }

        // Nearest query against hash 0: asset_1 (0x..01) is at distance 1,
        // asset_0 (itself) at distance 0. With threshold 1 we get exactly those.
        let results = reg.nearest_perceptual(&phash(&format!("{:016x}", 0u64)), 1);
        assert!(results.contains(&("asset_0", 0)));
        assert!(results.contains(&("asset_1", 1)));
        // 0x02 differs from 0x00 in 1 bit too (bit 1), so asset_2 is in range.
        assert!(results.contains(&("asset_2", 1)));
        // 0x03 differs in 2 bits → must NOT be present at threshold 1.
        assert!(!results.iter().any(|&(id, _)| id == "asset_3"));
        // Ensure ascending ordering by distance.
        assert!(results.windows(2).all(|w| w[0].1 <= w[1].1));
    }
}
