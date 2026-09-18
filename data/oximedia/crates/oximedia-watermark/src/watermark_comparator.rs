//! Watermark comparator: compare extracted watermarks against a reference
//! database using fuzzy matching, Hamming distance, and confidence scoring.
//!
//! This module provides:
//! - [`crate::watermark_comparator::WatermarkComparator`]: main entry point for comparison
//! - [`crate::watermark_comparator::FuzzyMatchConfig`]: tolerance parameters for fuzzy comparison
//! - [`crate::watermark_comparator::ComparisonResult`]: ranked list of matches with confidence scores
//! - [`crate::watermark_comparator::WatermarkPayloadDatabase`]: a lightweight payload-keyed database that
//!   stores raw byte payloads alongside owner/algorithm metadata
//! - Bit-error-rate and normalised Hamming distance helpers

// ---------------------------------------------------------------------------
// FuzzyMatchConfig
// ---------------------------------------------------------------------------

/// Configuration for fuzzy watermark matching.
#[derive(Debug, Clone)]
pub struct FuzzyMatchConfig {
    /// Maximum allowed bit-error rate (0.0 = exact, 0.5 = random).
    /// Candidates with BER above this value are discarded.
    pub max_ber: f64,
    /// Minimum confidence score in [0.0, 1.0] for a result to be returned.
    pub min_confidence: f64,
    /// Maximum number of matches to return (ranked by confidence).
    pub top_k: usize,
    /// Weight given to BER when computing the confidence score.
    /// Confidence = `ber_weight * (1 - ber/max_ber)` + `(1 - ber_weight)`.
    pub ber_weight: f64,
}

impl Default for FuzzyMatchConfig {
    fn default() -> Self {
        Self {
            max_ber: 0.1,
            min_confidence: 0.7,
            top_k: 5,
            ber_weight: 0.8,
        }
    }
}

// ---------------------------------------------------------------------------
// PayloadRecord
// ---------------------------------------------------------------------------

/// A single entry in the [`WatermarkPayloadDatabase`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadRecord {
    /// Unique identifier.
    pub id: u64,
    /// Raw watermark payload bytes.
    pub payload: Vec<u8>,
    /// Name or identifier of the rights owner.
    pub owner: String,
    /// Watermarking algorithm used.
    pub algorithm: String,
    /// Optional free-form notes.
    pub notes: Option<String>,
}

impl PayloadRecord {
    /// Create a new payload record.
    #[must_use]
    pub fn new(
        id: u64,
        payload: impl Into<Vec<u8>>,
        owner: impl Into<String>,
        algorithm: impl Into<String>,
    ) -> Self {
        Self {
            id,
            payload: payload.into(),
            owner: owner.into(),
            algorithm: algorithm.into(),
            notes: None,
        }
    }

    /// Attach notes.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

// ---------------------------------------------------------------------------
// WatermarkPayloadDatabase
// ---------------------------------------------------------------------------

/// In-memory database mapping raw watermark payloads to ownership metadata.
///
/// Unlike [`crate::watermark_database::WatermarkDatabase`] which stores
/// content hashes, this database stores the raw payload bytes and is designed
/// for use with the [`WatermarkComparator`].
#[derive(Debug, Clone, Default)]
pub struct WatermarkPayloadDatabase {
    records: Vec<PayloadRecord>,
    next_id: u64,
}

impl WatermarkPayloadDatabase {
    /// Create a new empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new payload record and return its ID.
    pub fn register(
        &mut self,
        payload: impl Into<Vec<u8>>,
        owner: impl Into<String>,
        algorithm: impl Into<String>,
        notes: Option<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut record = PayloadRecord::new(id, payload, owner, algorithm);
        if let Some(n) = notes {
            record = record.with_notes(n);
        }
        self.records.push(record);
        id
    }

    /// Return all records as a slice.
    #[must_use]
    pub fn all(&self) -> &[PayloadRecord] {
        &self.records
    }

    /// Find records by owner.
    #[must_use]
    pub fn find_by_owner(&self, owner: &str) -> Vec<&PayloadRecord> {
        self.records.iter().filter(|r| r.owner == owner).collect()
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ComparisonMatch
// ---------------------------------------------------------------------------

/// A single match between an extracted watermark and a database record.
#[derive(Debug, Clone)]
pub struct ComparisonMatch {
    /// The matching database record.
    pub record: PayloadRecord,
    /// Bit-error rate between extracted payload and stored payload.
    /// Range [0.0, 1.0]; 0.0 = exact match, 0.5 = uncorrelated.
    pub bit_error_rate: f64,
    /// Normalised Hamming distance (same as BER for bit sequences).
    pub hamming_distance: usize,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// True if the match is within the BER threshold (strict accept).
    pub is_accepted: bool,
}

// ---------------------------------------------------------------------------
// ComparisonResult
// ---------------------------------------------------------------------------

/// Result of a fuzzy watermark comparison.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Ranked matches (highest confidence first).
    pub matches: Vec<ComparisonMatch>,
    /// Best match (first element), if any.
    pub best_match: Option<ComparisonMatch>,
    /// Whether the best match is above the confidence threshold.
    pub is_identified: bool,
    /// Extracted payload that was queried.
    pub query_payload: Vec<u8>,
}

// ---------------------------------------------------------------------------
// WatermarkComparator
// ---------------------------------------------------------------------------

/// Compares extracted watermark payloads against a [`WatermarkPayloadDatabase`]
/// using configurable fuzzy matching.
pub struct WatermarkComparator {
    config: FuzzyMatchConfig,
}

impl WatermarkComparator {
    /// Create a new comparator with the given configuration.
    #[must_use]
    pub fn new(config: FuzzyMatchConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_comparator() -> Self {
        Self::new(FuzzyMatchConfig::default())
    }

    /// Compare an extracted payload against all entries in `db`.
    ///
    /// Returns a [`ComparisonResult`] with up to `config.top_k` ranked matches.
    #[must_use]
    pub fn compare(&self, extracted: &[u8], db: &WatermarkPayloadDatabase) -> ComparisonResult {
        let mut matches: Vec<ComparisonMatch> = db
            .all()
            .iter()
            .filter_map(|record| self.compare_single(extracted, record))
            .collect();

        // Sort by confidence descending.
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches.truncate(self.config.top_k);

        let best_match = matches.first().cloned();
        let is_identified = best_match
            .as_ref()
            .is_some_and(|m| m.confidence >= self.config.min_confidence);

        ComparisonResult {
            best_match,
            is_identified,
            matches,
            query_payload: extracted.to_vec(),
        }
    }

    /// Compare two raw payloads and return the bit-error rate.
    #[must_use]
    pub fn bit_error_rate(a: &[u8], b: &[u8]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let hamming = hamming_distance_bytes(a, b);
        let max_bits = a.len().max(b.len()) * 8;
        if max_bits == 0 {
            return 0.0;
        }
        hamming as f64 / max_bits as f64
    }

    // Internal: compare against a single record.
    fn compare_single(&self, extracted: &[u8], record: &PayloadRecord) -> Option<ComparisonMatch> {
        let stored = &record.payload;
        let hamming = hamming_distance_bytes(extracted, stored);
        let max_bits = extracted.len().max(stored.len()) * 8;
        let ber = if max_bits == 0 {
            0.0
        } else {
            hamming as f64 / max_bits as f64
        };

        if ber > self.config.max_ber {
            return None;
        }

        // Confidence: linear interpolation between min_confidence and 1.0
        // based on how far BER is below max_ber.
        let normalised_ber = if self.config.max_ber > 0.0 {
            ber / self.config.max_ber
        } else if ber == 0.0 {
            0.0
        } else {
            1.0
        };
        let confidence =
            self.config.ber_weight * (1.0 - normalised_ber) + (1.0 - self.config.ber_weight);
        let confidence = confidence.clamp(0.0, 1.0);

        let is_accepted = ber <= self.config.max_ber;

        Some(ComparisonMatch {
            record: record.clone(),
            bit_error_rate: ber,
            hamming_distance: hamming,
            confidence,
            is_accepted,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the Hamming distance (in bits) between two byte slices.
///
/// If they differ in length, the shorter slice is zero-padded virtually.
#[must_use]
pub fn hamming_distance_bytes(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let mut dist: usize = 0;

    // Count differing bits in the common prefix.
    for i in 0..min_len {
        dist += (a[i] ^ b[i]).count_ones() as usize;
    }

    // Treat extra bytes in the longer slice as XOR'd with 0x00.
    let extra_a = &a[min_len..];
    let extra_b = &b[min_len..];
    for &byte in extra_a {
        dist += byte.count_ones() as usize;
    }
    for &byte in extra_b {
        dist += byte.count_ones() as usize;
    }

    dist
}

/// Compute the normalised Hamming distance (BER) between two bit sequences
/// represented as byte slices.
#[must_use]
pub fn normalised_hamming(a: &[u8], b: &[u8]) -> f64 {
    WatermarkComparator::bit_error_rate(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db_with_payloads(payloads: &[&[u8]]) -> WatermarkPayloadDatabase {
        let mut db = WatermarkPayloadDatabase::new();
        for &p in payloads {
            db.register(p.to_vec(), "owner", "spread-spectrum", None);
        }
        db
    }

    #[test]
    fn test_exact_match_identified() {
        let db = make_db_with_payloads(&[b"hello"]);
        let cmp = WatermarkComparator::default_comparator();
        let result = cmp.compare(b"hello", &db);
        assert!(result.is_identified);
        assert_eq!(
            result
                .best_match
                .expect("exact match should have best_match")
                .bit_error_rate,
            0.0
        );
    }

    #[test]
    fn test_no_match_when_ber_too_high() {
        let db = make_db_with_payloads(&[b"\xFF\xFF\xFF\xFF"]);
        let cmp = WatermarkComparator::default_comparator();
        let result = cmp.compare(b"\x00\x00\x00\x00", &db);
        assert!(!result.is_identified);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn test_fuzzy_match_within_threshold() {
        // Flip 1 bit in a 4-byte payload (BER = 1/32 ≈ 3.1%)
        let payload = b"oxim";
        let mut extracted = payload.to_vec();
        extracted[0] ^= 0x01; // flip 1 bit

        let db = make_db_with_payloads(&[payload]);
        let config = FuzzyMatchConfig {
            max_ber: 0.1,
            min_confidence: 0.5,
            ..Default::default()
        };
        let cmp = WatermarkComparator::new(config);
        let result = cmp.compare(&extracted, &db);
        assert!(result.is_identified, "Should match with 1-bit flip");
    }

    #[test]
    fn test_top_k_limits_results() {
        let db = make_db_with_payloads(&[b"aaaaa", b"bbbbb", b"ccccc", b"ddddd", b"eeeee"]);
        let config = FuzzyMatchConfig {
            max_ber: 0.5,
            min_confidence: 0.0,
            top_k: 3,
            ..Default::default()
        };
        let cmp = WatermarkComparator::new(config);
        let result = cmp.compare(b"aaaaa", &db);
        assert!(result.matches.len() <= 3);
    }

    #[test]
    fn test_hamming_distance_same() {
        assert_eq!(hamming_distance_bytes(b"abc", b"abc"), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        // 0xFF XOR 0x00 = 0xFF => 8 bits differ
        assert_eq!(hamming_distance_bytes(&[0xFF], &[0x00]), 8);
    }

    #[test]
    fn test_hamming_distance_different_lengths() {
        // b = empty; extra byte 0xFF in a counts all 8 bits
        let a = &[0xFF];
        let b = &[];
        assert_eq!(hamming_distance_bytes(a, b), 8);
    }

    #[test]
    fn test_ber_zero_for_identical() {
        assert_eq!(WatermarkComparator::bit_error_rate(b"test", b"test"), 0.0);
    }

    #[test]
    fn test_ber_one_for_all_flipped() {
        let a = vec![0xFFu8; 4];
        let b = vec![0x00u8; 4];
        assert!(
            (WatermarkComparator::bit_error_rate(&a, &b) - 1.0).abs() < 1e-9,
            "BER should be 1.0"
        );
    }

    #[test]
    fn test_empty_db_returns_no_match() {
        let db = WatermarkPayloadDatabase::new();
        let cmp = WatermarkComparator::default_comparator();
        let result = cmp.compare(b"any", &db);
        assert!(!result.is_identified);
        assert!(result.matches.is_empty());
    }

    #[test]
    fn test_matches_sorted_by_confidence_desc() {
        // Two payloads; exact match should come first.
        let db = make_db_with_payloads(&[b"exact", b"exxct"]); // 2 bits differ in second
        let config = FuzzyMatchConfig {
            max_ber: 0.3,
            min_confidence: 0.0,
            top_k: 10,
            ..Default::default()
        };
        let cmp = WatermarkComparator::new(config);
        let result = cmp.compare(b"exact", &db);
        if result.matches.len() >= 2 {
            assert!(result.matches[0].confidence >= result.matches[1].confidence);
        }
    }

    #[test]
    fn test_confidence_is_one_for_exact_match() {
        let db = make_db_with_payloads(&[b"perfect"]);
        let cmp = WatermarkComparator::default_comparator();
        let result = cmp.compare(b"perfect", &db);
        let best = result
            .best_match
            .expect("exact match should produce best_match");
        assert!(
            (best.confidence - 1.0).abs() < 1e-9,
            "Confidence should be 1.0 for exact match, got {}",
            best.confidence
        );
    }

    #[test]
    fn test_normalised_hamming_helper() {
        let ber = normalised_hamming(b"\xFF", b"\x00");
        assert!((ber - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_query_payload_preserved_in_result() {
        let db = WatermarkPayloadDatabase::new();
        let cmp = WatermarkComparator::default_comparator();
        let result = cmp.compare(b"query-payload", &db);
        assert_eq!(result.query_payload, b"query-payload");
    }

    #[test]
    fn test_database_find_by_owner() {
        let mut db = WatermarkPayloadDatabase::new();
        db.register(b"payload1".to_vec(), "alice", "ss", None);
        db.register(b"payload2".to_vec(), "bob", "ss", None);
        let alice_records = db.find_by_owner("alice");
        assert_eq!(alice_records.len(), 1);
        assert_eq!(alice_records[0].owner, "alice");
    }

    #[test]
    fn test_database_register_increments_id() {
        let mut db = WatermarkPayloadDatabase::new();
        let id1 = db.register(b"p1".to_vec(), "o", "a", None);
        let id2 = db.register(b"p2".to_vec(), "o", "a", None);
        assert!(id2 > id1);
    }

    #[test]
    fn test_payload_record_with_notes() {
        let rec = PayloadRecord::new(1, b"data", "owner", "algo").with_notes("some note");
        assert!(rec.notes.is_some());
        assert_eq!(rec.notes.expect("notes should be present"), "some note");
    }
}
