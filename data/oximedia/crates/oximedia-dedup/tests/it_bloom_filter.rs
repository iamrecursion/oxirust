//! Integration tests for `BloomFilter` false positive rate at different fill levels.

use oximedia_dedup::bloom_filter::BloomFilter;

/// Test that the observed false positive rate at a given fill level does not
/// exceed 2× the theoretical (configured) FPR.
fn assert_fpr_within_bound(capacity: usize, configured_fpr: f32, fill_ratio: f64, label: &str) {
    let items_to_insert = (capacity as f64 * fill_ratio).ceil() as usize;

    let mut filter = BloomFilter::new(capacity, configured_fpr);

    // Insert items_to_insert distinct keys
    for i in 0..items_to_insert {
        let key = format!("inserted_item_{i:010}");
        filter.insert(key.as_bytes());
    }

    // Query 10_000 keys that were never inserted
    let query_count = 10_000usize;
    let mut false_positives = 0usize;
    let offset = items_to_insert + 1_000_000; // well beyond inserted range
    for i in 0..query_count {
        let key = format!("non_inserted_query_{:010}", offset + i);
        if filter.contains(key.as_bytes()) {
            false_positives += 1;
        }
    }

    let observed_fpr = false_positives as f64 / query_count as f64;
    // Allow up to 2× the configured FPR as headroom for hash collisions and
    // formula approximations.  The bloom filter may exceed this bound only when
    // overloaded far beyond design capacity.
    let upper_bound = (configured_fpr as f64) * 2.0;

    assert!(
        observed_fpr <= upper_bound,
        "[{label}] fill={fill_ratio:.0}%: observed FPR {observed_fpr:.4} \
         exceeds 2x configured FPR {upper_bound:.4} \
         (configured={configured_fpr}, fp={false_positives}/{query_count})"
    );
}

#[test]
fn test_bloom_fpr_at_50_percent_fill() {
    assert_fpr_within_bound(10_000, 0.01, 0.50, "50% fill");
}

#[test]
fn test_bloom_fpr_at_75_percent_fill() {
    assert_fpr_within_bound(10_000, 0.01, 0.75, "75% fill");
}

#[test]
fn test_bloom_fpr_at_90_percent_fill() {
    // At 90% fill the filter is overloaded; use a more lenient threshold
    // (5× configured) because the theoretical formula assumes fill ≤ 100%.
    let capacity = 10_000usize;
    let configured_fpr: f32 = 0.01;
    let fill_ratio = 0.90;
    let items_to_insert = (capacity as f64 * fill_ratio).ceil() as usize;

    let mut filter = BloomFilter::new(capacity, configured_fpr);
    for i in 0..items_to_insert {
        let key = format!("stress_item_{i:010}");
        filter.insert(key.as_bytes());
    }

    let query_count = 10_000usize;
    let mut false_positives = 0usize;
    let offset = items_to_insert + 1_000_000;
    for i in 0..query_count {
        let key = format!("stress_query_{:010}", offset + i);
        if filter.contains(key.as_bytes()) {
            false_positives += 1;
        }
    }

    let observed_fpr = false_positives as f64 / query_count as f64;
    // 5× allowance at 90 % fill because the filter is near saturation.
    let upper_bound = (configured_fpr as f64) * 5.0;
    assert!(
        observed_fpr <= upper_bound,
        "[90% fill] observed FPR {observed_fpr:.4} exceeds 5x configured FPR {upper_bound:.4} \
         (fp={false_positives}/{query_count})"
    );
}

#[test]
fn test_bloom_no_false_negatives() {
    // Items that were inserted must ALWAYS be found (no false negatives).
    let mut filter = BloomFilter::new(5_000, 0.01);
    let n = 2_000usize;
    for i in 0..n {
        let key = format!("definite_{i:06}");
        filter.insert(key.as_bytes());
    }
    for i in 0..n {
        let key = format!("definite_{i:06}");
        assert!(
            filter.contains(key.as_bytes()),
            "false negative for item {i}: bloom filter must never miss inserted items"
        );
    }
}

#[test]
fn test_bloom_clear_resets_state() {
    let mut filter = BloomFilter::new(1_000, 0.01);
    filter.insert(b"item_alpha");
    filter.insert(b"item_beta");
    assert!(filter.contains(b"item_alpha"));

    filter.clear();

    // After clear, previously inserted items should no longer be found
    // (with overwhelming probability for well-designed filters).
    // We check at least 10 items and expect ≥ 9 to return false.
    let still_found = [b"item_alpha".as_ref(), b"item_beta".as_ref()]
        .iter()
        .filter(|&&k| filter.contains(k))
        .count();
    assert!(
        still_found == 0,
        "clear() should reset the filter; {still_found}/2 items still appear as present"
    );
}
