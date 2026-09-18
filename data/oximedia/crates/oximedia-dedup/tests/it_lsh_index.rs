//! Integration tests for `BitLshIndex` recall accuracy on near-duplicate queries.

use oximedia_dedup::lsh_index::BitLshIndex;

/// Flip `k` pseudo-random bits in `hash` using a deterministic XOR mask.
fn flip_bits(hash: u64, k: u32, seed: u64) -> u64 {
    // Deterministic mask: spread bit flips across different positions.
    let mut mask = 0u64;
    let mut state = seed;
    let mut bits_flipped = 0u32;
    while bits_flipped < k {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bit_pos = (state >> 58) as u32; // 0-63
        let bit = 1u64 << bit_pos;
        if mask & bit == 0 {
            mask |= bit;
            bits_flipped += 1;
        }
    }
    hash ^ mask
}

fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[test]
fn test_lsh_recall_near_duplicates() {
    // Build an index with 100 synthetic perceptual hashes.
    let n = 100usize;
    let mut index = BitLshIndex::new(12, 6, 0xDEAD_BEEF_CAFE_1234);

    // Use deterministic pseudo-random hashes seeded from index.
    let hashes: Vec<u64> = (0..n)
        .map(|i| {
            let mut h = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            h ^= h >> 33;
            h
        })
        .collect();

    for (id, &hash) in hashes.iter().enumerate() {
        index.insert(id as u64, hash);
    }

    // For 10 near-duplicate queries (flip ≤ 3 bits), verify recall ≥ 70%.
    let query_count = 10usize;
    let mut found = 0usize;

    for q in 0..query_count {
        let original_id = (q * 10) as u64; // sample 10 evenly spaced originals
        let original_hash = hashes[original_id as usize];
        let flipped = flip_bits(original_hash, 3, original_id.wrapping_add(0xABCD));

        // Verify the flip actually produced a valid near-duplicate (hamming ≤ 3)
        assert!(
            hamming(original_hash, flipped) <= 3,
            "flip_bits produced hamming {} > 3",
            hamming(original_hash, flipped)
        );

        let candidates = index.query_candidates(flipped);
        // Check whether the original id appears among candidates
        if candidates
            .iter()
            .any(|(cand_id, _)| *cand_id == original_id)
        {
            found += 1;
        }
    }

    let recall = found as f64 / query_count as f64;
    assert!(
        recall >= 0.70,
        "LSH recall {:.1}% ({found}/{query_count}) is below the 70% threshold. \
         Increase num_tables or decrease bits_per_table if this fails consistently.",
        recall * 100.0
    );
}

#[test]
fn test_lsh_far_duplicates_not_retrieved() {
    // Far queries (hamming ≥ 20 from every indexed item) should rarely appear
    // as candidates for any specific indexed item.
    let n = 50usize;
    let mut index = BitLshIndex::new(12, 6, 0x1234_5678_9ABC_DEF0);

    let hashes: Vec<u64> = (0..n)
        .map(|i| {
            let v = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
            v ^ (v >> 30)
        })
        .collect();

    for (id, &hash) in hashes.iter().enumerate() {
        index.insert(id as u64, hash);
    }

    // Build a "far" query: invert the top 32 bits of a known hash so
    // hamming distance is exactly 32 from the original.
    let target_id = 7u64;
    let target_hash = hashes[target_id as usize];
    // Flip top 32 bits → hamming distance = 32
    let far_hash = target_hash ^ 0xFFFF_FFFF_0000_0000;
    assert_eq!(
        hamming(target_hash, far_hash),
        32,
        "test setup: expected hamming 32"
    );

    let candidates = index.query_candidates(far_hash);
    // The original (target_id) must NOT appear among candidates for the far query.
    let found_original = candidates.iter().any(|(cid, _)| *cid == target_id);
    assert!(
        !found_original,
        "LSH incorrectly returned the far-away item (id={target_id}) as a candidate \
         for a query with hamming distance 32"
    );
}

#[test]
fn test_lsh_find_near_duplicates_end_to_end() {
    // Insert 20 items where every pair (2k, 2k+1) is a near-duplicate (hamming ≤ 2).
    // `find_near_duplicates` should return at least 8 of the 10 pairs.
    let mut index = BitLshIndex::new(16, 5, 0xCAFE_BABE_0000_0001);

    let pairs = 10usize;
    for p in 0..pairs {
        let base: u64 = (p as u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(0x1111);
        let near = base ^ (1u64 << (p % 64)); // flip exactly 1 bit
        index.insert((2 * p) as u64, base);
        index.insert((2 * p + 1) as u64, near);
    }

    let duplicates = index.find_near_duplicates(2);
    // Collect which pair-slots were detected
    let detected_pairs: std::collections::HashSet<usize> = duplicates
        .iter()
        .filter_map(|&(id_a, id_b, dist)| {
            if dist <= 2 {
                let slot_a = (id_a / 2) as usize;
                let slot_b = (id_b / 2) as usize;
                if slot_a == slot_b {
                    return Some(slot_a);
                }
            }
            None
        })
        .collect();

    assert!(
        detected_pairs.len() >= 8,
        "find_near_duplicates detected only {}/{pairs} expected pairs; \
         all results: {duplicates:?}",
        detected_pairs.len()
    );
}
