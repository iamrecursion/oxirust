//! Validate extracted portions.
//!
//! This module provides functions to validate that extracted portions are playable.

/// Validate that a portion is playable.
pub fn validate_portion(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }

    // Check for valid structure
    has_valid_structure(data) && has_reasonable_content(data)
}

/// Check if data has valid structure.
fn has_valid_structure(data: &[u8]) -> bool {
    // Look for sync patterns
    has_sync_patterns(data)
}

/// Check if data has sync patterns.
fn has_sync_patterns(data: &[u8]) -> bool {
    let mut sync_count = 0;

    for i in 0..data.len().saturating_sub(3) {
        if data[i..i + 3] == [0x00, 0x00, 0x01] {
            sync_count += 1;
        }
    }

    // Should have at least some sync patterns
    sync_count > 0
}

/// Check if data has reasonable content.
fn has_reasonable_content(data: &[u8]) -> bool {
    // Check entropy
    let entropy = super::super::detect::analyze::calculate_entropy(data);

    // Reasonable entropy for media content
    entropy > 2.0 && entropy < 7.5
}

/// Calculate quality score for extracted portion.
pub fn calculate_quality_score(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut score = 0.0;

    // Entropy score (0-40 points)
    let entropy = super::super::detect::analyze::calculate_entropy(data);
    let entropy_score = if entropy > 4.0 && entropy < 7.0 {
        40.0
    } else {
        20.0
    };
    score += entropy_score;

    // Sync pattern score (0-30 points)
    let sync_score = if has_sync_patterns(data) { 30.0 } else { 0.0 };
    score += sync_score;

    // No corruption indicators (0-30 points)
    let corruption_score = if !has_corruption_indicators(data) {
        30.0
    } else {
        10.0
    };
    score += corruption_score;

    score
}

/// Check for corruption indicators.
fn has_corruption_indicators(data: &[u8]) -> bool {
    // Check for large runs of zeros
    let mut zero_count = 0;
    let mut max_zero_run = 0;

    for &byte in data {
        if byte == 0 {
            zero_count += 1;
            max_zero_run = max_zero_run.max(zero_count);
        } else {
            zero_count = 0;
        }
    }

    // Large runs of zeros indicate corruption
    max_zero_run > 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_portion_empty() {
        assert!(!validate_portion(&[]));
    }

    #[test]
    fn test_has_sync_patterns() {
        let data = vec![0x00, 0x00, 0x01, 0xBA, 0x00, 0x00];
        assert!(has_sync_patterns(&data));

        let no_sync = vec![0xFF; 10];
        assert!(!has_sync_patterns(&no_sync));
    }

    #[test]
    fn test_has_corruption_indicators() {
        let corrupt = vec![0; 2000];
        assert!(has_corruption_indicators(&corrupt));

        let valid = vec![1, 2, 3, 4, 5];
        assert!(!has_corruption_indicators(&valid));
    }
}
