//! Audio fingerprinting for content identification.
//!
//! Uses patent-free algorithms for audio fingerprinting.

use crate::error::SearchResult;
use crate::SearchResultItem;
use std::path::Path;
use uuid::Uuid;

/// Audio fingerprint index
pub struct AudioFingerprintIndex {
    index_path: std::path::PathBuf,
    fingerprints: Vec<(Uuid, Vec<u8>)>,
}

impl AudioFingerprintIndex {
    /// Create a new audio fingerprint index
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails
    pub fn new(index_path: &Path) -> SearchResult<Self> {
        if !index_path.exists() {
            std::fs::create_dir_all(index_path)?;
        }

        Ok(Self {
            index_path: index_path.to_path_buf(),
            fingerprints: Vec::new(),
        })
    }

    /// Add an audio fingerprint
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails
    pub fn add_document(&mut self, asset_id: Uuid, fingerprint: &[u8]) -> SearchResult<()> {
        self.fingerprints.push((asset_id, fingerprint.to_vec()));
        Ok(())
    }

    /// Search for similar audio
    ///
    /// # Errors
    ///
    /// Returns an error if search fails
    pub fn search_similar(
        &self,
        query_fingerprint: &[u8],
        limit: usize,
    ) -> SearchResult<Vec<SearchResultItem>> {
        let mut results: Vec<_> = self
            .fingerprints
            .iter()
            .map(|(id, fp)| {
                let similarity = self.hamming_similarity(query_fingerprint, fp);

                SearchResultItem {
                    asset_id: *id,
                    score: similarity,
                    title: None,
                    description: None,
                    file_path: String::new(),
                    mime_type: None,
                    duration_ms: None,
                    created_at: 0,
                    modified_at: None,
                    file_size: None,
                    matched_fields: vec!["audio".to_string()],
                    thumbnail_url: None,
                }
            })
            .collect();

        // Sort by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results.into_iter().take(limit).collect())
    }

    /// Calculate Hamming similarity
    fn hamming_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        if a.is_empty() || b.is_empty() || a.len() != b.len() {
            return 0.0;
        }

        let matching_bits = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_zeros())
            .sum::<u32>();

        matching_bits as f32 / (a.len() * 8) as f32
    }

    /// Commit changes
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails
    pub fn commit(&self) -> SearchResult<()> {
        Ok(())
    }

    /// Delete a fingerprint
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub fn delete(&mut self, asset_id: Uuid) -> SearchResult<()> {
        self.fingerprints.retain(|(id, _)| *id != asset_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_similarity() {
        let temp_dir = std::env::temp_dir().join("audio_fingerprint_test");
        let index = AudioFingerprintIndex::new(&temp_dir).expect("should succeed in test");

        let a = vec![0b11111111];
        let b = vec![0b11111111];
        assert!((index.hamming_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);

        let c = vec![0b11111111];
        let d = vec![0b00000000];
        assert!((index.hamming_similarity(&c, &d) - 0.0).abs() < f32::EPSILON);

        std::fs::remove_dir_all(temp_dir).ok();
    }
}
