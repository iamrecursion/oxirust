//! OCR text search implementation.

use crate::error::SearchResult;
use std::path::Path;
use uuid::Uuid;

/// OCR text index
pub struct OcrIndex {
    index_path: std::path::PathBuf,
    texts: Vec<(Uuid, String)>,
}

impl OcrIndex {
    /// Create a new OCR index
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
            texts: Vec::new(),
        })
    }

    /// Add OCR text for an asset
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails
    pub fn add_text(&mut self, asset_id: Uuid, text: &str) -> SearchResult<()> {
        self.texts.push((asset_id, text.to_string()));
        Ok(())
    }

    /// Search OCR text
    ///
    /// # Errors
    ///
    /// Returns an error if search fails
    pub fn search(&self, query: &str) -> SearchResult<Vec<Uuid>> {
        let query_lower = query.to_lowercase();
        let results: Vec<Uuid> = self
            .texts
            .iter()
            .filter(|(_, text)| text.to_lowercase().contains(&query_lower))
            .map(|(id, _)| *id)
            .collect();

        Ok(results)
    }

    /// Commit changes
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails
    pub fn commit(&self) -> SearchResult<()> {
        Ok(())
    }

    /// Delete OCR text for an asset
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub fn delete(&mut self, asset_id: Uuid) -> SearchResult<()> {
        self.texts.retain(|(id, _)| *id != asset_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_search() {
        let temp_dir = std::env::temp_dir().join("ocr_index_test");
        let mut index = OcrIndex::new(&temp_dir).expect("should succeed in test");

        let id = Uuid::new_v4();
        index
            .add_text(id, "Hello World")
            .expect("should succeed in test");

        let results = index.search("hello").expect("should succeed in test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], id);

        std::fs::remove_dir_all(temp_dir).ok();
    }
}
