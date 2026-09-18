//! Batch import system.

use crate::clip::Clip;
use std::path::PathBuf;

/// Options for batch import.
#[derive(Debug, Clone)]
pub struct BatchImportOptions {
    /// Auto-generate keywords from file names.
    pub auto_keywords: bool,

    /// Extract metadata from files.
    pub extract_metadata: bool,

    /// Create bins based on directory structure.
    pub create_bins: bool,
}

impl Default for BatchImportOptions {
    fn default() -> Self {
        Self {
            auto_keywords: true,
            extract_metadata: true,
            create_bins: true,
        }
    }
}

/// Batch importer for clips.
#[derive(Debug, Clone)]
pub struct BatchImporter {
    options: BatchImportOptions,
}

impl BatchImporter {
    /// Creates a new batch importer.
    #[must_use]
    pub fn new(options: BatchImportOptions) -> Self {
        Self { options }
    }

    /// Imports clips from file paths.
    #[must_use]
    pub fn import(&self, paths: Vec<PathBuf>) -> Vec<Clip> {
        paths
            .into_iter()
            .map(|path| {
                let mut clip = Clip::new(path.clone());

                if self.options.auto_keywords {
                    Self::add_auto_keywords(&mut clip, &path);
                }

                clip
            })
            .collect()
    }

    fn add_auto_keywords(clip: &mut Clip, path: &std::path::Path) {
        // Add file name without extension as keyword
        if let Some(stem) = path.file_stem() {
            if let Some(name) = stem.to_str() {
                clip.add_keyword(name.to_string());
            }
        }

        // Add parent directory name as keyword
        if let Some(parent) = path.parent() {
            if let Some(dir_name) = parent.file_name() {
                if let Some(name) = dir_name.to_str() {
                    clip.add_keyword(name.to_string());
                }
            }
        }
    }

    /// Imports clips with progress callback.
    #[allow(clippy::needless_pass_by_value)]
    pub fn import_with_progress<F>(&self, paths: Vec<PathBuf>, mut callback: F) -> Vec<Clip>
    where
        F: FnMut(usize, usize),
    {
        let total = paths.len();
        let mut clips = Vec::new();

        for (index, path) in paths.into_iter().enumerate() {
            let mut clip = Clip::new(path.clone());

            if self.options.auto_keywords {
                Self::add_auto_keywords(&mut clip, &path);
            }

            clips.push(clip);
            callback(index + 1, total);
        }

        clips
    }
}

impl Default for BatchImporter {
    fn default() -> Self {
        Self::new(BatchImportOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_import() {
        let paths = vec![
            PathBuf::from("/media/interview.mov"),
            PathBuf::from("/media/broll.mov"),
        ];

        let importer = BatchImporter::default();
        let clips = importer.import(paths);

        assert_eq!(clips.len(), 2);
        assert!(clips[0].keywords.contains(&"interview".to_string()));
        assert!(clips[0].keywords.contains(&"media".to_string()));
    }

    #[test]
    fn test_import_with_progress() {
        let paths = vec![PathBuf::from("/test1.mov"), PathBuf::from("/test2.mov")];

        let importer = BatchImporter::default();
        let mut progress_calls = 0;

        let clips = importer.import_with_progress(paths, |current, total| {
            progress_calls += 1;
            assert!(current <= total);
        });

        assert_eq!(clips.len(), 2);
        assert_eq!(progress_calls, 2);
    }
}
