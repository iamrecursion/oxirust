//! Media file scanner.

use crate::clip::Clip;
use crate::error::{ClipError, ClipResult};
use std::path::{Path, PathBuf};
use tokio::fs;

/// Media file scanner for discovering clips.
#[derive(Debug, Clone)]
pub struct MediaScanner {
    /// File extensions to scan.
    extensions: Vec<String>,

    /// Whether to scan recursively.
    recursive: bool,
}

impl MediaScanner {
    /// Creates a new media scanner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            extensions: vec![
                "mov".to_string(),
                "mp4".to_string(),
                "mxf".to_string(),
                "avi".to_string(),
                "mkv".to_string(),
                "m4v".to_string(),
            ],
            recursive: true,
        }
    }

    /// Sets the file extensions to scan.
    #[must_use]
    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.extensions = extensions;
        self
    }

    /// Sets whether to scan recursively.
    #[must_use]
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Scans a directory for media files.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub async fn scan(&self, path: impl AsRef<Path>) -> ClipResult<Vec<Clip>> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ClipError::FileNotFound(path.to_path_buf()));
        }

        let mut clips = Vec::new();
        self.scan_directory(path, &mut clips).await?;

        Ok(clips)
    }

    fn scan_directory<'a>(
        &'a self,
        path: &'a Path,
        clips: &'a mut Vec<Clip>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ClipResult<()>> + 'a>> {
        Box::pin(async move {
            let mut entries = fs::read_dir(path).await?;

            while let Some(entry) = entries.next_entry().await? {
                let entry_path = entry.path();

                if entry_path.is_dir() {
                    if self.recursive {
                        self.scan_directory(&entry_path, clips).await?;
                    }
                } else if self.is_media_file(&entry_path) {
                    clips.push(Clip::new(entry_path));
                }
            }

            Ok(())
        })
    }

    fn is_media_file(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            self.extensions.iter().any(|e| e.to_lowercase() == ext_str)
        } else {
            false
        }
    }

    /// Scans multiple directories.
    ///
    /// # Errors
    ///
    /// Returns an error if any directory cannot be read.
    pub async fn scan_multiple(&self, paths: &[PathBuf]) -> ClipResult<Vec<Clip>> {
        let mut all_clips = Vec::new();

        for path in paths {
            let clips = self.scan(path).await?;
            all_clips.extend(clips);
        }

        Ok(all_clips)
    }
}

impl Default for MediaScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_media_file() {
        let scanner = MediaScanner::new();

        assert!(scanner.is_media_file(Path::new("/test.mov")));
        assert!(scanner.is_media_file(Path::new("/test.MP4")));
        assert!(!scanner.is_media_file(Path::new("/test.txt")));
    }

    #[test]
    fn test_custom_extensions() {
        let scanner = MediaScanner::new().with_extensions(vec!["custom".to_string()]);

        assert!(scanner.is_media_file(Path::new("/test.custom")));
        assert!(!scanner.is_media_file(Path::new("/test.mov")));
    }
}
