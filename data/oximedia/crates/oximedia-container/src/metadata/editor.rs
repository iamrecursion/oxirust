//! High-level metadata editing API.
//!
//! Provides a convenient interface for reading and modifying metadata
//! in media files.

use crate::ContainerFormat;

use super::tags::{StandardTag, TagValue};
// `TagMap` is only referenced by `MetadataEditor` and its filesystem-based
// free functions below, all of which require file/async I/O that is not
// available on `wasm32` and are therefore gated out on that target.
#[cfg(not(target_arch = "wasm32"))]
use super::tags::TagMap;

#[cfg(not(target_arch = "wasm32"))]
use oximedia_core::OxiResult;
#[cfg(not(target_arch = "wasm32"))]
use oximedia_io::FileSource;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use super::reader::{detect_format, FlacMetadataReader, MatroskaMetadataReader, MetadataReader};
#[cfg(not(target_arch = "wasm32"))]
use super::util::MediaSourceExt;
#[cfg(not(target_arch = "wasm32"))]
use super::writer::{
    FlacMetadataWriter, MatroskaMetadataWriter, MetadataWriter, OggMetadataWriter,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::demux::Demuxer;
#[cfg(not(target_arch = "wasm32"))]
use crate::demux::MatroskaDemuxer;

/// Metadata format detection result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataFormat {
    /// FLAC uses Vorbis comments.
    Flac,
    /// Ogg (Vorbis, Opus) uses Vorbis comments.
    Ogg,
    /// Matroska/WebM uses native tags.
    Matroska,
    /// `WebM` (subset of Matroska).
    WebM,
}

impl From<ContainerFormat> for MetadataFormat {
    fn from(format: ContainerFormat) -> Self {
        match format {
            ContainerFormat::Flac => Self::Flac,
            ContainerFormat::Ogg => Self::Ogg,
            ContainerFormat::WebM => Self::WebM,
            _ => Self::Matroska, // Default fallback (includes Matroska)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// A metadata editor for media files.
///
/// Provides high-level operations for reading and writing metadata tags.
///
/// # Example
///
/// ```ignore
/// use oximedia_container::metadata::MetadataEditor;
///
/// let mut editor = MetadataEditor::open("audio.flac").await?;
///
/// // Read existing tags
/// if let Some(title) = editor.get_text("TITLE") {
///     println!("Current title: {}", title);
/// }
///
/// // Modify tags
/// editor.set("TITLE", "New Title");
/// editor.set("ARTIST", "New Artist");
/// editor.remove("COMMENT");
///
/// // Save changes
/// editor.save().await?;
/// ```
pub struct MetadataEditor {
    /// Path to the media file.
    path: PathBuf,
    /// Detected metadata format.
    format: MetadataFormat,
    /// Current tag map.
    tags: TagMap,
    /// Whether tags have been modified.
    modified: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl MetadataEditor {
    /// Opens a media file for metadata editing.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The format is not supported
    /// - Reading metadata fails
    pub async fn open(path: impl AsRef<Path>) -> OxiResult<Self> {
        let path = path.as_ref().to_path_buf();

        // Detect format
        let mut magic = [0u8; 8];
        let mut source_clone = FileSource::open(&path).await?;
        source_clone.read_exact(&mut magic).await?;

        let container_format = detect_format(&magic)?;
        let format = MetadataFormat::from(container_format);

        // Read metadata based on format
        let tags = match format {
            MetadataFormat::Flac => {
                let source = FileSource::open(&path).await?;
                FlacMetadataReader::read(source).await?
            }
            MetadataFormat::Ogg => {
                // For Ogg, we would need to use OggDemuxer
                // This is a simplified placeholder
                TagMap::new()
            }
            MetadataFormat::Matroska | MetadataFormat::WebM => {
                let source = FileSource::open(&path).await?;
                let mut demuxer = MatroskaDemuxer::new(source);
                demuxer.probe().await?;

                let tags = demuxer.tags();
                MatroskaMetadataReader::convert_tags(tags)
            }
        };

        Ok(Self {
            path,
            format,
            tags,
            modified: false,
        })
    }

    /// Returns the metadata format of the file.
    #[must_use]
    pub const fn format(&self) -> MetadataFormat {
        self.format
    }

    /// Returns true if tags have been modified.
    #[must_use]
    pub const fn is_modified(&self) -> bool {
        self.modified
    }

    /// Gets a tag value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&TagValue> {
        self.tags.get(key)
    }

    /// Gets a text tag value by key.
    #[must_use]
    pub fn get_text(&self, key: &str) -> Option<&str> {
        self.tags.get_text(key)
    }

    /// Gets all values for a tag key.
    #[must_use]
    pub fn get_all(&self, key: &str) -> &[TagValue] {
        self.tags.get_all(key)
    }

    /// Gets a standard tag value.
    #[must_use]
    pub fn get_standard(&self, tag: StandardTag) -> Option<&TagValue> {
        self.tags.get_standard(tag)
    }

    /// Sets a tag value, replacing any existing values.
    pub fn set(&mut self, key: impl AsRef<str>, value: impl Into<TagValue>) {
        self.tags.set(key, value);
        self.modified = true;
    }

    /// Adds a tag value without removing existing values.
    pub fn add(&mut self, key: impl AsRef<str>, value: impl Into<TagValue>) {
        self.tags.add(key, value);
        self.modified = true;
    }

    /// Sets a standard tag value.
    pub fn set_standard(&mut self, tag: StandardTag, value: impl Into<TagValue>) {
        self.tags.set_standard(tag, value);
        self.modified = true;
    }

    /// Removes a tag and all its values.
    ///
    /// Returns true if the tag existed.
    pub fn remove(&mut self, key: &str) -> bool {
        let removed = self.tags.remove(key);
        if removed {
            self.modified = true;
        }
        removed
    }

    /// Clears all tags.
    pub fn clear(&mut self) {
        if !self.tags.is_empty() {
            self.tags.clear();
            self.modified = true;
        }
    }

    /// Returns an iterator over all tag keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tags.keys()
    }

    /// Returns an iterator over all tag entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TagValue)> {
        self.tags.iter()
    }

    /// Returns a reference to the tag map.
    #[must_use]
    pub const fn tags(&self) -> &TagMap {
        &self.tags
    }

    /// Returns a mutable reference to the tag map.
    pub fn tags_mut(&mut self) -> &mut TagMap {
        self.modified = true;
        &mut self.tags
    }

    /// Saves metadata changes to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened for writing
    /// - Writing fails
    /// - The format doesn't support metadata writing
    pub async fn save(&mut self) -> OxiResult<()> {
        if !self.modified {
            return Ok(());
        }

        let mut source = FileSource::open(&self.path).await?;

        match self.format {
            MetadataFormat::Flac => {
                FlacMetadataWriter::write(&mut source, &self.tags).await?;
            }
            MetadataFormat::Ogg => {
                OggMetadataWriter::write(&mut source, &self.tags).await?;
            }
            MetadataFormat::Matroska | MetadataFormat::WebM => {
                MatroskaMetadataWriter::write(&mut source, &self.tags).await?;
            }
        }

        self.modified = false;
        Ok(())
    }

    /// Discards any unsaved changes and reloads metadata from the file.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    pub async fn reload(&mut self) -> OxiResult<()> {
        let new_editor = Self::open(&self.path).await?;
        self.tags = new_editor.tags;
        self.modified = false;
        Ok(())
    }

    // ─── Batch operations ───────────────────────────────────────────────

    /// Copies all tags from `source` into this editor, replacing existing values.
    ///
    /// This is equivalent to merging the source's tag map into this editor's.
    pub fn copy_all_from(&mut self, source: &TagMap) {
        self.tags.merge(source);
        self.modified = true;
    }

    /// Copies specific tags from `source` into this editor.
    ///
    /// Only tags whose keys are in `tag_keys` will be copied.
    /// Existing values for those keys will be replaced.
    pub fn copy_tags_from(&mut self, source: &TagMap, tag_keys: &[&str]) {
        for &key in tag_keys {
            let values = source.get_all(key);
            if !values.is_empty() {
                // Replace existing
                self.tags.remove(key);
                for val in values {
                    self.tags.add(key, val.clone());
                }
                self.modified = true;
            }
        }
    }

    /// Copies standard tags from `source` into this editor.
    ///
    /// Only the specified standard tags will be copied.
    pub fn copy_standard_tags_from(&mut self, source: &TagMap, tags: &[StandardTag]) {
        for &tag in tags {
            if let Some(value) = source.get_standard(tag) {
                self.tags.set_standard(tag, value.clone());
                self.modified = true;
            }
        }
    }

    /// Applies a batch of tag operations atomically.
    ///
    /// All operations are applied in order. If any operation would produce
    /// no change, it is silently skipped.
    pub fn apply_batch(&mut self, operations: &[BatchTagOperation]) {
        let mut any_change = false;
        for op in operations {
            match op {
                BatchTagOperation::Set { key, value } => {
                    self.tags.set(key.as_str(), value.clone());
                    any_change = true;
                }
                BatchTagOperation::Add { key, value } => {
                    self.tags.add(key.as_str(), value.clone());
                    any_change = true;
                }
                BatchTagOperation::Remove { key } => {
                    if self.tags.remove(key) {
                        any_change = true;
                    }
                }
                BatchTagOperation::Rename { from, to } => {
                    let values: Vec<TagValue> = self.tags.get_all(from).to_vec();
                    if !values.is_empty() {
                        self.tags.remove(from);
                        for val in values {
                            self.tags.add(to.as_str(), val);
                        }
                        any_change = true;
                    }
                }
                BatchTagOperation::SetStandard { tag, value } => {
                    self.tags.set_standard(*tag, value.clone());
                    any_change = true;
                }
                BatchTagOperation::RemoveAll => {
                    if !self.tags.is_empty() {
                        self.tags.clear();
                        any_change = true;
                    }
                }
                BatchTagOperation::ReplaceValue {
                    key,
                    old_value,
                    new_value,
                } => {
                    let values: Vec<TagValue> = self.tags.get_all(key).to_vec();
                    let old_text = old_value.as_str();
                    let has_match = values.iter().any(|v| v.as_text() == Some(old_text));
                    if has_match {
                        self.tags.remove(key);
                        for val in values {
                            if val.as_text() == Some(old_text) {
                                self.tags.add(key.as_str(), new_value.clone());
                            } else {
                                self.tags.add(key.as_str(), val);
                            }
                        }
                        any_change = true;
                    }
                }
                BatchTagOperation::PrefixValues { key, prefix } => {
                    let values: Vec<TagValue> = self.tags.get_all(key).to_vec();
                    if !values.is_empty() {
                        self.tags.remove(key);
                        for val in values {
                            if let Some(text) = val.as_text() {
                                let new_text = format!("{prefix}{text}");
                                self.tags.add(key.as_str(), TagValue::Text(new_text));
                            } else {
                                self.tags.add(key.as_str(), val);
                            }
                        }
                        any_change = true;
                    }
                }
            }
        }
        if any_change {
            self.modified = true;
        }
    }

    /// Returns a diff between this editor's tags and another tag map.
    ///
    /// Returns a list of changes needed to transform `other` into this editor's tags.
    #[must_use]
    pub fn diff(&self, other: &TagMap) -> Vec<TagDiff> {
        let mut diffs = Vec::new();

        // Tags present in self but not in other (added)
        for (key, value) in self.tags.iter() {
            if other.get(key).is_none() {
                diffs.push(TagDiff::Added {
                    key: key.to_string(),
                    value: value.clone(),
                });
            }
        }

        // Tags present in other but not in self (removed)
        for (key, _value) in other.iter() {
            if self.tags.get(key).is_none() {
                diffs.push(TagDiff::Removed {
                    key: key.to_string(),
                });
            }
        }

        // Tags present in both but with different values (modified)
        for (key, self_value) in self.tags.iter() {
            if let Some(other_value) = other.get(key) {
                if self_value != other_value {
                    diffs.push(TagDiff::Modified {
                        key: key.to_string(),
                        old_value: other_value.clone(),
                        new_value: self_value.clone(),
                    });
                }
            }
        }

        diffs
    }
}

/// A batch tag operation to apply to a metadata editor.
///
/// Operations are applied in order via [`MetadataEditor::apply_batch`].
#[derive(Debug, Clone)]
pub enum BatchTagOperation {
    /// Set a tag value (replaces existing).
    Set {
        /// Tag key.
        key: String,
        /// New value.
        value: TagValue,
    },
    /// Add a tag value (preserves existing).
    Add {
        /// Tag key.
        key: String,
        /// Value to add.
        value: TagValue,
    },
    /// Remove a tag entirely.
    Remove {
        /// Tag key to remove.
        key: String,
    },
    /// Rename a tag key (preserves values).
    Rename {
        /// Original key.
        from: String,
        /// New key.
        to: String,
    },
    /// Set a standard tag value.
    SetStandard {
        /// Standard tag identifier.
        tag: StandardTag,
        /// New value.
        value: TagValue,
    },
    /// Remove all tags.
    RemoveAll,
    /// Replace a specific text value within a tag.
    ReplaceValue {
        /// Tag key.
        key: String,
        /// Old text value to find.
        old_value: String,
        /// New value to replace with.
        new_value: TagValue,
    },
    /// Prefix all text values of a tag with a string.
    PrefixValues {
        /// Tag key.
        key: String,
        /// Prefix to add.
        prefix: String,
    },
}

impl BatchTagOperation {
    /// Creates a Set operation.
    #[must_use]
    pub fn set(key: impl Into<String>, value: impl Into<TagValue>) -> Self {
        Self::Set {
            key: key.into(),
            value: value.into(),
        }
    }

    /// Creates an Add operation.
    #[must_use]
    pub fn add(key: impl Into<String>, value: impl Into<TagValue>) -> Self {
        Self::Add {
            key: key.into(),
            value: value.into(),
        }
    }

    /// Creates a Remove operation.
    #[must_use]
    pub fn remove(key: impl Into<String>) -> Self {
        Self::Remove { key: key.into() }
    }

    /// Creates a Rename operation.
    #[must_use]
    pub fn rename(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self::Rename {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Creates a `SetStandard` operation.
    #[must_use]
    pub fn set_standard(tag: StandardTag, value: impl Into<TagValue>) -> Self {
        Self::SetStandard {
            tag,
            value: value.into(),
        }
    }

    /// Creates a `RemoveAll` operation.
    #[must_use]
    pub const fn remove_all() -> Self {
        Self::RemoveAll
    }

    /// Creates a `ReplaceValue` operation.
    #[must_use]
    pub fn replace_value(
        key: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<TagValue>,
    ) -> Self {
        Self::ReplaceValue {
            key: key.into(),
            old_value: old_value.into(),
            new_value: new_value.into(),
        }
    }

    /// Creates a `PrefixValues` operation.
    #[must_use]
    pub fn prefix_values(key: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self::PrefixValues {
            key: key.into(),
            prefix: prefix.into(),
        }
    }
}

/// A diff entry describing a change between two tag maps.
#[derive(Debug, Clone, PartialEq)]
pub enum TagDiff {
    /// A tag was added (present in new, absent in old).
    Added {
        /// Tag key.
        key: String,
        /// Added value.
        value: TagValue,
    },
    /// A tag was removed (present in old, absent in new).
    Removed {
        /// Tag key.
        key: String,
    },
    /// A tag value was modified.
    Modified {
        /// Tag key.
        key: String,
        /// Old value.
        old_value: TagValue,
        /// New value.
        new_value: TagValue,
    },
}

impl TagDiff {
    /// Returns the key of this diff entry.
    #[must_use]
    pub fn key(&self) -> &str {
        match self {
            Self::Added { key, .. } | Self::Removed { key, .. } | Self::Modified { key, .. } => key,
        }
    }

    /// Returns true if this is an addition.
    #[must_use]
    pub const fn is_added(&self) -> bool {
        matches!(self, Self::Added { .. })
    }

    /// Returns true if this is a removal.
    #[must_use]
    pub const fn is_removed(&self) -> bool {
        matches!(self, Self::Removed { .. })
    }

    /// Returns true if this is a modification.
    #[must_use]
    pub const fn is_modified(&self) -> bool {
        matches!(self, Self::Modified { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchMetadataEditor
// ─────────────────────────────────────────────────────────────────────────────

/// A single operation in a [`BatchMetadataEditor`] pipeline.
#[derive(Debug, Clone)]
pub enum MetadataOp {
    /// Set `key` to `value`, replacing any existing value.
    Set {
        /// Tag key (will be uppercased on application).
        key: String,
        /// New value.
        value: TagValue,
    },
    /// Remove `key` entirely.  A no-op if the key is absent.
    Remove {
        /// Tag key to remove.
        key: String,
    },
    /// Rename a key from `from` to `to`, preserving its value.
    /// Silently skipped if `from` is absent.
    Rename {
        /// Original key.
        from: String,
        /// Replacement key.
        to: String,
    },
    /// Set `key` to `value` only if the key is absent.
    SetIfAbsent {
        /// Tag key.
        key: String,
        /// Default value.
        value: TagValue,
    },
}

/// Builder-style batch metadata editor.
///
/// Accumulates a sequence of [`MetadataOp`] operations and applies them
/// atomically to a `HashMap<String, TagValue>` or to a media file via
/// [`apply_to_file`].
///
/// # Example
///
/// ```ignore
/// use std::collections::HashMap;
/// use oximedia_container::metadata::{BatchMetadataEditor, TagValue};
///
/// let mut map: HashMap<String, TagValue> = HashMap::new();
/// map.insert("TITLE".to_string(), "Old".into());
///
/// let applied = BatchMetadataEditor::new()
///     .set("TITLE", "New")
///     .set("ARTIST", "Artist")
///     .remove("COMMENT")
///     .apply(&mut map)
///     .expect("apply should succeed");
///
/// assert_eq!(applied, 2); // TITLE changed + ARTIST inserted
/// ```
///
/// [`apply_to_file`]: BatchMetadataEditor::apply_to_file
#[derive(Debug, Default)]
pub struct BatchMetadataEditor {
    operations: Vec<MetadataOp>,
}

impl BatchMetadataEditor {
    /// Creates a new empty editor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a [`MetadataOp::Set`] operation.
    #[must_use]
    pub fn set(mut self, key: impl Into<String>, value: impl Into<TagValue>) -> Self {
        self.operations.push(MetadataOp::Set {
            key: key.into(),
            value: value.into(),
        });
        self
    }

    /// Appends a [`MetadataOp::Remove`] operation.
    #[must_use]
    pub fn remove(mut self, key: impl Into<String>) -> Self {
        self.operations.push(MetadataOp::Remove { key: key.into() });
        self
    }

    /// Appends a [`MetadataOp::Rename`] operation.
    #[must_use]
    pub fn rename(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.operations.push(MetadataOp::Rename {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// Appends a [`MetadataOp::SetIfAbsent`] operation.
    #[must_use]
    pub fn set_if_absent(mut self, key: impl Into<String>, value: impl Into<TagValue>) -> Self {
        self.operations.push(MetadataOp::SetIfAbsent {
            key: key.into(),
            value: value.into(),
        });
        self
    }

    /// Applies all operations to `metadata` in order.
    ///
    /// Returns the number of operations that actually mutated the map (e.g.
    /// a `Set` that writes the same value that was already present does not
    /// count; a `Remove` on an absent key does not count).
    ///
    /// # Errors
    ///
    /// Currently infallible (returns `Ok`), but the `Result` return type
    /// allows future versions to add validating operations.
    pub fn apply(
        &self,
        metadata: &mut std::collections::HashMap<String, TagValue>,
    ) -> oximedia_core::OxiResult<usize> {
        let mut applied: usize = 0;
        for op in &self.operations {
            match op {
                MetadataOp::Set { key, value } => {
                    let changed = metadata
                        .get(key.as_str())
                        .map_or(true, |existing| existing != value);
                    metadata.insert(key.clone(), value.clone());
                    if changed {
                        applied += 1;
                    }
                }
                MetadataOp::Remove { key } => {
                    if metadata.remove(key.as_str()).is_some() {
                        applied += 1;
                    }
                }
                MetadataOp::Rename { from, to } => {
                    if let Some(value) = metadata.remove(from.as_str()) {
                        metadata.insert(to.clone(), value);
                        applied += 1;
                    }
                    // Absent `from` key → silently skip
                }
                MetadataOp::SetIfAbsent { key, value } => {
                    if !metadata.contains_key(key.as_str()) {
                        metadata.insert(key.clone(), value.clone());
                        applied += 1;
                    }
                }
            }
        }
        Ok(applied)
    }

    /// Applies all operations to the tags of the media file at `path`.
    ///
    /// The method:
    /// 1. Opens the file and reads its current tag map.
    /// 2. Copies tags into a `HashMap<String, TagValue>`.
    /// 3. Applies all operations via [`apply`].
    /// 4. Writes the modified tags back to the file.
    ///
    /// Returns the count of operations that changed the tag map.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, read, or written.
    ///
    /// [`apply`]: BatchMetadataEditor::apply
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply_to_file(&self, path: &std::path::Path) -> oximedia_core::OxiResult<usize> {
        use oximedia_core::OxiError;

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                OxiError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?;

        rt.block_on(async {
            let mut editor = MetadataEditor::open(path).await?;

            // Snapshot current tags into a HashMap
            let mut map: std::collections::HashMap<String, TagValue> = editor
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();

            let count = self.apply(&mut map)?;

            // Re-sync the editor from the modified HashMap
            editor.clear();
            for (k, v) in &map {
                editor.set(k, v.clone());
            }

            editor.save().await?;
            Ok(count)
        })
    }

    /// Returns the number of pending operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns `true` if no operations have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// Reads metadata from a media file without creating an editor.
///
/// This is a convenience function for read-only metadata access.
///
/// # Errors
///
/// Returns an error if reading or parsing fails.
pub async fn read_metadata(path: impl AsRef<Path>) -> OxiResult<TagMap> {
    let editor = MetadataEditor::open(path).await?;
    Ok(editor.tags)
}

#[cfg(not(target_arch = "wasm32"))]
/// Writes metadata to a file.
///
/// This is a convenience function for updating metadata without
/// reading existing tags first.
///
/// # Errors
///
/// Returns an error if writing fails.
pub async fn write_metadata(path: impl AsRef<Path>, tags: &TagMap) -> OxiResult<()> {
    let mut editor = MetadataEditor::open(path).await?;
    editor.tags = tags.clone();
    editor.modified = true;
    editor.save().await
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_format_from_container_format() {
        assert_eq!(
            MetadataFormat::from(ContainerFormat::Flac),
            MetadataFormat::Flac
        );
        assert_eq!(
            MetadataFormat::from(ContainerFormat::Ogg),
            MetadataFormat::Ogg
        );
        assert_eq!(
            MetadataFormat::from(ContainerFormat::Matroska),
            MetadataFormat::Matroska
        );
        assert_eq!(
            MetadataFormat::from(ContainerFormat::WebM),
            MetadataFormat::WebM
        );
    }

    #[test]
    fn test_metadata_editor_modification_tracking() {
        let editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        assert!(!editor.is_modified());
    }

    #[test]
    fn test_metadata_editor_set() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("TITLE", "Test");
        assert!(editor.is_modified());
        assert_eq!(editor.get_text("TITLE"), Some("Test"));
    }

    #[test]
    fn test_metadata_editor_add() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.add("ARTIST", "Artist 1");
        editor.add("ARTIST", "Artist 2");

        assert!(editor.is_modified());
        let artists = editor.get_all("ARTIST");
        assert_eq!(artists.len(), 2);
    }

    #[test]
    fn test_metadata_editor_remove() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("TITLE", "Test");
        editor.modified = false; // Reset flag

        assert!(editor.remove("TITLE"));
        assert!(editor.is_modified());
        assert!(!editor.remove("TITLE"));
    }

    #[test]
    fn test_metadata_editor_clear() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("TITLE", "Test");
        editor.set("ARTIST", "Test");
        editor.modified = false;

        editor.clear();
        assert!(editor.is_modified());
        assert!(editor.tags.is_empty());
    }

    #[test]
    fn test_metadata_editor_standard_tags() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set_standard(StandardTag::Title, "Test Title");
        assert_eq!(
            editor
                .get_standard(StandardTag::Title)
                .and_then(|v| v.as_text()),
            Some("Test Title")
        );
    }

    #[test]
    fn test_metadata_editor_iter() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("TITLE", "Title");
        editor.set("ARTIST", "Artist");

        let entries: Vec<_> = editor.iter().collect();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_metadata_editor_keys() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("TITLE", "Title");
        editor.set("ARTIST", "Artist");

        let keys: Vec<_> = editor.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"TITLE"));
        assert!(keys.contains(&"ARTIST"));
    }

    // ── Batch operation tests ───────────────────────────────────────────

    #[test]
    fn test_copy_all_from() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };
        editor.set("TITLE", "Original");

        let mut source = TagMap::new();
        source.set("TITLE", "Copied");
        source.set("ARTIST", "New Artist");
        source.set("ALBUM", "New Album");

        editor.copy_all_from(&source);

        assert!(editor.is_modified());
        assert_eq!(editor.get_text("TITLE"), Some("Copied")); // replaced
        assert_eq!(editor.get_text("ARTIST"), Some("New Artist"));
        assert_eq!(editor.get_text("ALBUM"), Some("New Album"));
    }

    #[test]
    fn test_copy_tags_from_selective() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        let mut source = TagMap::new();
        source.set("TITLE", "Source Title");
        source.set("ARTIST", "Source Artist");
        source.set("ALBUM", "Source Album");

        editor.copy_tags_from(&source, &["TITLE", "ALBUM"]);

        assert!(editor.is_modified());
        assert_eq!(editor.get_text("TITLE"), Some("Source Title"));
        assert_eq!(editor.get_text("ALBUM"), Some("Source Album"));
        assert!(editor.get_text("ARTIST").is_none()); // not copied
    }

    #[test]
    fn test_copy_tags_from_nonexistent() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        let source = TagMap::new();
        editor.copy_tags_from(&source, &["TITLE"]);

        assert!(!editor.is_modified()); // Nothing copied
    }

    #[test]
    fn test_copy_standard_tags_from() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        let mut source = TagMap::new();
        source.set_standard(StandardTag::Title, "Std Title");
        source.set_standard(StandardTag::Artist, "Std Artist");
        source.set_standard(StandardTag::Album, "Std Album");

        editor.copy_standard_tags_from(&source, &[StandardTag::Title, StandardTag::Album]);

        assert!(editor.is_modified());
        assert_eq!(
            editor
                .get_standard(StandardTag::Title)
                .and_then(|v| v.as_text()),
            Some("Std Title")
        );
        assert_eq!(
            editor
                .get_standard(StandardTag::Album)
                .and_then(|v| v.as_text()),
            Some("Std Album")
        );
        assert!(editor.get_standard(StandardTag::Artist).is_none());
    }

    #[test]
    fn test_apply_batch_set_and_add() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        let ops = vec![
            BatchTagOperation::set("TITLE", "Batch Title"),
            BatchTagOperation::set("ARTIST", "Main Artist"),
            BatchTagOperation::add("ARTIST", "Featured Artist"),
        ];

        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        assert_eq!(editor.get_text("TITLE"), Some("Batch Title"));
        assert_eq!(editor.get_all("ARTIST").len(), 2);
    }

    #[test]
    fn test_apply_batch_remove() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("TITLE", "Test");
        editor.set("ARTIST", "Test");
        editor.modified = false;

        let ops = vec![BatchTagOperation::remove("TITLE")];
        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        assert!(editor.get_text("TITLE").is_none());
        assert_eq!(editor.get_text("ARTIST"), Some("Test")); // untouched
    }

    #[test]
    fn test_apply_batch_rename() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("COMMENT", "My comment");
        editor.modified = false;

        let ops = vec![BatchTagOperation::rename("COMMENT", "DESCRIPTION")];
        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        assert!(editor.get_text("COMMENT").is_none());
        assert_eq!(editor.get_text("DESCRIPTION"), Some("My comment"));
    }

    #[test]
    fn test_apply_batch_set_standard() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        let ops = vec![
            BatchTagOperation::set_standard(StandardTag::Title, "Std Batch"),
            BatchTagOperation::set_standard(StandardTag::Genre, "Rock"),
        ];

        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        assert_eq!(
            editor
                .get_standard(StandardTag::Title)
                .and_then(|v| v.as_text()),
            Some("Std Batch")
        );
        assert_eq!(
            editor
                .get_standard(StandardTag::Genre)
                .and_then(|v| v.as_text()),
            Some("Rock")
        );
    }

    #[test]
    fn test_apply_batch_remove_all() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.set("TITLE", "Title");
        editor.set("ARTIST", "Artist");
        editor.modified = false;

        let ops = vec![BatchTagOperation::remove_all()];
        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        assert!(editor.tags().is_empty());
    }

    #[test]
    fn test_apply_batch_replace_value() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.add("ARTIST", "Old Artist");
        editor.add("ARTIST", "Keep This");
        editor.modified = false;

        let ops = vec![BatchTagOperation::replace_value(
            "ARTIST",
            "Old Artist",
            "New Artist",
        )];
        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        let artists = editor.get_all("ARTIST");
        assert_eq!(artists.len(), 2);
        // One should be "New Artist", other "Keep This"
        let texts: Vec<_> = artists.iter().filter_map(|v| v.as_text()).collect();
        assert!(texts.contains(&"New Artist"));
        assert!(texts.contains(&"Keep This"));
    }

    #[test]
    fn test_apply_batch_prefix_values() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        editor.add("GENRE", "Rock");
        editor.add("GENRE", "Metal");
        editor.modified = false;

        let ops = vec![BatchTagOperation::prefix_values("GENRE", "Heavy ")];
        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        let genres = editor.get_all("GENRE");
        let texts: Vec<_> = genres.iter().filter_map(|v| v.as_text()).collect();
        assert!(texts.contains(&"Heavy Rock"));
        assert!(texts.contains(&"Heavy Metal"));
    }

    #[test]
    fn test_apply_batch_no_change() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        // Remove a non-existent tag and clear empty tags
        let ops = vec![BatchTagOperation::remove("NONEXISTENT")];
        editor.apply_batch(&ops);

        assert!(!editor.is_modified());
    }

    #[test]
    fn test_apply_batch_complex_workflow() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        // Simulate a complex tag editing workflow
        let ops = vec![
            BatchTagOperation::set("TITLE", "My Song"),
            BatchTagOperation::set("ARTIST", "Band Name"),
            BatchTagOperation::set("ALBUM", "Album Title"),
            BatchTagOperation::set("DATE", "2024"),
            BatchTagOperation::set_standard(StandardTag::Genre, "Alternative"),
            BatchTagOperation::set("TRACKNUMBER", "5"),
            BatchTagOperation::set("TOTALTRACKS", "12"),
        ];

        editor.apply_batch(&ops);

        assert!(editor.is_modified());
        assert_eq!(editor.get_text("TITLE"), Some("My Song"));
        assert_eq!(editor.get_text("ARTIST"), Some("Band Name"));
        assert_eq!(editor.get_text("ALBUM"), Some("Album Title"));
        assert_eq!(editor.get_text("DATE"), Some("2024"));
        assert_eq!(editor.get_text("TRACKNUMBER"), Some("5"));
    }

    // ── TagDiff tests ───────────────────────────────────────────────────

    #[test]
    fn test_diff_added() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };
        editor.set("TITLE", "New");

        let other = TagMap::new();
        let diffs = editor.diff(&other);

        assert!(!diffs.is_empty());
        assert!(diffs.iter().any(|d| d.is_added() && d.key() == "TITLE"));
    }

    #[test]
    fn test_diff_removed() {
        let editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };

        let mut other = TagMap::new();
        other.set("TITLE", "Old");

        let diffs = editor.diff(&other);
        assert!(diffs.iter().any(|d| d.is_removed() && d.key() == "TITLE"));
    }

    #[test]
    fn test_diff_modified() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };
        editor.set("TITLE", "New");

        let mut other = TagMap::new();
        other.set("TITLE", "Old");

        let diffs = editor.diff(&other);
        assert!(diffs.iter().any(|d| d.is_modified() && d.key() == "TITLE"));
    }

    #[test]
    fn test_diff_no_changes() {
        let mut editor = MetadataEditor {
            path: PathBuf::from("test.flac"),
            format: MetadataFormat::Flac,
            tags: TagMap::new(),
            modified: false,
        };
        editor.set("TITLE", "Same");

        let mut other = TagMap::new();
        other.set("TITLE", "Same");

        let diffs = editor.diff(&other);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_tag_diff_methods() {
        let added = TagDiff::Added {
            key: "TITLE".to_string(),
            value: TagValue::Text("Test".to_string()),
        };
        assert!(added.is_added());
        assert!(!added.is_removed());
        assert!(!added.is_modified());
        assert_eq!(added.key(), "TITLE");

        let removed = TagDiff::Removed {
            key: "ARTIST".to_string(),
        };
        assert!(removed.is_removed());

        let modified = TagDiff::Modified {
            key: "ALBUM".to_string(),
            old_value: TagValue::Text("Old".to_string()),
            new_value: TagValue::Text("New".to_string()),
        };
        assert!(modified.is_modified());
    }

    // ── BatchTagOperation constructor tests ─────────────────────────────

    #[test]
    fn test_batch_op_constructors() {
        let _set = BatchTagOperation::set("TITLE", "Test");
        let _add = BatchTagOperation::add("ARTIST", "Test");
        let _remove = BatchTagOperation::remove("COMMENT");
        let _rename = BatchTagOperation::rename("OLD", "NEW");
        let _std = BatchTagOperation::set_standard(StandardTag::Title, "Test");
        let _clear = BatchTagOperation::remove_all();
        let _replace = BatchTagOperation::replace_value("ARTIST", "Old", "New");
        let _prefix = BatchTagOperation::prefix_values("GENRE", "Classic ");
    }

    // ── BatchMetadataEditor tests ────────────────────────────────────────

    #[test]
    fn test_batch_metadata_editor_set() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        let count = BatchMetadataEditor::new()
            .set("TITLE", TagValue::Text("Hello".to_string()))
            .set("ARTIST", TagValue::Text("World".to_string()))
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 2);
        assert_eq!(map.get("TITLE").and_then(|v| v.as_text()), Some("Hello"));
        assert_eq!(map.get("ARTIST").and_then(|v| v.as_text()), Some("World"));
    }

    #[test]
    fn test_batch_metadata_editor_remove() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        map.insert("COMMENT".to_string(), TagValue::Text("old".to_string()));
        let count = BatchMetadataEditor::new()
            .remove("COMMENT")
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 1);
        assert!(!map.contains_key("COMMENT"));
    }

    #[test]
    fn test_batch_metadata_editor_remove_absent() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        let count = BatchMetadataEditor::new()
            .remove("NONEXISTENT")
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_batch_metadata_editor_rename() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        map.insert("OLD_KEY".to_string(), TagValue::Text("value".to_string()));
        let count = BatchMetadataEditor::new()
            .rename("OLD_KEY", "NEW_KEY")
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 1);
        assert!(!map.contains_key("OLD_KEY"));
        assert_eq!(map.get("NEW_KEY").and_then(|v| v.as_text()), Some("value"));
    }

    #[test]
    fn test_batch_metadata_editor_rename_absent() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        let count = BatchMetadataEditor::new()
            .rename("MISSING", "TARGET")
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 0);
        assert!(!map.contains_key("TARGET"));
    }

    #[test]
    fn test_batch_metadata_editor_set_if_absent_missing() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        let count = BatchMetadataEditor::new()
            .set_if_absent("TITLE", TagValue::Text("Default".to_string()))
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 1);
        assert_eq!(map.get("TITLE").and_then(|v| v.as_text()), Some("Default"));
    }

    #[test]
    fn test_batch_metadata_editor_set_if_absent_present() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        map.insert("TITLE".to_string(), TagValue::Text("Existing".to_string()));
        let count = BatchMetadataEditor::new()
            .set_if_absent("TITLE", TagValue::Text("Default".to_string()))
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 0);
        assert_eq!(map.get("TITLE").and_then(|v| v.as_text()), Some("Existing"));
    }

    #[test]
    fn test_batch_metadata_editor_set_same_value_no_count() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        map.insert("TITLE".to_string(), TagValue::Text("Same".to_string()));
        let count = BatchMetadataEditor::new()
            .set("TITLE", TagValue::Text("Same".to_string()))
            .apply(&mut map)
            .expect("apply failed");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_batch_metadata_editor_combined() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        map.insert("TITLE".to_string(), TagValue::Text("Old".to_string()));
        map.insert("DELETE_ME".to_string(), TagValue::Text("bye".to_string()));

        let count = BatchMetadataEditor::new()
            .set("TITLE", TagValue::Text("New".to_string()))
            .remove("DELETE_ME")
            .set_if_absent("ARTIST", TagValue::Text("Unknown".to_string()))
            .apply(&mut map)
            .expect("apply failed");

        assert_eq!(count, 3);
        assert_eq!(map.get("TITLE").and_then(|v| v.as_text()), Some("New"));
        assert!(!map.contains_key("DELETE_ME"));
        assert_eq!(map.get("ARTIST").and_then(|v| v.as_text()), Some("Unknown"));
    }

    #[test]
    fn test_batch_metadata_editor_empty() {
        let mut map: std::collections::HashMap<String, TagValue> = std::collections::HashMap::new();
        let editor = BatchMetadataEditor::new();
        assert!(editor.is_empty());
        assert_eq!(editor.len(), 0);
        let count = editor.apply(&mut map).expect("apply failed");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_batch_metadata_editor_len() {
        let editor = BatchMetadataEditor::new()
            .set("A", TagValue::Text("1".to_string()))
            .remove("B")
            .rename("C", "D")
            .set_if_absent("E", TagValue::Text("5".to_string()));
        assert_eq!(editor.len(), 4);
        assert!(!editor.is_empty());
    }
}
