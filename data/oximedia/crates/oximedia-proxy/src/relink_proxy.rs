#![allow(dead_code)]

//! Proxy relinking engine for reconnecting proxies to moved or renamed source media.
//!
//! When source media files are moved, renamed, or migrated to a different
//! storage volume, the proxy-to-original links become stale. This module
//! provides tools to detect broken links and relink proxies to their
//! new source locations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Status of a single proxy link check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkHealth {
    /// The link is valid; the source file exists at the expected path.
    Valid,
    /// The source file is missing at the expected path.
    Broken,
    /// The link was successfully repaired by relinking.
    Relinked,
    /// The link could not be repaired.
    Unresolvable,
}

/// A record associating a proxy file with its original source media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyLinkRecord {
    /// Unique link identifier.
    pub link_id: String,
    /// Path to the proxy file.
    pub proxy_path: PathBuf,
    /// Expected path to the original source file.
    pub source_path: PathBuf,
    /// Current health status of the link.
    pub health: LinkHealth,
    /// File size of the source in bytes (for verification).
    pub source_size_bytes: u64,
    /// Optional checksum of the source for identity verification.
    pub source_checksum: Option<String>,
}

impl ProxyLinkRecord {
    /// Create a new link record.
    pub fn new(link_id: &str, proxy: &str, source: &str) -> Self {
        Self {
            link_id: link_id.to_string(),
            proxy_path: PathBuf::from(proxy),
            source_path: PathBuf::from(source),
            health: LinkHealth::Valid,
            source_size_bytes: 0,
            source_checksum: None,
        }
    }

    /// Set the expected source file size.
    pub fn with_source_size(mut self, bytes: u64) -> Self {
        self.source_size_bytes = bytes;
        self
    }

    /// Set the source checksum.
    pub fn with_checksum(mut self, checksum: &str) -> Self {
        self.source_checksum = Some(checksum.to_string());
        self
    }
}

/// A mapping rule that transforms old source paths to new ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelinkRule {
    /// Prefix to strip from old paths.
    pub old_prefix: String,
    /// Prefix to prepend to produce new paths.
    pub new_prefix: String,
    /// Optional file extension filter (e.g., ".mxf").
    pub extension_filter: Option<String>,
}

impl RelinkRule {
    /// Create a new relink rule that maps one path prefix to another.
    pub fn new(old_prefix: &str, new_prefix: &str) -> Self {
        Self {
            old_prefix: old_prefix.to_string(),
            new_prefix: new_prefix.to_string(),
            extension_filter: None,
        }
    }

    /// Limit this rule to files with a specific extension.
    pub fn with_extension(mut self, ext: &str) -> Self {
        self.extension_filter = Some(ext.to_string());
        self
    }

    /// Try to apply this rule to a source path, returning the new path if applicable.
    pub fn apply(&self, source: &Path) -> Option<PathBuf> {
        let source_str = source.to_string_lossy();
        if !source_str.starts_with(&self.old_prefix) {
            return None;
        }
        if let Some(ext) = &self.extension_filter {
            if let Some(file_ext) = source.extension() {
                let dot_ext = format!(".{}", file_ext.to_string_lossy());
                if dot_ext != *ext {
                    return None;
                }
            } else {
                return None;
            }
        }
        let remainder = &source_str[self.old_prefix.len()..];
        Some(PathBuf::from(format!("{}{}", self.new_prefix, remainder)))
    }
}

/// Result of a batch relink operation.
#[derive(Debug, Clone)]
pub struct RelinkReport {
    /// Total number of links checked.
    pub total_checked: usize,
    /// Number of links that were valid before relinking.
    pub already_valid: usize,
    /// Number of links successfully relinked.
    pub relinked: usize,
    /// Number of links that could not be resolved.
    pub unresolvable: usize,
    /// Map of link_id to new source path for relinked entries.
    pub relink_map: HashMap<String, PathBuf>,
}

/// Engine that manages proxy relinking operations.
#[derive(Debug)]
pub struct RelinkEngine {
    /// Rules to apply in order.
    rules: Vec<RelinkRule>,
}

impl RelinkEngine {
    /// Create a new relink engine with no rules.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a relink rule.
    pub fn add_rule(&mut self, rule: RelinkRule) {
        self.rules.push(rule);
    }

    /// Return the number of rules configured.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Attempt to relink a single record by applying rules in order.
    /// Returns the new path if a rule matched, or None.
    pub fn try_relink(&self, record: &ProxyLinkRecord) -> Option<PathBuf> {
        for rule in &self.rules {
            if let Some(new_path) = rule.apply(&record.source_path) {
                return Some(new_path);
            }
        }
        None
    }

    /// Run relinking on a batch of records, updating their health status.
    pub fn relink_batch(&self, records: &mut [ProxyLinkRecord]) -> RelinkReport {
        let total_checked = records.len();
        let mut already_valid = 0;
        let mut relinked = 0;
        let mut unresolvable = 0;
        let mut relink_map = HashMap::new();

        for record in records.iter_mut() {
            if record.health == LinkHealth::Valid {
                already_valid += 1;
                continue;
            }
            if record.health == LinkHealth::Broken {
                if let Some(new_path) = self.try_relink(record) {
                    record.source_path = new_path.clone();
                    record.health = LinkHealth::Relinked;
                    relink_map.insert(record.link_id.clone(), new_path);
                    relinked += 1;
                } else {
                    record.health = LinkHealth::Unresolvable;
                    unresolvable += 1;
                }
            }
        }

        RelinkReport {
            total_checked,
            already_valid,
            relinked,
            unresolvable,
            relink_map,
        }
    }

    /// Check all records and mark broken links (source file missing).
    /// This operates on in-memory path string checks only (no filesystem access).
    pub fn mark_broken_by_prefix(records: &mut [ProxyLinkRecord], missing_prefix: &str) {
        for record in records.iter_mut() {
            let source_str = record.source_path.to_string_lossy();
            if source_str.starts_with(missing_prefix) {
                record.health = LinkHealth::Broken;
            }
        }
    }

    /// Extract the filename from a path for matching purposes.
    pub fn filename(path: &Path) -> Option<String> {
        path.file_name().map(|n| n.to_string_lossy().to_string())
    }

    /// Build a lookup map from filename to link records for fuzzy relinking.
    pub fn build_filename_index(records: &[ProxyLinkRecord]) -> HashMap<String, Vec<usize>> {
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, record) in records.iter().enumerate() {
            if let Some(name) = Self::filename(&record.source_path) {
                index.entry(name).or_default().push(i);
            }
        }
        index
    }
}

impl Default for RelinkEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relink_rule_apply() {
        let rule = RelinkRule::new("/old/volume/", "/new/volume/");
        let result = rule.apply(Path::new("/old/volume/clips/a.mxf"));
        assert_eq!(result, Some(PathBuf::from("/new/volume/clips/a.mxf")));
    }

    #[test]
    fn test_relink_rule_no_match() {
        let rule = RelinkRule::new("/old/volume/", "/new/volume/");
        let result = rule.apply(Path::new("/other/path/a.mxf"));
        assert!(result.is_none());
    }

    #[test]
    fn test_relink_rule_with_extension() {
        let rule = RelinkRule::new("/old/", "/new/").with_extension(".mxf");
        let mxf = rule.apply(Path::new("/old/clip.mxf"));
        let mp4 = rule.apply(Path::new("/old/clip.mp4"));
        assert!(mxf.is_some());
        assert!(mp4.is_none());
    }

    #[test]
    fn test_link_record_new() {
        let rec = ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/src/a.mxf");
        assert_eq!(rec.link_id, "lk1");
        assert_eq!(rec.health, LinkHealth::Valid);
    }

    #[test]
    fn test_link_record_with_size() {
        let rec =
            ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/src/a.mxf").with_source_size(1_000_000);
        assert_eq!(rec.source_size_bytes, 1_000_000);
    }

    #[test]
    fn test_link_record_with_checksum() {
        let rec = ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/src/a.mxf").with_checksum("abc123");
        assert_eq!(rec.source_checksum, Some("abc123".to_string()));
    }

    #[test]
    fn test_engine_try_relink() {
        let mut engine = RelinkEngine::new();
        engine.add_rule(RelinkRule::new("/old/", "/new/"));
        let rec = ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/old/a.mxf");
        let result = engine.try_relink(&rec);
        assert_eq!(result, Some(PathBuf::from("/new/a.mxf")));
    }

    #[test]
    fn test_engine_try_relink_no_match() {
        let engine = RelinkEngine::new();
        let rec = ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/unknown/a.mxf");
        assert!(engine.try_relink(&rec).is_none());
    }

    #[test]
    fn test_relink_batch() {
        let mut engine = RelinkEngine::new();
        engine.add_rule(RelinkRule::new("/old/", "/new/"));

        let mut records = vec![
            ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/old/a.mxf"),
            ProxyLinkRecord::new("lk2", "/proxy/b.mp4", "/old/b.mxf"),
            ProxyLinkRecord::new("lk3", "/proxy/c.mp4", "/mystery/c.mxf"),
        ];
        // Mark all as broken first
        for r in &mut records {
            r.health = LinkHealth::Broken;
        }
        let report = engine.relink_batch(&mut records);
        assert_eq!(report.total_checked, 3);
        assert_eq!(report.relinked, 2);
        assert_eq!(report.unresolvable, 1);
    }

    #[test]
    fn test_relink_batch_already_valid() {
        let engine = RelinkEngine::new();
        let mut records = vec![ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/src/a.mxf")];
        let report = engine.relink_batch(&mut records);
        assert_eq!(report.already_valid, 1);
        assert_eq!(report.relinked, 0);
    }

    #[test]
    fn test_mark_broken_by_prefix() {
        let mut records = vec![
            ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/dead/a.mxf"),
            ProxyLinkRecord::new("lk2", "/proxy/b.mp4", "/alive/b.mxf"),
        ];
        RelinkEngine::mark_broken_by_prefix(&mut records, "/dead/");
        assert_eq!(records[0].health, LinkHealth::Broken);
        assert_eq!(records[1].health, LinkHealth::Valid);
    }

    #[test]
    fn test_filename_extraction() {
        assert_eq!(
            RelinkEngine::filename(Path::new("/a/b/clip.mxf")),
            Some("clip.mxf".to_string())
        );
    }

    #[test]
    fn test_build_filename_index() {
        let records = vec![
            ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/src/clip.mxf"),
            ProxyLinkRecord::new("lk2", "/proxy/b.mp4", "/other/clip.mxf"),
        ];
        let index = RelinkEngine::build_filename_index(&records);
        assert_eq!(
            index.get("clip.mxf").expect("should succeed in test").len(),
            2
        );
    }

    #[test]
    fn test_rule_count() {
        let mut engine = RelinkEngine::new();
        assert_eq!(engine.rule_count(), 0);
        engine.add_rule(RelinkRule::new("/a/", "/b/"));
        assert_eq!(engine.rule_count(), 1);
    }

    #[test]
    fn test_default_engine() {
        let engine = RelinkEngine::default();
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn test_relink_report_map() {
        let mut engine = RelinkEngine::new();
        engine.add_rule(RelinkRule::new("/old/", "/new/"));
        let mut records = vec![{
            let mut r = ProxyLinkRecord::new("lk1", "/proxy/a.mp4", "/old/a.mxf");
            r.health = LinkHealth::Broken;
            r
        }];
        let report = engine.relink_batch(&mut records);
        assert!(report.relink_map.contains_key("lk1"));
        assert_eq!(
            report
                .relink_map
                .get("lk1")
                .expect("should succeed in test"),
            &PathBuf::from("/new/a.mxf")
        );
    }
}
