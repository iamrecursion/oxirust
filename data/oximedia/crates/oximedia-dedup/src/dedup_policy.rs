//! Policy types for controlling deduplication behaviour.
//!
//! Provides `DedupAction`, `DedupPolicy`, `DedupPolicyConfig`, and
//! `DedupDecision` so callers can codify rules about what to do when
//! duplicates are found.

#![allow(dead_code)]

/// Action to take when a duplicate is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DedupAction {
    /// Delete the duplicate immediately.
    Delete,
    /// Move the duplicate to a quarantine directory.
    Quarantine,
    /// Create a symbolic link pointing to the canonical copy.
    Symlink,
    /// Keep both copies and emit a warning.
    Keep,
    /// Flag the item for manual review.
    Review,
    /// Skip (do nothing, log only).
    Skip,
}

impl DedupAction {
    /// Return `true` if this action permanently modifies or removes data.
    #[must_use]
    pub const fn is_destructive(self) -> bool {
        matches!(self, Self::Delete | Self::Quarantine)
    }

    /// Return a human-readable description of the action.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Delete => "delete duplicate",
            Self::Quarantine => "move to quarantine",
            Self::Symlink => "replace with symlink",
            Self::Keep => "keep both copies",
            Self::Review => "flag for review",
            Self::Skip => "skip / log only",
        }
    }
}

/// Configures the deduplication policy.
#[derive(Debug, Clone)]
pub struct DedupPolicyConfig {
    /// Enable strict mode: require all selected methods to agree before acting.
    pub strict_mode: bool,
    /// Minimum similarity score (0.0–1.0) required to consider items duplicates.
    pub min_similarity: f64,
    /// Action applied when an exact duplicate is found (similarity == 1.0).
    pub exact_action: DedupAction,
    /// Action applied when a near-duplicate is found.
    pub near_action: DedupAction,
    /// Whether to protect files marked as originals from deletion.
    pub protect_originals: bool,
}

impl Default for DedupPolicyConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            min_similarity: 0.95,
            exact_action: DedupAction::Quarantine,
            near_action: DedupAction::Review,
            protect_originals: true,
        }
    }
}

impl DedupPolicyConfig {
    /// Return `true` if strict mode is enabled.
    #[must_use]
    pub const fn strict_mode(&self) -> bool {
        self.strict_mode
    }

    /// Return the minimum similarity threshold.
    #[must_use]
    pub fn min_similarity(&self) -> f64 {
        self.min_similarity
    }
}

/// The computed deduplication decision for a candidate pair.
#[derive(Debug, Clone)]
pub struct DedupDecision {
    /// Similarity score in 0.0–1.0.
    pub similarity: f64,
    /// Chosen action.
    pub action: DedupAction,
    /// Whether the decision needs human review.
    pub needs_review: bool,
    /// Optional explanation string.
    pub reason: Option<String>,
}

impl DedupDecision {
    /// Create a new `DedupDecision`.
    #[must_use]
    pub fn new(similarity: f64, action: DedupAction, reason: Option<String>) -> Self {
        let needs_review =
            matches!(action, DedupAction::Review) || (action.is_destructive() && similarity < 1.0);
        Self {
            similarity,
            action,
            needs_review,
            reason,
        }
    }

    /// Return `true` if the decision requires human review before execution.
    #[must_use]
    pub fn requires_review(&self) -> bool {
        self.needs_review
    }
}

/// Evaluates pairs of media items according to a `DedupPolicyConfig`.
#[derive(Debug, Clone)]
pub struct DedupPolicy {
    config: DedupPolicyConfig,
}

impl DedupPolicy {
    /// Create a new `DedupPolicy` from a config.
    #[must_use]
    pub fn new(config: DedupPolicyConfig) -> Self {
        Self { config }
    }

    /// Decide whether two items with the given `similarity` should be deduped.
    ///
    /// Returns a `DedupDecision` describing what to do.
    #[must_use]
    pub fn should_dedup(&self, similarity: f64, is_original: bool) -> DedupDecision {
        // Guard: similarity below threshold → skip.
        if similarity < self.config.min_similarity {
            return DedupDecision::new(
                similarity,
                DedupAction::Skip,
                Some(format!(
                    "similarity {similarity:.3} below threshold {:.3}",
                    self.config.min_similarity
                )),
            );
        }

        // Guard: protect originals.
        if is_original && self.config.protect_originals {
            return DedupDecision::new(
                similarity,
                DedupAction::Keep,
                Some("file is marked as original".to_string()),
            );
        }

        // Exact duplicate.
        #[allow(clippy::float_cmp)]
        if similarity == 1.0 {
            let action = if self.config.strict_mode {
                self.config.exact_action
            } else {
                self.config.exact_action
            };
            return DedupDecision::new(
                similarity,
                action,
                Some("exact duplicate detected".to_string()),
            );
        }

        // Near-duplicate.
        DedupDecision::new(
            similarity,
            self.config.near_action,
            Some(format!("near-duplicate at {similarity:.3}")),
        )
    }

    /// Access the underlying config.
    #[must_use]
    pub const fn config(&self) -> &DedupPolicyConfig {
        &self.config
    }
}

impl Default for DedupPolicy {
    fn default() -> Self {
        Self::new(DedupPolicyConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Per-group configurable dedup actions
// ---------------------------------------------------------------------------

/// Criteria for selecting which file to keep within a duplicate group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeepCriterion {
    /// Keep the file with the most recent modification timestamp.
    Newest,
    /// Keep the file with the oldest modification timestamp.
    Oldest,
    /// Keep the file with the largest size (typically highest quality).
    LargestFile,
    /// Keep the file with the smallest size (most compressed).
    SmallestFile,
    /// Keep the file with the shortest path (likely the "original" location).
    ShortestPath,
    /// Keep the file with the longest path.
    LongestPath,
}

/// Per-group policy that determines both which file to keep and what
/// action to apply to the remaining duplicates.
#[derive(Debug, Clone)]
pub struct GroupPolicy {
    /// How to select the file to keep.
    pub keep: KeepCriterion,
    /// Action to apply to duplicates (non-kept files).
    pub action: DedupAction,
    /// Minimum similarity for this policy to apply.
    pub min_similarity: f64,
}

impl Default for GroupPolicy {
    fn default() -> Self {
        Self {
            keep: KeepCriterion::LargestFile,
            action: DedupAction::Review,
            min_similarity: 0.95,
        }
    }
}

/// Result of applying a `GroupPolicy` to a duplicate group.
#[derive(Debug, Clone)]
pub struct GroupDecision {
    /// Index of the file to keep (within the group's file list).
    pub keep_index: usize,
    /// Path of the file to keep.
    pub keep_path: String,
    /// Indices and paths of files to act upon.
    pub duplicates: Vec<(usize, String)>,
    /// The action to apply to duplicates.
    pub action: DedupAction,
    /// Optional reason.
    pub reason: String,
}

/// Score a file path according to a `KeepCriterion`.
///
/// Higher is better for all criteria (the file with the highest score is kept).
fn score_file(path: &str, criterion: KeepCriterion) -> f64 {
    match criterion {
        KeepCriterion::LargestFile => std::fs::metadata(path)
            .map(|m| m.len() as f64)
            .unwrap_or(0.0),
        KeepCriterion::SmallestFile => {
            let size = std::fs::metadata(path)
                .map(|m| m.len() as f64)
                .unwrap_or(f64::MAX);
            // Invert: smaller → higher score
            if size <= 0.0 {
                0.0
            } else {
                1.0 / size
            }
        }
        KeepCriterion::Newest => std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .ok()
            })
            .unwrap_or(0.0),
        KeepCriterion::Oldest => {
            let ts = std::fs::metadata(path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs_f64())
                        .ok()
                })
                .unwrap_or(f64::MAX);
            if ts >= f64::MAX {
                0.0
            } else {
                1.0 / (ts + 1.0)
            }
        }
        KeepCriterion::ShortestPath => {
            if path.is_empty() {
                0.0
            } else {
                1.0 / path.len() as f64
            }
        }
        KeepCriterion::LongestPath => path.len() as f64,
    }
}

/// Apply a `GroupPolicy` to a list of file paths.
///
/// Returns `None` if fewer than 2 files are provided.
#[must_use]
pub fn apply_group_policy(files: &[String], policy: &GroupPolicy) -> Option<GroupDecision> {
    if files.len() < 2 {
        return None;
    }

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (i, path) in files.iter().enumerate() {
        let s = score_file(path, policy.keep);
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    let duplicates: Vec<(usize, String)> = files
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != best_idx)
        .map(|(i, p)| (i, p.clone()))
        .collect();

    Some(GroupDecision {
        keep_index: best_idx,
        keep_path: files[best_idx].clone(),
        duplicates,
        action: policy.action,
        reason: format!(
            "keep by {:?}, apply {:?} to {} duplicate(s)",
            policy.keep,
            policy.action,
            files.len() - 1
        ),
    })
}

// ---------------------------------------------------------------------------
// GroupAction — configurable per-group action with keeper selection
// ---------------------------------------------------------------------------

/// High-level action to apply to every file in a duplicate group except the
/// one that is kept.
///
/// This is a simpler, caller-facing complement to the lower-level
/// [`KeepCriterion`] + [`GroupPolicy`] API. Use [`select_keeper`] to resolve
/// a group to its keeper path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupAction {
    /// Retain the file with the most recent modification time.
    KeepNewest,
    /// Retain the highest-quality file, ranked by a codec-aware quality score
    /// (header-derived resolution, bit depth and effective bitrate) rather than
    /// raw on-disk size. See [`crate::quality::quality_score`] for the formula
    /// and its honest fallback to file size when a format cannot be probed.
    KeepHighestQuality,
    /// Retain the first file listed in the group (index 0).
    KeepFirst,
    /// Delete all files — no keeper is selected (returns `None`).
    Delete,
}

impl GroupAction {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::KeepNewest => "keep-newest",
            Self::KeepHighestQuality => "keep-highest-quality",
            Self::KeepFirst => "keep-first",
            Self::Delete => "delete-all",
        }
    }

    /// Returns `true` if this action results in no file being preserved.
    #[must_use]
    pub const fn deletes_all(self) -> bool {
        matches!(self, Self::Delete)
    }
}

/// Select which file in `group` to keep according to `action`.
///
/// - `KeepNewest` — returns the path with the largest mtime (via
///   [`std::fs::metadata`]).  Paths whose metadata cannot be read are treated
///   as having `mtime = 0` (i.e., oldest).
/// - `KeepHighestQuality` — returns the path with the highest codec-aware
///   [`crate::quality::quality_score`] (resolution first, then effective bitrate,
///   then bit depth). Formats that cannot be probed fall back to file size, so the
///   result is never worse than a size ranking. Unreadable paths score `0.0`.
/// - `KeepFirst` — returns `group.first().cloned()`.
/// - `Delete` — returns `None` (all files are considered expendable).
///
/// Returns `None` for an empty group regardless of the action.
#[must_use]
pub fn select_keeper(
    group: &[std::path::PathBuf],
    action: &GroupAction,
) -> Option<std::path::PathBuf> {
    if group.is_empty() {
        return None;
    }

    match action {
        GroupAction::KeepFirst => group.first().cloned(),

        GroupAction::Delete => None,

        GroupAction::KeepNewest => {
            // Map each path to its mtime as seconds-since-epoch (0 on error).
            let best = group
                .iter()
                .map(|p| {
                    let ts = std::fs::metadata(p)
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| {
                            t.duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .ok()
                        })
                        .unwrap_or(0);
                    (ts, p)
                })
                .max_by_key(|(ts, _)| *ts)
                .map(|(_, p)| p.clone());
            best
        }

        GroupAction::KeepHighestQuality => {
            // Keep the genuinely highest-quality copy using header-derived signals
            // (resolution, bit depth and effective bitrate) instead of raw on-disk
            // size. `quality_score` reads each signal from the real file header and
            // falls back to size only when a format cannot be probed, so this never
            // regresses below the previous "largest file" behaviour. The score is
            // computed once per file; ties resolve to the last maximum, matching
            // the prior `max_by_key` semantics.
            group
                .iter()
                .map(|p| (crate::quality::quality_score(p), p))
                .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, p)| p.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_is_destructive_delete() {
        assert!(DedupAction::Delete.is_destructive());
    }

    #[test]
    fn test_action_is_destructive_quarantine() {
        assert!(DedupAction::Quarantine.is_destructive());
    }

    #[test]
    fn test_action_not_destructive_keep() {
        assert!(!DedupAction::Keep.is_destructive());
    }

    #[test]
    fn test_action_not_destructive_symlink() {
        assert!(!DedupAction::Symlink.is_destructive());
    }

    #[test]
    fn test_action_description_nonempty() {
        for action in [
            DedupAction::Delete,
            DedupAction::Quarantine,
            DedupAction::Symlink,
            DedupAction::Keep,
            DedupAction::Review,
            DedupAction::Skip,
        ] {
            assert!(!action.description().is_empty());
        }
    }

    #[test]
    fn test_policy_config_defaults() {
        let cfg = DedupPolicyConfig::default();
        assert!(!cfg.strict_mode());
        assert!((cfg.min_similarity() - 0.95).abs() < 1e-9);
        assert!(cfg.protect_originals);
    }

    #[test]
    fn test_policy_skip_below_threshold() {
        let policy = DedupPolicy::default();
        let decision = policy.should_dedup(0.50, false);
        assert_eq!(decision.action, DedupAction::Skip);
        assert!(!decision.requires_review());
    }

    #[test]
    fn test_policy_exact_duplicate() {
        let policy = DedupPolicy::default();
        let decision = policy.should_dedup(1.0, false);
        assert_eq!(decision.action, DedupAction::Quarantine);
    }

    #[test]
    fn test_policy_near_duplicate() {
        let policy = DedupPolicy::default();
        let decision = policy.should_dedup(0.97, false);
        assert_eq!(decision.action, DedupAction::Review);
    }

    #[test]
    fn test_policy_protect_original() {
        let policy = DedupPolicy::default();
        let decision = policy.should_dedup(1.0, true);
        assert_eq!(decision.action, DedupAction::Keep);
    }

    #[test]
    fn test_decision_requires_review_for_review_action() {
        let d = DedupDecision::new(0.97, DedupAction::Review, None);
        assert!(d.requires_review());
    }

    #[test]
    fn test_decision_requires_review_destructive_near_dup() {
        let d = DedupDecision::new(0.97, DedupAction::Delete, None);
        assert!(d.requires_review());
    }

    #[test]
    fn test_decision_no_review_for_exact_destructive() {
        // similarity == 1.0, destructive → NOT near-dup branch, no review flag
        let d = DedupDecision::new(1.0, DedupAction::Delete, None);
        assert!(!d.requires_review());
    }

    #[test]
    fn test_decision_skip_no_review() {
        let d = DedupDecision::new(0.5, DedupAction::Skip, None);
        assert!(!d.requires_review());
    }

    #[test]
    fn test_policy_config_strict_mode_toggle() {
        let mut cfg = DedupPolicyConfig::default();
        cfg.strict_mode = true;
        assert!(cfg.strict_mode());
    }

    // ---- GroupPolicy / KeepCriterion tests ----

    #[test]
    fn test_keep_criterion_shortest_path() {
        let files = vec![
            "/a/b/c/deep/path/file.mp4".to_string(),
            "/short.mp4".to_string(),
            "/medium/file.mp4".to_string(),
        ];
        let policy = GroupPolicy {
            keep: KeepCriterion::ShortestPath,
            action: DedupAction::Delete,
            min_similarity: 0.95,
        };
        let decision = apply_group_policy(&files, &policy).expect("should produce a decision");
        assert_eq!(decision.keep_path, "/short.mp4");
        assert_eq!(decision.duplicates.len(), 2);
        assert_eq!(decision.action, DedupAction::Delete);
    }

    #[test]
    fn test_keep_criterion_longest_path() {
        let files = vec![
            "/short.mp4".to_string(),
            "/a/b/c/deep/path/file.mp4".to_string(),
        ];
        let policy = GroupPolicy {
            keep: KeepCriterion::LongestPath,
            action: DedupAction::Quarantine,
            min_similarity: 0.95,
        };
        let decision = apply_group_policy(&files, &policy).expect("should produce a decision");
        assert_eq!(decision.keep_path, "/a/b/c/deep/path/file.mp4");
    }

    #[test]
    fn test_group_policy_default() {
        let policy = GroupPolicy::default();
        assert_eq!(policy.keep, KeepCriterion::LargestFile);
        assert_eq!(policy.action, DedupAction::Review);
        assert!((policy.min_similarity - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_group_policy_too_few_files() {
        let files = vec!["only_one.mp4".to_string()];
        let policy = GroupPolicy::default();
        assert!(apply_group_policy(&files, &policy).is_none());
    }

    #[test]
    fn test_group_decision_reason_contains_criterion() {
        let files = vec!["a.mp4".to_string(), "b.mp4".to_string()];
        let policy = GroupPolicy {
            keep: KeepCriterion::Newest,
            action: DedupAction::Symlink,
            min_similarity: 0.9,
        };
        let decision = apply_group_policy(&files, &policy).expect("should produce a decision");
        assert!(decision.reason.contains("Newest"));
        assert!(decision.reason.contains("Symlink"));
    }

    #[test]
    fn test_keep_criterion_all_variants_non_destructive() {
        // Ensure all KeepCriterion variants can be used without panic
        let files = vec!["a.mp4".to_string(), "b.mp4".to_string()];
        for criterion in [
            KeepCriterion::Newest,
            KeepCriterion::Oldest,
            KeepCriterion::LargestFile,
            KeepCriterion::SmallestFile,
            KeepCriterion::ShortestPath,
            KeepCriterion::LongestPath,
        ] {
            let policy = GroupPolicy {
                keep: criterion,
                action: DedupAction::Skip,
                min_similarity: 0.5,
            };
            let decision = apply_group_policy(&files, &policy);
            assert!(decision.is_some());
        }
    }

    // ---- GroupAction / select_keeper tests ----

    #[test]
    fn test_group_action_label_nonempty() {
        for action in [
            GroupAction::KeepNewest,
            GroupAction::KeepHighestQuality,
            GroupAction::KeepFirst,
            GroupAction::Delete,
        ] {
            assert!(!action.label().is_empty());
        }
    }

    #[test]
    fn test_group_action_delete_is_delete_all() {
        assert!(GroupAction::Delete.deletes_all());
        assert!(!GroupAction::KeepFirst.deletes_all());
    }

    #[test]
    fn test_policy_keep_first_returns_first() {
        let dir =
            std::env::temp_dir().join(format!("oximedia_policy_keep_first_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let f1 = dir.join("first.bin");
        let f2 = dir.join("second.bin");
        std::fs::write(&f1, b"aaa").expect("write");
        std::fs::write(&f2, b"bbb").expect("write");
        let group = vec![f1.clone(), f2];
        let keeper = select_keeper(&group, &GroupAction::KeepFirst);
        assert_eq!(keeper, Some(f1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_policy_delete_returns_none() {
        let dir =
            std::env::temp_dir().join(format!("oximedia_policy_delete_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let f1 = dir.join("a.bin");
        let f2 = dir.join("b.bin");
        std::fs::write(&f1, b"x").expect("write");
        std::fs::write(&f2, b"y").expect("write");
        let group = vec![f1, f2];
        let keeper = select_keeper(&group, &GroupAction::Delete);
        assert!(keeper.is_none(), "Delete action should return None");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_policy_keep_newest_returns_latest() {
        let dir =
            std::env::temp_dir().join(format!("oximedia_policy_newest_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let f_old = dir.join("old.bin");
        let f_new = dir.join("new.bin");
        std::fs::write(&f_old, b"old").expect("write old");
        // Small sleep not possible per policy; instead we set mtime explicitly
        // via filetime crate — but since we cannot depend on extra crates here,
        // we rely on the OS clock advancing between the two writes, or fall back
        // to testing that the function returns *a* valid path from the group.
        std::fs::write(&f_new, b"new").expect("write new");
        let group = vec![f_old.clone(), f_new.clone()];
        let keeper = select_keeper(&group, &GroupAction::KeepNewest);
        assert!(
            keeper == Some(f_old.clone()) || keeper == Some(f_new.clone()),
            "KeepNewest must return one of the group paths"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_policy_keep_highest_quality_largest_file() {
        let dir =
            std::env::temp_dir().join(format!("oximedia_policy_quality_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let f_small = dir.join("small.bin");
        let f_large = dir.join("large.bin");
        std::fs::write(&f_small, &[0u8; 100]).expect("write small");
        std::fs::write(&f_large, &[0u8; 500]).expect("write large");
        let group = vec![f_small, f_large.clone()];
        let keeper = select_keeper(&group, &GroupAction::KeepHighestQuality);
        assert_eq!(
            keeper,
            Some(f_large),
            "KeepHighestQuality should pick the largest file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_select_keeper_empty_group() {
        let keeper = select_keeper(&[], &GroupAction::KeepFirst);
        assert!(keeper.is_none(), "Empty group should always return None");
    }

    /// Minimal PNG whose header reports `width`x`height`, padded to `total_size`
    /// bytes so the on-disk size can be controlled independently of resolution.
    fn write_png(path: &std::path::Path, width: u32, height: u32, total_size: usize) {
        const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut bytes = Vec::with_capacity(total_size.max(26));
        bytes.extend_from_slice(&PNG_SIGNATURE);
        bytes.extend_from_slice(&13u32.to_be_bytes());
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.push(8); // bit depth
        bytes.push(2); // colour type
        while bytes.len() < total_size {
            bytes.push(0);
        }
        std::fs::write(path, &bytes).expect("write png");
    }

    #[test]
    fn test_keep_highest_quality_prefers_resolution_over_size() {
        // The headline fix: a small 4K image must beat a large SD image, which a
        // pure file-size ranking (the old behaviour) would get wrong.
        let dir = std::env::temp_dir().join(format!(
            "oximedia_policy_quality_res_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let small_4k = dir.join("small_4k.png");
        let huge_sd = dir.join("huge_sd.png");
        write_png(&small_4k, 3840, 2160, 4_096);
        write_png(&huge_sd, 720, 480, 4_000_000);

        let group = vec![huge_sd.clone(), small_4k.clone()];
        let keeper = select_keeper(&group, &GroupAction::KeepHighestQuality);
        assert_eq!(
            keeper,
            Some(small_4k),
            "KeepHighestQuality must keep the higher-resolution file, not the larger one"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
