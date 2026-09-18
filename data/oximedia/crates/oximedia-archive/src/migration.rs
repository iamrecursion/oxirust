//! Format migration planning for long-term digital preservation.
//!
//! Assesses file format risk, recommends migration paths, and estimates
//! the cost and time of migration batches.

#![allow(dead_code)]

/// NDSA / Library of Congress format-risk classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FormatRisk {
    /// Preferred: widely supported, openly specified, actively maintained.
    Preferred,
    /// Stable: well-documented, no known risks in the near term.
    Stable,
    /// At Risk: declining tool support or unclear long-term viability.
    AtRisk,
    /// Endangered: few remaining tools; migration recommended soon.
    Endangered,
    /// Obsolete: tools no longer maintained; migration urgent.
    Obsolete,
}

impl FormatRisk {
    /// Migration priority (higher = more urgent).
    #[must_use]
    pub const fn priority(&self) -> u8 {
        match self {
            Self::Preferred => 0,
            Self::Stable => 1,
            Self::AtRisk => 2,
            Self::Endangered => 3,
            Self::Obsolete => 4,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Preferred => "preferred",
            Self::Stable => "stable",
            Self::AtRisk => "at_risk",
            Self::Endangered => "endangered",
            Self::Obsolete => "obsolete",
        }
    }
}

/// A file format descriptor.
#[derive(Clone, Debug)]
pub struct FileFormat {
    /// Short name of the format (e.g., `"DPX"`).
    pub name: String,
    /// Canonical file extension without the leading dot (e.g., `"dpx"`).
    pub extension: String,
    /// MIME type (e.g., `"image/x-dpx"`).
    pub mime_type: String,
    /// Current risk status.
    pub risk: FormatRisk,
}

impl FileFormat {
    /// Create a new file format descriptor.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        extension: impl Into<String>,
        mime_type: impl Into<String>,
        risk: FormatRisk,
    ) -> Self {
        Self {
            name: name.into(),
            extension: extension.into(),
            mime_type: mime_type.into(),
            risk,
        }
    }

    // ── Common format presets ─────────────────────────────────────────────────

    /// SMPTE DPX (cinema standard, preferred for image sequences).
    #[must_use]
    pub fn dpx() -> Self {
        Self::new("DPX", "dpx", "image/x-dpx", FormatRisk::Preferred)
    }

    /// Apple ProRes 4444 (widely used, good tool support).
    #[must_use]
    pub fn prores_4444() -> Self {
        Self::new("ProRes 4444", "mov", "video/quicktime", FormatRisk::Stable)
    }

    /// Avid DNxHD (professional NLE format, good support).
    #[must_use]
    pub fn avid_dnxhd() -> Self {
        Self::new("Avid DNxHD", "mxf", "application/mxf", FormatRisk::Stable)
    }

    /// H.264 in MP4 container (delivery format; lossy).
    #[must_use]
    pub fn h264_mp4() -> Self {
        Self::new("H.264/MP4", "mp4", "video/mp4", FormatRisk::Stable)
    }

    /// MPEG-2 (older broadcast standard; declining support).
    #[must_use]
    pub fn mpeg2() -> Self {
        Self::new("MPEG-2", "mpg", "video/mpeg", FormatRisk::AtRisk)
    }

    /// DV (consumer/prosumer tape format; obsolescence risk).
    #[must_use]
    pub fn dv() -> Self {
        Self::new("DV", "dv", "video/x-dv", FormatRisk::Endangered)
    }

    /// Betacam SP (broadcast videotape; hardware-dependent).
    #[must_use]
    pub fn betacam() -> Self {
        Self::new(
            "Betacam SP",
            "betacam",
            "application/octet-stream",
            FormatRisk::Obsolete,
        )
    }
}

/// Describes a recommended migration from one format to another.
#[derive(Clone, Debug)]
pub struct MigrationPath {
    /// Source format.
    pub source: FileFormat,
    /// Recommended target format.
    pub target: FileFormat,
    /// Whether the migration involves quality loss.
    pub quality_loss: bool,
    /// Whether the original format can be recovered from the migrated file.
    pub reversible: bool,
    /// Human-readable notes about the migration.
    pub notes: String,
}

/// Plans format migrations based on format risk.
pub struct MigrationPlanner;

impl MigrationPlanner {
    /// Return a recommended migration path for `source`, or `None` if no
    /// migration is needed (format is `Preferred` or `Stable`).
    #[must_use]
    pub fn plan_migration(source: &FileFormat) -> Option<MigrationPath> {
        match source.risk {
            FormatRisk::Preferred | FormatRisk::Stable => None,
            FormatRisk::AtRisk => Some(MigrationPath {
                source: source.clone(),
                target: FileFormat::prores_4444(),
                quality_loss: false,
                reversible: false,
                notes: "Migrate to ProRes 4444 for better long-term tool support.".to_string(),
            }),
            FormatRisk::Endangered => Some(MigrationPath {
                source: source.clone(),
                target: FileFormat::dpx(),
                quality_loss: false,
                reversible: false,
                notes: "Urgent: migrate to DPX image sequence for lossless preservation."
                    .to_string(),
            }),
            FormatRisk::Obsolete => Some(MigrationPath {
                source: source.clone(),
                target: FileFormat::dpx(),
                quality_loss: false,
                reversible: false,
                notes: "Critical: format is obsolete — immediate migration to DPX required."
                    .to_string(),
            }),
        }
    }

    /// Estimate migration cost (CPU-hours per GB) for the given format.
    ///
    /// Returns a rough multiplier: higher for formats requiring complex decoding.
    #[must_use]
    pub fn estimate_cost_gb(format: &FileFormat, size_gb: f64) -> f64 {
        let cpu_hours_per_gb = match format.risk {
            FormatRisk::Preferred => 0.1,
            FormatRisk::Stable => 0.2,
            FormatRisk::AtRisk => 0.5,
            FormatRisk::Endangered => 1.5,
            FormatRisk::Obsolete => 4.0,
        };
        cpu_hours_per_gb * size_gb
    }
}

/// A planned batch of migrations.
#[derive(Clone, Debug)]
pub struct MigrationBatch {
    /// List of `(asset_id, migration_path)` pairs.
    pub items: Vec<(String, MigrationPath)>,
    /// Total size of all assets in GB.
    pub total_size_gb: f64,
    /// Estimated processing time in hours.
    pub estimated_hours: f64,
}

impl MigrationBatch {
    /// Create an empty migration batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            total_size_gb: 0.0,
            estimated_hours: 0.0,
        }
    }

    /// Add an item to the batch.
    pub fn add(&mut self, asset_id: impl Into<String>, path: MigrationPath, size_gb: f64) {
        let cost = MigrationPlanner::estimate_cost_gb(&path.source, size_gb);
        self.estimated_hours += cost;
        self.total_size_gb += size_gb;
        self.items.push((asset_id.into(), path));
    }

    /// Number of assets in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for MigrationBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_risk_priority_ordering() {
        assert!(FormatRisk::Obsolete.priority() > FormatRisk::Preferred.priority());
        assert!(FormatRisk::Endangered.priority() > FormatRisk::AtRisk.priority());
    }

    #[test]
    fn test_format_risk_labels_non_empty() {
        for risk in [
            FormatRisk::Preferred,
            FormatRisk::Stable,
            FormatRisk::AtRisk,
            FormatRisk::Endangered,
            FormatRisk::Obsolete,
        ] {
            assert!(!risk.label().is_empty());
        }
    }

    #[test]
    fn test_dpx_is_preferred() {
        assert_eq!(FileFormat::dpx().risk, FormatRisk::Preferred);
    }

    #[test]
    fn test_betacam_is_obsolete() {
        assert_eq!(FileFormat::betacam().risk, FormatRisk::Obsolete);
    }

    #[test]
    fn test_dv_is_endangered() {
        assert_eq!(FileFormat::dv().risk, FormatRisk::Endangered);
    }

    #[test]
    fn test_mpeg2_is_at_risk() {
        assert_eq!(FileFormat::mpeg2().risk, FormatRisk::AtRisk);
    }

    #[test]
    fn test_plan_migration_preferred_none() {
        let dpx = FileFormat::dpx();
        assert!(MigrationPlanner::plan_migration(&dpx).is_none());
    }

    #[test]
    fn test_plan_migration_at_risk_some() {
        let mpeg2 = FileFormat::mpeg2();
        let path = MigrationPlanner::plan_migration(&mpeg2);
        assert!(path.is_some());
        let path = path.expect("path should be valid");
        assert_eq!(path.target.risk, FormatRisk::Stable);
    }

    #[test]
    fn test_plan_migration_obsolete_to_dpx() {
        let betacam = FileFormat::betacam();
        let path = MigrationPlanner::plan_migration(&betacam).expect("path should be valid");
        assert_eq!(path.target.extension, "dpx");
    }

    #[test]
    fn test_estimate_cost_gb_preferred_cheap() {
        let dpx = FileFormat::dpx();
        let cost = MigrationPlanner::estimate_cost_gb(&dpx, 10.0);
        // 0.1 cpu-hours/GB * 10 GB = 1.0 — cheaper than at-risk formats
        assert!(cost <= 1.0, "cost = {cost}");
    }

    #[test]
    fn test_estimate_cost_gb_obsolete_expensive() {
        let betacam = FileFormat::betacam();
        let cost = MigrationPlanner::estimate_cost_gb(&betacam, 10.0);
        assert!(cost > 10.0, "cost = {cost}");
    }

    #[test]
    fn test_migration_batch_add() {
        let mut batch = MigrationBatch::new();
        let dv = FileFormat::dv();
        let path = MigrationPlanner::plan_migration(&dv).expect("path should be valid");
        batch.add("asset-001", path, 5.0);
        assert_eq!(batch.len(), 1);
        assert!((batch.total_size_gb - 5.0).abs() < 1e-9);
        assert!(batch.estimated_hours > 0.0);
    }

    #[test]
    fn test_migration_batch_is_empty() {
        let batch = MigrationBatch::new();
        assert!(batch.is_empty());
    }
}

// ── New migration-planning types ──────────────────────────────────────────────

/// Risk level of a migration operation.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MigrationRisk {
    /// Routine, well-understood migration.
    Low,
    /// Some complexity; proceed with review.
    Medium,
    /// Significant risk; expert sign-off required.
    High,
    /// Mission-critical; full risk management plan needed.
    Critical,
}

impl MigrationRisk {
    /// Numeric score (higher = more risky).
    #[must_use]
    pub const fn score(self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }
}

/// Degree to which a given format is supported by current tools.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatSupport {
    /// Actively maintained; no migration needed.
    FullySupported,
    /// Works but not recommended for new projects.
    LegacySupport,
    /// Will be removed in a future tool release.
    DeprecatedSoon,
    /// Tools no longer handle this format.
    Unsupported,
}

impl FormatSupport {
    /// Derive the migration risk implied by the support level.
    #[must_use]
    pub const fn migration_risk(self) -> MigrationRisk {
        match self {
            Self::FullySupported => MigrationRisk::Low,
            Self::LegacySupport => MigrationRisk::Medium,
            Self::DeprecatedSoon => MigrationRisk::High,
            Self::Unsupported => MigrationRisk::Critical,
        }
    }
}

/// Describes a single migration job (source → target format).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MigrationTask {
    /// Source format identifier.
    pub source_format: String,
    /// Target format identifier.
    pub target_format: String,
    /// Number of files to migrate.
    pub file_count: u32,
    /// Estimated time in hours.
    pub estimated_hours: f32,
}

impl MigrationTask {
    /// Returns `true` if the task covers more than 1 000 files.
    #[must_use]
    pub fn is_large(&self) -> bool {
        self.file_count > 1_000
    }
}

/// A collection of migration tasks forming a coherent plan.
#[allow(dead_code)]
#[derive(Default, Debug)]
pub struct MigrationPlan {
    /// Individual tasks.
    pub tasks: Vec<MigrationTask>,
    /// Free-text description.
    pub description: String,
}

impl MigrationPlan {
    /// Create an empty plan.
    #[must_use]
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            tasks: Vec::new(),
            description: description.into(),
        }
    }

    /// Append a task to the plan.
    pub fn add(&mut self, task: MigrationTask) {
        self.tasks.push(task);
    }

    /// Total number of files across all tasks.
    #[must_use]
    pub fn total_files(&self) -> u32 {
        self.tasks.iter().map(|t| t.file_count).sum()
    }

    /// Sum of estimated hours across all tasks.
    #[must_use]
    pub fn total_hours(&self) -> f32 {
        self.tasks.iter().map(|t| t.estimated_hours).sum()
    }

    /// Tasks where the source format has `High` or `Critical` support risk.
    ///
    /// Heuristic: more than 1 000 files OR estimated > 24 h counts as high-risk.
    #[must_use]
    pub fn high_risk_tasks(&self) -> Vec<&MigrationTask> {
        self.tasks
            .iter()
            .filter(|t| t.is_large() || t.estimated_hours > 24.0)
            .collect()
    }
}

#[cfg(test)]
mod migration_plan_tests {
    use super::*;

    fn small_task(src: &str, tgt: &str, n: u32, h: f32) -> MigrationTask {
        MigrationTask {
            source_format: src.to_string(),
            target_format: tgt.to_string(),
            file_count: n,
            estimated_hours: h,
        }
    }

    #[test]
    fn test_migration_risk_score_ordering() {
        assert!(MigrationRisk::Critical.score() > MigrationRisk::High.score());
        assert!(MigrationRisk::High.score() > MigrationRisk::Medium.score());
        assert!(MigrationRisk::Medium.score() > MigrationRisk::Low.score());
    }

    #[test]
    fn test_migration_risk_score_values() {
        assert_eq!(MigrationRisk::Low.score(), 1);
        assert_eq!(MigrationRisk::Critical.score(), 4);
    }

    #[test]
    fn test_format_support_fully_supported_low_risk() {
        assert_eq!(
            FormatSupport::FullySupported.migration_risk(),
            MigrationRisk::Low
        );
    }

    #[test]
    fn test_format_support_unsupported_critical_risk() {
        assert_eq!(
            FormatSupport::Unsupported.migration_risk(),
            MigrationRisk::Critical
        );
    }

    #[test]
    fn test_format_support_deprecated_high_risk() {
        assert_eq!(
            FormatSupport::DeprecatedSoon.migration_risk(),
            MigrationRisk::High
        );
    }

    #[test]
    fn test_migration_task_is_large_false() {
        let t = small_task("DV", "DPX", 500, 5.0);
        assert!(!t.is_large());
    }

    #[test]
    fn test_migration_task_is_large_true() {
        let t = small_task("DV", "DPX", 2_000, 40.0);
        assert!(t.is_large());
    }

    #[test]
    fn test_migration_plan_total_files() {
        let mut plan = MigrationPlan::new("Q1 migration");
        plan.add(small_task("A", "B", 100, 2.0));
        plan.add(small_task("C", "D", 200, 4.0));
        assert_eq!(plan.total_files(), 300);
    }

    #[test]
    fn test_migration_plan_total_hours() {
        let mut plan = MigrationPlan::new("test");
        plan.add(small_task("A", "B", 10, 1.5));
        plan.add(small_task("C", "D", 10, 2.5));
        assert!((plan.total_hours() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_migration_plan_high_risk_by_file_count() {
        let mut plan = MigrationPlan::new("big batch");
        plan.add(small_task("DV", "DPX", 5_000, 10.0));
        plan.add(small_task("MP4", "MOV", 50, 1.0));
        let hr = plan.high_risk_tasks();
        assert_eq!(hr.len(), 1);
        assert_eq!(hr[0].source_format, "DV");
    }

    #[test]
    fn test_migration_plan_high_risk_by_hours() {
        let mut plan = MigrationPlan::new("long job");
        plan.add(small_task("old", "new", 100, 30.0));
        assert_eq!(plan.high_risk_tasks().len(), 1);
    }

    #[test]
    fn test_migration_plan_no_high_risk() {
        let mut plan = MigrationPlan::new("easy");
        plan.add(small_task("A", "B", 50, 2.0));
        assert!(plan.high_risk_tasks().is_empty());
    }
}

// ── Dry-run and rollback support ──────────────────────────────────────────────

/// Outcome of a single dry-run migration step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DryRunOutcome {
    /// Migration would succeed.
    WouldSucceed,
    /// Migration would fail with the given reason.
    WouldFail(String),
    /// Migration was skipped (e.g. source format is already preferred).
    Skipped(String),
}

impl DryRunOutcome {
    /// Returns `true` if the outcome indicates success.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::WouldSucceed)
    }

    /// Returns `true` if the outcome indicates failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::WouldFail(_))
    }
}

impl std::fmt::Display for DryRunOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WouldSucceed => write!(f, "WOULD_SUCCEED"),
            Self::WouldFail(reason) => write!(f, "WOULD_FAIL: {reason}"),
            Self::Skipped(reason) => write!(f, "SKIPPED: {reason}"),
        }
    }
}

/// A single dry-run result entry.
#[derive(Clone, Debug)]
pub struct DryRunEntry {
    /// Asset identifier.
    pub asset_id: String,
    /// Source format.
    pub source_format: String,
    /// Target format.
    pub target_format: String,
    /// Predicted outcome.
    pub outcome: DryRunOutcome,
    /// Estimated duration in hours.
    pub estimated_hours: f64,
    /// Estimated output size in GB.
    pub estimated_output_size_gb: f64,
    /// Warnings (non-fatal issues).
    pub warnings: Vec<String>,
}

/// Report from a dry-run migration simulation.
#[derive(Clone, Debug)]
pub struct DryRunReport {
    /// Description of the migration plan.
    pub plan_description: String,
    /// Individual entries.
    pub entries: Vec<DryRunEntry>,
    /// Total estimated duration.
    pub total_estimated_hours: f64,
    /// Total estimated output size in GB.
    pub total_estimated_size_gb: f64,
}

impl DryRunReport {
    /// Number of entries that would succeed.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome.is_success())
            .count()
    }

    /// Number of entries that would fail.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome.is_failure())
            .count()
    }

    /// Number of entries that were skipped.
    #[must_use]
    pub fn skipped_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e.outcome, DryRunOutcome::Skipped(_)))
            .count()
    }

    /// Collect all warnings from all entries.
    #[must_use]
    pub fn all_warnings(&self) -> Vec<&str> {
        self.entries
            .iter()
            .flat_map(|e| e.warnings.iter().map(|w| w.as_str()))
            .collect()
    }

    /// Whether the overall dry-run passed (no failures).
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failure_count() == 0
    }
}

/// Asset descriptor for dry-run validation.
#[derive(Clone, Debug)]
pub struct DryRunAsset {
    /// Asset identifier.
    pub asset_id: String,
    /// Source format.
    pub source: FileFormat,
    /// Size in GB.
    pub size_gb: f64,
    /// Whether the file is readable.
    pub is_readable: bool,
    /// Whether sufficient disk space is available for the output.
    pub has_disk_space: bool,
}

/// Run a dry-run simulation of a migration plan.
///
/// This does not modify any files. It validates each asset against the
/// migration plan and predicts outcomes, sizes, and durations.
pub fn dry_run_migration(assets: &[DryRunAsset], plan_description: &str) -> DryRunReport {
    let mut entries = Vec::with_capacity(assets.len());
    let mut total_hours = 0.0;
    let mut total_size = 0.0;

    for asset in assets {
        let migration_path = MigrationPlanner::plan_migration(&asset.source);

        let (outcome, target_format, est_hours, est_size, warnings) = match migration_path {
            None => {
                let reason = format!(
                    "format '{}' is {} — no migration needed",
                    asset.source.name,
                    asset.source.risk.label()
                );
                (
                    DryRunOutcome::Skipped(reason),
                    asset.source.name.clone(),
                    0.0,
                    0.0,
                    Vec::new(),
                )
            }
            Some(ref path) => {
                let mut warns = Vec::new();

                // Check readability
                if !asset.is_readable {
                    (
                        DryRunOutcome::WouldFail(format!(
                            "source file for asset '{}' is not readable",
                            asset.asset_id
                        )),
                        path.target.name.clone(),
                        0.0,
                        0.0,
                        warns,
                    )
                } else if !asset.has_disk_space {
                    (
                        DryRunOutcome::WouldFail(format!(
                            "insufficient disk space for asset '{}'",
                            asset.asset_id
                        )),
                        path.target.name.clone(),
                        0.0,
                        0.0,
                        warns,
                    )
                } else {
                    let hours = MigrationPlanner::estimate_cost_gb(&asset.source, asset.size_gb);
                    // Estimate output size: lossless migrations are roughly 1:1,
                    // but format overhead can vary.
                    let output_size = asset.size_gb * if path.quality_loss { 0.3 } else { 1.05 };

                    if path.quality_loss {
                        warns.push("migration involves quality loss".to_string());
                    }
                    if asset.size_gb > 100.0 {
                        warns.push(format!(
                            "large asset ({:.1} GB) — migration may take {:.1} hours",
                            asset.size_gb, hours
                        ));
                    }
                    if asset.source.risk == FormatRisk::Obsolete {
                        warns.push(
                            "source format is OBSOLETE — verify migration output carefully"
                                .to_string(),
                        );
                    }

                    (
                        DryRunOutcome::WouldSucceed,
                        path.target.name.clone(),
                        hours,
                        output_size,
                        warns,
                    )
                }
            }
        };

        total_hours += est_hours;
        total_size += est_size;

        entries.push(DryRunEntry {
            asset_id: asset.asset_id.clone(),
            source_format: asset.source.name.clone(),
            target_format,
            outcome,
            estimated_hours: est_hours,
            estimated_output_size_gb: est_size,
            warnings,
        });
    }

    DryRunReport {
        plan_description: plan_description.to_string(),
        entries,
        total_estimated_hours: total_hours,
        total_estimated_size_gb: total_size,
    }
}

// ---------------------------------------------------------------------------
// Rollback tracking
// ---------------------------------------------------------------------------

/// Tracks the state of a migration for potential rollback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MigrationStepStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    InProgress,
    /// Completed successfully.
    Completed,
    /// Failed with the given error.
    Failed(String),
    /// Rolled back.
    RolledBack,
}

impl std::fmt::Display for MigrationStepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "PENDING"),
            Self::InProgress => write!(f, "IN_PROGRESS"),
            Self::Completed => write!(f, "COMPLETED"),
            Self::Failed(reason) => write!(f, "FAILED: {reason}"),
            Self::RolledBack => write!(f, "ROLLED_BACK"),
        }
    }
}

/// A single migration step with rollback information.
#[derive(Clone, Debug)]
pub struct MigrationStep {
    /// Step identifier.
    pub step_id: String,
    /// Asset identifier.
    pub asset_id: String,
    /// Source path.
    pub source_path: String,
    /// Backup path (where the original is preserved before migration).
    pub backup_path: Option<String>,
    /// Target path (where the migrated file will be written).
    pub target_path: String,
    /// Source format.
    pub source_format: String,
    /// Target format.
    pub target_format: String,
    /// Current status.
    pub status: MigrationStepStatus,
}

/// Journal that tracks migration steps for rollback support.
#[derive(Debug, Default)]
pub struct RollbackJournal {
    /// Steps in execution order.
    steps: Vec<MigrationStep>,
}

impl RollbackJournal {
    /// Create a new empty journal.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new step as pending.
    pub fn record_step(&mut self, step: MigrationStep) {
        self.steps.push(step);
    }

    /// Mark a step as in-progress.
    pub fn mark_in_progress(&mut self, step_id: &str) -> bool {
        if let Some(step) = self.steps.iter_mut().find(|s| s.step_id == step_id) {
            step.status = MigrationStepStatus::InProgress;
            true
        } else {
            false
        }
    }

    /// Mark a step as completed, recording the backup path.
    pub fn mark_completed(&mut self, step_id: &str, backup_path: Option<String>) -> bool {
        if let Some(step) = self.steps.iter_mut().find(|s| s.step_id == step_id) {
            step.status = MigrationStepStatus::Completed;
            step.backup_path = backup_path;
            true
        } else {
            false
        }
    }

    /// Mark a step as failed.
    pub fn mark_failed(&mut self, step_id: &str, reason: &str) -> bool {
        if let Some(step) = self.steps.iter_mut().find(|s| s.step_id == step_id) {
            step.status = MigrationStepStatus::Failed(reason.to_string());
            true
        } else {
            false
        }
    }

    /// Mark a step as rolled back.
    pub fn mark_rolled_back(&mut self, step_id: &str) -> bool {
        if let Some(step) = self.steps.iter_mut().find(|s| s.step_id == step_id) {
            step.status = MigrationStepStatus::RolledBack;
            true
        } else {
            false
        }
    }

    /// Get all completed steps (for potential rollback).
    #[must_use]
    pub fn completed_steps(&self) -> Vec<&MigrationStep> {
        self.steps
            .iter()
            .filter(|s| s.status == MigrationStepStatus::Completed)
            .collect()
    }

    /// Get all steps that need rollback (completed or in-progress).
    #[must_use]
    pub fn steps_needing_rollback(&self) -> Vec<&MigrationStep> {
        self.steps
            .iter()
            .filter(|s| {
                matches!(
                    s.status,
                    MigrationStepStatus::Completed | MigrationStepStatus::InProgress
                )
            })
            .rev() // Reverse order for rollback
            .collect()
    }

    /// Total number of steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Get a step by its ID.
    #[must_use]
    pub fn get_step(&self, step_id: &str) -> Option<&MigrationStep> {
        self.steps.iter().find(|s| s.step_id == step_id)
    }

    /// Summary of step statuses.
    #[must_use]
    pub fn summary(&self) -> RollbackSummary {
        let mut pending = 0;
        let mut in_progress = 0;
        let mut completed = 0;
        let mut failed = 0;
        let mut rolled_back = 0;

        for step in &self.steps {
            match &step.status {
                MigrationStepStatus::Pending => pending += 1,
                MigrationStepStatus::InProgress => in_progress += 1,
                MigrationStepStatus::Completed => completed += 1,
                MigrationStepStatus::Failed(_) => failed += 1,
                MigrationStepStatus::RolledBack => rolled_back += 1,
            }
        }

        RollbackSummary {
            total: self.steps.len(),
            pending,
            in_progress,
            completed,
            failed,
            rolled_back,
        }
    }
}

/// Summary counts of migration step statuses.
#[derive(Debug, Clone)]
pub struct RollbackSummary {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub failed: usize,
    pub rolled_back: usize,
}

impl RollbackSummary {
    /// Whether all steps completed successfully.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failed == 0 && self.pending == 0 && self.in_progress == 0
    }
}

#[cfg(test)]
mod dry_run_rollback_tests {
    use super::*;

    fn make_asset(
        id: &str,
        format: FileFormat,
        size_gb: f64,
        readable: bool,
        space: bool,
    ) -> DryRunAsset {
        DryRunAsset {
            asset_id: id.to_string(),
            source: format,
            size_gb,
            is_readable: readable,
            has_disk_space: space,
        }
    }

    // --- Dry-run tests ---

    #[test]
    fn test_dry_run_preferred_format_skipped() {
        let assets = vec![make_asset("a1", FileFormat::dpx(), 10.0, true, true)];
        let report = dry_run_migration(&assets, "test");
        assert_eq!(report.skipped_count(), 1);
        assert_eq!(report.success_count(), 0);
        assert!(report.passed());
    }

    #[test]
    fn test_dry_run_at_risk_succeeds() {
        let assets = vec![make_asset("a2", FileFormat::mpeg2(), 5.0, true, true)];
        let report = dry_run_migration(&assets, "migrate mpeg2");
        assert_eq!(report.success_count(), 1);
        assert!(report.total_estimated_hours > 0.0);
        assert!(report.total_estimated_size_gb > 0.0);
        assert!(report.passed());
    }

    #[test]
    fn test_dry_run_unreadable_fails() {
        let assets = vec![make_asset("a3", FileFormat::dv(), 1.0, false, true)];
        let report = dry_run_migration(&assets, "test");
        assert_eq!(report.failure_count(), 1);
        assert!(!report.passed());
    }

    #[test]
    fn test_dry_run_no_disk_space_fails() {
        let assets = vec![make_asset("a4", FileFormat::dv(), 1.0, true, false)];
        let report = dry_run_migration(&assets, "test");
        assert_eq!(report.failure_count(), 1);
        assert!(!report.passed());
    }

    #[test]
    fn test_dry_run_mixed_results() {
        let assets = vec![
            make_asset("ok", FileFormat::mpeg2(), 2.0, true, true),
            make_asset("skip", FileFormat::dpx(), 3.0, true, true),
            make_asset("fail", FileFormat::dv(), 1.0, false, true),
        ];
        let report = dry_run_migration(&assets, "mixed");
        assert_eq!(report.success_count(), 1);
        assert_eq!(report.skipped_count(), 1);
        assert_eq!(report.failure_count(), 1);
        assert!(!report.passed());
    }

    #[test]
    fn test_dry_run_large_asset_warning() {
        let assets = vec![make_asset("big", FileFormat::mpeg2(), 200.0, true, true)];
        let report = dry_run_migration(&assets, "big batch");
        let warnings = report.all_warnings();
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.contains("large asset")));
    }

    #[test]
    fn test_dry_run_obsolete_format_warning() {
        let assets = vec![make_asset("old", FileFormat::betacam(), 10.0, true, true)];
        let report = dry_run_migration(&assets, "obsolete");
        let warnings = report.all_warnings();
        assert!(warnings.iter().any(|w| w.contains("OBSOLETE")));
    }

    #[test]
    fn test_dry_run_empty_assets() {
        let report = dry_run_migration(&[], "empty");
        assert!(report.passed());
        assert_eq!(report.entries.len(), 0);
        assert!((report.total_estimated_hours).abs() < 1e-10);
    }

    #[test]
    fn test_dry_run_outcome_display() {
        assert_eq!(DryRunOutcome::WouldSucceed.to_string(), "WOULD_SUCCEED");
        assert!(DryRunOutcome::WouldFail("reason".into())
            .to_string()
            .contains("reason"));
        assert!(DryRunOutcome::Skipped("ok".into())
            .to_string()
            .contains("ok"));
    }

    // --- Rollback journal tests ---

    fn make_step(id: &str, asset: &str) -> MigrationStep {
        MigrationStep {
            step_id: id.to_string(),
            asset_id: asset.to_string(),
            source_path: format!("/source/{asset}.dv"),
            backup_path: None,
            target_path: format!("/target/{asset}.dpx"),
            source_format: "DV".to_string(),
            target_format: "DPX".to_string(),
            status: MigrationStepStatus::Pending,
        }
    }

    #[test]
    fn test_rollback_journal_record_and_query() {
        let mut journal = RollbackJournal::new();
        journal.record_step(make_step("s1", "video_a"));
        journal.record_step(make_step("s2", "video_b"));
        assert_eq!(journal.step_count(), 2);
    }

    #[test]
    fn test_rollback_journal_lifecycle() {
        let mut journal = RollbackJournal::new();
        journal.record_step(make_step("s1", "video_a"));

        assert!(journal.mark_in_progress("s1"));
        let step = journal.get_step("s1");
        assert_eq!(
            step.map(|s| &s.status),
            Some(&MigrationStepStatus::InProgress)
        );

        assert!(journal.mark_completed("s1", Some("/backup/video_a.dv".to_string())));
        let step = journal.get_step("s1");
        assert_eq!(
            step.map(|s| &s.status),
            Some(&MigrationStepStatus::Completed)
        );
        assert_eq!(
            step.and_then(|s| s.backup_path.as_deref()),
            Some("/backup/video_a.dv")
        );
    }

    #[test]
    fn test_rollback_journal_failed_step() {
        let mut journal = RollbackJournal::new();
        journal.record_step(make_step("s1", "video_a"));
        assert!(journal.mark_in_progress("s1"));
        assert!(journal.mark_failed("s1", "disk full"));

        let step = journal.get_step("s1");
        assert!(matches!(
            step.map(|s| &s.status),
            Some(MigrationStepStatus::Failed(_))
        ));
    }

    #[test]
    fn test_rollback_journal_steps_needing_rollback() {
        let mut journal = RollbackJournal::new();
        journal.record_step(make_step("s1", "a"));
        journal.record_step(make_step("s2", "b"));
        journal.record_step(make_step("s3", "c"));

        journal.mark_in_progress("s1");
        journal.mark_completed("s1", None);
        journal.mark_in_progress("s2");
        journal.mark_completed("s2", None);
        journal.mark_in_progress("s3");
        // s3 is still in-progress (crash scenario)

        let needing = journal.steps_needing_rollback();
        assert_eq!(needing.len(), 3);
        // Should be in reverse order
        assert_eq!(needing[0].step_id, "s3");
        assert_eq!(needing[1].step_id, "s2");
        assert_eq!(needing[2].step_id, "s1");
    }

    #[test]
    fn test_rollback_journal_mark_rolled_back() {
        let mut journal = RollbackJournal::new();
        journal.record_step(make_step("s1", "a"));
        journal.mark_in_progress("s1");
        journal.mark_completed("s1", None);
        assert!(journal.mark_rolled_back("s1"));

        let step = journal.get_step("s1");
        assert_eq!(
            step.map(|s| &s.status),
            Some(&MigrationStepStatus::RolledBack)
        );
    }

    #[test]
    fn test_rollback_journal_summary() {
        let mut journal = RollbackJournal::new();
        journal.record_step(make_step("s1", "a"));
        journal.record_step(make_step("s2", "b"));
        journal.record_step(make_step("s3", "c"));
        journal.record_step(make_step("s4", "d"));

        journal.mark_in_progress("s1");
        journal.mark_completed("s1", None);
        journal.mark_in_progress("s2");
        journal.mark_failed("s2", "error");
        journal.mark_in_progress("s3");
        journal.mark_completed("s3", None);
        journal.mark_rolled_back("s3");
        // s4 stays pending

        let summary = journal.summary();
        assert_eq!(summary.total, 4);
        assert_eq!(summary.completed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.rolled_back, 1);
        assert_eq!(summary.pending, 1);
        assert!(!summary.all_succeeded());
    }

    #[test]
    fn test_rollback_summary_all_succeeded() {
        let mut journal = RollbackJournal::new();
        journal.record_step(make_step("s1", "a"));
        journal.mark_in_progress("s1");
        journal.mark_completed("s1", None);

        let summary = journal.summary();
        assert!(summary.all_succeeded());
    }

    #[test]
    fn test_rollback_journal_missing_step_id() {
        let mut journal = RollbackJournal::new();
        assert!(!journal.mark_in_progress("nonexistent"));
        assert!(!journal.mark_completed("nonexistent", None));
        assert!(!journal.mark_failed("nonexistent", "err"));
        assert!(!journal.mark_rolled_back("nonexistent"));
    }

    #[test]
    fn test_migration_step_status_display() {
        assert_eq!(MigrationStepStatus::Pending.to_string(), "PENDING");
        assert_eq!(MigrationStepStatus::InProgress.to_string(), "IN_PROGRESS");
        assert_eq!(MigrationStepStatus::Completed.to_string(), "COMPLETED");
        assert_eq!(MigrationStepStatus::RolledBack.to_string(), "ROLLED_BACK");
        assert!(MigrationStepStatus::Failed("err".into())
            .to_string()
            .contains("err"));
    }
}
