//! Schema migration detection and planning for SymbolTable evolution.
//!
//! This module compares two [`SymbolTable`] versions, detects structural changes,
//! and produces a structured [`SchemaMigrationPlan`] with actionable steps.
//!
//! # Overview
//!
//! The migration engine:
//! 1. Snapshots both old and new schemas via [`SchemaSnapshot`]
//! 2. Diffs predicates, domains, and variable bindings
//! 3. Optionally detects renames using Dice-bigram similarity
//! 4. Classifies each [`SchemaChange`] by [`ChangeSeverity`]
//! 5. Generates ordered [`SchemaMigrationStep`]s
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{SymbolTable, DomainInfo, PredicateInfo};
//! use tensorlogic_adapters::schema_migration::{
//!     compute_migration, MigrationConfig, SchemaChange,
//! };
//!
//! let mut old = SymbolTable::new();
//! old.add_domain(DomainInfo::new("Person", 100)).unwrap();
//! let mut new_schema = SymbolTable::new();
//! new_schema.add_domain(DomainInfo::new("Person", 100)).unwrap();
//! new_schema.add_domain(DomainInfo::new("Animal", 50)).unwrap();
//!
//! let config = MigrationConfig::default();
//! let plan = compute_migration(&old, &new_schema, &config).unwrap();
//! assert!(!plan.is_empty());
//! ```

use std::collections::{HashMap, HashSet};

use crate::SymbolTable;

// ─────────────────────────────────────────────────────────────────────────────
// SchemaChange
// ─────────────────────────────────────────────────────────────────────────────

/// A single structural change detected between two schema versions.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaChange {
    /// A predicate was added in the new schema.
    PredicateAdded { name: String, arity: usize },
    /// A predicate was removed from the old schema.
    PredicateRemoved { name: String, arity: usize },
    /// A predicate exists in both versions but with a different arity.
    PredicateArityChanged {
        name: String,
        old_arity: usize,
        new_arity: usize,
    },
    /// A domain was added in the new schema.
    DomainAdded { name: String },
    /// A domain was removed from the old schema.
    DomainRemoved { name: String },
    /// A variable binding (rule entry) was added in the new schema.
    RuleAdded { name: String },
    /// A variable binding (rule entry) was removed from the old schema.
    RuleRemoved { name: String },
    /// A predicate was renamed (same arity, high name similarity).
    PredicateRenamed { old_name: String, new_name: String },
}

impl SchemaChange {
    /// Returns `true` when this change is considered breaking:
    /// removals and arity changes break existing consumers.
    pub fn is_breaking(&self) -> bool {
        matches!(
            self,
            SchemaChange::PredicateRemoved { .. }
                | SchemaChange::PredicateArityChanged { .. }
                | SchemaChange::DomainRemoved { .. }
                | SchemaChange::RuleRemoved { .. }
        )
    }

    /// Human-readable description of the change.
    pub fn description(&self) -> String {
        match self {
            SchemaChange::PredicateAdded { name, arity } => {
                format!("Predicate '{}' added (arity {})", name, arity)
            }
            SchemaChange::PredicateRemoved { name, arity } => {
                format!("Predicate '{}' removed (arity {})", name, arity)
            }
            SchemaChange::PredicateArityChanged {
                name,
                old_arity,
                new_arity,
            } => {
                format!(
                    "Predicate '{}' arity changed from {} to {}",
                    name, old_arity, new_arity
                )
            }
            SchemaChange::DomainAdded { name } => format!("Domain '{}' added", name),
            SchemaChange::DomainRemoved { name } => format!("Domain '{}' removed", name),
            SchemaChange::RuleAdded { name } => format!("Rule/variable '{}' added", name),
            SchemaChange::RuleRemoved { name } => format!("Rule/variable '{}' removed", name),
            SchemaChange::PredicateRenamed { old_name, new_name } => {
                format!("Predicate '{}' renamed to '{}'", old_name, new_name)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChangeSeverity
// ─────────────────────────────────────────────────────────────────────────────

/// Severity classification for a schema change.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChangeSeverity {
    /// Additive change — new predicate, domain, or rule.
    Info,
    /// Potentially risky change — rename or structural shift.
    Warning,
    /// Breaking change — removal or arity change.
    Breaking,
}

impl ChangeSeverity {
    /// Derive the severity from a [`SchemaChange`].
    pub fn from_change(change: &SchemaChange) -> Self {
        if change.is_breaking() {
            ChangeSeverity::Breaking
        } else {
            match change {
                SchemaChange::PredicateRenamed { .. } => ChangeSeverity::Warning,
                _ => ChangeSeverity::Info,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaMigrationStep
// ─────────────────────────────────────────────────────────────────────────────

/// A concrete migration step to transform the old schema into the new schema.
#[derive(Debug, Clone)]
pub enum SchemaMigrationStep {
    /// Create a new predicate slot.
    AddPredicate { name: String, arity: usize },
    /// Drop an existing predicate slot.
    RemovePredicate { name: String },
    /// Rename a predicate (non-destructive when reverse is possible).
    RenamePredicate { old_name: String, new_name: String },
    /// Extend a predicate by inserting a new positional argument.
    AddArityColumn {
        predicate: String,
        position: usize,
        default_value: String,
    },
    /// Shrink a predicate by removing a positional argument.
    RemoveArityColumn { predicate: String, position: usize },
    /// Register a new domain.
    AddDomain { name: String },
    /// Unregister an existing domain.
    RemoveDomain { name: String },
    /// Bind a new variable/rule entry.
    AddRule { name: String },
    /// Remove an existing variable/rule entry.
    RemoveRule { name: String },
}

impl SchemaMigrationStep {
    /// Human-readable description of the step.
    pub fn description(&self) -> String {
        match self {
            SchemaMigrationStep::AddPredicate { name, arity } => {
                format!("Add predicate '{}' with arity {}", name, arity)
            }
            SchemaMigrationStep::RemovePredicate { name } => {
                format!("Remove predicate '{}'", name)
            }
            SchemaMigrationStep::RenamePredicate { old_name, new_name } => {
                format!("Rename predicate '{}' → '{}'", old_name, new_name)
            }
            SchemaMigrationStep::AddArityColumn {
                predicate,
                position,
                default_value,
            } => {
                format!(
                    "Add column at position {} to predicate '{}' (default: '{}')",
                    position, predicate, default_value
                )
            }
            SchemaMigrationStep::RemoveArityColumn {
                predicate,
                position,
            } => {
                format!(
                    "Remove column at position {} from predicate '{}'",
                    position, predicate
                )
            }
            SchemaMigrationStep::AddDomain { name } => {
                format!("Add domain '{}'", name)
            }
            SchemaMigrationStep::RemoveDomain { name } => {
                format!("Remove domain '{}'", name)
            }
            SchemaMigrationStep::AddRule { name } => {
                format!("Add rule/variable '{}'", name)
            }
            SchemaMigrationStep::RemoveRule { name } => {
                format!("Remove rule/variable '{}'", name)
            }
        }
    }

    /// Returns `true` when this step is destructive (data may be lost).
    pub fn is_destructive(&self) -> bool {
        matches!(
            self,
            SchemaMigrationStep::RemovePredicate { .. }
                | SchemaMigrationStep::RemoveArityColumn { .. }
                | SchemaMigrationStep::RemoveDomain { .. }
                | SchemaMigrationStep::RemoveRule { .. }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaMigrationPlan
// ─────────────────────────────────────────────────────────────────────────────

/// The full migration plan produced by [`compute_migration`].
#[derive(Debug, Clone)]
pub struct SchemaMigrationPlan {
    /// All detected changes between old and new schema.
    pub changes: Vec<SchemaChange>,
    /// Ordered list of steps to apply to reach the new schema.
    pub steps: Vec<SchemaMigrationStep>,
    /// `true` when at least one change is breaking.
    pub has_breaking_changes: bool,
    /// Count of breaking changes.
    pub breaking_count: usize,
    /// Count of warning-level changes.
    pub warning_count: usize,
    /// Count of info-level changes.
    pub info_count: usize,
}

impl SchemaMigrationPlan {
    /// Returns `true` when no changes were detected.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Total number of detected changes.
    pub fn num_changes(&self) -> usize {
        self.changes.len()
    }

    /// Returns references to all breaking changes.
    pub fn breaking_changes(&self) -> Vec<&SchemaChange> {
        self.changes.iter().filter(|c| c.is_breaking()).collect()
    }

    /// Produces a multi-line human-readable summary report.
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Schema Migration Report ===\n");
        out.push_str(&format!("Total changes : {}\n", self.num_changes()));
        out.push_str(&format!("Breaking      : {}\n", self.breaking_count));
        out.push_str(&format!("Warnings      : {}\n", self.warning_count));
        out.push_str(&format!("Info          : {}\n", self.info_count));
        if !self.changes.is_empty() {
            out.push_str("\nChanges:\n");
            for change in &self.changes {
                let severity = ChangeSeverity::from_change(change);
                let tag = match severity {
                    ChangeSeverity::Breaking => "[BREAKING]",
                    ChangeSeverity::Warning => "[WARNING] ",
                    ChangeSeverity::Info => "[INFO]    ",
                };
                out.push_str(&format!("  {} {}\n", tag, change.description()));
            }
        }
        out
    }

    /// Produces a multi-line list of migration steps.
    pub fn format_steps(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Migration Steps ===\n");
        if self.steps.is_empty() {
            out.push_str("  (no steps required)\n");
        } else {
            for (idx, step) in self.steps.iter().enumerate() {
                let destructive = if step.is_destructive() {
                    " [DESTRUCTIVE]"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "  {:>3}. {}{}\n",
                    idx + 1,
                    step.description(),
                    destructive
                ));
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MigrationError
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for migration computation and validation.
#[derive(Debug, Clone)]
pub enum MigrationError {
    /// Two or more changes conflict with each other.
    ConflictingChanges(String),
    /// Multiple rename candidates found for a single removed predicate.
    AmbiguousRename { candidates: Vec<String> },
    /// The schema itself is malformed.
    InvalidSchema(String),
    /// Breaking changes were detected but `allow_breaking_changes` is `false`.
    BreakingChangesNotAllowed { count: usize },
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationError::ConflictingChanges(msg) => {
                write!(f, "Conflicting migration changes: {}", msg)
            }
            MigrationError::AmbiguousRename { candidates } => {
                write!(f, "Ambiguous rename: multiple candidates {:?}", candidates)
            }
            MigrationError::InvalidSchema(msg) => {
                write!(f, "Invalid schema: {}", msg)
            }
            MigrationError::BreakingChangesNotAllowed { count } => {
                write!(
                    f,
                    "Migration contains {} breaking change(s) but allow_breaking_changes is false",
                    count
                )
            }
        }
    }
}

impl std::error::Error for MigrationError {}

// ─────────────────────────────────────────────────────────────────────────────
// MigrationConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the migration engine.
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Attempt to detect predicate renames (same arity, similar name).
    pub detect_renames: bool,
    /// Minimum Dice-bigram similarity score `[0.0, 1.0]` to consider a rename.
    pub rename_similarity_threshold: f64,
    /// When `false`, [`compute_migration`] returns an error if breaking changes exist.
    pub allow_breaking_changes: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            detect_renames: true,
            rename_similarity_threshold: 0.7,
            allow_breaking_changes: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight snapshot of a [`SymbolTable`] for comparison purposes.
#[derive(Debug, Clone)]
pub struct SchemaSnapshot {
    /// Ordered list of predicate names.
    pub predicate_names: Vec<String>,
    /// Ordered list of domain names.
    pub domain_names: Vec<String>,
    /// Ordered list of variable/rule names.
    pub rule_names: Vec<String>,
    /// Predicate name → arity mapping.
    pub predicate_arities: HashMap<String, usize>,
}

impl SchemaSnapshot {
    /// Build a snapshot from a live [`SymbolTable`].
    pub fn from_symbol_table(table: &SymbolTable) -> Self {
        let predicate_names: Vec<String> = table.predicates.keys().cloned().collect();
        let domain_names: Vec<String> = table.domains.keys().cloned().collect();
        let rule_names: Vec<String> = table.variables.keys().cloned().collect();
        let predicate_arities: HashMap<String, usize> = table
            .predicates
            .iter()
            .map(|(name, info)| (name.clone(), info.arity))
            .collect();

        Self {
            predicate_names,
            domain_names,
            rule_names,
            predicate_arities,
        }
    }

    /// Number of predicates in the snapshot.
    pub fn predicate_count(&self) -> usize {
        self.predicate_names.len()
    }

    /// Number of domains in the snapshot.
    pub fn domain_count(&self) -> usize {
        self.domain_names.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// string_similarity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Dice coefficient on character bigrams of two strings.
///
/// Returns a value in `[0.0, 1.0]`, where `1.0` means identical strings.
/// Empty strings return `0.0`.
pub fn string_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let bigrams_a = collect_bigrams(a);
    let bigrams_b = collect_bigrams(b);

    if bigrams_a.is_empty() && bigrams_b.is_empty() {
        // Both have fewer than 2 chars; fall back to char equality
        return if a == b { 1.0 } else { 0.0 };
    }
    if bigrams_a.is_empty() || bigrams_b.is_empty() {
        return 0.0;
    }

    let total = bigrams_a.len() + bigrams_b.len();
    let common = count_common_bigrams(&bigrams_a, &bigrams_b);

    (2 * common) as f64 / total as f64
}

/// Collect all character bigrams from a string as `(char, char)` pairs.
fn collect_bigrams(s: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    chars.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Count how many bigrams from `a` appear in `b`, respecting multiplicity.
fn count_common_bigrams(a: &[(char, char)], b: &[(char, char)]) -> usize {
    // Build a frequency map for b
    let mut freq: HashMap<(char, char), usize> = HashMap::new();
    for &bigram in b {
        *freq.entry(bigram).or_insert(0) += 1;
    }

    let mut common = 0usize;
    let mut used: HashMap<(char, char), usize> = HashMap::new();
    for &bigram in a {
        let available = freq.get(&bigram).copied().unwrap_or(0);
        let already_used = used.get(&bigram).copied().unwrap_or(0);
        if already_used < available {
            common += 1;
            *used.entry(bigram).or_insert(0) += 1;
        }
    }
    common
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_migration
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the full migration plan needed to go from `old_schema` to `new_schema`.
///
/// Steps:
/// 1. Build snapshots of both schemas.
/// 2. Detect predicate additions, removals, and arity changes.
/// 3. Detect domain additions and removals.
/// 4. Detect variable/rule additions and removals.
/// 5. Optionally resolve renames by similarity.
/// 6. Synthesise `SchemaMigrationStep`s.
/// 7. Enforce `allow_breaking_changes` policy.
pub fn compute_migration(
    old_schema: &SymbolTable,
    new_schema: &SymbolTable,
    config: &MigrationConfig,
) -> Result<SchemaMigrationPlan, MigrationError> {
    let old_snap = SchemaSnapshot::from_symbol_table(old_schema);
    let new_snap = SchemaSnapshot::from_symbol_table(new_schema);

    let mut changes: Vec<SchemaChange> = Vec::new();

    // ── Predicates ──────────────────────────────────────────────────────────

    let old_pred_set: HashSet<&String> = old_snap.predicate_names.iter().collect();
    let new_pred_set: HashSet<&String> = new_snap.predicate_names.iter().collect();

    // Arity changes: present in both but arity differs
    for name in old_pred_set.intersection(&new_pred_set) {
        let old_arity = old_snap.predicate_arities.get(*name).copied().unwrap_or(0);
        let new_arity = new_snap.predicate_arities.get(*name).copied().unwrap_or(0);
        if old_arity != new_arity {
            changes.push(SchemaChange::PredicateArityChanged {
                name: (*name).clone(),
                old_arity,
                new_arity,
            });
        }
    }

    // Candidates for removal and addition
    let mut removed_preds: Vec<String> = old_pred_set
        .difference(&new_pred_set)
        .map(|s| (*s).clone())
        .collect();
    let mut added_preds: Vec<String> = new_pred_set
        .difference(&old_pred_set)
        .map(|s| (*s).clone())
        .collect();
    removed_preds.sort();
    added_preds.sort();

    // Rename detection
    if config.detect_renames {
        detect_predicate_renames(
            &mut removed_preds,
            &mut added_preds,
            &old_snap.predicate_arities,
            &new_snap.predicate_arities,
            config.rename_similarity_threshold,
            &mut changes,
        )?;
    }

    // Remaining removals and additions
    for name in &removed_preds {
        let arity = old_snap.predicate_arities.get(name).copied().unwrap_or(0);
        changes.push(SchemaChange::PredicateRemoved {
            name: name.clone(),
            arity,
        });
    }
    for name in &added_preds {
        let arity = new_snap.predicate_arities.get(name).copied().unwrap_or(0);
        changes.push(SchemaChange::PredicateAdded {
            name: name.clone(),
            arity,
        });
    }

    // ── Domains ─────────────────────────────────────────────────────────────

    let old_domain_set: HashSet<&String> = old_snap.domain_names.iter().collect();
    let new_domain_set: HashSet<&String> = new_snap.domain_names.iter().collect();

    let mut removed_domains: Vec<String> = old_domain_set
        .difference(&new_domain_set)
        .map(|s| (*s).clone())
        .collect();
    let mut added_domains: Vec<String> = new_domain_set
        .difference(&old_domain_set)
        .map(|s| (*s).clone())
        .collect();
    removed_domains.sort();
    added_domains.sort();

    for name in &removed_domains {
        changes.push(SchemaChange::DomainRemoved { name: name.clone() });
    }
    for name in &added_domains {
        changes.push(SchemaChange::DomainAdded { name: name.clone() });
    }

    // ── Variables / Rules ────────────────────────────────────────────────────

    let old_rule_set: HashSet<&String> = old_snap.rule_names.iter().collect();
    let new_rule_set: HashSet<&String> = new_snap.rule_names.iter().collect();

    let mut removed_rules: Vec<String> = old_rule_set
        .difference(&new_rule_set)
        .map(|s| (*s).clone())
        .collect();
    let mut added_rules: Vec<String> = new_rule_set
        .difference(&old_rule_set)
        .map(|s| (*s).clone())
        .collect();
    removed_rules.sort();
    added_rules.sort();

    for name in &removed_rules {
        changes.push(SchemaChange::RuleRemoved { name: name.clone() });
    }
    for name in &added_rules {
        changes.push(SchemaChange::RuleAdded { name: name.clone() });
    }

    // ── Severity counts ──────────────────────────────────────────────────────

    let mut breaking_count = 0usize;
    let mut warning_count = 0usize;
    let mut info_count = 0usize;
    for change in &changes {
        match ChangeSeverity::from_change(change) {
            ChangeSeverity::Breaking => breaking_count += 1,
            ChangeSeverity::Warning => warning_count += 1,
            ChangeSeverity::Info => info_count += 1,
        }
    }
    let has_breaking_changes = breaking_count > 0;

    if !config.allow_breaking_changes && has_breaking_changes {
        return Err(MigrationError::BreakingChangesNotAllowed {
            count: breaking_count,
        });
    }

    // ── Build steps ──────────────────────────────────────────────────────────

    let steps = build_migration_steps(
        &changes,
        &old_snap.predicate_arities,
        &new_snap.predicate_arities,
    );

    let plan = SchemaMigrationPlan {
        changes,
        steps,
        has_breaking_changes,
        breaking_count,
        warning_count,
        info_count,
    };

    Ok(plan)
}

/// Detect renames among removed/added predicates by Dice-bigram similarity.
/// Matched pairs are emitted as [`SchemaChange::PredicateRenamed`] and removed
/// from the `removed` / `added` vectors.
fn detect_predicate_renames(
    removed: &mut Vec<String>,
    added: &mut Vec<String>,
    old_arities: &HashMap<String, usize>,
    new_arities: &HashMap<String, usize>,
    threshold: f64,
    changes: &mut Vec<SchemaChange>,
) -> Result<(), MigrationError> {
    // Keep track of which names have been consumed
    let mut consumed_removed: HashSet<String> = HashSet::new();
    let mut consumed_added: HashSet<String> = HashSet::new();

    // For each removed predicate, find all added candidates with the same arity
    // and similarity >= threshold.
    for old_name in removed.iter() {
        let old_arity = old_arities.get(old_name).copied().unwrap_or(0);

        let mut candidates: Vec<(String, f64)> = added
            .iter()
            .filter(|new_name| !consumed_added.contains(*new_name))
            .filter(|new_name| new_arities.get(*new_name).copied().unwrap_or(0) == old_arity)
            .filter_map(|new_name| {
                let sim = string_similarity(old_name, new_name);
                if sim >= threshold {
                    Some((new_name.clone(), sim))
                } else {
                    None
                }
            })
            .collect();

        if candidates.is_empty() {
            continue;
        }

        // Sort descending by similarity for deterministic selection
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        // Check for ambiguous renames (multiple candidates with equal top score)
        let top_score = candidates[0].1;
        let top_candidates: Vec<String> = candidates
            .iter()
            .filter(|(_, s)| (s - top_score).abs() < f64::EPSILON)
            .map(|(n, _)| n.clone())
            .collect();

        if top_candidates.len() > 1 {
            return Err(MigrationError::AmbiguousRename {
                candidates: top_candidates,
            });
        }

        let new_name = candidates[0].0.clone();
        changes.push(SchemaChange::PredicateRenamed {
            old_name: old_name.clone(),
            new_name: new_name.clone(),
        });
        consumed_removed.insert(old_name.clone());
        consumed_added.insert(new_name);
    }

    // Remove consumed entries
    removed.retain(|n| !consumed_removed.contains(n));
    added.retain(|n| !consumed_added.contains(n));

    Ok(())
}

/// Translate detected [`SchemaChange`]s into ordered [`SchemaMigrationStep`]s.
fn build_migration_steps(
    changes: &[SchemaChange],
    old_arities: &HashMap<String, usize>,
    new_arities: &HashMap<String, usize>,
) -> Vec<SchemaMigrationStep> {
    let mut steps: Vec<SchemaMigrationStep> = Vec::new();

    for change in changes {
        match change {
            SchemaChange::PredicateAdded { name, arity } => {
                steps.push(SchemaMigrationStep::AddPredicate {
                    name: name.clone(),
                    arity: *arity,
                });
            }
            SchemaChange::PredicateRemoved { name, .. } => {
                steps.push(SchemaMigrationStep::RemovePredicate { name: name.clone() });
            }
            SchemaChange::PredicateArityChanged {
                name,
                old_arity,
                new_arity,
            } => {
                let old_a = old_arities.get(name).copied().unwrap_or(*old_arity);
                let new_a = new_arities.get(name).copied().unwrap_or(*new_arity);
                if new_a > old_a {
                    // Columns were added at the end
                    for pos in old_a..new_a {
                        steps.push(SchemaMigrationStep::AddArityColumn {
                            predicate: name.clone(),
                            position: pos,
                            default_value: "NULL".to_string(),
                        });
                    }
                } else {
                    // Columns were removed from the end
                    for pos in (new_a..old_a).rev() {
                        steps.push(SchemaMigrationStep::RemoveArityColumn {
                            predicate: name.clone(),
                            position: pos,
                        });
                    }
                }
            }
            SchemaChange::DomainAdded { name } => {
                steps.push(SchemaMigrationStep::AddDomain { name: name.clone() });
            }
            SchemaChange::DomainRemoved { name } => {
                steps.push(SchemaMigrationStep::RemoveDomain { name: name.clone() });
            }
            SchemaChange::RuleAdded { name } => {
                steps.push(SchemaMigrationStep::AddRule { name: name.clone() });
            }
            SchemaChange::RuleRemoved { name } => {
                steps.push(SchemaMigrationStep::RemoveRule { name: name.clone() });
            }
            SchemaChange::PredicateRenamed { old_name, new_name } => {
                steps.push(SchemaMigrationStep::RenamePredicate {
                    old_name: old_name.clone(),
                    new_name: new_name.clone(),
                });
            }
        }
    }

    steps
}

// ─────────────────────────────────────────────────────────────────────────────
// validate_plan
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a migration plan is internally self-consistent.
///
/// Checks for:
/// - Duplicate `AddPredicate` steps for the same name
/// - Duplicate `RemovePredicate` steps for the same name
/// - A predicate both added and removed in the same plan
pub fn validate_plan(plan: &SchemaMigrationPlan) -> Result<(), MigrationError> {
    let mut added_predicates: HashSet<String> = HashSet::new();
    let mut removed_predicates: HashSet<String> = HashSet::new();
    let mut added_domains: HashSet<String> = HashSet::new();
    let mut removed_domains: HashSet<String> = HashSet::new();

    for step in &plan.steps {
        match step {
            SchemaMigrationStep::AddPredicate { name, .. } => {
                check_not_duplicate(&mut added_predicates, name, "Predicate", "added")?;
                check_not_conflict(&removed_predicates, name, "Predicate")?;
            }
            SchemaMigrationStep::RemovePredicate { name } => {
                check_not_duplicate(&mut removed_predicates, name, "Predicate", "removed")?;
                check_not_conflict(&added_predicates, name, "Predicate")?;
            }
            SchemaMigrationStep::AddDomain { name } => {
                check_not_duplicate(&mut added_domains, name, "Domain", "added")?;
            }
            SchemaMigrationStep::RemoveDomain { name } => {
                check_not_duplicate(&mut removed_domains, name, "Domain", "removed")?;
                check_not_conflict(&added_domains, name, "Domain")?;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Returns an error when `name` is already present in `seen`.
fn check_not_duplicate(
    seen: &mut HashSet<String>,
    name: &str,
    kind: &str,
    action: &str,
) -> Result<(), MigrationError> {
    if !seen.insert(name.to_string()) {
        return Err(MigrationError::ConflictingChanges(format!(
            "{} '{}' is {} more than once",
            kind, name, action
        )));
    }
    Ok(())
}

/// Returns an error when `name` already exists in the opposing set (added vs removed).
fn check_not_conflict(
    opposing: &HashSet<String>,
    name: &str,
    kind: &str,
) -> Result<(), MigrationError> {
    if opposing.contains(name) {
        return Err(MigrationError::ConflictingChanges(format!(
            "{} '{}' is both added and removed",
            kind, name
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainInfo, PredicateInfo, SymbolTable};

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Build a minimal SymbolTable with one domain "D" and the given predicates.
    fn table_with_predicates(preds: &[(&str, usize)]) -> SymbolTable {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new("D", 1)).expect("add domain D");
        for &(name, arity) in preds {
            let domains: Vec<String> = (0..arity).map(|_| "D".to_string()).collect();
            t.add_predicate(PredicateInfo::new(name, domains))
                .expect("add predicate");
        }
        t
    }

    // ── string_similarity ────────────────────────────────────────────────────

    #[test]
    fn test_string_similarity_identical() {
        let sim = string_similarity("foo", "foo");
        assert!(
            (sim - 1.0).abs() < f64::EPSILON,
            "identical strings must have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_string_similarity_different() {
        let sim = string_similarity("abc", "xyz");
        assert!(
            sim < 0.5,
            "completely different strings should have similarity < 0.5, got {}",
            sim
        );
    }

    #[test]
    fn test_string_similarity_partial() {
        let sim = string_similarity("predicate", "predicat");
        assert!(
            sim > 0.7,
            "highly similar strings should exceed 0.7 similarity, got {}",
            sim
        );
    }

    // ── SchemaSnapshot ───────────────────────────────────────────────────────

    #[test]
    fn test_schema_snapshot_from_table() {
        let mut t = SymbolTable::new();
        t.add_domain(DomainInfo::new("Person", 100))
            .expect("domain");
        t.add_domain(DomainInfo::new("Animal", 50)).expect("domain");
        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()]);
        t.add_predicate(pred).expect("predicate");

        let snap = SchemaSnapshot::from_symbol_table(&t);
        assert_eq!(snap.domain_count(), 2);
        assert_eq!(snap.predicate_count(), 1);
        assert_eq!(snap.predicate_arities["knows"], 2);
    }

    // ── SchemaChange::is_breaking ────────────────────────────────────────────

    #[test]
    fn test_schema_change_is_breaking_removal() {
        let change = SchemaChange::PredicateRemoved {
            name: "foo".to_string(),
            arity: 1,
        };
        assert!(change.is_breaking());
    }

    #[test]
    fn test_schema_change_is_breaking_arity() {
        let change = SchemaChange::PredicateArityChanged {
            name: "foo".to_string(),
            old_arity: 1,
            new_arity: 2,
        };
        assert!(change.is_breaking());
    }

    #[test]
    fn test_schema_change_not_breaking_added() {
        let change = SchemaChange::PredicateAdded {
            name: "bar".to_string(),
            arity: 2,
        };
        assert!(!change.is_breaking());
    }

    // ── SchemaMigrationStep ──────────────────────────────────────────────────

    #[test]
    fn test_migration_step_is_destructive() {
        let step = SchemaMigrationStep::RemovePredicate {
            name: "old_pred".to_string(),
        };
        assert!(step.is_destructive());
    }

    #[test]
    fn test_migration_step_description_nonempty() {
        let step = SchemaMigrationStep::AddPredicate {
            name: "new_pred".to_string(),
            arity: 3,
        };
        let desc = step.description();
        assert!(!desc.is_empty(), "description must not be empty");
        assert!(desc.contains("new_pred"));
    }

    // ── compute_migration ────────────────────────────────────────────────────

    #[test]
    fn test_compute_migration_empty_schemas() {
        let old = SymbolTable::new();
        let new = SymbolTable::new();
        let config = MigrationConfig::default();
        let plan = compute_migration(&old, &new, &config).expect("migration");
        assert!(
            plan.is_empty(),
            "both empty schemas should yield empty plan"
        );
    }

    #[test]
    fn test_compute_migration_add_predicate() {
        let old = table_with_predicates(&[]);
        let new = table_with_predicates(&[("likes", 2)]);
        let config = MigrationConfig::default();
        let plan = compute_migration(&old, &new, &config).expect("migration");

        assert!(!plan.is_empty());
        let added = plan
            .changes
            .iter()
            .any(|c| matches!(c, SchemaChange::PredicateAdded { name, .. } if name == "likes"));
        assert!(added, "expected PredicateAdded for 'likes'");
    }

    #[test]
    fn test_compute_migration_remove_predicate() {
        let old = table_with_predicates(&[("old_pred", 1)]);
        let new = table_with_predicates(&[]);
        // Disable rename detection to guarantee we see a removal
        let config = MigrationConfig {
            detect_renames: false,
            ..Default::default()
        };
        let plan = compute_migration(&old, &new, &config).expect("migration");

        let removed = plan.changes.iter().any(
            |c| matches!(c, SchemaChange::PredicateRemoved { name, .. } if name == "old_pred"),
        );
        assert!(removed, "expected PredicateRemoved for 'old_pred'");
    }

    #[test]
    fn test_compute_migration_arity_change() {
        let old = table_with_predicates(&[("pred_a", 1)]);
        // Rebuild new table manually with changed arity
        let mut new = SymbolTable::new();
        new.add_domain(DomainInfo::new("D", 1)).expect("domain");
        new.add_predicate(PredicateInfo::new(
            "pred_a",
            vec!["D".to_string(), "D".to_string()],
        ))
        .expect("predicate");

        let config = MigrationConfig::default();
        let plan = compute_migration(&old, &new, &config).expect("migration");

        let arity_changed = plan.changes.iter().any(|c| {
            matches!(
                c,
                SchemaChange::PredicateArityChanged { name, old_arity: 1, new_arity: 2 }
                    if name == "pred_a"
            )
        });
        assert!(arity_changed, "expected PredicateArityChanged for 'pred_a'");
    }

    #[test]
    fn test_compute_migration_no_change() {
        let schema = table_with_predicates(&[("pred_x", 2)]);
        let config = MigrationConfig::default();
        let plan = compute_migration(&schema, &schema, &config).expect("migration");
        assert!(
            plan.is_empty(),
            "identical schemas must produce an empty plan"
        );
    }

    // ── SchemaMigrationPlan ──────────────────────────────────────────────────

    #[test]
    fn test_migration_plan_has_breaking() {
        let old = table_with_predicates(&[("to_remove", 1)]);
        let new = table_with_predicates(&[]);
        let config = MigrationConfig {
            detect_renames: false,
            ..Default::default()
        };
        let plan = compute_migration(&old, &new, &config).expect("migration");
        assert!(plan.has_breaking_changes);
        assert!(plan.breaking_count > 0);
    }

    #[test]
    fn test_migration_plan_format_report_nonempty() {
        let old = table_with_predicates(&[("p", 1)]);
        let new = table_with_predicates(&[("p", 2)]);
        let config = MigrationConfig::default();
        let plan = compute_migration(&old, &new, &config).expect("migration");
        let report = plan.format_report();
        assert!(
            !report.is_empty(),
            "format_report should return non-empty string"
        );
        assert!(report.contains("Migration Report"));
    }

    #[test]
    fn test_migration_plan_format_steps_nonempty() {
        let old = table_with_predicates(&[("p", 1)]);
        let new = table_with_predicates(&[("q", 1)]);
        let config = MigrationConfig {
            detect_renames: false,
            ..Default::default()
        };
        let plan = compute_migration(&old, &new, &config).expect("migration");
        let steps_str = plan.format_steps();
        assert!(!steps_str.is_empty());
        assert!(steps_str.contains("Migration Steps"));
    }

    // ── validate_plan ────────────────────────────────────────────────────────

    #[test]
    fn test_validate_plan_empty_ok() {
        let plan = SchemaMigrationPlan {
            changes: Vec::new(),
            steps: Vec::new(),
            has_breaking_changes: false,
            breaking_count: 0,
            warning_count: 0,
            info_count: 0,
        };
        assert!(validate_plan(&plan).is_ok());
    }

    // ── MigrationConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_migration_config_default() {
        let config = MigrationConfig::default();
        assert!(config.detect_renames);
        assert!(config.allow_breaking_changes);
        assert!(config.rename_similarity_threshold > 0.0);
        assert!(config.rename_similarity_threshold <= 1.0);
    }

    // ── MigrationError ───────────────────────────────────────────────────────

    #[test]
    fn test_migration_error_display() {
        let err = MigrationError::ConflictingChanges("test conflict".to_string());
        let msg = format!("{}", err);
        assert!(
            !msg.is_empty(),
            "error Display must produce non-empty string"
        );
        assert!(msg.contains("test conflict"));

        let err2 = MigrationError::AmbiguousRename {
            candidates: vec!["a".to_string(), "b".to_string()],
        };
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("Ambiguous"));

        let err3 = MigrationError::InvalidSchema("bad schema".to_string());
        let msg3 = format!("{}", err3);
        assert!(msg3.contains("bad schema"));
    }
}
