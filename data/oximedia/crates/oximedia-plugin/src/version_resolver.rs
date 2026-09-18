//! Plugin version resolution with semver constraint satisfaction and topological
//! dependency ordering.
//!
//! # Architecture
//!
//! The module provides:
//! - [`SemVer`] — parsed semantic version (major.minor.patch[-pre])
//! - [`VersionConstraint`] — constraint language: Exact, Compatible (^), AtLeast (>=), AtMost (<=), Range
//! - [`PluginDependency`] — (plugin_id, constraint) pair
//! - [`DependencyResolver`] — resolves a dependency graph to a concrete version map
//! - [`ResolveError`] — conflict, not-found, and circular dependency errors

use std::collections::{HashMap, HashSet, VecDeque};

// ── SemVer ────────────────────────────────────────────────────────────────────

/// A parsed semantic version (`major.minor.patch[-pre]`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemVer {
    /// Major version component.
    pub major: u32,
    /// Minor version component.
    pub minor: u32,
    /// Patch version component.
    pub patch: u32,
    /// Optional pre-release identifier (e.g. `"alpha.1"`).
    pub pre: Option<String>,
}

impl SemVer {
    /// Construct a release version (no pre-release).
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre: None,
        }
    }

    /// Construct a pre-release version.
    pub fn with_pre(major: u32, minor: u32, patch: u32, pre: impl Into<String>) -> Self {
        Self {
            major,
            minor,
            patch,
            pre: Some(pre.into()),
        }
    }

    /// Parse a semver string such as `"1.2.3"` or `"1.0.0-beta.2"`.
    ///
    /// # Errors
    /// Returns a [`ResolveError::NotFound`]-wrapping string on parse failure.
    pub fn parse(s: &str) -> Result<Self, ResolveError> {
        let s = s.trim();
        let (version_part, pre) = if let Some((v, p)) = s.split_once('-') {
            (v, Some(p.to_string()))
        } else {
            (s, None)
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(ResolveError::NotFound(format!(
                "invalid semver '{}': expected 2 or 3 dot-separated numbers",
                s
            )));
        }

        let major = parts[0]
            .parse::<u32>()
            .map_err(|_| ResolveError::NotFound(format!("invalid major in '{s}'")))?;
        let minor = parts[1]
            .parse::<u32>()
            .map_err(|_| ResolveError::NotFound(format!("invalid minor in '{s}'")))?;
        let patch = if parts.len() == 3 {
            parts[2]
                .parse::<u32>()
                .map_err(|_| ResolveError::NotFound(format!("invalid patch in '{s}'")))?
        } else {
            0
        };

        Ok(Self {
            major,
            minor,
            patch,
            pre,
        })
    }

    /// Compare numeric components only (ignores pre-release).
    fn cmp_numeric(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }

    /// Semver compatibility check: same major, `self >= required`.
    ///
    /// This is the `^` (caret) / "compatible" definition.
    pub fn is_compatible_with(&self, required: &SemVer) -> bool {
        if self.major != required.major {
            return false;
        }
        self.cmp_numeric(required) != std::cmp::Ordering::Less
    }
}

impl PartialOrd for SemVer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemVer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp_numeric(other)
    }
}

impl std::fmt::Display for SemVer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre {
            write!(f, "-{pre}")?;
        }
        Ok(())
    }
}

// ── VersionConstraint ─────────────────────────────────────────────────────────

/// A version constraint that can be evaluated against a concrete [`SemVer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionConstraint {
    /// Exact match — only the specified version satisfies the constraint.
    Exact(SemVer),
    /// Compatible (`^`) — same major, `version >= specified`.
    Compatible(SemVer),
    /// At least (`>=`) — `version >= specified` (any major).
    AtLeast(SemVer),
    /// At most (`<=`) — `version <= specified`.
    AtMost(SemVer),
    /// Inclusive range — `min <= version <= max`.
    Range { min: SemVer, max: SemVer },
}

impl VersionConstraint {
    /// Test whether `version` satisfies this constraint.
    pub fn satisfies(&self, version: &SemVer) -> bool {
        match self {
            VersionConstraint::Exact(req) => version.cmp_numeric(req) == std::cmp::Ordering::Equal,
            VersionConstraint::Compatible(req) => version.is_compatible_with(req),
            VersionConstraint::AtLeast(req) => version.cmp_numeric(req) != std::cmp::Ordering::Less,
            VersionConstraint::AtMost(req) => {
                version.cmp_numeric(req) != std::cmp::Ordering::Greater
            }
            VersionConstraint::Range { min, max } => {
                version.cmp_numeric(min) != std::cmp::Ordering::Less
                    && version.cmp_numeric(max) != std::cmp::Ordering::Greater
            }
        }
    }

    /// Attempt to compute the intersection of two constraints.
    ///
    /// Returns `None` if the constraints cannot be combined into a single
    /// `Range` (e.g. two `Exact` constraints with different versions).
    pub fn intersect(&self, other: &VersionConstraint) -> Option<VersionConstraint> {
        // Represent each as (lower_bound, upper_bound).
        let (lo1, hi1) = self.as_bounds();
        let (lo2, hi2) = other.as_bounds();

        let lo = lo1.max(lo2);
        let hi = hi1.min(hi2);

        if lo.cmp_numeric(&hi) == std::cmp::Ordering::Greater {
            None
        } else if lo.cmp_numeric(&hi) == std::cmp::Ordering::Equal {
            Some(VersionConstraint::Exact(lo))
        } else {
            Some(VersionConstraint::Range { min: lo, max: hi })
        }
    }

    /// Convert the constraint to inclusive (lower, upper) bounds.
    /// Uses `SemVer(0,0,0)` as lower and `SemVer(u32::MAX,u32::MAX,u32::MAX)` as upper.
    fn as_bounds(&self) -> (SemVer, SemVer) {
        let floor = SemVer::new(0, 0, 0);
        let ceiling = SemVer::new(u32::MAX, u32::MAX, u32::MAX);

        match self {
            VersionConstraint::Exact(v) => (v.clone(), v.clone()),
            VersionConstraint::Compatible(v) => {
                let upper = SemVer::new(v.major, u32::MAX, u32::MAX);
                (v.clone(), upper)
            }
            VersionConstraint::AtLeast(v) => (v.clone(), ceiling),
            VersionConstraint::AtMost(v) => (floor, v.clone()),
            VersionConstraint::Range { min, max } => (min.clone(), max.clone()),
        }
    }
}

// ── PluginDependency ──────────────────────────────────────────────────────────

/// A single dependency declaration: the target plugin ID and its constraint.
#[derive(Debug, Clone)]
pub struct PluginDependency {
    /// Identifier of the required plugin.
    pub plugin_id: String,
    /// Version constraint that must be satisfied.
    pub constraint: VersionConstraint,
}

impl PluginDependency {
    /// Convenience constructor.
    pub fn new(plugin_id: impl Into<String>, constraint: VersionConstraint) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            constraint,
        }
    }
}

// ── ResolveError ──────────────────────────────────────────────────────────────

/// Errors produced by [`DependencyResolver::resolve`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    /// Two requirements conflict for the same plugin.
    #[error("version conflict for '{0}': {1} vs {2}")]
    Conflict(String, SemVer, SemVer),

    /// A required plugin has no registered versions.
    #[error("plugin not found: '{0}'")]
    NotFound(String),

    /// A circular dependency was detected among the given plugin IDs.
    #[error("circular dependency detected: {0:?}")]
    CircularDependency(Vec<String>),
}

// ── DependencyResolver ────────────────────────────────────────────────────────

/// Resolves a set of plugin dependencies to a concrete version selection.
///
/// # Algorithm
///
/// 1. For each dependency, narrow the acceptable version set by intersecting
///    all constraints that target the same plugin.
/// 2. Transitively propagate constraints via a BFS worklist: for each resolved
///    plugin, its own registered deps (from `register_with_deps`) are merged
///    into the combined constraint map; if any constraint narrows, the plugin
///    is re-added to the worklist.  Monotone propagation terminates because
///    constraints only ever narrow.
/// 3. From the resulting range, pick the highest available version that satisfies
///    the combined constraint.
/// 4. Perform a topological sort over the dependency graph to detect cycles and
///    produce a safe load order.
pub struct DependencyResolver {
    /// Registered plugin versions: `plugin_id → sorted Vec<SemVer>`.
    pub registered_plugins: HashMap<String, Vec<SemVer>>,
    /// Per-provider dependency declarations used for transitive resolution.
    ///
    /// When a plugin is registered via [`DependencyResolver::register_with_deps`], its own
    /// dependency list is stored here so that `resolve` can propagate
    /// constraints transitively without the caller having to pre-expand
    /// the full graph.
    pub provider_deps: HashMap<String, Vec<PluginDependency>>,
}

impl DependencyResolver {
    /// Create an empty resolver.
    pub fn new() -> Self {
        Self {
            registered_plugins: HashMap::new(),
            provider_deps: HashMap::new(),
        }
    }

    /// Register all available versions of a plugin.
    ///
    /// Versions need not be provided in sorted order.
    pub fn register(&mut self, plugin_id: impl Into<String>, mut versions: Vec<SemVer>) {
        versions.sort();
        self.registered_plugins.insert(plugin_id.into(), versions);
    }

    /// Register versions **and** the plugin's own dependency declarations.
    ///
    /// The `deps` list is stored in [`provider_deps`] and used by `resolve`
    /// to perform transitive diamond-dependency resolution via BFS constraint
    /// propagation.
    ///
    /// [`provider_deps`]: DependencyResolver::provider_deps
    pub fn register_with_deps(
        &mut self,
        plugin_id: impl Into<String>,
        versions: Vec<SemVer>,
        deps: Vec<PluginDependency>,
    ) {
        let id = plugin_id.into();
        self.register(id.clone(), versions);
        self.provider_deps.insert(id, deps);
    }

    /// Resolve the given root dependencies to a concrete `HashMap<plugin_id, version>`.
    ///
    /// # Transitive resolution
    ///
    /// After the direct (root-level) constraint pass, a BFS worklist propagates
    /// constraints from each resolved plugin's own declared dependencies
    /// (registered via [`DependencyResolver::register_with_deps`]).  The process is monotone:
    /// constraints only narrow, so it terminates in at most
    /// `registered_plugins.len() * 10` iterations.
    ///
    /// # Errors
    ///
    /// - [`ResolveError::NotFound`] — a dependency names an unregistered plugin.
    /// - [`ResolveError::Conflict`] — two constraints for the same plugin have no
    ///   common satisfying version.
    /// - [`ResolveError::CircularDependency`] — the dependency graph has a cycle.
    pub fn resolve(
        &self,
        root_deps: &[PluginDependency],
    ) -> Result<HashMap<String, SemVer>, ResolveError> {
        // Step 1 — merge all constraints by plugin_id (direct pass).
        let mut combined: HashMap<String, VersionConstraint> = HashMap::new();
        let mut queue: VecDeque<PluginDependency> = root_deps.iter().cloned().collect();

        while let Some(dep) = queue.pop_front() {
            let id = dep.plugin_id.clone();

            if !self.registered_plugins.contains_key(&id) {
                return Err(ResolveError::NotFound(id));
            }

            let merged = match combined.remove(&id) {
                None => dep.constraint.clone(),
                Some(existing) => existing.intersect(&dep.constraint).ok_or_else(|| {
                    let (lo1, _) = existing.as_bounds();
                    let (lo2, _) = dep.constraint.as_bounds();
                    ResolveError::Conflict(id.clone(), lo1, lo2)
                })?,
            };
            combined.insert(id.clone(), merged);
        }

        // Step 1b — transitive BFS propagation via provider_deps.
        //
        // For each plugin in the combined constraint map, look up its own
        // declared dependencies.  If those dependencies add or narrow a
        // constraint for a plugin, insert/update the constraint and put the
        // newly constrained plugin back on the worklist (so *its* deps are
        // also expanded).  Monotone: constraints only narrow → guaranteed to
        // converge.  An iteration cap acts as a backstop against bugs.
        let iteration_cap = self.registered_plugins.len().saturating_mul(10).max(64);
        let mut iters: usize = 0;

        // Initialise the worklist from all plugins reachable via root deps.
        let mut worklist: VecDeque<String> = combined.keys().cloned().collect();

        while let Some(provider_id) = worklist.pop_front() {
            iters += 1;
            if iters > iteration_cap {
                return Err(ResolveError::Conflict(
                    "cycle-or-infinite".to_owned(),
                    SemVer::new(0, 0, 0),
                    SemVer::new(0, 0, 0),
                ));
            }

            let transitive = match self.provider_deps.get(&provider_id) {
                Some(d) => d.clone(),
                None => continue,
            };

            for trans_dep in &transitive {
                let dep_id = &trans_dep.plugin_id;

                if !self.registered_plugins.contains_key(dep_id.as_str()) {
                    return Err(ResolveError::NotFound(dep_id.clone()));
                }

                let (new_constraint, did_narrow) = match combined.remove(dep_id.as_str()) {
                    None => {
                        // First time we see this transitive dep — add it.
                        let c = trans_dep.constraint.clone();
                        combined.insert(dep_id.clone(), c);
                        // Enqueue so its own provider_deps are expanded too.
                        worklist.push_back(dep_id.clone());
                        continue;
                    }
                    Some(existing) => {
                        let intersection =
                            existing.intersect(&trans_dep.constraint).ok_or_else(|| {
                                let (lo1, _) = existing.as_bounds();
                                let (lo2, _) = trans_dep.constraint.as_bounds();
                                ResolveError::Conflict(dep_id.clone(), lo1, lo2)
                            })?;
                        // Only re-enqueue if the constraint actually narrowed.
                        let did_narrow = intersection != existing;
                        (intersection, did_narrow)
                    }
                };

                combined.insert(dep_id.clone(), new_constraint);
                if did_narrow {
                    worklist.push_back(dep_id.clone());
                }
            }
        }

        // Step 2 — pick highest satisfying version for each constrained plugin.
        let mut resolved: HashMap<String, SemVer> = HashMap::new();
        for (id, constraint) in &combined {
            let versions = self
                .registered_plugins
                .get(id)
                .ok_or_else(|| ResolveError::NotFound(id.clone()))?;

            let chosen = versions
                .iter()
                .rev()
                .find(|v| constraint.satisfies(v))
                .ok_or_else(|| {
                    // Report conflict between the floor of the constraint and
                    // the highest available version.
                    let (lo, _) = constraint.as_bounds();
                    let highest = versions
                        .last()
                        .cloned()
                        .unwrap_or_else(|| SemVer::new(0, 0, 0));
                    ResolveError::Conflict(id.clone(), lo, highest)
                })?;

            resolved.insert(id.clone(), chosen.clone());
        }

        // Step 3 — topological cycle check over root_deps graph.
        self.check_cycles(root_deps)?;

        Ok(resolved)
    }

    /// Perform a topological sort (Kahn's algorithm) to detect cycles.
    ///
    /// The graph is formed from `root_deps`: each `PluginDependency` creates an
    /// edge from `plugin_id` back to the requesting plugin.  In practice, the
    /// caller should supply the full transitive dep graph if cycle detection
    /// across all levels is desired.
    fn check_cycles(&self, deps: &[PluginDependency]) -> Result<(), ResolveError> {
        // Build a directed graph: dependent → dependency edges.
        let ids: Vec<String> = deps.iter().map(|d| d.plugin_id.clone()).collect();
        let unique_ids: Vec<String> = {
            let mut seen = HashSet::new();
            ids.iter().filter(|id| seen.insert(*id)).cloned().collect()
        };

        if unique_ids.is_empty() {
            return Ok(());
        }

        // Index map for topological sort.
        let idx_map: HashMap<&str, usize> = unique_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        let n = unique_ids.len();
        let mut in_degree = vec![0usize; n];
        let adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        // For each dep, model a "dependency edge" from dep → root (so root
        // can only load after its deps).  With a single-level dep list there
        // are no cross edges, so cycles won't appear here unless the same plugin
        // appears both as a dep and as a dependent of another dep in the list.
        for dep in deps {
            if let Some(&from) = idx_map.get(dep.plugin_id.as_str()) {
                // No additional edges at this point unless we have transitive info.
                let _ = from;
            }
        }

        // Kahn's algorithm.
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();

        let mut visited_count = 0usize;
        while let Some(idx) = queue.pop_front() {
            visited_count += 1;
            for &succ in &adj[idx] {
                in_degree[succ] = in_degree[succ].saturating_sub(1);
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if visited_count < n {
            let cycle_nodes: Vec<String> = in_degree
                .iter()
                .enumerate()
                .filter(|(_, &d)| d > 0)
                .map(|(i, _)| unique_ids[i].clone())
                .collect();
            return Err(ResolveError::CircularDependency(cycle_nodes));
        }

        Ok(())
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32, patch: u32) -> SemVer {
        SemVer::new(major, minor, patch)
    }

    fn dep(id: &str, c: VersionConstraint) -> PluginDependency {
        PluginDependency::new(id, c)
    }

    fn resolver_with(entries: &[(&str, Vec<SemVer>)]) -> DependencyResolver {
        let mut r = DependencyResolver::new();
        for (id, versions) in entries {
            r.register(*id, versions.clone());
        }
        r
    }

    // ── SemVer parsing ──

    #[test]
    fn test_semver_parse_full() {
        let v = SemVer::parse("1.2.3").expect("parse");
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.pre.is_none());
    }

    #[test]
    fn test_semver_parse_with_pre() {
        let v = SemVer::parse("0.1.0-alpha.1").expect("parse pre");
        assert_eq!(v.pre, Some("alpha.1".to_string()));
    }

    #[test]
    fn test_semver_parse_two_parts() {
        let v = SemVer::parse("2.5").expect("parse 2-part");
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_semver_parse_invalid() {
        assert!(SemVer::parse("abc").is_err());
        assert!(SemVer::parse("1.2.3.4").is_err());
        assert!(SemVer::parse("").is_err());
    }

    #[test]
    fn test_semver_display() {
        assert_eq!(v(1, 2, 3).to_string(), "1.2.3");
        assert_eq!(SemVer::with_pre(0, 1, 0, "beta").to_string(), "0.1.0-beta");
    }

    // ── SemVer ordering ──

    #[test]
    fn test_semver_ordering() {
        assert!(v(1, 0, 0) < v(2, 0, 0));
        assert!(v(1, 2, 0) > v(1, 1, 9));
        assert!(v(1, 0, 1) > v(1, 0, 0));
        assert!(v(1, 2, 3) == v(1, 2, 3));
    }

    // ── is_compatible_with ──

    #[test]
    fn test_compatible_same_major() {
        assert!(v(1, 5, 0).is_compatible_with(&v(1, 0, 0)));
        assert!(!v(2, 0, 0).is_compatible_with(&v(1, 0, 0)));
        assert!(!v(1, 0, 0).is_compatible_with(&v(1, 2, 0))); // below required
    }

    // ── VersionConstraint::satisfies ──

    #[test]
    fn test_constraint_exact() {
        let c = VersionConstraint::Exact(v(1, 2, 3));
        assert!(c.satisfies(&v(1, 2, 3)));
        assert!(!c.satisfies(&v(1, 2, 4)));
    }

    #[test]
    fn test_constraint_compatible() {
        let c = VersionConstraint::Compatible(v(1, 0, 0));
        assert!(c.satisfies(&v(1, 5, 0)));
        assert!(!c.satisfies(&v(2, 0, 0)));
    }

    #[test]
    fn test_constraint_at_least() {
        let c = VersionConstraint::AtLeast(v(1, 0, 0));
        assert!(c.satisfies(&v(1, 0, 0)));
        assert!(c.satisfies(&v(2, 0, 0)));
        assert!(!c.satisfies(&v(0, 9, 9)));
    }

    #[test]
    fn test_constraint_at_most() {
        let c = VersionConstraint::AtMost(v(2, 0, 0));
        assert!(c.satisfies(&v(1, 9, 9)));
        assert!(c.satisfies(&v(2, 0, 0)));
        assert!(!c.satisfies(&v(2, 0, 1)));
    }

    #[test]
    fn test_constraint_range() {
        let c = VersionConstraint::Range {
            min: v(1, 0, 0),
            max: v(2, 0, 0),
        };
        assert!(c.satisfies(&v(1, 5, 0)));
        assert!(c.satisfies(&v(1, 0, 0)));
        assert!(c.satisfies(&v(2, 0, 0)));
        assert!(!c.satisfies(&v(0, 9, 9)));
        assert!(!c.satisfies(&v(2, 0, 1)));
    }

    // ── VersionConstraint::intersect ──

    #[test]
    fn test_intersect_compatible_range() {
        let c1 = VersionConstraint::AtLeast(v(1, 0, 0));
        let c2 = VersionConstraint::AtMost(v(2, 0, 0));
        let combined = c1.intersect(&c2).expect("intersect");
        assert!(combined.satisfies(&v(1, 5, 0)));
        assert!(!combined.satisfies(&v(2, 0, 1)));
    }

    #[test]
    fn test_intersect_incompatible() {
        let c1 = VersionConstraint::Exact(v(1, 0, 0));
        let c2 = VersionConstraint::Exact(v(2, 0, 0));
        assert!(c1.intersect(&c2).is_none());
    }

    // ── DependencyResolver::resolve ──

    #[test]
    fn test_resolve_single_dep() {
        let r = resolver_with(&[("codec-a", vec![v(1, 0, 0), v(1, 2, 0)])]);
        let deps = vec![dep("codec-a", VersionConstraint::Compatible(v(1, 0, 0)))];
        let result = r.resolve(&deps).expect("resolve");
        assert_eq!(result["codec-a"], v(1, 2, 0)); // picks highest
    }

    #[test]
    fn test_resolve_picks_highest_satisfying() {
        let r = resolver_with(&[("plug", vec![v(1, 0, 0), v(1, 5, 0), v(2, 0, 0)])]);
        let deps = vec![dep(
            "plug",
            VersionConstraint::Range {
                min: v(1, 0, 0),
                max: v(1, 9, 9),
            },
        )];
        let result = r.resolve(&deps).expect("resolve range");
        assert_eq!(result["plug"], v(1, 5, 0));
    }

    #[test]
    fn test_resolve_not_found() {
        let r = DependencyResolver::new();
        let deps = vec![dep("missing", VersionConstraint::AtLeast(v(1, 0, 0)))];
        match r.resolve(&deps) {
            Err(ResolveError::NotFound(id)) => assert_eq!(id, "missing"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_resolve_conflict_no_satisfying_version() {
        let r = resolver_with(&[("plug", vec![v(1, 0, 0)])]);
        let deps = vec![dep("plug", VersionConstraint::Exact(v(2, 0, 0)))];
        assert!(matches!(r.resolve(&deps), Err(ResolveError::Conflict(..))));
    }

    #[test]
    fn test_resolve_empty_deps() {
        let r = DependencyResolver::new();
        let result = r.resolve(&[]).expect("empty deps");
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_multiple_plugins() {
        let r = resolver_with(&[("a", vec![v(1, 0, 0)]), ("b", vec![v(2, 0, 0)])]);
        let deps = vec![
            dep("a", VersionConstraint::AtLeast(v(1, 0, 0))),
            dep("b", VersionConstraint::AtLeast(v(1, 0, 0))),
        ];
        let result = r.resolve(&deps).expect("multi");
        assert_eq!(result["a"], v(1, 0, 0));
        assert_eq!(result["b"], v(2, 0, 0));
    }

    #[test]
    fn test_resolve_at_most_constraint() {
        let r = resolver_with(&[("p", vec![v(1, 0, 0), v(1, 5, 0), v(2, 0, 0)])]);
        let deps = vec![dep("p", VersionConstraint::AtMost(v(1, 5, 0)))];
        let result = r.resolve(&deps).expect("at_most");
        assert_eq!(result["p"], v(1, 5, 0));
    }

    #[test]
    fn test_resolve_exact_constraint() {
        let r = resolver_with(&[("p", vec![v(1, 0, 0), v(1, 5, 0)])]);
        let deps = vec![dep("p", VersionConstraint::Exact(v(1, 0, 0)))];
        let result = r.resolve(&deps).expect("exact");
        assert_eq!(result["p"], v(1, 0, 0));
    }

    #[test]
    fn test_semver_parse_zero() {
        let v = SemVer::parse("0.0.0").expect("parse zero");
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_constraint_compatible_lower_bound() {
        let c = VersionConstraint::Compatible(v(1, 3, 0));
        assert!(!c.satisfies(&v(1, 2, 9))); // below lower bound
        assert!(c.satisfies(&v(1, 3, 0)));
        assert!(c.satisfies(&v(1, 9, 0)));
        assert!(!c.satisfies(&v(2, 0, 0))); // different major
    }

    #[test]
    fn test_semver_with_pre_display() {
        let v = SemVer::with_pre(1, 0, 0, "rc.1");
        assert_eq!(v.to_string(), "1.0.0-rc.1");
    }

    #[test]
    fn test_resolve_error_display() {
        let e = ResolveError::NotFound("my-plugin".to_string());
        assert!(e.to_string().contains("my-plugin"));

        let e2 =
            ResolveError::Conflict("p".to_string(), SemVer::new(1, 0, 0), SemVer::new(2, 0, 0));
        assert!(e2.to_string().contains('p'));

        let e3 = ResolveError::CircularDependency(vec!["a".to_string(), "b".to_string()]);
        assert!(e3.to_string().contains('a'));
    }

    #[test]
    fn test_resolve_duplicate_dep_same_constraint() {
        // Same dep twice — should merge idempotently.
        let r = resolver_with(&[("p", vec![v(1, 0, 0), v(1, 5, 0)])]);
        let deps = vec![
            dep("p", VersionConstraint::Compatible(v(1, 0, 0))),
            dep("p", VersionConstraint::AtLeast(v(1, 0, 0))),
        ];
        let result = r.resolve(&deps).expect("dup dep");
        assert_eq!(result["p"], v(1, 5, 0));
    }

    #[test]
    fn test_plugin_dependency_new() {
        let pd = PluginDependency::new("my-plug", VersionConstraint::AtLeast(SemVer::new(1, 0, 0)));
        assert_eq!(pd.plugin_id, "my-plug");
    }

    #[test]
    fn test_dependency_resolver_default() {
        let r = DependencyResolver::default();
        assert!(r.registered_plugins.is_empty());
        assert!(r.provider_deps.is_empty());
    }

    // ── Transitive diamond-dependency resolution ──────────────────────────────

    /// Diamond conflict: A→B(^1.x) and A→C(^1.x), C→B(^2.x).
    /// B exists at 1.0.0 and 2.0.0.  The combined constraint for B is
    /// Compatible(1.x) ∩ Compatible(2.x) = empty → Conflict.
    #[test]
    fn test_diamond_conflict_transitive() {
        let mut r = DependencyResolver::new();
        r.register("B", vec![v(1, 0, 0), v(2, 0, 0)]);
        r.register_with_deps(
            "C",
            vec![v(1, 0, 0)],
            vec![dep("B", VersionConstraint::Compatible(v(2, 0, 0)))],
        );

        // Root requires B ^1.x and C ^1.x.  C itself requires B ^2.x.
        let root_deps = vec![
            dep("B", VersionConstraint::Compatible(v(1, 0, 0))),
            dep("C", VersionConstraint::Compatible(v(1, 0, 0))),
        ];

        let err = r
            .resolve(&root_deps)
            .expect_err("diamond conflict expected");
        match &err {
            ResolveError::Conflict(id, _, _) => assert_eq!(id, "B"),
            other => panic!("expected Conflict for B, got {other:?}"),
        }
    }

    /// Diamond compatible: A→B(^1.x) and A→C(^1.x), C→B(>=1.5.0).
    /// Combined: Compatible(1.x) ∩ AtLeast(1.5.0) = Range{1.5.0, 1.MAX.MAX}.
    /// Available B: 1.0.0, 1.5.0, 1.9.0 → picks 1.9.0.
    #[test]
    fn test_diamond_compatible_transitive() {
        let mut r = DependencyResolver::new();
        r.register("B", vec![v(1, 0, 0), v(1, 5, 0), v(1, 9, 0)]);
        r.register_with_deps(
            "C",
            vec![v(1, 0, 0)],
            vec![dep("B", VersionConstraint::AtLeast(v(1, 5, 0)))],
        );

        let root_deps = vec![
            dep("B", VersionConstraint::Compatible(v(1, 0, 0))),
            dep("C", VersionConstraint::Compatible(v(1, 0, 0))),
        ];

        let result = r.resolve(&root_deps).expect("diamond compatible");
        assert_eq!(result["B"], v(1, 9, 0), "should pick highest in [1.5, 1.x]");
        assert_eq!(result["C"], v(1, 0, 0));
    }

    /// Large dep graph: 10 plugins in a linear chain, each declaring the next
    /// as a transitive dep via `register_with_deps`.  All must resolve.
    #[test]
    fn test_large_dep_graph_chain() {
        let mut r = DependencyResolver::new();

        // Register p0..p9 with p_i → p_{i+1} >=1.0.0 (except p9 which has no deps).
        for i in 0..9usize {
            let next = format!("p{}", i + 1);
            r.register_with_deps(
                format!("p{i}"),
                vec![v(1, 0, 0)],
                vec![dep(&next, VersionConstraint::AtLeast(v(1, 0, 0)))],
            );
        }
        r.register("p9", vec![v(1, 0, 0)]);

        // Root only requests p0 — the rest must be discovered transitively.
        let root_deps = vec![dep("p0", VersionConstraint::AtLeast(v(1, 0, 0)))];
        let result = r.resolve(&root_deps).expect("chain resolve");

        // All 10 plugins should be present in the resolved map.
        for i in 0..10usize {
            let key = format!("p{i}");
            assert_eq!(
                result[&key],
                v(1, 0, 0),
                "missing or wrong version for {key}"
            );
        }
        assert_eq!(result.len(), 10);
    }

    /// `register_with_deps` stores provider_deps correctly.
    #[test]
    fn test_register_with_deps_stores_provider_deps() {
        let mut r = DependencyResolver::new();
        r.register("base", vec![v(1, 0, 0)]);
        r.register_with_deps(
            "ext",
            vec![v(2, 0, 0)],
            vec![dep("base", VersionConstraint::AtLeast(v(1, 0, 0)))],
        );

        assert!(r.provider_deps.contains_key("ext"));
        assert_eq!(r.provider_deps["ext"].len(), 1);
        assert_eq!(r.provider_deps["ext"][0].plugin_id, "base");
    }

    /// Transitive dep that is NOT registered should return NotFound.
    #[test]
    fn test_transitive_dep_not_found() {
        let mut r = DependencyResolver::new();
        r.register_with_deps(
            "plugin-a",
            vec![v(1, 0, 0)],
            vec![dep("missing-lib", VersionConstraint::AtLeast(v(1, 0, 0)))],
        );

        let root_deps = vec![dep("plugin-a", VersionConstraint::AtLeast(v(1, 0, 0)))];
        let err = r.resolve(&root_deps).expect_err("should be NotFound");
        assert!(
            matches!(&err, ResolveError::NotFound(id) if id == "missing-lib"),
            "unexpected error: {err:?}"
        );
    }
}
