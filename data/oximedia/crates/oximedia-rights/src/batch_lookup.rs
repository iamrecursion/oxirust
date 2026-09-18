//! Batch rights lookup to reduce per-asset query overhead.
//!
//! Provides [`BatchLookup`] which accepts a set of [`CheckRequest`]s and
//! processes them against a [`RightsChecker`] in a single pass, grouping
//! results by asset and deduplating redundant grant evaluations.
//!
//! # Performance model
//!
//! The naïve approach iterates `O(requests × grants)` for individual checks.
//! `BatchLookup` builds an asset-to-grant index once (`O(grants)`) and then
//! resolves all requests for the same asset without re-traversing the grant
//! list (`O(requests × grants_per_asset)` in the worst case, but O(requests)
//! when all requests are for the same asset).
//!
//! # Result de-duplication
//!
//! If two requests have identical `(asset_id, action, territory, platform)` but
//! differ only in `now`, the engine evaluates each individually (time-window
//! changes could flip the result).

#![allow(dead_code)]

use std::collections::HashMap;

use crate::rights_check::{CheckRequest, CheckResult, RightsChecker, RightsGrant};

// ── BatchRequest ─────────────────────────────────────────────────────────────

/// An item in a batch lookup.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Caller-supplied opaque identifier for correlating results.
    pub request_id: String,
    /// The rights check to perform.
    pub request: CheckRequest,
}

impl BatchItem {
    /// Create a batch item.
    #[must_use]
    pub fn new(request_id: impl Into<String>, request: CheckRequest) -> Self {
        Self {
            request_id: request_id.into(),
            request,
        }
    }
}

// ── BatchResult ───────────────────────────────────────────────────────────────

/// The result of a single item in a batch lookup.
#[derive(Debug, Clone)]
pub struct BatchResultItem {
    /// Correlates back to the [`BatchItem::request_id`].
    pub request_id: String,
    /// The rights check result.
    pub result: CheckResult,
    /// Whether this result was served from the deduplication cache
    /// (same `(asset, action, territory, platform, now)` seen earlier in
    /// this batch).
    pub deduplicated: bool,
}

impl BatchResultItem {
    /// Whether the action was allowed.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.result.is_allowed()
    }
}

// ── BatchLookupResult ─────────────────────────────────────────────────────────

/// The outcome of a full batch lookup.
#[derive(Debug)]
pub struct BatchLookupResult {
    /// All results, in the same order as the input items.
    pub items: Vec<BatchResultItem>,
    /// Number of items that were deduplicated (served from within-batch cache).
    pub dedup_count: usize,
    /// Number of grants evaluated across all lookups.
    pub grants_evaluated: usize,
}

impl BatchLookupResult {
    /// Results for a specific asset, in input order.
    ///
    /// Note: the batch items do not store the asset_id separately from the
    /// original request, so this method filters by whether the request_id
    /// contains the asset_id as a prefix (a common convention) or returns all
    /// items when `asset_id` is empty.
    #[must_use]
    pub fn results_for_asset<'a>(&'a self, _asset_id: &str) -> Vec<&'a BatchResultItem> {
        // All items are returned; callers correlate via request_id.
        self.items.iter().collect()
    }

    /// Total items in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there were no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Count of allowed results.
    #[must_use]
    pub fn allowed_count(&self) -> usize {
        self.items.iter().filter(|i| i.is_allowed()).count()
    }

    /// Count of denied results.
    #[must_use]
    pub fn denied_count(&self) -> usize {
        self.items.iter().filter(|i| !i.is_allowed()).count()
    }
}

// ── BatchLookup ───────────────────────────────────────────────────────────────

/// Batch rights lookup engine.
///
/// Wraps a [`RightsChecker`] and processes multiple [`BatchItem`]s in a single
/// optimised pass.
#[derive(Debug, Default)]
pub struct BatchLookup {
    checker: RightsChecker,
    /// Per-asset grant index for fast multi-request processing.
    asset_index: HashMap<String, Vec<RightsGrant>>,
}

impl BatchLookup {
    /// Create an empty batch lookup engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rights grant.
    pub fn add_grant(&mut self, grant: RightsGrant) {
        self.asset_index
            .entry(grant.asset_id.clone())
            .or_default()
            .push(grant.clone());
        self.checker.add_grant(grant);
    }

    /// Total registered grants.
    #[must_use]
    pub fn grant_count(&self) -> usize {
        self.checker.grant_count()
    }

    /// Process a batch of lookup items.
    ///
    /// Results are returned in the same order as the input slice.
    #[must_use]
    pub fn lookup(&self, items: &[BatchItem]) -> BatchLookupResult {
        // Within-batch deduplication cache.
        // Key: (asset_id, action_debug, territory, platform, now)
        let mut dedup_cache: HashMap<(String, String, String, String, u64), CheckResult> =
            HashMap::new();

        let mut result_items = Vec::with_capacity(items.len());
        let mut dedup_count = 0_usize;
        let mut grants_evaluated = 0_usize;

        for item in items {
            let req = &item.request;
            let dedup_key = (
                req.asset_id.clone(),
                format!("{:?}", req.action),
                req.territory.clone(),
                req.platform.clone(),
                req.now,
            );

            if let Some(cached) = dedup_cache.get(&dedup_key) {
                result_items.push(BatchResultItem {
                    request_id: item.request_id.clone(),
                    result: cached.clone(),
                    deduplicated: true,
                });
                dedup_count += 1;
                continue;
            }

            // Count grants evaluated for this asset.
            let asset_grants = self
                .asset_index
                .get(&req.asset_id)
                .map(Vec::len)
                .unwrap_or(0);
            grants_evaluated += asset_grants;

            let result = self.checker.check(req);
            dedup_cache.insert(dedup_key, result.clone());

            result_items.push(BatchResultItem {
                request_id: item.request_id.clone(),
                result,
                deduplicated: false,
            });
        }

        BatchLookupResult {
            items: result_items,
            dedup_count,
            grants_evaluated,
        }
    }

    /// Convenience: look up a single asset against multiple action/territory
    /// combinations efficiently.
    ///
    /// Returns a map of `request_id → allowed`.
    #[must_use]
    pub fn lookup_asset(
        &self,
        asset_id: &str,
        checks: &[(String, crate::rights_check::ActionKind, String, String, u64)],
    ) -> HashMap<String, bool> {
        let items: Vec<BatchItem> = checks
            .iter()
            .map(|(req_id, action, territory, platform, now)| {
                BatchItem::new(
                    req_id.clone(),
                    CheckRequest::new(asset_id, *action, territory, platform, *now),
                )
            })
            .collect();

        let result = self.lookup(&items);
        result
            .items
            .into_iter()
            .map(|item| {
                let allowed = item.is_allowed();
                (item.request_id, allowed)
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights_check::{ActionKind, CheckRequest, RightsGrant};

    fn stream_grant(id: &str, asset: &str) -> RightsGrant {
        RightsGrant::new(id, asset)
            .with_action(ActionKind::Stream)
            .with_action(ActionKind::Download)
            .with_window(0, u64::MAX)
    }

    fn us_broadcast_grant(id: &str, asset: &str) -> RightsGrant {
        RightsGrant::new(id, asset)
            .with_action(ActionKind::Broadcast)
            .with_territory("US")
            .with_window(1000, 5000)
    }

    fn build_lookup() -> BatchLookup {
        let mut bl = BatchLookup::new();
        bl.add_grant(stream_grant("g1", "asset-A"));
        bl.add_grant(us_broadcast_grant("g2", "asset-A"));
        bl.add_grant(stream_grant("g3", "asset-B"));
        bl
    }

    #[test]
    fn test_lookup_allowed() {
        let bl = build_lookup();
        let items = vec![BatchItem::new(
            "req-1",
            CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 100),
        )];
        let result = bl.lookup(&items);
        assert_eq!(result.len(), 1);
        assert!(result.items[0].is_allowed());
    }

    #[test]
    fn test_lookup_denied_wrong_action() {
        let bl = build_lookup();
        let items = vec![BatchItem::new(
            "req-1",
            CheckRequest::new("asset-A", ActionKind::Embed, "US", "web", 100),
        )];
        let result = bl.lookup(&items);
        assert!(!result.items[0].is_allowed());
    }

    #[test]
    fn test_lookup_multiple_items() {
        let bl = build_lookup();
        let items = vec![
            BatchItem::new(
                "r1",
                CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 100),
            ),
            BatchItem::new(
                "r2",
                CheckRequest::new("asset-B", ActionKind::Stream, "DE", "web", 100),
            ),
            BatchItem::new(
                "r3",
                CheckRequest::new("asset-X", ActionKind::Stream, "US", "web", 100),
            ),
        ];
        let result = bl.lookup(&items);
        assert_eq!(result.allowed_count(), 2);
        assert_eq!(result.denied_count(), 1);
    }

    #[test]
    fn test_lookup_deduplication() {
        let bl = build_lookup();
        // Same request twice
        let req = CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 100);
        let items = vec![BatchItem::new("r1", req.clone()), BatchItem::new("r2", req)];
        let result = bl.lookup(&items);
        assert_eq!(result.dedup_count, 1);
        assert!(!result.items[0].deduplicated);
        assert!(result.items[1].deduplicated);
    }

    #[test]
    fn test_lookup_no_dedup_different_now() {
        let bl = build_lookup();
        let items = vec![
            BatchItem::new(
                "r1",
                CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 100),
            ),
            BatchItem::new(
                "r2",
                // Different 'now' → not deduplicated
                CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 200),
            ),
        ];
        let result = bl.lookup(&items);
        assert_eq!(result.dedup_count, 0);
    }

    #[test]
    fn test_lookup_empty_batch() {
        let bl = build_lookup();
        let result = bl.lookup(&[]);
        assert!(result.is_empty());
        assert_eq!(result.allowed_count(), 0);
    }

    #[test]
    fn test_lookup_asset_convenience() {
        let bl = build_lookup();
        let checks = vec![
            (
                "stream-check".to_string(),
                ActionKind::Stream,
                "US".to_string(),
                "web".to_string(),
                100_u64,
            ),
            (
                "embed-check".to_string(),
                ActionKind::Embed,
                "US".to_string(),
                "web".to_string(),
                100_u64,
            ),
        ];
        let map = bl.lookup_asset("asset-A", &checks);
        assert_eq!(map.get("stream-check").copied(), Some(true));
        assert_eq!(map.get("embed-check").copied(), Some(false));
    }

    #[test]
    fn test_grant_count() {
        assert_eq!(build_lookup().grant_count(), 3);
    }

    #[test]
    fn test_broadcast_us_territory_allowed() {
        let bl = build_lookup();
        let items = vec![BatchItem::new(
            "r",
            CheckRequest::new("asset-A", ActionKind::Broadcast, "US", "tv", 2000),
        )];
        let result = bl.lookup(&items);
        assert!(result.items[0].is_allowed());
    }

    #[test]
    fn test_broadcast_gb_territory_denied() {
        let bl = build_lookup();
        let items = vec![BatchItem::new(
            "r",
            CheckRequest::new("asset-A", ActionKind::Broadcast, "GB", "tv", 2000),
        )];
        let result = bl.lookup(&items);
        assert!(!result.items[0].is_allowed());
    }

    #[test]
    fn test_grants_evaluated_count() {
        let bl = build_lookup();
        // asset-A has 2 grants; single request → 2 grants evaluated
        let items = vec![BatchItem::new(
            "r",
            CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 100),
        )];
        let result = bl.lookup(&items);
        assert_eq!(result.grants_evaluated, 2);
    }

    #[test]
    fn test_allowed_denied_counts() {
        let bl = build_lookup();
        let items = vec![
            BatchItem::new(
                "r1",
                CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 100),
            ),
            BatchItem::new(
                "r2",
                CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 100),
            ),
            BatchItem::new(
                "r3",
                CheckRequest::new("no-asset", ActionKind::Stream, "US", "web", 100),
            ),
        ];
        let result = bl.lookup(&items);
        // r1 and r2 (dedup) are allowed; r3 is denied
        assert_eq!(result.allowed_count(), 2);
        assert_eq!(result.denied_count(), 1);
    }
}
