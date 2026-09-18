//! Proxy cache warming: pre-generate proxies for frequently/recently accessed originals.
//!
//! Editing latency is dominated by the moment an editor scrubs to a clip whose
//! proxy is *not* yet on disk. [`ProxyCacheWarmer`] amortizes that cost by
//! ranking originals with a transparent scoring heuristic and proactively
//! queuing the highest-value proxies for generation *before* they are demanded.
//!
//! # Scoring
//!
//! Each candidate is scored as
//!
//! ```text
//! score = w_freq * hit_count
//!       + w_recency * recency_decay(age)
//!       + codec_pref_bonus
//! ```
//!
//! where `age = now - last_access` (saturating, in seconds) and
//! [`WarmingConfig::recency_decay`] is `1 / (1 + age / tau)` — a smooth,
//! strictly-decreasing function of age. The clock is **never** read inside
//! scoring: callers pass `now` explicitly so the heuristic is fully
//! deterministic and unit-testable.
//!
//! # Why a self-contained candidate type
//!
//! The on-disk [`crate::registry::ProxyEntry`] is a *serialized* record and does
//! not carry per-asset access telemetry. Rather than break that schema, the
//! warmer consumes a lightweight [`WarmCandidate`] snapshot that the caller
//! populates from whatever access metadata is available — the [`CacheManager`]
//! (`access_count` / `last_access`), [`crate::proxy_cache`] (`hit_count` /
//! `last_access_ms`), or, as a coarse fallback, a registry entry's `created_at`.
//! [`WarmCandidate::from_registry_record`] wires the common registry path.
//!
//! [`CacheManager`]: crate::cache::CacheManager
//!
//! # Example
//!
//! ```
//! use oximedia_proxy::cache::{ProxyCacheWarmer, WarmingConfig, WarmCandidate};
//! use oximedia_proxy::spec::ProxyCodec;
//! use std::path::PathBuf;
//!
//! let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
//! let now = 10_000u64;
//! let candidates = vec![
//!     WarmCandidate::new(PathBuf::from("/media/a.mov"), 9, now - 5, ProxyCodec::H264),
//!     WarmCandidate::new(PathBuf::from("/media/b.mov"), 1, now - 9_000, ProxyCodec::H264),
//! ];
//! let selected = warmer.select_candidates(&candidates, now);
//! assert_eq!(selected[0].original_path, PathBuf::from("/media/a.mov"));
//!
//! // Injected generation closure — unit code performs no real transcode.
//! let result = warmer.warm(&selected, |_candidate| Ok(()));
//! assert_eq!(result.generated, selected.len());
//! ```

use crate::registry::RegistryRecord;
use crate::spec::ProxyCodec;
use std::path::PathBuf;

/// Default frequency weight (`w_freq`): each prior access adds this to the score.
pub const DEFAULT_FREQ_WEIGHT: f64 = 1.0;

/// Default recency weight (`w_recency`): a brand-new access adds up to this much.
pub const DEFAULT_RECENCY_WEIGHT: f64 = 5.0;

/// Default codec-preference bonus added when a candidate matches `codec_pref`.
pub const DEFAULT_CODEC_PREF_BONUS: f64 = 2.0;

/// Default recency time-constant `tau`, in seconds (one hour).
///
/// At `age == tau` the recency term has decayed to exactly half its peak.
pub const DEFAULT_RECENCY_TAU_SECS: f64 = 3_600.0;

/// Default upper bound on the number of proxies warmed in a single pass.
pub const DEFAULT_MAX_CANDIDATES: usize = 16;

/// Configuration controlling how candidates are scored and how many are warmed.
///
/// All fields are public for ergonomic struct-update construction; prefer the
/// `with_*` builders for clamped, intention-revealing changes.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmingConfig {
    /// Weight applied to the access/hit count (`w_freq`).
    pub freq_weight: f64,
    /// Weight applied to the recency-decay term (`w_recency`).
    pub recency_weight: f64,
    /// Bonus added when a candidate's codec equals [`Self::codec_pref`].
    pub codec_pref_bonus: f64,
    /// Time-constant `tau` (seconds) for the recency decay; clamped to be positive.
    pub recency_tau_secs: f64,
    /// Preferred proxy codec; candidates matching it receive [`Self::codec_pref_bonus`].
    pub codec_pref: Option<ProxyCodec>,
    /// Maximum number of candidates to warm in a single [`ProxyCacheWarmer::select_candidates`] /
    /// [`ProxyCacheWarmer::warm`] pass.
    pub max_candidates: usize,
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            freq_weight: DEFAULT_FREQ_WEIGHT,
            recency_weight: DEFAULT_RECENCY_WEIGHT,
            codec_pref_bonus: DEFAULT_CODEC_PREF_BONUS,
            recency_tau_secs: DEFAULT_RECENCY_TAU_SECS,
            codec_pref: None,
            max_candidates: DEFAULT_MAX_CANDIDATES,
        }
    }
}

impl WarmingConfig {
    /// Create a configuration with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the frequency weight (`w_freq`).
    #[must_use]
    pub fn with_freq_weight(mut self, weight: f64) -> Self {
        self.freq_weight = weight;
        self
    }

    /// Set the recency weight (`w_recency`).
    #[must_use]
    pub fn with_recency_weight(mut self, weight: f64) -> Self {
        self.recency_weight = weight;
        self
    }

    /// Set the recency time-constant `tau`, in seconds.
    ///
    /// Non-positive values are clamped to a tiny positive epsilon so the decay
    /// function can never divide by zero.
    #[must_use]
    pub fn with_recency_tau_secs(mut self, tau: f64) -> Self {
        self.recency_tau_secs = tau;
        self
    }

    /// Set the preferred codec and the bonus awarded for matching it.
    #[must_use]
    pub fn with_codec_pref(mut self, codec: ProxyCodec, bonus: f64) -> Self {
        self.codec_pref = Some(codec);
        self.codec_pref_bonus = bonus;
        self
    }

    /// Set the maximum number of candidates warmed per pass.
    #[must_use]
    pub fn with_max_candidates(mut self, max: usize) -> Self {
        self.max_candidates = max;
        self
    }

    /// Effective (always-positive) recency time-constant used by the decay.
    fn effective_tau(&self) -> f64 {
        if self.recency_tau_secs.is_finite() && self.recency_tau_secs > 0.0 {
            self.recency_tau_secs
        } else {
            f64::EPSILON
        }
    }

    /// Recency decay as a function of `age` (seconds): `1 / (1 + age / tau)`.
    ///
    /// Strictly decreasing in `age`, equal to `1.0` at `age == 0`, and
    /// asymptotically approaching `0.0` as `age` grows. Negative ages (which
    /// should not occur for a saturating `now - last_access`) are treated as `0`.
    #[must_use]
    pub fn recency_decay(&self, age_secs: f64) -> f64 {
        let age = if age_secs.is_finite() && age_secs > 0.0 {
            age_secs
        } else {
            0.0
        };
        1.0 / (1.0 + age / self.effective_tau())
    }
}

/// A single original eligible for proxy warming, with its access telemetry.
///
/// This is a *snapshot* the caller assembles from live access metadata; the
/// warmer treats it as immutable input and never mutates persistent state.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmCandidate {
    /// Path to the original (high-resolution) source asset.
    pub original_path: PathBuf,
    /// Number of times this original (or its proxy) has been accessed.
    pub hit_count: u64,
    /// Unix-seconds timestamp of the most recent access.
    pub last_access: u64,
    /// Codec the warmed proxy would use (drives the codec-preference bonus).
    pub codec: ProxyCodec,
}

impl WarmCandidate {
    /// Create a candidate from explicit access telemetry.
    #[must_use]
    pub fn new(
        original_path: PathBuf,
        hit_count: u64,
        last_access: u64,
        codec: ProxyCodec,
    ) -> Self {
        Self {
            original_path,
            hit_count,
            last_access,
            codec,
        }
    }

    /// Derive a candidate from a [`RegistryRecord`] plus externally-tracked
    /// access stats.
    ///
    /// The codec is taken from the record's first proxy entry (the registry's
    /// canonical variant); if the record has no proxies — and therefore nothing
    /// to inform a codec choice — this returns `None`. Use [`Self::new`] when the
    /// codec is known independently.
    #[must_use]
    pub fn from_registry_record(
        record: &RegistryRecord,
        hit_count: u64,
        last_access: u64,
    ) -> Option<Self> {
        let codec = record.proxies.first()?.spec.codec.clone();
        Some(Self {
            original_path: record.original_path.clone(),
            hit_count,
            last_access,
            codec,
        })
    }
}

/// Outcome of a [`ProxyCacheWarmer::warm`] pass.
///
/// Invariants: `queued == generated + failed` (every queued candidate is either
/// generated or fails), and `queued` equals the number of candidates passed in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarmingResult {
    /// Number of candidates queued for generation in this pass.
    pub queued: usize,
    /// Number of proxies the generation closure reported as successfully produced.
    pub generated: usize,
    /// Number of proxies the generation closure reported as failed.
    pub failed: usize,
}

/// Pre-generates proxies for frequently / recently accessed originals.
///
/// Construct with a [`WarmingConfig`], rank originals via
/// [`Self::select_candidates`], then drive generation through
/// [`Self::warm`] with an injected closure. The warmer itself performs no I/O
/// and runs no encoders — that responsibility belongs to the closure, which
/// keeps the scoring/selection logic trivially unit-testable.
#[derive(Debug, Clone, Default)]
pub struct ProxyCacheWarmer {
    config: WarmingConfig,
}

impl ProxyCacheWarmer {
    /// Create a warmer with the supplied configuration.
    #[must_use]
    pub fn new(config: WarmingConfig) -> Self {
        Self { config }
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &WarmingConfig {
        &self.config
    }

    /// Score a single candidate as of `now` (Unix seconds).
    ///
    /// `now` is supplied by the caller; the wall clock is never read here, so
    /// scoring is deterministic. Ages are computed with a saturating subtraction
    /// so a `last_access` in the future cannot produce a negative age.
    #[must_use]
    pub fn score(&self, candidate: &WarmCandidate, now: u64) -> f64 {
        let age_secs = now.saturating_sub(candidate.last_access) as f64;
        let freq_term = self.config.freq_weight * candidate.hit_count as f64;
        let recency_term = self.config.recency_weight * self.config.recency_decay(age_secs);
        let codec_term = match &self.config.codec_pref {
            Some(pref) if *pref == candidate.codec => self.config.codec_pref_bonus,
            _ => 0.0,
        };
        freq_term + recency_term + codec_term
    }

    /// Select the top-scoring candidates, highest score first.
    ///
    /// The number returned is capped at [`WarmingConfig::max_candidates`]. Ties
    /// in score are broken deterministically by ascending `original_path`, so the
    /// ordering is stable across runs regardless of input order or `HashMap`
    /// iteration nondeterminism. An empty input yields an empty result.
    #[must_use]
    pub fn select_candidates(&self, candidates: &[WarmCandidate], now: u64) -> Vec<WarmCandidate> {
        self.select_candidates_limited(candidates, now, self.config.max_candidates)
    }

    /// Like [`Self::select_candidates`] but with an explicit `limit` overriding
    /// the configured maximum.
    #[must_use]
    pub fn select_candidates_limited(
        &self,
        candidates: &[WarmCandidate],
        now: u64,
        limit: usize,
    ) -> Vec<WarmCandidate> {
        if limit == 0 || candidates.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(f64, &WarmCandidate)> =
            candidates.iter().map(|c| (self.score(c, now), c)).collect();

        // Descending by score; deterministic tie-break on path keeps selection
        // stable even though scores are floats and input order is arbitrary.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.original_path.cmp(&b.1.original_path))
        });

        scored
            .into_iter()
            .take(limit)
            .map(|(_, c)| c.clone())
            .collect()
    }

    /// Warm the supplied candidates by invoking `generate_fn` for each.
    ///
    /// The closure is the *only* place real work happens; it receives one
    /// candidate at a time and returns `Ok(())` on success or `Err(_)` on
    /// failure. Failures are counted, not propagated — one unfetchable original
    /// must not abort warming of the rest. The returned [`WarmingResult`]
    /// satisfies `queued == generated + failed == candidates.len()`.
    pub fn warm<F>(&self, candidates: &[WarmCandidate], mut generate_fn: F) -> WarmingResult
    where
        F: FnMut(&WarmCandidate) -> crate::Result<()>,
    {
        let mut result = WarmingResult {
            queued: candidates.len(),
            generated: 0,
            failed: 0,
        };

        for candidate in candidates {
            match generate_fn(candidate) {
                Ok(()) => result.generated += 1,
                Err(_) => result.failed += 1,
            }
        }

        result
    }

    /// Convenience: select then warm in a single call.
    ///
    /// Equivalent to `self.warm(&self.select_candidates(candidates, now), generate_fn)`,
    /// returning both the ranked selection and the warming outcome.
    pub fn select_and_warm<F>(
        &self,
        candidates: &[WarmCandidate],
        now: u64,
        generate_fn: F,
    ) -> (Vec<WarmCandidate>, WarmingResult)
    where
        F: FnMut(&WarmCandidate) -> crate::Result<()>,
    {
        let selected = self.select_candidates(candidates, now);
        let result = self.warm(&selected, generate_fn);
        (selected, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ProxyEntry;
    use crate::spec::{ProxyResolutionMode, ProxySpec};
    use std::cell::Cell;
    use std::path::Path;

    fn candidate(path: &str, hits: u64, last_access: u64, codec: ProxyCodec) -> WarmCandidate {
        WarmCandidate::new(PathBuf::from(path), hits, last_access, codec)
    }

    fn make_spec(codec: ProxyCodec) -> ProxySpec {
        ProxySpec::new(
            "Test",
            ProxyResolutionMode::ScaleFactor(0.25),
            codec,
            2_000_000,
        )
    }

    #[test]
    fn top_k_selection_orders_by_score_descending() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 10_000u64;
        // `high` has both more hits AND a more recent access than `low`/`mid`.
        let candidates = vec![
            candidate("/m/low.mov", 1, now - 9_000, ProxyCodec::H264),
            candidate("/m/high.mov", 20, now - 1, ProxyCodec::H264),
            candidate("/m/mid.mov", 5, now - 100, ProxyCodec::H264),
        ];
        let selected = warmer.select_candidates(&candidates, now);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].original_path, PathBuf::from("/m/high.mov"));
        assert_eq!(selected[1].original_path, PathBuf::from("/m/mid.mov"));
        assert_eq!(selected[2].original_path, PathBuf::from("/m/low.mov"));

        // Scores must be monotonically non-increasing in selection order.
        let s0 = warmer.score(&selected[0], now);
        let s1 = warmer.score(&selected[1], now);
        let s2 = warmer.score(&selected[2], now);
        assert!(
            s0 >= s1 && s1 >= s2,
            "scores not descending: {s0} {s1} {s2}"
        );
    }

    #[test]
    fn select_respects_max_candidates_limit() {
        let config = WarmingConfig::default().with_max_candidates(2);
        let warmer = ProxyCacheWarmer::new(config);
        let now = 1_000u64;
        let candidates = vec![
            candidate("/m/a.mov", 10, now - 1, ProxyCodec::H264),
            candidate("/m/b.mov", 9, now - 2, ProxyCodec::H264),
            candidate("/m/c.mov", 8, now - 3, ProxyCodec::H264),
            candidate("/m/d.mov", 7, now - 4, ProxyCodec::H264),
        ];
        let selected = warmer.select_candidates(&candidates, now);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].original_path, PathBuf::from("/m/a.mov"));
        assert_eq!(selected[1].original_path, PathBuf::from("/m/b.mov"));
    }

    #[test]
    fn tie_break_is_stable_by_path() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 500u64;
        // Identical telemetry => identical scores; only the path differs.
        let candidates = vec![
            candidate("/m/zzz.mov", 5, now - 10, ProxyCodec::H264),
            candidate("/m/aaa.mov", 5, now - 10, ProxyCodec::H264),
            candidate("/m/mmm.mov", 5, now - 10, ProxyCodec::H264),
        ];
        let selected = warmer.select_candidates(&candidates, now);
        assert_eq!(selected[0].original_path, PathBuf::from("/m/aaa.mov"));
        assert_eq!(selected[1].original_path, PathBuf::from("/m/mmm.mov"));
        assert_eq!(selected[2].original_path, PathBuf::from("/m/zzz.mov"));
    }

    #[test]
    fn recency_decay_strictly_decreases_with_age() {
        let config = WarmingConfig::default();
        let ages = [0.0, 1.0, 60.0, 600.0, 3_600.0, 36_000.0, 360_000.0];
        let mut prev = f64::INFINITY;
        for &age in &ages {
            let d = config.recency_decay(age);
            assert!(
                d < prev,
                "decay not strictly decreasing at age {age}: {d} >= {prev}"
            );
            assert!(
                (0.0..=1.0).contains(&d),
                "decay out of [0,1] at age {age}: {d}"
            );
            prev = d;
        }
        // Anchor values: 1.0 at age 0, exactly 0.5 at age == tau.
        assert!((config.recency_decay(0.0) - 1.0).abs() < 1e-12);
        assert!((config.recency_decay(DEFAULT_RECENCY_TAU_SECS) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn recency_term_dominates_when_frequency_equal() {
        // Two candidates with equal hits: the more-recently-accessed one wins.
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 100_000u64;
        let fresh = candidate("/m/fresh.mov", 3, now - 10, ProxyCodec::H264);
        let stale = candidate("/m/stale.mov", 3, now - 50_000, ProxyCodec::H264);
        assert!(warmer.score(&fresh, now) > warmer.score(&stale, now));
    }

    #[test]
    fn codec_preference_bonus_is_honoured() {
        let config = WarmingConfig::default().with_codec_pref(ProxyCodec::ProRes422Proxy, 100.0);
        let warmer = ProxyCacheWarmer::new(config);
        let now = 1_000u64;
        // `preferred` has FEWER hits but its codec matches the preference; the
        // large bonus must lift it above the otherwise-stronger `other`.
        let preferred = candidate("/m/pref.mov", 1, now - 10, ProxyCodec::ProRes422Proxy);
        let other = candidate("/m/other.mov", 5, now - 10, ProxyCodec::H264);
        assert!(warmer.score(&preferred, now) > warmer.score(&other, now));

        let selected = warmer.select_candidates(&[other, preferred], now);
        assert_eq!(selected[0].original_path, PathBuf::from("/m/pref.mov"));
    }

    #[test]
    fn codec_preference_absent_gives_no_bonus() {
        // Without a configured preference, codec must not affect the score.
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 1_000u64;
        let a = candidate("/m/a.mov", 4, now - 5, ProxyCodec::ProRes422Proxy);
        let b = candidate("/m/b.mov", 4, now - 5, ProxyCodec::H264);
        assert!((warmer.score(&a, now) - warmer.score(&b, now)).abs() < 1e-12);
    }

    #[test]
    fn progress_counters_all_success() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 1_000u64;
        let candidates = vec![
            candidate("/m/a.mov", 3, now - 1, ProxyCodec::H264),
            candidate("/m/b.mov", 2, now - 2, ProxyCodec::H264),
            candidate("/m/c.mov", 1, now - 3, ProxyCodec::H264),
        ];
        let selected = warmer.select_candidates(&candidates, now);
        let result = warmer.warm(&selected, |_| Ok(()));
        assert_eq!(result.queued, selected.len());
        assert_eq!(result.generated, selected.len());
        assert_eq!(result.failed, 0);
        assert_eq!(result.generated + result.failed, result.queued);
    }

    #[test]
    fn progress_counters_with_injected_failures() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 1_000u64;
        let candidates = vec![
            candidate("/m/a.mov", 5, now - 1, ProxyCodec::H264),
            candidate("/m/b.mov", 4, now - 2, ProxyCodec::H264),
            candidate("/m/c.mov", 3, now - 3, ProxyCodec::H264),
            candidate("/m/d.mov", 2, now - 4, ProxyCodec::H264),
        ];
        let selected = warmer.select_candidates(&candidates, now);

        // Fail every other invocation, regardless of which paths land where.
        let toggle = Cell::new(false);
        let result = warmer.warm(&selected, |_candidate| {
            let fail = toggle.get();
            toggle.set(!fail);
            if fail {
                Err(crate::ProxyError::GenerationError("injected".into()))
            } else {
                Ok(())
            }
        });

        assert_eq!(result.queued, 4);
        assert_eq!(result.generated, 2);
        assert_eq!(result.failed, 2);
        assert_eq!(result.generated + result.failed, result.queued);
    }

    #[test]
    fn warm_visits_every_candidate_exactly_once() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 1_000u64;
        let candidates = vec![
            candidate("/m/a.mov", 3, now - 1, ProxyCodec::H264),
            candidate("/m/b.mov", 2, now - 2, ProxyCodec::H264),
        ];
        let count = Cell::new(0usize);
        let result = warmer.warm(&candidates, |_| {
            count.set(count.get() + 1);
            Ok(())
        });
        assert_eq!(count.get(), 2);
        assert_eq!(result.queued, 2);
    }

    #[test]
    fn empty_registry_yields_no_candidates_and_no_panic() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let selected = warmer.select_candidates(&[], 1_000);
        assert!(selected.is_empty());

        let result = warmer.warm(&selected, |_| Ok(()));
        assert_eq!(result.queued, 0);
        assert_eq!(result.generated, 0);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn zero_limit_yields_no_candidates() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 1_000u64;
        let candidates = vec![candidate("/m/a.mov", 9, now - 1, ProxyCodec::H264)];
        let selected = warmer.select_candidates_limited(&candidates, now, 0);
        assert!(selected.is_empty());
    }

    #[test]
    fn future_last_access_does_not_panic_and_caps_recency() {
        // `last_access` in the future => saturating age of 0 => peak recency.
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 100u64;
        let c = candidate("/m/future.mov", 0, now + 5_000, ProxyCodec::H264);
        let score = warmer.score(&c, now);
        // age == 0 => recency_decay == 1.0 => score == recency_weight (hits == 0).
        assert!((score - DEFAULT_RECENCY_WEIGHT).abs() < 1e-12);
    }

    #[test]
    fn from_registry_record_uses_first_proxy_codec() {
        let mut record = RegistryRecord::new(PathBuf::from("/src/clip.mov"));
        record.add_proxy(ProxyEntry::new(
            PathBuf::from("/proxy/clip.mov"),
            make_spec(ProxyCodec::ProRes422Proxy),
        ));
        let cand =
            WarmCandidate::from_registry_record(&record, 7, 1_234).expect("record has a proxy");
        assert_eq!(cand.original_path, PathBuf::from("/src/clip.mov"));
        assert_eq!(cand.hit_count, 7);
        assert_eq!(cand.last_access, 1_234);
        assert_eq!(cand.codec, ProxyCodec::ProRes422Proxy);
    }

    #[test]
    fn from_registry_record_none_when_no_proxies() {
        let record = RegistryRecord::new(PathBuf::from("/src/empty.mov"));
        assert!(WarmCandidate::from_registry_record(&record, 1, 1).is_none());
    }

    #[test]
    fn select_and_warm_round_trip() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default().with_max_candidates(2));
        let now = 1_000u64;
        let candidates = vec![
            candidate("/m/a.mov", 9, now - 1, ProxyCodec::H264),
            candidate("/m/b.mov", 8, now - 2, ProxyCodec::H264),
            candidate("/m/c.mov", 7, now - 3, ProxyCodec::H264),
        ];
        let (selected, result) = warmer.select_and_warm(&candidates, now, |_| Ok(()));
        assert_eq!(selected.len(), 2);
        assert_eq!(result.queued, 2);
        assert_eq!(result.generated, 2);
    }

    #[test]
    fn config_builders_round_trip() {
        let config = WarmingConfig::new()
            .with_freq_weight(2.5)
            .with_recency_weight(7.0)
            .with_recency_tau_secs(120.0)
            .with_codec_pref(ProxyCodec::Vp9, 3.0)
            .with_max_candidates(4);
        assert!((config.freq_weight - 2.5).abs() < 1e-12);
        assert!((config.recency_weight - 7.0).abs() < 1e-12);
        assert!((config.recency_tau_secs - 120.0).abs() < 1e-12);
        assert_eq!(config.codec_pref, Some(ProxyCodec::Vp9));
        assert!((config.codec_pref_bonus - 3.0).abs() < 1e-12);
        assert_eq!(config.max_candidates, 4);
    }

    #[test]
    fn non_positive_tau_is_clamped_safely() {
        // tau <= 0 must not yield NaN/Inf; decay stays finite and in [0, 1].
        let config = WarmingConfig::default().with_recency_tau_secs(0.0);
        let d0 = config.recency_decay(0.0);
        let d_big = config.recency_decay(1_000.0);
        assert!(d0.is_finite() && (0.0..=1.0).contains(&d0));
        assert!(d_big.is_finite() && (0.0..=1.0).contains(&d_big));
        assert!(d0 >= d_big);
    }

    #[test]
    fn score_uses_supplied_now_deterministically() {
        // Same candidate, two different `now` values => later `now` (larger age)
        // gives a strictly smaller recency contribution.
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let c = candidate("/m/x.mov", 2, 1_000, ProxyCodec::H264);
        let early = warmer.score(&c, 1_010);
        let late = warmer.score(&c, 5_000);
        assert!(early > late);
    }

    #[test]
    fn select_does_not_depend_on_input_order() {
        let warmer = ProxyCacheWarmer::new(WarmingConfig::default());
        let now = 1_000u64;
        let a = candidate("/m/a.mov", 10, now - 1, ProxyCodec::H264);
        let b = candidate("/m/b.mov", 5, now - 2, ProxyCodec::H264);
        let c = candidate("/m/c.mov", 1, now - 3, ProxyCodec::H264);

        let forward = warmer.select_candidates(&[a.clone(), b.clone(), c.clone()], now);
        let reversed = warmer.select_candidates(&[c, b, a], now);
        let forward_paths: Vec<&Path> = forward.iter().map(|x| x.original_path.as_path()).collect();
        let reversed_paths: Vec<&Path> =
            reversed.iter().map(|x| x.original_path.as_path()).collect();
        assert_eq!(forward_paths, reversed_paths);
    }
}
