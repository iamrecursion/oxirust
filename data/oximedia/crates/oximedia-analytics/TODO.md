# oximedia-analytics TODO

## Current Status
- 22 modules: `session`, `retention`, `ab_testing`, `engagement`, `error`, `bandit`, `cohort`,
  `funnel`, `quantile`, `attribution`, `realtime`, `geo_device`, `anomaly`, `ctr`,
  `event_buffer`, `fingerprint`, `heatmap`, `multivariate`, `percentile`, `recommendation`,
  `segment_retention`, `weighted_retention`
- 23 source files total
- 327 unit tests passing (+ 6 doc-tests)
- Viewer session tracking with playback events, attention heatmaps, reservoir-sampled heatmaps
- Audience retention curves, segment retention, weighted demographic retention, incremental computation
- A/B testing (frequentist + Bayesian), multi-armed bandit (epsilon-greedy + Thompson sampling)
- Engagement scoring, linear regression + EMA trend analysis, time-series decomposition
- Cohort analysis, funnel analysis, churn prediction, loyalty scoring
- Real-time sliding-window aggregation, CTR tracking, geographic/device breakdowns
- Watch-time attribution, multivariate testing, content recommendation

## Enhancements
- [x] Add configurable significance level (alpha) to `winning_variant` in `ab_testing` (implemented 2026-06-01: alpha_to_critical_z gating in winning_variant_with_alpha; src/ab_testing.rs:306)
- [x] Implement Bayesian A/B testing as alternative to frequentist z-test in `ab_testing` (verified 2026-05-16; src/ab_testing.rs — bandit.rs Thompson sampling wired)
- [x] Add multi-armed bandit (epsilon-greedy, Thompson sampling) for adaptive experiments in `ab_testing` (verified 2026-05-16; src/bandit.rs)
- [x] Implement segment-level retention analysis in `retention` (retention by content chapter/segment) (verified 2026-05-16; src/segment_retention.rs)
- [x] Add weighted retention curves in `retention` accounting for viewer demographics (verified 2026-05-16; src/weighted_retention.rs)
- [x] Implement time-series decomposition (trend + seasonality) in `engagement` trend analysis (verified 2026-05-16; src/engagement.rs)
- [x] Add exponential moving average option alongside linear regression in `linear_regression_slope`
- [x] Implement funnel analysis in `session` (track viewer progression through content milestones) (verified 2026-05-16; src/funnel.rs)
- [x] Add session replay reconstruction from `PlaybackEvent` sequences for debugging (verified 2026-05-16; src/replay.rs)

## New Features
- [x] Add cohort analysis module: group viewers by first-view date and track retention over time (verified 2026-05-16; src/cohort.rs)
- [x] Implement churn prediction based on engagement score decline patterns (verified 2026-06-01: predict_churn/ChurnConfig/ChurnRisk/ChurnAssessment — src/funnel.rs:152-230)
- [x] Add real-time analytics aggregation: sliding window metrics (concurrent viewers, bitrate stats) (verified 2026-05-16; src/realtime.rs)
- [x] Implement content recommendation scoring based on engagement similarity (verified 2026-05-16; src/recommendation.rs)
- [x] Add geographic/device breakdowns for session metrics
- [x] Implement watch time attribution: allocate credit to content segments for total engagement (verified 2026-05-16; src/attribution.rs)
- [x] Add multivariate testing support (test multiple variables simultaneously) in `ab_testing` (verified 2026-05-16; src/multivariate.rs)
- [x] Implement click-through rate (CTR) tracking for thumbnails and previews (verified 2026-05-16; src/ctr.rs)
- [x] Add viewer loyalty scoring: frequency, recency, duration weighted composite (verified 2026-06-01: compute_loyalty/LoyaltyScore/LoyaltyComponents/LoyaltyWeights — src/funnel.rs:242-345)

## Performance
- [x] Add streaming/incremental computation to `compute_retention` for large viewer datasets (verified 2026-06-01: IncrementalRetentionState/compute_retention_incremental — src/retention.rs:142,257)
- [x] Implement approximate quantile computation (t-digest) for percentile metrics at scale (verified 2026-05-16; src/quantile.rs, src/percentile.rs)
- [x] Use integer arithmetic for `assign_variant` FNV-1a hash instead of string operations (verified 2026-06-05: FNV-1a already operates on bytes via u32 arithmetic; pinned via test_fnv1a_exact_hash_pin — src/ab_testing.rs)
- [x] Add batch processing for `analyze_session` across multiple sessions in parallel (implemented 2026-06-01: rayon par_iter in analyze_sessions_batch; src/session.rs:206)
- [x] Implement reservoir sampling for memory-bounded attention heatmap generation

## Testing
- [x] Test `assign_variant` distribution uniformity with chi-squared test over 10K+ assignments (added 2026-06-01: test_assign_variant_chi_squared_uniformity — tests/wave15_tests.rs)
- [x] Add tests for `winning_variant` with known p-value outcomes (verify statistical correctness) (added 2026-06-01: test_winning_variant_known_z_score, test_winning_variant_alpha_gates_significance — tests/wave15_tests.rs)
- [x] Test `compute_retention` with synthetic viewer data: 100% retention, 50% linear drop, step function (added 2026-06-01: test_compute_retention_synthetic_curves — tests/wave15_tests.rs)
- [x] Test `drop_off_points` detection accuracy with synthetic retention curves (added 2026-06-05: test_drop_off_detects_large_drop, test_drop_off_empty_result_when_no_drop, test_drop_off_strict_boundary — src/retention.rs)
- [x] Test `linear_regression_slope` against known regression datasets (added 2026-06-05: test_slope_known_answer, test_slope_negative, test_slope_fewer_than_two_points, test_slope_zero_denominator — src/engagement.rs)
- [ ] Add test for `attention_heatmap` with overlapping and non-overlapping session segments

## Documentation
- [ ] Add example workflow: create experiment -> assign users -> collect metrics -> determine winner
- [ ] Document engagement score formula and weight tuning recommendations
- [ ] Add retention curve interpretation guide (what constitutes good/bad retention)
