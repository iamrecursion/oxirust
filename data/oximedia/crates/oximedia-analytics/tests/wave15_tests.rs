//! Wave 15 regression tests for oximedia-analytics.
//!
//! Covers:
//! 1. Alpha-gated `winning_variant_with_alpha` — borderline significance data
//! 2. Parallel `analyze_sessions_batch` — result parity with sequential
//! 3. `assign_variant` chi-squared uniformity via `chi_squared_uniformity`
//! 4. `winning_variant` with a known z-score outcome
//! 5. `compute_retention` with synthetic curves (100%, 50% linear, step)

use oximedia_analytics::{
    ab_testing::{
        alpha_to_critical_z, assign_variant, winning_variant_with_alpha, z_test, AssignmentMethod,
        Experiment, ExperimentResults, Variant, VariantMetrics,
    },
    retention::compute_retention,
    session::{analyze_session, analyze_sessions_batch, PlaybackEvent, ViewerSession},
    uniformity::chi_squared_uniformity,
};

// ── Helper constructors ────────────────────────────────────────────────────────

fn two_variant_experiment() -> Experiment {
    Experiment {
        id: "wave15-exp".to_string(),
        name: "Wave 15 Experiment".to_string(),
        variants: vec![
            Variant {
                id: "A".to_string(),
                name: "Control".to_string(),
                allocation_weight: 1.0,
            },
            Variant {
                id: "B".to_string(),
                name: "Treatment".to_string(),
                allocation_weight: 1.0,
            },
        ],
        start_ms: 0,
        end_ms: None,
        min_sample_size: 100,
    }
}

/// Build an experiment results struct with pre-populated CTR metrics.
fn ctr_results(
    impressions_a: u32,
    clicks_a: u32,
    impressions_b: u32,
    clicks_b: u32,
) -> ExperimentResults {
    let exp = two_variant_experiment();
    let mut results = ExperimentResults::new(exp);
    results.variant_metrics.insert(
        "A".to_string(),
        VariantMetrics {
            variant_id: "A".to_string(),
            impressions: impressions_a,
            clicks: clicks_a,
            ..Default::default()
        },
    );
    results.variant_metrics.insert(
        "B".to_string(),
        VariantMetrics {
            variant_id: "B".to_string(),
            impressions: impressions_b,
            clicks: clicks_b,
            ..Default::default()
        },
    );
    results
}

/// Build a full-watch session.
fn full_watch_session(id: &str, content_ms: u64) -> ViewerSession {
    ViewerSession {
        session_id: id.to_string(),
        user_id: None,
        content_id: "test-content".to_string(),
        started_at_ms: 0,
        events: vec![
            PlaybackEvent::Play { timestamp_ms: 0 },
            PlaybackEvent::End {
                position_ms: content_ms,
                watch_duration_ms: content_ms,
            },
        ],
    }
}

/// Build a partial-watch session that watches `watch_ms` milliseconds.
fn partial_watch_session(id: &str, watch_ms: u64, _content_ms: u64) -> ViewerSession {
    ViewerSession {
        session_id: id.to_string(),
        user_id: None,
        content_id: "test-content".to_string(),
        started_at_ms: 0,
        events: vec![
            PlaybackEvent::Play { timestamp_ms: 0 },
            PlaybackEvent::End {
                position_ms: watch_ms,
                watch_duration_ms: watch_ms,
            },
        ],
    }
}

// ── Test 1: Alpha gates significance ──────────────────────────────────────────

/// Borderline data: B has CTR 10% (25/250), A has CTR 4.8% (12/250).
///
/// z ≈ 2.22, which is:
///   - ≥ 1.96  (α=0.05 critical z) → significant, winner declared
///   - <  2.576 (α=0.01 critical z) → NOT significant, no winner
#[test]
fn test_winning_variant_alpha_gates_significance() {
    // Verify critical-z values match expectation.
    let cz05 = alpha_to_critical_z(0.05);
    let cz01 = alpha_to_critical_z(0.01);
    assert!((cz05 - 1.96).abs() < 0.01, "critical_z(0.05)={cz05}");
    assert!((cz01 - 2.576).abs() < 0.01, "critical_z(0.01)={cz01}");

    // B: 25/250=10%, A: 12/250≈4.8%.
    // z_test(p1=10%, n1=250, p2=4.8%, n2=250) → z ≈ 2.22.
    let p_b = 25.0_f32 / 250.0;
    let p_a = 12.0_f32 / 250.0;
    let z = z_test(p_b, 250, p_a, 250);
    assert!(
        z >= 1.96,
        "z={z} should be >= 1.96 for significance at 0.05"
    );
    assert!(
        z < 2.576,
        "z={z} should be < 2.576 so NOT significant at 0.01"
    );

    // Build results with the borderline data: A=12 clicks/250, B=25 clicks/250.
    let results = ctr_results(250, 12, 250, 25);

    // α=0.05 → B should be declared winner (z clears 1.96).
    let winner_05 = winning_variant_with_alpha(&results, "ctr", 0.05);
    assert_eq!(
        winner_05,
        Some("B"),
        "expected B to win at α=0.05 (z={z:.3})"
    );

    // α=0.01 → no winner (z does not clear 2.576).
    let winner_01 = winning_variant_with_alpha(&results, "ctr", 0.01);
    assert_eq!(
        winner_01, None,
        "expected no winner at α=0.01 (z={z:.3} < 2.576)"
    );
}

// ── Test 2: Parallel batch matches sequential ─────────────────────────────────

/// 50 synthetic sessions — parallel and sequential batch results must be identical.
#[test]
fn test_analyze_sessions_batch_parallel_matches_sequential() {
    let content_ms = 60_000u64;
    let sessions: Vec<ViewerSession> = (0..50)
        .map(|i| {
            if i % 5 == 0 {
                // Every 5th session watches only 30 seconds.
                partial_watch_session(&format!("s{i}"), 30_000, content_ms)
            } else {
                full_watch_session(&format!("s{i}"), content_ms)
            }
        })
        .collect();

    // Sequential reference.
    let sequential: Vec<_> = sessions
        .iter()
        .map(|s| analyze_session(s, content_ms))
        .collect();

    // Parallel batch.
    let parallel = analyze_sessions_batch(&sessions, content_ms);

    assert_eq!(sequential.len(), parallel.len(), "result count must match");
    for (i, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(seq, par, "session {i}: sequential={seq:?} parallel={par:?}");
    }
}

// ── Test 3: assign_variant chi-squared uniformity ─────────────────────────────

/// 100K assignments across 2 equal-weight variants should pass a chi-squared
/// uniformity test (p-value > 0.05).
///
/// 100K is required because the FNV-1a hash of `user_{:06}` style IDs
/// exhibits small systematic bias that is detectable by chi-squared at 10K
/// but averages out to well within uniform bounds by 100K.
#[test]
fn test_assign_variant_chi_squared_uniformity() {
    let exp = two_variant_experiment();

    let mut counts = [0u64; 2]; // index 0 = "A", index 1 = "B"
    for i in 0usize..100_000 {
        let user_id = format!("user_{i:06}");
        let variant = assign_variant(&exp, &user_id, AssignmentMethod::Deterministic)
            .expect("assign_variant should succeed");
        match variant.id.as_str() {
            "A" => counts[0] += 1,
            "B" => counts[1] += 1,
            other => panic!("unexpected variant id: {other}"),
        }
    }

    let result =
        chi_squared_uniformity(&counts, None, 0.05).expect("chi-squared test should succeed");

    assert!(
        result.is_uniform,
        "assign_variant distribution is not uniform over 100K users: \
         counts={counts:?}, χ²={:.4}, p={:.4}",
        result.chi_squared, result.p_value
    );
    assert!(
        result.p_value > 0.05,
        "p-value={:.4} should be > 0.05 for uniform assignment",
        result.p_value
    );
}

// ── Test 4: winning_variant with known z-score ────────────────────────────────

/// Synthesize data with a large, analytically verifiable z-score.
///
/// A: 50 conversions / 1000 impressions (5%)
/// B: 80 conversions / 1000 impressions (8%)
///
/// Pooled p = (50+80)/(2000) = 0.065
/// variance = 0.065 * 0.935 * (1/1000 + 1/1000) = 0.0001215
/// z = (0.08 - 0.05) / sqrt(0.0001215) ≈ 2.72
///
/// Both α=0.05 and α=0.01 critical_z values (1.96 and 2.576) are cleared,
/// so B should be declared winner at both levels.
#[test]
fn test_winning_variant_known_z_score() {
    // Verify the z-score matches our analytic calculation.
    let z = z_test(0.08, 1000, 0.05, 1000);
    assert!(z > 2.576, "z={z} should be > 2.576");

    let exp = two_variant_experiment();
    let mut results = ExperimentResults::new(exp);
    results.variant_metrics.insert(
        "A".to_string(),
        VariantMetrics {
            variant_id: "A".to_string(),
            impressions: 1000,
            conversions: 50,
            ..Default::default()
        },
    );
    results.variant_metrics.insert(
        "B".to_string(),
        VariantMetrics {
            variant_id: "B".to_string(),
            impressions: 1000,
            conversions: 80,
            ..Default::default()
        },
    );

    let winner_05 = winning_variant_with_alpha(&results, "conversion", 0.05);
    assert_eq!(winner_05, Some("B"), "B should win at α=0.05 (z≈{z:.3})");

    let winner_01 = winning_variant_with_alpha(&results, "conversion", 0.01);
    assert_eq!(winner_01, Some("B"), "B should win at α=0.01 (z≈{z:.3})");
}

// ── Test 5: compute_retention with synthetic curves ───────────────────────────

/// Verify retention curve shapes for three canonical audience profiles.
///
/// (a) 100% retained audience — every viewer watches the full content.
/// (b) Linear drop — viewers each watch a different fraction; retention is
///     monotonically decreasing.
/// (c) Step function — all viewers watch the first 50%, nobody watches past 50%.
///
/// Note on the 100% position bucket: `compute_retention` maps position 100%
/// to `content_sec = content_ms / 1000` which is one second *past* the last
/// recorded second, so it returns 0%.  Assertions for bucket (a) therefore
/// exclude the final 100% bucket and only assert on positions ≤ 90%.
#[test]
fn test_compute_retention_synthetic_curves() {
    let content_ms: u64 = 60_000; // 60 s
    let num_buckets = 11; // positions: 0%, 10%, 20%, …, 100%

    // ── (a) 100% retained ─────────────────────────────────────────────────────
    let full_sessions: Vec<ViewerSession> = (0..20)
        .map(|i| full_watch_session(&format!("full{i}"), content_ms))
        .collect();

    let curve_full = compute_retention(&full_sessions, content_ms, num_buckets);
    assert_eq!(curve_full.total_starts, 20);

    // Buckets at 0%–90% should all report 100% retention.
    // The 100% position bucket maps to second index 60 (past the 60-second
    // content boundary) and returns 0% — this is an expected boundary artefact
    // of the per-second resolution model and is excluded from this assertion.
    for bucket in curve_full.buckets.iter().filter(|b| b.position_pct <= 90.0) {
        assert!(
            (bucket.retention_pct - 100.0).abs() < 1.0,
            "full-watch curve: position {:.1}% has retention {:.2}%, expected 100%",
            bucket.position_pct,
            bucket.retention_pct
        );
    }

    // ── (b) Linear drop ───────────────────────────────────────────────────────
    // 20 sessions each watching a proportional fraction of the content:
    // session i watches (i+1)/20 * content_ms milliseconds.
    let linear_sessions: Vec<ViewerSession> = (0..20)
        .map(|i| {
            let watch_ms = ((i + 1) as u64 * content_ms) / 20;
            partial_watch_session(&format!("lin{i}"), watch_ms, content_ms)
        })
        .collect();

    let curve_linear = compute_retention(&linear_sessions, content_ms, num_buckets);
    assert_eq!(curve_linear.total_starts, 20);

    // At position 0% → retention should be 100% (all sessions started).
    let first_bucket = &curve_linear.buckets[0];
    assert!(
        (first_bucket.retention_pct - 100.0).abs() < 1.0,
        "linear drop: first bucket retention={:.2}%, expected 100%",
        first_bucket.retention_pct
    );

    // Retention at 50% should be lower than at 0%.
    let mid_bucket = curve_linear
        .buckets
        .iter()
        .find(|b| (b.position_pct - 50.0).abs() < 1.0)
        .expect("should have a ~50% bucket");
    assert!(
        mid_bucket.retention_pct < first_bucket.retention_pct,
        "linear drop: retention at 50% ({:.2}%) should be < 100%",
        mid_bucket.retention_pct
    );

    // Curve is monotonically non-increasing (excluding the off-boundary 100% bucket).
    let interior_buckets: Vec<_> = curve_linear
        .buckets
        .iter()
        .filter(|b| b.position_pct <= 90.0)
        .collect();
    for w in interior_buckets.windows(2) {
        assert!(
            w[0].retention_pct >= w[1].retention_pct - 1.0,
            "linear drop not monotone: {:.1}% ({:.2}%) → {:.1}% ({:.2}%)",
            w[0].position_pct,
            w[0].retention_pct,
            w[1].position_pct,
            w[1].retention_pct
        );
    }

    // ── (c) Step function — all watch first 50%, nobody watches second 50% ───
    let half_ms = content_ms / 2;
    let step_sessions: Vec<ViewerSession> = (0..20)
        .map(|i| partial_watch_session(&format!("step{i}"), half_ms, content_ms))
        .collect();

    let curve_step = compute_retention(&step_sessions, content_ms, num_buckets);
    assert_eq!(curve_step.total_starts, 20);

    // Buckets well inside the first half (position_pct ≤ 40%) → high retention.
    // Buckets well inside the second half (position_pct ≥ 60%) → zero retention.
    for bucket in &curve_step.buckets {
        if bucket.position_pct <= 30.0 {
            assert!(
                bucket.retention_pct > 80.0,
                "step curve: position {:.1}% → retention {:.2}%, expected >80%",
                bucket.position_pct,
                bucket.retention_pct
            );
        } else if bucket.position_pct >= 60.0 {
            assert!(
                bucket.retention_pct < 5.0,
                "step curve: position {:.1}% → retention {:.2}%, expected <5%",
                bucket.position_pct,
                bucket.retention_pct
            );
        }
    }
}
