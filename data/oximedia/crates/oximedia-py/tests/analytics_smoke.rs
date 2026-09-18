//! Smoke tests for the `oximedia.analytics` Python bindings.
//!
//! Same embedded-interpreter pattern as `tests/ml_smoke.rs`. These tests
//! specifically cover `attention_heatmap` and `compute_engagement`, which
//! take `Vec<PyRef<ViewerSession>>` and therefore need real Python objects
//! (GIL) to construct — the native `#[cfg(test)]` unit tests in
//! `src/analytics_py.rs` cover everything that does not require the GIL.

use oximedia_py::analytics_py::register_submodule;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn prepare_analytics_env(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let parent = PyModule::new(py, "oximedia_test_parent")?;
    register_submodule(&parent)?;
    let analytics = parent.getattr("analytics")?;
    let globals = PyDict::new(py);
    globals.set_item("analytics", analytics)?;
    Ok(globals)
}

#[test]
fn session_tracking_and_analyze() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_analytics_env(py).expect("register analytics");
        py.run(
            c_str!(
                "s = analytics.ViewerSession('sess-1', 'content-1', 0)\n\
                 s.track_play(0)\n\
                 s.track_end(10000, 10000)\n\
                 assert s.event_count() == 2\n\
                 assert s.session_id == 'sess-1'\n\
                 assert s.content_id == 'content-1'\n\
                 m = s.analyze(10000)\n\
                 assert abs(m.completion_pct - 100.0) < 1e-3\n\
                 assert m.total_watch_ms == 10000\n"
            ),
            Some(&globals),
            None,
        )
        .expect("session tracking runs");
    });
}

#[test]
fn module_level_analyze_session_matches_method() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_analytics_env(py).expect("register analytics");
        py.run(
            c_str!(
                "s = analytics.ViewerSession('sess-2', 'content-1', 0)\n\
                 s.track_play(0)\n\
                 s.track_end(5000, 5000)\n\
                 m1 = s.analyze(10000)\n\
                 m2 = analytics.analyze_session(s, 10000)\n\
                 assert m1.total_watch_ms == m2.total_watch_ms\n\
                 assert m1.completion_pct == m2.completion_pct\n"
            ),
            Some(&globals),
            None,
        )
        .expect("module-level analyze_session runs");
    });
}

#[test]
fn attention_heatmap_peaks_at_one() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_analytics_env(py).expect("register analytics");
        py.run(
            c_str!(
                "s = analytics.ViewerSession('sess-3', 'content-1', 0)\n\
                 s.track_play(0)\n\
                 s.track_end(10000, 10000)\n\
                 heat = analytics.attention_heatmap([s], 10000, 2000)\n\
                 assert len(heat) > 0\n\
                 peak = max(intensity for _pos, intensity in heat)\n\
                 assert abs(peak - 1.0) < 1e-6\n"
            ),
            Some(&globals),
            None,
        )
        .expect("attention_heatmap runs");
    });
}

#[test]
fn attention_heatmap_empty_sessions_is_empty() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_analytics_env(py).expect("register analytics");
        py.run(
            c_str!("assert analytics.attention_heatmap([], 10000, 2000) == []\n"),
            Some(&globals),
            None,
        )
        .expect("empty attention_heatmap runs");
    });
}

#[test]
fn compute_engagement_full_watch_scores_high() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_analytics_env(py).expect("register analytics");
        py.run(
            c_str!(
                "s0 = analytics.ViewerSession('sess-a', 'content-1', 0)\n\
                 s0.track_play(0)\n\
                 s0.track_end(10000, 10000)\n\
                 s1 = analytics.ViewerSession('sess-b', 'content-1', 0)\n\
                 s1.track_play(0)\n\
                 s1.track_end(10000, 10000)\n\
                 score = analytics.compute_engagement([s0, s1], 10000)\n\
                 assert score.content_id == 'content-1'\n\
                 assert score.watch_time_score > 0.9, score.watch_time_score\n\
                 assert score.completion_score > 0.9, score.completion_score\n\
                 assert score.score > 0.5, score.score\n\
                 assert score.social_score == 0.0\n"
            ),
            Some(&globals),
            None,
        )
        .expect("compute_engagement runs");
    });
}
