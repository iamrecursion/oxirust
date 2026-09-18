//! Smoke tests for the `oximedia.neural` Python bindings.
//!
//! These tests drive the PyO3 classes via a minimal embedded Python
//! interpreter (same pattern as `tests/ml_smoke.rs`) so the full
//! Python-facing contract — argument binding, `__repr__`/`__len__`, and
//! error propagation as real `ValueError` exceptions — is exercised without
//! needing a maturin-built wheel.

use oximedia_py::neural_py::register_submodule;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build a parent module and register the `neural` submodule into it,
/// returning a Python dict with `neural` bound as a top-level name.
fn prepare_neural_env(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let parent = PyModule::new(py, "oximedia_test_parent")?;
    register_submodule(&parent)?;
    let neural = parent.getattr("neural")?;
    let globals = PyDict::new(py);
    globals.set_item("neural", neural)?;
    Ok(globals)
}

#[test]
fn tensor_construct_and_shape() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "t = neural.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])\n\
                 assert t.shape() == [2, 2]\n\
                 assert t.ndim() == 2\n\
                 assert t.numel() == 4\n\
                 assert len(t) == 4\n\
                 assert t.to_list() == [1.0, 2.0, 3.0, 4.0]\n"
            ),
            Some(&globals),
            None,
        )
        .expect("tensor runs");
    });
}

#[test]
fn tensor_zeros_and_ones() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "z = neural.Tensor.zeros([3])\n\
                 assert z.to_list() == [0.0, 0.0, 0.0]\n\
                 o = neural.Tensor.ones([2])\n\
                 assert o.to_list() == [1.0, 1.0]\n"
            ),
            Some(&globals),
            None,
        )
        .expect("zeros/ones run");
    });
}

#[test]
fn tensor_shape_mismatch_raises_value_error() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "try:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}neural.Tensor([1.0, 2.0, 3.0], [2, 2])\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}raise AssertionError('expected error')\n\
                 except ValueError:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}pass\n"
            ),
            Some(&globals),
            None,
        )
        .expect("shape mismatch raises ValueError");
    });
}

#[test]
fn scene_classifier_classifies_zero_vector() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "clf = neural.SceneClassifier()\n\
                 features = [0.0] * neural.SceneClassifier.input_dim()\n\
                 idx, conf = clf.classify(features)\n\
                 assert 0 <= idx < neural.SceneClassifier.num_classes()\n\
                 assert abs(conf - 0.1) < 1e-4\n"
            ),
            Some(&globals),
            None,
        )
        .expect("scene classifier runs");
    });
}

#[test]
fn scene_classifier_wrong_dim_raises_value_error() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "clf = neural.SceneClassifier()\n\
                 try:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}clf.classify([0.0, 0.0])\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}raise AssertionError('expected error')\n\
                 except ValueError:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}pass\n"
            ),
            Some(&globals),
            None,
        )
        .expect("wrong-dim classify raises ValueError");
    });
}

#[test]
fn thumbnail_ranker_scores_in_range() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "ranker = neural.ThumbnailRanker()\n\
                 score = ranker.score([0.0] * neural.ThumbnailRanker.input_dim())\n\
                 assert abs(score - 0.5) < 1e-4\n"
            ),
            Some(&globals),
            None,
        )
        .expect("thumbnail ranker runs");
    });
}

#[test]
fn sr_upscaler_doubles_frame_size() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "up = neural.SrUpscaler()\n\
                 frame = [0.5] * (8 * 8)\n\
                 out = up.upscale_2x(frame, 8, 8)\n\
                 assert len(out) == 16 * 16\n"
            ),
            Some(&globals),
            None,
        )
        .expect("sr upscaler runs");
    });
}

#[test]
fn feature_extractor_returns_128_dims() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "fe = neural.FeatureExtractor()\n\
                 frame = [0.5] * (32 * 32)\n\
                 features = fe.extract(frame, 32, 32)\n\
                 assert len(features) == neural.FeatureExtractor.feature_dim()\n"
            ),
            Some(&globals),
            None,
        )
        .expect("feature extractor runs");
    });
}

#[test]
fn scene_class_name_resolves() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_neural_env(py).expect("register neural");
        py.run(
            c_str!(
                "assert neural.scene_class_name(0) == 'Static'\n\
                 assert neural.scene_class_name(99) == 'Unknown'\n"
            ),
            Some(&globals),
            None,
        )
        .expect("scene_class_name runs");
    });
}
