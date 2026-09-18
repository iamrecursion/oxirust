//! Smoke tests for the `oximedia.ml` Python bindings.
//!
//! These tests drive the PyO3 classes via a minimal embedded Python
//! interpreter so the full Python-facing contract is exercised without
//! needing a real ONNX model on disk. We use the pure-Rust paths
//! (device enumeration, model zoo, heuristic shot boundary detection,
//! face-embedding cosine math).
//!
//! Gated on the `ml` feature so the default (non-ML) build still
//! compiles without pulling in oximedia-ml.

#![cfg(feature = "ml")]

use oximedia_py::ml_py::register_submodule;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build a parent module and register the `ml` submodule into it,
/// returning a Python dict with `ml` bound as a top-level name.
fn prepare_ml_env(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let parent = PyModule::new(py, "oximedia_test_parent")?;
    register_submodule(&parent)?;
    let ml = parent.getattr("ml")?;
    let globals = PyDict::new(py);
    globals.set_item("ml", ml)?;
    Ok(globals)
}

#[test]
fn device_cpu_is_available_and_named() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "dev = ml.MlDeviceType.cpu()\n\
                 assert dev.name == 'cpu'\n\
                 assert dev.display_name == 'CPU'\n\
                 assert dev.is_available() is True\n"
            ),
            Some(&globals),
            None,
        )
        .expect("cpu device runs");
    });
}

#[test]
fn auto_device_is_available() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "dev = ml.MlDeviceType.auto()\n\
                 assert dev.is_available() is True\n\
                 top_level = ml.auto_device()\n\
                 assert top_level.is_available() is True\n"
            ),
            Some(&globals),
            None,
        )
        .expect("auto device runs");
    });
}

#[test]
fn device_from_name_round_trip() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "d = ml.MlDeviceType.from_name('cpu')\n\
                 assert d.name == 'cpu'\n\
                 a = ml.MlDeviceType.from_name('auto')\n\
                 assert a.is_available() is True\n\
                 try:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}ml.MlDeviceType.from_name('unknown-gpu')\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}raise AssertionError('expected error')\n\
                 except ValueError:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}pass\n"
            ),
            Some(&globals),
            None,
        )
        .expect("from_name runs");
    });
}

#[test]
fn device_list_available_contains_cpu() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "devs = ml.MlDeviceType.list_available()\n\
                 assert any(d.name == 'cpu' for d in devs)\n\
                 top_level = ml.available_devices()\n\
                 assert any(d.name == 'cpu' for d in top_level)\n"
            ),
            Some(&globals),
            None,
        )
        .expect("list_available runs");
    });
}

#[test]
fn model_zoo_defaults_contain_scene_and_shot() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "zoo = ml.MlModelZoo.with_defaults()\n\
                 assert len(zoo) == 2\n\
                 scene = zoo.get('places365/resnet18')\n\
                 assert scene is not None\n\
                 assert scene.task == 'scene-classification'\n\
                 assert scene.input_size == (224, 224)\n\
                 shot = zoo.get('transnet-v2')\n\
                 assert shot is not None\n\
                 assert shot.task == 'shot-boundary'\n"
            ),
            Some(&globals),
            None,
        )
        .expect("zoo runs");
    });
}

#[test]
fn model_zoo_empty_has_no_entries() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "zoo = ml.MlModelZoo()\n\
                 assert len(zoo) == 0\n\
                 assert zoo.get('places365/resnet18') is None\n"
            ),
            Some(&globals),
            None,
        )
        .expect("empty zoo runs");
    });
}

#[test]
fn shot_boundary_heuristic_runs_on_constant_frames() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "import numpy as np\n\
                 det = ml.ShotBoundaryDetector.heuristic()\n\
                 assert det.has_model is False\n\
                 assert 0.0 < det.threshold <= 1.0\n\
                 frames = np.zeros((2, 27, 48, 3), dtype=np.uint8)\n\
                 boundaries = det.run(frames)\n\
                 assert len(boundaries) == 0\n"
            ),
            Some(&globals),
            None,
        )
        .expect("heuristic runs");
    });
}

#[test]
fn shot_boundary_heuristic_rejects_wrong_channels() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "import numpy as np\n\
                 det = ml.ShotBoundaryDetector.heuristic()\n\
                 bad = np.zeros((1, 27, 48, 4), dtype=np.uint8)\n\
                 try:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}det.run(bad)\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}raise AssertionError('expected error')\n\
                 except ValueError:\n\
                 \u{0020}\u{0020}\u{0020}\u{0020}pass\n"
            ),
            Some(&globals),
            None,
        )
        .expect("rejects wrong channels");
    });
}

#[test]
fn face_embedding_cosine_self_is_one() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "emb = ml.FaceEmbedding.from_raw([0.1, 0.2, 0.3, 0.4])\n\
                 sim = emb.cosine_similarity(emb)\n\
                 assert abs(sim - 1.0) < 1e-5\n"
            ),
            Some(&globals),
            None,
        )
        .expect("self cosine runs");
    });
}

#[test]
fn face_embedding_cosine_orthogonal_is_zero() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "a = ml.FaceEmbedding.from_raw([1.0, 0.0, 0.0, 0.0])\n\
                 b = ml.FaceEmbedding.from_raw([0.0, 1.0, 0.0, 0.0])\n\
                 sim = a.cosine_similarity(b)\n\
                 assert abs(sim) < 1e-5\n"
            ),
            Some(&globals),
            None,
        )
        .expect("orthogonal cosine runs");
    });
}

#[test]
fn face_embedding_from_raw_is_unit_norm() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_ml_env(py).expect("register ml");
        py.run(
            c_str!(
                "emb = ml.FaceEmbedding.from_raw([3.0, 4.0])\n\
                 vs = emb.to_list()\n\
                 norm = (vs[0] ** 2 + vs[1] ** 2) ** 0.5\n\
                 assert abs(norm - 1.0) < 1e-5\n\
                 assert len(emb) == 2\n"
            ),
            Some(&globals),
            None,
        )
        .expect("l2 norm runs");
    });
}
