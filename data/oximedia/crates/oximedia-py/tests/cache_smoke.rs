//! Smoke tests for the `oximedia.cache` Python bindings.
//!
//! Same embedded-interpreter pattern as `tests/ml_smoke.rs`.

use oximedia_py::cache_py::register_submodule;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn prepare_cache_env(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let parent = PyModule::new(py, "oximedia_test_parent")?;
    register_submodule(&parent)?;
    let cache = parent.getattr("cache")?;
    let globals = PyDict::new(py);
    globals.set_item("cache", cache)?;
    Ok(globals)
}

#[test]
fn put_get_and_len() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_cache_env(py).expect("register cache");
        py.run(
            c_str!(
                "c = cache.LruCache(4)\n\
                 c.put('a', b'hello')\n\
                 assert c.get('a') == b'hello'\n\
                 assert len(c) == 1\n\
                 assert c.get('missing') is None\n"
            ),
            Some(&globals),
            None,
        )
        .expect("put/get runs");
    });
}

#[test]
fn eviction_at_capacity() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_cache_env(py).expect("register cache");
        py.run(
            c_str!(
                "c = cache.LruCache(2)\n\
                 c.put('a', b'1')\n\
                 c.put('b', b'2')\n\
                 c.put('c', b'3')\n\
                 assert not c.contains('a')\n\
                 assert c.contains('b')\n\
                 assert c.contains('c')\n"
            ),
            Some(&globals),
            None,
        )
        .expect("eviction runs");
    });
}

#[test]
fn stats_track_hits_and_misses() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_cache_env(py).expect("register cache");
        py.run(
            c_str!(
                "c = cache.LruCache(4)\n\
                 c.put('a', b'x')\n\
                 c.get('a')\n\
                 c.get('nope')\n\
                 s = c.stats()\n\
                 assert s.hits == 1\n\
                 assert s.misses == 1\n\
                 assert abs(s.hit_rate() - 0.5) < 1e-9\n"
            ),
            Some(&globals),
            None,
        )
        .expect("stats runs");
    });
}

#[test]
fn pin_survives_eviction() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_cache_env(py).expect("register cache");
        py.run(
            c_str!(
                "c = cache.LruCache(2)\n\
                 c.put_pinned('keep', b'1')\n\
                 c.put('b', b'2')\n\
                 c.put('c', b'3')\n\
                 assert c.contains('keep')\n"
            ),
            Some(&globals),
            None,
        )
        .expect("pin runs");
    });
}

#[test]
fn ttl_expires_and_purges() {
    pyo3::Python::initialize();
    Python::attach(|py| {
        let globals = prepare_cache_env(py).expect("register cache");
        py.run(
            c_str!(
                "import time\n\
                 c = cache.LruCache(4)\n\
                 c.put_with_ttl_ms('a', b'1', 0)\n\
                 c.put('b', b'2')\n\
                 time.sleep(0.01)\n\
                 assert c.get('a') is None\n\
                 c2 = cache.LruCache(4)\n\
                 c2.put_with_ttl_ms('x', b'1', 0)\n\
                 c2.put('y', b'2')\n\
                 time.sleep(0.01)\n\
                 assert c2.purge_expired() == 1\n\
                 assert len(c2) == 1\n"
            ),
            Some(&globals),
            None,
        )
        .expect("ttl runs");
    });
}
