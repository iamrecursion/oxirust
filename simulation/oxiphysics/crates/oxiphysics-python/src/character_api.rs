// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the oxiphysics character module.
//!
//! Exposes the kinematic capsule character controller to Python.
//!
//! The collision backend of the controller is a closure in Rust. From Python it
//! is a callable: [`PyCharacterController::move_and_slide`] wraps it into the
//! closure the engine expects, so any Python collision code (or a test double)
//! can drive the capsule.

use oxiphysics::character::{CharacterConfig, CharacterController, CharacterShape, SweepHit};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySequence;
use std::cell::RefCell;

/// Normals shorter than this are no direction.
const MIN_NORMAL_LENGTH: f64 = 1e-12;

/// `Ok` if every component is finite.
fn check_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be finite, got {values:?}"
        )))
    }
}

/// Convert what a Python sweep callable returned into the engine's hit.
///
/// `None` means the path is free; otherwise the value must be a pair
/// `(toi, normal)` of a finite time of impact and a non-zero finite normal,
/// which is normalised (the engine projects onto the plane of a unit normal).
fn sweep_hit_from_python(outcome: &Bound<'_, PyAny>) -> PyResult<Option<SweepHit>> {
    if outcome.is_none() {
        return Ok(None);
    }
    let shape_error = || {
        PyTypeError::new_err(
            "the sweep callable must return None (free path) or a (toi, normal) pair",
        )
    };
    let pair = outcome.cast::<PySequence>().map_err(|_| shape_error())?;
    if pair.len()? != 2 {
        return Err(shape_error());
    }
    let toi: f64 = pair.get_item(0)?.extract()?;
    let normal: [f64; 3] = pair.get_item(1)?.extract()?;
    if !toi.is_finite() {
        return Err(PyValueError::new_err(format!(
            "the time of impact must be finite, got {toi}"
        )));
    }
    check_finite("the hit normal", &normal)?;
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length < MIN_NORMAL_LENGTH {
        return Err(PyValueError::new_err(
            "the hit normal must not be the zero vector",
        ));
    }
    Ok(Some(SweepHit {
        toi,
        normal: [normal[0] / length, normal[1] / length, normal[2] / length],
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// PyCharacterController
// ─────────────────────────────────────────────────────────────────────────────

/// Kinematic capsule character controller with sweep-and-slide, step-up,
/// and slope handling.
///
/// `move_and_slide(delta, sweep)` moves the capsule with collision response.
/// The collision backend is the Python callable `sweep(origin, direction)`
/// that the controller calls to probe the world, `jump(speed)` launches the
/// capsule along the up axis, and `apply_gravity(gy, dt)` is the helper that
/// moves the capsule under gravity without any collision test.
#[pyclass(name = "CharacterController")]
pub struct PyCharacterController {
    inner: CharacterController,
}

#[pymethods]
impl PyCharacterController {
    /// Create a character controller at the given position.
    ///
    /// `radius` — capsule radius (m). `half_height` — capsule half-height (m).
    #[new]
    pub fn new(position: [f64; 3], radius: f64, half_height: f64) -> Self {
        let shape = CharacterShape {
            radius,
            half_height,
        };
        let config = CharacterConfig::default();
        Self {
            inner: CharacterController::new(position, shape, config),
        }
    }

    /// Current world-space position.
    pub fn position(&self) -> [f64; 3] {
        self.inner.position
    }

    /// Set the world-space position directly (teleport).
    pub fn set_position(&mut self, position: [f64; 3]) {
        self.inner.position = position;
    }

    /// Current velocity.
    pub fn velocity(&self) -> [f64; 3] {
        self.inner.velocity
    }

    /// Set velocity.
    pub fn set_velocity(&mut self, velocity: [f64; 3]) {
        self.inner.velocity = velocity;
    }

    /// Whether the controller is currently grounded.
    pub fn is_grounded(&self) -> bool {
        self.inner.is_grounded
    }

    /// Apply gravity for `dt` seconds (without collision).
    ///
    /// `gy` is the signed gravity component along y (`-9.81` pulls down). The
    /// velocity changes by `gy * dt` and then the position moves by the new
    /// velocity times `dt`, with no collision test at all; this is only for
    /// capsules that move freely. A capsule that collides integrates its own
    /// velocity and calls `move_and_slide`, which resolves the collisions.
    pub fn apply_gravity(&mut self, gy: f64, dt: f64) {
        self.inner.velocity[1] += gy * dt;
        self.inner.position[0] += self.inner.velocity[0] * dt;
        self.inner.position[1] += self.inner.velocity[1] * dt;
        self.inner.position[2] += self.inner.velocity[2] * dt;
    }

    /// Set max slope degrees.
    pub fn set_max_slope_deg(&mut self, deg: f64) {
        self.inner.config.max_slope_deg = deg;
    }

    /// Set step offset (max height that can be stepped over).
    pub fn set_step_offset(&mut self, offset: f64) {
        self.inner.config.step_offset = offset;
    }

    /// Normal of the ground surface the capsule stands on, or `None` when it
    /// is not grounded.
    pub fn ground_normal(&self) -> Option<[f64; 3]> {
        self.inner.ground_normal
    }

    /// Start a jump: set the velocity component along the up axis to `speed`
    /// (the horizontal components are kept) and leave the ground.
    ///
    /// Raises `ValueError` if `speed` is not finite.
    pub fn jump(&mut self, speed: f64) -> PyResult<()> {
        check_finite("speed", &[speed])?;
        self.inner.jump(speed);
        Ok(())
    }

    /// Move the capsule by `delta` with collision response and return
    /// `(translation, is_grounded)`: the displacement that was actually
    /// applied and whether the capsule stands on a walkable surface afterwards
    /// (see `ground_normal`).
    ///
    /// `sweep(origin, direction)` is the collision backend. The controller
    /// calls it with the world-space position of the capsule origin and the
    /// direction vector, scaled by the length of the motion to test (both
    /// 3-tuples of floats), and expects
    ///
    /// * `None` when the path is free, or
    /// * `(toi, normal)` for a hit: `toi` is the fraction in `[0, 1]` of
    ///   `direction` travelled before the contact (other values are clamped)
    ///   and `normal` the surface normal pointing away from the surface (it is
    ///   normalised).
    ///
    /// The capsule advances to the surface and slides along it with the rest of
    /// the motion; a hit on a surface steeper than the maximum slope is a wall
    /// that the capsule steps over when it is lower than the step offset, and
    /// a character that was grounded stays glued to the floor over small drops.
    ///
    /// Raises `TypeError` if `sweep` is not callable or returns something else
    /// than described, `ValueError` for a non-finite `delta`, time of impact or
    /// normal and for a zero normal, and whatever `sweep` raises. After an
    /// error the capsule is exactly as it was before the call.
    pub fn move_and_slide(
        &mut self,
        delta: [f64; 3],
        sweep: &Bound<'_, PyAny>,
    ) -> PyResult<([f64; 3], bool)> {
        check_finite("delta", &delta)?;
        if !sweep.is_callable() {
            return Err(PyTypeError::new_err("sweep must be callable"));
        }
        let before = self.inner.clone();
        // The engine's callback cannot fail, so the first Python error is kept
        // here, the remaining probes of the move report a free path without
        // calling Python again, and the error is raised once the move is over.
        let failure: RefCell<Option<PyErr>> = RefCell::new(None);
        let moved = self.inner.move_and_slide(delta, |origin, direction| {
            if failure.borrow().is_some() {
                return None;
            }
            let outcome = sweep
                .call1((
                    (origin[0], origin[1], origin[2]),
                    (direction[0], direction[1], direction[2]),
                ))
                .and_then(|outcome| sweep_hit_from_python(&outcome));
            match outcome {
                Ok(hit) => hit,
                Err(err) => {
                    *failure.borrow_mut() = Some(err);
                    None
                }
            }
        });
        if let Some(err) = failure.into_inner() {
            self.inner = before;
            return Err(err);
        }
        Ok((moved.translation, moved.is_grounded))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCharacterController>()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::PyDict;
    use std::ffi::CStr;

    /// Run `f` with the interpreter of the test process attached.
    fn with_python<R>(f: impl for<'py> FnOnce(Python<'py>) -> R) -> R {
        Python::initialize();
        Python::attach(f)
    }

    /// Run `source` and return the global namespace it defines.
    fn run_source<'py>(py: Python<'py>, source: &CStr) -> Bound<'py, PyDict> {
        let globals = PyDict::new(py);
        py.run(source, Some(&globals), None)
            .expect("the Python source compiles and runs");
        globals
    }

    /// The function `sweep` defined by `source`.
    fn python_sweep<'py>(py: Python<'py>, source: &CStr) -> Bound<'py, PyAny> {
        run_source(py, source)
            .get_item("sweep")
            .expect("globals lookup")
            .expect("the source defines sweep")
    }

    /// A sweep that never hits anything.
    const FREE: &CStr = c"def sweep(origin, direction):\n    return None\n";

    /// A wall whose normal leans 45 degrees towards -x and +y; only a probe that moves
    /// into it hits.
    const WALL: &CStr = c"
def sweep(origin, direction):
    n = (-0.7071067811865476, 0.7071067811865476, 0.0)
    if direction[0] * n[0] + direction[1] * n[1] + direction[2] * n[2] < -1e-9:
        return (0.0, n)
    return None
";

    /// A floor half way down every downward probe.
    const FLOOR: &CStr = c"
def sweep(origin, direction):
    if direction[1] < -1e-9:
        return (0.5, (0.0, 1.0, 0.0))
    return None
";

    /// A character whose 45 degree wall counts as a wall (the default slope
    /// limit of 45 degrees would put it exactly on the boundary) and that does
    /// not step over anything.
    fn wall_climber() -> PyCharacterController {
        let mut ctrl = PyCharacterController::new([0.0, 1.0, 0.0], 0.4, 0.9);
        ctrl.set_max_slope_deg(30.0);
        ctrl.set_step_offset(0.0);
        ctrl
    }

    fn assert_close(got: [f64; 3], expected: [f64; 3]) {
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-9, "got {got:?}, expected {expected:?}");
        }
    }

    #[test]
    fn test_character_controller_instantiation() {
        let ctrl = PyCharacterController::new([0.0, 1.0, 0.0], 0.4, 0.9);
        let pos = ctrl.position();
        assert!((pos[1] - 1.0).abs() < 1e-9);
        assert!(!ctrl.is_grounded());
        assert_eq!(ctrl.ground_normal(), None);
    }

    #[test]
    fn test_character_apply_gravity() {
        let mut ctrl = PyCharacterController::new([0.0, 10.0, 0.0], 0.4, 0.9);
        ctrl.apply_gravity(-9.81, 1.0);
        // Semi-implicit Euler: the velocity changes first, the position moves by it.
        assert_close(ctrl.velocity(), [0.0, -9.81, 0.0]);
        assert_close(ctrl.position(), [0.0, 10.0 - 9.81, 0.0]);
    }

    #[test]
    fn test_move_and_slide_on_a_free_path_applies_the_whole_delta() {
        with_python(|py| {
            let sweep = python_sweep(py, FREE);
            let mut ctrl = PyCharacterController::new([1.0, 2.0, 3.0], 0.4, 0.9);
            let (translation, grounded) = ctrl
                .move_and_slide([0.5, -0.25, 2.0], &sweep)
                .expect("a free path");
            assert_eq!(translation, [0.5, -0.25, 2.0]);
            assert!(!grounded);
            assert_eq!(ctrl.position(), [1.5, 1.75, 5.0]);
        });
    }

    #[test]
    fn test_move_and_slide_projects_the_translation_onto_a_45_degree_wall() {
        with_python(|py| {
            let sweep = python_sweep(py, WALL);
            let mut ctrl = wall_climber();
            let (translation, grounded) = ctrl
                .move_and_slide([1.0, 0.0, 0.0], &sweep)
                .expect("sliding along a wall");
            // The motion into the wall is removed, the rest slides up the incline:
            // (1, 0, 0) - n (n . (1, 0, 0)) = (0.5, 0.5, 0).
            assert_close(translation, [0.5, 0.5, 0.0]);
            let n = [-0.5f64.sqrt(), 0.5f64.sqrt(), 0.0];
            let along_normal =
                translation[0] * n[0] + translation[1] * n[1] + translation[2] * n[2];
            assert!(
                along_normal.abs() < 1e-9,
                "the translation lies in the wall plane"
            );
            assert!(!grounded, "a wall is not a floor");
            assert_eq!(ctrl.ground_normal(), None);
            assert_close(ctrl.position(), [0.5, 1.5, 0.0]);
            // The component along the wall is untouched.
            let mut ctrl = wall_climber();
            let (translation, _) = ctrl
                .move_and_slide([1.0, 0.0, 1.0], &sweep)
                .expect("sliding along a wall");
            assert_close(translation, [0.5, 0.5, 1.0]);
        });
    }

    #[test]
    fn test_move_and_slide_normalises_the_hit_normal_and_accepts_lists() {
        with_python(|py| {
            let sweep = python_sweep(
                py,
                c"
def sweep(origin, direction):
    n = (-2.0, 2.0, 0.0)
    if direction[0] * n[0] + direction[1] * n[1] + direction[2] * n[2] < -1e-9:
        return [0.0, [-2.0, 2.0, 0.0]]
    return None
",
            );
            let mut ctrl = wall_climber();
            let (translation, _) = ctrl
                .move_and_slide([1.0, 0.0, 0.0], &sweep)
                .expect("a list pair with a long normal");
            // The wall of length-2 normal (-2, 2, 0) is the same 45 degree wall.
            assert_close(translation, [0.5, 0.5, 0.0]);
        });
    }

    #[test]
    fn test_move_and_slide_lands_on_a_floor_and_reports_the_ground_normal() {
        with_python(|py| {
            let sweep = python_sweep(py, FLOOR);
            let mut ctrl = PyCharacterController::new([0.0, 5.0, 0.0], 0.4, 0.9);
            let (translation, grounded) = ctrl
                .move_and_slide([0.0, -1.0, 0.0], &sweep)
                .expect("falling onto a floor");
            // Half way down minus the 1 mm skin width.
            assert_close(translation, [0.0, -0.499, 0.0]);
            assert!(grounded);
            assert!(ctrl.is_grounded());
            assert_eq!(ctrl.ground_normal(), Some([0.0, 1.0, 0.0]));
        });
    }

    #[test]
    fn test_jump_replaces_the_vertical_velocity_and_leaves_the_ground() {
        with_python(|py| {
            let sweep = python_sweep(py, FLOOR);
            let mut ctrl = PyCharacterController::new([0.0, 5.0, 0.0], 0.4, 0.9);
            ctrl.move_and_slide([0.0, -1.0, 0.0], &sweep)
                .expect("falling onto a floor");
            assert!(ctrl.is_grounded());
            ctrl.set_velocity([1.0, -3.0, 2.0]);
            ctrl.jump(5.0).expect("a finite speed");
            assert_eq!(ctrl.velocity(), [1.0, 5.0, 2.0]);
            assert!(!ctrl.is_grounded());
            assert_eq!(ctrl.ground_normal(), None);
        });
    }

    #[test]
    fn test_jump_rejects_a_non_finite_speed() {
        with_python(|py| {
            let mut ctrl = PyCharacterController::new([0.0; 3], 0.4, 0.9);
            ctrl.set_velocity([1.0, 2.0, 3.0]);
            for speed in [f64::NAN, f64::INFINITY] {
                let err = ctrl.jump(speed).expect_err("a non-finite speed");
                assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            }
            assert_eq!(ctrl.velocity(), [1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn test_move_and_slide_raises_what_the_sweep_raises_and_restores_the_controller() {
        with_python(|py| {
            let globals = run_source(
                py,
                c"
calls = []
def sweep(origin, direction):
    calls.append(origin)
    if len(calls) == 1:
        return (0.5, (-1.0, 0.0, 0.0))
    raise RuntimeError('boom')
",
            );
            let sweep = globals
                .get_item("sweep")
                .expect("lookup")
                .expect("sweep is defined");
            let mut ctrl = PyCharacterController::new([0.0, 1.0, 0.0], 0.4, 0.9);
            ctrl.set_velocity([0.5, 0.0, 0.0]);
            let err = ctrl
                .move_and_slide([1.0, 0.0, 0.0], &sweep)
                .expect_err("the second probe raises");
            assert!(err.is_instance_of::<PyRuntimeError>(py), "{err}");
            assert!(err.to_string().contains("boom"), "{err}");
            // The first probe advanced the capsule half way, the error undoes it.
            assert_eq!(ctrl.position(), [0.0, 1.0, 0.0]);
            assert_eq!(ctrl.velocity(), [0.5, 0.0, 0.0]);
            assert!(!ctrl.is_grounded());
            // After the failure Python is not called again within the move.
            let calls = globals
                .get_item("calls")
                .expect("lookup")
                .expect("calls is defined");
            assert_eq!(calls.len().expect("a list"), 2);
        });
    }

    #[test]
    fn test_move_and_slide_rejects_bad_arguments_and_results() {
        with_python(|py| {
            let mut ctrl = PyCharacterController::new([0.0, 1.0, 0.0], 0.4, 0.9);
            let not_callable = 5i32.into_pyobject(py).expect("an int").into_any();
            let err = ctrl
                .move_and_slide([1.0, 0.0, 0.0], &not_callable)
                .expect_err("an int is not a sweep callable");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            let free = python_sweep(py, FREE);
            let err = ctrl
                .move_and_slide([f64::NAN, 0.0, 0.0], &free)
                .expect_err("a NaN delta");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let err = ctrl
                .move_and_slide([0.0, f64::INFINITY, 0.0], &free)
                .expect_err("an infinite delta");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let wrong_type: [&CStr; 3] = [
                c"def sweep(o, d):\n    return 5\n",
                c"def sweep(o, d):\n    return (0.5, 1.0, 2.0)\n",
                c"def sweep(o, d):\n    return 'ab'\n",
            ];
            for source in wrong_type {
                let err = ctrl
                    .move_and_slide([1.0, 0.0, 0.0], &python_sweep(py, source))
                    .expect_err("neither None nor a (toi, normal) pair");
                assert!(err.is_instance_of::<PyTypeError>(py), "{source:?}: {err}");
            }
            let wrong_value: [&CStr; 4] = [
                c"def sweep(o, d):\n    return (float('nan'), (0.0, 1.0, 0.0))\n",
                c"def sweep(o, d):\n    return (0.5, (0.0, 0.0, 0.0))\n",
                c"def sweep(o, d):\n    return (0.5, (0.0, float('inf'), 0.0))\n",
                c"def sweep(o, d):\n    return (0.5, (0.0, 1.0))\n",
            ];
            for source in wrong_value {
                let err = ctrl
                    .move_and_slide([1.0, 0.0, 0.0], &python_sweep(py, source))
                    .expect_err("a malformed hit");
                assert!(err.is_instance_of::<PyValueError>(py), "{source:?}: {err}");
            }
            assert_eq!(
                ctrl.position(),
                [0.0, 1.0, 0.0],
                "no failed call moved the capsule"
            );
        });
    }
}
