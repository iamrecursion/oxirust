// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the oxiphysics event_bus module.
//!
//! Exposes physics event publish/subscribe functionality to Python.
//!
//! Python code queues events with [`PyEventBus::publish`], registers listener
//! callables with [`PyEventBus::subscribe`] and delivers the queued events to
//! them with [`PyEventBus::flush`]. A listener receives one `dict` per event:
//! `"kind"` holds the event kind and the other keys are the fields of the
//! event, the same names `publish` takes.
//!
//! | `kind`            | fields                                                          |
//! |-------------------|-----------------------------------------------------------------|
//! | `CollisionBegin`  | `body_a`, `body_b`, `contact_normal` (3 floats), `penetration_depth` |
//! | `CollisionEnd`    | `body_a`, `body_b`                                              |
//! | `BodySleep`       | `body`                                                          |
//! | `BodyWake`        | `body`                                                          |
//! | `ConstraintBreak` | `constraint_id`, `accumulated_impulse`                          |
//! | `BodyEnterRegion` | `body`, `region_id`                                             |
//! | `BodyLeaveRegion` | `body`, `region_id`                                             |
//! | `StepCompleted`   | `step`, `sim_time`, `dt`                                        |
//! | `UserDefined`     | `id`, `data` (`bytes`)                                          |
//!
//! Note: `EventBus` stores `Box<dyn Fn>` listeners which are `!Sync`.
//! The pyclass is therefore marked `unsendable` — it cannot be shared
//! across Python threads.

use oxiphysics::event_bus::{EventBus, ListenerId, PhysicsEvent};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;

/// The event kinds `publish` accepts and the fields each one takes.
const EVENT_KINDS: [(&str, &[&str]); 9] = [
    (
        "CollisionBegin",
        &["body_a", "body_b", "contact_normal", "penetration_depth"],
    ),
    ("CollisionEnd", &["body_a", "body_b"]),
    ("BodySleep", &["body"]),
    ("BodyWake", &["body"]),
    ("ConstraintBreak", &["constraint_id", "accumulated_impulse"]),
    ("BodyEnterRegion", &["body", "region_id"]),
    ("BodyLeaveRegion", &["body", "region_id"]),
    ("StepCompleted", &["step", "sim_time", "dt"]),
    ("UserDefined", &["id", "data"]),
];

/// The fields of an event given to `publish` as keyword arguments.
struct EventFields<'a, 'py> {
    kind: &'a str,
    fields: Option<&'a Bound<'py, PyDict>>,
}

impl<'a, 'py> EventFields<'a, 'py> {
    /// Fields of an event of kind `kind` that has the field names `allowed`.
    ///
    /// Raises `TypeError` for a field that the kind does not have.
    fn new(
        kind: &'a str,
        allowed: &[&str],
        fields: Option<&'a Bound<'py, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(dict) = fields {
            for key in dict.keys() {
                let name: String = key.extract()?;
                if !allowed.contains(&name.as_str()) {
                    return Err(PyTypeError::new_err(format!(
                        "a {kind} event has no field '{name}', its fields are {}",
                        allowed.join(", ")
                    )));
                }
            }
        }
        Ok(Self { kind, fields })
    }

    /// The field `name` converted to `T`.
    ///
    /// Raises `TypeError` if the field is missing; a value that does not
    /// convert raises the error of the conversion (`TypeError`,
    /// `OverflowError`, ...) with the field named in its message.
    fn get<T: FromPyObjectOwned<'py>>(&self, name: &str) -> PyResult<T> {
        let kind = self.kind;
        let Some(value) = self
            .fields
            .map(|dict| dict.get_item(name))
            .transpose()?
            .flatten()
        else {
            return Err(PyTypeError::new_err(format!(
                "a {kind} event needs the field '{name}'"
            )));
        };
        value.extract::<T>().map_err(|err| {
            let err: PyErr = err.into();
            let py = value.py();
            PyErr::from_type(
                err.get_type(py),
                format!("field '{name}' of a {kind} event: {}", err.value(py)),
            )
        })
    }
}

/// Build the engine event of kind `kind` from the keyword arguments of `publish`.
///
/// Raises `ValueError` for an unknown kind and `TypeError` for a missing,
/// unknown or mistyped field.
fn event_from_python(kind: &str, fields: Option<&Bound<'_, PyDict>>) -> PyResult<PhysicsEvent> {
    let Some((_, allowed)) = EVENT_KINDS.iter().find(|(name, _)| *name == kind) else {
        let kinds: Vec<&str> = EVENT_KINDS.iter().map(|(name, _)| *name).collect();
        return Err(PyValueError::new_err(format!(
            "unknown event kind '{kind}', the kinds are {}",
            kinds.join(", ")
        )));
    };
    let fields = EventFields::new(kind, allowed, fields)?;
    let event = match kind {
        "CollisionBegin" => PhysicsEvent::CollisionBegin {
            body_a: fields.get("body_a")?,
            body_b: fields.get("body_b")?,
            contact_normal: fields.get("contact_normal")?,
            penetration_depth: fields.get("penetration_depth")?,
        },
        "CollisionEnd" => PhysicsEvent::CollisionEnd {
            body_a: fields.get("body_a")?,
            body_b: fields.get("body_b")?,
        },
        "BodySleep" => PhysicsEvent::BodySleep {
            body: fields.get("body")?,
        },
        "BodyWake" => PhysicsEvent::BodyWake {
            body: fields.get("body")?,
        },
        "ConstraintBreak" => PhysicsEvent::ConstraintBreak {
            constraint_id: fields.get("constraint_id")?,
            accumulated_impulse: fields.get("accumulated_impulse")?,
        },
        "BodyEnterRegion" => PhysicsEvent::BodyEnterRegion {
            body: fields.get("body")?,
            region_id: fields.get("region_id")?,
        },
        "BodyLeaveRegion" => PhysicsEvent::BodyLeaveRegion {
            body: fields.get("body")?,
            region_id: fields.get("region_id")?,
        },
        "StepCompleted" => PhysicsEvent::StepCompleted {
            step: fields.get("step")?,
            sim_time: fields.get("sim_time")?,
            dt: fields.get("dt")?,
        },
        _ => PhysicsEvent::UserDefined {
            id: fields.get("id")?,
            data: fields.get("data")?,
        },
    };
    Ok(event)
}

/// The `dict` a listener receives for `event`: `"kind"` plus the event's fields.
fn event_to_python<'py>(py: Python<'py>, event: &PhysicsEvent) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("kind", event.kind_name())?;
    match event {
        PhysicsEvent::CollisionBegin {
            body_a,
            body_b,
            contact_normal,
            penetration_depth,
        } => {
            dict.set_item("body_a", body_a)?;
            dict.set_item("body_b", body_b)?;
            dict.set_item("contact_normal", contact_normal)?;
            dict.set_item("penetration_depth", penetration_depth)?;
        }
        PhysicsEvent::CollisionEnd { body_a, body_b } => {
            dict.set_item("body_a", body_a)?;
            dict.set_item("body_b", body_b)?;
        }
        PhysicsEvent::BodySleep { body } | PhysicsEvent::BodyWake { body } => {
            dict.set_item("body", body)?;
        }
        PhysicsEvent::ConstraintBreak {
            constraint_id,
            accumulated_impulse,
        } => {
            dict.set_item("constraint_id", constraint_id)?;
            dict.set_item("accumulated_impulse", accumulated_impulse)?;
        }
        PhysicsEvent::BodyEnterRegion { body, region_id }
        | PhysicsEvent::BodyLeaveRegion { body, region_id } => {
            dict.set_item("body", body)?;
            dict.set_item("region_id", region_id)?;
        }
        PhysicsEvent::StepCompleted { step, sim_time, dt } => {
            dict.set_item("step", step)?;
            dict.set_item("sim_time", sim_time)?;
            dict.set_item("dt", dt)?;
        }
        PhysicsEvent::UserDefined { id, data } => {
            dict.set_item("id", id)?;
            dict.set_item("data", PyBytes::new(py, data))?;
        }
    }
    Ok(dict)
}

// ─────────────────────────────────────────────────────────────────────────────
// PyEventBus
// ─────────────────────────────────────────────────────────────────────────────

/// Physics event bus for publish/subscribe patterns.
///
/// `publish(kind, **fields)` queues an event, `subscribe(callback)` registers
/// a listener and `flush()` delivers the queued events to the listeners in
/// subscription order (see the module documentation for the kinds and fields).
///
/// Note: this object is **not thread-safe** (unsendable) because it stores
/// listener closures that are `!Sync`.
#[pyclass(name = "EventBus", unsendable)]
pub struct PyEventBus {
    inner: EventBus,
    /// Subscription ids given to Python and the engine listener behind each.
    subscriptions: HashMap<u64, ListenerId>,
    /// The id the next subscription gets.
    next_subscription: u64,
}

impl Default for PyEventBus {
    fn default() -> Self {
        Self {
            inner: EventBus::new(),
            subscriptions: HashMap::new(),
            next_subscription: 0,
        }
    }
}

#[pymethods]
impl PyEventBus {
    /// Create a new event bus.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue an event of the given kind; `flush()` delivers it to the listeners.
    ///
    /// The fields are keyword arguments, for example
    /// `bus.publish("BodySleep", body=3)` or
    /// `bus.publish("CollisionBegin", body_a=0, body_b=1,
    /// contact_normal=[0, 1, 0], penetration_depth=0.01)`; the module
    /// documentation lists every kind with its fields.
    ///
    /// Raises `ValueError` for an unknown kind and `TypeError` for a missing,
    /// unknown or mistyped field. Nothing is queued after an error.
    #[pyo3(signature = (kind, **fields))]
    pub fn publish(&mut self, kind: &str, fields: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let event = event_from_python(kind, fields)?;
        self.inner.publish(event);
        Ok(())
    }

    /// Register `callback(event)` and return its subscription id.
    ///
    /// `flush()` calls the callback once per queued event, in the order the
    /// events were published, with a `dict` that holds the event `"kind"` and its
    /// fields. An exception the callback raises is printed (`sys.excepthook`) and
    /// does not stop the delivery to the other listeners. A callback must not
    /// call methods of the bus it is registered with while `flush()` runs: the
    /// bus is borrowed then, so such a call raises `RuntimeError` inside the
    /// callback.
    ///
    /// Raises `TypeError` if `callback` is not callable.
    pub fn subscribe(&mut self, callback: &Bound<'_, PyAny>) -> PyResult<u64> {
        if !callback.is_callable() {
            return Err(PyTypeError::new_err("callback must be callable"));
        }
        let callback: Py<PyAny> = callback.clone().unbind();
        let listener = self.inner.subscribe(move |event: &PhysicsEvent| {
            Python::attach(|py| {
                let outcome =
                    event_to_python(py, event).and_then(|dict| callback.call1(py, (dict,)));
                if let Err(err) = outcome {
                    err.print(py);
                }
            });
        });
        let id = self.next_subscription;
        self.next_subscription += 1;
        self.subscriptions.insert(id, listener);
        Ok(id)
    }

    /// Remove the listener with the subscription id `subscription_id` that
    /// `subscribe` returned. Returns `True` if it was registered.
    pub fn unsubscribe(&mut self, subscription_id: u64) -> bool {
        match self.subscriptions.remove(&subscription_id) {
            Some(listener) => self.inner.unsubscribe(listener),
            None => false,
        }
    }

    /// Number of pending (unflushed) events.
    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    /// Number of registered listeners.
    pub fn listener_count(&self) -> usize {
        self.inner.listener_count()
    }

    /// Total number of events ever published.
    pub fn total_published(&self) -> u64 {
        self.inner.total_published()
    }

    /// Total number of events dispatched to listeners.
    pub fn total_dispatched(&self) -> u64 {
        self.inner.total_dispatched()
    }

    /// Drain and return all pending events as a list of description strings.
    ///
    /// Each string is the `Display` representation of the event.
    /// The internal queue is cleared.
    pub fn drain_as_strings(&mut self) -> Vec<String> {
        self.inner
            .drain()
            .into_iter()
            .map(|e| e.to_string())
            .collect()
    }

    /// Return the kind names of all pending events without consuming them.
    pub fn pending_kind_names(&self) -> Vec<String> {
        self.inner
            .pending()
            .iter()
            .map(|e| e.kind_name().to_owned())
            .collect()
    }

    /// Clear all pending events without dispatching them.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Flush all pending events to registered listeners, in the order the
    /// events were published and the listeners were subscribed.
    pub fn flush(&mut self) {
        self.inner.flush();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEventBus>()?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyOverflowError;
    use pyo3::types::PyList;
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

    /// The global `name` of a namespace.
    fn global<'py>(globals: &Bound<'py, PyDict>, name: &str) -> Bound<'py, PyAny> {
        globals
            .get_item(name)
            .expect("globals lookup")
            .unwrap_or_else(|| panic!("the source defines {name}"))
    }

    /// The dict a Python expression evaluates to.
    fn dict_of<'py>(py: Python<'py>, expression: &CStr) -> Bound<'py, PyDict> {
        py.eval(expression, None, None)
            .expect("the expression evaluates")
            .cast_into::<PyDict>()
            .expect("the expression is a dict")
    }

    /// Source of a listener `collect` that appends every event to `received`.
    const COLLECTOR: &CStr = c"received = []\ndef collect(event):\n    received.append(event)\n";

    /// Replace `sys.stderr` by a buffer; returns the buffer and the old stream.
    fn capture_stderr<'py>(py: Python<'py>) -> (Bound<'py, PyAny>, Bound<'py, PyAny>) {
        let sys = py.import("sys").expect("sys");
        let buffer = py
            .import("io")
            .and_then(|io| io.getattr("StringIO"))
            .and_then(|class| class.call0())
            .expect("an io.StringIO");
        let previous = sys.getattr("stderr").expect("sys.stderr");
        sys.setattr("stderr", &buffer).expect("replace sys.stderr");
        (buffer, previous)
    }

    /// Put the stream `capture_stderr` replaced back and return what was written.
    fn restore_stderr(
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        previous: Bound<'_, PyAny>,
    ) -> String {
        py.import("sys")
            .and_then(|sys| sys.setattr("stderr", previous))
            .expect("restore sys.stderr");
        buffer
            .call_method0("getvalue")
            .and_then(|text| text.extract::<String>())
            .expect("the captured text")
    }

    #[test]
    fn test_event_bus_instantiation() {
        let mut bus = PyEventBus::new();
        assert_eq!(bus.pending_count(), 0);
        assert_eq!(bus.listener_count(), 0);
        assert_eq!(bus.total_published(), 0);
        bus.clear();
        assert_eq!(bus.pending_count(), 0);
    }

    #[test]
    fn test_event_bus_drain_empty() {
        let mut bus = PyEventBus::new();
        let strings = bus.drain_as_strings();
        assert!(strings.is_empty());
    }

    #[test]
    fn test_subscribed_callback_fires_once_after_publish_and_flush() {
        with_python(|py| {
            let globals = run_source(py, COLLECTOR);
            let collect = global(&globals, "collect");
            let received = global(&globals, "received");
            let mut bus = PyEventBus::new();
            let id = bus.subscribe(&collect).expect("a callable");
            assert_eq!(id, 0);
            assert_eq!(bus.listener_count(), 1);
            let fields = dict_of(py, c"{'body': 3}");
            bus.publish("BodySleep", Some(&fields))
                .expect("a valid event");
            assert_eq!(bus.pending_count(), 1);
            assert_eq!(bus.pending_kind_names(), ["BodySleep"]);
            assert_eq!(
                received.len().expect("a list"),
                0,
                "flush delivers, publish only queues"
            );
            bus.flush();
            assert_eq!(received.len().expect("a list"), 1, "exactly one delivery");
            let event = received.get_item(0).expect("the event");
            let expected = dict_of(py, c"{'kind': 'BodySleep', 'body': 3}");
            assert!(event.eq(&expected).expect("dict comparison"), "{event}");
            assert_eq!(bus.pending_count(), 0);
            assert_eq!(bus.total_published(), 1);
            assert_eq!(bus.total_dispatched(), 1);
            // A second flush has nothing left to deliver.
            bus.flush();
            assert_eq!(received.len().expect("a list"), 1);
        });
    }

    #[test]
    fn test_every_event_kind_reaches_the_listener_with_its_fields() {
        let cases: [(&str, &CStr); 9] = [
            (
                "CollisionBegin",
                c"{'body_a': 1, 'body_b': 2, 'contact_normal': [0.0, 1.0, 0.0], 'penetration_depth': 0.25}",
            ),
            ("CollisionEnd", c"{'body_a': 1, 'body_b': 2}"),
            ("BodySleep", c"{'body': 7}"),
            ("BodyWake", c"{'body': 7}"),
            (
                "ConstraintBreak",
                c"{'constraint_id': 4, 'accumulated_impulse': 12.5}",
            ),
            ("BodyEnterRegion", c"{'body': 3, 'region_id': 9}"),
            ("BodyLeaveRegion", c"{'body': 3, 'region_id': 9}"),
            (
                "StepCompleted",
                c"{'step': 42, 'sim_time': 0.7, 'dt': 0.016666666666666666}",
            ),
            ("UserDefined", c"{'id': 5, 'data': b'\\x00\\x01\\xff'}"),
        ];
        with_python(|py| {
            let globals = run_source(py, COLLECTOR);
            let collect = global(&globals, "collect");
            let received = global(&globals, "received");
            let mut bus = PyEventBus::new();
            bus.subscribe(&collect).expect("a callable");
            let mut expected_events = Vec::new();
            for (kind, literal) in cases {
                let fields = dict_of(py, literal);
                bus.publish(kind, Some(&fields))
                    .unwrap_or_else(|err| panic!("{kind}: {err}"));
                let expected = fields.copy().expect("a copy of the fields");
                expected.set_item("kind", kind).expect("kind key");
                expected_events.push(expected);
            }
            let kinds: Vec<&str> = cases.iter().map(|(kind, _)| *kind).collect();
            assert_eq!(bus.pending_kind_names(), kinds);
            bus.flush();
            assert_eq!(received.len().expect("a list"), cases.len());
            for (index, expected) in expected_events.iter().enumerate() {
                let event = received.get_item(index).expect("a delivered event");
                assert!(
                    event.eq(expected).expect("dict comparison"),
                    "event {index}: {event} vs {expected}"
                );
            }
            assert_eq!(bus.total_dispatched(), cases.len() as u64);
        });
    }

    #[test]
    fn test_published_events_use_the_engine_event_kinds() {
        with_python(|py| {
            let mut bus = PyEventBus::new();
            let fields = dict_of(
                py,
                c"{'body_a': 0, 'body_b': 1, 'contact_normal': [0.0, 1.0, 0.0], 'penetration_depth': 0.01}",
            );
            bus.publish("CollisionBegin", Some(&fields))
                .expect("a valid event");
            let step = dict_of(py, c"{'step': 1, 'sim_time': 0.5, 'dt': 0.25}");
            bus.publish("StepCompleted", Some(&step))
                .expect("a valid event");
            assert_eq!(
                bus.drain_as_strings(),
                [
                    "CollisionBegin(0↔1, depth=0.0100m)",
                    "StepCompleted(step=1, t=0.5000s, dt=0.250000s)"
                ]
            );
        });
    }

    #[test]
    fn test_publish_rejects_unknown_kinds_and_bad_fields() {
        with_python(|py| {
            let mut bus = PyEventBus::new();
            let err = bus.publish("Explosion", None).expect_err("an unknown kind");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            let message = err.to_string();
            assert!(
                message.contains("Explosion") && message.contains("BodySleep"),
                "{message}"
            );
            let missing = bus
                .publish("BodySleep", None)
                .expect_err("no fields at all");
            assert!(missing.is_instance_of::<PyTypeError>(py), "{missing}");
            assert!(missing.to_string().contains("'body'"), "{missing}");
            let partial = dict_of(py, c"{'body_a': 1}");
            let err = bus
                .publish("CollisionEnd", Some(&partial))
                .expect_err("body_b is missing");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            assert!(err.to_string().contains("'body_b'"), "{err}");
            let extra = dict_of(py, c"{'body': 1, 'extra': 2}");
            let err = bus
                .publish("BodySleep", Some(&extra))
                .expect_err("an unknown field");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            assert!(err.to_string().contains("'extra'"), "{err}");
            let mistyped = dict_of(py, c"{'body': 'three'}");
            let err = bus
                .publish("BodySleep", Some(&mistyped))
                .expect_err("a str id");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            assert!(err.to_string().contains("'body'"), "{err}");
            let negative = dict_of(py, c"{'body': -1}");
            let err = bus
                .publish("BodySleep", Some(&negative))
                .expect_err("a negative id");
            assert!(err.is_instance_of::<PyOverflowError>(py), "{err}");
            let text_payload = dict_of(py, c"{'id': 1, 'data': 'text'}");
            let err = bus
                .publish("UserDefined", Some(&text_payload))
                .expect_err("text is not a payload");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            let short_normal = dict_of(
                py,
                c"{'body_a': 0, 'body_b': 1, 'contact_normal': [0.0, 1.0], 'penetration_depth': 0.0}",
            );
            let err = bus
                .publish("CollisionBegin", Some(&short_normal))
                .expect_err("a normal needs three components");
            assert!(err.is_instance_of::<PyValueError>(py), "{err}");
            assert_eq!(bus.pending_count(), 0, "nothing is queued after an error");
            assert_eq!(bus.total_published(), 0);
        });
    }

    #[test]
    fn test_listener_exceptions_are_printed_and_do_not_stop_the_delivery() {
        with_python(|py| {
            let globals = run_source(
                py,
                c"
received = []
def bad(event):
    raise ValueError('listener failed')
def good(event):
    received.append(event['kind'])
",
            );
            let mut bus = PyEventBus::new();
            bus.subscribe(&global(&globals, "bad")).expect("a callable");
            bus.subscribe(&global(&globals, "good"))
                .expect("a callable");
            let fields = dict_of(py, c"{'body': 1}");
            bus.publish("BodyWake", Some(&fields))
                .expect("a valid event");
            let (buffer, previous) = capture_stderr(py);
            bus.flush();
            let printed = restore_stderr(py, &buffer, previous);
            let received = global(&globals, "received");
            let expected = py.eval(c"['BodyWake']", None, None).expect("a list");
            assert!(
                received.eq(&expected).expect("list comparison"),
                "{received}"
            );
            assert!(printed.contains("ValueError"), "{printed:?}");
            assert!(printed.contains("listener failed"), "{printed:?}");
            assert_eq!(bus.total_dispatched(), 2, "both listeners were called");
        });
    }

    #[test]
    fn test_unsubscribe_stops_the_delivery() {
        with_python(|py| {
            let globals = run_source(py, COLLECTOR);
            let collect = global(&globals, "collect");
            let received = global(&globals, "received");
            let mut bus = PyEventBus::new();
            let id = bus.subscribe(&collect).expect("a callable");
            let other = bus.subscribe(&collect).expect("a callable");
            assert_ne!(id, other);
            let fields = dict_of(py, c"{'body': 2}");
            bus.publish("BodySleep", Some(&fields))
                .expect("a valid event");
            bus.flush();
            assert_eq!(
                received.len().expect("a list"),
                2,
                "both subscriptions deliver"
            );
            assert!(bus.unsubscribe(id));
            assert!(!bus.unsubscribe(id), "a second removal finds nothing");
            assert!(!bus.unsubscribe(999), "an unknown id finds nothing");
            assert_eq!(bus.listener_count(), 1);
            bus.publish("BodySleep", Some(&fields))
                .expect("a valid event");
            bus.flush();
            assert_eq!(
                received.len().expect("a list"),
                3,
                "only the other listener is left"
            );
            assert!(bus.unsubscribe(other));
            assert_eq!(bus.listener_count(), 0);
            bus.publish("BodySleep", Some(&fields))
                .expect("a valid event");
            bus.flush();
            assert_eq!(received.len().expect("a list"), 3);
        });
    }

    #[test]
    fn test_listeners_run_in_subscription_order_for_every_event() {
        with_python(|py| {
            let globals = run_source(
                py,
                c"
log = []
def first(event):
    log.append(('first', event['kind']))
def second(event):
    log.append(('second', event['kind']))
",
            );
            let mut bus = PyEventBus::new();
            bus.subscribe(&global(&globals, "first"))
                .expect("a callable");
            bus.subscribe(&global(&globals, "second"))
                .expect("a callable");
            let fields = dict_of(py, c"{'body': 1}");
            bus.publish("BodySleep", Some(&fields))
                .expect("a valid event");
            bus.publish("BodyWake", Some(&fields))
                .expect("a valid event");
            bus.flush();
            let log = global(&globals, "log");
            let expected = py
                .eval(
                    c"[('first', 'BodySleep'), ('second', 'BodySleep'), ('first', 'BodyWake'), ('second', 'BodyWake')]",
                    None,
                    None,
                )
                .expect("a list");
            assert!(log.eq(&expected).expect("list comparison"), "{log}");
            assert_eq!(bus.total_dispatched(), 4);
        });
    }

    #[test]
    fn test_subscribe_requires_a_callable() {
        with_python(|py| {
            let mut bus = PyEventBus::new();
            let not_callable = py.eval(c"42", None, None).expect("an int");
            let err = bus
                .subscribe(&not_callable)
                .expect_err("an int is not callable");
            assert!(err.is_instance_of::<PyTypeError>(py), "{err}");
            assert_eq!(bus.listener_count(), 0);
        });
    }

    #[test]
    fn test_a_listener_cannot_reenter_the_bus_during_flush() {
        with_python(|py| {
            let bus = Bound::new(py, PyEventBus::new()).expect("a Python object");
            let globals = PyDict::new(py);
            globals.set_item("bus", &bus).expect("a global");
            py.run(
                c"def reenter(event):\n    bus.publish('BodyWake', body=1)\n",
                Some(&globals),
                None,
            )
            .expect("the listener compiles");
            let reenter = global(&globals, "reenter");
            bus.call_method1("subscribe", (reenter,))
                .expect("a callable");
            let kwargs = PyDict::new(py);
            kwargs.set_item("body", 4).expect("a field");
            bus.call_method("publish", ("BodySleep",), Some(&kwargs))
                .expect("a valid event");
            let (buffer, previous) = capture_stderr(py);
            bus.call_method0("flush")
                .expect("flush survives the failing listener");
            let printed = restore_stderr(py, &buffer, previous);
            assert!(printed.contains("Already borrowed"), "{printed:?}");
            let pending: usize = bus
                .call_method0("pending_count")
                .and_then(|count| count.extract())
                .expect("a count");
            assert_eq!(pending, 0, "the re-entrant publish did not queue anything");
        });
    }

    #[test]
    fn test_publish_and_subscribe_from_python_keyword_arguments() {
        with_python(|py| {
            let globals = run_source(py, COLLECTOR);
            let received = global(&globals, "received");
            let bus = Bound::new(py, PyEventBus::new()).expect("a Python object");
            let id: u64 = bus
                .call_method1("subscribe", (global(&globals, "collect"),))
                .and_then(|id| id.extract())
                .expect("a subscription id");
            let kwargs = dict_of(py, c"{'constraint_id': 8, 'accumulated_impulse': 3.5}");
            bus.call_method("publish", ("ConstraintBreak",), Some(&kwargs))
                .expect("keyword arguments");
            bus.call_method0("flush").expect("flush");
            let expected = py
                .eval(
                    c"[{'kind': 'ConstraintBreak', 'constraint_id': 8, 'accumulated_impulse': 3.5}]",
                    None,
                    None,
                )
                .expect("a list");
            assert!(
                received.eq(&expected).expect("list comparison"),
                "{received}"
            );
            let removed: bool = bus
                .call_method1("unsubscribe", (id,))
                .and_then(|removed| removed.extract())
                .expect("a bool");
            assert!(removed);
            let list = received.cast_into::<PyList>().expect("a list");
            assert_eq!(list.len(), 1);
        });
    }
}
