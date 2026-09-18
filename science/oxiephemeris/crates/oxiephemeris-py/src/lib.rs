//! Python bindings for `OxiEphemeris`.
//!
//! A thin `pyo3` wrapper over the [`oxiephemeris_chart`] facade: every
//! function parses its arguments, calls the shared compute layer, and
//! returns either a Python `dict` (the stable JSON view) or a `str` (RDF
//! Turtle / N-Triples). The astronomy, the astrology, and the Linked Open
//! Data all live in the facade, so Python results are bit-identical to the
//! CLI's and the WASM binding's.
//!
//! DE ephemeris data is passed as `bytes` (read the DE440/DE441 file in
//! Python and hand the buffer in), so the binding needs no filesystem
//! access and works anywhere the data can be loaded.
#![forbid(unsafe_code)]
// `#[pyfunction]` signatures receive their `String`/owned arguments by
// value (pyo3 extracts them from Python objects), so passing them on by
// value to the facade is the natural shape, not a needless move. And a
// chart request legitimately has many optional knobs (system, sidereal,
// rulership, dut1, base_iri, …), each a keyword argument, so the arity is
// the API, not accidental complexity.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

use oxiephemeris_chart::calendar::CalendarKind;
use oxiephemeris_chart::comparison::PersonInput;
use oxiephemeris_chart::error::ChartError;
use oxiephemeris_chart::request::{parse_house_system, NatalSpec};
use oxiephemeris_chart::{
    chart_to_json, chart_to_rdf, comparison_to_json, comparison_to_rdf, composite_to_json,
    composite_to_rdf, natal_chart, RdfFormat,
};
use oxiephemeris_core::time::{julday as core_julday, revjul as core_revjul, JulianDate};
use oxiephemeris_de::DeFile;

/// Maps a facade error to a Python `ValueError`.
fn err(e: ChartError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Maps any `Display` error to a Python `ValueError`.
fn verr(msg: impl core::fmt::Display) -> PyErr {
    PyValueError::new_err(msg.to_string())
}

/// Parses a DE ephemeris buffer.
fn parse_de(bytes: &[u8]) -> PyResult<DeFile<'_>> {
    DeFile::parse(bytes).map_err(verr)
}

/// One person's birth data, read from either a Python mapping
/// (`{"date": ..., "lat": ..., "lon": ...}`) or an object exposing the
/// same names as attributes.
struct PersonArg {
    date: String,
    lat: f64,
    lon: f64,
}

/// Reads a named field, preferring subscription (`obj["name"]`, for
/// dicts) and falling back to attribute access (`obj.name`, for plain
/// objects).
fn person_field<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    match obj.get_item(name) {
        Ok(item) => Ok(item),
        Err(_) => obj.getattr(name),
    }
}

impl PersonArg {
    /// Extracts a `PersonArg` from a dict-or-object argument.
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            date: person_field(obj, "date")?.extract()?,
            lat: person_field(obj, "lat")?.extract()?,
            lon: person_field(obj, "lon")?.extract()?,
        })
    }
}

impl PersonArg {
    fn to_input(&self, cal: CalendarKind) -> PersonInput {
        PersonInput {
            date: self.date.clone(),
            cal,
            lat_deg: self.lat,
            lon_deg: self.lon,
        }
    }
}

/// Julian Date of a calendar date/time.
///
/// `calendar` is `"gregorian"` (default) or `"julian"`.
#[pyfunction]
#[pyo3(signature = (year, month, day, hours=0.0, calendar="gregorian"))]
fn julday(year: i32, month: u8, day: u8, hours: f64, calendar: &str) -> PyResult<f64> {
    let cal = CalendarKind::parse(calendar).map_err(err)?.to_core();
    Ok(core_julday(cal, year, month, day, hours)
        .map_err(verr)?
        .value())
}

/// Calendar date/time of a Julian Date, as a dict
/// `{year, month, day, hours}`.
#[pyfunction]
#[pyo3(signature = (jd, calendar="gregorian"))]
fn revjul<'py>(py: Python<'py>, jd: f64, calendar: &str) -> PyResult<Bound<'py, PyAny>> {
    let cal = CalendarKind::parse(calendar).map_err(err)?.to_core();
    let (date, hours) = core_revjul(JulianDate::new(jd, 0.0), cal).map_err(verr)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("year", date.year)?;
    dict.set_item("month", date.month)?;
    dict.set_item("day", date.day)?;
    dict.set_item("hours", hours)?;
    Ok(dict.into_any())
}

/// Builds and resolves a natal request from Python kwargs.
#[allow(clippy::too_many_arguments)]
fn natal_request(
    date: String,
    lat: f64,
    lon: f64,
    alt: f64,
    dut1: f64,
    cal: String,
    system: String,
    sidereal: Option<String>,
    rulership: String,
) -> PyResult<oxiephemeris_chart::NatalRequest> {
    NatalSpec {
        date,
        lat,
        lon,
        alt,
        dut1,
        cal,
        system,
        sidereal,
        rulership,
    }
    .into_request()
    .map_err(err)
}

/// Computes a natal chart, returning the stable JSON view as a dict.
#[pyfunction]
#[pyo3(signature = (de, date, lat, lon, *, alt=0.0, dut1=0.0, cal="gregorian".to_owned(),
                    system="placidus".to_owned(), sidereal=None, rulership="traditional".to_owned()))]
#[allow(clippy::too_many_arguments)]
fn natal<'py>(
    py: Python<'py>,
    de: &Bound<'py, PyBytes>,
    date: String,
    lat: f64,
    lon: f64,
    alt: f64,
    dut1: f64,
    cal: String,
    system: String,
    sidereal: Option<String>,
    rulership: String,
) -> PyResult<Bound<'py, PyAny>> {
    let req = natal_request(date, lat, lon, alt, dut1, cal, system, sidereal, rulership)?;
    let de = parse_de(de.as_bytes())?;
    let chart = natal_chart(&de, &req).map_err(err)?;
    let json = chart_to_json(&chart).map_err(err)?;
    json_to_py(py, &json)
}

/// Computes a natal chart, returning RDF Turtle (or N-Triples).
#[pyfunction]
#[pyo3(signature = (de, date, lat, lon, *, alt=0.0, dut1=0.0, cal="gregorian".to_owned(),
                    system="placidus".to_owned(), sidereal=None, rulership="traditional".to_owned(),
                    base_iri=None, ntriples=false))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn natal_rdf(
    de: &Bound<'_, PyBytes>,
    date: String,
    lat: f64,
    lon: f64,
    alt: f64,
    dut1: f64,
    cal: String,
    system: String,
    sidereal: Option<String>,
    rulership: String,
    base_iri: Option<String>,
    ntriples: bool,
) -> PyResult<String> {
    let req = natal_request(date, lat, lon, alt, dut1, cal, system, sidereal, rulership)?;
    let de = parse_de(de.as_bytes())?;
    let chart = natal_chart(&de, &req).map_err(err)?;
    chart_to_rdf(&chart, base_iri.as_deref(), None, fmt(ntriples)).map_err(err)
}

/// Synastry cross-aspects between two natal charts, as a dict.
#[pyfunction]
#[pyo3(signature = (de, a, b, *, system="placidus".to_owned(), dut1=0.0, cal="gregorian".to_owned()))]
fn synastry<'py>(
    py: Python<'py>,
    de: &Bound<'py, PyBytes>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    system: String,
    dut1: f64,
    cal: String,
) -> PyResult<Bound<'py, PyAny>> {
    let a = PersonArg::from_py(a)?;
    let b = PersonArg::from_py(b)?;
    let (de, cmp) = synastry_model(de, &a, &b, &system, dut1, &cal)?;
    let _ = de;
    let json = comparison_to_json(&cmp).map_err(err)?;
    json_to_py(py, &json)
}

/// Synastry as RDF Turtle (or N-Triples).
#[pyfunction]
#[pyo3(signature = (de, a, b, *, system="placidus".to_owned(), dut1=0.0, cal="gregorian".to_owned(),
                    base_iri=None, ntriples=false))]
#[allow(clippy::too_many_arguments)]
fn synastry_rdf(
    de: &Bound<'_, PyBytes>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    system: String,
    dut1: f64,
    cal: String,
    base_iri: Option<String>,
    ntriples: bool,
) -> PyResult<String> {
    let a = PersonArg::from_py(a)?;
    let b = PersonArg::from_py(b)?;
    let (_de, cmp) = synastry_model(de, &a, &b, &system, dut1, &cal)?;
    comparison_to_rdf(&cmp, base_iri.as_deref(), fmt(ntriples)).map_err(err)
}

/// Transit aspects to a natal chart, as a dict.
#[pyfunction]
#[pyo3(signature = (de, natal, transit, *, system="placidus".to_owned(), dut1=0.0, cal="gregorian".to_owned()))]
fn transit<'py>(
    py: Python<'py>,
    de: &Bound<'py, PyBytes>,
    natal: &Bound<'py, PyAny>,
    transit: String,
    system: String,
    dut1: f64,
    cal: String,
) -> PyResult<Bound<'py, PyAny>> {
    let natal = PersonArg::from_py(natal)?;
    let cal_kind = CalendarKind::parse(&cal).map_err(err)?;
    let sys = parse_house_system(&system).map_err(err)?;
    let de = parse_de(de.as_bytes())?;
    let cmp = oxiephemeris_chart::transit(
        &de,
        &natal.to_input(cal_kind),
        &transit,
        cal_kind,
        sys,
        dut1,
    )
    .map_err(err)?;
    let json = comparison_to_json(&cmp).map_err(err)?;
    json_to_py(py, &json)
}

/// Secondary-progressed chart aspects to the natal, as a dict.
#[pyfunction]
#[pyo3(signature = (de, natal, target, *, system="placidus".to_owned(), dut1=0.0, cal="gregorian".to_owned()))]
fn progress<'py>(
    py: Python<'py>,
    de: &Bound<'py, PyBytes>,
    natal: &Bound<'py, PyAny>,
    target: String,
    system: String,
    dut1: f64,
    cal: String,
) -> PyResult<Bound<'py, PyAny>> {
    let natal = PersonArg::from_py(natal)?;
    let cal_kind = CalendarKind::parse(&cal).map_err(err)?;
    let sys = parse_house_system(&system).map_err(err)?;
    let de = parse_de(de.as_bytes())?;
    let cmp = oxiephemeris_chart::progression(
        &de,
        &natal.to_input(cal_kind),
        &target,
        cal_kind,
        sys,
        dut1,
    )
    .map_err(err)?;
    let json = comparison_to_json(&cmp).map_err(err)?;
    json_to_py(py, &json)
}

/// Midpoint composite chart of two natal charts, as a dict.
#[pyfunction]
#[pyo3(signature = (de, a, b, *, system="placidus".to_owned(), dut1=0.0, cal="gregorian".to_owned(), base_iri=None))]
fn composite<'py>(
    py: Python<'py>,
    de: &Bound<'py, PyBytes>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    system: String,
    dut1: f64,
    cal: String,
    base_iri: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let a = PersonArg::from_py(a)?;
    let b = PersonArg::from_py(b)?;
    let cal_kind = CalendarKind::parse(&cal).map_err(err)?;
    let sys = parse_house_system(&system).map_err(err)?;
    let de = parse_de(de.as_bytes())?;
    let comp = oxiephemeris_chart::composite(
        &de,
        &a.to_input(cal_kind),
        &b.to_input(cal_kind),
        sys,
        dut1,
        base_iri.as_deref(),
    )
    .map_err(err)?;
    let json = composite_to_json(&comp).map_err(err)?;
    json_to_py(py, &json)
}

/// Composite chart as RDF Turtle (or N-Triples).
#[pyfunction]
#[pyo3(signature = (de, a, b, *, system="placidus".to_owned(), dut1=0.0, cal="gregorian".to_owned(),
                    base_iri=None, ntriples=false))]
#[allow(clippy::too_many_arguments)]
fn composite_rdf(
    de: &Bound<'_, PyBytes>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    system: String,
    dut1: f64,
    cal: String,
    base_iri: Option<String>,
    ntriples: bool,
) -> PyResult<String> {
    let a = PersonArg::from_py(a)?;
    let b = PersonArg::from_py(b)?;
    let cal_kind = CalendarKind::parse(&cal).map_err(err)?;
    let sys = parse_house_system(&system).map_err(err)?;
    let de = parse_de(de.as_bytes())?;
    let comp = oxiephemeris_chart::composite(
        &de,
        &a.to_input(cal_kind),
        &b.to_input(cal_kind),
        sys,
        dut1,
        base_iri.as_deref(),
    )
    .map_err(err)?;
    composite_to_rdf(&comp, base_iri.as_deref(), fmt(ntriples)).map_err(err)
}

/// Shared synastry computation (returns the DE handle too, to keep it
/// alive for the borrow).
fn synastry_model<'a>(
    de: &'a Bound<'_, PyBytes>,
    a: &PersonArg,
    b: &PersonArg,
    system: &str,
    dut1: f64,
    cal: &str,
) -> PyResult<(DeFile<'a>, oxiephemeris_rdf::model::ChartComparison)> {
    let cal_kind = CalendarKind::parse(cal).map_err(err)?;
    let sys = parse_house_system(system).map_err(err)?;
    let de_file = parse_de(de.as_bytes())?;
    let cmp = oxiephemeris_chart::synastry(
        &de_file,
        &a.to_input(cal_kind),
        &b.to_input(cal_kind),
        sys,
        dut1,
    )
    .map_err(err)?;
    Ok((de_file, cmp))
}

/// `RdfFormat` from a `ntriples` flag.
const fn fmt(ntriples: bool) -> RdfFormat {
    if ntriples {
        RdfFormat::NTriples
    } else {
        RdfFormat::Turtle
    }
}

/// Parses a JSON string into a Python object.
fn json_to_py<'py>(py: Python<'py>, json: &str) -> PyResult<Bound<'py, PyAny>> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(verr)?;
    pythonize::pythonize(py, &value).map_err(PyErr::from)
}

/// The `oxiephemeris` Python module.
#[pymodule]
fn oxiephemeris(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(julday, m)?)?;
    m.add_function(wrap_pyfunction!(revjul, m)?)?;
    m.add_function(wrap_pyfunction!(natal, m)?)?;
    m.add_function(wrap_pyfunction!(natal_rdf, m)?)?;
    m.add_function(wrap_pyfunction!(synastry, m)?)?;
    m.add_function(wrap_pyfunction!(synastry_rdf, m)?)?;
    m.add_function(wrap_pyfunction!(transit, m)?)?;
    m.add_function(wrap_pyfunction!(progress, m)?)?;
    m.add_function(wrap_pyfunction!(composite, m)?)?;
    m.add_function(wrap_pyfunction!(composite_rdf, m)?)?;
    Ok(())
}
