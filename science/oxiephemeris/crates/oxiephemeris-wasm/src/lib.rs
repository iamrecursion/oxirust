//! WebAssembly bindings for `OxiEphemeris`.
//!
//! A thin `wasm-bindgen` wrapper over the [`oxiephemeris_chart`] facade,
//! so a browser can compute a full natal chart — signs, houses, aspects,
//! dignities, Arabic Parts, declinations — and its Linked Open Data
//! (Turtle) **entirely offline**, given a JPL DE ephemeris buffer.
//!
//! # The boundary is JSON strings
//!
//! Every function takes the DE bytes as a `Uint8Array` (`&[u8]`) plus a
//! JSON **request string**, and returns a JSON (or Turtle) **string**.
//! Strings cross the wasm/JS boundary robustly without a schema-coupled
//! binary layout, and the request/response shapes are exactly the facade's
//! stable JSON — the same the Python binding and the CLI emit.
#![forbid(unsafe_code)]
// `#[wasm_bindgen]` functions receive their `String`/`Option<String>`
// arguments by value across the JS boundary, so passing them on by value
// is the natural shape rather than a needless move.
#![allow(clippy::needless_pass_by_value)]

use wasm_bindgen::prelude::*;

use oxiephemeris_chart::calendar::CalendarKind;
use oxiephemeris_chart::comparison::PersonInput;
use oxiephemeris_chart::error::ChartError;
use oxiephemeris_chart::request::{parse_house_system, NatalSpec};
use oxiephemeris_chart::{
    chart_to_json, chart_to_rdf, comparison_to_json, comparison_to_rdf, composite_to_json,
    composite_to_rdf, natal_chart, RdfFormat,
};
use oxiephemeris_de::DeFile;

/// Installs a panic hook that logs Rust panics to the browser console.
/// Call once from JavaScript before the other functions for readable
/// diagnostics.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Maps a facade error to a `JsError`.
fn err(e: ChartError) -> JsError {
    JsError::new(&e.to_string())
}

/// Maps any `Display` error to a `JsError`.
fn jerr(msg: impl core::fmt::Display) -> JsError {
    JsError::new(&msg.to_string())
}

/// Deserializes a JSON request string of type `T`.
fn spec<T: serde::de::DeserializeOwned>(request_json: &str) -> Result<T, JsError> {
    serde_json::from_str(request_json).map_err(jerr)
}

/// A two-person request (synastry / composite).
#[derive(serde::Deserialize)]
struct PairJson {
    a: PersonJson,
    b: PersonJson,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default)]
    dut1: f64,
    #[serde(default = "default_cal")]
    cal: String,
}

/// A transit request.
#[derive(serde::Deserialize)]
struct TransitJson {
    natal: PersonJson,
    transit: String,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default)]
    dut1: f64,
    #[serde(default = "default_cal")]
    cal: String,
}

/// A progression request.
#[derive(serde::Deserialize)]
struct ProgressJson {
    natal: PersonJson,
    target: String,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default)]
    dut1: f64,
    #[serde(default = "default_cal")]
    cal: String,
}

/// One person in a pair/transit/progression request.
#[derive(serde::Deserialize)]
struct PersonJson {
    date: String,
    lat: f64,
    lon: f64,
}

fn default_system() -> String {
    "placidus".to_owned()
}
fn default_cal() -> String {
    "gregorian".to_owned()
}

impl PersonJson {
    fn to_input(&self, cal: CalendarKind) -> PersonInput {
        PersonInput {
            date: self.date.clone(),
            cal,
            lat_deg: self.lat,
            lon_deg: self.lon,
        }
    }
}

/// Computes a natal chart from DE bytes and a JSON request, returning the
/// stable JSON view as a string.
///
/// The request is a [`oxiephemeris_chart::NatalSpec`]:
/// `{"date","lat","lon","alt"?,"dut1"?,"cal"?,"system"?,"sidereal"?,"rulership"?}`.
///
/// # Errors
///
/// Returns a `JsError` for a malformed request, unparseable DE bytes, or a
/// computation failure.
#[wasm_bindgen]
pub fn natal_json(de: &[u8], request_json: &str) -> Result<String, JsError> {
    let request = spec::<NatalSpec>(request_json)?
        .into_request()
        .map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let chart = natal_chart(&de, &request).map_err(err)?;
    chart_to_json(&chart).map_err(err)
}

/// Computes a natal chart, returning RDF Turtle.
///
/// # Errors
///
/// As [`natal_json`], plus RDF-emission failures.
#[wasm_bindgen]
pub fn natal_turtle(
    de: &[u8],
    request_json: &str,
    base_iri: Option<String>,
) -> Result<String, JsError> {
    let request = spec::<NatalSpec>(request_json)?
        .into_request()
        .map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let chart = natal_chart(&de, &request).map_err(err)?;
    chart_to_rdf(&chart, base_iri.as_deref(), None, RdfFormat::Turtle).map_err(err)
}

/// Computes a synastry between two natal charts, returning JSON.
///
/// The request is `{"a":{date,lat,lon},"b":{...},"system"?,"dut1"?,"cal"?}`.
///
/// # Errors
///
/// As [`natal_json`].
#[wasm_bindgen]
pub fn synastry_json(de: &[u8], request_json: &str) -> Result<String, JsError> {
    let req = spec::<PairJson>(request_json)?;
    let cal = CalendarKind::parse(&req.cal).map_err(err)?;
    let sys = parse_house_system(&req.system).map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let cmp = oxiephemeris_chart::synastry(
        &de,
        &req.a.to_input(cal),
        &req.b.to_input(cal),
        sys,
        req.dut1,
    )
    .map_err(err)?;
    comparison_to_json(&cmp).map_err(err)
}

/// Synastry as RDF Turtle.
///
/// # Errors
///
/// As [`synastry_json`], plus RDF-emission failures.
#[wasm_bindgen]
pub fn synastry_turtle(
    de: &[u8],
    request_json: &str,
    base_iri: Option<String>,
) -> Result<String, JsError> {
    let req = spec::<PairJson>(request_json)?;
    let cal = CalendarKind::parse(&req.cal).map_err(err)?;
    let sys = parse_house_system(&req.system).map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let cmp = oxiephemeris_chart::synastry(
        &de,
        &req.a.to_input(cal),
        &req.b.to_input(cal),
        sys,
        req.dut1,
    )
    .map_err(err)?;
    comparison_to_rdf(&cmp, base_iri.as_deref(), RdfFormat::Turtle).map_err(err)
}

/// Computes a transit-to-natal comparison, returning JSON.
///
/// The request is `{"natal":{date,lat,lon},"transit":"<iso>","system"?,"dut1"?,"cal"?}`.
///
/// # Errors
///
/// As [`natal_json`].
#[wasm_bindgen]
pub fn transit_json(de: &[u8], request_json: &str) -> Result<String, JsError> {
    let req = spec::<TransitJson>(request_json)?;
    let cal = CalendarKind::parse(&req.cal).map_err(err)?;
    let sys = parse_house_system(&req.system).map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let cmp = oxiephemeris_chart::transit(
        &de,
        &req.natal.to_input(cal),
        &req.transit,
        cal,
        sys,
        req.dut1,
    )
    .map_err(err)?;
    comparison_to_json(&cmp).map_err(err)
}

/// Computes a secondary-progression comparison, returning JSON.
///
/// The request is `{"natal":{date,lat,lon},"target":"<iso>","system"?,"dut1"?,"cal"?}`.
///
/// # Errors
///
/// As [`natal_json`].
#[wasm_bindgen]
pub fn progress_json(de: &[u8], request_json: &str) -> Result<String, JsError> {
    let req = spec::<ProgressJson>(request_json)?;
    let cal = CalendarKind::parse(&req.cal).map_err(err)?;
    let sys = parse_house_system(&req.system).map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let cmp = oxiephemeris_chart::progression(
        &de,
        &req.natal.to_input(cal),
        &req.target,
        cal,
        sys,
        req.dut1,
    )
    .map_err(err)?;
    comparison_to_json(&cmp).map_err(err)
}

/// Computes a midpoint composite chart, returning JSON.
///
/// The request is the same shape as [`synastry_json`], plus an optional
/// `"base_iri"`.
///
/// # Errors
///
/// As [`natal_json`].
#[wasm_bindgen]
pub fn composite_json(de: &[u8], request_json: &str) -> Result<String, JsError> {
    let req = spec::<PairJson>(request_json)?;
    let cal = CalendarKind::parse(&req.cal).map_err(err)?;
    let sys = parse_house_system(&req.system).map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let comp = oxiephemeris_chart::composite(
        &de,
        &req.a.to_input(cal),
        &req.b.to_input(cal),
        sys,
        req.dut1,
        None,
    )
    .map_err(err)?;
    composite_to_json(&comp).map_err(err)
}

/// Composite chart as RDF Turtle.
///
/// # Errors
///
/// As [`composite_json`], plus RDF-emission failures.
#[wasm_bindgen]
pub fn composite_turtle(
    de: &[u8],
    request_json: &str,
    base_iri: Option<String>,
) -> Result<String, JsError> {
    let req = spec::<PairJson>(request_json)?;
    let cal = CalendarKind::parse(&req.cal).map_err(err)?;
    let sys = parse_house_system(&req.system).map_err(err)?;
    let de = DeFile::parse(de).map_err(jerr)?;
    let comp = oxiephemeris_chart::composite(
        &de,
        &req.a.to_input(cal),
        &req.b.to_input(cal),
        sys,
        req.dut1,
        base_iri.as_deref(),
    )
    .map_err(err)?;
    composite_to_rdf(&comp, base_iri.as_deref(), RdfFormat::Turtle).map_err(err)
}
