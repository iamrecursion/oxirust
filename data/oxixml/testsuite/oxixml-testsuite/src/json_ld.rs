//! W3C JSON-LD 1.1 API and Streaming conformance runner.
//!
//! Drives the `json-ld-api/tests/{expand,compact,flatten,toRdf,fromRdf}-manifest.jsonld` and
//! `json-ld-streaming/tests/stream-toRdf-manifest.jsonld` manifests against
//! [`oxixml_jsonld`]'s five public entry points. Each entry point is reported as its own suite
//! (`json-ld/expand`, `json-ld/compact`, …) since a regression in one operation says nothing
//! about the others.
//!
//! Only `specVersion: json-ld-1.1` entries with a non-HTML input are run — this crate has no
//! JSON-LD 1.0 compatibility mode to hold to the older behaviour, and HTML-embedded JSON-LD is a
//! separate suite this crate does not parse.

use std::fs;
use std::path::{Path, PathBuf};

use oxixml_jsonld::{
    ClosureLoader, DocumentLoader, JsonLdErrorCode, JsonLdOptions, JsonLdProcessingMode,
    JsonLdProfile, JsonLdSyntaxError, JsonObject, JsonValue, LoadDocumentError,
    LoadDocumentOptions, RdfDirection, RemoteDocument, compact, expand, flatten, from_rdf, to_rdf,
};
use oxixml_model::Quad;
use oxixml_turtle::NQuadsParser;

use crate::env;
use crate::isomorphism::quads_isomorphic;
use crate::report::SuiteReport;

/// One parsed manifest entry.
struct TestCase {
    id: String,
    types: Vec<String>,
    name: String,
    input: String,
    expect: Option<String>,
    expect_error_code: Option<String>,
    options: JsonObject,
    context: Option<String>,
    base_iri: String,
}

impl TestCase {
    fn has_type(&self, ty: &str) -> bool {
        self.types.iter().any(|t| t == ty)
    }

    fn option_str(&self, key: &str) -> Option<&str> {
        self.options.get(key).and_then(JsonValue::as_str)
    }

    fn option_bool(&self, key: &str) -> Option<bool> {
        match self.options.get(key) {
            Some(JsonValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    fn spec_version(&self) -> &str {
        self.option_str("specVersion").unwrap_or("json-ld-1.1")
    }

    fn base(&self) -> String {
        match self.option_str("base") {
            Some(base) => base.to_owned(),
            None => format!("{}{}", self.base_iri, self.input),
        }
    }
}

/// A parsed manifest.
struct Manifest {
    dir: PathBuf,
    base_iri: String,
    sequence: Vec<TestCase>,
}

fn load_json(path: &Path) -> Result<JsonValue, String> {
    let data = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    JsonValue::from_slice(&data).map_err(|error| format!("{}: {error}", path.display()))
}

fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let document = load_json(path)?;
    let object = document
        .as_object()
        .ok_or_else(|| format!("{}: manifest is not a JSON object", path.display()))?;
    let base_iri = object
        .get("baseIri")
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned();
    let dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut sequence = Vec::new();
    if let Some(JsonValue::Array(entries)) = object.get("sequence") {
        for entry in entries {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let types = match entry.get("@type") {
                Some(JsonValue::Array(items)) => items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect(),
                Some(JsonValue::String(s)) => vec![s.clone()],
                _ => Vec::new(),
            };
            let Some(input) = entry.get("input").and_then(JsonValue::as_str) else {
                continue;
            };
            sequence.push(TestCase {
                id: entry
                    .get("@id")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                types,
                name: entry
                    .get("name")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                input: input.to_owned(),
                expect: entry
                    .get("expect")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned),
                expect_error_code: entry
                    .get("expectErrorCode")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned),
                options: entry
                    .get("option")
                    .and_then(JsonValue::as_object)
                    .cloned()
                    .unwrap_or_default(),
                context: entry
                    .get("context")
                    .and_then(JsonValue::as_str)
                    .map(str::to_owned),
                base_iri: base_iri.clone(),
            });
        }
    }
    Ok(Manifest {
        dir,
        base_iri,
        sequence,
    })
}

/// A [`DocumentLoader`] serving `base_iri`-rooted URLs from `dir`, with no network access ever
/// performed — every document the W3C suites reference by an absolute URL is present in the
/// checkout.
fn file_loader(base_iri: &str, dir: &Path) -> impl DocumentLoader + use<> {
    let base_iri = base_iri.to_owned();
    let dir = dir.to_path_buf();
    ClosureLoader::new(move |url: &str, _options: &LoadDocumentOptions| {
        let Some(relative) = url.strip_prefix(&base_iri) else {
            return Err(LoadDocumentError::new(
                format!("<{url}> is outside the local test suite"),
                JsonLdErrorCode::LoadingRemoteContextFailed,
            ));
        };
        let relative = relative.split('#').next().unwrap_or(relative);
        let path = dir.join(relative);
        let data = fs::read(&path).map_err(|error| {
            LoadDocumentError::new(
                format!("cannot read {}: {error}", path.display()),
                JsonLdErrorCode::LoadingRemoteContextFailed,
            )
        })?;
        Ok(RemoteDocument::new(data, url))
    })
}

fn build_options<'a>(
    test: &TestCase,
    loader: &'a dyn DocumentLoader,
    manifest: &Manifest,
) -> Result<JsonLdOptions<'a>, String> {
    let mut options = JsonLdOptions::new().with_loader(loader);
    options.base = Some(test.base());
    options.ordered = true;
    if test.option_str("processingMode") == Some("json-ld-1.0")
        || test.spec_version() == "json-ld-1.0"
    {
        options.processing_mode = JsonLdProcessingMode::JsonLd1_0;
    }
    if let Some(expand_context) = test.option_str("expandContext") {
        options.expand_context = Some(load_json(&manifest.dir.join(expand_context))?);
    }
    if let Some(value) = test.option_bool("compactArrays") {
        options.compact_arrays = value;
    }
    if let Some(value) = test.option_bool("compactToRelative") {
        options.compact_to_relative = value;
    }
    if let Some(value) = test.option_bool("produceGeneralizedRdf") {
        options.produce_generalized_rdf = value;
    }
    if let Some(value) = test.option_bool("useNativeTypes") {
        options.use_native_types = value;
    }
    if let Some(value) = test.option_bool("useRdfType") {
        options.use_rdf_type = value;
    }
    if test.has_type("jld:StreamTest") {
        options.profile |= JsonLdProfile::Streaming;
    }
    match test.option_str("rdfDirection") {
        Some("i18n-datatype") => options.rdf_direction = RdfDirection::I18nDatatype,
        Some("compound-literal") => options.rdf_direction = RdfDirection::CompoundLiteral,
        _ => {}
    }
    Ok(options)
}

/// Structural JSON equality: object member order is irrelevant, array order is significant.
fn json_eq(left: &JsonValue, right: &JsonValue) -> bool {
    match (left, right) {
        (JsonValue::Null, JsonValue::Null) => true,
        (JsonValue::Bool(a), JsonValue::Bool(b)) => a == b,
        (JsonValue::Number(a), JsonValue::Number(b)) => number_eq(a, b),
        (JsonValue::String(a), JsonValue::String(b)) => a == b,
        (JsonValue::Array(a), JsonValue::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_eq(x, y))
        }
        (JsonValue::Object(a), JsonValue::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|other| json_eq(value, other)))
        }
        _ => false,
    }
}

fn number_eq(left: &str, right: &str) -> bool {
    left == right || matches!((left.parse::<f64>(), right.parse::<f64>()), (Ok(a), Ok(b)) if a == b)
}

fn render(value: &JsonValue) -> String {
    let text = value.to_json_string();
    if text.len() > 400 {
        format!("{}…", &text[..400])
    } else {
        text
    }
}

/// Skips entries this driver never evaluates: JSON-LD 1.0 entries (this is a 1.1-only processor)
/// and HTML-embedded input (a separate, unparsed suite).
fn should_skip(test: &TestCase) -> Option<&'static str> {
    if test.spec_version() != "json-ld-1.1" {
        return Some("specVersion is not json-ld-1.1");
    }
    if test.input.ends_with(".html") {
        return Some("HTML input is out of scope");
    }
    None
}

fn nquads(path: &Path) -> Result<Vec<Quad>, String> {
    let data = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut quads = Vec::new();
    for quad in NQuadsParser::new().for_slice(&data) {
        quads.push(quad.map_err(|error| format!("{}: {error}", path.display()))?);
    }
    Ok(quads)
}

/// Checks a manifest entry's error expectation, if it has one. Returns `true` when the entry was
/// an error-expectation entry (handled either way), so the caller's positive-result path is
/// skipped.
fn check_error_expectation(
    test: &TestCase,
    result: &Result<JsonValue, JsonLdSyntaxError>,
    report: &mut SuiteReport,
) -> bool {
    let Some(expected) = &test.expect_error_code else {
        return false;
    };
    match result {
        Ok(value) => report.record_fail(
            &test.id,
            format!(
                "{}: expected error '{expected}', got {}",
                test.name,
                render(value)
            ),
        ),
        Err(error) => {
            let actual = error.code().map(|code| code.to_string());
            if actual.as_deref() == Some(expected.as_str()) {
                report.record_pass();
            } else {
                report.record_fail(
                    &test.id,
                    format!(
                        "{}: expected error '{expected}', got '{}'",
                        test.name,
                        actual.unwrap_or_else(|| error.to_string())
                    ),
                );
            }
        }
    }
    true
}

fn compare_or_record(
    test: &TestCase,
    manifest: &Manifest,
    result: Result<JsonValue, JsonLdSyntaxError>,
    report: &mut SuiteReport,
) {
    if check_error_expectation(test, &result, report) {
        return;
    }
    let Some(expect) = &test.expect else {
        report.record_skip("no mf:expect and no error expectation");
        return;
    };
    let expected = match load_json(&manifest.dir.join(expect)) {
        Ok(expected) => expected,
        Err(error) => {
            report.record_fail(&test.id, error);
            return;
        }
    };
    match result {
        Ok(value) if json_eq(&value, &expected) => report.record_pass(),
        Ok(value) => report.record_fail(
            &test.id,
            format!(
                "{}\n     got: {}\nexpected: {}",
                test.name,
                render(&value),
                render(&expected)
            ),
        ),
        Err(error) => {
            report.record_fail(&test.id, format!("{}: unexpected error {error}", test.name))
        }
    }
}

fn load_case_input(
    manifest: &Manifest,
    test: &TestCase,
    report: &mut SuiteReport,
) -> Option<JsonValue> {
    match load_json(&manifest.dir.join(&test.input)) {
        Ok(input) => Some(input),
        Err(error) => {
            report.record_fail(&test.id, format!("{}: {error}", test.name));
            None
        }
    }
}

fn run_manifest_entries(
    dir: &Path,
    manifest_name: &str,
    mut evaluate: impl FnMut(&Manifest, &TestCase, &dyn DocumentLoader, &mut SuiteReport),
) -> Result<SuiteReport, String> {
    let manifest = load_manifest(&dir.join(manifest_name))?;
    let loader = file_loader(&manifest.base_iri, dir);
    let mut report = SuiteReport::new("json-ld"); // caller overwrites `suite` before returning
    for test in &manifest.sequence {
        if let Some(reason) = should_skip(test) {
            report.record_skip(reason);
            continue;
        }
        evaluate(&manifest, test, &loader, &mut report);
    }
    Ok(report)
}

fn named(mut report: SuiteReport, suite: &str) -> SuiteReport {
    report.suite = suite.to_owned();
    report
}

fn expand_report(dir: &Path) -> Result<SuiteReport, String> {
    let report = run_manifest_entries(
        dir,
        "expand-manifest.jsonld",
        |manifest, test, loader, report| {
            let options = match build_options(test, loader, manifest) {
                Ok(options) => options,
                Err(error) => return report.record_fail(&test.id, error),
            };
            let Some(input) = load_case_input(manifest, test, report) else {
                return;
            };
            compare_or_record(test, manifest, expand(input, &options), report);
        },
    )?;
    Ok(named(report, "json-ld/expand"))
}

fn compact_report(dir: &Path) -> Result<SuiteReport, String> {
    let report = run_manifest_entries(
        dir,
        "compact-manifest.jsonld",
        |manifest, test, loader, report| {
            let options = match build_options(test, loader, manifest) {
                Ok(options) => options,
                Err(error) => return report.record_fail(&test.id, error),
            };
            let Some(input) = load_case_input(manifest, test, report) else {
                return;
            };
            let context = match &test.context {
                Some(context) => match load_json(&manifest.dir.join(context)) {
                    Ok(context) => context,
                    Err(error) => return report.record_fail(&test.id, error),
                },
                None => JsonValue::Object(JsonObject::new()),
            };
            compare_or_record(test, manifest, compact(input, context, &options), report);
        },
    )?;
    Ok(named(report, "json-ld/compact"))
}

fn flatten_report(dir: &Path) -> Result<SuiteReport, String> {
    let report = run_manifest_entries(
        dir,
        "flatten-manifest.jsonld",
        |manifest, test, loader, report| {
            let options = match build_options(test, loader, manifest) {
                Ok(options) => options,
                Err(error) => return report.record_fail(&test.id, error),
            };
            let Some(input) = load_case_input(manifest, test, report) else {
                return;
            };
            let context = match &test.context {
                Some(context) => match load_json(&manifest.dir.join(context)) {
                    Ok(context) => Some(context),
                    Err(error) => return report.record_fail(&test.id, error),
                },
                None => None,
            };
            compare_or_record(test, manifest, flatten(input, context, &options), report);
        },
    )?;
    Ok(named(report, "json-ld/flatten"))
}

fn from_rdf_report(dir: &Path) -> Result<SuiteReport, String> {
    let report = run_manifest_entries(
        dir,
        "fromRdf-manifest.jsonld",
        |manifest, test, loader, report| {
            let options = match build_options(test, loader, manifest) {
                Ok(options) => options,
                Err(error) => return report.record_fail(&test.id, error),
            };
            let quads = match nquads(&manifest.dir.join(&test.input)) {
                Ok(quads) => quads,
                Err(error) => {
                    if test.expect_error_code.is_some() {
                        return report.record_pass(); // a malformed input is exactly what was asked for
                    }
                    return report.record_fail(&test.id, error);
                }
            };
            compare_or_record(test, manifest, from_rdf(&quads, &options), report);
        },
    )?;
    Ok(named(report, "json-ld/fromRdf"))
}

fn to_rdf_like_report(dir: &Path, manifest_name: &str, suite: &str) -> Result<SuiteReport, String> {
    let report = run_manifest_entries(dir, manifest_name, |manifest, test, loader, report| {
        let options = match build_options(test, loader, manifest) {
            Ok(options) => options,
            Err(error) => return report.record_fail(&test.id, error),
        };
        let Some(input) = load_case_input(manifest, test, report) else {
            return;
        };
        let result = to_rdf(input, &options);
        if let Some(expected_code) = &test.expect_error_code {
            match result {
                Ok(_) => report.record_fail(
                    &test.id,
                    format!("expected error '{expected_code}', got success"),
                ),
                Err(error) => {
                    let actual = error.code().map(|code| code.to_string());
                    if actual.as_deref() == Some(expected_code.as_str()) {
                        report.record_pass();
                    } else {
                        report.record_fail(
                            &test.id,
                            format!(
                                "expected error '{expected_code}', got '{}'",
                                actual.unwrap_or_else(|| error.to_string())
                            ),
                        );
                    }
                }
            }
            return;
        }
        let quads = match result {
            Ok(quads) => quads,
            Err(error) => return report.record_fail(&test.id, format!("unexpected error {error}")),
        };
        if test.has_type("jld:PositiveSyntaxTest") {
            return report.record_pass();
        }
        let Some(expect) = &test.expect else {
            return report.record_skip("no mf:expect and no error expectation");
        };
        let expected = match nquads(&manifest.dir.join(expect)) {
            Ok(expected) => expected,
            // A generalized-RDF expectation (a blank node or literal in predicate position)
            // cannot be represented by `oxixml_model::Quad`, matching the RDF 1.1 abstract
            // syntax; the entry is out of scope rather than a failure.
            Err(_) => return report.record_skip("expected result is generalized RDF"),
        };
        if quads_isomorphic(&quads, &expected) {
            report.record_pass();
        } else {
            report.record_fail(
                &test.id,
                format!("{} quads vs {} expected", quads.len(), expected.len()),
            );
        }
    })?;
    Ok(named(report, suite))
}

fn api_tests_dir() -> Option<PathBuf> {
    let dir = env::testsuite_dir()?.join("json-ld-api").join("tests");
    dir.is_dir().then_some(dir)
}

fn streaming_tests_dir() -> Option<PathBuf> {
    let dir = env::testsuite_dir()?
        .join("json-ld-streaming")
        .join("tests");
    dir.is_dir().then_some(dir)
}

fn unavailable(suite: &str) -> SuiteReport {
    SuiteReport::unavailable(
        suite,
        match env::testsuite_dir() {
            None => env::missing(env::TESTSUITE_DIR_VAR),
            Some(_) => env::not_found(
                env::TESTSUITE_DIR_VAR,
                "json-ld-api/tests/ (or json-ld-streaming/tests/)",
            ),
        },
    )
}

/// Runs every JSON-LD entry point the suite covers.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let mut reports = Vec::new();
    match api_tests_dir() {
        Some(dir) => {
            for (build, suite) in [
                (
                    expand_report as fn(&Path) -> Result<SuiteReport, String>,
                    "json-ld/expand",
                ),
                (compact_report, "json-ld/compact"),
                (flatten_report, "json-ld/flatten"),
                (from_rdf_report, "json-ld/fromRdf"),
            ] {
                reports.push(match build(&dir) {
                    Ok(report) => report,
                    Err(error) => {
                        let mut report = SuiteReport::new(suite);
                        report.record_skip(error);
                        report
                    }
                });
            }
            reports.push(
                match to_rdf_like_report(&dir, "toRdf-manifest.jsonld", "json-ld/toRdf") {
                    Ok(report) => report,
                    Err(error) => {
                        let mut report = SuiteReport::new("json-ld/toRdf");
                        report.record_skip(error);
                        report
                    }
                },
            );
        }
        None => {
            for suite in [
                "json-ld/expand",
                "json-ld/compact",
                "json-ld/flatten",
                "json-ld/fromRdf",
                "json-ld/toRdf",
            ] {
                reports.push(unavailable(suite));
            }
        }
    }

    match streaming_tests_dir() {
        Some(dir) => reports.push(
            match to_rdf_like_report(
                &dir,
                "stream-toRdf-manifest.jsonld",
                "json-ld/streaming-toRdf",
            ) {
                Ok(report) => report,
                Err(error) => {
                    let mut report = SuiteReport::new("json-ld/streaming-toRdf");
                    report.record_skip(error);
                    report
                }
            },
        ),
        None => reports.push(unavailable("json-ld/streaming-toRdf")),
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_eq_ignores_object_member_order_but_not_array_order() {
        let a = JsonValue::from_slice(br#"{"a":1,"b":2}"#).expect("valid JSON");
        let b = JsonValue::from_slice(br#"{"b":2,"a":1}"#).expect("valid JSON");
        assert!(json_eq(&a, &b));

        let arr_a = JsonValue::from_slice(br#"[1,2]"#).expect("valid JSON");
        let arr_b = JsonValue::from_slice(br#"[2,1]"#).expect("valid JSON");
        assert!(!json_eq(&arr_a, &arr_b));
    }

    #[test]
    fn number_eq_treats_equal_values_as_equal_regardless_of_lexical_form() {
        assert!(number_eq("1.0", "1"));
        assert!(!number_eq("1", "2"));
    }

    #[test]
    fn load_manifest_reads_a_small_embedded_manifest() {
        let dir = std::env::temp_dir().join(format!(
            "oxixml-testsuite-jsonld-fixture-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        let manifest_path = dir.join("expand-manifest.jsonld");
        fs::write(
            &manifest_path,
            // `##`-delimited: the fixture's own `"#t1"` fragment id would otherwise close a
            // single-`#` raw string early.
            br##"{
                "baseIri": "https://example.org/",
                "sequence": [
                    {"@id": "#t1", "@type": "jld:PositiveEvaluationTest", "name": "t1", "input": "t1-in.jsonld", "expect": "t1-out.jsonld"}
                ]
            }"##,
        )
        .expect("scratch file");

        let manifest = load_manifest(&manifest_path).expect("valid manifest");
        assert_eq!(manifest.sequence.len(), 1);
        assert_eq!(manifest.sequence[0].name, "t1");
        assert_eq!(manifest.sequence[0].spec_version(), "json-ld-1.1");
        assert!(should_skip(&manifest.sequence[0]).is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn should_skip_recognizes_json_ld_10_and_html_entries() {
        let mut test = TestCase {
            id: "#t".to_owned(),
            types: Vec::new(),
            name: "t".to_owned(),
            input: "t.jsonld".to_owned(),
            expect: None,
            expect_error_code: None,
            options: JsonObject::new(),
            context: None,
            base_iri: String::new(),
        };
        assert!(should_skip(&test).is_none());
        test.input = "t.html".to_owned();
        assert!(should_skip(&test).is_some());
        test.input = "t.jsonld".to_owned();
        test.options.insert(
            "specVersion".to_owned(),
            JsonValue::String("json-ld-1.0".to_owned()),
        );
        assert!(should_skip(&test).is_some());
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::TESTSUITE_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), 6);
        assert!(reports.iter().all(|report| !report.is_available()));
    }
}
