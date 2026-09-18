//! Shared machinery for the W3C JSON-LD test-suite harnesses.
//!
//! Suites are discovered through environment variables and the harness silently reports "skipped"
//! when they are not set, so the crate's tests remain runnable without any external checkout:
//!
//! * `OXIXML_TESTSUITE_DIR` — the root of the oxigraph testsuite checkout, which contains the
//!   `json-ld-api` and `json-ld-streaming` W3C suites.
//!
//! Nothing in here hardcodes an absolute path.

#![allow(dead_code)]

use oxixml_jsonld::{
    ClosureLoader, DocumentLoader, JsonLdErrorCode, JsonLdOptions, JsonLdProcessingMode,
    JsonLdProfile, JsonObject, JsonValue, LoadDocumentError, LoadDocumentOptions, RdfDirection,
    RemoteDocument,
};
use oxixml_model::{BlankNode, Quad, Term};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// The root of the oxigraph testsuite checkout, if configured.
#[must_use]
pub fn testsuite_dir() -> Option<PathBuf> {
    let raw = std::env::var_os("OXIXML_TESTSUITE_DIR")?;
    let path = PathBuf::from(raw);
    path.is_dir().then_some(path)
}

/// A parsed W3C manifest entry.
#[derive(Debug, Clone)]
pub struct TestCase {
    /// The fragment identifier, e.g. `#t0001`.
    pub id: String,
    /// The `rdf:type` values, e.g. `jld:PositiveEvaluationTest`.
    pub types: Vec<String>,
    /// The human-readable name.
    pub name: String,
    /// The `input` value, relative to the manifest.
    pub input: String,
    /// The `expect` value (positive evaluation tests), relative to the manifest.
    pub expect: Option<String>,
    /// The `expectErrorCode` value (negative evaluation tests).
    pub expect_error_code: Option<String>,
    /// The `option` map.
    pub options: JsonObject,
    /// The `context` value used by compaction tests.
    pub context: Option<String>,
    /// The requested profile IRIs (`option.specVersion` aside), from the entry's `option`.
    pub base_iri: String,
}

impl TestCase {
    /// Whether the entry declares `type`.
    #[must_use]
    pub fn has_type(&self, ty: &str) -> bool {
        self.types.iter().any(|t| t == ty)
    }

    /// Whether the entry is a negative evaluation test.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.has_type("jld:NegativeEvaluationTest")
    }

    /// Whether the entry is a positive *syntax* test (parse must succeed; output is not compared).
    #[must_use]
    pub fn is_positive_syntax(&self) -> bool {
        self.has_type("jld:PositiveSyntaxTest")
    }

    fn option_str(&self, key: &str) -> Option<&str> {
        self.options.get(key).and_then(JsonValue::as_str)
    }

    fn option_bool(&self, key: &str) -> Option<bool> {
        match self.options.get(key) {
            Some(JsonValue::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    /// The `specVersion` the entry targets (`json-ld-1.1` when unspecified).
    #[must_use]
    pub fn spec_version(&self) -> &str {
        self.option_str("specVersion").unwrap_or("json-ld-1.1")
    }

    /// Whether the entry is normative (defaults to `true`).
    #[must_use]
    pub fn normative(&self) -> bool {
        self.option_bool("normative").unwrap_or(true)
    }

    /// The effective base IRI of the test document.
    #[must_use]
    pub fn base(&self) -> String {
        match self.option_str("base") {
            Some(base) => base.to_owned(),
            None => format!("{}{}", self.base_iri, self.input),
        }
    }
}

/// A parsed manifest.
pub struct Manifest {
    /// The directory the manifest lives in.
    pub dir: PathBuf,
    /// The manifest's `baseIri`.
    pub base_iri: String,
    /// The test entries.
    pub sequence: Vec<TestCase>,
}

/// Loads a JSON document from disk into the crate's [`JsonValue`] tree.
pub fn load_json(path: &Path) -> Result<JsonValue, String> {
    let data = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    JsonValue::from_slice(&data).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parses a W3C JSON-LD manifest.
pub fn load_manifest(path: &Path) -> Result<Manifest, String> {
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

/// Builds a [`DocumentLoader`] serving `base_iri`-rooted URLs from `dir`.
///
/// The W3C suites reference their own contexts by absolute `https://w3c.github.io/...` URLs; the
/// files are present in the checkout, so the harness resolves them locally. No network access is
/// ever performed.
pub fn file_loader(base_iri: &str, dir: &Path) -> impl DocumentLoader + use<> {
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
        let data = fs::read(&path).map_err(|e| {
            LoadDocumentError::new(
                format!("cannot read {}: {e}", path.display()),
                JsonLdErrorCode::LoadingRemoteContextFailed,
            )
        })?;
        Ok(RemoteDocument::new(data, url))
    })
}

/// Builds the [`JsonLdOptions`] a manifest entry asks for.
pub fn build_options<'a>(
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
        let path = manifest.dir.join(expand_context);
        options.expand_context = Some(load_json(&path)?);
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
#[must_use]
pub fn json_eq(left: &JsonValue, right: &JsonValue) -> bool {
    match (left, right) {
        (JsonValue::Null, JsonValue::Null) => true,
        (JsonValue::Bool(a), JsonValue::Bool(b)) => a == b,
        (JsonValue::Number(a), JsonValue::Number(b)) => number_eq(a, b),
        (JsonValue::String(a), JsonValue::String(b)) => a == b,
        (JsonValue::Array(a), JsonValue::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_eq(x, y))
        }
        (JsonValue::Object(a), JsonValue::Object(b)) => {
            if a.len() != b.len() {
                return false;
            }
            a.iter()
                .all(|(key, value)| b.get(key).is_some_and(|other| json_eq(value, other)))
        }
        _ => false,
    }
}

fn number_eq(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    match (left.parse::<f64>(), right.parse::<f64>()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// A pretty one-line rendering used in failure reports.
#[must_use]
pub fn render(value: &JsonValue) -> String {
    let text = value.to_json_string();
    if text.len() > 600 {
        format!("{}…", &text[..600])
    } else {
        text
    }
}

/// Whether two RDF datasets are isomorphic (equal up to blank node renaming).
#[must_use]
pub fn datasets_isomorphic(left: &[Quad], right: &[Quad]) -> bool {
    // An RDF dataset is a set: the suite's expectation files may repeat a statement.
    let left: Vec<Quad> = deduplicate(left);
    let right: Vec<Quad> = deduplicate(right);
    let (left, right) = (left.as_slice(), right.as_slice());
    if left.len() != right.len() {
        return false;
    }
    let left_set: BTreeSet<String> = left.iter().map(quad_key).collect();
    let right_set: BTreeSet<String> = right.iter().map(quad_key).collect();
    if left_set.len() != right_set.len() {
        return false;
    }
    if left_set == right_set {
        return true;
    }
    let left_blanks = blank_nodes(left);
    let right_blanks = blank_nodes(right);
    if left_blanks.len() != right_blanks.len() {
        return false;
    }
    // Signature-based bucketing keeps the search space tiny for the suite's small datasets.
    let left_sig = signatures(left, &left_blanks);
    let right_sig = signatures(right, &right_blanks);
    let mut order: Vec<String> = left_blanks.iter().cloned().collect();
    order.sort_by_key(|b| left_sig.get(b).map_or(0, |s| usize::MAX - s.len()));
    let mut mapping = HashMap::new();
    let mut used = HashSet::new();
    search(
        &order,
        0,
        &mut mapping,
        &mut used,
        &left_sig,
        &right_sig,
        &right_blanks,
        &right_set,
        left,
    )
}

#[allow(clippy::too_many_arguments)]
fn search(
    order: &[String],
    index: usize,
    mapping: &mut HashMap<String, String>,
    used: &mut HashSet<String>,
    left_sig: &BTreeMap<String, BTreeSet<String>>,
    right_sig: &BTreeMap<String, BTreeSet<String>>,
    right_blanks: &BTreeSet<String>,
    right_set: &BTreeSet<String>,
    left: &[Quad],
) -> bool {
    if index == order.len() {
        let renamed: BTreeSet<String> = left.iter().map(|q| rename_key(q, mapping)).collect();
        return &renamed == right_set;
    }
    let candidate_source = &order[index];
    let source_signature = left_sig.get(candidate_source);
    for candidate in right_blanks {
        if used.contains(candidate) {
            continue;
        }
        if source_signature.map(BTreeSet::len) != right_sig.get(candidate).map(BTreeSet::len) {
            continue;
        }
        mapping.insert(candidate_source.clone(), candidate.clone());
        used.insert(candidate.clone());
        if search(
            order,
            index + 1,
            mapping,
            used,
            left_sig,
            right_sig,
            right_blanks,
            right_set,
            left,
        ) {
            return true;
        }
        used.remove(candidate);
        mapping.remove(candidate_source);
    }
    false
}

fn deduplicate(quads: &[Quad]) -> Vec<Quad> {
    let mut seen = HashSet::new();
    quads
        .iter()
        .filter(|quad| seen.insert(quad_key(quad)))
        .cloned()
        .collect()
}

fn blank_nodes(quads: &[Quad]) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for quad in quads {
        if let oxixml_model::NamedOrBlankNode::BlankNode(b) = &quad.subject {
            set.insert(b.as_str().to_owned());
        }
        if let Term::BlankNode(b) = &quad.object {
            set.insert(b.as_str().to_owned());
        }
        if let oxixml_model::GraphName::BlankNode(b) = &quad.graph_name {
            set.insert(b.as_str().to_owned());
        }
    }
    set
}

fn signatures(quads: &[Quad], blanks: &BTreeSet<String>) -> BTreeMap<String, BTreeSet<String>> {
    let mut map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for blank in blanks {
        map.insert(blank.clone(), BTreeSet::new());
    }
    for quad in quads {
        let key = anonymised_key(quad);
        for blank in quad_blanks(quad) {
            if let Some(entry) = map.get_mut(&blank) {
                entry.insert(key.clone());
            }
        }
    }
    map
}

fn quad_blanks(quad: &Quad) -> Vec<String> {
    let mut out = Vec::new();
    if let oxixml_model::NamedOrBlankNode::BlankNode(b) = &quad.subject {
        out.push(b.as_str().to_owned());
    }
    if let Term::BlankNode(b) = &quad.object {
        out.push(b.as_str().to_owned());
    }
    if let oxixml_model::GraphName::BlankNode(b) = &quad.graph_name {
        out.push(b.as_str().to_owned());
    }
    out
}

fn quad_key(quad: &Quad) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "{} {} {} {}",
        quad.subject, quad.predicate, quad.object, quad.graph_name
    );
    out
}

fn anonymised_key(quad: &Quad) -> String {
    let mut anonymous = quad.clone();
    if let oxixml_model::NamedOrBlankNode::BlankNode(_) = &anonymous.subject {
        anonymous.subject = BlankNode::new_unchecked("x").into();
    }
    if let Term::BlankNode(_) = &anonymous.object {
        anonymous.object = BlankNode::new_unchecked("x").into();
    }
    if let oxixml_model::GraphName::BlankNode(_) = &anonymous.graph_name {
        anonymous.graph_name = BlankNode::new_unchecked("x").into();
    }
    quad_key(&anonymous)
}

fn rename_key(quad: &Quad, mapping: &HashMap<String, String>) -> String {
    let mut renamed = quad.clone();
    if let oxixml_model::NamedOrBlankNode::BlankNode(b) = &renamed.subject {
        if let Some(target) = mapping.get(b.as_str()) {
            renamed.subject = BlankNode::new_unchecked(target.clone()).into();
        }
    }
    if let Term::BlankNode(b) = &renamed.object {
        if let Some(target) = mapping.get(b.as_str()) {
            renamed.object = BlankNode::new_unchecked(target.clone()).into();
        }
    }
    if let oxixml_model::GraphName::BlankNode(b) = &renamed.graph_name {
        if let Some(target) = mapping.get(b.as_str()) {
            renamed.graph_name = BlankNode::new_unchecked(target.clone()).into();
        }
    }
    quad_key(&renamed)
}

/// The outcome tally of one manifest run.
#[derive(Default)]
pub struct Report {
    /// Number of entries that produced the expected result.
    pub passed: usize,
    /// Number of entries that did not.
    pub failed: usize,
    /// Number of entries the harness deliberately did not run.
    pub skipped: usize,
    /// Human-readable failure descriptions, capped for readability.
    pub failures: Vec<String>,
}

impl Report {
    /// Records a success.
    pub fn pass(&mut self) {
        self.passed += 1;
    }

    /// Records a skip.
    pub fn skip(&mut self) {
        self.skipped += 1;
    }

    /// Records a failure.
    ///
    /// Only the first few descriptions are kept so a regression report stays readable; set
    /// `OXIXML_JSONLD_VERBOSE=1` to keep every one of them.
    pub fn fail(&mut self, description: impl Into<String>) {
        self.failed += 1;
        let cap = if std::env::var_os("OXIXML_JSONLD_VERBOSE").is_some() {
            usize::MAX
        } else {
            20
        };
        if self.failures.len() < cap {
            self.failures.push(description.into());
        }
    }

    /// The ratio of passed entries over the entries that were actually run.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        let run = self.passed + self.failed;
        if run == 0 {
            1.0
        } else {
            self.passed as f64 / run as f64
        }
    }

    /// A one-line summary.
    #[must_use]
    pub fn summary(&self, label: &str) -> String {
        format!(
            "{label}: {} passed, {} failed, {} skipped ({:.1}%)",
            self.passed,
            self.failed,
            self.skipped,
            self.ratio() * 100.0
        )
    }
}
