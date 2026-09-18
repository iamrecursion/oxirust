//! RDF Bulk Import/Export for tensorlogic-oxirs-bridge
//!
//! Provides efficient bulk parsing and serialization of RDF triples in
//! N-Triples, TSV, and simplified Turtle formats without external parsing deps.

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during bulk RDF I/O operations.
#[derive(Debug, Clone)]
pub enum BulkIoError {
    /// A line could not be parsed.
    ParseError { line: usize, message: String },
    /// An IRI value is syntactically invalid.
    InvalidIri(String),
    /// A triple is structurally invalid.
    InvalidTriple(String),
    /// An I/O-level error (string form for Clone compat).
    IoError(String),
    /// A prefix was referenced but never declared.
    UnknownPrefix(String),
    /// The input contained no triples or was entirely empty.
    EmptyInput,
    /// A line had unexpected content.
    MalformedLine { line: usize, content: String },
}

impl std::fmt::Display for BulkIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError { line, message } => {
                write!(f, "parse error at line {line}: {message}")
            }
            Self::InvalidIri(iri) => write!(f, "invalid IRI: {iri}"),
            Self::InvalidTriple(msg) => write!(f, "invalid triple: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::UnknownPrefix(prefix) => write!(f, "unknown prefix: {prefix}"),
            Self::EmptyInput => write!(f, "input is empty"),
            Self::MalformedLine { line, content } => {
                write!(f, "malformed line {line}: {content}")
            }
        }
    }
}

impl std::error::Error for BulkIoError {}

// ─────────────────────────────────────────────────────────────────────────────
// RdfTriple
// ─────────────────────────────────────────────────────────────────────────────

/// A raw RDF triple where subject, predicate and object are stored as strings.
///
/// IRIs are stored with angle-bracket delimiters (`<…>`), literals with
/// double-quote delimiters.  Prefixed names (e.g. `rdf:type`) are supported
/// as input but are always expanded before storage.
#[derive(Debug, Clone, PartialEq)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl RdfTriple {
    /// Construct a new triple from any string-like arguments.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            subject: s.into(),
            predicate: p.into(),
            object: o.into(),
        }
    }

    /// Returns `true` when the object is an RDF literal (starts with `"`).
    pub fn is_literal_object(&self) -> bool {
        self.object.starts_with('"')
    }

    /// Returns `true` when the object looks like an IRI: starts with `<` or
    /// contains a colon (prefixed name or bare IRI string).
    pub fn is_iri_object(&self) -> bool {
        self.object.starts_with('<') || self.object.contains(':')
    }

    /// Serialise the triple as a single N-Triples line: `<s> <p> <o> .`
    ///
    /// If terms are already angle-bracket-enclosed they are emitted as-is;
    /// bare strings are wrapped in `<…>`.
    pub fn to_ntriples(&self) -> String {
        let s = wrap_iri_if_needed(&self.subject);
        let p = wrap_iri_if_needed(&self.predicate);
        let o = wrap_iri_if_needed(&self.object);
        format!("{s} {p} {o} .")
    }

    /// Serialise the triple as a TSV line: `subject\tpredicate\tobject`
    /// (without angle brackets).
    pub fn to_tsv(&self) -> String {
        let s = strip_angle_brackets(&self.subject);
        let p = strip_angle_brackets(&self.predicate);
        let o = strip_angle_brackets(&self.object);
        format!("{s}\t{p}\t{o}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NamespaceRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Maps short prefixes to base IRIs and provides expand/contract operations.
#[derive(Debug, Clone, Default)]
pub struct NamespaceRegistry {
    prefixes: HashMap<String, String>,
}

impl NamespaceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a prefix→base-IRI mapping.
    pub fn register(&mut self, prefix: impl Into<String>, iri: impl Into<String>) {
        self.prefixes.insert(prefix.into(), iri.into());
    }

    /// Expand a prefixed name such as `rdf:type` to `<base + "type">`.
    ///
    /// Returns `None` when the prefix is not registered or the value does not
    /// contain a colon separator.
    pub fn expand(&self, prefixed: &str) -> Option<String> {
        // Already an absolute IRI – return as-is wrapped in angle brackets.
        if prefixed.starts_with('<') && prefixed.ends_with('>') {
            return Some(prefixed.to_string());
        }
        let colon_pos = prefixed.find(':')?;
        let prefix = &prefixed[..colon_pos];
        let local = &prefixed[colon_pos + 1..];
        let base = self.prefixes.get(prefix)?;
        Some(format!("<{base}{local}>"))
    }

    /// Contract an angle-bracket IRI to a prefixed name.
    ///
    /// The IRI may be given with or without angle brackets.  Returns `None`
    /// when no registered prefix matches.
    pub fn contract(&self, iri: &str) -> Option<String> {
        let bare = strip_angle_brackets(iri);
        // Find the longest matching prefix base to avoid ambiguous contractions.
        let mut best: Option<(&str, &str)> = None; // (prefix_name, base_iri)
        for (prefix, base) in &self.prefixes {
            if bare.starts_with(base.as_str()) {
                let is_longer = best.as_ref().is_none_or(|(_, b)| base.len() > b.len());
                if is_longer {
                    best = Some((prefix.as_str(), base.as_str()));
                }
            }
        }
        let (prefix_name, base_iri) = best?;
        let local = &bare[base_iri.len()..];
        Some(format!("{prefix_name}:{local}"))
    }

    /// Number of registered prefixes.
    pub fn num_prefixes(&self) -> usize {
        self.prefixes.len()
    }

    /// Create a registry pre-loaded with common RDF/OWL/XSD/Schema prefixes.
    pub fn with_common_prefixes() -> Self {
        let mut reg = Self::new();
        reg.register("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        reg.register("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        reg.register("owl", "http://www.w3.org/2002/07/owl#");
        reg.register("xsd", "http://www.w3.org/2001/XMLSchema#");
        reg.register("schema", "https://schema.org/");
        reg
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BulkFormat
// ─────────────────────────────────────────────────────────────────────────────

/// Supported serialisation formats for bulk RDF I/O.
#[derive(Debug, Clone, PartialEq)]
pub enum BulkFormat {
    /// One triple per line: `<s> <p> <o> .`
    NTriples,
    /// Tab-separated values: `subject\tpredicate\tobject`
    Tsv,
    /// Simplified Turtle with `@prefix` declarations.
    Turtle,
}

// ─────────────────────────────────────────────────────────────────────────────
// BulkIoStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics produced by a bulk import or export operation.
#[derive(Debug, Clone)]
pub struct BulkIoStats {
    pub triples_processed: usize,
    pub triples_skipped: usize,
    pub parse_errors: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub literal_count: usize,
    pub format: BulkFormat,
}

impl BulkIoStats {
    /// Compute statistics from a slice of triples.
    pub fn compute(triples: &[RdfTriple], format: BulkFormat) -> Self {
        let mut subjects: HashSet<&str> = HashSet::new();
        let mut predicates: HashSet<&str> = HashSet::new();
        let mut objects: HashSet<&str> = HashSet::new();
        let mut literal_count = 0usize;

        for t in triples {
            subjects.insert(t.subject.as_str());
            predicates.insert(t.predicate.as_str());
            objects.insert(t.object.as_str());
            if t.is_literal_object() {
                literal_count += 1;
            }
        }

        Self {
            triples_processed: triples.len(),
            triples_skipped: 0,
            parse_errors: 0,
            unique_subjects: subjects.len(),
            unique_predicates: predicates.len(),
            unique_objects: objects.len(),
            literal_count,
            format,
        }
    }

    /// Return a human-readable one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "format={:?} triples={} skipped={} errors={} \
             subjects={} predicates={} objects={} literals={}",
            self.format,
            self.triples_processed,
            self.triples_skipped,
            self.parse_errors,
            self.unique_subjects,
            self.unique_predicates,
            self.unique_objects,
            self.literal_count,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RdfBulkImporter
// ─────────────────────────────────────────────────────────────────────────────

/// Parses bulk RDF text into a `Vec<RdfTriple>`.
pub struct RdfBulkImporter {
    pub registry: NamespaceRegistry,
    pub skip_errors: bool,
    pub max_triples: Option<usize>,
}

impl RdfBulkImporter {
    /// Create a new importer with an empty namespace registry.
    pub fn new() -> Self {
        Self {
            registry: NamespaceRegistry::new(),
            skip_errors: false,
            max_triples: None,
        }
    }

    /// Create a new importer using the supplied namespace registry.
    pub fn with_registry(registry: NamespaceRegistry) -> Self {
        Self {
            registry,
            skip_errors: false,
            max_triples: None,
        }
    }

    /// Configure whether parse errors should be skipped (default: `false`).
    pub fn with_skip_errors(mut self, skip: bool) -> Self {
        self.skip_errors = skip;
        self
    }

    /// Stop importing after `max` triples have been collected.
    pub fn with_max_triples(mut self, max: usize) -> Self {
        self.max_triples = Some(max);
        self
    }

    // ── Public parse entry points ────────────────────────────────────────────

    /// Parse N-Triples format (`<s> <p> <o> .` per line).
    pub fn parse_ntriples(
        &self,
        input: &str,
    ) -> Result<(Vec<RdfTriple>, BulkIoStats), BulkIoError> {
        let mut triples: Vec<RdfTriple> = Vec::new();
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for (idx, raw_line) in input.lines().enumerate() {
            let line_num = idx + 1;
            let line = raw_line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match Self::parse_ntriples_triple(line) {
                Ok(triple) => {
                    triples.push(triple);
                    if let Some(max) = self.max_triples {
                        if triples.len() >= max {
                            break;
                        }
                    }
                }
                Err(e) => {
                    errors += 1;
                    if self.skip_errors {
                        skipped += 1;
                        let _ = e; // intentionally consumed
                    } else {
                        return Err(BulkIoError::ParseError {
                            line: line_num,
                            message: e.to_string(),
                        });
                    }
                }
            }
        }

        let mut stats = BulkIoStats::compute(&triples, BulkFormat::NTriples);
        stats.triples_skipped = skipped;
        stats.parse_errors = errors;
        Ok((triples, stats))
    }

    /// Parse TSV format (`subject\tpredicate\tobject` per line).
    pub fn parse_tsv(&self, input: &str) -> Result<(Vec<RdfTriple>, BulkIoStats), BulkIoError> {
        let mut triples: Vec<RdfTriple> = Vec::new();
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for (idx, raw_line) in input.lines().enumerate() {
            let line_num = idx + 1;
            let line = raw_line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            if parts.len() < 3 {
                errors += 1;
                if self.skip_errors {
                    skipped += 1;
                    continue;
                } else {
                    return Err(BulkIoError::MalformedLine {
                        line: line_num,
                        content: line.to_string(),
                    });
                }
            }

            let triple = RdfTriple::new(parts[0].trim(), parts[1].trim(), parts[2].trim());
            triples.push(triple);

            if let Some(max) = self.max_triples {
                if triples.len() >= max {
                    break;
                }
            }
        }

        let mut stats = BulkIoStats::compute(&triples, BulkFormat::Tsv);
        stats.triples_skipped = skipped;
        stats.parse_errors = errors;
        Ok((triples, stats))
    }

    /// Parse simplified Turtle: `@prefix` declarations followed by triple lines.
    ///
    /// Blank node syntax and nested structures are not supported in this
    /// lightweight implementation.
    pub fn parse_turtle(&self, input: &str) -> Result<(Vec<RdfTriple>, BulkIoStats), BulkIoError> {
        // Build a local registry by cloning the importer's registry, then
        // augment it with any @prefix declarations found in the input.
        let mut local_reg = self.registry.clone();
        let mut triples: Vec<RdfTriple> = Vec::new();
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for (idx, raw_line) in input.lines().enumerate() {
            let line_num = idx + 1;
            let line = raw_line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Handle @prefix declarations.
            if line.to_ascii_lowercase().starts_with("@prefix") {
                if let Some((prefix, base_iri)) = Self::parse_turtle_prefix(line) {
                    local_reg.register(prefix, base_iri);
                }
                continue;
            }

            // Triple lines: possibly ending with `.`
            let triple_str = line.trim_end_matches('.').trim();
            if triple_str.is_empty() {
                continue;
            }

            match self.parse_turtle_triple(triple_str, &local_reg) {
                Ok(triple) => {
                    triples.push(triple);
                    if let Some(max) = self.max_triples {
                        if triples.len() >= max {
                            break;
                        }
                    }
                }
                Err(e) => {
                    errors += 1;
                    if self.skip_errors {
                        skipped += 1;
                        let _ = (line_num, e);
                    } else {
                        return Err(BulkIoError::ParseError {
                            line: line_num,
                            message: e.to_string(),
                        });
                    }
                }
            }
        }

        let mut stats = BulkIoStats::compute(&triples, BulkFormat::Turtle);
        stats.triples_skipped = skipped;
        stats.parse_errors = errors;
        Ok((triples, stats))
    }

    /// Auto-detect the format and dispatch to the appropriate parser.
    ///
    /// Detection logic (first non-blank, non-comment line):
    /// - Starts with `@prefix` → Turtle
    /// - Contains `\t` → TSV
    /// - Contains `<` and ends with `.` → N-Triples
    /// - Falls back to N-Triples
    pub fn parse_auto(&self, input: &str) -> Result<(Vec<RdfTriple>, BulkIoStats), BulkIoError> {
        let detected = detect_format(input);
        match detected {
            BulkFormat::Turtle => self.parse_turtle(input),
            BulkFormat::Tsv => self.parse_tsv(input),
            BulkFormat::NTriples => self.parse_ntriples(input),
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Parse a single N-Triples line.
    fn parse_ntriples_triple(line: &str) -> Result<RdfTriple, BulkIoError> {
        // Strip trailing ` .` or `.`
        let stripped = line.trim_end_matches('.').trim();

        // Tokenise respecting quoted literals.
        let tokens = tokenize_nt_line(stripped)?;

        if tokens.len() < 3 {
            return Err(BulkIoError::InvalidTriple(format!(
                "expected 3 tokens, got {} in: {line}",
                tokens.len()
            )));
        }

        Ok(RdfTriple::new(&tokens[0], &tokens[1], &tokens[2]))
    }

    /// Parse a `@prefix` declaration and return `(prefix_name, base_iri)`.
    fn parse_turtle_prefix(line: &str) -> Option<(String, String)> {
        // Expected: `@prefix rdf: <http://...> .`
        let after_keyword = line
            .strip_prefix("@prefix")
            .or_else(|| line.strip_prefix("@PREFIX"))?
            .trim();

        let colon_pos = after_keyword.find(':')?;
        let prefix_name = after_keyword[..colon_pos].trim().to_string();

        let rest = after_keyword[colon_pos + 1..].trim();
        // Extract the IRI between < and >
        let iri_start = rest.find('<')? + 1;
        let iri_end = rest.find('>')?;
        if iri_start >= iri_end {
            return None;
        }
        let base_iri = rest[iri_start..iri_end].to_string();
        Some((prefix_name, base_iri))
    }

    /// Parse a single Turtle triple line using a local namespace registry.
    fn parse_turtle_triple(
        &self,
        line: &str,
        reg: &NamespaceRegistry,
    ) -> Result<RdfTriple, BulkIoError> {
        let tokens = tokenize_nt_line(line)?;
        if tokens.len() < 3 {
            return Err(BulkIoError::InvalidTriple(format!(
                "expected 3 tokens, got {} in: {line}",
                tokens.len()
            )));
        }

        let s = expand_token(&tokens[0], reg)?;
        let p = expand_token(&tokens[1], reg)?;
        let o = expand_token(&tokens[2], reg)?;

        Ok(RdfTriple::new(s, p, o))
    }
}

impl Default for RdfBulkImporter {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RdfBulkExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Serialises a slice of `RdfTriple` to various text formats.
pub struct RdfBulkExporter {
    pub registry: NamespaceRegistry,
    pub sort_triples: bool,
}

impl RdfBulkExporter {
    /// Create a new exporter with an empty namespace registry.
    pub fn new() -> Self {
        Self {
            registry: NamespaceRegistry::new(),
            sort_triples: false,
        }
    }

    /// Create a new exporter using the supplied namespace registry.
    pub fn with_registry(registry: NamespaceRegistry) -> Self {
        Self {
            registry,
            sort_triples: false,
        }
    }

    /// Configure whether triples should be sorted before serialisation.
    pub fn with_sort(mut self, sort: bool) -> Self {
        self.sort_triples = sort;
        self
    }

    /// Export as N-Triples (one triple per line).
    pub fn export_ntriples(&self, triples: &[RdfTriple]) -> String {
        let sorted;
        let triples = if self.sort_triples {
            sorted = sort_triples_slice(triples);
            sorted.as_slice()
        } else {
            triples
        };

        triples
            .iter()
            .map(|t| t.to_ntriples())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export as TSV (tab-separated, no angle brackets).
    pub fn export_tsv(&self, triples: &[RdfTriple]) -> String {
        let sorted;
        let triples = if self.sort_triples {
            sorted = sort_triples_slice(triples);
            sorted.as_slice()
        } else {
            triples
        };

        triples
            .iter()
            .map(|t| t.to_tsv())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Export as simplified Turtle with `@prefix` declarations at the top.
    pub fn export_turtle(&self, triples: &[RdfTriple]) -> String {
        let sorted;
        let triples = if self.sort_triples {
            sorted = sort_triples_slice(triples);
            sorted.as_slice()
        } else {
            triples
        };

        let mut output = String::new();

        // Emit prefix declarations (sorted for determinism).
        let mut prefix_entries: Vec<(&String, &String)> = self.registry.prefixes.iter().collect();
        prefix_entries.sort_by_key(|(k, _)| k.as_str());
        for (prefix, base_iri) in &prefix_entries {
            output.push_str(&format!("@prefix {prefix}: <{base_iri}> .\n"));
        }

        if !prefix_entries.is_empty() {
            output.push('\n');
        }

        for triple in triples {
            let s = self.contract_or_keep(&triple.subject);
            let p = self.contract_or_keep(&triple.predicate);
            let o = self.contract_or_keep(&triple.object);
            output.push_str(&format!("{s} {p} {o} .\n"));
        }

        output
    }

    /// Attempt to contract a term using the registry; keep original if none found.
    fn contract_or_keep(&self, term: &str) -> String {
        self.registry
            .contract(term)
            .unwrap_or_else(|| term.to_string())
    }
}

impl Default for RdfBulkExporter {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the format of an RDF text block by inspecting the first meaningful line.
fn detect_format(input: &str) -> BulkFormat {
    for raw in input.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.to_ascii_lowercase().starts_with("@prefix") {
            return BulkFormat::Turtle;
        }
        if line.contains('\t') {
            return BulkFormat::Tsv;
        }
        // N-Triples heuristic: contains '<' and line ends with '.'
        if line.contains('<') && line.ends_with('.') {
            return BulkFormat::NTriples;
        }
        // Default fall-through
        break;
    }
    BulkFormat::NTriples
}

/// Tokenise a (possibly literal-containing) N-Triples or Turtle line into at
/// most three tokens while respecting `"…"` quoted strings and `<…>` IRIs.
fn tokenize_nt_line(line: &str) -> Result<Vec<String>, BulkIoError> {
    let mut tokens: Vec<String> = Vec::with_capacity(3);
    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    let mut i = 0usize;

    while i < len && tokens.len() < 3 {
        // Skip whitespace between tokens
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        match chars[i] {
            // IRI enclosed in angle brackets
            '<' => {
                let start = i;
                i += 1;
                while i < len && chars[i] != '>' {
                    i += 1;
                }
                if i >= len {
                    return Err(BulkIoError::InvalidIri(chars[start..].iter().collect()));
                }
                i += 1; // consume '>'
                tokens.push(chars[start..i].iter().collect());
            }
            // Literal enclosed in double quotes
            '"' => {
                let start = i;
                i += 1;
                // Skip escaped characters and find end of literal string
                while i < len {
                    if chars[i] == '\\' {
                        i += 2; // skip escape sequence
                    } else if chars[i] == '"' {
                        i += 1; // consume closing quote
                        break;
                    } else {
                        i += 1;
                    }
                }
                // Handle optional ^^<datatype> or @lang suffix
                if i < len && chars[i] == '^' && i + 1 < len && chars[i + 1] == '^' {
                    i += 2;
                    // Consume the datatype IRI
                    if i < len && chars[i] == '<' {
                        i += 1;
                        while i < len && chars[i] != '>' {
                            i += 1;
                        }
                        if i < len {
                            i += 1; // consume '>'
                        }
                    }
                } else if i < len && chars[i] == '@' {
                    i += 1;
                    while i < len && (chars[i].is_alphanumeric() || chars[i] == '-') {
                        i += 1;
                    }
                }
                tokens.push(chars[start..i].iter().collect());
            }
            // Prefixed name or bare token (e.g. `rdf:type`, `a`)
            _ => {
                let start = i;
                while i < len && !chars[i].is_whitespace() {
                    i += 1;
                }
                tokens.push(chars[start..i].iter().collect());
            }
        }
    }

    Ok(tokens)
}

/// Expand a single token (IRI or prefixed name) using the provided registry.
fn expand_token(token: &str, reg: &NamespaceRegistry) -> Result<String, BulkIoError> {
    if token.starts_with('<') {
        // Already an absolute IRI
        return Ok(token.to_string());
    }
    if token.starts_with('"') {
        // Literal – no expansion needed
        return Ok(token.to_string());
    }
    // Turtle shorthand `a` for rdf:type
    if token == "a" {
        return reg
            .expand("rdf:type")
            .ok_or_else(|| BulkIoError::UnknownPrefix("rdf".to_string()));
    }
    if token.contains(':') {
        return reg.expand(token).ok_or_else(|| {
            let prefix = token.split(':').next().unwrap_or(token).to_string();
            BulkIoError::UnknownPrefix(prefix)
        });
    }
    // Bare term – return as-is wrapped in angle brackets
    Ok(format!("<{token}>"))
}

/// Wrap a term in `<…>` if it is not already enclosed and is not a literal.
fn wrap_iri_if_needed(term: &str) -> String {
    if term.starts_with('<') || term.starts_with('"') {
        term.to_string()
    } else {
        format!("<{term}>")
    }
}

/// Strip angle brackets from an IRI term, leaving literals untouched.
fn strip_angle_brackets(term: &str) -> &str {
    if term.starts_with('<') && term.ends_with('>') {
        &term[1..term.len() - 1]
    } else {
        term
    }
}

/// Produce a sorted clone of a triple slice for deterministic output.
fn sort_triples_slice(triples: &[RdfTriple]) -> Vec<RdfTriple> {
    let mut v = triples.to_vec();
    v.sort_by(|a, b| {
        a.subject
            .cmp(&b.subject)
            .then_with(|| a.predicate.cmp(&b.predicate))
            .then_with(|| a.object.cmp(&b.object))
    });
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RdfTriple ────────────────────────────────────────────────────────────

    #[test]
    fn test_rdf_triple_new() {
        let t = RdfTriple::new(
            "<http://example.org/s>",
            "<http://example.org/p>",
            "<http://example.org/o>",
        );
        assert_eq!(t.subject, "<http://example.org/s>");
        assert_eq!(t.predicate, "<http://example.org/p>");
        assert_eq!(t.object, "<http://example.org/o>");
    }

    #[test]
    fn test_rdf_triple_is_literal() {
        let t_lit = RdfTriple::new("<s>", "<p>", "\"hello\"");
        assert!(t_lit.is_literal_object());
        assert!(!t_lit.is_iri_object());

        let t_iri = RdfTriple::new("<s>", "<p>", "<http://example.org/o>");
        assert!(!t_iri.is_literal_object());
        assert!(t_iri.is_iri_object());
    }

    #[test]
    fn test_rdf_triple_to_ntriples() {
        let t = RdfTriple::new(
            "<http://example.org/s>",
            "<http://example.org/p>",
            "<http://example.org/o>",
        );
        assert_eq!(
            t.to_ntriples(),
            "<http://example.org/s> <http://example.org/p> <http://example.org/o> ."
        );
    }

    #[test]
    fn test_rdf_triple_to_tsv() {
        let t = RdfTriple::new(
            "<http://example.org/s>",
            "<http://example.org/p>",
            "<http://example.org/o>",
        );
        let tsv = t.to_tsv();
        let parts: Vec<&str> = tsv.splitn(3, '\t').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "http://example.org/s");
        assert_eq!(parts[1], "http://example.org/p");
        assert_eq!(parts[2], "http://example.org/o");
    }

    // ── NamespaceRegistry ────────────────────────────────────────────────────

    #[test]
    fn test_namespace_registry_register_expand() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/");
        let expanded = reg.expand("ex:Person");
        assert_eq!(expanded, Some("<http://example.org/Person>".to_string()));
    }

    #[test]
    fn test_namespace_registry_contract() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/");
        let contracted = reg.contract("<http://example.org/Person>");
        assert_eq!(contracted, Some("ex:Person".to_string()));
    }

    #[test]
    fn test_namespace_registry_with_common_prefixes() {
        let reg = NamespaceRegistry::with_common_prefixes();
        assert!(reg.num_prefixes() >= 4);
        assert_eq!(
            reg.expand("rdf:type"),
            Some("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>".to_string())
        );
        assert_eq!(
            reg.expand("xsd:string"),
            Some("<http://www.w3.org/2001/XMLSchema#string>".to_string())
        );
    }

    #[test]
    fn test_namespace_registry_unknown_prefix() {
        let reg = NamespaceRegistry::new();
        assert!(reg.expand("unknown:foo").is_none());
    }

    // ── RdfBulkImporter – N-Triples ──────────────────────────────────────────

    #[test]
    fn test_bulk_importer_parse_ntriples_basic() {
        let input = "<http://a.org/s> <http://a.org/p> <http://a.org/o> .";
        let importer = RdfBulkImporter::new();
        let (triples, stats) = importer.parse_ntriples(input).expect("parse failed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "<http://a.org/s>");
        assert_eq!(stats.triples_processed, 1);
    }

    #[test]
    fn test_bulk_importer_parse_ntriples_blank_lines() {
        let input = "\n# comment\n<http://a.org/s> <http://a.org/p> <http://a.org/o> .\n\n";
        let importer = RdfBulkImporter::new();
        let (triples, _stats) = importer.parse_ntriples(input).expect("parse failed");
        assert_eq!(triples.len(), 1);
    }

    // ── RdfBulkImporter – TSV ────────────────────────────────────────────────

    #[test]
    fn test_bulk_importer_parse_tsv_basic() {
        let input =
            "http://a.org/s\thttp://a.org/p\thttp://a.org/o\nhttp://a.org/s2\thttp://a.org/p\thttp://a.org/o2";
        let importer = RdfBulkImporter::new();
        let (triples, stats) = importer.parse_tsv(input).expect("parse failed");
        assert_eq!(triples.len(), 2);
        assert_eq!(stats.triples_processed, 2);
        assert_eq!(triples[0].subject, "http://a.org/s");
    }

    // ── RdfBulkImporter – Turtle ─────────────────────────────────────────────

    #[test]
    fn test_bulk_importer_parse_turtle_with_prefix() {
        let input = "@prefix ex: <http://example.org/> .\nex:Alice ex:knows ex:Bob .";
        let importer = RdfBulkImporter::new();
        let (triples, _stats) = importer.parse_turtle(input).expect("parse failed");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, "<http://example.org/Alice>");
        assert_eq!(triples[0].object, "<http://example.org/Bob>");
    }

    // ── RdfBulkImporter – Auto-detect ────────────────────────────────────────

    #[test]
    fn test_bulk_importer_parse_auto_detects_ntriples() {
        let input = "<http://a.org/s> <http://a.org/p> <http://a.org/o> .";
        let importer = RdfBulkImporter::new();
        let (triples, stats) = importer.parse_auto(input).expect("parse failed");
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.format, BulkFormat::NTriples);
    }

    #[test]
    fn test_bulk_importer_parse_auto_detects_tsv() {
        let input = "http://a.org/s\thttp://a.org/p\thttp://a.org/o";
        let importer = RdfBulkImporter::new();
        let (triples, stats) = importer.parse_auto(input).expect("parse failed");
        assert_eq!(triples.len(), 1);
        assert_eq!(stats.format, BulkFormat::Tsv);
    }

    // ── max_triples ──────────────────────────────────────────────────────────

    #[test]
    fn test_bulk_importer_max_triples() {
        let input = "<http://a.org/s1> <http://a.org/p> <http://a.org/o1> .\n\
                     <http://a.org/s2> <http://a.org/p> <http://a.org/o2> .\n\
                     <http://a.org/s3> <http://a.org/p> <http://a.org/o3> .";
        let importer = RdfBulkImporter::new().with_max_triples(2);
        let (triples, _stats) = importer.parse_ntriples(input).expect("parse failed");
        assert_eq!(triples.len(), 2);
    }

    // ── RdfBulkExporter ──────────────────────────────────────────────────────

    #[test]
    fn test_bulk_exporter_ntriples_roundtrip() {
        let input = "<http://a.org/s> <http://a.org/p> <http://a.org/o> .";
        let importer = RdfBulkImporter::new();
        let (triples, _) = importer.parse_ntriples(input).expect("parse");
        let exporter = RdfBulkExporter::new();
        let output = exporter.export_ntriples(&triples);
        // Re-parse the exported output
        let (triples2, _) = importer.parse_ntriples(&output).expect("re-parse");
        assert_eq!(triples, triples2);
    }

    #[test]
    fn test_bulk_exporter_tsv() {
        let triples = vec![RdfTriple::new(
            "<http://a.org/s>",
            "<http://a.org/p>",
            "<http://a.org/o>",
        )];
        let exporter = RdfBulkExporter::new();
        let tsv = exporter.export_tsv(&triples);
        assert!(tsv.contains('\t'));
        assert!(!tsv.contains('<')); // angle brackets stripped
    }

    #[test]
    fn test_bulk_exporter_turtle_has_prefixes() {
        let mut reg = NamespaceRegistry::new();
        reg.register("ex", "http://example.org/");
        let triples = vec![RdfTriple::new(
            "<http://example.org/s>",
            "<http://example.org/p>",
            "<http://example.org/o>",
        )];
        let exporter = RdfBulkExporter::with_registry(reg);
        let turtle = exporter.export_turtle(&triples);
        assert!(turtle.contains("@prefix ex:"));
    }

    // ── BulkIoStats ──────────────────────────────────────────────────────────

    #[test]
    fn test_bulk_io_stats_compute() {
        let triples = vec![
            RdfTriple::new("<http://a.org/s>", "<http://a.org/p>", "<http://a.org/o>"),
            RdfTriple::new("<http://a.org/s>", "<http://a.org/p>", "\"literal\""),
        ];
        let stats = BulkIoStats::compute(&triples, BulkFormat::NTriples);
        assert_eq!(stats.triples_processed, 2);
        assert_eq!(stats.unique_subjects, 1);
        assert_eq!(stats.literal_count, 1);
    }

    #[test]
    fn test_bulk_io_stats_summary_nonempty() {
        let triples = vec![RdfTriple::new("<s>", "<p>", "<o>")];
        let stats = BulkIoStats::compute(&triples, BulkFormat::Tsv);
        let summary = stats.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("triples=1"));
    }
}
