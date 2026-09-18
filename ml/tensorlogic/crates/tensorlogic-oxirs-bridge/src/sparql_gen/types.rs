//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::render_pattern;

/// A SPARQL graph pattern — the building blocks of a WHERE clause.
#[derive(Debug, Clone)]
pub enum GraphPattern {
    /// A simple triple pattern.
    Triple(TriplePattern),
    /// `OPTIONAL { ... }`.
    Optional(Vec<GraphPattern>),
    /// `{ ... } UNION { ... }`.
    Union(Vec<GraphPattern>, Vec<GraphPattern>),
    /// `FILTER ( ... )`.
    Filter(SparqlFilter),
    /// `BIND ( expr AS ?var )`.
    Bind(SparqlExpr, String),
    /// `VALUES (?v1 ?v2 …) { (t1 t2) … }` — inline value table.
    Values(Vec<String>, Vec<Vec<SparqlTerm>>),
    /// An anonymous group `{ pattern* }`.
    Group(Vec<GraphPattern>),
}
/// A single SPARQL term used in triple patterns and filter expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum SparqlTerm {
    /// A SPARQL variable: `?name`.
    Variable(String),
    /// An IRI: `<http://...>` or `prefix:local`.
    Iri(String),
    /// A plain string literal: `"value"`.
    Literal(String),
    /// A numeric literal: `42.0`.
    NumericLiteral(f64),
    /// A boolean literal: `true` / `false`.
    BoolLiteral(bool),
}
impl SparqlTerm {
    /// Create a variable term.
    pub fn var(name: impl Into<String>) -> Self {
        SparqlTerm::Variable(name.into())
    }
    /// Create an IRI term.
    pub fn iri(uri: impl Into<String>) -> Self {
        SparqlTerm::Iri(uri.into())
    }
    /// Create a plain string literal term.
    pub fn literal(s: impl Into<String>) -> Self {
        SparqlTerm::Literal(s.into())
    }
    /// Render this term as a SPARQL string fragment.
    ///
    /// * `Variable("x")` → `"?x"`
    /// * `Iri("http://example.org/foo")` → `"<http://example.org/foo>"`
    /// * `Iri("ex:foo")` → `"ex:foo"` (no angle brackets for prefixed names)
    /// * `Literal("hello")` → `"\"hello\""`
    /// * `NumericLiteral(3.14)` → `"3.14"`
    /// * `BoolLiteral(true)` → `"true"`
    pub fn as_sparql_string(&self) -> String {
        match self {
            SparqlTerm::Variable(name) => format!("?{name}"),
            SparqlTerm::Iri(uri) => {
                if uri.starts_with("http://")
                    || uri.starts_with("https://")
                    || uri.starts_with("urn:")
                {
                    format!("<{uri}>")
                } else {
                    uri.clone()
                }
            }
            SparqlTerm::Literal(s) => {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            }
            SparqlTerm::NumericLiteral(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15_f64 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            SparqlTerm::BoolLiteral(b) => {
                if *b {
                    "true".to_owned()
                } else {
                    "false".to_owned()
                }
            }
        }
    }
}
/// A fully constructed SPARQL SELECT query ready for rendering.
#[derive(Debug, Clone)]
pub struct SparqlQuery {
    /// Prefix declarations: `(prefix_label, iri)`.
    pub prefixes: Vec<(String, String)>,
    /// Projected variable names (empty → `SELECT *`).
    pub select_vars: Vec<String>,
    /// Whether to add `DISTINCT`.
    pub distinct: bool,
    /// WHERE clause patterns.
    pub patterns: Vec<GraphPattern>,
    /// `LIMIT n`.
    pub limit: Option<usize>,
    /// `OFFSET n`.
    pub offset: Option<usize>,
    /// `ORDER BY ?var`.
    pub order_by: Option<String>,
}
impl SparqlQuery {
    /// Create an empty SPARQL SELECT query.
    pub fn new() -> Self {
        SparqlQuery {
            prefixes: Vec::new(),
            select_vars: Vec::new(),
            distinct: false,
            patterns: Vec::new(),
            limit: None,
            offset: None,
            order_by: None,
        }
    }
    /// Add a prefix declaration.
    pub fn with_prefix(mut self, prefix: &str, iri: &str) -> Self {
        self.prefixes.push((prefix.to_owned(), iri.to_owned()));
        self
    }
    /// Specify projected variables (`SELECT ?a ?b ...`).
    pub fn select(mut self, vars: &[&str]) -> Self {
        for v in vars {
            self.select_vars.push((*v).to_owned());
        }
        self
    }
    /// Add `DISTINCT` to the query.
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }
    /// Add `LIMIT n`.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }
    /// Add `OFFSET n`.
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }
    /// Add `ORDER BY ?var`.
    pub fn order_by(mut self, var: &str) -> Self {
        self.order_by = Some(var.to_owned());
        self
    }
    /// Append a graph pattern to the WHERE clause.
    pub fn add_pattern(mut self, p: GraphPattern) -> Self {
        self.patterns.push(p);
        self
    }
    /// Render the full query as a SPARQL 1.1 string.
    pub fn to_sparql(&self) -> String {
        let mut out = String::new();
        for (prefix, iri) in &self.prefixes {
            out.push_str(&format!("PREFIX {prefix}: <{iri}>\n"));
        }
        if !self.prefixes.is_empty() {
            out.push('\n');
        }
        out.push_str("SELECT ");
        if self.distinct {
            out.push_str("DISTINCT ");
        }
        if self.select_vars.is_empty() {
            out.push('*');
        } else {
            let projected: Vec<String> = self.select_vars.iter().map(|v| format!("?{v}")).collect();
            out.push_str(&projected.join(" "));
        }
        out.push('\n');
        out.push_str("WHERE {\n");
        let indent = "  ";
        for pattern in &self.patterns {
            out.push_str(&render_pattern(pattern, indent, 1));
        }
        out.push('}');
        if let Some(ref var) = self.order_by {
            out.push_str(&format!("\nORDER BY ?{var}"));
        }
        if let Some(lim) = self.limit {
            out.push_str(&format!("\nLIMIT {lim}"));
        }
        if let Some(off) = self.offset {
            out.push_str(&format!("\nOFFSET {off}"));
        }
        out
    }
}
/// Configuration for the SPARQL generator.
#[derive(Debug, Clone)]
pub struct SparqlGenConfig {
    /// Base IRI prefix for predicates (default `"http://tensorlogic.org/ont#"`).
    pub base_prefix: String,
    /// Prefix string for auto-generated variable names (default `"v_"`).
    pub variable_prefix: String,
    /// Render `∃x. P(x)` as `OPTIONAL { ?x ... }` instead of simply introducing `?x`.
    pub use_optional_for_exists: bool,
    /// Indentation string (default two spaces).
    pub indent: String,
    /// Maximum recursion depth before returning an error (default 50).
    pub max_depth: usize,
}
/// A SPARQL expression used inside `BIND ( expr AS ?var )`.
#[derive(Debug, Clone)]
pub enum SparqlExpr {
    Term(SparqlTerm),
    Add(Box<SparqlExpr>, Box<SparqlExpr>),
    Sub(Box<SparqlExpr>, Box<SparqlExpr>),
    Mul(Box<SparqlExpr>, Box<SparqlExpr>),
    Div(Box<SparqlExpr>, Box<SparqlExpr>),
    /// `STR(?x)`.
    Str(Box<SparqlExpr>),
    /// `LANG(?x)`.
    Lang(Box<SparqlExpr>),
}
impl SparqlExpr {
    pub(super) fn as_sparql_string(&self) -> String {
        match self {
            SparqlExpr::Term(t) => t.as_sparql_string(),
            SparqlExpr::Add(l, r) => {
                format!("({} + {})", l.as_sparql_string(), r.as_sparql_string())
            }
            SparqlExpr::Sub(l, r) => {
                format!("({} - {})", l.as_sparql_string(), r.as_sparql_string())
            }
            SparqlExpr::Mul(l, r) => {
                format!("({} * {})", l.as_sparql_string(), r.as_sparql_string())
            }
            SparqlExpr::Div(l, r) => {
                format!("({} / {})", l.as_sparql_string(), r.as_sparql_string())
            }
            SparqlExpr::Str(inner) => format!("STR({})", inner.as_sparql_string()),
            SparqlExpr::Lang(inner) => format!("LANG({})", inner.as_sparql_string()),
        }
    }
}
/// A SPARQL filter expression used inside `FILTER ( ... )`.
#[derive(Debug, Clone)]
pub enum SparqlFilter {
    Equals(SparqlTerm, SparqlTerm),
    NotEquals(SparqlTerm, SparqlTerm),
    LessThan(SparqlTerm, SparqlTerm),
    GreaterThan(SparqlTerm, SparqlTerm),
    LessOrEqual(SparqlTerm, SparqlTerm),
    GreaterOrEqual(SparqlTerm, SparqlTerm),
    And(Box<SparqlFilter>, Box<SparqlFilter>),
    Or(Box<SparqlFilter>, Box<SparqlFilter>),
    Not(Box<SparqlFilter>),
    /// `BOUND(?var)`.
    Bound(String),
    /// `isIRI(?term)`.
    IsIri(SparqlTerm),
    /// `REGEX(?var, "pattern")`.
    Regex(SparqlTerm, String),
    /// Raw boolean literal filter.
    BoolValue(bool),
    /// `NOT EXISTS { ... }`.
    NotExists(Vec<GraphPattern>),
}
/// A SPARQL triple pattern: `(subject predicate object)`.
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: SparqlTerm,
    pub predicate: SparqlTerm,
    pub object: SparqlTerm,
}
impl TriplePattern {
    /// Create a new triple pattern.
    pub fn new(subject: SparqlTerm, predicate: SparqlTerm, object: SparqlTerm) -> Self {
        TriplePattern {
            subject,
            predicate,
            object,
        }
    }
    /// Render as a SPARQL triple pattern string (without trailing dot or newline).
    pub fn as_sparql_string(&self) -> String {
        format!(
            "{} {} {}",
            self.subject.as_sparql_string(),
            self.predicate.as_sparql_string(),
            self.object.as_sparql_string(),
        )
    }
}
/// Errors that can occur during SPARQL query generation from a [`tensorlogic_ir::TLExpr`].
#[derive(Debug)]
pub enum SparqlGenError {
    /// An expression variant is not yet supported for SPARQL generation.
    UnsupportedExpr(String),
    /// A variable name conflicts with previously introduced variables.
    AmbiguousVariable(String),
    /// The input expression produces no patterns (empty query body).
    EmptyQuery,
}
