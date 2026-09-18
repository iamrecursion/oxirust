//! Programmatic SPARQL query builder — a fluent builder API that constructs
//! SPARQL query strings without requiring manual string formatting.
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_oxirs_bridge::sparql_builder::{
//!     SelectQuery, WhereClause, SparqlTerm, SparqlFilter,
//! };
//!
//! let query = SelectQuery::new()
//!     .prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
//!     .prefix("ex", "http://example.org/")
//!     .select("x")
//!     .select("name")
//!     .where_clause(
//!         WhereClause::new()
//!             .triple(
//!                 SparqlTerm::var("x"),
//!                 SparqlTerm::prefixed("rdf", "type"),
//!                 SparqlTerm::prefixed("ex", "Person"),
//!             )
//!             .filter(SparqlFilter::gt("age", 18.0)),
//!     )
//!     .limit(100)
//!     .build()
//!     .expect("valid query");
//!
//! assert!(query.contains("SELECT"));
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can be produced by the SPARQL query builder.
#[derive(Debug, Clone)]
pub enum SparqlBuilderError {
    /// The WHERE clause contained no patterns (SELECT query requires at least one).
    EmptyWhereClause,
    /// A variable name was syntactically invalid.
    InvalidVariableName(String),
    /// Mutually exclusive modifiers were both set.
    ConflictingModifiers(String),
}

impl fmt::Display for SparqlBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparqlBuilderError::EmptyWhereClause => {
                write!(f, "SPARQL builder error: WHERE clause is empty")
            }
            SparqlBuilderError::InvalidVariableName(name) => {
                write!(
                    f,
                    "SPARQL builder error: invalid variable name '{name}' \
                     (must start with a letter or '_')"
                )
            }
            SparqlBuilderError::ConflictingModifiers(msg) => {
                write!(f, "SPARQL builder error: conflicting modifiers — {msg}")
            }
        }
    }
}

impl std::error::Error for SparqlBuilderError {}

// ---------------------------------------------------------------------------
// validate_variable_name
// ---------------------------------------------------------------------------

/// Validate a SPARQL variable name.
///
/// A valid name must start with a letter (ASCII or Unicode) or `_`, followed
/// by letters, digits, `_`, `-`, or `.`.
pub fn validate_variable_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
}

// ---------------------------------------------------------------------------
// SparqlLiteral
// ---------------------------------------------------------------------------

/// An RDF literal with an optional datatype or language tag.
#[derive(Debug, Clone, PartialEq)]
pub struct SparqlLiteral {
    /// The lexical value of the literal.
    pub value: String,
    /// Optional datatype IRI (e.g. `"xsd:integer"`).
    pub datatype: Option<String>,
    /// Optional BCP-47 language tag (e.g. `"en"`).
    pub lang_tag: Option<String>,
}

// ---------------------------------------------------------------------------
// SparqlTerm
// ---------------------------------------------------------------------------

/// A SPARQL term: IRI, prefixed name, literal, blank node, or variable.
#[derive(Debug, Clone, PartialEq)]
pub enum SparqlTerm {
    /// A full IRI, e.g. `<http://example.org/foo>`.
    Iri(String),
    /// A prefixed name, e.g. `rdf:type`.
    PrefixedName(String, String),
    /// An RDF literal.
    Literal(SparqlLiteral),
    /// A blank node, e.g. `_:b0`.
    BlankNode(String),
    /// A SPARQL variable, e.g. `?x`.
    Variable(String),
}

impl SparqlTerm {
    /// Create an IRI term.  If the string is not already wrapped in `<…>`,
    /// the angle brackets are added automatically when rendered.
    pub fn iri(s: impl Into<String>) -> Self {
        SparqlTerm::Iri(s.into())
    }

    /// Create a variable term.
    pub fn var(name: impl Into<String>) -> Self {
        SparqlTerm::Variable(name.into())
    }

    /// Create a plain string literal.
    pub fn literal(value: impl Into<String>) -> Self {
        SparqlTerm::Literal(SparqlLiteral {
            value: value.into(),
            datatype: None,
            lang_tag: None,
        })
    }

    /// Create a typed literal, e.g. `"42"^^xsd:integer`.
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        SparqlTerm::Literal(SparqlLiteral {
            value: value.into(),
            datatype: Some(datatype.into()),
            lang_tag: None,
        })
    }

    /// Create a language-tagged literal, e.g. `"hello"@en`.
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        SparqlTerm::Literal(SparqlLiteral {
            value: value.into(),
            datatype: None,
            lang_tag: Some(lang.into()),
        })
    }

    /// Create a blank-node term.
    pub fn blank(id: impl Into<String>) -> Self {
        SparqlTerm::BlankNode(id.into())
    }

    /// Create a prefixed-name term.
    pub fn prefixed(prefix: impl Into<String>, local: impl Into<String>) -> Self {
        SparqlTerm::PrefixedName(prefix.into(), local.into())
    }

    /// Render this term as a SPARQL string fragment.
    pub fn to_sparql(&self) -> String {
        match self {
            SparqlTerm::Variable(name) => format!("?{name}"),
            SparqlTerm::Iri(iri) => {
                if iri.starts_with('<') && iri.ends_with('>') {
                    iri.clone()
                } else {
                    format!("<{iri}>")
                }
            }
            SparqlTerm::PrefixedName(prefix, local) => format!("{prefix}:{local}"),
            SparqlTerm::BlankNode(id) => format!("_:{id}"),
            SparqlTerm::Literal(lit) => render_literal(lit),
        }
    }
}

/// Render an [`SparqlLiteral`] to its SPARQL string representation.
fn render_literal(lit: &SparqlLiteral) -> String {
    // Escape double-quotes inside the value.
    let escaped = lit.value.replace('\\', "\\\\").replace('"', "\\\"");
    let base = format!("\"{escaped}\"");
    if let Some(lang) = &lit.lang_tag {
        format!("{base}@{lang}")
    } else if let Some(dt) = &lit.datatype {
        format!("{base}^^{dt}")
    } else {
        base
    }
}

// ---------------------------------------------------------------------------
// TriplePattern (builder variant)
// ---------------------------------------------------------------------------

/// A SPARQL triple pattern `(subject predicate object)`.
#[derive(Debug, Clone)]
pub struct BuilderTriplePattern {
    /// The subject term.
    pub subject: SparqlTerm,
    /// The predicate term.
    pub predicate: SparqlTerm,
    /// The object term.
    pub object: SparqlTerm,
}

impl BuilderTriplePattern {
    /// Create a new triple pattern.
    pub fn new(s: SparqlTerm, p: SparqlTerm, o: SparqlTerm) -> Self {
        BuilderTriplePattern {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    /// Render this triple pattern as a SPARQL line, e.g. `?s rdf:type ?o .`
    pub fn to_sparql(&self) -> String {
        format!(
            "{} {} {} .",
            self.subject.to_sparql(),
            self.predicate.to_sparql(),
            self.object.to_sparql()
        )
    }
}

// ---------------------------------------------------------------------------
// SparqlFilter
// ---------------------------------------------------------------------------

/// A SPARQL FILTER expression stored as a raw expression string.
#[derive(Debug, Clone)]
pub struct SparqlFilter {
    /// The expression string, e.g. `?age > 18`.
    pub expression: String,
}

impl SparqlFilter {
    /// Create a filter from a raw expression string.
    pub fn new(expr: impl Into<String>) -> Self {
        SparqlFilter {
            expression: expr.into(),
        }
    }

    /// `?var > value`
    pub fn gt(var: &str, value: f64) -> Self {
        SparqlFilter::new(format!("?{var} > {value}"))
    }

    /// `?var < value`
    pub fn lt(var: &str, value: f64) -> Self {
        SparqlFilter::new(format!("?{var} < {value}"))
    }

    /// `?var = other`
    pub fn eq(var: &str, other: &str) -> Self {
        SparqlFilter::new(format!("?{var} = {other}"))
    }

    /// `langMatches(lang(?var), "lang")`
    pub fn lang_matches(var: &str, lang: &str) -> Self {
        SparqlFilter::new(format!("langMatches(lang(?{var}), \"{lang}\")"))
    }

    /// `regex(?var, "pattern")`
    pub fn regex(var: &str, pattern: &str) -> Self {
        SparqlFilter::new(format!("regex(?{var}, \"{pattern}\")"))
    }

    /// Logical AND of two filters.
    pub fn and(left: SparqlFilter, right: SparqlFilter) -> Self {
        SparqlFilter::new(format!("({}) && ({})", left.expression, right.expression))
    }

    /// Logical OR of two filters.
    pub fn or(left: SparqlFilter, right: SparqlFilter) -> Self {
        SparqlFilter::new(format!("({}) || ({})", left.expression, right.expression))
    }

    /// Logical NOT of a filter.
    pub fn negate(inner: SparqlFilter) -> Self {
        SparqlFilter::new(format!("!({})", inner.expression))
    }

    /// Render as a `FILTER (...)` clause line.
    pub fn to_sparql(&self) -> String {
        format!("FILTER ({})", self.expression)
    }
}

impl std::ops::Not for SparqlFilter {
    type Output = SparqlFilter;

    fn not(self) -> Self::Output {
        SparqlFilter::negate(self)
    }
}

// ---------------------------------------------------------------------------
// OrderDirection
// ---------------------------------------------------------------------------

/// ORDER BY direction for a single variable.
#[derive(Debug, Clone)]
pub enum OrderDirection {
    /// `ASC(?var)`
    Asc(String),
    /// `DESC(?var)`
    Desc(String),
}

impl OrderDirection {
    /// Render as an ORDER BY term, e.g. `ASC(?x)`.
    pub fn to_sparql(&self) -> String {
        match self {
            OrderDirection::Asc(var) => format!("ASC(?{var})"),
            OrderDirection::Desc(var) => format!("DESC(?{var})"),
        }
    }
}

// ---------------------------------------------------------------------------
// WhereClause
// ---------------------------------------------------------------------------

/// A single item inside a WHERE clause.
#[derive(Debug, Clone)]
pub enum WhereClauseItem {
    /// A triple pattern.
    Triple(BuilderTriplePattern),
    /// A FILTER clause.
    Filter(SparqlFilter),
    /// An OPTIONAL block.
    Optional(WhereClause),
    /// A UNION of two sub-patterns.
    Union(WhereClause, WhereClause),
    /// A BIND assignment: `BIND (<expr> AS ?<var>)`.
    Bind(String, String),
    /// A VALUES inline data block: `VALUES ?var { val1 val2 … }`.
    Values(String, Vec<SparqlTerm>),
    /// A multi-variable VALUES inline table: `VALUES (?v1 ?v2 …) { (t1 t2) … }`.
    ValuesMulti(Vec<String>, Vec<Vec<SparqlTerm>>),
}

/// A SPARQL WHERE clause composed of triple patterns, filters, optional
/// blocks, unions, binds, and inline VALUES blocks.
#[derive(Debug, Clone, Default)]
pub struct WhereClause {
    patterns: Vec<WhereClauseItem>,
}

impl WhereClause {
    /// Create an empty WHERE clause.
    pub fn new() -> Self {
        WhereClause::default()
    }

    /// Append a triple pattern.
    pub fn triple(mut self, s: SparqlTerm, p: SparqlTerm, o: SparqlTerm) -> Self {
        self.patterns
            .push(WhereClauseItem::Triple(BuilderTriplePattern::new(s, p, o)));
        self
    }

    /// Append a FILTER.
    pub fn filter(mut self, f: SparqlFilter) -> Self {
        self.patterns.push(WhereClauseItem::Filter(f));
        self
    }

    /// Append an OPTIONAL block.
    pub fn optional(mut self, inner: WhereClause) -> Self {
        self.patterns.push(WhereClauseItem::Optional(inner));
        self
    }

    /// Append a UNION of two sub-clauses.
    pub fn union(mut self, left: WhereClause, right: WhereClause) -> Self {
        self.patterns.push(WhereClauseItem::Union(left, right));
        self
    }

    /// Append a BIND assignment.
    pub fn bind(mut self, expr: impl Into<String>, var: impl Into<String>) -> Self {
        self.patterns
            .push(WhereClauseItem::Bind(expr.into(), var.into()));
        self
    }

    /// Append an inline VALUES block.
    pub fn values(mut self, var: impl Into<String>, vals: Vec<SparqlTerm>) -> Self {
        self.patterns
            .push(WhereClauseItem::Values(var.into(), vals));
        self
    }

    /// Append a multi-variable inline VALUES table.
    pub fn values_multi(
        mut self,
        vars: Vec<impl Into<String>>,
        rows: Vec<Vec<SparqlTerm>>,
    ) -> Self {
        let vars: Vec<String> = vars.into_iter().map(Into::into).collect();
        self.patterns.push(WhereClauseItem::ValuesMulti(vars, rows));
        self
    }

    /// Returns `true` when the clause has no items.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Number of top-level items in this clause.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Render the full WHERE block with `{` / `}` and 2-space indentation.
    pub fn to_sparql(&self) -> String {
        let mut out = String::from("WHERE {\n");
        for item in &self.patterns {
            render_where_item(&mut out, item, 1);
        }
        out.push('}');
        out
    }

    /// Render only the inner content (without the outer `WHERE { … }` wrapper),
    /// used for nested clauses like OPTIONAL and UNION.
    fn render_inner(&self, depth: usize) -> String {
        let mut out = String::from("{\n");
        for item in &self.patterns {
            render_where_item(&mut out, item, depth + 1);
        }
        push_indent(&mut out, depth);
        out.push('}');
        out
    }
}

/// Push `depth * 2` spaces of indentation into `buf`.
fn push_indent(buf: &mut String, depth: usize) {
    for _ in 0..depth * 2 {
        buf.push(' ');
    }
}

/// Render a single [`WhereClauseItem`] into `buf` at the given indentation
/// depth.
fn render_where_item(buf: &mut String, item: &WhereClauseItem, depth: usize) {
    match item {
        WhereClauseItem::Triple(tp) => {
            push_indent(buf, depth);
            buf.push_str(&tp.to_sparql());
            buf.push('\n');
        }
        WhereClauseItem::Filter(f) => {
            push_indent(buf, depth);
            buf.push_str(&f.to_sparql());
            buf.push('\n');
        }
        WhereClauseItem::Optional(inner) => {
            push_indent(buf, depth);
            buf.push_str("OPTIONAL ");
            buf.push_str(&inner.render_inner(depth));
            buf.push('\n');
        }
        WhereClauseItem::Union(left, right) => {
            push_indent(buf, depth);
            buf.push_str(&left.render_inner(depth));
            buf.push_str(" UNION ");
            buf.push_str(&right.render_inner(depth));
            buf.push('\n');
        }
        WhereClauseItem::Bind(expr, var) => {
            push_indent(buf, depth);
            buf.push_str(&format!("BIND ({expr} AS ?{var})\n"));
        }
        WhereClauseItem::Values(var, vals) => {
            push_indent(buf, depth);
            let vals_str: Vec<String> = vals.iter().map(|v| v.to_sparql()).collect();
            buf.push_str(&format!("VALUES ?{var} {{ {} }}\n", vals_str.join(" ")));
        }
        WhereClauseItem::ValuesMulti(vars, rows) => {
            push_indent(buf, depth);
            let var_list: Vec<String> = vars.iter().map(|v| format!("?{v}")).collect();
            let row_strs: Vec<String> = rows
                .iter()
                .map(|row| {
                    let terms: Vec<String> = row.iter().map(|t| t.to_sparql()).collect();
                    format!("({})", terms.join(" "))
                })
                .collect();
            buf.push_str(&format!(
                "VALUES ({}) {{ {} }}\n",
                var_list.join(" "),
                row_strs.join(" ")
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// SelectQuery
// ---------------------------------------------------------------------------

/// A fluent builder for SPARQL SELECT queries.
#[derive(Debug, Clone)]
pub struct SelectQuery {
    /// PREFIX declarations: `(prefix, iri)`.
    pub prefixes: Vec<(String, String)>,
    /// Variables to project.  Empty means `SELECT *`.
    pub projection: Vec<String>,
    /// Whether to apply `DISTINCT`.
    pub distinct: bool,
    /// Whether to apply `REDUCED`.
    pub reduced: bool,
    /// The WHERE clause.
    pub where_clause: WhereClause,
    /// ORDER BY terms.
    pub order_by: Vec<OrderDirection>,
    /// GROUP BY variables.
    pub group_by: Vec<String>,
    /// HAVING filter (only meaningful when GROUP BY is set).
    pub having: Option<SparqlFilter>,
    /// LIMIT value.
    pub limit: Option<usize>,
    /// OFFSET value.
    pub offset: Option<usize>,
}

impl SelectQuery {
    /// Create a new, empty SELECT query builder.
    pub fn new() -> Self {
        SelectQuery {
            prefixes: Vec::new(),
            projection: Vec::new(),
            distinct: false,
            reduced: false,
            where_clause: WhereClause::new(),
            order_by: Vec::new(),
            group_by: Vec::new(),
            having: None,
            limit: None,
            offset: None,
        }
    }

    /// Add a PREFIX declaration.
    pub fn prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.push((prefix.into(), iri.into()));
        self
    }

    /// Add a variable to the SELECT projection.
    pub fn select(mut self, var: impl Into<String>) -> Self {
        self.projection.push(var.into());
        self
    }

    /// Set projection to `SELECT *`.
    pub fn select_all(mut self) -> Self {
        self.projection.clear();
        self
    }

    /// Add `DISTINCT` to the query.
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Add `REDUCED` to the query.
    pub fn reduced(mut self) -> Self {
        self.reduced = true;
        self
    }

    /// Set the WHERE clause.
    pub fn where_clause(mut self, clause: WhereClause) -> Self {
        self.where_clause = clause;
        self
    }

    /// Add an `ORDER BY ASC(?var)` term.
    pub fn order_by_asc(mut self, var: impl Into<String>) -> Self {
        self.order_by.push(OrderDirection::Asc(var.into()));
        self
    }

    /// Add an `ORDER BY DESC(?var)` term.
    pub fn order_by_desc(mut self, var: impl Into<String>) -> Self {
        self.order_by.push(OrderDirection::Desc(var.into()));
        self
    }

    /// Add a GROUP BY variable.
    pub fn group_by(mut self, var: impl Into<String>) -> Self {
        self.group_by.push(var.into());
        self
    }

    /// Set the HAVING filter.
    pub fn having(mut self, filter: SparqlFilter) -> Self {
        self.having = Some(filter);
        self
    }

    /// Set the LIMIT.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Set the OFFSET.
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Build the SPARQL query string, performing validation first.
    pub fn build(&self) -> Result<String, SparqlBuilderError> {
        // Validate: at least one pattern in WHERE
        if self.where_clause.is_empty() {
            return Err(SparqlBuilderError::EmptyWhereClause);
        }

        // Validate: DISTINCT and REDUCED are mutually exclusive
        if self.distinct && self.reduced {
            return Err(SparqlBuilderError::ConflictingModifiers(
                "DISTINCT and REDUCED cannot both be set".to_string(),
            ));
        }

        // Validate variable names in projection
        for var in &self.projection {
            if !validate_variable_name(var) {
                return Err(SparqlBuilderError::InvalidVariableName(var.clone()));
            }
        }

        Ok(self.build_unchecked())
    }

    /// Build the SPARQL query string without validation.
    pub fn build_unchecked(&self) -> String {
        let mut out = String::new();

        // PREFIX declarations
        for (prefix, iri) in &self.prefixes {
            let iri_str = if iri.starts_with('<') && iri.ends_with('>') {
                iri.clone()
            } else {
                format!("<{iri}>")
            };
            out.push_str(&format!("PREFIX {prefix}: {iri_str}\n"));
        }

        if !self.prefixes.is_empty() {
            out.push('\n');
        }

        // SELECT [DISTINCT|REDUCED] [vars | *]
        out.push_str("SELECT");
        if self.distinct {
            out.push_str(" DISTINCT");
        } else if self.reduced {
            out.push_str(" REDUCED");
        }

        if self.projection.is_empty() {
            out.push_str(" *");
        } else {
            for var in &self.projection {
                out.push_str(&format!(" ?{var}"));
            }
        }
        out.push('\n');

        // WHERE { … }
        out.push_str(&self.where_clause.to_sparql());
        out.push('\n');

        // GROUP BY
        if !self.group_by.is_empty() {
            let vars: Vec<String> = self.group_by.iter().map(|v| format!("?{v}")).collect();
            out.push_str(&format!("GROUP BY {}\n", vars.join(" ")));
        }

        // HAVING
        if let Some(having) = &self.having {
            out.push_str(&format!("HAVING ({})\n", having.expression));
        }

        // ORDER BY
        if !self.order_by.is_empty() {
            let terms: Vec<String> = self.order_by.iter().map(|o| o.to_sparql()).collect();
            out.push_str(&format!("ORDER BY {}\n", terms.join(" ")));
        }

        // LIMIT
        if let Some(limit) = self.limit {
            out.push_str(&format!("LIMIT {limit}\n"));
        }

        // OFFSET
        if let Some(offset) = self.offset {
            out.push_str(&format!("OFFSET {offset}\n"));
        }

        out
    }
}

impl Default for SelectQuery {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AskQuery
// ---------------------------------------------------------------------------

/// A fluent builder for SPARQL ASK queries.
#[derive(Debug, Clone)]
pub struct AskQuery {
    /// PREFIX declarations.
    pub prefixes: Vec<(String, String)>,
    /// The WHERE clause.
    pub where_clause: WhereClause,
}

impl AskQuery {
    /// Create a new, empty ASK query builder.
    pub fn new() -> Self {
        AskQuery {
            prefixes: Vec::new(),
            where_clause: WhereClause::new(),
        }
    }

    /// Add a PREFIX declaration.
    pub fn prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.push((prefix.into(), iri.into()));
        self
    }

    /// Set the WHERE clause.
    pub fn where_clause(mut self, clause: WhereClause) -> Self {
        self.where_clause = clause;
        self
    }

    /// Build the SPARQL ASK query string.
    pub fn build(&self) -> String {
        let mut out = String::new();

        for (prefix, iri) in &self.prefixes {
            let iri_str = if iri.starts_with('<') && iri.ends_with('>') {
                iri.clone()
            } else {
                format!("<{iri}>")
            };
            out.push_str(&format!("PREFIX {prefix}: {iri_str}\n"));
        }

        if !self.prefixes.is_empty() {
            out.push('\n');
        }

        out.push_str("ASK\n");
        out.push_str(&self.where_clause.to_sparql());
        out.push('\n');
        out
    }
}

impl Default for AskQuery {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SparqlTerm rendering -----------------------------------------------

    #[test]
    fn test_sparql_term_variable_render() {
        assert_eq!(SparqlTerm::var("x").to_sparql(), "?x");
    }

    #[test]
    fn test_sparql_term_iri_render_bare() {
        // Bare IRI gets wrapped in angle brackets.
        assert_eq!(
            SparqlTerm::iri("http://example.org/foo").to_sparql(),
            "<http://example.org/foo>"
        );
    }

    #[test]
    fn test_sparql_term_iri_render_already_wrapped() {
        // Already-wrapped IRI is used as-is.
        assert_eq!(
            SparqlTerm::iri("<http://example.org/foo>").to_sparql(),
            "<http://example.org/foo>"
        );
    }

    #[test]
    fn test_sparql_term_literal_render() {
        assert_eq!(SparqlTerm::literal("hello").to_sparql(), "\"hello\"");
    }

    #[test]
    fn test_sparql_term_typed_literal_render() {
        assert_eq!(
            SparqlTerm::typed_literal("42", "xsd:integer").to_sparql(),
            "\"42\"^^xsd:integer"
        );
    }

    #[test]
    fn test_sparql_term_lang_literal_render() {
        assert_eq!(
            SparqlTerm::lang_literal("hello", "en").to_sparql(),
            "\"hello\"@en"
        );
    }

    #[test]
    fn test_sparql_term_prefixed_render() {
        assert_eq!(SparqlTerm::prefixed("rdf", "type").to_sparql(), "rdf:type");
    }

    #[test]
    fn test_sparql_term_blank_node_render() {
        assert_eq!(SparqlTerm::blank("b0").to_sparql(), "_:b0");
    }

    // --- TriplePattern ------------------------------------------------------

    #[test]
    fn test_triple_pattern_to_sparql() {
        let tp = BuilderTriplePattern::new(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("o"),
        );
        assert_eq!(tp.to_sparql(), "?s rdf:type ?o .");
    }

    // --- SparqlFilter -------------------------------------------------------

    #[test]
    fn test_filter_gt() {
        let f = SparqlFilter::gt("x", 18.0);
        assert_eq!(f.to_sparql(), "FILTER (?x > 18)");
    }

    #[test]
    fn test_filter_lt() {
        let f = SparqlFilter::lt("age", 65.0);
        assert_eq!(f.to_sparql(), "FILTER (?age < 65)");
    }

    #[test]
    fn test_filter_and() {
        let left = SparqlFilter::gt("x", 0.0);
        let right = SparqlFilter::lt("x", 100.0);
        let combined = SparqlFilter::and(left, right);
        assert!(combined.to_sparql().contains("&&"));
    }

    #[test]
    fn test_filter_or() {
        let left = SparqlFilter::gt("x", 0.0);
        let right = SparqlFilter::lt("x", 100.0);
        let combined = SparqlFilter::or(left, right);
        assert!(combined.to_sparql().contains("||"));
    }

    #[test]
    fn test_filter_not() {
        let inner = SparqlFilter::gt("x", 18.0);
        let negated = SparqlFilter::negate(inner);
        assert!(negated.to_sparql().contains("!("));
    }

    #[test]
    fn test_filter_lang_matches() {
        let f = SparqlFilter::lang_matches("label", "en");
        assert!(f.to_sparql().contains("langMatches"));
    }

    #[test]
    fn test_filter_regex() {
        let f = SparqlFilter::regex("name", "^Alice");
        assert!(f.to_sparql().contains("regex"));
        assert!(f.to_sparql().contains("^Alice"));
    }

    // --- WhereClause --------------------------------------------------------

    #[test]
    fn test_where_clause_empty() {
        let wc = WhereClause::new();
        assert!(wc.is_empty());
        assert_eq!(wc.pattern_count(), 0);
    }

    #[test]
    fn test_where_clause_triple() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("o"),
        );
        assert!(!wc.is_empty());
        assert_eq!(wc.pattern_count(), 1);
    }

    #[test]
    fn test_where_clause_to_sparql_contains_triple() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("o"),
        );
        let rendered = wc.to_sparql();
        assert!(rendered.contains("?s rdf:type ?o ."));
    }

    #[test]
    fn test_where_clause_optional() {
        let inner = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("ex", "email"),
            SparqlTerm::var("email"),
        );
        let wc = WhereClause::new()
            .triple(
                SparqlTerm::var("s"),
                SparqlTerm::prefixed("rdf", "type"),
                SparqlTerm::prefixed("ex", "Person"),
            )
            .optional(inner);
        let rendered = wc.to_sparql();
        assert!(rendered.contains("OPTIONAL"));
    }

    #[test]
    fn test_where_clause_union() {
        let left = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("ex", "name"),
            SparqlTerm::var("name"),
        );
        let right = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("ex", "label"),
            SparqlTerm::var("name"),
        );
        let wc = WhereClause::new().union(left, right);
        let rendered = wc.to_sparql();
        assert!(rendered.contains("UNION"));
    }

    #[test]
    fn test_where_clause_filter() {
        let wc = WhereClause::new()
            .triple(
                SparqlTerm::var("s"),
                SparqlTerm::prefixed("ex", "age"),
                SparqlTerm::var("age"),
            )
            .filter(SparqlFilter::gt("age", 18.0));
        let rendered = wc.to_sparql();
        assert!(rendered.contains("FILTER"));
    }

    // --- SelectQuery --------------------------------------------------------

    #[test]
    fn test_select_query_build_basic() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("x"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("type"),
        );
        let query = SelectQuery::new()
            .select("x")
            .where_clause(wc)
            .build()
            .expect("should build successfully");

        assert!(query.contains("SELECT"));
        assert!(query.contains("?x"));
        assert!(query.contains("WHERE"));
        assert!(query.contains("rdf:type"));
    }

    #[test]
    fn test_select_query_build_distinct() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("x"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("t"),
        );
        let query = SelectQuery::new()
            .select("x")
            .distinct()
            .where_clause(wc)
            .build()
            .expect("should build successfully");

        assert!(query.contains("DISTINCT"));
    }

    #[test]
    fn test_select_query_build_limit_offset() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("x"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("t"),
        );
        let query = SelectQuery::new()
            .select("x")
            .where_clause(wc)
            .limit(10)
            .offset(5)
            .build()
            .expect("should build successfully");

        assert!(query.contains("LIMIT 10"));
        assert!(query.contains("OFFSET 5"));
    }

    #[test]
    fn test_select_query_build_order_by() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("x"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("t"),
        );
        let query = SelectQuery::new()
            .select("x")
            .where_clause(wc)
            .order_by_asc("x")
            .build()
            .expect("should build successfully");

        assert!(query.contains("ORDER BY ASC(?x)"));
    }

    #[test]
    fn test_select_query_empty_where_error() {
        let result = SelectQuery::new().select("x").build();
        assert!(matches!(result, Err(SparqlBuilderError::EmptyWhereClause)));
    }

    #[test]
    fn test_select_query_distinct_and_reduced_conflict() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("x"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::var("t"),
        );
        let result = SelectQuery::new()
            .select("x")
            .distinct()
            .reduced()
            .where_clause(wc)
            .build();
        assert!(matches!(
            result,
            Err(SparqlBuilderError::ConflictingModifiers(_))
        ));
    }

    #[test]
    fn test_select_query_with_prefix() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("x"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::prefixed("ex", "Person"),
        );
        let query = SelectQuery::new()
            .prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            .prefix("ex", "http://example.org/")
            .select("x")
            .where_clause(wc)
            .build()
            .expect("should build successfully");

        assert!(query.contains("PREFIX rdf:"));
        assert!(query.contains("PREFIX ex:"));
    }

    #[test]
    fn test_select_query_build_star() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::var("p"),
            SparqlTerm::var("o"),
        );
        let query = SelectQuery::new()
            .select_all()
            .where_clause(wc)
            .build()
            .expect("should build successfully");

        assert!(query.contains("SELECT *"));
    }

    // --- AskQuery -----------------------------------------------------------

    #[test]
    fn test_ask_query_build() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::prefixed("ex", "Person"),
        );
        let query = AskQuery::new().where_clause(wc).build();
        assert!(query.starts_with("ASK"));
    }

    #[test]
    fn test_ask_query_with_prefix() {
        let wc = WhereClause::new().triple(
            SparqlTerm::var("s"),
            SparqlTerm::prefixed("rdf", "type"),
            SparqlTerm::prefixed("ex", "Thing"),
        );
        let query = AskQuery::new()
            .prefix("ex", "http://example.org/")
            .where_clause(wc)
            .build();
        assert!(query.contains("PREFIX ex:"));
        assert!(query.contains("ASK"));
    }

    // --- validate_variable_name ---------------------------------------------

    #[test]
    fn test_validate_variable_name_valid() {
        assert!(validate_variable_name("x"));
        assert!(validate_variable_name("myVar"));
        assert!(validate_variable_name("_private"));
        assert!(validate_variable_name("foo_bar"));
    }

    #[test]
    fn test_validate_variable_name_invalid() {
        assert!(!validate_variable_name(""));
        assert!(!validate_variable_name("123abc"));
        assert!(!validate_variable_name("?x"));
    }

    // --- OrderDirection -----------------------------------------------------

    #[test]
    fn test_order_direction_asc() {
        assert_eq!(OrderDirection::Asc("x".to_string()).to_sparql(), "ASC(?x)");
    }

    #[test]
    fn test_order_direction_desc() {
        assert_eq!(
            OrderDirection::Desc("score".to_string()).to_sparql(),
            "DESC(?score)"
        );
    }

    // --- SparqlBuilderError Display -----------------------------------------

    #[test]
    fn test_error_display() {
        let e = SparqlBuilderError::EmptyWhereClause;
        let s = e.to_string();
        assert!(s.contains("empty"));

        let e2 = SparqlBuilderError::InvalidVariableName("123bad".to_string());
        let s2 = e2.to_string();
        assert!(s2.contains("123bad"));

        let e3 = SparqlBuilderError::ConflictingModifiers("test".to_string());
        let s3 = e3.to_string();
        assert!(s3.contains("test"));
    }
}
