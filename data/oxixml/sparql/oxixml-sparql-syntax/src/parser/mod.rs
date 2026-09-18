//! Hand-written recursive-descent parser over the full SPARQL 1.1 grammar.
//!
//! The parser directly builds the [`crate::algebra`] representation while parsing, following the
//! translation rules of [SPARQL 1.1 §18.2](https://www.w3.org/TR/sparql11-query/#sparqlQuery).

mod build;
mod expr;
mod pattern;
mod query;
#[cfg(feature = "sparql-12")]
mod star;
mod terminal;
mod update;

use crate::algebra::AggregateExpression;
use crate::error::SparqlSyntaxError;
use crate::query::Query;
use crate::term::{BlankNode, NamedNode, Variable};
use crate::update::Update;
use oxixml_iri::Iri;
use oxixml_model::OxString;
use std::collections::{HashMap, HashSet};

/// Internal parser error carrying a byte position and a fatality flag.
///
/// Fatal errors are semantic validation failures that must abort parsing; non-fatal errors are
/// recoverable and let ordered-choice alternatives be tried.
#[derive(Clone)]
pub(crate) struct PErr {
    pub(crate) pos: usize,
    pub(crate) msg: &'static str,
    pub(crate) fatal: bool,
}

/// Internal parser result type.
pub(crate) type R<T> = Result<T, PErr>;

/// Keeps whichever error reached the furthest position (better diagnostics).
pub(crate) fn pick_furthest(current: Option<PErr>, new: PErr) -> Option<PErr> {
    match current {
        Some(c) if c.pos >= new.pos => Some(c),
        _ => Some(new),
    }
}

/// Ordered-choice combinator with fatal-error short-circuit and furthest-error tracking.
macro_rules! alt {
    ($self:ident, $($m:expr),+ $(,)?) => {{
        let __start = $self.pos;
        let mut __best: Option<$crate::parser::PErr> = None;
        'alt: {
            $(
                match $m {
                    Ok(v) => break 'alt Ok(v),
                    Err(e) => {
                        if e.fatal { break 'alt Err(e); }
                        __best = $crate::parser::pick_furthest(__best, e);
                        $self.pos = __start;
                    }
                }
            )+
            Err(__best.unwrap_or_else(|| $self.mk_perr("no matching alternative")))
        }
    }};
}
pub(crate) use alt;

/// The parser state for a single SPARQL document.
pub(crate) struct Parser<'a> {
    pub(crate) input: &'a str,
    pub(crate) pos: usize,
    pub(crate) base_iri: Option<Iri<OxString>>,
    pub(crate) prefixes: HashMap<String, String>,
    /// IRIs registered as custom aggregate functions, see
    /// [`SparqlParser::with_custom_aggregate_function`](crate::SparqlParser::with_custom_aggregate_function).
    pub(crate) custom_aggregate_functions: HashSet<NamedNode>,
    pub(crate) used_bnodes: HashSet<String>,
    pub(crate) currently_used_bnodes: HashSet<String>,
    pub(crate) aggregates: Vec<Vec<(Variable, AggregateExpression)>>,
    pub(crate) bnode_counter: u64,
    pub(crate) var_counter: u64,
    /// Prefix shared by every parser-generated blank node label and variable name.
    ///
    /// It is a run of `g` longer than any run of `g` in the parsed text, so no identifier written
    /// by the user can ever start with it, let alone be equal to a generated one. See
    /// [`Parser::fresh_prefix`].
    pub(crate) fresh_prefix: String,
    pub(crate) furthest: usize,
    /// Set while the argument of an aggregate is parsed, so nested aggregates can be rejected.
    pub(crate) in_aggregate: bool,
    /// Cleared while an expression that may not contain aggregates is parsed (`FILTER`, `BIND`
    /// and `GROUP BY`), see [SPARQL 1.1 §18.2.4.1](https://www.w3.org/TR/sparql11-query/#convertGroupAggSelectExpressions).
    pub(crate) aggregates_allowed: bool,
    /// Current nesting depth of the recursive-descent productions, see [`Parser::enter_recursion`].
    pub(crate) depth: u32,
}

/// Maximum nesting depth of parenthesized expressions, group graph patterns, property paths and
/// collections.
///
/// The parser is a plain recursive-descent one, so an unbounded nesting depth would let a short
/// adversarial query (a few kilobytes of `(((…)))` or `{{{…}}}`) overflow the stack and abort the
/// process. Any nesting past this ceiling is reported as a regular syntax error instead.
///
/// The value is chosen so that the deepest production chain (the built-in call ordered choice,
/// whose frame is by far the largest) stays well inside a 2 MiB thread stack in a debug build,
/// while remaining far above the nesting depth of any real-world query.
pub(crate) const MAX_RECURSION_DEPTH: u32 = 48;

/// Applies the [codepoint escape sequences](https://www.w3.org/TR/sparql12-query/#codepointEscape)
/// pre-processing step of SPARQL 1.2 §19.2.
///
/// `\uXXXX` and `\UXXXXXXXX` are replaced by the codepoint they denote *before* tokenizing, and
/// anywhere in the document. The scan is a plain left-to-right one, so `"\\u0041"` yields the
/// (invalid) string escape `\A` rather than the six characters `A`, which is exactly what the
/// specification prescribes.
#[cfg(feature = "sparql-12")]
fn decode_codepoint_escapes(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains('\\') {
        return std::borrow::Cow::Borrowed(text);
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut last_copied = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            i += 1;
            continue;
        }
        let (prefix, width) = match bytes.get(i + 1) {
            Some(b'u') => (2, 4),
            Some(b'U') => (2, 8),
            _ => {
                i += 1;
                continue;
            }
        };
        let start = i + prefix;
        let Some(digits) = text.get(start..start + width) else {
            i += 1;
            continue;
        };
        let Some(decoded) = u32::from_str_radix(digits, 16)
            .ok()
            .and_then(char::from_u32)
        else {
            i += 1;
            continue;
        };
        out.push_str(&text[last_copied..i]);
        out.push(decoded);
        i = start + width;
        last_copied = i;
    }
    if last_copied == 0 {
        return std::borrow::Cow::Borrowed(text);
    }
    out.push_str(&text[last_copied..]);
    std::borrow::Cow::Owned(out)
}

/// Parses a SPARQL query.
pub(crate) fn parse_query(
    text: &str,
    base_iri: Option<Iri<OxString>>,
    prefixes: HashMap<String, String>,
    custom_aggregate_functions: HashSet<NamedNode>,
) -> Result<Query, SparqlSyntaxError> {
    #[cfg(feature = "sparql-12")]
    let decoded = decode_codepoint_escapes(text);
    #[cfg(feature = "sparql-12")]
    let text = &*decoded;
    let mut parser = Parser::new(text, base_iri, prefixes, custom_aggregate_functions);
    match parser.query_unit() {
        Ok(query) => {
            parser.ws();
            if parser.pos == text.len() {
                Ok(query)
            } else {
                Err(SparqlSyntaxError::from_offsets(
                    "Unexpected trailing characters after the query",
                    text,
                    parser.pos..text.len(),
                ))
            }
        }
        Err(e) => Err(parser.into_error(e)),
    }
}

/// Parses a SPARQL update.
pub(crate) fn parse_update(
    text: &str,
    base_iri: Option<Iri<OxString>>,
    prefixes: HashMap<String, String>,
    custom_aggregate_functions: HashSet<NamedNode>,
) -> Result<Update, SparqlSyntaxError> {
    #[cfg(feature = "sparql-12")]
    let decoded = decode_codepoint_escapes(text);
    #[cfg(feature = "sparql-12")]
    let text = &*decoded;
    let mut parser = Parser::new(text, base_iri, prefixes, custom_aggregate_functions);
    match parser.update_unit() {
        Ok(operations) => {
            parser.ws();
            if parser.pos == text.len() {
                Ok(Update {
                    base_iri: parser.base_iri.clone(),
                    operations,
                })
            } else {
                Err(SparqlSyntaxError::from_offsets(
                    "Unexpected trailing characters after the update",
                    text,
                    parser.pos..text.len(),
                ))
            }
        }
        Err(e) => Err(parser.into_error(e)),
    }
}

impl<'a> Parser<'a> {
    fn new(
        input: &'a str,
        base_iri: Option<Iri<OxString>>,
        prefixes: HashMap<String, String>,
        custom_aggregate_functions: HashSet<NamedNode>,
    ) -> Self {
        Self {
            input,
            pos: 0,
            base_iri,
            prefixes,
            custom_aggregate_functions,
            used_bnodes: HashSet::new(),
            currently_used_bnodes: HashSet::new(),
            aggregates: Vec::new(),
            bnode_counter: 0,
            var_counter: 0,
            fresh_prefix: Self::fresh_prefix(input),
            furthest: 0,
            in_aggregate: false,
            aggregates_allowed: true,
            depth: 0,
        }
    }

    /// Computes a prefix that cannot occur anywhere in `input`.
    ///
    /// The prefix is a run of `g` one longer than the longest run of `g` of the parsed text, so it
    /// is not a substring of `input`; consequently no blank node label and no variable name
    /// written by the user starts with it. Generated names stay deterministic for a given input,
    /// which keeps `parse(x).to_string()` stable.
    fn fresh_prefix(input: &str) -> String {
        /// Above this length the deterministic prefix is replaced by a random one, so that a
        /// pathological input cannot make every generated name grow with the input size.
        const MAX_LEN: usize = 16;

        let mut longest = 0_usize;
        let mut current = 0_usize;
        for byte in input.bytes() {
            if byte == b'g' {
                current += 1;
                longest = longest.max(current);
                if longest >= MAX_LEN {
                    // A run of `g` that long never happens in real SPARQL: fall back to a unique
                    // (hexadecimal, hence `g`-free) prefix.
                    return BlankNode::default().as_str().to_owned();
                }
            } else {
                current = 0;
            }
        }
        "g".repeat(longest + 1)
    }

    fn into_error(self, e: PErr) -> SparqlSyntaxError {
        let pos = e.pos.max(self.furthest);
        SparqlSyntaxError::from_offsets(e.msg, self.input, pos..self.input.len().min(pos + 1))
    }

    // --- error constructors -------------------------------------------------

    pub(crate) fn mk_perr(&mut self, msg: &'static str) -> PErr {
        self.furthest = self.furthest.max(self.pos);
        PErr {
            pos: self.pos,
            msg,
            fatal: false,
        }
    }

    pub(crate) fn mk_err<T>(&mut self, msg: &'static str) -> R<T> {
        Err(self.mk_perr(msg))
    }

    pub(crate) fn fatal<T>(&mut self, msg: &'static str) -> R<T> {
        self.furthest = self.furthest.max(self.pos);
        Err(PErr {
            pos: self.pos,
            msg,
            fatal: true,
        })
    }

    // --- cursor helpers -----------------------------------------------------

    #[inline]
    pub(crate) fn rest(&self) -> &'a str {
        &self.input[self.pos..]
    }

    #[inline]
    pub(crate) fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    /// Consumes whitespace and comments (the grammar's `_` production).
    pub(crate) fn ws(&mut self) {
        loop {
            let bytes = self.input.as_bytes();
            let mut moved = false;
            while self.pos < bytes.len() {
                match bytes[self.pos] {
                    b' ' | b'\t' | b'\n' | b'\r' => {
                        self.pos += 1;
                        moved = true;
                    }
                    _ => break,
                }
            }
            if self.pos < bytes.len() && bytes[self.pos] == b'#' {
                self.pos += 1;
                while self.pos < bytes.len() && bytes[self.pos] != b'\r' && bytes[self.pos] != b'\n'
                {
                    self.pos += 1;
                }
                moved = true;
            }
            if !moved {
                break;
            }
        }
    }

    // --- literal / keyword matching ----------------------------------------

    /// Matches a case-sensitive literal (used for punctuation and lowercase tokens).
    pub(crate) fn eat(&mut self, s: &str) -> R<()> {
        if self.rest().as_bytes().starts_with(s.as_bytes()) {
            self.pos += s.len();
            Ok(())
        } else {
            self.mk_err("expected token")
        }
    }

    /// Optionally matches a case-sensitive literal, returning whether it matched.
    pub(crate) fn try_str(&mut self, s: &str) -> bool {
        if self.rest().as_bytes().starts_with(s.as_bytes()) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    /// Matches a case-insensitive keyword of exactly `s.len()` bytes.
    ///
    /// SPARQL tokenizes with the longest-token rule, so a keyword only matches when it is not
    /// immediately followed by a name character: `GROUPBY` is a single (invalid) token and never
    /// `GROUP` followed by `BY`, and `VALUES:x` is the prefixed name of the prefix `VALUES`.
    pub(crate) fn kw(&mut self, s: &str) -> R<()> {
        if self.try_kw(s) {
            Ok(())
        } else {
            self.mk_err("expected keyword")
        }
    }

    /// Optionally matches a case-insensitive keyword, honouring the longest-token rule.
    pub(crate) fn try_kw(&mut self, s: &str) -> bool {
        let rest = self.rest().as_bytes();
        if rest.len() < s.len() || !rest[..s.len()].eq_ignore_ascii_case(s.as_bytes()) {
            return false;
        }
        let continues_name = matches!(
            self.input[self.pos + s.len()..].chars().next(),
            Some(c) if terminal::is_pn_chars(c) || c == ':'
        );
        if continues_name {
            return false;
        }
        self.pos += s.len();
        true
    }

    // --- fresh identifiers (deterministic per parse) ------------------------

    /// A fresh blank node for `[]`, collections and path expansion.
    ///
    /// The label is built from [`Parser::fresh_prefix`], which cannot be the start of any label
    /// written in the parsed text: a generated blank node is therefore never merged with a
    /// user-written one such as `_:g0`.
    pub(crate) fn fresh_bnode(&mut self) -> BlankNode {
        let id = self.bnode_counter;
        self.bnode_counter += 1;
        BlankNode::new_unchecked(format!("{}b{id}", self.fresh_prefix))
    }

    /// A fresh internal variable for aggregates and grouping.
    ///
    /// As for [`Parser::fresh_bnode`], the name cannot collide with a variable of the parsed
    /// text, so a query using `?0` or `?v0` keeps its own bindings.
    pub(crate) fn fresh_var(&mut self) -> Variable {
        let id = self.var_counter;
        self.var_counter += 1;
        Variable::new_unchecked(format!("{}v{id}", self.fresh_prefix))
    }

    // --- recursion guard ----------------------------------------------------

    /// Enters a nested production, failing when [`MAX_RECURSION_DEPTH`] is exceeded.
    ///
    /// Every successful call must be paired with a [`Parser::leave_recursion`] call.
    pub(crate) fn enter_recursion(&mut self) -> R<()> {
        if self.depth >= MAX_RECURSION_DEPTH {
            return self.fatal("The query is nested too deeply");
        }
        self.depth += 1;
        Ok(())
    }

    /// Leaves a nested production entered through [`Parser::enter_recursion`].
    pub(crate) fn leave_recursion(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Leaves `n` nested productions entered through [`Parser::enter_recursion`].
    ///
    /// Used by the flat-chain productions (`conditional_or_expression`, `path_sequence`,
    /// the `UNION` loop, …): a chain of `n + 1` operator-joined elements is folded into a
    /// left-deep tree `n` levels deep, and every fold level is charged with one
    /// [`Parser::enter_recursion`] call *before* the operand that grows the tree by that level is
    /// parsed — never after the whole chain has already been built — so an over-long chain is
    /// rejected before the oversized tree it would have produced is ever constructed (that tree's
    /// `Drop`/`Display`/`fmt_sse` are just as natively recursive as this parser's own call stack,
    /// so a value that is never built is never dropped either). Charging one unit per fold before
    /// parsing the next operand, rather than checking the total once after collecting every
    /// operand, also keeps the guard compositional: while a chain's charge is still live, an
    /// operand that is itself a long chain (or genuinely nested) is checked against the combined
    /// depth, not against a freshly reset counter.
    ///
    /// All `n` charges are released together once the whole chain is folded, so a chain's own
    /// depth never leaks into whatever the caller parses next.
    pub(crate) fn leave_recursion_by(&mut self, n: u32) {
        self.depth = self.depth.saturating_sub(n);
    }
}
