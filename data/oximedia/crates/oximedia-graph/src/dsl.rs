//! Graph DSL parser for describing filter pipelines as text.
//!
//! This module provides a simple text-based domain-specific language for
//! defining media processing filter graphs.  The DSL is designed to be
//! human-readable and easy to produce programmatically.
//!
//! # Syntax
//!
//! A graph description is a sequence of **pipeline chains** separated by
//! semicolons (`;`) or newlines.  Each chain is a sequence of **node
//! specifications** connected by arrow tokens (`->`).
//!
//! ```text
//! source -> scale(1920,1080) -> encoder -> sink
//! source_audio -> normalize -> aac_encoder -> mux
//! ```
//!
//! A **node specification** has the form:
//!
//! ```text
//! name
//! name(arg1, arg2, …)
//! label:name
//! label:name(arg1, arg2, …)
//! ```
//!
//! Where:
//! - `name` is an ASCII identifier (letters, digits, underscores, hyphens).
//! - `label` is an optional unique alias used to identify the node when
//!   building the graph (useful when the same filter type is used more than
//!   once).
//! - Arguments (`arg1`, `arg2`, …) are positional string tokens passed to the
//!   filter constructor.  Quoted strings (`"…"`) preserve embedded spaces.
//!
//! ## Multi-branch graphs
//!
//! Fan-out and fan-in topologies require explicit node labels.  The same label
//! can appear in multiple chains to express shared nodes:
//!
//! ```text
//! source -> split
//! split -> scale(1280,720) -> sink_hd
//! split -> scale(640,360) -> sink_sd
//! ```
//!
//! ## Comments
//!
//! Lines that start with `#` (after optional whitespace) are ignored.
//!
//! # Example
//!
//! ```
//! use oximedia_graph::dsl::{parse_graph_dsl, GraphDescription};
//!
//! let input = "source -> scale(1920,1080) -> encoder -> sink";
//! let desc = parse_graph_dsl(input).expect("parse should succeed");
//!
//! // Four nodes and three edges
//! assert_eq!(desc.nodes.len(), 4);
//! assert_eq!(desc.edges.len(), 3);
//! ```

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::missing_errors_doc)]

use std::fmt;

use crate::error::{GraphError, GraphResult};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed description of a filter graph.
///
/// Produced by [`parse_graph_dsl`].
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDescription {
    /// Unique node specifications, deduplicated by their label.
    pub nodes: Vec<NodeSpec>,
    /// Directed edges between node labels.
    pub edges: Vec<EdgeSpec>,
}

impl GraphDescription {
    /// Returns `true` if the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns `true` if a node with the given label exists.
    #[must_use]
    pub fn contains_node(&self, label: &str) -> bool {
        self.nodes.iter().any(|n| n.label == label)
    }

    /// Look up a node by label.
    #[must_use]
    pub fn node(&self, label: &str) -> Option<&NodeSpec> {
        self.nodes.iter().find(|n| n.label == label)
    }
}

/// Specification of a single node in the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeSpec {
    /// Unique label used to identify this node in edges.
    ///
    /// If the DSL input did not provide an explicit `label:name` prefix the
    /// label is synthesised from the filter name and an auto-increment counter
    /// (e.g. `scale_0`, `scale_1`).
    pub label: String,
    /// Filter type name (the identifier before any argument list).
    pub filter: String,
    /// Positional arguments provided in the parenthesised argument list.
    pub args: Vec<String>,
}

impl NodeSpec {
    /// Create a node specification with no arguments.
    #[must_use]
    pub fn new(label: impl Into<String>, filter: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            filter: filter.into(),
            args: Vec::new(),
        }
    }

    /// Create a node specification with arguments.
    #[must_use]
    pub fn with_args(
        label: impl Into<String>,
        filter: impl Into<String>,
        args: Vec<String>,
    ) -> Self {
        Self {
            label: label.into(),
            filter: filter.into(),
            args,
        }
    }
}

impl fmt::Display for NodeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.label, self.filter)?;
        if !self.args.is_empty() {
            write!(f, "({})", self.args.join(", "))?;
        }
        Ok(())
    }
}

/// A directed edge from one node to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeSpec {
    /// Label of the source node.
    pub from: String,
    /// Label of the destination node.
    pub to: String,
}

impl EdgeSpec {
    /// Create a new edge specification.
    #[must_use]
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
        }
    }
}

impl fmt::Display for EdgeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.from, self.to)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse error
// ─────────────────────────────────────────────────────────────────────────────

/// A parse error with position information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Human-readable description of the problem.
    pub message: String,
    /// 1-based line number where the error occurred (if known).
    pub line: Option<usize>,
    /// 1-based column number where the error occurred (if known).
    pub column: Option<usize>,
}

impl ParseError {
    fn at(line: usize, column: usize, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            line: Some(line),
            column: Some(column),
        }
    }

    fn simple(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            line: None,
            column: None,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.column) {
            (Some(l), Some(c)) => write!(f, "parse error at {}:{}: {}", l, c, self.message),
            (Some(l), None) => write!(f, "parse error at line {}: {}", l, self.message),
            _ => write!(f, "parse error: {}", self.message),
        }
    }
}

impl From<ParseError> for GraphError {
    fn from(e: ParseError) -> Self {
        GraphError::ConfigurationError(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokeniser
// ─────────────────────────────────────────────────────────────────────────────

/// Token kinds produced by the tokeniser.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TokenKind {
    /// An identifier or bare argument value.
    Ident(String),
    /// A quoted string argument.
    Quoted(String),
    /// The `->` arrow connecting nodes.
    Arrow,
    /// `(` opening an argument list.
    LParen,
    /// `)` closing an argument list.
    RParen,
    /// `,` separating arguments.
    Comma,
    /// `:` separating a label from a filter name.
    Colon,
    /// `;` or newline — chain separator.
    ChainSep,
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    line: usize,
    col: usize,
}

/// Tokenise `input` into a flat `Vec<Token>`.
///
/// Lines starting with `#` (after optional leading whitespace) are treated as
/// comments and skipped entirely.
fn tokenise(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();
    let mut line = 1usize;
    let mut line_start = 0usize;

    while let Some(&(idx, ch)) = chars.peek() {
        let col = idx - line_start + 1;

        match ch {
            // Skip spaces and tabs.
            ' ' | '\t' | '\r' => {
                chars.next();
            }

            // Newline — chain separator (unless it's just whitespace between tokens).
            '\n' => {
                chars.next();
                // Emit ChainSep only when there are tokens already queued
                // (avoids leading separators from blank lines).
                if !tokens.is_empty() {
                    // Don't emit a duplicate ChainSep.
                    let last_is_sep = matches!(
                        tokens.last().map(|t: &Token| &t.kind),
                        Some(TokenKind::ChainSep)
                    );
                    if !last_is_sep {
                        tokens.push(Token {
                            kind: TokenKind::ChainSep,
                            line,
                            col,
                        });
                    }
                }
                line += 1;
                line_start = idx + 1;
            }

            // Comment — skip until end of line.
            '#' => {
                while let Some(&(_, c)) = chars.peek() {
                    if c == '\n' {
                        break;
                    }
                    chars.next();
                }
            }

            // `->`  arrow  OR  negative number literal (e.g. `-14`).
            '-' => {
                chars.next();
                match chars.peek() {
                    Some(&(_, '>')) => {
                        chars.next();
                        tokens.push(Token {
                            kind: TokenKind::Arrow,
                            line,
                            col,
                        });
                    }
                    // Negative integer/float argument: `-` followed by a digit.
                    Some(&(_, c)) if c.is_ascii_digit() => {
                        let mut num = String::from('-');
                        while let Some(&(_, c)) = chars.peek() {
                            if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
                                num.push(c);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        tokens.push(Token {
                            kind: TokenKind::Ident(num),
                            line,
                            col,
                        });
                    }
                    _ => {
                        return Err(ParseError::at(
                            line,
                            col,
                            "unexpected '-'; did you mean '->'?",
                        ));
                    }
                }
            }

            '(' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::LParen,
                    line,
                    col,
                });
            }
            ')' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::RParen,
                    line,
                    col,
                });
            }
            ',' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Comma,
                    line,
                    col,
                });
            }
            ':' => {
                chars.next();
                tokens.push(Token {
                    kind: TokenKind::Colon,
                    line,
                    col,
                });
            }
            ';' => {
                chars.next();
                let last_is_sep = matches!(
                    tokens.last().map(|t: &Token| &t.kind),
                    Some(TokenKind::ChainSep)
                );
                if !last_is_sep {
                    tokens.push(Token {
                        kind: TokenKind::ChainSep,
                        line,
                        col,
                    });
                }
            }

            // Quoted string.
            '"' => {
                chars.next();
                let mut s = String::new();
                let mut closed = false;
                while let Some(&(_, c)) = chars.peek() {
                    chars.next();
                    if c == '"' {
                        closed = true;
                        break;
                    }
                    if c == '\\' {
                        // Escape sequence — consume one more character.
                        if let Some(&(_, escaped)) = chars.peek() {
                            chars.next();
                            match escaped {
                                'n' => s.push('\n'),
                                't' => s.push('\t'),
                                '"' => s.push('"'),
                                '\\' => s.push('\\'),
                                other => {
                                    s.push('\\');
                                    s.push(other);
                                }
                            }
                        }
                    } else {
                        s.push(c);
                    }
                }
                if !closed {
                    return Err(ParseError::at(line, col, "unterminated string literal"));
                }
                tokens.push(Token {
                    kind: TokenKind::Quoted(s),
                    line,
                    col,
                });
            }

            // Identifier: letters, digits, '_', '-' (but not '-->' prefix).
            c if c.is_ascii_alphanumeric() || c == '_' => {
                let mut ident = String::new();
                while let Some(&(_, c)) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        // Avoid consuming the '-' of '->'.
                        if c == '-' {
                            // Peek two characters ahead.
                            let rest: String = {
                                let mut tmp = chars.clone();
                                tmp.next(); // consume '-'
                                tmp.peek().map(|&(_, x)| x).map_or(String::new(), |x| {
                                    let mut s = String::from('-');
                                    s.push(x);
                                    s
                                })
                            };
                            if rest == "->" {
                                break;
                            }
                        }
                        ident.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token {
                    kind: TokenKind::Ident(ident),
                    line,
                    col,
                });
            }

            // Unrecognised character.
            other => {
                return Err(ParseError::at(
                    line,
                    col,
                    format!("unexpected character '{other}'"),
                ));
            }
        }
    }

    Ok(tokens)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser
// ─────────────────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Counter used to generate unique labels when none are provided.
    counters: std::collections::HashMap<String, usize>,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            counters: std::collections::HashMap::new(),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next_token(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        self.pos += 1;
        t
    }

    /// Skip chain separators; returns `true` if any were consumed.
    fn skip_seps(&mut self) -> bool {
        let mut skipped = false;
        while matches!(self.peek().map(|t| &t.kind), Some(TokenKind::ChainSep)) {
            self.pos += 1;
            skipped = true;
        }
        skipped
    }

    /// Generate an auto-label for `filter_name`.
    fn auto_label(&mut self, filter_name: &str) -> String {
        let count = self.counters.entry(filter_name.to_owned()).or_insert(0);
        let label = format!("{}_{}", filter_name, count);
        *count += 1;
        label
    }

    /// Parse `label:filter_name(args…)` or just `filter_name(args…)`.
    fn parse_node_spec(&mut self) -> Result<NodeSpec, ParseError> {
        // Peek to decide if next pattern is `ident : ident` (label:name).
        let label_or_filter = match self.peek() {
            Some(Token {
                kind: TokenKind::Ident(s),
                ..
            }) => s.clone(),
            Some(t) => {
                return Err(ParseError::at(
                    t.line,
                    t.col,
                    format!("expected node name, found {:?}", t.kind),
                ));
            }
            None => {
                return Err(ParseError::simple("unexpected end of input in node spec"));
            }
        };
        self.pos += 1; // consume the first ident

        // Check if next token is `:` — if so this is `label : filter_name`.
        let (label, filter) = if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::Colon)) {
            self.pos += 1; // consume `:`
            let filter = match self.peek() {
                Some(Token {
                    kind: TokenKind::Ident(s),
                    ..
                }) => s.clone(),
                Some(t) => {
                    return Err(ParseError::at(
                        t.line,
                        t.col,
                        "expected filter name after ':'",
                    ));
                }
                None => {
                    return Err(ParseError::simple("expected filter name after ':'"));
                }
            };
            self.pos += 1; // consume filter ident
            (label_or_filter, filter)
        } else {
            // No label — derive one from filter name.
            let filter = label_or_filter.clone();
            let label = self.auto_label(&filter);
            (label, filter)
        };

        // Optionally parse `( arg, arg, … )`.
        let args = if matches!(self.peek().map(|t| &t.kind), Some(TokenKind::LParen)) {
            self.pos += 1; // consume `(`
            self.parse_args()?
        } else {
            Vec::new()
        };

        Ok(NodeSpec {
            label,
            filter,
            args,
        })
    }

    /// Parse a comma-separated argument list, consuming the closing `)`.
    fn parse_args(&mut self) -> Result<Vec<String>, ParseError> {
        let mut args = Vec::new();
        loop {
            // Clone the kind to avoid borrow-checker issues when we mutate pos.
            let kind_opt = self.peek().map(|t| t.kind.clone());
            match kind_opt {
                Some(TokenKind::RParen) => {
                    self.pos += 1; // consume `)`
                    break;
                }
                Some(TokenKind::Ident(s)) => {
                    args.push(s);
                    self.pos += 1;
                }
                Some(TokenKind::Quoted(s)) => {
                    args.push(s);
                    self.pos += 1;
                }
                Some(TokenKind::Comma) => {
                    self.pos += 1; // skip comma
                }
                _ => {
                    if let Some(t) = self.peek() {
                        return Err(ParseError::at(
                            t.line,
                            t.col,
                            format!("unexpected token in argument list: {:?}", t.kind),
                        ));
                    }
                    return Err(ParseError::simple("unterminated argument list"));
                }
            }
        }
        Ok(args)
    }

    /// Parse one pipeline chain: `node (-> node)*`.
    fn parse_chain(
        &mut self,
        nodes: &mut Vec<NodeSpec>,
        edges: &mut Vec<EdgeSpec>,
    ) -> Result<(), ParseError> {
        let first = self.parse_node_spec()?;
        let mut prev_label = first.label.clone();
        if !nodes.iter().any(|n| n.label == first.label) {
            nodes.push(first);
        }

        loop {
            // Clone the kind to avoid borrow conflicts when mutating self.
            let kind_opt = self.peek().map(|t| t.kind.clone());
            match kind_opt {
                Some(TokenKind::Arrow) => {
                    self.pos += 1; // consume `->`
                    let next = self.parse_node_spec()?;
                    let next_label = next.label.clone();
                    // Add edge.
                    edges.push(EdgeSpec::new(prev_label.clone(), next_label.clone()));
                    // Add node only if not already seen.
                    if !nodes.iter().any(|n| n.label == next.label) {
                        nodes.push(next);
                    }
                    prev_label = next_label;
                }
                // Chain ended.
                Some(TokenKind::ChainSep) | None => {
                    break;
                }
                _ => {
                    if let Some(t) = self.peek() {
                        return Err(ParseError::at(
                            t.line,
                            t.col,
                            format!("expected '->' or end of chain, found {:?}", t.kind),
                        ));
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    /// Parse the full input and return a [`GraphDescription`].
    fn parse(mut self) -> Result<GraphDescription, ParseError> {
        let mut nodes: Vec<NodeSpec> = Vec::new();
        let mut edges: Vec<EdgeSpec> = Vec::new();

        // Skip leading separators.
        self.skip_seps();

        while self.peek().is_some() {
            self.parse_chain(&mut nodes, &mut edges)?;
            self.skip_seps();
        }

        Ok(GraphDescription { nodes, edges })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a text-based graph DSL description into a [`GraphDescription`].
///
/// # Syntax
///
/// See the [module-level documentation][self] for a full description of the
/// supported syntax.
///
/// # Errors
///
/// Returns [`GraphError::ConfigurationError`] (wrapping a [`ParseError`]) if
/// the input is syntactically invalid.
///
/// # Example
///
/// ```
/// use oximedia_graph::dsl::parse_graph_dsl;
///
/// let dsl = "source -> scale(1920,1080) -> h264_encoder -> mp4_sink";
/// let desc = parse_graph_dsl(dsl).expect("parse should succeed");
/// assert_eq!(desc.nodes.len(), 4);
/// assert_eq!(desc.edges.len(), 3);
/// ```
pub fn parse_graph_dsl(input: &str) -> GraphResult<GraphDescription> {
    let tokens = tokenise(input).map_err(GraphError::from)?;
    let parser = Parser::new(tokens);
    parser.parse().map_err(GraphError::from)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic chain parsing ──────────────────────────────────────────────────

    #[test]
    fn test_parse_simple_chain() {
        let dsl = "source -> scale(1920,1080) -> encoder -> sink";
        let desc = parse_graph_dsl(dsl).expect("parse should succeed");
        assert_eq!(desc.nodes.len(), 4);
        assert_eq!(desc.edges.len(), 3);
    }

    #[test]
    fn test_parse_single_node() {
        let desc = parse_graph_dsl("source").expect("parse should succeed");
        assert_eq!(desc.nodes.len(), 1);
        assert_eq!(desc.edges.len(), 0);
        assert_eq!(desc.nodes[0].filter, "source");
    }

    #[test]
    fn test_parse_node_with_args() {
        let desc = parse_graph_dsl("scale(1280,720)").expect("parse should succeed");
        assert_eq!(desc.nodes[0].filter, "scale");
        assert_eq!(desc.nodes[0].args, vec!["1280", "720"]);
    }

    #[test]
    fn test_parse_explicit_label() {
        let desc = parse_graph_dsl("my_src:source -> sink").expect("parse should succeed");
        assert_eq!(desc.nodes[0].label, "my_src");
        assert_eq!(desc.nodes[0].filter, "source");
    }

    // ── edge correctness ─────────────────────────────────────────────────────

    #[test]
    fn test_edges_connect_sequential_nodes() {
        let desc = parse_graph_dsl("a -> b -> c").expect("parse should succeed");
        assert_eq!(desc.edges.len(), 2);
        assert_eq!(desc.edges[0].from, desc.nodes[0].label);
        assert_eq!(desc.edges[0].to, desc.nodes[1].label);
        assert_eq!(desc.edges[1].from, desc.nodes[1].label);
        assert_eq!(desc.edges[1].to, desc.nodes[2].label);
    }

    // ── multi-chain (newline-separated) ─────────────────────────────────────

    #[test]
    fn test_parse_multiline_chains() {
        let dsl = "src -> filter_a\nfilter_b -> sink";
        let desc = parse_graph_dsl(dsl).expect("parse should succeed");
        // 4 distinct nodes, 2 edges.
        assert_eq!(desc.nodes.len(), 4);
        assert_eq!(desc.edges.len(), 2);
    }

    #[test]
    fn test_parse_semicolon_separated_chains() {
        let dsl = "src -> enc; src2 -> enc2";
        let desc = parse_graph_dsl(dsl).expect("parse should succeed");
        assert_eq!(desc.nodes.len(), 4);
        assert_eq!(desc.edges.len(), 2);
    }

    // ── shared node (fan-out) ────────────────────────────────────────────────

    #[test]
    fn test_shared_node_deduplication() {
        let dsl = "tee:split\ntee -> branch_a\ntee -> branch_b";
        let desc = parse_graph_dsl(dsl).expect("parse should succeed");
        // "tee" should appear only once in nodes.
        let tee_count = desc.nodes.iter().filter(|n| n.label == "tee").count();
        assert_eq!(tee_count, 1, "shared node must be deduplicated");
    }

    // ── comments ────────────────────────────────────────────────────────────

    #[test]
    fn test_comments_are_ignored() {
        let dsl = "# this is a comment\nsource -> sink\n# another comment";
        let desc = parse_graph_dsl(dsl).expect("parse should succeed");
        assert_eq!(desc.nodes.len(), 2);
    }

    // ── empty/whitespace input ───────────────────────────────────────────────

    #[test]
    fn test_empty_input() {
        let desc = parse_graph_dsl("").expect("parse should succeed");
        assert!(desc.is_empty());
    }

    #[test]
    fn test_whitespace_only_input() {
        let desc = parse_graph_dsl("   \n  \t  ").expect("parse should succeed");
        assert!(desc.is_empty());
    }

    // ── quoted arguments ─────────────────────────────────────────────────────

    #[test]
    fn test_quoted_args_preserve_spaces() {
        let desc =
            parse_graph_dsl(r#"watermark("hello world",50,50)"#).expect("parse should succeed");
        assert_eq!(desc.nodes[0].args[0], "hello world");
    }

    // ── GraphDescription helpers ─────────────────────────────────────────────

    #[test]
    fn test_contains_node() {
        let desc = parse_graph_dsl("src -> sink").expect("parse should succeed");
        // Auto-labels are src_0, sink_0.
        assert!(desc.contains_node("src_0"));
        assert!(!desc.contains_node("nonexistent"));
    }

    #[test]
    fn test_node_lookup() {
        let desc = parse_graph_dsl("my:scale(1920,1080)").expect("parse should succeed");
        let node = desc.node("my").expect("node should exist");
        assert_eq!(node.filter, "scale");
        assert_eq!(node.args, vec!["1920", "1080"]);
    }

    // ── display ──────────────────────────────────────────────────────────────

    #[test]
    fn test_node_spec_display_no_args() {
        let n = NodeSpec::new("my_src", "source");
        assert_eq!(n.to_string(), "my_src:source");
    }

    #[test]
    fn test_node_spec_display_with_args() {
        let n = NodeSpec::with_args("s0", "scale", vec!["1920".into(), "1080".into()]);
        assert_eq!(n.to_string(), "s0:scale(1920, 1080)");
    }

    #[test]
    fn test_edge_spec_display() {
        let e = EdgeSpec::new("src", "sink");
        assert_eq!(e.to_string(), "src -> sink");
    }

    // ── error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_unterminated_string_returns_error() {
        let result = parse_graph_dsl(r#"node("unterminated)"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_unexpected_char_returns_error() {
        let result = parse_graph_dsl("node @ other");
        assert!(result.is_err());
    }

    #[test]
    fn test_bare_arrow_returns_error() {
        let result = parse_graph_dsl("-> sink");
        assert!(result.is_err());
    }

    // ── complex pipeline ─────────────────────────────────────────────────────

    #[test]
    fn test_complex_pipeline() {
        let dsl = r#"
            # Full transcode pipeline
            input:source -> deinterlace -> normalize(loudness,-14)
            input -> scale(1920,1080) -> h264:encoder(crf,23) -> mux:mp4_sink
        "#;
        let desc = parse_graph_dsl(dsl).expect("parse should succeed");
        // Nodes: input, deinterlace_0, normalize_0, scale_0, h264, mux
        assert!(desc.nodes.len() >= 4);
        assert!(desc.edges.len() >= 3);
        // input shared across two chains.
        let input_count = desc.nodes.iter().filter(|n| n.label == "input").count();
        assert_eq!(input_count, 1, "shared 'input' node must be deduplicated");
    }

    #[test]
    fn test_auto_label_uniqueness() {
        // Two anonymous scale nodes should get different auto-labels.
        let dsl = "src -> scale(1920,1080)\nsrc2 -> scale(640,360)";
        let desc = parse_graph_dsl(dsl).expect("parse should succeed");
        let scale_labels: Vec<&str> = desc
            .nodes
            .iter()
            .filter(|n| n.filter == "scale")
            .map(|n| n.label.as_str())
            .collect();
        assert_eq!(scale_labels.len(), 2);
        assert_ne!(
            scale_labels[0], scale_labels[1],
            "auto-labels must be unique"
        );
    }
}
