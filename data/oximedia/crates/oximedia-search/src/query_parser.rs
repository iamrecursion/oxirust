//! Query parser — tokenises user search strings into structured query tokens.
#![allow(dead_code)]

// ── QueryField ────────────────────────────────────────────────────────────────

/// A named field that may appear in a structured query (`field:value`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryField {
    /// The media title.
    Title,
    /// The media description.
    Description,
    /// Tags / keywords.
    Tag,
    /// Audio codec name.
    Codec,
    /// Video resolution (e.g. `1920x1080`).
    Resolution,
    /// Duration expressed in seconds.
    Duration,
    /// Free-form / catch-all field.
    Any,
}

impl QueryField {
    /// Canonical field name as it appears in a search string.
    #[must_use]
    pub fn field_name(&self) -> &'static str {
        match self {
            QueryField::Title => "title",
            QueryField::Description => "description",
            QueryField::Tag => "tag",
            QueryField::Codec => "codec",
            QueryField::Resolution => "resolution",
            QueryField::Duration => "duration",
            QueryField::Any => "*",
        }
    }

    /// Parse a field name string (case-insensitive) into a `QueryField`.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "title" => QueryField::Title,
            "description" | "desc" => QueryField::Description,
            "tag" | "tags" => QueryField::Tag,
            "codec" => QueryField::Codec,
            "resolution" | "res" => QueryField::Resolution,
            "duration" | "dur" => QueryField::Duration,
            _ => QueryField::Any,
        }
    }
}

// ── QueryToken ────────────────────────────────────────────────────────────────

/// A single token produced by the query parser.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryToken {
    /// A plain search term.
    Term(String),
    /// A phrase (multi-word, quoted).
    Phrase(String),
    /// A field-restricted term, e.g. `title:hello`.
    FieldTerm {
        /// The target field.
        field: QueryField,
        /// The value to match.
        value: String,
    },
    /// A range constraint, e.g. `duration:[60 TO 120]`.
    Range {
        /// The target field.
        field: QueryField,
        /// Lower bound (inclusive), or `None` for open.
        from: Option<f64>,
        /// Upper bound (inclusive), or `None` for open.
        to: Option<f64>,
    },
    /// Boolean AND operator.
    And,
    /// Boolean OR operator.
    Or,
    /// Boolean NOT / negation operator.
    Not,
    /// Wildcard suffix term, e.g. `vid*`.
    Wildcard(String),
}

// ── QueryParser ───────────────────────────────────────────────────────────────

/// A simple query parser that converts a search string into a `Vec<QueryToken>`.
///
/// Supported syntax:
/// - `word` → `Term`
/// - `"multi word"` → `Phrase`
/// - `field:value` → `FieldTerm`
/// - `field:[from TO to]` → `Range`
/// - `AND`, `OR`, `NOT` (case-insensitive) → boolean operators
/// - `word*` → `Wildcard`
#[derive(Debug, Default)]
pub struct QueryParser {
    /// When true, unknown fields fall through as plain Terms.
    pub lenient: bool,
}

impl QueryParser {
    /// Create a new strict parser.
    #[must_use]
    pub fn new() -> Self {
        Self { lenient: false }
    }

    /// Create a new lenient parser that never rejects unknown field names.
    #[must_use]
    pub fn lenient() -> Self {
        Self { lenient: true }
    }

    /// Parse `input` into a vector of `QueryToken`s.
    #[must_use]
    pub fn parse(&self, input: &str) -> Vec<QueryToken> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();

        while let Some(&ch) = chars.peek() {
            match ch {
                // Skip whitespace
                ' ' | '\t' | '\n' => {
                    chars.next();
                }
                // Quoted phrase
                '"' => {
                    chars.next(); // consume opening quote
                    let mut phrase = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == '"' {
                            chars.next();
                            break;
                        }
                        phrase.push(c);
                        chars.next();
                    }
                    if !phrase.is_empty() {
                        tokens.push(QueryToken::Phrase(phrase));
                    }
                }
                // Range bracket: field:[from TO to]
                // (handled when we encounter '[' after "field:")
                _ => {
                    // Collect a raw token until whitespace or quote
                    let mut raw = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == ' ' || c == '\t' || c == '\n' || c == '"' {
                            break;
                        }
                        raw.push(c);
                        chars.next();
                    }
                    if raw.is_empty() {
                        continue;
                    }

                    // Boolean keywords
                    match raw.to_ascii_uppercase().as_str() {
                        "AND" => {
                            tokens.push(QueryToken::And);
                            continue;
                        }
                        "OR" => {
                            tokens.push(QueryToken::Or);
                            continue;
                        }
                        "NOT" => {
                            tokens.push(QueryToken::Not);
                            continue;
                        }
                        _ => {}
                    }

                    // field:[from TO to]
                    if let Some(colon_pos) = raw.find(':') {
                        let field_str = &raw[..colon_pos];
                        let rest = &raw[colon_pos + 1..];
                        let field = QueryField::from_name(field_str);

                        if rest.starts_with('[') {
                            // Range token — collect until ']'
                            let mut range_raw = rest.to_string();
                            if !range_raw.contains(']') {
                                // Continue reading from chars
                                while let Some(&c) = chars.peek() {
                                    range_raw.push(c);
                                    chars.next();
                                    if c == ']' {
                                        break;
                                    }
                                }
                            }
                            // Parse [from TO to]
                            let inner = range_raw.trim_start_matches('[').trim_end_matches(']');
                            let parts: Vec<&str> = inner.split_ascii_whitespace().collect();
                            let from = parts.first().and_then(|s| s.parse::<f64>().ok());
                            let to = parts.last().and_then(|s| s.parse::<f64>().ok());
                            tokens.push(QueryToken::Range { field, from, to });
                            continue;
                        }

                        // Plain field:value
                        if rest.ends_with('*') {
                            let value = rest.trim_end_matches('*').to_string();
                            tokens.push(QueryToken::Wildcard(format!(
                                "{}:{}*",
                                field.field_name(),
                                value
                            )));
                        } else {
                            tokens.push(QueryToken::FieldTerm {
                                field,
                                value: rest.to_string(),
                            });
                        }
                        continue;
                    }

                    // Wildcard suffix
                    if raw.ends_with('*') {
                        tokens.push(QueryToken::Wildcard(raw));
                        continue;
                    }

                    // Plain term
                    tokens.push(QueryToken::Term(raw));
                }
            }
        }

        tokens
    }

    /// Return the number of boolean operators in a parsed token list.
    #[must_use]
    pub fn count_operators(tokens: &[QueryToken]) -> usize {
        tokens
            .iter()
            .filter(|t| matches!(t, QueryToken::And | QueryToken::Or | QueryToken::Not))
            .count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_term() {
        let p = QueryParser::new();
        let tokens = p.parse("hello");
        assert_eq!(tokens, vec![QueryToken::Term("hello".into())]);
    }

    #[test]
    fn test_parse_multiple_terms() {
        let p = QueryParser::new();
        let tokens = p.parse("foo bar baz");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], QueryToken::Term("foo".into()));
    }

    #[test]
    fn test_parse_phrase() {
        let p = QueryParser::new();
        let tokens = p.parse("\"hello world\"");
        assert_eq!(tokens, vec![QueryToken::Phrase("hello world".into())]);
    }

    #[test]
    fn test_parse_field_term() {
        let p = QueryParser::new();
        let tokens = p.parse("title:sunset");
        assert_eq!(
            tokens,
            vec![QueryToken::FieldTerm {
                field: QueryField::Title,
                value: "sunset".into(),
            }]
        );
    }

    #[test]
    fn test_parse_and_operator() {
        let p = QueryParser::new();
        let tokens = p.parse("foo AND bar");
        assert!(tokens.contains(&QueryToken::And));
    }

    #[test]
    fn test_parse_or_operator() {
        let p = QueryParser::new();
        let tokens = p.parse("cat OR dog");
        assert!(tokens.contains(&QueryToken::Or));
    }

    #[test]
    fn test_parse_not_operator() {
        let p = QueryParser::new();
        let tokens = p.parse("NOT spam");
        assert_eq!(tokens[0], QueryToken::Not);
    }

    #[test]
    fn test_parse_wildcard() {
        let p = QueryParser::new();
        let tokens = p.parse("vid*");
        assert_eq!(tokens, vec![QueryToken::Wildcard("vid*".into())]);
    }

    #[test]
    fn test_parse_range() {
        let p = QueryParser::new();
        let tokens = p.parse("duration:[60 TO 120]");
        assert!(matches!(
            tokens[0],
            QueryToken::Range {
                field: QueryField::Duration,
                ..
            }
        ));
    }

    #[test]
    fn test_query_field_field_name() {
        assert_eq!(QueryField::Title.field_name(), "title");
        assert_eq!(QueryField::Tag.field_name(), "tag");
        assert_eq!(QueryField::Any.field_name(), "*");
    }

    #[test]
    fn test_query_field_from_name_case_insensitive() {
        assert_eq!(QueryField::from_name("TITLE"), QueryField::Title);
        assert_eq!(QueryField::from_name("desc"), QueryField::Description);
        assert_eq!(QueryField::from_name("unknown"), QueryField::Any);
    }

    #[test]
    fn test_count_operators() {
        let p = QueryParser::new();
        let tokens = p.parse("a AND b OR c NOT d");
        let count = QueryParser::count_operators(&tokens);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_parse_empty_string() {
        let p = QueryParser::new();
        let tokens = p.parse("");
        assert!(tokens.is_empty());
    }
}
