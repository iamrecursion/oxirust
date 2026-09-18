//! The tokenizer of the [RELAX NG compact syntax][rnc], §2.
//!
//! The five stages the specification describes are all here: BOM stripping,
//! newline normalization, `\x{…}` escape interpretation, and then the
//! longest-match tokenization of §2.5. Comments (`#`) and documentation lines
//! (`##`) are dropped by the lexer: documentation translates to an element in the
//! compatibility-annotations namespace, which simplification rule 4.1 removes
//! again, so discarding it here changes nothing that a validator can observe.
//!
//! [rnc]: https://relaxng.org/compact-20021121.html

use crate::error::{Position, SchemaError, syntax};

/// One token of the compact syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Token {
    /// An `identifier`: an NCName that is not a keyword, or `\` plus an NCName.
    Ident(String),
    /// One of the twenty keywords.
    Keyword(String),
    /// A `CName`: `prefix:local`.
    CName(String, String),
    /// An `nsName`: `prefix:*`.
    NsName(String),
    /// A `literal`, with `~` concatenation already applied.
    Literal(String),
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `?`
    Question,
    /// `,`
    Comma,
    /// `|`
    Pipe,
    /// `&`
    Amp,
    /// `-`
    Minus,
    /// `=`
    Equal,
    /// `|=`
    PipeEqual,
    /// `&=`
    AmpEqual,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `>>`
    Follow,
    /// The end of the input.
    Eof,
}

impl Token {
    /// A human-readable spelling, for diagnostics.
    pub(super) fn describe(&self) -> String {
        match self {
            Self::Ident(name) => format!("identifier `{name}`"),
            Self::Keyword(name) => format!("keyword `{name}`"),
            Self::CName(prefix, local) => format!("name `{prefix}:{local}`"),
            Self::NsName(prefix) => format!("name `{prefix}:*`"),
            Self::Literal(_) => "a literal".to_owned(),
            Self::Star => "`*`".to_owned(),
            Self::Plus => "`+`".to_owned(),
            Self::Question => "`?`".to_owned(),
            Self::Comma => "`,`".to_owned(),
            Self::Pipe => "`|`".to_owned(),
            Self::Amp => "`&`".to_owned(),
            Self::Minus => "`-`".to_owned(),
            Self::Equal => "`=`".to_owned(),
            Self::PipeEqual => "`|=`".to_owned(),
            Self::AmpEqual => "`&=`".to_owned(),
            Self::LBrace => "`{`".to_owned(),
            Self::RBrace => "`}`".to_owned(),
            Self::LParen => "`(`".to_owned(),
            Self::RParen => "`)`".to_owned(),
            Self::LBracket => "`[`".to_owned(),
            Self::RBracket => "`]`".to_owned(),
            Self::Follow => "`>>`".to_owned(),
            Self::Eof => "the end of the schema".to_owned(),
        }
    }
}

/// The twenty keywords of the compact syntax.
pub(super) const KEYWORDS: [&str; 19] = [
    "attribute",
    "default",
    "datatypes",
    "div",
    "element",
    "empty",
    "external",
    "grammar",
    "include",
    "inherit",
    "list",
    "mixed",
    "namespace",
    "notAllowed",
    "parent",
    "start",
    "string",
    "text",
    "token",
];

/// A source character, tagged with whether it is a *newline marker*.
///
/// §2.3 turns every line break into a newline marker, and §2.4 says that
/// `\x{A}` produces "a character, as for all other escape sequences, not a
/// newline marker". The distinction is observable: a literal may not contain a
/// newline marker, but it may contain a `#xA` character that came from an
/// escape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ch {
    value: char,
    newline_marker: bool,
}

/// Turns the text of a compact-syntax schema into tokens.
pub(super) fn tokenize(source: &str) -> Result<Vec<(Token, Position)>, SchemaError> {
    let normalized = normalize(source);
    let chars = interpret_escapes(&normalized)?;
    Lexer {
        chars,
        index: 0,
        line: 1,
        column: 1,
    }
    .run()
}

/// Strips a leading byte-order mark and normalizes newlines to newline markers.
fn normalize(source: &str) -> Vec<Ch> {
    let source = source.strip_prefix('\u{feff}').unwrap_or(source);
    let mut out = Vec::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                let _ = chars.next();
            }
            out.push(Ch {
                value: '\n',
                newline_marker: true,
            });
        } else if c == '\n' {
            out.push(Ch {
                value: '\n',
                newline_marker: true,
            });
        } else {
            out.push(Ch {
                value: c,
                newline_marker: false,
            });
        }
    }
    out
}

/// Replaces every `\x{…}` escape sequence by the character it denotes.
fn interpret_escapes(chars: &[Ch]) -> Result<Vec<Ch>, SchemaError> {
    if !chars.iter().any(|c| c.value == '\\') {
        return Ok(chars.to_vec());
    }
    let mut out = Vec::with_capacity(chars.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index].value != '\\' {
            out.push(chars[index]);
            index += 1;
            continue;
        }
        // escapeOpen ::= "\" xs "{"
        let mut cursor = index + 1;
        let mut xs = 0;
        while cursor < chars.len() && chars[cursor].value == 'x' {
            xs += 1;
            cursor += 1;
        }
        if xs == 0 || cursor >= chars.len() || chars[cursor].value != '{' {
            // Not an escape sequence: this is the `\NCName` identifier escape.
            out.push(chars[index]);
            index += 1;
            continue;
        }
        cursor += 1;
        let start = cursor;
        let mut value: u32 = 0;
        while cursor < chars.len() && chars[cursor].value != '}' {
            let digit = chars[cursor]
                .value
                .to_digit(16)
                .ok_or_else(|| syntax("a `\\x{…}` escape must contain hexadecimal digits"))?;
            value = value.saturating_mul(16).saturating_add(digit);
            cursor += 1;
        }
        if cursor >= chars.len() || cursor == start {
            return Err(syntax("an unterminated or empty `\\x{…}` escape sequence"));
        }
        let replacement = char::from_u32(value)
            .filter(|c| oxixml_dom::is_xml_char(*c))
            .ok_or_else(|| {
                syntax(format!(
                    "the escape `\\x{{{value:X}}}` does not denote an XML character"
                ))
            })?;
        out.push(Ch {
            value: replacement,
            newline_marker: false,
        });
        index = cursor + 1;
    }
    Ok(out)
}

struct Lexer {
    chars: Vec<Ch>,
    index: usize,
    line: u32,
    column: u32,
}

impl Lexer {
    fn position(&self) -> Position {
        Position::new(self.line, self.column)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).map(|c| c.value)
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.index + offset).map(|c| c.value)
    }

    fn at_newline_marker(&self) -> bool {
        self.chars.get(self.index).is_some_and(|c| c.newline_marker)
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.index).copied()?;
        self.index += 1;
        if c.newline_marker {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c.value)
    }

    fn starts_with(&self, text: &str) -> bool {
        text.chars()
            .enumerate()
            .all(|(offset, c)| self.peek_at(offset) == Some(c))
    }

    fn eat(&mut self, text: &str) -> bool {
        if !self.starts_with(text) {
            return false;
        }
        for _ in text.chars() {
            let _ = self.bump();
        }
        true
    }

    fn run(mut self) -> Result<Vec<(Token, Position)>, SchemaError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_separators();
            let position = self.position();
            let Some(c) = self.peek() else {
                tokens.push((Token::Eof, position));
                return Ok(tokens);
            };
            let token = match c {
                '"' | '\'' => Token::Literal(self.literal()?),
                '*' => {
                    let _ = self.bump();
                    Token::Star
                }
                '+' => {
                    let _ = self.bump();
                    Token::Plus
                }
                '?' => {
                    let _ = self.bump();
                    Token::Question
                }
                ',' => {
                    let _ = self.bump();
                    Token::Comma
                }
                '-' => {
                    let _ = self.bump();
                    Token::Minus
                }
                '{' => {
                    let _ = self.bump();
                    Token::LBrace
                }
                '}' => {
                    let _ = self.bump();
                    Token::RBrace
                }
                '(' => {
                    let _ = self.bump();
                    Token::LParen
                }
                ')' => {
                    let _ = self.bump();
                    Token::RParen
                }
                '[' => {
                    let _ = self.bump();
                    Token::LBracket
                }
                ']' => {
                    let _ = self.bump();
                    Token::RBracket
                }
                '=' => {
                    let _ = self.bump();
                    Token::Equal
                }
                '|' => {
                    let _ = self.bump();
                    if self.eat("=") {
                        Token::PipeEqual
                    } else {
                        Token::Pipe
                    }
                }
                '&' => {
                    let _ = self.bump();
                    if self.eat("=") {
                        Token::AmpEqual
                    } else {
                        Token::Amp
                    }
                }
                '>' => {
                    if self.eat(">>") {
                        Token::Follow
                    } else {
                        return Err(syntax("`>` must be part of `>>`").at(position));
                    }
                }
                '\\' => {
                    let _ = self.bump();
                    let name = self.ncname();
                    if name.is_empty() {
                        return Err(syntax("`\\` must be followed by a name").at(position));
                    }
                    Token::Ident(name)
                }
                c if is_name_start(c) => self.name_token()?,
                other => {
                    return Err(syntax(format!("`{other}` cannot start a token")).at(position));
                }
            };
            tokens.push((token, position));
        }
    }

    /// `separator ::= whitespace | "#" [^#] restOfLine | "#"`, plus the
    /// documentation lines that are dropped.
    fn skip_separators(&mut self) {
        loop {
            match self.peek() {
                // `separator ::= [#x9 #xA #x20 &newline;]`. A `#xD` that
                // survives escape interpretation is *not* a separator.
                Some(' ' | '\t' | '\n') => {
                    let _ = self.bump();
                }
                Some('#') => {
                    while self.peek().is_some() {
                        if self.peek() == Some('\n') {
                            break;
                        }
                        let _ = self.bump();
                    }
                }
                _ => return,
            }
        }
    }

    fn ncname(&mut self) -> String {
        let mut name = String::new();
        if self.peek().is_some_and(is_name_start) {
            if let Some(c) = self.bump() {
                name.push(c);
            }
        } else {
            return name;
        }
        while self.peek().is_some_and(is_name_char) {
            if let Some(c) = self.bump() {
                name.push(c);
            }
        }
        name
    }

    fn name_token(&mut self) -> Result<Token, SchemaError> {
        let name = self.ncname();
        if self.peek() == Some(':') {
            if self.peek_at(1) == Some('*') {
                let _ = self.bump();
                let _ = self.bump();
                return Ok(Token::NsName(name));
            }
            if self.peek_at(1).is_some_and(is_name_start) {
                let _ = self.bump();
                let local = self.ncname();
                return Ok(Token::CName(name, local));
            }
        }
        if KEYWORDS.contains(&name.as_str()) {
            Ok(Token::Keyword(name))
        } else {
            Ok(Token::Ident(name))
        }
    }

    /// `literal ::= literalSegment | literalSegment "~" literal`
    fn literal(&mut self) -> Result<String, SchemaError> {
        let mut value = self.literal_segment()?;
        loop {
            let mark = (self.index, self.line, self.column);
            self.skip_separators();
            if self.peek() != Some('~') {
                self.index = mark.0;
                self.line = mark.1;
                self.column = mark.2;
                return Ok(value);
            }
            let _ = self.bump();
            self.skip_separators();
            value.push_str(&self.literal_segment()?);
        }
    }

    fn literal_segment(&mut self) -> Result<String, SchemaError> {
        let position = self.position();
        for (open, triple) in [("\"\"\"", true), ("'''", true), ("\"", false), ("'", false)] {
            if !self.eat(open) {
                continue;
            }
            let mut value = String::new();
            loop {
                if self.eat(open) {
                    return Ok(value);
                }
                let is_newline_marker = self.at_newline_marker();
                let Some(c) = self.bump() else {
                    return Err(syntax("an unterminated literal").at(position));
                };
                if !triple && is_newline_marker {
                    return Err(
                        syntax("a single-quoted literal must not contain a line break")
                            .at(position),
                    );
                }
                value.push(c);
            }
        }
        Err(syntax("expected a literal").at(position))
    }
}

/// The start of an `NCName`, as [`crate::name`] defines it.
fn is_name_start(c: char) -> bool {
    crate::name::is_ncname_start_char(c)
}

/// The continuation of an `NCName`, as [`crate::name`] defines it.
fn is_name_char(c: char) -> bool {
    crate::name::is_ncname_char(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(source: &str) -> Vec<Token> {
        tokenize(source)
            .expect("a lexable schema")
            .into_iter()
            .map(|(token, _)| token)
            .collect()
    }

    #[test]
    fn keywords_and_identifiers_are_distinguished() {
        assert_eq!(
            tokens("element foo \\element"),
            [
                Token::Keyword("element".to_owned()),
                Token::Ident("foo".to_owned()),
                Token::Ident("element".to_owned()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn colon_forms() {
        assert_eq!(
            tokens("xsd:int p:*"),
            [
                Token::CName("xsd".to_owned(), "int".to_owned()),
                Token::NsName("p".to_owned()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn literals_concatenate_and_nest_quotes() {
        assert_eq!(
            tokens(r#" "a" ~ "b" "#),
            [Token::Literal("ab".to_owned()), Token::Eof]
        );
        assert_eq!(
            tokens("\"\"\"a\"b\"\"\""),
            [Token::Literal("a\"b".to_owned()), Token::Eof]
        );
    }

    #[test]
    fn comments_and_documentation_are_dropped() {
        assert_eq!(
            tokens("# comment\n## documentation\nempty"),
            [Token::Keyword("empty".to_owned()), Token::Eof]
        );
    }

    #[test]
    fn escapes_are_interpreted_before_tokenizing() {
        assert_eq!(
            tokens("\\x{66}oo"),
            [Token::Ident("foo".to_owned()), Token::Eof]
        );
        // The replacement is not rescanned for further escapes.
        assert_eq!(
            tokens("\"\\x{5C}x{5C}\""),
            [Token::Literal("\\x{5C}".to_owned()), Token::Eof]
        );
    }

    #[test]
    fn combining_operators() {
        assert_eq!(
            tokens("a |= b &= c >> d"),
            [
                Token::Ident("a".to_owned()),
                Token::PipeEqual,
                Token::Ident("b".to_owned()),
                Token::AmpEqual,
                Token::Ident("c".to_owned()),
                Token::Follow,
                Token::Ident("d".to_owned()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn bad_input_is_rejected() {
        assert!(tokenize("\"unterminated").is_err());
        assert!(tokenize("\\x{}").is_err());
        assert!(tokenize("%").is_err());
    }
}
