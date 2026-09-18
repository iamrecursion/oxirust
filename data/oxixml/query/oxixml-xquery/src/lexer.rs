//! The XQuery 3.1 tokenizer.
//!
//! Like XPath, XQuery has no reserved words: `element`, `for`, `switch` and `div` are ordinary
//! names wherever a name is expected. The lexer therefore recognizes no keywords at all — it
//! produces names, literals and punctuation, and the parser decides what a name means from its
//! position.
//!
//! Two things distinguish this lexer from the XPath one:
//!
//! * a string literal resolves the five predefined entity references and character references,
//!   which XPath's does not. [`Token::Str`] records whether any were present, because a literal
//!   that used one cannot be handed back to the XPath compiler verbatim;
//! * the scanner is *resettable*. XQuery's grammar needs unbounded lookahead — `element` may
//!   begin a constructor, a kind test or a function call, and `<` may begin a comparison or a
//!   direct element constructor — so the parser saves a position, tries an alternative, and
//!   rewinds when it fails. [`Lexer::position`] and [`Lexer::seek`] are what make that cheap.

use crate::error::{XQueryError, XQueryResult};

/// A lexical token of the XQuery 3.1 grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Token {
    /// An `NCName` or a lexical `QName` (`prefix:local`).
    Name(String),
    /// `Q{uri}local` — a `URIQualifiedName`.
    BracedName {
        /// The namespace URI between the braces.
        uri: String,
        /// The local part.
        local: String,
    },
    /// `Q{uri}*` — a wildcard in an explicitly named namespace.
    BracedWildcard {
        /// The namespace URI between the braces.
        uri: String,
    },
    /// An `IntegerLiteral`, in its source spelling.
    Integer(String),
    /// A `DecimalLiteral`, in its source spelling.
    Decimal(String),
    /// A `DoubleLiteral`, in its source spelling.
    Double(String),
    /// A `StringLiteral`. The value has doubled quotation marks, entity references and character
    /// references resolved; `expanded` is true when a reference was resolved, which means the
    /// source spelling is not valid XPath.
    Str {
        /// The literal's value.
        value: String,
        /// Whether an entity or character reference was resolved.
        expanded: bool,
    },
    /// `$`
    Dollar,
    /// `@`
    At,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `%`
    Percent,
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `[`
    LeftBracket,
    /// `]`
    RightBracket,
    /// `{`
    LeftBrace,
    /// `}`
    RightBrace,
    /// `(#`
    PragmaOpen,
    /// `#)`
    PragmaClose,
    /// `/`
    Slash,
    /// `//`
    DoubleSlash,
    /// `::`
    ColonColon,
    /// `:`
    Colon,
    /// `:=`
    Assign,
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `?`
    Question,
    /// `!`
    Bang,
    /// `#`
    Hash,
    /// `|`
    Pipe,
    /// `.`
    Dot,
    /// `..`
    DotDot,
    /// `=`
    Equal,
    /// `!=`
    NotEqual,
    /// `<`
    Less,
    /// `<=`
    LessEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterEqual,
    /// `<<`
    Precedes,
    /// `>>`
    Follows,
    /// `||`
    Concat,
    /// `=>`
    Arrow,
    /// `` ` `` — the start of a string constructor.
    Backtick,
    /// The end of the query.
    Eof,
}

impl Token {
    /// A short rendering used in `XPST0003` diagnostics.
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Name(name) => format!("the name `{name}`"),
            Self::BracedName { uri, local } => format!("the name `Q{{{uri}}}{local}`"),
            Self::BracedWildcard { uri } => format!("the wildcard `Q{{{uri}}}*`"),
            Self::Integer(text) | Self::Decimal(text) | Self::Double(text) => {
                format!("the number `{text}`")
            }
            Self::Str { value, .. } => format!("the string `{value}`"),
            Self::Eof => "the end of the query".to_owned(),
            other => format!("`{}`", other.spelling()),
        }
    }

    /// The punctuation's source spelling, for diagnostics.
    fn spelling(&self) -> &'static str {
        match self {
            Self::Dollar => "$",
            Self::At => "@",
            Self::Comma => ",",
            Self::Semicolon => ";",
            Self::Percent => "%",
            Self::LeftParen => "(",
            Self::RightParen => ")",
            Self::LeftBracket => "[",
            Self::RightBracket => "]",
            Self::LeftBrace => "{",
            Self::RightBrace => "}",
            Self::PragmaOpen => "(#",
            Self::PragmaClose => "#)",
            Self::Slash => "/",
            Self::DoubleSlash => "//",
            Self::ColonColon => "::",
            Self::Colon => ":",
            Self::Assign => ":=",
            Self::Star => "*",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Question => "?",
            Self::Bang => "!",
            Self::Hash => "#",
            Self::Pipe => "|",
            Self::Dot => ".",
            Self::DotDot => "..",
            Self::Equal => "=",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Precedes => "<<",
            Self::Follows => ">>",
            Self::Concat => "||",
            Self::Arrow => "=>",
            Self::Backtick => "`",
            _ => "a token",
        }
    }
}

/// A token together with the byte range it occupies in the query source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Spanned {
    pub(crate) token: Token,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// Whether `c` may start an XML `Name`.
#[must_use]
pub(crate) fn is_name_start(c: char) -> bool {
    c == '_'
        || c.is_ascii_alphabetic()
        || matches!(c as u32,
            0xC0..=0xD6 | 0xD8..=0xF6 | 0xF8..=0x2FF | 0x370..=0x37D | 0x37F..=0x1FFF
            | 0x200C..=0x200D | 0x2070..=0x218F | 0x2C00..=0x2FEF | 0x3001..=0xD7FF
            | 0xF900..=0xFDCF | 0xFDF0..=0xFFFD | 0x10000..=0xEFFFF)
}

/// Whether `c` may continue an XML `Name`.
#[must_use]
pub(crate) fn is_name_char(c: char) -> bool {
    is_name_start(c)
        || c == '-'
        || c == '.'
        || c.is_ascii_digit()
        || matches!(c as u32, 0xB7 | 0x300..=0x36F | 0x203F..=0x2040)
}

/// The scanner the parser drives.
#[derive(Debug, Clone)]
pub(crate) struct Lexer<'a> {
    source: &'a str,
    position: usize,
}

impl<'a> Lexer<'a> {
    /// A scanner positioned at the start of `source`.
    pub(crate) const fn new(source: &'a str) -> Self {
        Self {
            source,
            position: 0,
        }
    }

    /// The whole query text.
    pub(crate) const fn source(&self) -> &'a str {
        self.source
    }

    /// The byte offset the next token will be scanned from.
    pub(crate) const fn position(&self) -> usize {
        self.position
    }

    /// Moves the scanner to `position`, which is how the parser backtracks.
    pub(crate) const fn seek(&mut self, position: usize) {
        self.position = position;
    }

    fn rest(&self) -> &'a str {
        self.source.get(self.position..).unwrap_or_default()
    }

    fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self, c: char) {
        self.position += c.len_utf8();
    }

    /// Skips whitespace and `(: nested (: comments :) :)`.
    ///
    /// # Errors
    ///
    /// `XPST0003` when a comment is never closed.
    pub(crate) fn skip_trivia(&mut self) -> XQueryResult<()> {
        loop {
            match self.peek_char() {
                Some(c) if c.is_whitespace() => self.bump(c),
                Some('(') if self.rest().starts_with("(:") && !self.rest().starts_with("(:#") => {
                    self.skip_comment()?;
                }
                _ => return Ok(()),
            }
        }
    }

    fn skip_comment(&mut self) -> XQueryResult<()> {
        let start = self.position;
        let mut depth = 0_usize;
        loop {
            let rest = self.rest();
            if rest.starts_with("(:") {
                depth += 1;
                self.position += 2;
            } else if rest.starts_with(":)") {
                depth -= 1;
                self.position += 2;
                if depth == 0 {
                    return Ok(());
                }
            } else {
                match self.peek_char() {
                    Some(c) => self.bump(c),
                    None => {
                        return Err(XQueryError::syntax("the comment is never closed", start));
                    }
                }
            }
        }
    }

    /// Scans the next token.
    ///
    /// # Errors
    ///
    /// `XPST0003` for any lexical error.
    pub(crate) fn next_token(&mut self) -> XQueryResult<Spanned> {
        self.skip_trivia()?;
        let start = self.position;
        let Some(c) = self.peek_char() else {
            return Ok(Spanned {
                token: Token::Eof,
                start,
                end: start,
            });
        };
        let token = match c {
            '"' | '\'' => self.scan_string(c)?,
            '0'..='9' => self.scan_number()?,
            '.' => self.scan_dot()?,
            'Q' if self.rest().starts_with("Q{") => self.scan_braced_name()?,
            c if is_name_start(c) => self.scan_name(),
            _ => self.scan_punctuation(c)?,
        };
        Ok(Spanned {
            token,
            start,
            end: self.position,
        })
    }

    fn scan_dot(&mut self) -> XQueryResult<Token> {
        let rest = self.rest();
        if rest.starts_with("..") {
            self.position += 2;
            return Ok(Token::DotDot);
        }
        let follows_digit = rest
            .get(1..)
            .and_then(|tail| tail.chars().next())
            .is_some_and(|c| c.is_ascii_digit());
        if follows_digit {
            return self.scan_number();
        }
        self.position += 1;
        Ok(Token::Dot)
    }

    fn scan_name(&mut self) -> Token {
        let start = self.position;
        while let Some(c) = self.peek_char() {
            if is_name_char(c) {
                self.bump(c);
            } else {
                break;
            }
        }
        // `prefix:local` is one token, but `prefix:*`, `prefix::` and `prefix:=` are not.
        let saved = self.position;
        if self.peek_char() == Some(':') {
            self.position += 1;
            let follows = self.peek_char();
            if follows == Some(':') || follows == Some('*') || follows == Some('=') {
                self.position = saved;
            } else if follows.is_some_and(is_name_start) {
                while let Some(c) = self.peek_char() {
                    if is_name_char(c) {
                        self.bump(c);
                    } else {
                        break;
                    }
                }
            } else {
                self.position = saved;
            }
        }
        Token::Name(
            self.source
                .get(start..self.position)
                .unwrap_or_default()
                .to_owned(),
        )
    }

    /// `BracedURILiteral ::= "Q" "{" (PredefinedEntityRef | CharRef | [^&{}])* "}"`.
    ///
    /// The references are expanded, and the resulting URI is whitespace-normalized as the
    /// `xs:anyURI` whitespace facet prescribes — collapsed, not merely trimmed.
    fn scan_braced_name(&mut self) -> XQueryResult<Token> {
        let start = self.position;
        self.position += 2;
        let mut raw = String::new();
        loop {
            match self.peek_char() {
                Some('}') => break,
                Some('{') => {
                    return Err(XQueryError::syntax(
                        "a braced URI literal may not contain `{`",
                        start,
                    ));
                }
                Some('&') => {
                    let at = self.position;
                    let rest = self.rest().get(1..).unwrap_or_default();
                    let Some(relative) = rest.find(';') else {
                        return Err(XQueryError::syntax(
                            "the entity or character reference is never closed",
                            at,
                        ));
                    };
                    let body = rest.get(..relative).unwrap_or_default().to_owned();
                    raw.push_str(&resolve_reference(&body, at)?);
                    self.position += 1 + relative + 1;
                }
                Some(c) => {
                    raw.push(c);
                    self.bump(c);
                }
                None => {
                    return Err(XQueryError::syntax(
                        "the braced URI literal is never closed",
                        start,
                    ));
                }
            }
        }
        let uri = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        self.position += 1;
        match self.peek_char() {
            Some('*') => {
                self.position += 1;
                Ok(Token::BracedWildcard { uri })
            }
            Some(c) if is_name_start(c) => {
                let local_start = self.position;
                while let Some(c) = self.peek_char() {
                    if is_name_char(c) {
                        self.bump(c);
                    } else {
                        break;
                    }
                }
                let local = self
                    .source
                    .get(local_start..self.position)
                    .unwrap_or_default()
                    .to_owned();
                Ok(Token::BracedName { uri, local })
            }
            _ => Err(XQueryError::syntax(
                "a braced URI literal must be followed by a local name or `*`",
                start,
            )),
        }
    }

    fn scan_number(&mut self) -> XQueryResult<Token> {
        let start = self.position;
        let mut seen_dot = false;
        while let Some(c) = self.peek_char() {
            match c {
                '0'..='9' => self.bump(c),
                '.' if !seen_dot => {
                    seen_dot = true;
                    self.bump(c);
                }
                _ => break,
            }
        }
        let mantissa_end = self.position;
        if matches!(self.peek_char(), Some('e' | 'E')) {
            let saved = self.position;
            self.position += 1;
            if matches!(self.peek_char(), Some('+' | '-')) {
                self.position += 1;
            }
            if self.peek_char().is_some_and(|c| c.is_ascii_digit()) {
                while let Some(c) = self.peek_char() {
                    if c.is_ascii_digit() {
                        self.bump(c);
                    } else {
                        break;
                    }
                }
                let text = self.source.get(start..self.position).unwrap_or_default();
                return Ok(Token::Double(text.to_owned()));
            }
            self.position = saved;
        }
        let text = self.source.get(start..mantissa_end).unwrap_or_default();
        if seen_dot {
            Ok(Token::Decimal(text.to_owned()))
        } else {
            Ok(Token::Integer(text.to_owned()))
        }
    }

    fn scan_string(&mut self, quote: char) -> XQueryResult<Token> {
        let start = self.position;
        self.position += quote.len_utf8();
        let mut value = String::new();
        let mut expanded = false;
        loop {
            let Some(c) = self.peek_char() else {
                return Err(XQueryError::syntax(
                    "the string literal is never closed",
                    start,
                ));
            };
            if c == quote {
                self.position += quote.len_utf8();
                if self.peek_char() == Some(quote) {
                    value.push(quote);
                    self.position += quote.len_utf8();
                    continue;
                }
                return Ok(Token::Str { value, expanded });
            }
            if c == '&' {
                let resolved = self.scan_reference()?;
                value.push_str(&resolved);
                expanded = true;
                continue;
            }
            value.push(c);
            self.bump(c);
        }
    }

    /// Resolves one `&…;` reference. Shared with the direct-constructor scanner.
    ///
    /// # Errors
    ///
    /// `XPST0003` for an unknown entity, `XQST0090` for a character reference that does not
    /// denote a legal XML character.
    pub(crate) fn scan_reference(&mut self) -> XQueryResult<String> {
        let start = self.position;
        self.position += 1;
        let body_start = self.position;
        loop {
            match self.peek_char() {
                Some(';') => break,
                Some(c) if c != '&' && c != '<' && !c.is_whitespace() => self.bump(c),
                _ => {
                    return Err(XQueryError::syntax(
                        "the entity or character reference is never closed",
                        start,
                    ));
                }
            }
        }
        let body = self
            .source
            .get(body_start..self.position)
            .unwrap_or_default();
        self.position += 1;
        resolve_reference(body, start)
    }

    fn scan_punctuation(&mut self, c: char) -> XQueryResult<Token> {
        let rest = self.rest();
        for (spelling, token) in [
            ("(#", Token::PragmaOpen),
            ("#)", Token::PragmaClose),
            ("//", Token::DoubleSlash),
            ("::", Token::ColonColon),
            (":=", Token::Assign),
            ("!=", Token::NotEqual),
            ("<<", Token::Precedes),
            (">>", Token::Follows),
            ("<=", Token::LessEqual),
            (">=", Token::GreaterEqual),
            ("||", Token::Concat),
            ("=>", Token::Arrow),
        ] {
            if rest.starts_with(spelling) {
                self.position += spelling.len();
                return Ok(token);
            }
        }
        let single = match c {
            '$' => Token::Dollar,
            '@' => Token::At,
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            '%' => Token::Percent,
            '(' => Token::LeftParen,
            ')' => Token::RightParen,
            '[' => Token::LeftBracket,
            ']' => Token::RightBracket,
            '{' => Token::LeftBrace,
            '}' => Token::RightBrace,
            '/' => Token::Slash,
            ':' => Token::Colon,
            '*' => Token::Star,
            '+' => Token::Plus,
            '-' => Token::Minus,
            '?' => Token::Question,
            '!' => Token::Bang,
            '#' => Token::Hash,
            '|' => Token::Pipe,
            '=' => Token::Equal,
            '<' => Token::Less,
            '>' => Token::Greater,
            '`' => Token::Backtick,
            _ => {
                return Err(XQueryError::syntax(
                    format!("unexpected character `{c}`"),
                    self.position,
                ));
            }
        };
        self.bump(c);
        Ok(single)
    }
}

/// Resolves the body of a `&…;` reference — `lt`, `#38`, `#x2603` and the rest.
///
/// # Errors
///
/// `XQST0090` when a character reference is out of range, `XPST0003` for an unknown entity name.
pub(crate) fn resolve_reference(body: &str, offset: usize) -> XQueryResult<String> {
    match body {
        "lt" => return Ok("<".to_owned()),
        "gt" => return Ok(">".to_owned()),
        "amp" => return Ok("&".to_owned()),
        "quot" => return Ok("\"".to_owned()),
        "apos" => return Ok("'".to_owned()),
        _ => {}
    }
    let Some(digits) = body.strip_prefix('#') else {
        return Err(XQueryError::syntax(
            format!("`&{body};` is not a predefined entity reference"),
            offset,
        ));
    };
    // The *grammar* of a character reference is `&#[0-9]+;` or `&#x[0-9a-fA-F]+;`; anything else
    // — an empty, mistyped or uppercase-`X` reference — is a syntax error, `XPST0003`. Only a
    // well-formed reference that names a character XML forbids is `XQST0090`.
    let code = match digits.strip_prefix('x') {
        Some(hex) if !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()) => {
            u32::from_str_radix(hex, 16).ok()
        }
        Some(_) => None,
        None if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) => {
            digits.parse::<u32>().ok()
        }
        None => {
            return Err(XQueryError::syntax(
                format!("`&{body};` is not a well-formed character reference"),
                offset,
            ));
        }
    };
    let invalid = || {
        XQueryError::new(
            crate::error::XQueryErrorCode::Xqst0090,
            format!("`&{body};` does not denote a legal XML character"),
        )
        .at(offset)
    };
    let Some(code) = code else {
        // A reference whose digits do not fit in a code point is out of range, not mistyped.
        if digits
            .strip_prefix('x')
            .is_some_and(|hex| !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()))
            || digits.chars().all(|c| c.is_ascii_digit()) && !digits.is_empty()
        {
            return Err(invalid());
        }
        return Err(XQueryError::syntax(
            format!("`&{body};` is not a well-formed character reference"),
            offset,
        ));
    };
    if !is_xml_char(code) {
        return Err(invalid());
    }
    char::from_u32(code)
        .map(|c| c.to_string())
        .ok_or_else(invalid)
}

/// The four characters of XML 1.0's `S` production, which is what "white space" means throughout
/// the XQuery grammar. Narrower than Rust's Unicode `White_Space`, so text that begins with a
/// no-break space begins with content.
pub(crate) const XML_SPACE: [char; 4] = [' ', '\t', '\r', '\n'];

/// Whether `code` is a legal XML 1.0 character.
#[must_use]
pub(crate) const fn is_xml_char(code: u32) -> bool {
    matches!(code, 0x9 | 0xA | 0xD | 0x20..=0xD7FF | 0xE000..=0xFFFD | 0x10000..=0x10FFFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(source: &str) -> XQueryResult<Vec<Token>> {
        let mut lexer = Lexer::new(source);
        let mut out = Vec::new();
        loop {
            let spanned = lexer.next_token()?;
            if spanned.token == Token::Eof {
                return Ok(out);
            }
            out.push(spanned.token);
        }
    }

    #[test]
    fn names_absorb_hyphens_but_not_axis_separators() -> XQueryResult<()> {
        assert_eq!(tokens("x-1")?, vec![Token::Name("x-1".to_owned())]);
        assert_eq!(
            tokens("x - 1")?,
            vec![
                Token::Name("x".to_owned()),
                Token::Minus,
                Token::Integer("1".to_owned())
            ]
        );
        assert_eq!(
            tokens("child::p")?,
            vec![
                Token::Name("child".to_owned()),
                Token::ColonColon,
                Token::Name("p".to_owned())
            ]
        );
        assert_eq!(
            tokens("p:*")?,
            vec![Token::Name("p".to_owned()), Token::Colon, Token::Star]
        );
        assert_eq!(tokens("p:local")?, vec![Token::Name("p:local".to_owned())]);
        Ok(())
    }

    #[test]
    fn numbers_follow_the_three_literal_productions() -> XQueryResult<()> {
        assert_eq!(tokens("42")?, vec![Token::Integer("42".to_owned())]);
        assert_eq!(tokens("4.2")?, vec![Token::Decimal("4.2".to_owned())]);
        assert_eq!(tokens(".5")?, vec![Token::Decimal(".5".to_owned())]);
        assert_eq!(tokens("4e2")?, vec![Token::Double("4e2".to_owned())]);
        assert_eq!(tokens("4.2E-3")?, vec![Token::Double("4.2E-3".to_owned())]);
        // `1e` is an integer followed by a name, not a malformed double.
        assert_eq!(
            tokens("1e")?,
            vec![Token::Integer("1".to_owned()), Token::Name("e".to_owned())]
        );
        Ok(())
    }

    #[test]
    fn string_literals_resolve_references() -> XQueryResult<()> {
        assert_eq!(
            tokens(r#""a&amp;b""#)?,
            vec![Token::Str {
                value: "a&b".to_owned(),
                expanded: true
            }]
        );
        assert_eq!(
            tokens(r#""he said ""hi""""#)?,
            vec![Token::Str {
                value: "he said \"hi\"".to_owned(),
                expanded: false
            }]
        );
        assert_eq!(
            tokens("'&#x41;'")?,
            vec![Token::Str {
                value: "A".to_owned(),
                expanded: true
            }]
        );
        Ok(())
    }

    #[test]
    fn comments_nest_and_are_skipped() -> XQueryResult<()> {
        assert_eq!(
            tokens("1 (: outer (: inner :) still :) + 2")?,
            vec![
                Token::Integer("1".to_owned()),
                Token::Plus,
                Token::Integer("2".to_owned())
            ]
        );
        assert!(tokens("1 (: never closed").is_err());
        Ok(())
    }

    #[test]
    fn braced_names_are_single_tokens() -> XQueryResult<()> {
        assert_eq!(
            tokens("Q{urn:x}local")?,
            vec![Token::BracedName {
                uri: "urn:x".to_owned(),
                local: "local".to_owned()
            }]
        );
        assert_eq!(
            tokens("Q{urn:x}*")?,
            vec![Token::BracedWildcard {
                uri: "urn:x".to_owned()
            }]
        );
        Ok(())
    }

    #[test]
    fn a_bad_character_reference_is_xqst0090() {
        let failure = tokens("'&#xD800;'").expect_err("a surrogate is not an XML character");
        assert!(failure.is(crate::error::XQueryErrorCode::Xqst0090));
    }

    #[test]
    fn the_scanner_rewinds() -> XQueryResult<()> {
        let mut lexer = Lexer::new("a b");
        let first = lexer.next_token()?;
        let saved = lexer.position();
        let second = lexer.next_token()?;
        lexer.seek(saved);
        assert_eq!(lexer.next_token()?, second);
        assert_eq!(first.token, Token::Name("a".to_owned()));
        Ok(())
    }
}
