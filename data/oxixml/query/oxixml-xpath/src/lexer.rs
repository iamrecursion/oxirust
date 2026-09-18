//! The XPath 3.1 tokenizer.
//!
//! XPath has no reserved words: `div`, `and`, `to` and `element` are ordinary names wherever a
//! name is expected, and operators only where an operator is expected. The lexer therefore does
//! **no** keyword recognition at all — it produces names, literals and punctuation, and the
//! recursive-descent parser decides what a name means from its position. This is the division of
//! labour the specification's own note on
//! [lexical states](https://www.w3.org/TR/xpath-31/#extra-grammatical-constraints) describes,
//! and it is why `child::div`, `a div b` and `div(1)` all tokenize identically here.
//!
//! Four lexical decisions *are* made here, because the parser cannot make them:
//!
//! * `-` is an `NCName` character, so `x-1` is one name while `x - 1` and `1-1` are three
//!   tokens. This follows XPath's own terminal definitions.
//! * `pre:local` is a single [`Token::Name`], but `pre:*` and `pre::` are not: they become
//!   `Name`,`Colon`,`Star` and `Name`,`ColonColon` respectively.
//! * `Q{uri}local` and `Q{uri}*` are single tokens, since the braced URI literal has its own
//!   character rules.
//! * the *host language* settles whether a `PredefinedEntityRef` or a `CharRef` expands inside a
//!   string literal and inside a braced URI literal. XPath's grammar has neither; XQuery's has
//!   both, so an engine compiling query text asks for the XQuery lexis with
//!   [`crate::StaticContext::with_xquery_host_language`].

use crate::error::{XPathError, XPathResult};

/// One lexical token together with the byte offset it starts at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Spanned {
    pub(crate) token: Token,
    pub(crate) offset: usize,
}

/// A lexical token of the XPath 3.1 grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Token {
    /// An `NCName` or a lexical `QName` (`prefix:local`).
    Name(String),
    /// `Q{uri}local` — a `URIQualifiedName`.
    BracedName { uri: String, local: String },
    /// `Q{uri}*` — a wildcard in an explicitly named namespace.
    BracedWildcard { uri: String },
    /// An `IntegerLiteral`, in its source spelling.
    Integer(String),
    /// A `DecimalLiteral`, in its source spelling.
    Decimal(String),
    /// A `DoubleLiteral`, in its source spelling.
    Double(String),
    /// A `StringLiteral`, with doubled quotation marks already resolved.
    Str(String),
    /// `$`
    Dollar,
    /// `@`
    At,
    /// `,`
    Comma,
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
            Self::Str(text) => format!("the string literal \"{text}\""),
            other => format!("`{}`", other.spelling()),
        }
    }

    /// How many characters this token occupies in the source, for the productions the grammar
    /// marks `/* ws: explicit */` and that therefore have to be recognized by adjacency.
    ///
    /// Only the tokens such a production is built from need to answer exactly: a `Name` (an
    /// `NCName` or a lexical `QName`, both spelled as written), and the one-character
    /// punctuation. Everything else reports its punctuation spelling, which is never consulted.
    pub(crate) fn width(&self) -> usize {
        match self {
            Self::Name(name) => name.chars().count(),
            Self::Integer(text) | Self::Decimal(text) | Self::Double(text) => text.chars().count(),
            other => other.spelling().chars().count(),
        }
    }

    /// The source spelling of a punctuation token.
    pub(crate) fn spelling(&self) -> &'static str {
        match self {
            Self::Dollar => "$",
            Self::At => "@",
            Self::Comma => ",",
            Self::LeftParen => "(",
            Self::RightParen => ")",
            Self::LeftBracket => "[",
            Self::RightBracket => "]",
            Self::LeftBrace => "{",
            Self::RightBrace => "}",
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
            _ => "a literal",
        }
    }
}

/// The XML `NameStartChar` production, minus `:` — that is, the first character of an `NCName`.
#[must_use]
pub(crate) fn is_ncname_start(character: char) -> bool {
    matches!(character,
        'A'..='Z'
        | '_'
        | 'a'..='z'
        | '\u{C0}'..='\u{D6}'
        | '\u{D8}'..='\u{F6}'
        | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}'
        | '\u{37F}'..='\u{1FFF}'
        | '\u{200C}'..='\u{200D}'
        | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}'
        | '\u{3001}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}'
        | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}')
}

/// The XML `NameChar` production, minus `:` — a non-initial character of an `NCName`.
#[must_use]
pub(crate) fn is_ncname_char(character: char) -> bool {
    is_ncname_start(character)
        || matches!(character,
            '-' | '.'
            | '0'..='9'
            | '\u{B7}'
            | '\u{300}'..='\u{36F}'
            | '\u{203F}'..='\u{2040}')
}

/// Whether `text` is a complete `NCName` — a name with no colon in it.
#[must_use]
pub(crate) fn is_ncname(text: &str) -> bool {
    let mut characters = text.chars();
    characters.next().is_some_and(is_ncname_start) && characters.all(is_ncname_char)
}

/// Tokenizes `source`, skipping whitespace and (possibly nested) `(: … :)` comments.
///
/// `references` selects the host language's lexis: with it set, a `PredefinedEntityRef` or a
/// `CharRef` expands inside a string literal and inside a braced URI literal, as XQuery's
/// grammar allows and XPath's does not. See
/// [`crate::StaticContext::with_xquery_host_language`].
///
/// # Errors
///
/// [`XPathErrorCode::Xpst0003`](crate::XPathErrorCode::Xpst0003) for an unterminated string
/// literal or comment, a malformed number or braced URI literal, or a character that cannot
/// begin a token.
pub(crate) fn tokenize_with_references(
    source: &str,
    references: bool,
) -> XPathResult<Vec<Spanned>> {
    let mut lexer = Lexer::new(source);
    lexer.xquery_lexis = references;
    lexer.run()
}

struct Lexer<'a> {
    source: &'a str,
    chars: Vec<char>,
    offsets: Vec<usize>,
    index: usize,
    tokens: Vec<Spanned>,
    /// Whether entity and character references expand inside `Q{…}`, as they do in XQuery.
    xquery_lexis: bool,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        let mut chars = Vec::new();
        let mut offsets = Vec::new();
        for (offset, character) in source.char_indices() {
            chars.push(character);
            offsets.push(offset);
        }
        offsets.push(source.len());
        Self {
            source,
            chars,
            offsets,
            index: 0,
            tokens: Vec::new(),
            xquery_lexis: false,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).copied()
    }

    fn peek_at(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.index + ahead).copied()
    }

    fn offset(&self) -> usize {
        self.offsets
            .get(self.index)
            .copied()
            .unwrap_or(self.source.len())
    }

    fn push(&mut self, offset: usize, token: Token) {
        self.tokens.push(Spanned { token, offset });
    }

    fn run(mut self) -> XPathResult<Vec<Spanned>> {
        loop {
            self.skip_ignorable()?;
            let offset = self.offset();
            let Some(character) = self.peek() else {
                return Ok(self.tokens);
            };
            match character {
                '0'..='9' => self.number(offset)?,
                '.' if matches!(self.peek_at(1), Some('0'..='9')) => self.number(offset)?,
                '"' | '\'' => self.string(offset, character)?,
                _ if is_ncname_start(character) => self.name(offset)?,
                _ => self.punctuation(offset, character)?,
            }
        }
    }

    fn skip_ignorable(&mut self) -> XPathResult<()> {
        loop {
            match self.peek() {
                Some(' ' | '\t' | '\r' | '\n') => self.index += 1,
                Some('(') if self.peek_at(1) == Some(':') => self.comment()?,
                _ => return Ok(()),
            }
        }
    }

    fn comment(&mut self) -> XPathResult<()> {
        let start = self.offset();
        let mut depth = 0_usize;
        loop {
            match (self.peek(), self.peek_at(1)) {
                (Some('('), Some(':')) => {
                    depth += 1;
                    self.index += 2;
                }
                (Some(':'), Some(')')) => {
                    depth -= 1;
                    self.index += 2;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                (Some(_), _) => self.index += 1,
                (None, _) => {
                    return Err(XPathError::syntax(start, "unterminated `(:` comment"));
                }
            }
        }
    }

    fn string(&mut self, offset: usize, quote: char) -> XPathResult<()> {
        self.index += 1;
        let mut value = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(XPathError::syntax(offset, "unterminated string literal"));
                }
                Some(character) if character == quote => {
                    self.index += 1;
                    if self.peek() == Some(quote) {
                        value.push(quote);
                        self.index += 1;
                    } else {
                        self.push(offset, Token::Str(value));
                        return Ok(());
                    }
                }
                // XQuery 3.1 §A.2.1 admits `PredefinedEntityRef` and `CharRef` inside a string
                // literal where XPath 3.1 §A.2.1 has neither, so `'a&#x2d;pi'` is `a-pi` in a
                // query and nine characters in an expression.
                Some('&') if self.xquery_lexis => {
                    let resolved = self.reference(offset)?;
                    value.push_str(&resolved);
                }
                Some(character) => {
                    value.push(character);
                    self.index += 1;
                }
            }
        }
    }

    fn number(&mut self, offset: usize) -> XPathResult<()> {
        let start = self.index;
        let mut fractional = false;
        let mut exponent = false;
        while matches!(self.peek(), Some('0'..='9')) {
            self.index += 1;
        }
        if self.peek() == Some('.') {
            fractional = true;
            self.index += 1;
            while matches!(self.peek(), Some('0'..='9')) {
                self.index += 1;
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let after_sign = match self.peek_at(1) {
                Some('+' | '-') => 2,
                _ => 1,
            };
            if matches!(self.peek_at(after_sign), Some('0'..='9')) {
                exponent = true;
                self.index += after_sign;
                while matches!(self.peek(), Some('0'..='9')) {
                    self.index += 1;
                }
            }
        }
        let text: String = self.chars[start..self.index].iter().collect();
        if matches!(self.peek(), Some(character) if is_ncname_start(character)) {
            // `1e` or `2x`: the specification requires a numeric literal to be followed by a
            // delimiting character.
            return Err(XPathError::syntax(
                offset,
                format!("the numeric literal `{text}` runs into the following name"),
            ));
        }
        let token = if exponent {
            Token::Double(text)
        } else if fractional {
            Token::Decimal(text)
        } else {
            Token::Integer(text)
        };
        self.push(offset, token);
        Ok(())
    }

    fn read_ncname(&mut self) -> String {
        let start = self.index;
        if matches!(self.peek(), Some(character) if is_ncname_start(character)) {
            self.index += 1;
            while matches!(self.peek(), Some(character) if is_ncname_char(character)) {
                self.index += 1;
            }
        }
        self.chars[start..self.index].iter().collect()
    }

    /// Whether the two tokens just emitted are an adjacent `*` `:`, which makes the name now
    /// being read the tail of the `"*" ":" NCName` wildcard.
    ///
    /// The `Wildcard` production is `/* ws: explicit */` (XPath 3.1 §A.2.4.1) and its third part
    /// is an `NCName`, not a `QName`, so the longest match ends at the first colon: `map{*:b:b}`
    /// is the key `*:b`, a colon, and the value `b` (`MapConstructor-019`, `-029`).
    fn after_star_colon(&self, offset: usize) -> bool {
        let [.., star, colon] = self.tokens.as_slice() else {
            return false;
        };
        star.token == Token::Star
            && colon.token == Token::Colon
            && star.offset + 1 == colon.offset
            && colon.offset + 1 == offset
    }

    fn name(&mut self, offset: usize) -> XPathResult<()> {
        let first = self.read_ncname();
        if first == "Q" && self.peek() == Some('{') {
            return self.braced_name(offset);
        }
        if self.after_star_colon(offset) {
            self.push(offset, Token::Name(first));
            return Ok(());
        }
        if self.peek() == Some(':')
            && matches!(self.peek_at(1), Some(character) if is_ncname_start(character))
        {
            self.index += 1;
            let local = self.read_ncname();
            self.push(offset, Token::Name(format!("{first}:{local}")));
            return Ok(());
        }
        self.push(offset, Token::Name(first));
        Ok(())
    }

    fn braced_name(&mut self, offset: usize) -> XPathResult<()> {
        self.index += 1;
        let mut raw = String::new();
        loop {
            match self.peek() {
                None | Some('{') => {
                    return Err(XPathError::syntax(
                        offset,
                        "unterminated `Q{` braced URI literal",
                    ));
                }
                Some('}') => break,
                // A reference expands to *content*, never to punctuation: an expanded `}` stays
                // part of the URI rather than closing the literal.
                Some('&') if self.xquery_lexis => {
                    raw.push_str(&self.reference(offset)?);
                }
                Some(character) => {
                    raw.push(character);
                    self.index += 1;
                }
            }
        }
        self.index += 1;
        let uri = collapse_whitespace(&raw);
        if self.peek() == Some('*') {
            self.index += 1;
            self.push(offset, Token::BracedWildcard { uri });
            return Ok(());
        }
        let local = self.read_ncname();
        if local.is_empty() {
            return Err(XPathError::syntax(
                offset,
                "`Q{…}` must be followed by an NCName or `*`",
            ));
        }
        self.push(offset, Token::BracedName { uri, local });
        Ok(())
    }

    /// Reads one `PredefinedEntityRef` or `CharRef`, with the cursor on its `&`, and returns the
    /// text it stands for.
    fn reference(&mut self, offset: usize) -> XPathResult<String> {
        let start = self.index + 1;
        let mut end = start;
        while !matches!(self.chars.get(end), Some(';') | None) {
            end += 1;
        }
        if self.chars.get(end).is_none() {
            return Err(XPathError::syntax(
                offset,
                "the entity or character reference is never closed",
            ));
        }
        let body: String = self.chars[start..end].iter().collect();
        self.index = end + 1;
        let resolved = match body.as_str() {
            "lt" => "<".to_owned(),
            "gt" => ">".to_owned(),
            "amp" => "&".to_owned(),
            "quot" => "\"".to_owned(),
            "apos" => "'".to_owned(),
            _ => {
                let digits = body.strip_prefix('#').ok_or_else(|| {
                    XPathError::syntax(
                        offset,
                        format!("`&{body};` is not a predefined entity reference"),
                    )
                })?;
                let code = match digits.strip_prefix('x') {
                    Some(hex) if !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()) => {
                        u32::from_str_radix(hex, 16).ok()
                    }
                    Some(_) => None,
                    None if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) => {
                        digits.parse::<u32>().ok()
                    }
                    None => {
                        return Err(XPathError::syntax(
                            offset,
                            format!("`&{body};` is not a well-formed character reference"),
                        ));
                    }
                };
                let character = code.and_then(char::from_u32).ok_or_else(|| {
                    XPathError::syntax(
                        offset,
                        format!("`&{body};` does not denote a legal XML character"),
                    )
                })?;
                character.to_string()
            }
        };
        Ok(resolved)
    }

    fn punctuation(&mut self, offset: usize, character: char) -> XPathResult<()> {
        let second = self.peek_at(1);
        let (token, width) = match (character, second) {
            ('/', Some('/')) => (Token::DoubleSlash, 2),
            ('/', _) => (Token::Slash, 1),
            (':', Some(':')) => (Token::ColonColon, 2),
            (':', Some('=')) => (Token::Assign, 2),
            (':', _) => (Token::Colon, 1),
            ('!', Some('=')) => (Token::NotEqual, 2),
            ('!', _) => (Token::Bang, 1),
            ('<', Some('<')) => (Token::Precedes, 2),
            ('<', Some('=')) => (Token::LessEqual, 2),
            ('<', _) => (Token::Less, 1),
            ('>', Some('>')) => (Token::Follows, 2),
            ('>', Some('=')) => (Token::GreaterEqual, 2),
            ('>', _) => (Token::Greater, 1),
            ('|', Some('|')) => (Token::Concat, 2),
            ('|', _) => (Token::Pipe, 1),
            ('=', Some('>')) => (Token::Arrow, 2),
            ('=', _) => (Token::Equal, 1),
            ('.', Some('.')) => (Token::DotDot, 2),
            ('.', _) => (Token::Dot, 1),
            ('$', _) => (Token::Dollar, 1),
            ('@', _) => (Token::At, 1),
            (',', _) => (Token::Comma, 1),
            ('(', _) => (Token::LeftParen, 1),
            (')', _) => (Token::RightParen, 1),
            ('[', _) => (Token::LeftBracket, 1),
            (']', _) => (Token::RightBracket, 1),
            ('{', _) => (Token::LeftBrace, 1),
            ('}', _) => (Token::RightBrace, 1),
            ('*', _) => (Token::Star, 1),
            ('+', _) => (Token::Plus, 1),
            ('-', _) => (Token::Minus, 1),
            ('?', _) => (Token::Question, 1),
            ('#', _) => (Token::Hash, 1),
            _ => {
                return Err(XPathError::syntax(
                    offset,
                    format!("`{character}` cannot start a token"),
                ));
            }
        };
        self.index += width;
        self.push(offset, token);
        Ok(())
    }
}

/// Applies the `collapse` whitespace facet, which XPath 3.1 §3.1.2 requires of the URI inside a
/// `BracedURILiteral`.
fn collapse_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut pending = false;
    for character in input.chars() {
        if matches!(character, ' ' | '\t' | '\r' | '\n') {
            pending = !out.is_empty();
        } else {
            if pending {
                out.push(' ');
                pending = false;
            }
            out.push(character);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> Vec<Token> {
        tokenize_with_references(source, false)
            .expect("a well-formed expression")
            .into_iter()
            .map(|spanned| spanned.token)
            .collect()
    }

    #[test]
    fn names_absorb_hyphens_but_not_spaced_minus_signs() {
        assert_eq!(lex("x-1"), vec![Token::Name("x-1".to_owned())]);
        assert_eq!(
            lex("x - 1"),
            vec![
                Token::Name("x".to_owned()),
                Token::Minus,
                Token::Integer("1".to_owned())
            ]
        );
        assert_eq!(
            lex("1-1"),
            vec![
                Token::Integer("1".to_owned()),
                Token::Minus,
                Token::Integer("1".to_owned())
            ]
        );
    }

    #[test]
    fn colons_separate_axes_wildcards_and_qualified_names() {
        assert_eq!(lex("a:b"), vec![Token::Name("a:b".to_owned())]);
        assert_eq!(
            lex("child::a"),
            vec![
                Token::Name("child".to_owned()),
                Token::ColonColon,
                Token::Name("a".to_owned())
            ]
        );
        assert_eq!(
            lex("pre:*"),
            vec![Token::Name("pre".to_owned()), Token::Colon, Token::Star]
        );
        assert_eq!(
            lex("*:local"),
            vec![Token::Star, Token::Colon, Token::Name("local".to_owned())]
        );
        assert_eq!(
            lex("$x := 1"),
            vec![
                Token::Dollar,
                Token::Name("x".to_owned()),
                Token::Assign,
                Token::Integer("1".to_owned())
            ]
        );
    }

    #[test]
    fn braced_uri_literals() {
        assert_eq!(
            lex("Q{urn:x}local"),
            vec![Token::BracedName {
                uri: "urn:x".to_owned(),
                local: "local".to_owned()
            }]
        );
        assert_eq!(
            lex("Q{ urn:x }*"),
            vec![Token::BracedWildcard {
                uri: "urn:x".to_owned()
            }]
        );
        assert!(tokenize_with_references("Q{urn:x", false).is_err());
        assert!(tokenize_with_references("Q{urn:x}", false).is_err());
        // `Qname` is an ordinary name.
        assert_eq!(lex("Qname"), vec![Token::Name("Qname".to_owned())]);
    }

    #[test]
    fn numeric_literals_are_classified() {
        assert_eq!(lex("42"), vec![Token::Integer("42".to_owned())]);
        assert_eq!(lex("4.2"), vec![Token::Decimal("4.2".to_owned())]);
        assert_eq!(lex("4."), vec![Token::Decimal("4.".to_owned())]);
        assert_eq!(lex(".5"), vec![Token::Decimal(".5".to_owned())]);
        assert_eq!(lex("4.2e1"), vec![Token::Double("4.2e1".to_owned())]);
        assert_eq!(lex("4E-1"), vec![Token::Double("4E-1".to_owned())]);
        // `e` not followed by digits is a separate name.
        assert_eq!(
            lex("1.5 eq 2"),
            vec![
                Token::Decimal("1.5".to_owned()),
                Token::Name("eq".to_owned()),
                Token::Integer("2".to_owned())
            ]
        );
        assert!(tokenize_with_references("1e", false).is_err());
    }

    #[test]
    fn string_literals_resolve_doubled_quotes() {
        assert_eq!(lex(r#""a""b""#), vec![Token::Str(r#"a"b"#.to_owned())]);
        assert_eq!(lex("'a''b'"), vec![Token::Str("a'b".to_owned())]);
        assert_eq!(lex("\"it's\""), vec![Token::Str("it's".to_owned())]);
        assert!(tokenize_with_references("\"unterminated", false).is_err());
    }

    #[test]
    fn comments_nest_and_are_skipped() {
        assert_eq!(
            lex("1 (: a (: nested :) comment :) + 2"),
            vec![
                Token::Integer("1".to_owned()),
                Token::Plus,
                Token::Integer("2".to_owned())
            ]
        );
        assert!(tokenize_with_references("1 (: unterminated", false).is_err());
    }

    #[test]
    fn dots_and_slashes() {
        assert_eq!(lex(".."), vec![Token::DotDot]);
        assert_eq!(lex("."), vec![Token::Dot]);
        assert_eq!(lex("//"), vec![Token::DoubleSlash]);
        assert_eq!(
            lex("a!b"),
            vec![
                Token::Name("a".to_owned()),
                Token::Bang,
                Token::Name("b".to_owned())
            ]
        );
        assert_eq!(
            lex("a=>b"),
            vec![
                Token::Name("a".to_owned()),
                Token::Arrow,
                Token::Name("b".to_owned())
            ]
        );
    }

    #[test]
    fn offsets_point_at_the_first_character() {
        let tokens = tokenize_with_references("  1 + 2", false).expect("valid");
        assert_eq!(tokens[0].offset, 2);
        assert_eq!(tokens[1].offset, 4);
        assert_eq!(tokens[2].offset, 6);
    }

    #[test]
    fn unknown_characters_are_rejected() {
        assert!(tokenize_with_references("1 ` 2", false).is_err());
        assert!(tokenize_with_references("\u{a3}", false).is_err());
    }
}
