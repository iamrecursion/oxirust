//! CTL (Color Transform Language) lexer for ACES color transforms.
//!
//! Tokenizes CTL source code into a sequence of [`CtlToken`]s for subsequent
//! parsing. Handles whitespace, line/block comments, identifiers, keywords,
//! numeric literals (integer and float), and operator/punctuation symbols.

use crate::error::ColorError;

// ─────────────────────────────────────────────────────────────────────────────
// Token types
// ─────────────────────────────────────────────────────────────────────────────

/// A single token produced by the CTL lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum CtlToken {
    // ── Literals ──────────────────────────────────────────────────────────────
    /// A floating-point literal, e.g. `1.0`, `3.14f`, `-0.5`.
    FloatLit(f64),
    /// An integer literal that does not contain a decimal point, e.g. `2`.
    IntLit(i64),
    /// Boolean literal (`true` / `false`).
    BoolLit(bool),

    // ── Keywords ──────────────────────────────────────────────────────────────
    /// `float`
    KwFloat,
    /// `int`
    KwInt,
    /// `bool`
    KwBool,
    /// `void`
    KwVoid,
    /// `if`
    KwIf,
    /// `else`
    KwElse,
    /// `input`
    KwInput,
    /// `output`
    KwOutput,
    /// `varying`
    KwVarying,
    /// `uniform`
    KwUniform,
    /// `return`
    KwReturn,

    // ── Identifiers ───────────────────────────────────────────────────────────
    /// Any identifier that is not a keyword, e.g. `rOut`, `myVar`.
    Ident(String),

    // ── Operators ─────────────────────────────────────────────────────────────
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `=` (assignment)
    Assign,
    /// `==` (equality comparison)
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `&&`
    And,
    /// `||`
    Or,
    /// `!`
    Bang,

    // ── Punctuation ───────────────────────────────────────────────────────────
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `;`
    Semi,
    /// `,`
    Comma,
    /// `.`
    Dot,
}

// ─────────────────────────────────────────────────────────────────────────────
// Lexer state
// ─────────────────────────────────────────────────────────────────────────────

struct Lexer<'s> {
    src: &'s [u8],
    pos: usize,
}

impl<'s> Lexer<'s> {
    fn new(source: &'s str) -> Self {
        Self {
            src: source.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.src.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    /// Skip ASCII whitespace characters.
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Skip a `//` line comment (already consumed `//`).
    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.advance() {
            if ch == b'\n' {
                break;
            }
        }
    }

    /// Skip a `/* … */` block comment (already consumed `/*`).
    fn skip_block_comment(&mut self) -> Result<(), ColorError> {
        loop {
            match self.advance() {
                None => return Err(ColorError::Parse("unterminated block comment".to_string())),
                Some(b'*') if self.peek() == Some(b'/') => {
                    self.pos += 1; // consume '/'
                    return Ok(());
                }
                _ => {}
            }
        }
    }

    /// Lex a numeric literal (integer or float).
    ///
    /// Called after the first digit has been *peeked* (not consumed).
    fn lex_number(&mut self) -> Result<CtlToken, ColorError> {
        let start = self.pos;
        let mut is_float = false;

        // Consume digits
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }

        // Optional fractional part
        if self.peek() == Some(b'.') && matches!(self.peek2(), Some(b'0'..=b'9') | None) {
            // peek2 might be None (e.g. "1." at end of input) — still a float
            is_float = true;
            self.pos += 1; // consume '.'
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        // Optional exponent
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        // Optional trailing 'f' / 'F' (GLSL / CTL style float suffix)
        if matches!(self.peek(), Some(b'f' | b'F')) {
            is_float = true;
            self.pos += 1;
        }

        let slice = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|e| ColorError::Parse(format!("invalid UTF-8 in numeric literal: {e}")))?;

        // Strip trailing 'f'/'F' before parsing
        let numeric_str = slice.trim_end_matches(['f', 'F']);

        if is_float {
            let val: f64 = numeric_str
                .parse()
                .map_err(|_| ColorError::Parse(format!("invalid float literal: '{slice}'")))?;
            Ok(CtlToken::FloatLit(val))
        } else {
            let val: i64 = numeric_str
                .parse()
                .map_err(|_| ColorError::Parse(format!("invalid integer literal: '{slice}'")))?;
            Ok(CtlToken::IntLit(val))
        }
    }

    /// Lex an identifier or keyword. Called after the first alphabetic/`_` byte
    /// has been *peeked*.
    fn lex_ident_or_kw(&mut self) -> CtlToken {
        let start = self.pos;
        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-')
        ) {
            self.pos += 1;
        }
        // Safety: slice is always valid UTF-8 since we only accepted ASCII bytes.
        let word = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        match word {
            "float" => CtlToken::KwFloat,
            "int" => CtlToken::KwInt,
            "bool" => CtlToken::KwBool,
            "void" => CtlToken::KwVoid,
            "if" => CtlToken::KwIf,
            "else" => CtlToken::KwElse,
            "input" => CtlToken::KwInput,
            "output" => CtlToken::KwOutput,
            "varying" => CtlToken::KwVarying,
            "uniform" => CtlToken::KwUniform,
            "return" => CtlToken::KwReturn,
            "true" => CtlToken::BoolLit(true),
            "false" => CtlToken::BoolLit(false),
            other => CtlToken::Ident(other.to_string()),
        }
    }

    /// Produce the next token, or `None` at end of input.
    fn next_token(&mut self) -> Result<Option<CtlToken>, ColorError> {
        loop {
            self.skip_whitespace();
            let ch = match self.peek() {
                None => return Ok(None),
                Some(c) => c,
            };

            // ── Comments ──────────────────────────────────────────────────────
            if ch == b'/' {
                match self.peek2() {
                    Some(b'/') => {
                        self.pos += 2;
                        self.skip_line_comment();
                        continue;
                    }
                    Some(b'*') => {
                        self.pos += 2;
                        self.skip_block_comment()?;
                        continue;
                    }
                    _ => {
                        self.pos += 1;
                        return Ok(Some(CtlToken::Slash));
                    }
                }
            }

            // ── Numeric literals ──────────────────────────────────────────────
            if ch.is_ascii_digit() {
                return self.lex_number().map(Some);
            }

            // A leading dot followed by a digit is also a float (e.g. `.5`)
            if ch == b'.' {
                if matches!(self.peek2(), Some(b'0'..=b'9')) {
                    // Insert a leading '0' conceptually: treat as float starting at '.'
                    let start = self.pos;
                    self.pos += 1; // consume '.'
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.pos += 1;
                    }
                    if matches!(self.peek(), Some(b'f' | b'F')) {
                        self.pos += 1;
                    }
                    let slice = std::str::from_utf8(&self.src[start..self.pos])
                        .map_err(|e| ColorError::Parse(format!("UTF-8 error: {e}")))?;
                    let numeric = slice.trim_end_matches(['f', 'F']);
                    let val: f64 = numeric.parse().map_err(|_| {
                        ColorError::Parse(format!("invalid float literal: '{slice}'"))
                    })?;
                    return Ok(Some(CtlToken::FloatLit(val)));
                }
                self.pos += 1;
                return Ok(Some(CtlToken::Dot));
            }

            // ── Identifiers / keywords ─────────────────────────────────────
            if ch.is_ascii_alphabetic() || ch == b'_' {
                return Ok(Some(self.lex_ident_or_kw()));
            }

            // ── Two-character operators ────────────────────────────────────
            self.pos += 1; // consume `ch`
            let tok = match ch {
                b'+' => CtlToken::Plus,
                b'-' => CtlToken::Minus,
                b'*' => CtlToken::Star,
                b'(' => CtlToken::LParen,
                b')' => CtlToken::RParen,
                b'{' => CtlToken::LBrace,
                b'}' => CtlToken::RBrace,
                b'[' => CtlToken::LBracket,
                b']' => CtlToken::RBracket,
                b';' => CtlToken::Semi,
                b',' => CtlToken::Comma,
                b'=' => {
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        CtlToken::Eq
                    } else {
                        CtlToken::Assign
                    }
                }
                b'!' => {
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        CtlToken::Ne
                    } else {
                        CtlToken::Bang
                    }
                }
                b'<' => {
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        CtlToken::Le
                    } else {
                        CtlToken::Lt
                    }
                }
                b'>' => {
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        CtlToken::Ge
                    } else {
                        CtlToken::Gt
                    }
                }
                b'&' => {
                    if self.peek() == Some(b'&') {
                        self.pos += 1;
                        CtlToken::And
                    } else {
                        return Err(ColorError::Parse(
                            "single '&' is not a valid CTL operator; did you mean '&&'?"
                                .to_string(),
                        ));
                    }
                }
                b'|' => {
                    if self.peek() == Some(b'|') {
                        self.pos += 1;
                        CtlToken::Or
                    } else {
                        return Err(ColorError::Parse(
                            "single '|' is not a valid CTL operator; did you mean '||'?"
                                .to_string(),
                        ));
                    }
                }
                other => {
                    return Err(ColorError::Parse(format!(
                        "unexpected character '{}' (0x{:02X})",
                        other as char, other
                    )))
                }
            };
            return Ok(Some(tok));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenize a CTL source string into a flat list of [`CtlToken`]s.
///
/// # Errors
///
/// Returns [`ColorError::Parse`] on any lexical error (unknown character,
/// unterminated block comment, malformed numeric literal, etc.).
pub fn tokenize(source: &str) -> Result<Vec<CtlToken>, ColorError> {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    while let Some(tok) = lexer.next_token()? {
        tokens.push(tok);
    }
    Ok(tokens)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_source() {
        let toks = tokenize("").expect("tokenize empty");
        assert!(toks.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let toks = tokenize("   \n\t  ").expect("tokenize whitespace");
        assert!(toks.is_empty());
    }

    #[test]
    fn test_line_comment_skipped() {
        let toks = tokenize("// this is a comment\nfloat").expect("tokenize line comment");
        assert_eq!(toks, vec![CtlToken::KwFloat]);
    }

    #[test]
    fn test_block_comment_skipped() {
        let toks = tokenize("/* block */ float").expect("tokenize block comment");
        assert_eq!(toks, vec![CtlToken::KwFloat]);
    }

    #[test]
    fn test_keywords() {
        let src = "float int bool void if else input output varying uniform return";
        let toks = tokenize(src).expect("tokenize source");
        assert_eq!(
            toks,
            vec![
                CtlToken::KwFloat,
                CtlToken::KwInt,
                CtlToken::KwBool,
                CtlToken::KwVoid,
                CtlToken::KwIf,
                CtlToken::KwElse,
                CtlToken::KwInput,
                CtlToken::KwOutput,
                CtlToken::KwVarying,
                CtlToken::KwUniform,
                CtlToken::KwReturn,
            ]
        );
    }

    #[test]
    fn test_bool_literals() {
        let toks = tokenize("true false").expect("tokenize booleans");
        assert_eq!(
            toks,
            vec![CtlToken::BoolLit(true), CtlToken::BoolLit(false)]
        );
    }

    #[test]
    fn test_float_literals() {
        let toks = tokenize("1.0 3.14 0.5f 2.0F 1e3 1.5e-2").expect("tokenize floats");
        assert!(matches!(toks[0], CtlToken::FloatLit(_)));
        assert!(matches!(toks[1], CtlToken::FloatLit(_)));
        assert!(matches!(toks[2], CtlToken::FloatLit(_)));
        assert!(matches!(toks[3], CtlToken::FloatLit(_)));
        assert!(matches!(toks[4], CtlToken::FloatLit(_)));
        assert!(matches!(toks[5], CtlToken::FloatLit(_)));
        assert_eq!(toks.len(), 6);
    }

    #[test]
    fn test_integer_literal() {
        let toks = tokenize("42").expect("tokenize integer");
        assert_eq!(toks, vec![CtlToken::IntLit(42)]);
    }

    #[test]
    fn test_operators() {
        let src = "+ - * / = == != < > <= >= && || !";
        let toks = tokenize(src).expect("tokenize source");
        assert_eq!(
            toks,
            vec![
                CtlToken::Plus,
                CtlToken::Minus,
                CtlToken::Star,
                CtlToken::Slash,
                CtlToken::Assign,
                CtlToken::Eq,
                CtlToken::Ne,
                CtlToken::Lt,
                CtlToken::Gt,
                CtlToken::Le,
                CtlToken::Ge,
                CtlToken::And,
                CtlToken::Or,
                CtlToken::Bang,
            ]
        );
    }

    #[test]
    fn test_punctuation() {
        let toks = tokenize("( ) { } [ ] ; ,").expect("tokenize punctuation");
        assert_eq!(
            toks,
            vec![
                CtlToken::LParen,
                CtlToken::RParen,
                CtlToken::LBrace,
                CtlToken::RBrace,
                CtlToken::LBracket,
                CtlToken::RBracket,
                CtlToken::Semi,
                CtlToken::Comma,
            ]
        );
    }

    #[test]
    fn test_identifier() {
        let toks = tokenize("rOut gIn myVar").expect("tokenize identifiers");
        assert_eq!(
            toks,
            vec![
                CtlToken::Ident("rOut".to_string()),
                CtlToken::Ident("gIn".to_string()),
                CtlToken::Ident("myVar".to_string()),
            ]
        );
    }

    #[test]
    fn test_unterminated_block_comment_error() {
        let result = tokenize("/* unterminated");
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_char_error() {
        let result = tokenize("@");
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_declaration() {
        let src = "float x = 1.0;";
        let toks = tokenize(src).expect("tokenize source");
        assert_eq!(
            toks,
            vec![
                CtlToken::KwFloat,
                CtlToken::Ident("x".to_string()),
                CtlToken::Assign,
                CtlToken::FloatLit(1.0),
                CtlToken::Semi,
            ]
        );
    }

    #[test]
    fn test_ctl_main_signature_tokens() {
        let src = "void main(output varying float rOut, input varying float rIn)";
        let toks = tokenize(src).expect("tokenize source");
        // Spot-check a few positions
        assert_eq!(toks[0], CtlToken::KwVoid);
        assert_eq!(toks[1], CtlToken::Ident("main".to_string()));
        assert_eq!(toks[2], CtlToken::LParen);
        assert_eq!(toks[3], CtlToken::KwOutput);
    }

    #[test]
    fn test_leading_dot_float() {
        let toks = tokenize(".5").expect("tokenize leading dot float");
        assert_eq!(toks, vec![CtlToken::FloatLit(0.5)]);
    }
}
