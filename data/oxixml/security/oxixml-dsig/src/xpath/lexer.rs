//! The XPath 1.0 tokenizer.
//!
//! XPath's lexical structure is context-sensitive, and
//! [§3.7](https://www.w3.org/TR/1999/REC-xpath-19991116/#exprlex) states the
//! three disambiguation rules that make it tractable. All three are implemented
//! here, because getting any of them wrong silently changes what an expression
//! selects:
//!
//! 1. If the preceding token is none of `@`, `::`, `(`, `[`, `,` or an
//!    operator, then `*` is the multiply operator and an `NCName` that spells
//!    `and`, `or`, `mod` or `div` is that operator. So `a * b` multiplies while
//!    `a[*]` tests for any child, and `div div div` divides one `<div>` by
//!    another.
//! 2. An `NCName` immediately followed by `(` is a node type or a function
//!    name, never a name test — `comment()` is not "the child named
//!    `comment`".
//! 3. An `NCName` immediately followed by `::` is an axis name.
//!
//! Whitespace between a name and the `(` or `::` is allowed and skipped when
//! applying rules 2 and 3.

use std::fmt;

/// One XPath token.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token {
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `[`
    LeftBracket,
    /// `]`
    RightBracket,
    /// `.`
    Dot,
    /// `..`
    DotDot,
    /// `@`
    At,
    /// `,`
    Comma,
    /// `::`
    ColonColon,
    /// `/`
    Slash,
    /// `//`
    SlashSlash,
    /// `|`
    Pipe,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `=`
    Equal,
    /// `!=`
    NotEqual,
    /// `<`
    Less,
    /// `<=`
    LessOrEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterOrEqual,
    /// `*` used as the multiply operator.
    Star,
    /// `*` used as a name test.
    NameTestStar,
    /// `prefix:*`
    NameTestPrefixStar(String),
    /// `and`, `or`, `mod` or `div`.
    OperatorName(&'static str),
    /// An axis name, without the `::`.
    AxisName(String),
    /// A function name or node type, without the `(`.
    FunctionName {
        /// The prefix, if written.
        prefix: Option<String>,
        /// The local part.
        local: String,
    },
    /// A name test.
    Name {
        /// The prefix, if written.
        prefix: Option<String>,
        /// The local part.
        local: String,
    },
    /// `$name`
    Variable {
        /// The prefix, if written.
        prefix: Option<String>,
        /// The local part.
        local: String,
    },
    /// A quoted string.
    Literal(String),
    /// A number.
    Number(f64),
}

impl Token {
    /// Whether a following `*` or `and`/`or`/`mod`/`div` is an operator,
    /// per disambiguation rule 1.
    const fn precedes_operator(&self) -> bool {
        !matches!(
            self,
            Self::At
                | Self::ColonColon
                | Self::LeftParen
                | Self::LeftBracket
                | Self::Comma
                | Self::Slash
                | Self::SlashSlash
                | Self::Pipe
                | Self::Plus
                | Self::Minus
                | Self::Equal
                | Self::NotEqual
                | Self::Less
                | Self::LessOrEqual
                | Self::Greater
                | Self::GreaterOrEqual
                | Self::Star
                | Self::OperatorName(_)
        )
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeftParen => f.write_str("("),
            Self::RightParen => f.write_str(")"),
            Self::LeftBracket => f.write_str("["),
            Self::RightBracket => f.write_str("]"),
            Self::Dot => f.write_str("."),
            Self::DotDot => f.write_str(".."),
            Self::At => f.write_str("@"),
            Self::Comma => f.write_str(","),
            Self::ColonColon => f.write_str("::"),
            Self::Slash => f.write_str("/"),
            Self::SlashSlash => f.write_str("//"),
            Self::Pipe => f.write_str("|"),
            Self::Plus => f.write_str("+"),
            Self::Minus => f.write_str("-"),
            Self::Equal => f.write_str("="),
            Self::NotEqual => f.write_str("!="),
            Self::Less => f.write_str("<"),
            Self::LessOrEqual => f.write_str("<="),
            Self::Greater => f.write_str(">"),
            Self::GreaterOrEqual => f.write_str(">="),
            Self::Star | Self::NameTestStar => f.write_str("*"),
            Self::NameTestPrefixStar(prefix) => write!(f, "{prefix}:*"),
            Self::OperatorName(name) => f.write_str(name),
            Self::AxisName(name) => write!(f, "{name}::"),
            Self::FunctionName { prefix, local } | Self::Name { prefix, local } => match prefix {
                Some(prefix) => write!(f, "{prefix}:{local}"),
                None => f.write_str(local),
            },
            Self::Variable { prefix, local } => match prefix {
                Some(prefix) => write!(f, "${prefix}:{local}"),
                None => write!(f, "${local}"),
            },
            Self::Literal(text) => write!(f, "{text:?}"),
            Self::Number(value) => write!(f, "{value}"),
        }
    }
}

/// Splits an XPath expression into tokens.
///
/// # Errors
///
/// Returns a human-readable message for an unterminated string, a stray
/// character or a malformed number.
pub(crate) fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let characters: Vec<char> = input.chars().collect();
    let mut tokens: Vec<Token> = Vec::new();
    let mut index = 0_usize;

    while index < characters.len() {
        let Some(&current) = characters.get(index) else {
            break;
        };
        if current.is_whitespace() {
            index += 1;
            continue;
        }
        let previous_precedes_operator = tokens.last().is_some_and(Token::precedes_operator);

        match current {
            '(' => push_simple(&mut tokens, &mut index, Token::LeftParen),
            ')' => push_simple(&mut tokens, &mut index, Token::RightParen),
            '[' => push_simple(&mut tokens, &mut index, Token::LeftBracket),
            ']' => push_simple(&mut tokens, &mut index, Token::RightBracket),
            '@' => push_simple(&mut tokens, &mut index, Token::At),
            ',' => push_simple(&mut tokens, &mut index, Token::Comma),
            '|' => push_simple(&mut tokens, &mut index, Token::Pipe),
            '+' => push_simple(&mut tokens, &mut index, Token::Plus),
            '-' => push_simple(&mut tokens, &mut index, Token::Minus),
            '=' => push_simple(&mut tokens, &mut index, Token::Equal),
            '*' => {
                index += 1;
                tokens.push(if previous_precedes_operator {
                    Token::Star
                } else {
                    Token::NameTestStar
                });
            }
            '/' => {
                if characters.get(index + 1) == Some(&'/') {
                    index += 2;
                    tokens.push(Token::SlashSlash);
                } else {
                    index += 1;
                    tokens.push(Token::Slash);
                }
            }
            '.' => match characters.get(index + 1) {
                Some('.') => {
                    index += 2;
                    tokens.push(Token::DotDot);
                }
                Some(digit) if digit.is_ascii_digit() => {
                    tokens.push(read_number(&characters, &mut index)?);
                }
                _ => {
                    index += 1;
                    tokens.push(Token::Dot);
                }
            },
            '!' => {
                if characters.get(index + 1) == Some(&'=') {
                    index += 2;
                    tokens.push(Token::NotEqual);
                } else {
                    return Err("`!` must be followed by `=`".to_owned());
                }
            }
            '<' => {
                if characters.get(index + 1) == Some(&'=') {
                    index += 2;
                    tokens.push(Token::LessOrEqual);
                } else {
                    index += 1;
                    tokens.push(Token::Less);
                }
            }
            '>' => {
                if characters.get(index + 1) == Some(&'=') {
                    index += 2;
                    tokens.push(Token::GreaterOrEqual);
                } else {
                    index += 1;
                    tokens.push(Token::Greater);
                }
            }
            ':' => {
                if characters.get(index + 1) == Some(&':') {
                    index += 2;
                    tokens.push(Token::ColonColon);
                } else {
                    return Err("a stray `:` is not valid XPath".to_owned());
                }
            }
            '\'' | '"' => tokens.push(read_literal(&characters, &mut index, current)?),
            '$' => {
                index += 1;
                let (prefix, local) = read_qualified_name(&characters, &mut index)?;
                tokens.push(Token::Variable { prefix, local });
            }
            digit if digit.is_ascii_digit() => tokens.push(read_number(&characters, &mut index)?),
            first if is_name_start(first) => {
                let (prefix, local) = read_qualified_name(&characters, &mut index)?;
                tokens.push(classify_name(
                    &characters,
                    index,
                    prefix,
                    local,
                    previous_precedes_operator,
                )?);
            }
            other => {
                return Err(format!(
                    "unexpected character {other:?} in an XPath expression"
                ));
            }
        }
    }
    Ok(tokens)
}

fn push_simple(tokens: &mut Vec<Token>, index: &mut usize, token: Token) {
    *index += 1;
    tokens.push(token);
}

/// Applies disambiguation rules 1–3 to a name that has just been read.
fn classify_name(
    characters: &[char],
    index: usize,
    prefix: Option<String>,
    local: String,
    previous_precedes_operator: bool,
) -> Result<Token, String> {
    // Rule 1: an operator name, but only where an operator can appear.
    if previous_precedes_operator && prefix.is_none() {
        if let Some(operator) = match local.as_str() {
            "and" => Some("and"),
            "or" => Some("or"),
            "mod" => Some("mod"),
            "div" => Some("div"),
            _ => None,
        } {
            return Ok(Token::OperatorName(operator));
        }
    }

    let mut lookahead = index;
    while characters.get(lookahead).is_some_and(|c| c.is_whitespace()) {
        lookahead += 1;
    }
    // Rule 3: an axis name.
    if characters.get(lookahead) == Some(&':') && characters.get(lookahead + 1) == Some(&':') {
        return match prefix {
            Some(_) => Err(format!(
                "an axis name cannot be prefixed: `{local}` is preceded by a prefix"
            )),
            None => Ok(Token::AxisName(local)),
        };
    }
    // Rule 2: a node type or a function name.
    if characters.get(lookahead) == Some(&'(') {
        return Ok(Token::FunctionName { prefix, local });
    }
    // `prefix:*` was already turned into a name by `read_qualified_name`.
    if local == "*" {
        return match prefix {
            Some(prefix) => Ok(Token::NameTestPrefixStar(prefix)),
            None => Ok(Token::NameTestStar),
        };
    }
    Ok(Token::Name { prefix, local })
}

fn read_literal(characters: &[char], index: &mut usize, quote: char) -> Result<Token, String> {
    *index += 1;
    let mut text = String::new();
    loop {
        match characters.get(*index) {
            None => return Err("an XPath string literal is not terminated".to_owned()),
            Some(&character) if character == quote => {
                *index += 1;
                return Ok(Token::Literal(text));
            }
            Some(&character) => {
                text.push(character);
                *index += 1;
            }
        }
    }
}

fn read_number(characters: &[char], index: &mut usize) -> Result<Token, String> {
    let start = *index;
    while characters.get(*index).is_some_and(char::is_ascii_digit) {
        *index += 1;
    }
    if characters.get(*index) == Some(&'.') {
        *index += 1;
        while characters.get(*index).is_some_and(char::is_ascii_digit) {
            *index += 1;
        }
    }
    let text: String = characters
        .get(start..*index)
        .unwrap_or_default()
        .iter()
        .collect();
    text.parse::<f64>()
        .map(Token::Number)
        .map_err(|_| format!("`{text}` is not an XPath number"))
}

/// Reads an `NCName`, an `NCName:NCName` or an `NCName:*`.
fn read_qualified_name(
    characters: &[char],
    index: &mut usize,
) -> Result<(Option<String>, String), String> {
    let first = read_ncname(characters, index)?;
    if characters.get(*index) == Some(&':') && characters.get(*index + 1) != Some(&':') {
        *index += 1;
        if characters.get(*index) == Some(&'*') {
            *index += 1;
            return Ok((Some(first), "*".to_owned()));
        }
        let second = read_ncname(characters, index)?;
        return Ok((Some(first), second));
    }
    Ok((None, first))
}

fn read_ncname(characters: &[char], index: &mut usize) -> Result<String, String> {
    let start = *index;
    match characters.get(*index) {
        Some(&first) if is_name_start(first) => *index += 1,
        Some(&other) => return Err(format!("{other:?} cannot start an XML name")),
        None => return Err("an XML name was expected".to_owned()),
    }
    while characters.get(*index).copied().is_some_and(is_name_char) {
        *index += 1;
    }
    Ok(characters
        .get(start..*index)
        .unwrap_or_default()
        .iter()
        .collect())
}

/// The `NameStartChar` production of
/// [XML Names](https://www.w3.org/TR/xml-names/#NT-NCName), minus `:` because
/// an `NCName` cannot contain one.
fn is_name_start(character: char) -> bool {
    matches!(character,
        'A'..='Z' | '_' | 'a'..='z'
        | '\u{c0}'..='\u{d6}' | '\u{d8}'..='\u{f6}' | '\u{f8}'..='\u{2ff}'
        | '\u{370}'..='\u{37d}' | '\u{37f}'..='\u{1fff}'
        | '\u{200c}'..='\u{200d}' | '\u{2070}'..='\u{218f}'
        | '\u{2c00}'..='\u{2fef}' | '\u{3001}'..='\u{d7ff}'
        | '\u{f900}'..='\u{fdcf}' | '\u{fdf0}'..='\u{fffd}'
        | '\u{10000}'..='\u{effff}')
}

/// The `NameChar` production, minus `:`.
fn is_name_char(character: char) -> bool {
    is_name_start(character)
        || matches!(character,
            '-' | '.' | '0'..='9' | '\u{b7}'
            | '\u{300}'..='\u{36f}' | '\u{203f}'..='\u{2040}')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(input: &str) -> Vec<Token> {
        tokenize(input).expect("the test expression tokenizes")
    }

    #[test]
    fn a_star_is_a_name_test_or_a_multiplication_depending_on_context() {
        assert_eq!(tokens("*"), [Token::NameTestStar]);
        assert_eq!(
            tokens("a * b"),
            [
                Token::Name {
                    prefix: None,
                    local: "a".to_owned()
                },
                Token::Star,
                Token::Name {
                    prefix: None,
                    local: "b".to_owned()
                }
            ]
        );
        assert_eq!(
            tokens("a[*]"),
            [
                Token::Name {
                    prefix: None,
                    local: "a".to_owned()
                },
                Token::LeftBracket,
                Token::NameTestStar,
                Token::RightBracket
            ]
        );
    }

    #[test]
    fn div_can_be_both_an_element_name_and_an_operator() {
        assert_eq!(
            tokens("div div div"),
            [
                Token::Name {
                    prefix: None,
                    local: "div".to_owned()
                },
                Token::OperatorName("div"),
                Token::Name {
                    prefix: None,
                    local: "div".to_owned()
                }
            ]
        );
    }

    #[test]
    fn a_name_before_a_parenthesis_is_a_function() {
        assert_eq!(
            tokens("comment()"),
            [
                Token::FunctionName {
                    prefix: None,
                    local: "comment".to_owned()
                },
                Token::LeftParen,
                Token::RightParen
            ]
        );
        assert_eq!(
            tokens("ancestor-or-self :: node ()").first(),
            Some(&Token::AxisName("ancestor-or-self".to_owned()))
        );
    }

    #[test]
    fn prefixed_wildcards_and_names_are_distinguished() {
        assert_eq!(tokens("ds:*"), [Token::NameTestPrefixStar("ds".to_owned())]);
        assert_eq!(
            tokens("ds:Signature"),
            [Token::Name {
                prefix: Some("ds".to_owned()),
                local: "Signature".to_owned()
            }]
        );
    }

    #[test]
    fn numbers_literals_and_variables() {
        assert_eq!(tokens("1"), [Token::Number(1.0)]);
        assert_eq!(tokens(".5"), [Token::Number(0.5)]);
        assert_eq!(tokens("1.25"), [Token::Number(1.25)]);
        assert_eq!(tokens("'a\"b'"), [Token::Literal("a\"b".to_owned())]);
        assert_eq!(
            tokens("$x"),
            [Token::Variable {
                prefix: None,
                local: "x".to_owned()
            }]
        );
    }

    #[test]
    fn abbreviations_tokenize() {
        assert_eq!(
            tokens("//@a/../."),
            [
                Token::SlashSlash,
                Token::At,
                Token::Name {
                    prefix: None,
                    local: "a".to_owned()
                },
                Token::Slash,
                Token::DotDot,
                Token::Slash,
                Token::Dot
            ]
        );
    }

    #[test]
    fn an_unterminated_literal_is_an_error() {
        assert!(tokenize("'abc").is_err());
        assert!(tokenize("a ! b").is_err());
        assert!(tokenize("a : b").is_err());
    }
}
