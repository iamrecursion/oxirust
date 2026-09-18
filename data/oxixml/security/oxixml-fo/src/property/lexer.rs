//! Tokenizer for the XSL property value expression language.
//!
//! The token set and the disambiguation rules follow
//! [XSL 1.1 §5.9.11](https://www.w3.org/TR/xsl11/#d0e6017) exactly:
//!
//! * an NCName immediately (or after whitespace) followed by `(` is a function
//!   name;
//! * a number ends at the first character that is neither a digit nor `.`, so a
//!   unit name can follow without intervening whitespace;
//! * an NCName immediately following a number is a unit name;
//! * an NCName following a *numeric* is an operator name (`div` or `mod`).

use crate::error::{FoError, Result};

/// A binary operator of the expression language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operator {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `div`
    Divide,
    /// `mod`
    Modulo,
}

/// A single token of a property value expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token {
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `,`
    Comma,
    /// An operator.
    Operator(Operator),
    /// A number with an optional unit name, e.g. `12` or `1.5em`.
    Numeric(f64, Option<String>),
    /// A percentage, e.g. `110%`.
    Percent(f64),
    /// A hexadecimal colour, e.g. `#f0f0f0`.
    Color(String),
    /// A quoted string literal.
    Literal(String),
    /// An NCName used as a function name (it is followed by `(`).
    Function(String),
    /// An NCName used as a keyword or enumeration token.
    Name(String),
}

fn is_ncname_start(c: char) -> bool {
    c == '_'
        || c.is_alphabetic()
        || matches!(c as u32,
            0xC0..=0xD6 | 0xD8..=0xF6 | 0xF8..=0x2FF | 0x370..=0x37D | 0x37F..=0x1FFF
            | 0x200C..=0x200D | 0x2070..=0x218F | 0x2C00..=0x2FEF | 0x3001..=0xD7FF
            | 0xF900..=0xFDCF | 0xFDF0..=0xFFFD | 0x10000..=0xEFFFF)
}

fn is_ncname_char(c: char) -> bool {
    is_ncname_start(c)
        || c == '-'
        || c == '.'
        || c.is_ascii_digit()
        || matches!(c as u32, 0xB7 | 0x300..=0x36F | 0x203F..=0x2040)
}

/// Tokenizes a property value expression.
///
/// `property` is used only for error reporting.
pub(crate) fn tokenize(property: &str, input: &str) -> Result<Vec<Token>> {
    let mut tokens: Vec<Token> = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut index = 0usize;

    while index < chars.len() {
        let current = chars[index];
        if current.is_whitespace() {
            index += 1;
            continue;
        }
        match current {
            '(' => {
                tokens.push(Token::LeftParen);
                index += 1;
            }
            ')' => {
                tokens.push(Token::RightParen);
                index += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                index += 1;
            }
            '+' => {
                tokens.push(Token::Operator(Operator::Add));
                index += 1;
            }
            '-' => {
                tokens.push(Token::Operator(Operator::Subtract));
                index += 1;
            }
            '*' => {
                tokens.push(Token::Operator(Operator::Multiply));
                index += 1;
            }
            '#' => {
                let start = index;
                index += 1;
                while index < chars.len() && chars[index].is_ascii_hexdigit() {
                    index += 1;
                }
                if index == start + 1 {
                    return Err(FoError::expression(property, "empty colour literal `#`"));
                }
                tokens.push(Token::Color(chars[start..index].iter().collect()));
            }
            '"' | '\'' => {
                let quote = current;
                index += 1;
                let start = index;
                while index < chars.len() && chars[index] != quote {
                    index += 1;
                }
                if index >= chars.len() {
                    return Err(FoError::expression(property, "unterminated string literal"));
                }
                tokens.push(Token::Literal(chars[start..index].iter().collect()));
                index += 1;
            }
            '.' | '0'..='9' => {
                let start = index;
                while index < chars.len() && (chars[index].is_ascii_digit() || chars[index] == '.')
                {
                    index += 1;
                }
                let text: String = chars[start..index].iter().collect();
                let number: f64 = text.parse().map_err(|_| {
                    FoError::expression(property, format!("`{text}` is not a number"))
                })?;
                if index < chars.len() && chars[index] == '%' {
                    index += 1;
                    tokens.push(Token::Percent(number));
                } else if index < chars.len() && is_ncname_start(chars[index]) {
                    let unit_start = index;
                    while index < chars.len() && is_ncname_char(chars[index]) {
                        index += 1;
                    }
                    let unit: String = chars[unit_start..index].iter().collect();
                    tokens.push(Token::Numeric(number, Some(unit)));
                } else {
                    tokens.push(Token::Numeric(number, None));
                }
            }
            c if is_ncname_start(c) => {
                let start = index;
                while index < chars.len() && is_ncname_char(chars[index]) {
                    index += 1;
                }
                let name: String = chars[start..index].iter().collect();
                let mut lookahead = index;
                while lookahead < chars.len() && chars[lookahead].is_whitespace() {
                    lookahead += 1;
                }
                let followed_by_paren = chars.get(lookahead) == Some(&'(');
                if followed_by_paren {
                    tokens.push(Token::Function(name));
                } else if name == "div" && follows_operand(&tokens) {
                    tokens.push(Token::Operator(Operator::Divide));
                } else if name == "mod" && follows_operand(&tokens) {
                    tokens.push(Token::Operator(Operator::Modulo));
                } else {
                    tokens.push(Token::Name(name));
                }
            }
            other => {
                return Err(FoError::expression(
                    property,
                    format!("unexpected character `{other}`"),
                ));
            }
        }
    }
    Ok(tokens)
}

/// Whether the previous token can end an operand, which decides whether `div`
/// and `mod` are operator names or enumeration tokens.
fn follows_operand(tokens: &[Token]) -> bool {
    matches!(
        tokens.last(),
        Some(
            Token::Numeric(..)
                | Token::Percent(_)
                | Token::RightParen
                | Token::Name(_)
                | Token::Color(_)
                | Token::Literal(_)
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(input: &str) -> Vec<Token> {
        tokenize("test", input).expect("tokenizes")
    }

    #[test]
    fn numbers_terminate_before_unit_names() {
        assert_eq!(
            lex("10pt"),
            vec![Token::Numeric(10.0, Some("pt".to_owned()))]
        );
        assert_eq!(
            lex("10pt-2pt"),
            vec![Token::Numeric(10.0, Some("pt-2pt".to_owned()))],
            "XSL 1.1 §5.9.5 note: `10pt-2pt` lexes as a single unit name"
        );
        assert_eq!(
            lex("10pt - 2pt"),
            vec![
                Token::Numeric(10.0, Some("pt".to_owned())),
                Token::Operator(Operator::Subtract),
                Token::Numeric(2.0, Some("pt".to_owned())),
            ]
        );
    }

    #[test]
    fn function_names_are_recognised_by_the_following_paren() {
        assert_eq!(
            lex("from-parent(start-indent)"),
            vec![
                Token::Function("from-parent".to_owned()),
                Token::LeftParen,
                Token::Name("start-indent".to_owned()),
                Token::RightParen,
            ]
        );
    }

    #[test]
    fn div_and_mod_are_operators_only_after_an_operand() {
        assert_eq!(
            lex("5 div 2"),
            vec![
                Token::Numeric(5.0, None),
                Token::Operator(Operator::Divide),
                Token::Numeric(2.0, None),
            ]
        );
        assert_eq!(lex("mod"), vec![Token::Name("mod".to_owned())]);
    }

    #[test]
    fn percentages_and_colours() {
        assert_eq!(lex("110%"), vec![Token::Percent(110.0)]);
        assert_eq!(lex("#f0f"), vec![Token::Color("#f0f".to_owned())]);
    }

    #[test]
    fn unterminated_literals_are_rejected() {
        assert!(tokenize("test", "\"abc").is_err());
        assert!(tokenize("test", "#").is_err());
        assert!(tokenize("test", "@").is_err());
    }
}
