//! Attribute value templates — the `{...}` expressions that may appear inside an attribute.
//!
//! [XSLT 3.0 §5.5](https://www.w3.org/TR/xslt-30/#attribute-value-templates) specifies the
//! syntax exactly: `{{` and `}}` are literal braces, an unescaped `{` opens an expression, and
//! the expression runs to the matching `}`, with braces inside a string literal ignored. A
//! stray `}` outside an expression is a static error.

use crate::error::{XsltError, XsltResult};

/// One piece of a parsed attribute value template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AvtPiece {
    /// Literal text, with `{{` and `}}` already reduced.
    Literal(String),
    /// The source text of an embedded expression, without its braces.
    Expression(String),
}

/// Splits an attribute value into its literal and expression pieces.
///
/// # Errors
///
/// `XTSE0350` for an unmatched `{`, `XTSE0370` for an unescaped `}`.
pub(crate) fn parse(value: &str) -> XsltResult<Vec<AvtPiece>> {
    let mut pieces = Vec::new();
    let mut literal = String::new();
    let characters: Vec<char> = value.chars().collect();
    let mut index = 0;
    while index < characters.len() {
        match characters[index] {
            '{' if characters.get(index + 1) == Some(&'{') => {
                literal.push('{');
                index += 2;
            }
            '}' if characters.get(index + 1) == Some(&'}') => {
                literal.push('}');
                index += 2;
            }
            '}' => {
                return Err(XsltError::stat1c(
                    "XTSE0370",
                    "an unescaped `}` in an attribute value template must be written `}}`",
                ));
            }
            '{' => {
                if !literal.is_empty() {
                    pieces.push(AvtPiece::Literal(std::mem::take(&mut literal)));
                }
                let (expression, next) = scan_expression(&characters, index + 1)?;
                pieces.push(AvtPiece::Expression(expression));
                index = next;
            }
            other => {
                literal.push(other);
                index += 1;
            }
        }
    }
    if !literal.is_empty() {
        pieces.push(AvtPiece::Literal(literal));
    }
    Ok(pieces)
}

/// Reads the expression that starts at `start`, returning it and the index just past its `}`.
///
/// Three constructs of the embedded expression hide a brace from the scan, and all three are the
/// expression's own lexical rules rather than the template's: a string literal, a nested
/// `{`…`}` (a map constructor, an inline function body), and an
/// [XPath comment](https://www.w3.org/TR/xpath-31/#comments), whose `(:`…`:)` delimiters nest and
/// whose content is not tokenized at all. The suite writes each of them: `cvt-024` puts a `}` in
/// a string literal, `cvt-022` a map constructor, and `cvt-023`
/// `{(:implausible } :) 'Rings'}` — a `}` inside a comment, which used to end the expression and
/// leave the real terminator looking like the stray `}` of `XTSE0370`.
fn scan_expression(characters: &[char], start: usize) -> XsltResult<(String, usize)> {
    let mut depth = 1_usize;
    let mut index = start;
    let mut expression = String::new();
    let mut quote: Option<char> = None;
    // Nesting depth of `(:`…`:)`; zero outside a comment.
    let mut comment = 0_usize;
    while index < characters.len() {
        let character = characters[index];
        if comment > 0 {
            expression.push(character);
            if character == '(' && characters.get(index + 1) == Some(&':') {
                comment += 1;
                expression.push(':');
                index += 2;
                continue;
            }
            if character == ':' && characters.get(index + 1) == Some(&')') {
                comment -= 1;
                expression.push(')');
                index += 2;
                continue;
            }
            index += 1;
            continue;
        }
        match quote {
            Some(open) => {
                expression.push(character);
                if character == open {
                    // A doubled quote inside an XPath string literal is an escaped quote.
                    if characters.get(index + 1) == Some(&open) {
                        expression.push(open);
                        index += 1;
                    } else {
                        quote = None;
                    }
                }
            }
            None => match character {
                '(' if characters.get(index + 1) == Some(&':') => {
                    comment = 1;
                    expression.push('(');
                    expression.push(':');
                    index += 2;
                    continue;
                }
                '\'' | '"' => {
                    quote = Some(character);
                    expression.push(character);
                }
                '{' => {
                    depth += 1;
                    expression.push(character);
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((expression, index + 1));
                    }
                    expression.push(character);
                }
                other => expression.push(other),
            },
        }
        index += 1;
    }
    Err(XsltError::stat1c(
        "XTSE0350",
        "an attribute value template has a `{` with no matching `}`",
    ))
}

/// Whether the value contains anything that needs evaluating.
pub(crate) fn is_constant(pieces: &[AvtPiece]) -> bool {
    pieces
        .iter()
        .all(|piece| matches!(piece, AvtPiece::Literal(_)))
}

/// The literal text of a template that has no expressions.
pub(crate) fn literal_text(pieces: &[AvtPiece]) -> String {
    pieces
        .iter()
        .map(|piece| match piece {
            AvtPiece::Literal(text) => text.as_str(),
            AvtPiece::Expression(_) => "",
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_one_literal() -> XsltResult<()> {
        assert_eq!(parse("abc")?, vec![AvtPiece::Literal("abc".to_owned())]);
        assert!(is_constant(&parse("abc")?));
        Ok(())
    }

    #[test]
    fn doubled_braces_are_literal() -> XsltResult<()> {
        assert_eq!(
            parse("a{{b}}c")?,
            vec![AvtPiece::Literal("a{b}c".to_owned())]
        );
        Ok(())
    }

    #[test]
    fn expressions_are_extracted() -> XsltResult<()> {
        assert_eq!(
            parse("x{@a}y{1 + 2}")?,
            vec![
                AvtPiece::Literal("x".to_owned()),
                AvtPiece::Expression("@a".to_owned()),
                AvtPiece::Literal("y".to_owned()),
                AvtPiece::Expression("1 + 2".to_owned()),
            ]
        );
        Ok(())
    }

    #[test]
    fn braces_inside_a_string_literal_are_ignored() -> XsltResult<()> {
        assert_eq!(
            parse("{concat('{', '}')}")?,
            vec![AvtPiece::Expression("concat('{', '}')".to_owned())]
        );
        Ok(())
    }

    #[test]
    fn nested_braces_belong_to_the_expression() -> XsltResult<()> {
        assert_eq!(
            parse("{map{'a':1}?a}")?,
            vec![AvtPiece::Expression("map{'a':1}?a".to_owned())]
        );
        Ok(())
    }

    /// `cvt-023`: a `}` inside an XPath comment is part of the comment, and the comment's own
    /// delimiters nest.
    #[test]
    fn braces_inside_an_xpath_comment_are_ignored() -> XsltResult<()> {
        assert_eq!(
            parse("{(:implausible } :) 'Rings'}")?,
            vec![AvtPiece::Expression(
                "(:implausible } :) 'Rings'".to_owned()
            )]
        );
        assert_eq!(
            parse("{(: a (: } :) b :)1}")?,
            vec![AvtPiece::Expression("(: a (: } :) b :)1".to_owned())]
        );
        // A comment opener inside a string literal is text, not a comment.
        assert_eq!(
            parse("{'(:'}")?,
            vec![AvtPiece::Expression("'(:'".to_owned())]
        );
        Ok(())
    }

    #[test]
    fn unbalanced_braces_are_static_errors() {
        assert_eq!(parse("{a").expect_err("unclosed").code(), Some("XTSE0350"));
        assert_eq!(parse("a}").expect_err("stray").code(), Some("XTSE0370"));
    }
}
