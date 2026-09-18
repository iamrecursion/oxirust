//! Contains a parser for an XML processing instruction.

use crate::errors::SyntaxError;
use crate::parser::Parser;
use crate::utils::{is_whitespace, memchr_iter};

/// A parser that search a `?>` sequence in the slice.
///
/// To use a parser create an instance of parser and [`feed`] data into it.
/// After successful search the parser will return [`Some`] with position where
/// processing instruction is ended (the position after `?>`). If search was
/// unsuccessful, a [`None`] will be returned. You typically would expect positive
/// result of search, so that you should feed new data until you get it.
///
/// NOTE: after successful match the parser does not returned to the initial
/// state and should not be used anymore. Create a new parser if you want to perform
/// new search.
///
/// # Example
///
/// ```
/// use oxixml_xml::parser::{Parser, PiParser};
///
/// let mut parser = PiParser::default();
///
/// // Parse `<?instruction with = 'some > and ?' inside?>and the text follow...`
/// // splitted into three chunks
/// assert_eq!(parser.feed(b"<?instruction"), None);
/// // ...get new chunk of data
/// assert_eq!(parser.feed(b" with = 'some > and ?"), None);
/// // ...get another chunk of data
/// assert_eq!(parser.feed(b"' inside?>and the text follow..."), Some(9));
/// //                       ^        ^
/// //                       0        9
/// ```
///
/// [`feed`]: Self::feed()
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PiParser(
    /// A flag that indicates was the `bytes` in the previous attempt to find the
    /// end ended with `?`.
    pub bool,
);

impl Parser for PiParser {
    /// Determines the end position of a processing instruction in the provided slice.
    /// Processing instruction ends on the first occurrence of `?>` which cannot be
    /// escaped.
    ///
    /// [Section 2.6]: Parameter entity references MUST NOT be recognized within
    /// processing instructions, so parser do not search for them.
    ///
    /// Returns position after the `?>` or `None` if such sequence was not found.
    ///
    /// # Parameters
    /// - `bytes`: a slice to find the end of a processing instruction.
    ///   Should contain text in ASCII-compatible encoding
    ///
    /// [Section 2.6]: https://www.w3.org/TR/xml11/#sec-pi
    #[inline]
    fn feed(&mut self, bytes: &[u8]) -> Option<usize> {
        for i in memchr_iter(b'>', bytes) {
            match i {
                0 if self.0 => return Some(0),
                // If the previous byte is `?`, then we found `?>`
                i if i > 0 && bytes[i - 1] == b'?' => return Some(i),
                _ => {}
            }
        }
        self.0 = bytes.last().copied() == Some(b'?');
        None
    }

    #[inline]
    fn eof_error(self, content: &[u8]) -> SyntaxError {
        // Check if content starts with "<?xml" followed by whitespace, '?' or end.
        // This determines whether to report an unclosed XML declaration or PI.
        //
        // quick-xml 0.41 parity: the lexer works on bytes and therefore requires
        // an ASCII-compatible encoding, so this discrimination does not fire for
        // a document encoded in UTF-16 (or any other ASCII-incompatible
        // encoding), which is reported as an unclosed PI instead. Such input has
        // to be transcoded before it reaches the lexer, which is what
        // [`DecodingReader`](crate::encoding::DecodingReader) is for. The
        // behaviour is pinned by `unclosed_utf16_xml_declaration_is_a_pi` below.
        let is_xml_decl = content.starts_with(b"<?xml")
            && content
                .get(5)
                .is_none_or(|&b| is_whitespace(b) || b == b'?');
        if is_xml_decl {
            SyntaxError::UnclosedXmlDecl
        } else {
            SyntaxError::UnclosedPI
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns `Ok(pos)` with the position in the buffer where processing
    /// instruction is ended.
    ///
    /// Returns `Err(internal_state)` if parsing is not done yet.
    fn parse_pi(bytes: &[u8], had_question_mark: bool) -> Result<usize, bool> {
        let mut parser = PiParser(had_question_mark);
        match parser.feed(bytes) {
            Some(i) => Ok(i),
            None => Err(parser.0),
        }
    }

    // Comments shows which character was seen the last before calling `feed`.
    // `x` means any character, pipe denotes start of the buffer that passed to `feed`
    #[test]
    fn pi() {
        assert_eq!(parse_pi(b"", false), Err(false)); // x|
        assert_eq!(parse_pi(b"", true), Err(false)); // ?|

        assert_eq!(parse_pi(b"?", false), Err(true)); // x|?
        assert_eq!(parse_pi(b"?", true), Err(true)); // ?|?

        assert_eq!(parse_pi(b">", false), Err(false)); // x|>
        assert_eq!(parse_pi(b">", true), Ok(0)); // ?|>

        assert_eq!(parse_pi(b"?>", false), Ok(1)); // x|?>
        assert_eq!(parse_pi(b"?>", true), Ok(1)); // ?|?>

        assert_eq!(parse_pi(b">?>", false), Ok(2)); // x|>?>
        assert_eq!(parse_pi(b">?>", true), Ok(0)); // ?|>?>
    }

    /// Pins the documented quick-xml 0.41 parity limitation of [`PiParser::eof_error`]:
    /// the discrimination between an unclosed XML declaration and an unclosed
    /// processing instruction is a byte comparison, so it only works for
    /// ASCII-compatible encodings. UTF-16 input has to be transcoded with
    /// [`DecodingReader`](crate::encoding::DecodingReader) first.
    #[test]
    fn unclosed_xml_declaration_needs_an_ascii_compatible_encoding() {
        assert_eq!(
            PiParser(false).eof_error(b"<?xml version='1.0'"),
            SyntaxError::UnclosedXmlDecl
        );
        // The same declaration in UTF-16 LE is not recognized as one.
        let utf16: Vec<u8> = "<?xml version='1.0'"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(PiParser(false).eof_error(&utf16), SyntaxError::UnclosedPI);
    }
}
