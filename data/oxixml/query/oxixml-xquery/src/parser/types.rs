//! Names and type syntax.
//!
//! A `SequenceType` is never turned into a tree here: this crate hands every type check to
//! [`oxixml_xpath`] as the operand of an `instance of` or `treat as`, so the parser only has to
//! find where a type *ends* and keep its source text. That is both less code and — because the
//! XPath compiler is the one that has to agree with itself about what `element(a, xs:string?)`
//! means — more accurate.

use super::Parser;
use crate::ast::unexpected;
use crate::error::XQueryResult;
use crate::lexer::Token;

impl Parser<'_> {
    /// Consumes an `EQName` and returns its lexical form, with a braced name spelled
    /// `Q{uri}local` so that the XPath compiler resolves it the same way.
    pub(crate) fn eqname(&mut self, what: &str) -> XQueryResult<(String, usize)> {
        match self.peek().clone() {
            Token::Name(name) => {
                let end = self.advance()?.end;
                Ok((name, end))
            }
            Token::BracedName { uri, local } => {
                let end = self.advance()?.end;
                Ok((format!("Q{{{uri}}}{local}"), end))
            }
            other => Err(self.error(unexpected(&other, what))),
        }
    }

    /// Consumes an `NCName` — a name with no prefix and no braced URI.
    pub(crate) fn ncname(&mut self, what: &str) -> XQueryResult<String> {
        match self.peek().clone() {
            Token::Name(name) if !name.contains(':') => {
                self.advance()?;
                Ok(name)
            }
            other => Err(self.error(unexpected(&other, what))),
        }
    }

    /// Consumes a `URILiteral` — a string literal naming a namespace or a location.
    pub(crate) fn uri_literal(&mut self, what: &str) -> XQueryResult<String> {
        match self.peek().clone() {
            Token::Str { value, .. } => {
                self.advance()?;
                Ok(value)
            }
            other => Err(self.error(unexpected(&other, what))),
        }
    }

    /// `TypeDeclaration ::= "as" SequenceType`, when it is present.
    pub(crate) fn optional_type_declaration(&mut self) -> XQueryResult<Option<String>> {
        if !self.at_keyword("as") {
            return Ok(None);
        }
        self.advance()?;
        Ok(Some(self.sequence_type_source()?))
    }

    /// Consumes a `SequenceType` and returns its source text, with any annotation assertions
    /// blanked out so that the text can be handed to the XPath compiler verbatim.
    pub(crate) fn sequence_type_source(&mut self) -> XQueryResult<String> {
        let start = self.offset();
        self.scan_sequence_type()?;
        Ok(self.blanked_text(start, self.last_end()))
    }

    /// Consumes an `ItemType` — a `SequenceType` with no occurrence indicator — and returns its
    /// source text. `declare context item as …` is the one place the grammar wants this.
    pub(crate) fn item_type_source(&mut self) -> XQueryResult<String> {
        let start = self.offset();
        self.scan_item_type()?;
        if matches!(self.peek(), Token::Question | Token::Star | Token::Plus) {
            return Err(self.error("an item type may not carry an occurrence indicator".to_owned()));
        }
        Ok(self.blanked_text(start, self.last_end()))
    }

    /// The source text of `start..end`, with every recorded blank range replaced by spaces.
    fn blanked_text(&self, start: usize, end: usize) -> String {
        let mut text = self.text(start, end).to_owned();
        for &(from, to) in self.blanked_ranges() {
            if from < start || to > end {
                continue;
            }
            let (from, to) = (from - start, to - start);
            // Safe because a blank range always spans whole tokens, so both ends are on
            // character boundaries, and a space is one byte wide.
            if text.is_char_boundary(from) && text.is_char_boundary(to) {
                text.replace_range(from..to, &" ".repeat(to - from));
            }
        }
        text
    }

    fn scan_sequence_type(&mut self) -> XQueryResult<()> {
        if self.at_keyword("empty-sequence") && self.next_is(&Token::LeftParen)? {
            self.advance()?;
            self.balanced_parens()?;
            return Ok(());
        }
        self.scan_item_type()?;
        if matches!(self.peek(), Token::Question | Token::Star | Token::Plus) {
            self.advance()?;
        }
        Ok(())
    }

    fn scan_item_type(&mut self) -> XQueryResult<()> {
        // `FunctionTest ::= Annotation* (AnyFunctionTest | TypedFunctionTest)`. What an
        // annotation assertion *means* is implementation-defined; this processor checks that
        // the name is not in a reserved namespace and then ignores it.
        if self.peek() == &Token::Percent {
            let names = self.annotations()?;
            self.note_type_annotations(names);
            if !self.at_keyword("function") {
                return Err(self.error(unexpected(self.peek(), "`function` after an annotation")));
            }
        }
        match self.peek().clone() {
            Token::LeftParen => {
                self.balanced_parens()?;
                Ok(())
            }
            Token::Name(name) => {
                if self.next_is(&Token::LeftParen)? {
                    self.advance()?;
                    self.balanced_parens()?;
                    if name == "function" && self.at_keyword("as") {
                        self.advance()?;
                        self.scan_sequence_type()?;
                    }
                    return Ok(());
                }
                self.advance()?;
                Ok(())
            }
            Token::BracedName { .. } => {
                self.advance()?;
                if self.peek() == &Token::LeftParen {
                    self.balanced_parens()?;
                }
                Ok(())
            }
            other => Err(self.error(unexpected(&other, "an item type"))),
        }
    }

    /// Consumes a `SingleType ::= SimpleTypeName "?"?` and returns its source text.
    pub(crate) fn single_type_source(&mut self) -> XQueryResult<String> {
        let start = self.offset();
        self.eqname("a type name")?;
        if self.peek() == &Token::Question {
            self.advance()?;
        }
        Ok(self.text(start, self.last_end()).to_owned())
    }

    /// Whether the keyword `name`, which is current, opens a computed node constructor rather
    /// than being an ordinary name test.
    pub(crate) fn starts_computed_constructor(&mut self, name: &str) -> XQueryResult<bool> {
        let state = self.state();
        self.advance()?;
        let opens = match name {
            "text" | "comment" | "document" => self.peek() == &Token::LeftBrace,
            "element" | "attribute" | "processing-instruction" | "namespace" => {
                self.peek() == &Token::LeftBrace
                    || (matches!(self.peek(), Token::Name(_) | Token::BracedName { .. })
                        && self.next_is(&Token::LeftBrace)?)
            }
            _ => false,
        };
        self.restore(state);
        Ok(opens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence_type(source: &str) -> XQueryResult<String> {
        let mut parser = Parser::new(source)?;
        parser.sequence_type_source()
    }

    #[test]
    fn sequence_types_are_scanned_whole() -> XQueryResult<()> {
        for source in [
            "xs:integer",
            "xs:integer?",
            "item()*",
            "empty-sequence()",
            "element(a, xs:string?)",
            "node()+",
            "map(xs:string, item()*)",
            "array(*)",
            "function(*)",
            "function(xs:integer) as xs:integer",
            "Q{urn:x}local*",
            "(xs:integer | xs:string)",
        ] {
            assert_eq!(sequence_type(source)?, source, "scanning `{source}`");
        }
        Ok(())
    }

    #[test]
    fn a_type_stops_before_the_next_token() -> XQueryResult<()> {
        let mut parser = Parser::new("xs:integer* return 1")?;
        assert_eq!(parser.sequence_type_source()?, "xs:integer*");
        assert!(parser.at_keyword("return"));
        Ok(())
    }

    #[test]
    fn braced_names_keep_their_spelling() -> XQueryResult<()> {
        let mut parser = Parser::new("Q{urn:x}local")?;
        assert_eq!(parser.eqname("a name")?.0, "Q{urn:x}local");
        Ok(())
    }

    #[test]
    fn constructor_lookahead_distinguishes_name_tests() -> XQueryResult<()> {
        let mut parser = Parser::new("element foo { 1 }")?;
        assert!(parser.starts_computed_constructor("element")?);
        let mut parser = Parser::new("element / b")?;
        assert!(!parser.starts_computed_constructor("element")?);
        let mut parser = Parser::new("text { 'x' }")?;
        assert!(parser.starts_computed_constructor("text")?);
        Ok(())
    }
}
