//! Parsing of the `children` and `Mixed` content-model productions.
//!
//! Implements XML 1.0 §3.2.1 `[47]`–`[50]` (`children`, `cp`, `choice`, `seq`)
//! and §3.2.2 `[51]` (`Mixed`). The opening `(` and, for mixed content, the
//! `#PCDATA` keyword have already been consumed by the caller.

use crate::error::{DtdError, DtdErrorKind};
use crate::model::{ContentParticle, ContentSpec, MixedContent, Occurrence};
use crate::parser::subset::SubsetParser;

/// The greatest nesting depth of parenthesised groups an element content model
/// may reach before [`DtdErrorKind::ContentModelDepth`] is raised.
///
/// XML 1.0 sets no limit, but every real content model is a handful of levels
/// deep; the cap exists so that a hostile `<!ELEMENT e ((((…))))>` becomes a
/// clean error rather than a stack overflow. It is well below the depth at
/// which any supported platform's stack runs out, in debug builds included.
pub const MAX_CONTENT_MODEL_DEPTH: usize = 256;

/// The greatest number of child-element positions (leaf particles) a `children` content
/// model may name before [`DtdErrorKind::ContentModelPositions`] is raised.
///
/// [`MAX_CONTENT_MODEL_DEPTH`] only bounds parenthesis-*nesting* depth, which guards the
/// recursive-descent parser's own call stack; it does nothing for a single, flat, repeatable
/// group naming thousands of distinct children (`(a|b|…|z9999)+`), which sits at nesting depth
/// one. That shape turns out to be cubic — not merely superlinear — in leaf count to compile:
/// every leaf's `follow` set ends up containing every other leaf (from the repetition), so
/// subsequently determinizing the automaton (`PositionAutomaton::determinize` in
/// [`crate::automaton`]) creates one DFA state per leaf and does a linear scan of a
/// linear-sized follow set for each of a linear number of candidate symbols at every one of
/// those states. Measured directly against this exact worst-case shape (a debug build, on a
/// machine shared with concurrent builds, so the numbers themselves are illustrative rather
/// than a precise guarantee — but the cubic growth is reproducible): a choice group this wide
/// took on the order of a hundred milliseconds to compile, twice as wide took roughly ten
/// times as long, and four times as wide took roughly eighty times as long, one single
/// ordinary-looking declaration each time. The DFA-state cap of the internal subset
/// construction does not help here: this shape never creates more states than it has leaves,
/// far below that cap, so the full cubic cost is paid before it could ever trigger. No
/// legitimate content model comes close to this limit — the widest choice group in the W3C XML
/// conformance test suite names seven alternatives — so this cap keeps an ordinary DTD entirely
/// unaffected while keeping the cost of one hostile declaration to, worst case, low double-digit
/// milliseconds even in an unoptimized build.
pub const MAX_CONTENT_MODEL_POSITIONS: usize = 128;

impl SubsetParser<'_> {
    /// Parses the remainder of a `Mixed` content model, after `'(' S? '#PCDATA'`.
    ///
    /// ```text
    /// Mixed ::= '(' S? '#PCDATA' (S? '|' S? Name)* S? ')*'
    ///         | '(' S? '#PCDATA' S? ')'
    /// ```
    pub(crate) fn parse_mixed_tail(&mut self, open_frame: u64) -> Result<ContentSpec, DtdError> {
        let mut names: Vec<String> = Vec::new();
        loop {
            self.skip_whitespace(true)?;
            match self.scanner.peek() {
                Some(')') => {
                    self.scanner.next_char();
                    self.note_group_nesting(open_frame);
                    if names.is_empty() {
                        // `(#PCDATA)` and `(#PCDATA)*` are both legal.
                        self.scanner.try_consume("*");
                        return Ok(ContentSpec::Mixed(MixedContent::pcdata_only()));
                    }
                    if !self.scanner.try_consume("*") {
                        let found = self.describe_next();
                        return Err(self.error(DtdErrorKind::Unexpected {
                            expected: "`*` after a mixed content model with element types"
                                .to_owned(),
                            found,
                        }));
                    }
                    return Ok(ContentSpec::Mixed(MixedContent::new(names)));
                }
                Some('|') => {
                    self.scanner.next_char();
                    self.skip_whitespace(true)?;
                    names.push(self.read_name()?);
                }
                _ => {
                    let found = self.describe_next();
                    return Err(self.error(DtdErrorKind::Unexpected {
                        expected: "`|` or `)`".to_owned(),
                        found,
                    }));
                }
            }
        }
    }

    /// Parses the remainder of a `choice` or `seq`, after its opening `(` and
    /// any following white space.
    ///
    /// ```text
    /// choice ::= '(' S? cp ( S? '|' S? cp )+ S? ')'
    /// seq    ::= '(' S? cp ( S? ',' S? cp )* S? ')'
    /// ```
    ///
    /// A group with a single member and no separator is a `seq` of one.
    pub(crate) fn parse_group_tail(
        &mut self,
        open_frame: u64,
        depth: usize,
    ) -> Result<ContentParticle, DtdError> {
        // The grammar is mutually recursive (`cp` may be a `choice`/`seq`,
        // whose members are `cp`s again), so a hostile declaration nesting
        // parentheses tens of thousands deep would otherwise exhaust the
        // stack. This is the same reasoning as the entity-expansion budget.
        if depth > MAX_CONTENT_MODEL_DEPTH {
            return Err(self.error(DtdErrorKind::ContentModelDepth(MAX_CONTENT_MODEL_DEPTH)));
        }
        let mut particles = vec![self.parse_content_particle(depth)?];
        self.skip_whitespace(true)?;

        let separator = match self.scanner.peek() {
            Some(c @ ('|' | ',')) => Some(c),
            Some(')') => None,
            _ => {
                let found = self.describe_next();
                return Err(self.error(DtdErrorKind::Unexpected {
                    expected: "`|`, `,` or `)`".to_owned(),
                    found,
                }));
            }
        };

        if let Some(separator) = separator {
            while self.scanner.peek() == Some(separator) {
                self.scanner.next_char();
                self.skip_whitespace(true)?;
                particles.push(self.parse_content_particle(depth)?);
                self.skip_whitespace(true)?;
            }
        }

        self.expect_char(')')?;
        self.note_group_nesting(open_frame);
        let occurrence = self.read_occurrence();
        Ok(match separator {
            Some('|') => ContentParticle::Choice {
                particles,
                occurrence,
            },
            _ => ContentParticle::Sequence {
                particles,
                occurrence,
            },
        })
    }

    /// Parses one `cp ::= (Name | choice | seq) ('?' | '*' | '+')?`.
    fn parse_content_particle(&mut self, depth: usize) -> Result<ContentParticle, DtdError> {
        if self.scanner.peek() == Some('(') {
            self.scanner.next_char();
            let open_frame = self.scanner.last_frame_id();
            self.skip_whitespace(true)?;
            return self.parse_group_tail(open_frame, depth + 1);
        }
        let name = self.read_name()?;
        Ok(ContentParticle::Name {
            name,
            occurrence: self.read_occurrence(),
        })
    }

    /// VC: Proper Group/PE Nesting (XML 1.0 §3.2.1) — records that the `)` just
    /// consumed came from a different entity than its `(`.
    fn note_group_nesting(&mut self, open_frame: u64) {
        if self.scanner.last_frame_id() != open_frame {
            self.group_nesting_error = true;
        }
    }

    /// Consumes an optional `?`, `*` or `+`.
    fn read_occurrence(&mut self) -> Occurrence {
        match self.scanner.peek() {
            Some('?') => {
                self.scanner.next_char();
                Occurrence::Optional
            }
            Some('*') => {
                self.scanner.next_char();
                Occurrence::ZeroOrMore
            }
            Some('+') => {
                self.scanner.next_char();
                Occurrence::OneOrMore
            }
            _ => Occurrence::One,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::error::DtdErrorKind;
    use crate::{DtdError, DtdParser};

    fn model(text: &str) -> Result<String, DtdError> {
        let dtd = DtdParser::new().parse_internal_subset(&format!("<!ELEMENT a {text}>"))?;
        Ok(dtd
            .element("a")
            .map(|declaration| declaration.content().to_string())
            .unwrap_or_default())
    }

    #[test]
    fn keyword_models_parse() -> Result<(), DtdError> {
        assert_eq!(model("EMPTY")?, "EMPTY");
        assert_eq!(model("ANY")?, "ANY");
        Ok(())
    }

    #[test]
    fn mixed_models_parse_in_both_forms() -> Result<(), DtdError> {
        assert_eq!(model("(#PCDATA)")?, "(#PCDATA)");
        assert_eq!(model("(#PCDATA)*")?, "(#PCDATA)");
        assert_eq!(model("( #PCDATA | b | c )*")?, "(#PCDATA|b|c)*");
        Ok(())
    }

    #[test]
    fn mixed_models_with_names_require_the_star() {
        let error = model("(#PCDATA|b)").expect_err("`*` is required");
        assert!(matches!(error.kind(), DtdErrorKind::Unexpected { .. }));
    }

    #[test]
    fn nested_groups_and_occurrences_parse() -> Result<(), DtdError> {
        assert_eq!(model("(b)")?, "(b)");
        assert_eq!(model("(b, c)")?, "(b,c)");
        assert_eq!(model("(b | c)+")?, "(b|c)+");
        assert_eq!(model("(b, (c | d)*, e?)")?, "(b,(c|d)*,e?)");
        Ok(())
    }

    #[test]
    fn mixing_separators_in_one_group_is_rejected() {
        let error = model("(b, c | d)").expect_err("`,` and `|` may not be mixed");
        assert!(matches!(error.kind(), DtdErrorKind::Unexpected { .. }));
    }

    #[test]
    fn an_empty_group_is_rejected() {
        assert!(model("()").is_err());
        assert!(model("(b|)").is_err());
    }
}
