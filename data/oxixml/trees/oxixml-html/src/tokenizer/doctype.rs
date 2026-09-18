//! The DOCTYPE states
//! ([HTML § 13.2.5.53](https://html.spec.whatwg.org/multipage/parsing.html#doctype-state)
//! through
//! [§ 13.2.5.68](https://html.spec.whatwg.org/multipage/parsing.html#bogus-doctype-state)).

use super::data::REPLACEMENT;
use super::{State, Tokenizer, is_html_whitespace};
use crate::error::codes;
use crate::token::Doctype;

/// Which of the two DOCTYPE identifiers a quoted-identifier state is filling
/// in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Identifier {
    Public,
    System,
}

impl Tokenizer<'_> {
    pub(super) fn step_doctype(&mut self) {
        match self.state {
            State::Doctype => self.doctype_state(),
            State::BeforeDoctypeName => self.before_doctype_name_state(),
            State::DoctypeName => self.doctype_name_state(),
            State::AfterDoctypeName => self.after_doctype_name_state(),
            State::AfterDoctypePublicKeyword => self.after_doctype_public_keyword_state(),
            State::BeforeDoctypePublicIdentifier => self.before_doctype_public_identifier_state(),
            State::DoctypePublicIdentifierDoubleQuoted => {
                self.doctype_identifier_state(Identifier::Public, '"');
            }
            State::DoctypePublicIdentifierSingleQuoted => {
                self.doctype_identifier_state(Identifier::Public, '\'');
            }
            State::AfterDoctypePublicIdentifier => self.after_doctype_public_identifier_state(),
            State::BetweenDoctypePublicAndSystemIdentifiers => {
                self.between_doctype_public_and_system_identifiers_state();
            }
            State::AfterDoctypeSystemKeyword => self.after_doctype_system_keyword_state(),
            State::BeforeDoctypeSystemIdentifier => self.before_doctype_system_identifier_state(),
            State::DoctypeSystemIdentifierDoubleQuoted => {
                self.doctype_identifier_state(Identifier::System, '"');
            }
            State::DoctypeSystemIdentifierSingleQuoted => {
                self.doctype_identifier_state(Identifier::System, '\'');
            }
            State::AfterDoctypeSystemIdentifier => self.after_doctype_system_identifier_state(),
            State::BogusDoctype => self.bogus_doctype_state(),
            _ => unreachable!("step_doctype dispatched a state it does not own"),
        }
    }

    /// Emits the DOCTYPE token with the force-quirks flag set, then end of
    /// file — the recovery every DOCTYPE state shares.
    fn doctype_eof(&mut self) {
        self.error(codes::EOF_IN_DOCTYPE);
        self.doctype.force_quirks = true;
        self.emit_doctype();
        self.emit_eof();
    }

    /// Emits the DOCTYPE token and returns to the data state.
    fn doctype_close(&mut self) {
        self.state = State::Data;
        self.emit_doctype();
    }

    fn new_doctype(&mut self) {
        self.doctype = Doctype::default();
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#doctype-state>
    fn doctype_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => self.state = State::BeforeDoctypeName,
            Some('>') => {
                self.unconsume('>');
                self.state = State::BeforeDoctypeName;
            }
            Some(ch) => {
                self.error(codes::MISSING_WHITESPACE_BEFORE_DOCTYPE_NAME);
                self.unconsume(ch);
                self.state = State::BeforeDoctypeName;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.new_doctype();
                self.doctype_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#before-doctype-name-state>
    fn before_doctype_name_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('\0') => {
                self.error(codes::UNEXPECTED_NULL_CHARACTER);
                self.new_doctype();
                self.doctype.name = Some(REPLACEMENT.to_string());
                self.state = State::DoctypeName;
            }
            Some('>') => {
                self.error(codes::MISSING_DOCTYPE_NAME);
                self.new_doctype();
                self.doctype.force_quirks = true;
                self.doctype_close();
            }
            Some(ch) => {
                self.new_doctype();
                self.doctype.name = Some(ch.to_ascii_lowercase().to_string());
                self.state = State::DoctypeName;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.new_doctype();
                self.doctype_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#doctype-name-state>
    fn doctype_name_state(&mut self) {
        loop {
            match self.consume() {
                Some(ch) if is_html_whitespace(ch) => {
                    self.state = State::AfterDoctypeName;
                    return;
                }
                Some('>') => {
                    self.doctype_close();
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.doctype
                        .name
                        .get_or_insert_with(String::new)
                        .push(REPLACEMENT);
                }
                Some(ch) => self
                    .doctype
                    .name
                    .get_or_insert_with(String::new)
                    .push(ch.to_ascii_lowercase()),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.doctype_eof();
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-doctype-name-state>
    fn after_doctype_name_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('>') => self.doctype_close(),
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.unconsume(ch);
                // `PUBLIC` and `SYSTEM` are six bytes each, counted from the
                // character just put back.
                if self.needs_lookahead(6) {
                    return;
                }
                if self.consume_if_match("PUBLIC", true) {
                    self.state = State::AfterDoctypePublicKeyword;
                } else if self.consume_if_match("SYSTEM", true) {
                    self.state = State::AfterDoctypeSystemKeyword;
                } else {
                    self.error(codes::INVALID_CHARACTER_SEQUENCE_AFTER_DOCTYPE_NAME);
                    self.doctype.force_quirks = true;
                    self.state = State::BogusDoctype;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-doctype-public-keyword-state>
    fn after_doctype_public_keyword_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {
                self.state = State::BeforeDoctypePublicIdentifier;
            }
            Some('"') => {
                self.error(codes::MISSING_WHITESPACE_AFTER_DOCTYPE_PUBLIC_KEYWORD);
                self.doctype.public_id = Some(String::new());
                self.state = State::DoctypePublicIdentifierDoubleQuoted;
            }
            Some('\'') => {
                self.error(codes::MISSING_WHITESPACE_AFTER_DOCTYPE_PUBLIC_KEYWORD);
                self.doctype.public_id = Some(String::new());
                self.state = State::DoctypePublicIdentifierSingleQuoted;
            }
            Some('>') => {
                self.error(codes::MISSING_DOCTYPE_PUBLIC_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.doctype_close();
            }
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.error(codes::MISSING_QUOTE_BEFORE_DOCTYPE_PUBLIC_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.unconsume(ch);
                self.state = State::BogusDoctype;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#before-doctype-public-identifier-state>
    fn before_doctype_public_identifier_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('"') => {
                self.doctype.public_id = Some(String::new());
                self.state = State::DoctypePublicIdentifierDoubleQuoted;
            }
            Some('\'') => {
                self.doctype.public_id = Some(String::new());
                self.state = State::DoctypePublicIdentifierSingleQuoted;
            }
            Some('>') => {
                self.error(codes::MISSING_DOCTYPE_PUBLIC_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.doctype_close();
            }
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.error(codes::MISSING_QUOTE_BEFORE_DOCTYPE_PUBLIC_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.unconsume(ch);
                self.state = State::BogusDoctype;
            }
        }
    }

    /// The four quoted-identifier states, which differ only in which identifier
    /// they fill in and which quote closes them.
    fn doctype_identifier_state(&mut self, identifier: Identifier, quote: char) {
        loop {
            match self.consume() {
                Some(ch) if ch == quote => {
                    self.state = match identifier {
                        Identifier::Public => State::AfterDoctypePublicIdentifier,
                        Identifier::System => State::AfterDoctypeSystemIdentifier,
                    };
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.identifier_mut(identifier).push(REPLACEMENT);
                }
                Some('>') => {
                    self.error(match identifier {
                        Identifier::Public => codes::ABRUPT_DOCTYPE_PUBLIC_IDENTIFIER,
                        Identifier::System => codes::ABRUPT_DOCTYPE_SYSTEM_IDENTIFIER,
                    });
                    self.doctype.force_quirks = true;
                    self.doctype_close();
                    return;
                }
                Some(ch) => self.identifier_mut(identifier).push(ch),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.doctype_eof();
                    return;
                }
            }
        }
    }

    fn identifier_mut(&mut self, identifier: Identifier) -> &mut String {
        match identifier {
            Identifier::Public => self.doctype.public_id.get_or_insert_with(String::new),
            Identifier::System => self.doctype.system_id.get_or_insert_with(String::new),
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-doctype-public-identifier-state>
    fn after_doctype_public_identifier_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {
                self.state = State::BetweenDoctypePublicAndSystemIdentifiers;
            }
            Some('>') => self.doctype_close(),
            Some('"') => {
                self.error(codes::MISSING_WHITESPACE_BETWEEN_DOCTYPE_PUBLIC_AND_SYSTEM_IDENTIFIERS);
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierDoubleQuoted;
            }
            Some('\'') => {
                self.error(codes::MISSING_WHITESPACE_BETWEEN_DOCTYPE_PUBLIC_AND_SYSTEM_IDENTIFIERS);
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierSingleQuoted;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.error(codes::MISSING_QUOTE_BEFORE_DOCTYPE_SYSTEM_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.unconsume(ch);
                self.state = State::BogusDoctype;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#between-doctype-public-and-system-identifiers-state>
    fn between_doctype_public_and_system_identifiers_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('>') => self.doctype_close(),
            Some('"') => {
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierDoubleQuoted;
            }
            Some('\'') => {
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierSingleQuoted;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.error(codes::MISSING_QUOTE_BEFORE_DOCTYPE_SYSTEM_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.unconsume(ch);
                self.state = State::BogusDoctype;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-doctype-system-keyword-state>
    fn after_doctype_system_keyword_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {
                self.state = State::BeforeDoctypeSystemIdentifier;
            }
            Some('"') => {
                self.error(codes::MISSING_WHITESPACE_AFTER_DOCTYPE_SYSTEM_KEYWORD);
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierDoubleQuoted;
            }
            Some('\'') => {
                self.error(codes::MISSING_WHITESPACE_AFTER_DOCTYPE_SYSTEM_KEYWORD);
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierSingleQuoted;
            }
            Some('>') => {
                self.error(codes::MISSING_DOCTYPE_SYSTEM_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.doctype_close();
            }
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.error(codes::MISSING_QUOTE_BEFORE_DOCTYPE_SYSTEM_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.unconsume(ch);
                self.state = State::BogusDoctype;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#before-doctype-system-identifier-state>
    fn before_doctype_system_identifier_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('"') => {
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierDoubleQuoted;
            }
            Some('\'') => {
                self.doctype.system_id = Some(String::new());
                self.state = State::DoctypeSystemIdentifierSingleQuoted;
            }
            Some('>') => {
                self.error(codes::MISSING_DOCTYPE_SYSTEM_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.doctype_close();
            }
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.error(codes::MISSING_QUOTE_BEFORE_DOCTYPE_SYSTEM_IDENTIFIER);
                self.doctype.force_quirks = true;
                self.unconsume(ch);
                self.state = State::BogusDoctype;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-doctype-system-identifier-state>
    fn after_doctype_system_identifier_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('>') => self.doctype_close(),
            None => {
                if self.suspended {
                    return;
                }
                self.doctype_eof();
            }
            Some(ch) => {
                self.error(codes::UNEXPECTED_CHARACTER_AFTER_DOCTYPE_SYSTEM_IDENTIFIER);
                self.unconsume(ch);
                self.state = State::BogusDoctype;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#bogus-doctype-state>
    fn bogus_doctype_state(&mut self) {
        loop {
            match self.consume() {
                Some('>') => {
                    self.doctype_close();
                    return;
                }
                Some('\0') => self.error(codes::UNEXPECTED_NULL_CHARACTER),
                Some(_) => {}
                None => {
                    if self.suspended {
                        return;
                    }
                    self.emit_doctype();
                    self.emit_eof();
                    return;
                }
            }
        }
    }
}
