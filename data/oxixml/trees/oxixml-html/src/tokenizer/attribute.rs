//! The attribute states and the self-closing start tag state
//! ([HTML § 13.2.5.32](https://html.spec.whatwg.org/multipage/parsing.html#before-attribute-name-state)
//! through
//! [§ 13.2.5.40](https://html.spec.whatwg.org/multipage/parsing.html#self-closing-start-tag-state)).

use super::data::REPLACEMENT;
use super::{State, Tokenizer, is_html_whitespace};
use crate::error::codes;
use crate::token::Attribute;

impl Tokenizer<'_> {
    pub(super) fn step_attribute(&mut self) {
        match self.state {
            State::BeforeAttributeName => self.before_attribute_name_state(),
            State::AttributeName => self.attribute_name_state(),
            State::AfterAttributeName => self.after_attribute_name_state(),
            State::BeforeAttributeValue => self.before_attribute_value_state(),
            State::AttributeValueDoubleQuoted => self.attribute_value_quoted_state('"'),
            State::AttributeValueSingleQuoted => self.attribute_value_quoted_state('\''),
            State::AttributeValueUnquoted => self.attribute_value_unquoted_state(),
            State::AfterAttributeValueQuoted => self.after_attribute_value_quoted_state(),
            State::SelfClosingStartTag => self.self_closing_start_tag_state(),
            _ => unreachable!("step_attribute dispatched a state it does not own"),
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#before-attribute-name-state>
    fn before_attribute_name_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            // Reconsume in the after attribute name state.
            Some(ch @ ('/' | '>')) => {
                self.unconsume(ch);
                self.state = State::AfterAttributeName;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.state = State::AfterAttributeName;
            }
            Some('=') => {
                self.error(codes::UNEXPECTED_EQUALS_SIGN_BEFORE_ATTRIBUTE_NAME);
                self.finish_attribute();
                self.attribute = Some(Attribute::new("=", ""));
                self.state = State::AttributeName;
            }
            Some(ch) => {
                self.finish_attribute();
                self.attribute = Some(Attribute::default());
                self.unconsume(ch);
                self.state = State::AttributeName;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#attribute-name-state>
    fn attribute_name_state(&mut self) {
        loop {
            match self.consume() {
                Some(ch) if is_html_whitespace(ch) => {
                    self.unconsume(ch);
                    self.state = State::AfterAttributeName;
                    return;
                }
                Some(ch @ ('/' | '>')) => {
                    self.unconsume(ch);
                    self.state = State::AfterAttributeName;
                    return;
                }
                None => {
                    if self.suspended {
                        return;
                    }
                    self.state = State::AfterAttributeName;
                    return;
                }
                Some('=') => {
                    self.state = State::BeforeAttributeValue;
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.attribute_name_mut().push(REPLACEMENT);
                }
                Some(ch @ ('"' | '\'' | '<')) => {
                    self.error(codes::UNEXPECTED_CHARACTER_IN_ATTRIBUTE_NAME);
                    self.attribute_name_mut().push(ch);
                }
                Some(ch) => self.attribute_name_mut().push(ch.to_ascii_lowercase()),
            }
        }
    }

    fn attribute_name_mut(&mut self) -> &mut String {
        &mut self.attribute.get_or_insert_with(Attribute::default).name
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-attribute-name-state>
    fn after_attribute_name_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('/') => self.state = State::SelfClosingStartTag,
            Some('=') => self.state = State::BeforeAttributeValue,
            Some('>') => {
                self.state = State::Data;
                self.emit_current_tag();
            }
            Some(ch) => {
                self.finish_attribute();
                self.attribute = Some(Attribute::default());
                self.unconsume(ch);
                self.state = State::AttributeName;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_TAG);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#before-attribute-value-state>
    fn before_attribute_value_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            Some('"') => self.state = State::AttributeValueDoubleQuoted,
            Some('\'') => self.state = State::AttributeValueSingleQuoted,
            Some('>') => {
                self.error(codes::MISSING_ATTRIBUTE_VALUE);
                self.state = State::Data;
                self.emit_current_tag();
            }
            Some(ch) => {
                self.unconsume(ch);
                self.state = State::AttributeValueUnquoted;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.state = State::AttributeValueUnquoted;
            }
        }
    }

    /// The double- and single-quoted attribute value states, which differ only
    /// in the quote that closes them.
    fn attribute_value_quoted_state(&mut self, quote: char) {
        loop {
            if let Some(run) = self.take_run(&[quote, '&', '\0']) {
                self.attribute_value_mut().push_str(&run);
                continue;
            }
            match self.consume() {
                Some(ch) if ch == quote => {
                    self.state = State::AfterAttributeValueQuoted;
                    return;
                }
                Some('&') => {
                    self.return_state = self.state;
                    self.state = State::CharacterReference;
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.attribute_value_mut().push(REPLACEMENT);
                }
                Some(ch) => self.attribute_value_mut().push(ch),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::EOF_IN_TAG);
                    self.emit_eof();
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#attribute-value-(unquoted)-state>
    fn attribute_value_unquoted_state(&mut self) {
        loop {
            match self.consume() {
                Some(ch) if is_html_whitespace(ch) => {
                    self.state = State::BeforeAttributeName;
                    return;
                }
                Some('&') => {
                    self.return_state = State::AttributeValueUnquoted;
                    self.state = State::CharacterReference;
                    return;
                }
                Some('>') => {
                    self.state = State::Data;
                    self.emit_current_tag();
                    return;
                }
                Some('\0') => {
                    self.error(codes::UNEXPECTED_NULL_CHARACTER);
                    self.attribute_value_mut().push(REPLACEMENT);
                }
                Some(ch @ ('"' | '\'' | '<' | '=' | '`')) => {
                    self.error(codes::UNEXPECTED_CHARACTER_IN_UNQUOTED_ATTRIBUTE_VALUE);
                    self.attribute_value_mut().push(ch);
                }
                Some(ch) => self.attribute_value_mut().push(ch),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::EOF_IN_TAG);
                    self.emit_eof();
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-attribute-value-(quoted)-state>
    fn after_attribute_value_quoted_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => self.state = State::BeforeAttributeName,
            Some('/') => self.state = State::SelfClosingStartTag,
            Some('>') => {
                self.state = State::Data;
                self.emit_current_tag();
            }
            Some(ch) => {
                self.error(codes::MISSING_WHITESPACE_BETWEEN_ATTRIBUTES);
                self.unconsume(ch);
                self.state = State::BeforeAttributeName;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_TAG);
                self.emit_eof();
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#self-closing-start-tag-state>
    fn self_closing_start_tag_state(&mut self) {
        match self.consume() {
            Some('>') => {
                self.tag.self_closing = true;
                self.state = State::Data;
                self.emit_current_tag();
            }
            Some(ch) => {
                self.error(codes::UNEXPECTED_SOLIDUS_IN_TAG);
                self.unconsume(ch);
                self.state = State::BeforeAttributeName;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_TAG);
                self.emit_eof();
            }
        }
    }
}
