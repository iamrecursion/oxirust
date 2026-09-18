//! The character reference states
//! ([HTML § 13.2.5.72](https://html.spec.whatwg.org/multipage/parsing.html#character-reference-state)
//! through
//! [§ 13.2.5.80](https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-end-state)).

use super::{State, Tokenizer};
use crate::entities;
use crate::error::codes;

/// The replacement table of the
/// [numeric character reference end state](https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-end-state),
/// which reinterprets the windows-1252 code points that pages write as if they
/// were Unicode.
const WINDOWS_1252: [(u32, u32); 27] = [
    (0x80, 0x20ac),
    (0x82, 0x201a),
    (0x83, 0x0192),
    (0x84, 0x201e),
    (0x85, 0x2026),
    (0x86, 0x2020),
    (0x87, 0x2021),
    (0x88, 0x02c6),
    (0x89, 0x2030),
    (0x8a, 0x0160),
    (0x8b, 0x2039),
    (0x8c, 0x0152),
    (0x8e, 0x017d),
    (0x91, 0x2018),
    (0x92, 0x2019),
    (0x93, 0x201c),
    (0x94, 0x201d),
    (0x95, 0x2022),
    (0x96, 0x2013),
    (0x97, 0x2014),
    (0x98, 0x02dc),
    (0x99, 0x2122),
    (0x9a, 0x0161),
    (0x9b, 0x203a),
    (0x9c, 0x0153),
    (0x9e, 0x017e),
    (0x9f, 0x0178),
];

/// A "control" in the sense of the HTML standard: a C0 control or a code point
/// in `U+007F`–`U+009F`.
const fn is_control(code: u32) -> bool {
    code <= 0x1f || (code >= 0x7f && code <= 0x9f)
}

/// A noncharacter: `U+FDD0`–`U+FDEF` and the last two code points of every
/// plane.
const fn is_noncharacter(code: u32) -> bool {
    (code >= 0xfdd0 && code <= 0xfdef) || (code & 0xfffe) == 0xfffe
}

impl Tokenizer<'_> {
    pub(super) fn step_charref(&mut self) {
        match self.state {
            State::CharacterReference => self.character_reference_state(),
            State::NamedCharacterReference => self.named_character_reference_state(),
            State::AmbiguousAmpersand => self.ambiguous_ampersand_state(),
            State::NumericCharacterReference => self.numeric_character_reference_state(),
            State::HexadecimalCharacterReferenceStart => {
                self.numeric_start_state(true);
            }
            State::DecimalCharacterReferenceStart => self.numeric_start_state(false),
            State::HexadecimalCharacterReference => self.numeric_digits_state(true),
            State::DecimalCharacterReference => self.numeric_digits_state(false),
            State::NumericCharacterReferenceEnd => self.numeric_character_reference_end_state(),
            _ => unreachable!("step_charref dispatched a state it does not own"),
        }
    }

    /// "Flush code points consumed as a character reference": the temporary
    /// buffer goes into the attribute value when the reference was inside one,
    /// and into character tokens otherwise.
    fn flush_code_points(&mut self) {
        let buffer = std::mem::take(&mut self.temp_buffer);
        if self.return_state.is_attribute_value() {
            self.attribute_value_mut().push_str(&buffer);
        } else {
            self.emit_str(&buffer);
        }
        self.temp_buffer = buffer;
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#character-reference-state>
    fn character_reference_state(&mut self) {
        self.temp_buffer.clear();
        self.temp_buffer.push('&');
        match self.consume() {
            Some(ch) if ch.is_ascii_alphanumeric() => {
                self.unconsume(ch);
                self.state = State::NamedCharacterReference;
            }
            Some('#') => {
                self.temp_buffer.push('#');
                self.state = State::NumericCharacterReference;
            }
            other => {
                if self.suspended {
                    return;
                }
                self.flush_code_points();
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = self.return_state;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#named-character-reference-state>
    fn named_character_reference_state(&mut self) {
        // The longest match is over at most `MAX_NAME_LEN` bytes, and the
        // attribute-value special case below peeks at the character after it,
        // so one more byte than that settles this state for good.
        if self.needs_lookahead(entities::MAX_NAME_LEN + 1) {
            return;
        }
        match entities::longest_match(self.rest()) {
            Some((length, replacement)) => {
                let name = self.rest()[..length].to_owned();
                self.pos += length;
                self.temp_buffer.push_str(&name);
                let ends_with_semicolon = name.ends_with(';');
                let next = self.rest().chars().next();
                if self.return_state.is_attribute_value()
                    && !ends_with_semicolon
                    && next.is_some_and(|ch| ch == '=' || ch.is_ascii_alphanumeric())
                {
                    self.flush_code_points();
                    self.state = self.return_state;
                    return;
                }
                if !ends_with_semicolon {
                    self.error(codes::MISSING_SEMICOLON_AFTER_CHARACTER_REFERENCE);
                }
                self.temp_buffer.clear();
                self.temp_buffer.push_str(replacement);
                self.flush_code_points();
                self.state = self.return_state;
            }
            None => {
                self.flush_code_points();
                self.state = State::AmbiguousAmpersand;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#ambiguous-ampersand-state>
    fn ambiguous_ampersand_state(&mut self) {
        loop {
            match self.consume() {
                Some(ch) if ch.is_ascii_alphanumeric() => {
                    if self.return_state.is_attribute_value() {
                        self.attribute_value_mut().push(ch);
                    } else {
                        self.emit_char(ch);
                    }
                }
                Some(';') => {
                    self.error(codes::UNKNOWN_NAMED_CHARACTER_REFERENCE);
                    self.unconsume(';');
                    self.state = self.return_state;
                    return;
                }
                other => {
                    if self.suspended {
                        return;
                    }
                    if let Some(ch) = other {
                        self.unconsume(ch);
                    }
                    self.state = self.return_state;
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-state>
    fn numeric_character_reference_state(&mut self) {
        self.character_reference_code = 0;
        match self.consume() {
            Some(ch @ ('x' | 'X')) => {
                self.temp_buffer.push(ch);
                self.state = State::HexadecimalCharacterReferenceStart;
            }
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::DecimalCharacterReferenceStart;
            }
        }
    }

    /// The hexadecimal and decimal "start" states.
    fn numeric_start_state(&mut self, hexadecimal: bool) {
        // This state looks at the next character without consuming it, so it
        // needs one to be there.
        if self.needs_lookahead(1) {
            return;
        }
        let digit = self.rest().chars().next().is_some_and(|ch| {
            if hexadecimal {
                ch.is_ascii_hexdigit()
            } else {
                ch.is_ascii_digit()
            }
        });
        if digit {
            self.state = if hexadecimal {
                State::HexadecimalCharacterReference
            } else {
                State::DecimalCharacterReference
            };
        } else {
            self.error(codes::ABSENCE_OF_DIGITS_IN_NUMERIC_CHARACTER_REFERENCE);
            self.flush_code_points();
            self.state = self.return_state;
        }
    }

    /// The hexadecimal and decimal digit-accumulating states.
    fn numeric_digits_state(&mut self, hexadecimal: bool) {
        loop {
            match self.consume() {
                Some(ch) if hexadecimal && ch.is_ascii_hexdigit() => {
                    self.accumulate(16, digit_value(ch));
                }
                Some(ch) if !hexadecimal && ch.is_ascii_digit() => {
                    self.accumulate(10, digit_value(ch));
                }
                Some(';') => {
                    self.state = State::NumericCharacterReferenceEnd;
                    return;
                }
                other => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::MISSING_SEMICOLON_AFTER_CHARACTER_REFERENCE);
                    if let Some(ch) = other {
                        self.unconsume(ch);
                    }
                    self.state = State::NumericCharacterReferenceEnd;
                    return;
                }
            }
        }
    }

    /// Multiplies the accumulated code by the base and adds a digit, saturating
    /// so that an absurdly long run of digits cannot overflow.
    fn accumulate(&mut self, base: u32, digit: u32) {
        self.character_reference_code = self
            .character_reference_code
            .saturating_mul(base)
            .saturating_add(digit);
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#numeric-character-reference-end-state>
    fn numeric_character_reference_end_state(&mut self) {
        let mut code = self.character_reference_code;
        if code == 0 {
            self.error(codes::NULL_CHARACTER_REFERENCE);
            code = 0xfffd;
        } else if code > 0x0010_ffff {
            self.error(codes::CHARACTER_REFERENCE_OUTSIDE_UNICODE_RANGE);
            code = 0xfffd;
        } else if (0xd800..=0xdfff).contains(&code) {
            self.error(codes::SURROGATE_CHARACTER_REFERENCE);
            code = 0xfffd;
        } else {
            if is_noncharacter(code) {
                self.error(codes::NONCHARACTER_CHARACTER_REFERENCE);
            }
            if code == 0x0d || (is_control(code) && !matches!(code, 0x09 | 0x0a | 0x0c | 0x20)) {
                self.error(codes::CONTROL_CHARACTER_REFERENCE);
                if let Some(&(_, replacement)) =
                    WINDOWS_1252.iter().find(|&&(from, _)| from == code)
                {
                    code = replacement;
                }
            }
        }
        let ch = char::from_u32(code).unwrap_or('\u{fffd}');
        self.temp_buffer.clear();
        self.temp_buffer.push(ch);
        self.flush_code_points();
        self.state = self.return_state;
    }
}

/// The numeric value of an ASCII hexadecimal digit.
fn digit_value(ch: char) -> u32 {
    ch.to_digit(16).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{is_control, is_noncharacter};

    #[test]
    fn control_and_noncharacter_classification() {
        assert!(is_control(0x00));
        assert!(is_control(0x9f));
        assert!(!is_control(0xa0));
        assert!(is_noncharacter(0xfffe));
        assert!(is_noncharacter(0x1_ffff));
        assert!(is_noncharacter(0xfdd0));
        assert!(!is_noncharacter(0xfdf0));
    }
}
