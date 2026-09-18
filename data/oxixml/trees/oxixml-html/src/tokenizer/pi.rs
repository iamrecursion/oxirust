//! This crate's optional `<?…?>` processing instruction extension.
//!
//! **These states are not in the HTML standard.**
//! [HTML § 13.2.5.6](https://html.spec.whatwg.org/multipage/parsing.html#tag-open-state)
//! makes `<?` an `unexpected-question-mark-instead-of-tag-name` parse error
//! that is reconsumed in the bogus comment state, and that is what this
//! tokenizer does unless [`Tokenizer::set_processing_instructions`] turns the
//! extension on.
//!
//! With the extension on, `<?php echo 1 ?>` in a template produces a
//! processing instruction token rather than a comment, so a pipeline that
//! round-trips such markup keeps it. The restrictions are that the target must
//! match `[A-Za-z_][A-Za-z0-9_-]*` and must not be `xml` or `xml-stylesheet`;
//! anything else falls back to the bogus comment state with the leading `?`
//! restored, which is byte-for-byte the standard's behaviour.

use super::{State, Tokenizer, is_html_whitespace};
use crate::error::codes;
use crate::token::ProcessingInstruction;

impl Tokenizer<'_> {
    pub(super) fn step_processing_instruction(&mut self) {
        match self.state {
            State::ProcessingInstructionOpen => self.open_state(),
            State::ProcessingInstructionTarget => self.target_state(),
            State::AfterProcessingInstructionTarget => self.after_target_state(),
            State::ProcessingInstructionData => self.instruction_data_state(),
            State::ProcessingInstructionQuestionable => self.questionable_state(),
            _ => unreachable!("step_processing_instruction dispatched a state it does not own"),
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#processing-instruction-open-state>
    fn open_state(&mut self) {
        match self.consume() {
            Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => {
                self.unconsume(ch);
                self.state = State::ProcessingInstructionTarget;
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_PROCESSING_INSTRUCTION);
                self.emit_eof();
            }
            Some(ch) => {
                self.error(codes::INVALID_FIRST_CHARACTER_OF_PROCESSING_INSTRUCTION_TARGET);
                self.temp_buffer_to_comment();
                self.unconsume(ch);
                self.state = State::BogusComment;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#processing-instruction-target-state>
    fn target_state(&mut self) {
        loop {
            match self.consume() {
                Some(ch) if is_html_whitespace(ch) || ch == '?' || ch == '>' => {
                    self.unconsume(ch);
                    if self.temp_buffer.eq_ignore_ascii_case("xml")
                        || self.temp_buffer.eq_ignore_ascii_case("xml-stylesheet")
                    {
                        self.error(codes::DISALLOWED_PROCESSING_INSTRUCTION_TARGET);
                        self.temp_buffer_to_comment();
                        self.state = State::BogusComment;
                    } else {
                        self.processing_instruction = ProcessingInstruction {
                            target: std::mem::take(&mut self.temp_buffer),
                            data: String::new(),
                        };
                        self.state = State::AfterProcessingInstructionTarget;
                    }
                    return;
                }
                Some(ch) if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' => {
                    self.temp_buffer.push(ch);
                }
                None => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::EOF_IN_PROCESSING_INSTRUCTION);
                    self.emit_eof();
                    return;
                }
                Some(ch) => {
                    self.error(codes::INVALID_PROCESSING_INSTRUCTION_TARGET);
                    self.temp_buffer_to_comment();
                    self.unconsume(ch);
                    self.state = State::BogusComment;
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#after-processing-instruction-target-state>
    fn after_target_state(&mut self) {
        match self.consume() {
            Some(ch) if is_html_whitespace(ch) => {}
            other => {
                if self.suspended {
                    return;
                }
                if let Some(ch) = other {
                    self.unconsume(ch);
                }
                self.state = State::ProcessingInstructionData;
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#processing-instruction-data-state>
    fn instruction_data_state(&mut self) {
        loop {
            if let Some(run) = self.take_run(&['?', '>']) {
                self.processing_instruction.data.push_str(&run);
                continue;
            }
            match self.consume() {
                Some('?') => {
                    self.state = State::ProcessingInstructionQuestionable;
                    return;
                }
                Some('>') => {
                    self.state = State::Data;
                    self.emit_processing_instruction();
                    return;
                }
                Some(ch) => self.processing_instruction.data.push(ch),
                None => {
                    if self.suspended {
                        return;
                    }
                    self.error(codes::EOF_IN_PROCESSING_INSTRUCTION);
                    self.emit_eof();
                    return;
                }
            }
        }
    }

    /// <https://html.spec.whatwg.org/multipage/parsing.html#processing-instruction-questionable-state>
    fn questionable_state(&mut self) {
        match self.consume() {
            Some('>') => {
                self.state = State::Data;
                self.emit_processing_instruction();
            }
            None => {
                if self.suspended {
                    return;
                }
                self.error(codes::EOF_IN_PROCESSING_INSTRUCTION);
                self.emit_eof();
            }
            Some(ch) => {
                self.processing_instruction.data.push('?');
                self.unconsume(ch);
                self.state = State::ProcessingInstructionData;
            }
        }
    }
}
