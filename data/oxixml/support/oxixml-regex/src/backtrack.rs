//! A back-tracking interpreter for patterns that use back-references.
//!
//! Back-references make a pattern non-regular, so the linear-time Pike VM cannot
//! evaluate them. This module compiles such a pattern to a small instruction set
//! and runs it with an explicit heap-allocated back-track stack (the native call
//! stack stays at constant depth). Two safeguards keep pathological patterns
//! from hanging:
//!
//! * every executed step, at every position a search tries, is counted against
//!   a single budget the caller threads through for the duration of one
//!   logical operation (see [`find_from`]); exhausting it returns
//!   [`RegexError::BudgetExceeded`] instead of looping forever. This bounds
//!   the *total* work done, not just the worst single starting position: a
//!   single position that back-tracks catastrophically (for example
//!   `(a+)+\1$` against a long non-matching string) exhausts it immediately,
//!   and so, eventually, does a pattern that is merely expensive -- never
//!   catastrophic -- at each of unboundedly many positions;
//! * unbounded repetitions carry a loop guard ([`BtInst::MarkLoop`] /
//!   [`BtInst::LoopBack`]) so a body that matches the empty string cannot spin.

use std::rc::Rc;

use crate::ast::Ast;
use crate::charclass::{ClassExpr, any_variant};
use crate::error::RegexError;
use crate::pikevm::{Captures, at_end, at_start};

/// The number of interpreter steps a single search is allowed before it is
/// abandoned with [`RegexError::BudgetExceeded`].
///
/// A "search" is every starting position tried in service of finding *one*
/// match: [`Regex::is_match`](crate::Regex::is_match) and
/// [`Regex::find`](crate::Regex::find) each spend one full budget looking for
/// their single answer, and [`Regex::replace`](crate::Regex::replace) /
/// [`Regex::tokenize`](crate::Regex::tokenize) each spend one full budget
/// across *every* match found during that one call, since each is a single
/// logical operation with one result. [`crate::Matches`] (the iterator behind
/// [`Regex::find_iter`](crate::Regex::find_iter)) instead gives each match it
/// yields its own fresh budget, deliberately: an iterator walking a huge
/// document is expected to keep finding matches for as long as the document
/// has more to give, and charging that against one lifetime total would
/// eventually report [`RegexError::BudgetExceeded`] on perfectly ordinary
/// input. See [`find_from`] for where the within-one-search sharing happens.
pub(crate) const DEFAULT_BUDGET: u64 = 1_000_000;

/// A back-tracking instruction.
#[derive(Debug, Clone)]
pub(crate) enum BtInst {
    /// Consume one input character if it matches a literal.
    Literal(char),
    /// Consume one input character if it matches a class.
    Class(Rc<ClassExpr>),
    /// Accept the match.
    Match,
    /// Continue at the given instruction.
    Jmp(usize),
    /// Try the first target, keeping the second for back-tracking.
    Split(usize, usize),
    /// Record the current position into a capture slot.
    Save(usize),
    /// Match the text previously captured by a group (empty if it did not
    /// participate).
    Backref(usize),
    /// Zero-width start assertion.
    AssertStart,
    /// Zero-width end assertion.
    AssertEnd,
    /// Push the current position as the entry point of a repetition.
    MarkLoop,
    /// Pop a loop entry; jump to the first target if the position advanced,
    /// otherwise to the second (the loop exit).
    LoopBack(usize, usize),
}

/// A compiled back-tracking program.
#[derive(Debug, Clone)]
pub(crate) struct BtProgram {
    /// The instruction sequence.
    pub insts: Vec<BtInst>,
    /// The number of capturing groups (excluding the whole-match group 0).
    pub group_count: usize,
}

/// The same expansion guard the Pike compiler uses.
const EXPANSION_LIMIT: usize = 200_000;

/// Compiles `ast` into a back-tracking [`BtProgram`].
pub(crate) fn compile(ast: &Ast, group_count: usize) -> Result<BtProgram, RegexError> {
    let mut insts = Vec::new();
    insts.push(BtInst::Save(0));
    emit(ast, &mut insts)?;
    insts.push(BtInst::Save(1));
    insts.push(BtInst::Match);
    Ok(BtProgram { insts, group_count })
}

fn check_size(insts: &[BtInst]) -> Result<(), RegexError> {
    if insts.len() > EXPANSION_LIMIT {
        return Err(RegexError::InvalidPattern(
            "pattern expands to too many instructions".to_owned(),
        ));
    }
    Ok(())
}

fn emit(node: &Ast, insts: &mut Vec<BtInst>) -> Result<(), RegexError> {
    check_size(insts)?;
    match node {
        Ast::Empty => {}
        Ast::Literal(c) => insts.push(BtInst::Literal(*c)),
        Ast::Class(cl) => insts.push(BtInst::Class(Rc::new(cl.clone()))),
        Ast::AnchorStart => insts.push(BtInst::AssertStart),
        Ast::AnchorEnd => insts.push(BtInst::AssertEnd),
        Ast::Backref(n) => insts.push(BtInst::Backref(*n)),
        Ast::Concat(items) => {
            for item in items {
                emit(item, insts)?;
            }
        }
        Ast::Alt(branches) => emit_alt(branches, insts)?,
        Ast::Group { index, inner } => {
            if let Some(i) = index {
                insts.push(BtInst::Save(2 * i));
            }
            emit(inner, insts)?;
            if let Some(i) = index {
                insts.push(BtInst::Save(2 * i + 1));
            }
        }
        Ast::Repeat {
            inner,
            min,
            max,
            greedy,
        } => emit_repeat(inner, *min, *max, *greedy, insts)?,
    }
    Ok(())
}

fn emit_alt(branches: &[Ast], insts: &mut Vec<BtInst>) -> Result<(), RegexError> {
    let mut jmp_positions = Vec::new();
    let last = branches.len().saturating_sub(1);
    for (i, branch) in branches.iter().enumerate() {
        if i < last {
            let split_pos = insts.len();
            insts.push(BtInst::Split(0, 0));
            emit(branch, insts)?;
            let jmp_pos = insts.len();
            insts.push(BtInst::Jmp(0));
            jmp_positions.push(jmp_pos);
            let next = insts.len();
            insts[split_pos] = BtInst::Split(split_pos + 1, next);
        } else {
            emit(branch, insts)?;
        }
    }
    let end = insts.len();
    for pos in jmp_positions {
        insts[pos] = BtInst::Jmp(end);
    }
    Ok(())
}

fn emit_repeat(
    inner: &Ast,
    min: u32,
    max: Option<u32>,
    greedy: bool,
    insts: &mut Vec<BtInst>,
) -> Result<(), RegexError> {
    let min = min as usize;
    if min > EXPANSION_LIMIT {
        // See the same guard in `crate::program::emit_repeat`: an unrollable
        // lower bound over a non-nullable body compiles to an instruction that
        // always fails, which is the right answer for every input shorter than
        // the bound.
        if !inner.is_nullable() {
            insts.push(BtInst::Class(Rc::new(ClassExpr::never())));
            return Ok(());
        }
        return Err(RegexError::InvalidPattern(
            "quantifier lower bound is too large".to_owned(),
        ));
    }
    for _ in 0..min {
        emit(inner, insts)?;
        check_size(insts)?;
    }
    match max {
        None => emit_star(inner, greedy, insts)?,
        Some(max) => {
            let max = max as usize;
            let optional = max.saturating_sub(min);
            if optional > EXPANSION_LIMIT {
                return Err(RegexError::InvalidPattern(
                    "quantifier upper bound is too large".to_owned(),
                ));
            }
            emit_optionals(inner, optional, greedy, insts)?;
        }
    }
    Ok(())
}

/// Emits `inner*` with a loop guard so an empty-matching body cannot spin.
fn emit_star(inner: &Ast, greedy: bool, insts: &mut Vec<BtInst>) -> Result<(), RegexError> {
    let split_pos = insts.len();
    insts.push(BtInst::Split(0, 0));
    let mark_pos = insts.len();
    insts.push(BtInst::MarkLoop);
    emit(inner, insts)?;
    let back_pos = insts.len();
    insts.push(BtInst::LoopBack(split_pos, 0));
    let after = insts.len();
    insts[split_pos] = if greedy {
        BtInst::Split(mark_pos, after)
    } else {
        BtInst::Split(after, mark_pos)
    };
    insts[back_pos] = BtInst::LoopBack(split_pos, after);
    check_size(insts)
}

/// Emits `count` nested optional copies of `inner`.
fn emit_optionals(
    inner: &Ast,
    count: usize,
    greedy: bool,
    insts: &mut Vec<BtInst>,
) -> Result<(), RegexError> {
    if count == 0 {
        return Ok(());
    }
    let split_pos = insts.len();
    insts.push(BtInst::Split(0, 0));
    emit(inner, insts)?;
    emit_optionals(inner, count - 1, greedy, insts)?;
    let after = insts.len();
    insts[split_pos] = if greedy {
        BtInst::Split(split_pos + 1, after)
    } else {
        BtInst::Split(after, split_pos + 1)
    };
    check_size(insts)
}

/// A back-tracking thread.
#[derive(Clone)]
struct BtThread {
    pc: usize,
    sp: usize,
    caps: Captures,
    loops: Vec<usize>,
}

/// Attempts to match `program` anchored at position `start`.
///
/// `budget` is not a fresh allowance for this position: it is the number of
/// interpreter steps left in the whole search this attempt is one part of
/// (see [`find_from`]), and every step taken here is deducted from it in
/// place, so that the cost of trying many positions (or many matches; see
/// [`Regex::find_from`](crate::Regex::find_from)) adds up against one shared
/// total instead of each attempt getting its own fresh budget. Returns
/// `Ok(Some(caps))` on success, `Ok(None)` on failure, and
/// `Err(BudgetExceeded)` the moment the shared budget runs out.
fn match_at(
    program: &BtProgram,
    chars: &[char],
    start: usize,
    full: bool,
    ci: bool,
    multiline: bool,
    budget: &mut u64,
) -> Result<Option<Captures>, RegexError> {
    let slots = 2 * (program.group_count + 1);
    let len = chars.len();
    let mut stack: Vec<BtThread> = vec![BtThread {
        pc: 0,
        sp: start,
        caps: vec![None; slots],
        loops: Vec::new(),
    }];

    while let Some(mut t) = stack.pop() {
        loop {
            if *budget == 0 {
                return Err(RegexError::BudgetExceeded);
            }
            *budget -= 1;
            match &program.insts[t.pc] {
                BtInst::Literal(l) => {
                    if t.sp < len && char_matches(chars[t.sp], *l, ci) {
                        t.pc += 1;
                        t.sp += 1;
                    } else {
                        break;
                    }
                }
                BtInst::Class(cl) => {
                    if t.sp < len && cl.contains(chars[t.sp], ci) {
                        t.pc += 1;
                        t.sp += 1;
                    } else {
                        break;
                    }
                }
                BtInst::Match => {
                    if !full || t.sp == len {
                        return Ok(Some(t.caps));
                    }
                    break;
                }
                BtInst::Jmp(x) => t.pc = *x,
                BtInst::Split(a, b) => {
                    let mut alt = t.clone();
                    alt.pc = *b;
                    stack.push(alt);
                    t.pc = *a;
                }
                BtInst::Save(slot) => {
                    if let Some(cell) = t.caps.get_mut(*slot) {
                        *cell = Some(t.sp);
                    }
                    t.pc += 1;
                }
                BtInst::Backref(n) => {
                    let group_start = t.caps.get(2 * n).copied().flatten();
                    let group_end = t.caps.get(2 * n + 1).copied().flatten();
                    match (group_start, group_end) {
                        (Some(gs), Some(ge)) => {
                            let width = ge - gs;
                            if t.sp + width <= len && backref_matches(chars, t.sp, gs, ge, ci) {
                                t.sp += width;
                                t.pc += 1;
                            } else {
                                break;
                            }
                        }
                        // A group that did not participate matches the empty
                        // string.
                        _ => t.pc += 1,
                    }
                }
                BtInst::AssertStart => {
                    if at_start(t.sp, chars, multiline) {
                        t.pc += 1;
                    } else {
                        break;
                    }
                }
                BtInst::AssertEnd => {
                    if at_end(t.sp, chars, multiline) {
                        t.pc += 1;
                    } else {
                        break;
                    }
                }
                BtInst::MarkLoop => {
                    t.loops.push(t.sp);
                    t.pc += 1;
                }
                BtInst::LoopBack(target, exit) => {
                    let entry = t.loops.pop().unwrap_or(t.sp);
                    if t.sp > entry {
                        t.pc = *target;
                    } else {
                        t.pc = *exit;
                    }
                }
            }
        }
    }
    Ok(None)
}

fn char_matches(input: char, literal: char, ci: bool) -> bool {
    if ci {
        any_variant(input, |v| v == literal)
    } else {
        input == literal
    }
}

fn backref_matches(chars: &[char], sp: usize, gs: usize, ge: usize, ci: bool) -> bool {
    let width = ge - gs;
    if sp + width > chars.len() {
        return false;
    }
    for offset in 0..width {
        let a = chars[sp + offset];
        let b = chars[gs + offset];
        let equal = if ci {
            any_variant(a, |v| v == b)
        } else {
            a == b
        };
        if !equal {
            return false;
        }
    }
    true
}

/// Searches for the left-most match starting at or after `from`, returning the
/// capture slots (with slot 0/1 holding the match extent).
///
/// `budget` is shared across every `start` position this loop tries: it is
/// not reset between them, so the total interpreter work performed by this
/// one call is what is bounded, not the cost of any single position in
/// isolation. A caller that needs to run more than one such search as a
/// single logical operation can extend that sharing further by reusing the
/// same `budget` across those calls too, and [`Regex::replace`]
/// (crate::Regex::replace) / [`Regex::tokenize`](crate::Regex::tokenize) do
/// exactly that across every match found during one call, so a pattern whose
/// per-match cost is individually small still cannot buy unbounded total
/// work by matching over and over within that one `replace`/`tokenize` call.
/// [`crate::Matches`] makes the opposite choice on purpose and gives each
/// match its own fresh budget instead — see its type-level documentation.
pub(crate) fn find_from(
    program: &BtProgram,
    chars: &[char],
    from: usize,
    ci: bool,
    multiline: bool,
    budget: &mut u64,
) -> Result<Option<Captures>, RegexError> {
    for start in from..=chars.len() {
        if let Some(caps) = match_at(program, chars, start, false, ci, multiline, budget)? {
            return Ok(Some(caps));
        }
    }
    Ok(None)
}

/// Returns whether any substring at or after position 0 matches (the semantics
/// of `fn:matches`).
pub(crate) fn is_match_search(
    program: &BtProgram,
    chars: &[char],
    ci: bool,
    multiline: bool,
    budget: &mut u64,
) -> Result<bool, RegexError> {
    Ok(find_from(program, chars, 0, ci, multiline, budget)?.is_some())
}

/// Returns whether the whole input matches when anchored at the start (the
/// XSD whole-string semantics). Not reached in normal use, since XSD patterns
/// never contain back-references, but kept for completeness.
pub(crate) fn is_match_full(
    program: &BtProgram,
    chars: &[char],
    ci: bool,
    multiline: bool,
    budget: &mut u64,
) -> Result<bool, RegexError> {
    Ok(match_at(program, chars, 0, true, ci, multiline, budget)?.is_some())
}
