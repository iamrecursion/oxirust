//! Compilation of a back-reference-free AST to a Pike-VM program.
//!
//! The instruction set is the classic Thompson/Pike design: character
//! consumers, unconditional jumps, priority-ordered splits, capture saves, and
//! zero-width anchor assertions. Repetitions are expanded into splits and
//! jumps; counted repetitions are unrolled with a guard against pathological
//! expansion sizes. Programs produced here never contain a back-reference —
//! patterns that need one are routed to the back-tracking interpreter instead.

use std::rc::Rc;

use crate::ast::Ast;
use crate::charclass::{ClassExpr, any_variant};
use crate::error::RegexError;

/// The upper bound on the number of times a counted quantifier body may be
/// unrolled, and on the total program size, to keep compilation bounded.
pub(crate) const EXPANSION_LIMIT: usize = 200_000;

/// A single-character matcher stored inside a [`Inst::Char`] instruction.
#[derive(Debug, Clone)]
pub(crate) enum Matcher {
    /// Matches exactly one literal character (case-folded when requested).
    Literal(char),
    /// Matches a character class.
    Class(Rc<ClassExpr>),
}

impl Matcher {
    /// Returns `true` if `c` is accepted, honouring case-insensitivity.
    pub(crate) fn matches(&self, c: char, ci: bool) -> bool {
        match self {
            Matcher::Literal(l) => {
                if ci {
                    any_variant(c, |v| v == *l)
                } else {
                    c == *l
                }
            }
            Matcher::Class(cl) => cl.contains(c, ci),
        }
    }
}

/// A Pike-VM instruction.
#[derive(Debug, Clone)]
pub(crate) enum Inst {
    /// Consume one input character if it matches, then advance.
    Char(Matcher),
    /// Accept the match.
    Match,
    /// Continue at the given instruction index.
    Jmp(usize),
    /// Try the first target with priority, the second on back-track.
    Split(usize, usize),
    /// Record the current input position into a capture slot.
    Save(usize),
    /// Zero-width assertion: succeed only at the start of the string (or of a
    /// line, in multi-line mode).
    AssertStart,
    /// Zero-width assertion: succeed only at the end of the string (or of a
    /// line, in multi-line mode).
    AssertEnd,
}

/// A compiled program together with its capture-group count.
#[derive(Debug, Clone)]
pub(crate) struct Program {
    /// The instruction sequence.
    pub insts: Vec<Inst>,
    /// The number of capturing groups (excluding the whole-match group 0).
    pub group_count: usize,
}

/// Compiles `ast` into a Pike-VM [`Program`]. The whole match is bracketed with
/// `Save(0)` / `Save(1)`.
pub(crate) fn compile(ast: &Ast, group_count: usize) -> Result<Program, RegexError> {
    let mut insts = Vec::new();
    insts.push(Inst::Save(0));
    emit(ast, &mut insts)?;
    insts.push(Inst::Save(1));
    insts.push(Inst::Match);
    Ok(Program { insts, group_count })
}

fn check_size(insts: &[Inst]) -> Result<(), RegexError> {
    if insts.len() > EXPANSION_LIMIT {
        return Err(RegexError::InvalidPattern(
            "pattern expands to too many instructions".to_owned(),
        ));
    }
    Ok(())
}

fn emit(node: &Ast, insts: &mut Vec<Inst>) -> Result<(), RegexError> {
    check_size(insts)?;
    match node {
        Ast::Empty => {}
        Ast::Literal(c) => insts.push(Inst::Char(Matcher::Literal(*c))),
        Ast::Class(cl) => insts.push(Inst::Char(Matcher::Class(Rc::new(cl.clone())))),
        Ast::AnchorStart => insts.push(Inst::AssertStart),
        Ast::AnchorEnd => insts.push(Inst::AssertEnd),
        Ast::Concat(items) => {
            for item in items {
                emit(item, insts)?;
            }
        }
        Ast::Alt(branches) => emit_alt(branches, insts)?,
        Ast::Group { index, inner } => {
            if let Some(i) = index {
                insts.push(Inst::Save(2 * i));
            }
            emit(inner, insts)?;
            if let Some(i) = index {
                insts.push(Inst::Save(2 * i + 1));
            }
        }
        Ast::Repeat {
            inner,
            min,
            max,
            greedy,
        } => emit_repeat(inner, *min, *max, *greedy, insts)?,
        Ast::Backref(_) => {
            return Err(RegexError::InvalidPattern(
                "internal error: back-reference reached the Pike compiler".to_owned(),
            ));
        }
    }
    Ok(())
}

fn emit_alt(branches: &[Ast], insts: &mut Vec<Inst>) -> Result<(), RegexError> {
    let mut jmp_positions = Vec::new();
    let last = branches.len().saturating_sub(1);
    for (i, branch) in branches.iter().enumerate() {
        if i < last {
            let split_pos = insts.len();
            insts.push(Inst::Split(0, 0));
            emit(branch, insts)?;
            let jmp_pos = insts.len();
            insts.push(Inst::Jmp(0));
            jmp_positions.push(jmp_pos);
            let next = insts.len();
            insts[split_pos] = Inst::Split(split_pos + 1, next);
        } else {
            emit(branch, insts)?;
        }
    }
    let end = insts.len();
    for pos in jmp_positions {
        insts[pos] = Inst::Jmp(end);
    }
    Ok(())
}

fn emit_repeat(
    inner: &Ast,
    min: u32,
    max: Option<u32>,
    greedy: bool,
    insts: &mut Vec<Inst>,
) -> Result<(), RegexError> {
    let min = min as usize;
    if min > EXPANSION_LIMIT {
        // A lower bound this large cannot be unrolled, but the pattern is
        // still well formed: `fn:matches('aaa', 'a{2147483647}')` is `false`,
        // not `FORX0002`. When the body cannot match the empty string the
        // repetition needs at least `min` characters, so against any input
        // shorter than that it simply fails — and `Regex` refuses to run a
        // program compiled this way against an input that long. A nullable
        // body has no such floor, and is still rejected.
        if !inner.is_nullable() {
            insts.push(Inst::Char(Matcher::Class(Rc::new(ClassExpr::never()))));
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

/// Emits `inner*` (or `inner*?` when not greedy).
fn emit_star(inner: &Ast, greedy: bool, insts: &mut Vec<Inst>) -> Result<(), RegexError> {
    let split_pos = insts.len();
    insts.push(Inst::Split(0, 0));
    emit(inner, insts)?;
    insts.push(Inst::Jmp(split_pos));
    let after = insts.len();
    insts[split_pos] = if greedy {
        Inst::Split(split_pos + 1, after)
    } else {
        Inst::Split(after, split_pos + 1)
    };
    check_size(insts)
}

/// Emits `count` nested optional copies of `inner`, giving `inner{0,count}`.
fn emit_optionals(
    inner: &Ast,
    count: usize,
    greedy: bool,
    insts: &mut Vec<Inst>,
) -> Result<(), RegexError> {
    if count == 0 {
        return Ok(());
    }
    let split_pos = insts.len();
    insts.push(Inst::Split(0, 0));
    emit(inner, insts)?;
    emit_optionals(inner, count - 1, greedy, insts)?;
    let after = insts.len();
    insts[split_pos] = if greedy {
        Inst::Split(split_pos + 1, after)
    } else {
        Inst::Split(after, split_pos + 1)
    };
    check_size(insts)
}
