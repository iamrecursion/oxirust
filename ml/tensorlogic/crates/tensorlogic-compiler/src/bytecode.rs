//! Stack-based bytecode VM for TensorLogic expressions.
//!
//! This module provides a compiler from [`TLExpr`] to a flat [`BytecodeProgram`]
//! and a lightweight virtual machine that executes it. Repeated evaluation of
//! compiled expressions is faster than recursive interpretation because the
//! expression tree is only traversed once during compilation; subsequent
//! executions only walk the flat instruction array.
//!
//! # Quick Start
//!
//! ```rust
//! use tensorlogic_compiler::bytecode::{compile, execute, VmEnv, VmValue};
//! use tensorlogic_ir::TLExpr;
//!
//! // Compile 2.0 + 3.0
//! let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
//! let program = compile(&expr).unwrap();
//!
//! let env = VmEnv::new();
//! let result = execute(&program, &env).unwrap();
//! assert_eq!(result, VmValue::Num(5.0));
//! ```

use std::collections::HashMap;
use tensorlogic_ir::TLExpr;

// ─────────────────────────────────────────────────────────────────────────────
// Instruction set
// ─────────────────────────────────────────────────────────────────────────────

/// Stack-based VM instruction.
///
/// Instructions operate on a `Vec<VmValue>` stack and an instruction pointer.
/// All binary operations pop two values and push one; all unary operations pop
/// one and push one.
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // ── Stack management ──────────────────────────────────────────────────
    /// Push a numeric constant onto the stack.
    PushNum(f64),
    /// Push a boolean constant onto the stack.
    PushBool(bool),
    /// Push a symbol literal onto the stack.
    PushSym(String),
    /// Discard the top-of-stack value.
    Pop,
    /// Duplicate the top-of-stack value.
    Dup,

    // ── Arithmetic (binary, both args must be Num) ────────────────────────
    /// Pop b, pop a; push a + b.
    Add,
    /// Pop b, pop a; push a - b.
    Sub,
    /// Pop b, pop a; push a * b.
    Mul,
    /// Pop b, pop a; push a / b (error on zero divisor).
    Div,
    /// Pop b, pop a; push a ^ b.
    Pow,
    /// Pop b, pop a; push a % b.
    Mod,
    /// Pop a; push -a.
    Neg,
    /// Pop a; push |a|.
    Abs,
    /// Pop a; push √a.
    Sqrt,
    /// Pop a; push e^a.
    Exp,
    /// Pop a; push ln(a).
    Log,
    /// Pop b, pop a; push min(a, b).
    Min,
    /// Pop b, pop a; push max(a, b).
    Max,

    // ── Comparison (result pushed as Bool) ────────────────────────────────
    /// Pop b, pop a; push a == b.
    Eq,
    /// Pop b, pop a; push a != b.
    Ne,
    /// Pop b, pop a; push a < b.
    Lt,
    /// Pop b, pop a; push a <= b.
    Le,
    /// Pop b, pop a; push a > b.
    Gt,
    /// Pop b, pop a; push a >= b.
    Ge,

    // ── Boolean logic ─────────────────────────────────────────────────────
    /// Pop b, pop a; push a && b (both must be truthy-compatible).
    And,
    /// Pop b, pop a; push a || b.
    Or,
    /// Pop a; push !a.
    Not,

    // ── Control flow ──────────────────────────────────────────────────────
    /// If TOS is falsy, jump to absolute instruction index; otherwise fall through.
    /// TOS is consumed.
    JumpIfFalse(usize),
    /// If TOS is truthy, jump to absolute instruction index; otherwise fall through.
    /// TOS is consumed.
    JumpIfTrue(usize),
    /// Unconditional jump to absolute instruction index.
    Jump(usize),

    // ── Variables ─────────────────────────────────────────────────────────
    /// Push the value of the named variable from the execution environment.
    LoadVar(String),
    /// Pop TOS and bind it to the named variable in the execution environment.
    StoreVar(String),

    // ── Fuzzy operations ──────────────────────────────────────────────────
    /// Product t-norm: pop b, pop a; push a * b.
    TNorm,
    /// Probabilistic sum t-conorm: pop b, pop a; push a + b - a*b.
    TCoNorm,
    /// Standard fuzzy NOT: pop a; push 1.0 - a.
    FuzzyNot,

    // ── Termination ───────────────────────────────────────────────────────
    /// Stop execution; TOS is the result.
    Halt,
}

// ─────────────────────────────────────────────────────────────────────────────
// BytecodeProgram
// ─────────────────────────────────────────────────────────────────────────────

/// A compiled, flat sequence of [`Instruction`]s.
#[derive(Debug, Clone)]
pub struct BytecodeProgram {
    /// The ordered list of instructions.
    pub instructions: Vec<Instruction>,
}

impl Default for BytecodeProgram {
    fn default() -> Self {
        Self::new()
    }
}

impl BytecodeProgram {
    /// Create an empty program.
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Append an instruction and return its absolute index.
    pub fn push(&mut self, instr: Instruction) -> usize {
        let idx = self.instructions.len();
        self.instructions.push(instr);
        idx
    }

    /// Patch the jump target of the instruction at `idx`.
    ///
    /// Panics in debug builds if `idx` does not contain a jump instruction.
    pub fn patch_jump(&mut self, idx: usize, target: usize) {
        match &mut self.instructions[idx] {
            Instruction::JumpIfFalse(t) | Instruction::JumpIfTrue(t) | Instruction::Jump(t) => {
                *t = target;
            }
            other => {
                debug_assert!(
                    false,
                    "patch_jump called on non-jump instruction: {:?}",
                    other
                );
            }
        }
    }

    /// Return the number of instructions in the program.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Return `true` if the program contains no instructions.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VmValue
// ─────────────────────────────────────────────────────────────────────────────

/// A runtime value on the VM stack.
#[derive(Debug, Clone, PartialEq)]
pub enum VmValue {
    /// A 64-bit floating-point number.
    Num(f64),
    /// A boolean flag.
    Bool(bool),
    /// A symbol (named constant), used in pattern matching.
    Sym(String),
}

impl VmValue {
    /// Extract the numeric payload or return a [`VmError::TypeMismatch`].
    pub fn as_num(&self) -> Result<f64, VmError> {
        match self {
            VmValue::Num(n) => Ok(*n),
            VmValue::Bool(_) => Err(VmError::TypeMismatch {
                expected: "Num",
                got: "Bool",
            }),
            VmValue::Sym(_) => Err(VmError::TypeMismatch {
                expected: "Num",
                got: "Sym",
            }),
        }
    }

    /// Extract the boolean payload or return a [`VmError::TypeMismatch`].
    pub fn as_bool(&self) -> Result<bool, VmError> {
        match self {
            VmValue::Bool(b) => Ok(*b),
            VmValue::Num(_) => Err(VmError::TypeMismatch {
                expected: "Bool",
                got: "Num",
            }),
            VmValue::Sym(_) => Err(VmError::TypeMismatch {
                expected: "Bool",
                got: "Sym",
            }),
        }
    }

    /// Numeric: non-zero is truthy; Boolean: direct value; Symbol: always truthy.
    pub fn is_truthy(&self) -> bool {
        match self {
            VmValue::Num(n) => *n != 0.0,
            VmValue::Bool(b) => *b,
            VmValue::Sym(s) => !s.is_empty(),
        }
    }

    /// Return a static type name string for error messages.
    #[allow(dead_code)]
    fn type_name(&self) -> &'static str {
        match self {
            VmValue::Num(_) => "Num",
            VmValue::Bool(_) => "Bool",
            VmValue::Sym(_) => "Sym",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VmError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during VM execution.
#[derive(Debug)]
pub enum VmError {
    /// A pop or peek was attempted on an empty stack.
    StackUnderflow,
    /// An operation received a value of an unexpected type.
    TypeMismatch {
        /// The type that was expected.
        expected: &'static str,
        /// The type that was actually present.
        got: &'static str,
    },
    /// A `LoadVar` instruction referenced a name not present in the environment.
    UnboundVariable(String),
    /// A `Div` instruction was attempted with a zero denominator.
    DivisionByZero,
    /// The instruction pointer jumped outside the program bounds.
    InvalidInstruction(usize),
    /// `execute` was called with a program that contains no instructions.
    ProgramEmpty,
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmError::StackUnderflow => write!(f, "VM stack underflow"),
            VmError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {}, got {}", expected, got)
            }
            VmError::UnboundVariable(name) => {
                write!(f, "unbound variable: '{}'", name)
            }
            VmError::DivisionByZero => write!(f, "division by zero"),
            VmError::InvalidInstruction(ip) => {
                write!(f, "invalid instruction pointer: {}", ip)
            }
            VmError::ProgramEmpty => write!(f, "program contains no instructions"),
        }
    }
}

impl std::error::Error for VmError {}

// ─────────────────────────────────────────────────────────────────────────────
// VmEnv
// ─────────────────────────────────────────────────────────────────────────────

/// Variable environment passed to the VM at execution time.
///
/// The VM does **not** modify the caller's environment; `StoreVar` writes into
/// a local clone that is discarded when execution ends.
#[derive(Debug, Clone, Default)]
pub struct VmEnv {
    vars: HashMap<String, VmValue>,
}

impl VmEnv {
    /// Create an empty environment.
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    /// Bind a variable to an arbitrary [`VmValue`].
    pub fn set(&mut self, name: impl Into<String>, val: VmValue) {
        self.vars.insert(name.into(), val);
    }

    /// Convenience helper: bind a variable to a numeric value.
    pub fn set_num(&mut self, name: impl Into<String>, val: f64) {
        self.set(name, VmValue::Num(val));
    }

    /// Convenience helper: bind a variable to a boolean value.
    pub fn set_bool(&mut self, name: impl Into<String>, val: bool) {
        self.set(name, VmValue::Bool(val));
    }

    /// Look up a variable by name.
    pub fn get(&self, name: &str) -> Option<&VmValue> {
        self.vars.get(name)
    }

    /// Return the number of bindings in the environment.
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// Return `true` if the environment contains no bindings.
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VmStats
// ─────────────────────────────────────────────────────────────────────────────

/// Execution statistics collected during a single VM run.
#[derive(Debug, Default, Clone)]
pub struct VmStats {
    /// Total number of instructions dispatched (including the final `Halt`).
    pub instructions_executed: usize,
    /// The highest stack depth observed at any point during execution.
    pub max_stack_depth: usize,
    /// Number of conditional or unconditional jumps that were actually taken.
    pub jumps_taken: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompileError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during bytecode compilation.
#[derive(Debug)]
pub enum CompileError {
    /// The expression contains a variant that the bytecode compiler does not
    /// support (e.g. quantifiers, modal operators, lambda, …).
    UnsupportedExpr(String),
    /// The expression tree is deeper than the configured `max_depth` limit.
    MaxDepthExceeded,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnsupportedExpr(desc) => {
                write!(f, "unsupported expression in bytecode compiler: {}", desc)
            }
            CompileError::MaxDepthExceeded => {
                write!(f, "expression depth exceeds configured maximum")
            }
        }
    }
}

impl std::error::Error for CompileError {}

// ─────────────────────────────────────────────────────────────────────────────
// Compiler internals
// ─────────────────────────────────────────────────────────────────────────────

/// Internal compilation state passed through recursive descent.
struct Compiler {
    program: BytecodeProgram,
    max_depth: usize,
}

impl Compiler {
    fn new(max_depth: usize) -> Self {
        Self {
            program: BytecodeProgram::new(),
            max_depth,
        }
    }

    /// Recursively compile `expr` into `self.program`, respecting `depth`.
    fn compile_expr(&mut self, expr: &TLExpr, depth: usize) -> Result<(), CompileError> {
        if depth > self.max_depth {
            return Err(CompileError::MaxDepthExceeded);
        }

        match expr {
            // ── Numeric literal ───────────────────────────────────────────
            TLExpr::Constant(c) => {
                self.program.push(Instruction::PushNum(*c));
            }

            // ── Zero-arity predicates are treated as variable loads ────────
            TLExpr::Pred { name, args } if args.is_empty() => {
                self.program.push(Instruction::LoadVar(name.clone()));
            }

            // ── Arithmetic ────────────────────────────────────────────────
            TLExpr::Add(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Add);
            }
            TLExpr::Sub(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Sub);
            }
            TLExpr::Mul(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Mul);
            }
            TLExpr::Div(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Div);
            }
            TLExpr::Pow(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Pow);
            }
            TLExpr::Mod(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Mod);
            }
            TLExpr::Abs(a) => {
                self.compile_expr(a, depth + 1)?;
                self.program.push(Instruction::Abs);
            }
            TLExpr::Sqrt(a) => {
                self.compile_expr(a, depth + 1)?;
                self.program.push(Instruction::Sqrt);
            }
            TLExpr::Exp(a) => {
                self.compile_expr(a, depth + 1)?;
                self.program.push(Instruction::Exp);
            }
            TLExpr::Log(a) => {
                self.compile_expr(a, depth + 1)?;
                self.program.push(Instruction::Log);
            }
            TLExpr::Min(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Min);
            }
            TLExpr::Max(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Max);
            }

            // ── Comparison ────────────────────────────────────────────────
            TLExpr::Eq(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Eq);
            }
            TLExpr::Lt(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Lt);
            }
            TLExpr::Gt(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Gt);
            }
            TLExpr::Lte(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Le);
            }
            TLExpr::Gte(a, b) => {
                self.compile_expr(a, depth + 1)?;
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Ge);
            }

            // ── Boolean logic with short-circuit jumps ────────────────────
            //
            // And(a, b):
            //   compile(a)
            //   JumpIfFalse(end)   ; pops a; if false pushes Bool(false) and jumps to end
            //                      ; if true falls through (a consumed; stack unchanged)
            //   compile(b)         ; result of b is the final value (a was truthy)
            //   Not, Not           ; coerce numeric b to Bool if needed (identity on Bool)
            //   end:
            //
            // Note: JumpIfFalse CONSUMES the condition value from the stack. When it
            // does NOT jump (a is truthy) the stack is one entry shorter, so we only
            // need b's result at the end. When it DOES jump it pushes Bool(false) at
            // the target, so both paths leave exactly one value on the stack.
            TLExpr::And(a, b) => {
                self.compile_expr(a, depth + 1)?;
                // Emit placeholder jump; we'll patch the target after compiling b.
                let jump_idx = self.program.push(Instruction::JumpIfFalse(0));
                // a was truthy (and consumed); compile b — its value is the And result.
                self.compile_expr(b, depth + 1)?;
                // Coerce b's result to Bool so both paths leave a Bool on the stack.
                self.program.push(Instruction::Not);
                self.program.push(Instruction::Not);
                let end = self.program.len();
                self.program.patch_jump(jump_idx, end);
            }

            // Or(a, b):
            //   compile(a)
            //   JumpIfTrue(end)    ; pops a; if true pushes Bool(true) and jumps to end
            //   compile(b)         ; result of b is the final value (a was falsy)
            //   Not, Not           ; coerce to Bool
            //   end:
            TLExpr::Or(a, b) => {
                self.compile_expr(a, depth + 1)?;
                let jump_idx = self.program.push(Instruction::JumpIfTrue(0));
                self.compile_expr(b, depth + 1)?;
                self.program.push(Instruction::Not);
                self.program.push(Instruction::Not);
                let end = self.program.len();
                self.program.patch_jump(jump_idx, end);
            }

            TLExpr::Not(a) => {
                self.compile_expr(a, depth + 1)?;
                self.program.push(Instruction::Not);
            }

            // ── Conditional ───────────────────────────────────────────────
            //
            // IfThenElse(cond, t, f):
            //   compile(cond)
            //   JumpIfFalse(else_branch)
            //   compile(t)
            //   Jump(end)
            //   else_branch: compile(f)
            //   end:
            TLExpr::IfThenElse {
                condition,
                then_branch,
                else_branch,
            } => {
                self.compile_expr(condition, depth + 1)?;
                let jf_idx = self.program.push(Instruction::JumpIfFalse(0));
                self.compile_expr(then_branch, depth + 1)?;
                let jump_idx = self.program.push(Instruction::Jump(0));
                // patch the JumpIfFalse to point here
                let else_start = self.program.len();
                self.program.patch_jump(jf_idx, else_start);
                self.compile_expr(else_branch, depth + 1)?;
                let end = self.program.len();
                self.program.patch_jump(jump_idx, end);
            }

            // ── Let binding ───────────────────────────────────────────────
            //
            // Let { var, value, body }:
            //   compile(value)
            //   StoreVar(var)
            //   compile(body)
            TLExpr::Let { var, value, body } => {
                self.compile_expr(value, depth + 1)?;
                self.program.push(Instruction::StoreVar(var.clone()));
                self.compile_expr(body, depth + 1)?;
            }

            // ── Fuzzy operations ──────────────────────────────────────────
            //
            // The bytecode VM uses a single product t-norm / probabilistic-sum
            // t-conorm regardless of the `kind` tag.  More sophisticated
            // dispatch can be added later without changing the instruction set.
            TLExpr::TNorm { left, right, .. } => {
                self.compile_expr(left, depth + 1)?;
                self.compile_expr(right, depth + 1)?;
                self.program.push(Instruction::TNorm);
            }
            TLExpr::TCoNorm { left, right, .. } => {
                self.compile_expr(left, depth + 1)?;
                self.compile_expr(right, depth + 1)?;
                self.program.push(Instruction::TCoNorm);
            }
            TLExpr::FuzzyNot { expr: inner, .. } => {
                self.compile_expr(inner, depth + 1)?;
                self.program.push(Instruction::FuzzyNot);
            }

            // ── Symbol literal ────────────────────────────────────────────
            TLExpr::SymbolLiteral(s) => {
                self.program.push(Instruction::PushSym(s.clone()));
            }

            // ── Pattern matching ──────────────────────────────────────────
            //
            // Lower `Match { scrutinee, arms }` to nested IfThenElse at the
            // bytecode level.  The scrutinee is compiled once and stored in a
            // fresh temporary variable to avoid re-evaluation.
            TLExpr::Match { scrutinee, arms } => {
                if arms.is_empty() {
                    return Err(CompileError::UnsupportedExpr(
                        "Match with no arms".to_string(),
                    ));
                }
                // Store scrutinee in a fresh temp.
                self.compile_expr(scrutinee, depth + 1)?;
                let tmp = format!("__match_scrutinee_{depth}");
                self.program.push(Instruction::StoreVar(tmp.clone()));

                // Build nested IfThenElse from the arms (last arm = wildcard).
                let (wildcard_body, non_wildcard) = arms
                    .split_last()
                    .ok_or_else(|| CompileError::UnsupportedExpr("Empty Match arms".into()))?;

                // Inline compile the chain: iterate non-wildcard arms in
                // reverse, wrapping around the accumulated else-branch.
                // We achieve this by emitting the structure directly.
                self.emit_match_chain(&tmp, non_wildcard, &wildcard_body.1, depth)?;
            }

            // ── Unsupported variants ──────────────────────────────────────
            other => {
                return Err(CompileError::UnsupportedExpr(format!("{:?}", other)));
            }
        }

        Ok(())
    }

    /// Emit a chain of conditional jumps for the non-wildcard arms of a Match.
    ///
    /// The scrutinee has already been stored in `scrutinee_var`.
    /// `arms` is the slice of (Pattern, body) excluding the wildcard tail.
    /// `else_body` is the wildcard-arm body.
    fn emit_match_chain(
        &mut self,
        scrutinee_var: &str,
        arms: &[(tensorlogic_ir::MatchPattern, Box<TLExpr>)],
        else_body: &TLExpr,
        depth: usize,
    ) -> Result<(), CompileError> {
        if arms.is_empty() {
            // Only wildcard — compile the else body directly.
            return self.compile_expr(else_body, depth + 1);
        }

        // Emit current arm: if scrutinee == rhs { body } else { rest }
        let (pat, body) = &arms[0];
        let remaining = &arms[1..];

        // Condition: load scrutinee, push rhs, compare.
        self.program
            .push(Instruction::LoadVar(scrutinee_var.to_string()));
        match pat {
            tensorlogic_ir::MatchPattern::ConstNumber(n) => {
                self.program.push(Instruction::PushNum(*n));
            }
            tensorlogic_ir::MatchPattern::ConstSymbol(s) => {
                self.program.push(Instruction::PushSym(s.clone()));
            }
            tensorlogic_ir::MatchPattern::Wildcard => {
                return Err(CompileError::UnsupportedExpr(
                    "Wildcard in non-tail position".into(),
                ));
            }
        }
        self.program.push(Instruction::Eq);

        // JumpIfFalse → else branch
        let jf_idx = self.program.push(Instruction::JumpIfFalse(0));
        // Then branch: compile arm body
        self.compile_expr(body, depth + 1)?;
        // Jump past else
        let jump_idx = self.program.push(Instruction::Jump(0));
        // Else branch:
        let else_start = self.program.len();
        self.program.patch_jump(jf_idx, else_start);
        // Recurse for remaining arms
        self.emit_match_chain(scrutinee_var, remaining, else_body, depth)?;
        let end = self.program.len();
        self.program.patch_jump(jump_idx, end);

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public compile API
// ─────────────────────────────────────────────────────────────────────────────

/// Default maximum expression depth allowed during compilation.
pub const DEFAULT_MAX_DEPTH: usize = 512;

/// Compile a [`TLExpr`] to a [`BytecodeProgram`].
///
/// The final instruction appended is always [`Instruction::Halt`].
///
/// # Errors
///
/// Returns [`CompileError::UnsupportedExpr`] when the expression tree contains
/// variants that cannot be translated to bytecode (e.g. quantifiers, modal
/// operators, lambda abstractions).
///
/// Returns [`CompileError::MaxDepthExceeded`] if the default depth limit
/// ([`DEFAULT_MAX_DEPTH`]) is surpassed.
pub fn compile(expr: &TLExpr) -> Result<BytecodeProgram, CompileError> {
    compile_with_config(expr, DEFAULT_MAX_DEPTH)
}

/// Compile a [`TLExpr`] to a [`BytecodeProgram`] with an explicit depth limit.
///
/// `max_depth` controls how deeply the compiler will recurse into nested
/// sub-expressions before emitting [`CompileError::MaxDepthExceeded`].
pub fn compile_with_config(
    expr: &TLExpr,
    max_depth: usize,
) -> Result<BytecodeProgram, CompileError> {
    let mut compiler = Compiler::new(max_depth);
    compiler.compile_expr(expr, 0)?;
    compiler.program.push(Instruction::Halt);
    Ok(compiler.program)
}

// ─────────────────────────────────────────────────────────────────────────────
// VM executor
// ─────────────────────────────────────────────────────────────────────────────

/// Execute a [`BytecodeProgram`] and return the top-of-stack value after `Halt`.
///
/// A mutable local copy of `env` is used so that [`Instruction::StoreVar`]
/// does not modify the caller's environment.
pub fn execute(program: &BytecodeProgram, env: &VmEnv) -> Result<VmValue, VmError> {
    let (val, _stats) = execute_with_stats(program, env)?;
    Ok(val)
}

/// Execute a [`BytecodeProgram`] and return both the result and execution statistics.
pub fn execute_with_stats(
    program: &BytecodeProgram,
    env: &VmEnv,
) -> Result<(VmValue, VmStats), VmError> {
    if program.is_empty() {
        return Err(VmError::ProgramEmpty);
    }

    let mut stack: Vec<VmValue> = Vec::with_capacity(16);
    // Local mutable copy so StoreVar doesn't mutate the caller's env.
    let mut local_env = env.clone();
    let mut ip: usize = 0;
    let mut stats = VmStats::default();

    loop {
        if ip >= program.instructions.len() {
            return Err(VmError::InvalidInstruction(ip));
        }

        let instr = &program.instructions[ip];
        stats.instructions_executed += 1;

        match instr {
            // ── Stack management ─────────────────────────────────────────
            Instruction::PushNum(n) => {
                stack.push(VmValue::Num(*n));
                ip += 1;
            }
            Instruction::PushBool(b) => {
                stack.push(VmValue::Bool(*b));
                ip += 1;
            }
            Instruction::PushSym(s) => {
                stack.push(VmValue::Sym(s.clone()));
                ip += 1;
            }
            Instruction::Pop => {
                stack.pop().ok_or(VmError::StackUnderflow)?;
                ip += 1;
            }
            Instruction::Dup => {
                let top = stack.last().ok_or(VmError::StackUnderflow)?.clone();
                stack.push(top);
                ip += 1;
            }

            // ── Arithmetic ───────────────────────────────────────────────
            Instruction::Add => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a + b));
                ip += 1;
            }
            Instruction::Sub => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a - b));
                ip += 1;
            }
            Instruction::Mul => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a * b));
                ip += 1;
            }
            Instruction::Div => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                if b == 0.0 {
                    return Err(VmError::DivisionByZero);
                }
                stack.push(VmValue::Num(a / b));
                ip += 1;
            }
            Instruction::Pow => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a.powf(b)));
                ip += 1;
            }
            Instruction::Mod => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a % b));
                ip += 1;
            }
            Instruction::Neg => {
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(-a));
                ip += 1;
            }
            Instruction::Abs => {
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a.abs()));
                ip += 1;
            }
            Instruction::Sqrt => {
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a.sqrt()));
                ip += 1;
            }
            Instruction::Exp => {
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a.exp()));
                ip += 1;
            }
            Instruction::Log => {
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a.ln()));
                ip += 1;
            }
            Instruction::Min => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a.min(b)));
                ip += 1;
            }
            Instruction::Max => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Num(a.max(b)));
                ip += 1;
            }

            // ── Comparison ───────────────────────────────────────────────
            Instruction::Eq => {
                let b = pop_value(&mut stack)?;
                let a = pop_value(&mut stack)?;
                stack.push(VmValue::Bool(values_equal(&a, &b)));
                ip += 1;
            }
            Instruction::Ne => {
                let b = pop_value(&mut stack)?;
                let a = pop_value(&mut stack)?;
                stack.push(VmValue::Bool(!values_equal(&a, &b)));
                ip += 1;
            }
            Instruction::Lt => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Bool(a < b));
                ip += 1;
            }
            Instruction::Le => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Bool(a <= b));
                ip += 1;
            }
            Instruction::Gt => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Bool(a > b));
                ip += 1;
            }
            Instruction::Ge => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                stack.push(VmValue::Bool(a >= b));
                ip += 1;
            }

            // ── Boolean logic ─────────────────────────────────────────────
            Instruction::And => {
                let b = pop_value(&mut stack)?;
                let a = pop_value(&mut stack)?;
                stack.push(VmValue::Bool(a.is_truthy() && b.is_truthy()));
                ip += 1;
            }
            Instruction::Or => {
                let b = pop_value(&mut stack)?;
                let a = pop_value(&mut stack)?;
                stack.push(VmValue::Bool(a.is_truthy() || b.is_truthy()));
                ip += 1;
            }
            Instruction::Not => {
                let a = pop_value(&mut stack)?;
                stack.push(VmValue::Bool(!a.is_truthy()));
                ip += 1;
            }

            // ── Control flow ──────────────────────────────────────────────
            Instruction::JumpIfFalse(target) => {
                let target = *target;
                let cond = pop_value(&mut stack)?;
                if !cond.is_truthy() {
                    // Push a Bool(false) so the result is still on the stack at the
                    // jump target — the caller is responsible for having set up the
                    // stack correctly, but we preserve the false value here so that
                    // short-circuit And returns the correct result.
                    stack.push(VmValue::Bool(false));
                    ip = target;
                    stats.jumps_taken += 1;
                } else {
                    ip += 1;
                }
            }
            Instruction::JumpIfTrue(target) => {
                let target = *target;
                let cond = pop_value(&mut stack)?;
                if cond.is_truthy() {
                    // Similarly preserve the true value for short-circuit Or.
                    stack.push(VmValue::Bool(true));
                    ip = target;
                    stats.jumps_taken += 1;
                } else {
                    ip += 1;
                }
            }
            Instruction::Jump(target) => {
                ip = *target;
                stats.jumps_taken += 1;
            }

            // ── Variables ─────────────────────────────────────────────────
            Instruction::LoadVar(name) => {
                let val = local_env
                    .get(name)
                    .ok_or_else(|| VmError::UnboundVariable(name.clone()))?
                    .clone();
                stack.push(val);
                ip += 1;
            }
            Instruction::StoreVar(name) => {
                let val = pop_value(&mut stack)?;
                local_env.set(name.clone(), val);
                ip += 1;
            }

            // ── Fuzzy operations ──────────────────────────────────────────
            Instruction::TNorm => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                // Product t-norm: T(a, b) = a * b
                stack.push(VmValue::Num(a * b));
                ip += 1;
            }
            Instruction::TCoNorm => {
                let b = pop_num(&mut stack)?;
                let a = pop_num(&mut stack)?;
                // Probabilistic sum: S(a, b) = a + b - a*b
                stack.push(VmValue::Num(a + b - a * b));
                ip += 1;
            }
            Instruction::FuzzyNot => {
                let a = pop_num(&mut stack)?;
                // Standard fuzzy NOT: N(a) = 1 - a
                stack.push(VmValue::Num(1.0 - a));
                ip += 1;
            }

            // ── Termination ───────────────────────────────────────────────
            Instruction::Halt => {
                let result = stack.pop().ok_or(VmError::StackUnderflow)?;
                // Update final stats
                if stats.max_stack_depth < stack.len() + 1 {
                    stats.max_stack_depth = stack.len() + 1;
                }
                return Ok((result, stats));
            }
        }

        // Track maximum stack depth after each instruction.
        if stack.len() > stats.max_stack_depth {
            stats.max_stack_depth = stack.len();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pop the top-of-stack value (any type).
#[inline]
fn pop_value(stack: &mut Vec<VmValue>) -> Result<VmValue, VmError> {
    stack.pop().ok_or(VmError::StackUnderflow)
}

/// Pop the top-of-stack value and coerce it to `f64`.
#[inline]
fn pop_num(stack: &mut Vec<VmValue>) -> Result<f64, VmError> {
    let val = stack.pop().ok_or(VmError::StackUnderflow)?;
    match val {
        VmValue::Num(n) => Ok(n),
        VmValue::Bool(_) => Err(VmError::TypeMismatch {
            expected: "Num",
            got: "Bool",
        }),
        VmValue::Sym(_) => Err(VmError::TypeMismatch {
            expected: "Num",
            got: "Sym",
        }),
    }
}

/// Value equality that is aware of the two possible [`VmValue`] variants.
#[inline]
fn values_equal(a: &VmValue, b: &VmValue) -> bool {
    match (a, b) {
        (VmValue::Num(x), VmValue::Num(y)) => x == y,
        (VmValue::Bool(x), VmValue::Bool(y)) => x == y,
        (VmValue::Sym(x), VmValue::Sym(y)) => x == y,
        _ => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{FuzzyNegationKind, TCoNormKind, TLExpr, TNormKind};

    // ── Helper: compile + execute in one step ──────────────────────────────
    fn eval(expr: TLExpr) -> VmValue {
        let prog = compile(&expr).expect("compile failed");
        let env = VmEnv::new();
        execute(&prog, &env).expect("execute failed")
    }

    fn eval_env(expr: TLExpr, env: &VmEnv) -> VmValue {
        let prog = compile(&expr).expect("compile failed");
        execute(&prog, env).expect("execute failed")
    }

    // ── 1. Constant compile shape ──────────────────────────────────────────
    #[test]
    fn test_compile_constant_shape() {
        let val = std::f64::consts::PI;
        let prog = compile(&TLExpr::Constant(val)).expect("compile failed");
        assert_eq!(prog.len(), 2, "should be [PushNum(PI), Halt]");
        assert_eq!(prog.instructions[0], Instruction::PushNum(val));
        assert_eq!(prog.instructions[1], Instruction::Halt);
    }

    // ── 2. Execute single PushNum ──────────────────────────────────────────
    #[test]
    fn test_execute_push_num() {
        let mut prog = BytecodeProgram::new();
        prog.push(Instruction::PushNum(5.0));
        prog.push(Instruction::Halt);
        let env = VmEnv::new();
        let result = execute(&prog, &env).expect("execute failed");
        assert_eq!(result, VmValue::Num(5.0));
    }

    // ── 3. Add ────────────────────────────────────────────────────────────
    #[test]
    fn test_add() {
        let expr = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
        assert_eq!(eval(expr), VmValue::Num(5.0));
    }

    // ── 4. Sub ────────────────────────────────────────────────────────────
    #[test]
    fn test_sub() {
        let expr = TLExpr::sub(TLExpr::Constant(10.0), TLExpr::Constant(4.0));
        assert_eq!(eval(expr), VmValue::Num(6.0));
    }

    // ── 5. Mul ────────────────────────────────────────────────────────────
    #[test]
    fn test_mul() {
        let expr = TLExpr::mul(TLExpr::Constant(3.0), TLExpr::Constant(4.0));
        assert_eq!(eval(expr), VmValue::Num(12.0));
    }

    // ── 6. Div ────────────────────────────────────────────────────────────
    #[test]
    fn test_div() {
        let expr = TLExpr::div(TLExpr::Constant(10.0), TLExpr::Constant(2.0));
        assert_eq!(eval(expr), VmValue::Num(5.0));
    }

    // ── 7. Pow ────────────────────────────────────────────────────────────
    #[test]
    fn test_pow() {
        let expr = TLExpr::pow(TLExpr::Constant(2.0), TLExpr::Constant(8.0));
        assert_eq!(eval(expr), VmValue::Num(256.0));
    }

    // ── 8. Eq true ────────────────────────────────────────────────────────
    #[test]
    fn test_eq_true() {
        let expr = TLExpr::eq(TLExpr::Constant(3.0), TLExpr::Constant(3.0));
        assert_eq!(eval(expr), VmValue::Bool(true));
    }

    // ── 9. Lt true ────────────────────────────────────────────────────────
    #[test]
    fn test_lt_true() {
        let expr = TLExpr::lt(TLExpr::Constant(1.0), TLExpr::Constant(2.0));
        assert_eq!(eval(expr), VmValue::Bool(true));
    }

    // ── 10. And false ─────────────────────────────────────────────────────
    #[test]
    fn test_and_false() {
        let expr = TLExpr::and(
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(1.0)),
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(2.0)),
        );
        assert_eq!(eval(expr), VmValue::Bool(false));
    }

    // ── 11. Or true ───────────────────────────────────────────────────────
    #[test]
    fn test_or_true() {
        let expr = TLExpr::or(
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(2.0)),
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(1.0)),
        );
        assert_eq!(eval(expr), VmValue::Bool(true));
    }

    // ── 12. Not false → true ──────────────────────────────────────────────
    #[test]
    fn test_not_false_to_true() {
        let expr = TLExpr::negate(TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(2.0)));
        assert_eq!(eval(expr), VmValue::Bool(true));
    }

    // ── 13. Short-circuit And: jump taken when first arg is false ─────────
    #[test]
    fn test_short_circuit_and_jump() {
        // First argument is false → second should be skipped entirely.
        let expr = TLExpr::and(
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(2.0)), // false
            TLExpr::eq(TLExpr::Constant(3.0), TLExpr::Constant(3.0)), // true (never reached)
        );
        let prog = compile(&expr).expect("compile failed");
        let env = VmEnv::new();
        let (result, stats) = execute_with_stats(&prog, &env).expect("execute failed");
        assert_eq!(result, VmValue::Bool(false));
        assert!(stats.jumps_taken > 0, "JumpIfFalse should have been taken");
    }

    // ── 14. Short-circuit Or: jump taken when first arg is true ───────────
    #[test]
    fn test_short_circuit_or_jump() {
        let expr = TLExpr::or(
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(1.0)), // true
            TLExpr::eq(TLExpr::Constant(3.0), TLExpr::Constant(4.0)), // false (never reached)
        );
        let prog = compile(&expr).expect("compile failed");
        let env = VmEnv::new();
        let (result, stats) = execute_with_stats(&prog, &env).expect("execute failed");
        assert_eq!(result, VmValue::Bool(true));
        assert!(stats.jumps_taken > 0, "JumpIfTrue should have been taken");
    }

    // ── 15. IfThenElse true branch ────────────────────────────────────────
    #[test]
    fn test_ite_true_branch() {
        let expr = TLExpr::if_then_else(
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(1.0)),
            TLExpr::Constant(1.0),
            TLExpr::Constant(2.0),
        );
        assert_eq!(eval(expr), VmValue::Num(1.0));
    }

    // ── 16. IfThenElse false branch ───────────────────────────────────────
    #[test]
    fn test_ite_false_branch() {
        let expr = TLExpr::if_then_else(
            TLExpr::eq(TLExpr::Constant(1.0), TLExpr::Constant(2.0)),
            TLExpr::Constant(1.0),
            TLExpr::Constant(2.0),
        );
        assert_eq!(eval(expr), VmValue::Num(2.0));
    }

    // ── 17. LoadVar retrieves value from VmEnv ────────────────────────────
    #[test]
    fn test_load_var() {
        let expr = TLExpr::pred("x", vec![]);
        let mut env = VmEnv::new();
        env.set_num("x", 42.0);
        assert_eq!(eval_env(expr, &env), VmValue::Num(42.0));
    }

    // ── 18. Let binding roundtrip ─────────────────────────────────────────
    #[test]
    fn test_let_binding() {
        // let y = 7.0 in y * 2.0
        let expr = TLExpr::Let {
            var: "y".to_string(),
            value: Box::new(TLExpr::Constant(7.0)),
            body: Box::new(TLExpr::mul(
                TLExpr::pred("y", vec![]),
                TLExpr::Constant(2.0),
            )),
        };
        let env = VmEnv::new();
        assert_eq!(eval_env(expr, &env), VmValue::Num(14.0));
    }

    // ── 19. VmError::StackUnderflow ───────────────────────────────────────
    #[test]
    fn test_stack_underflow() {
        let mut prog = BytecodeProgram::new();
        prog.push(Instruction::Add); // no operands
        prog.push(Instruction::Halt);
        let env = VmEnv::new();
        let err = execute(&prog, &env).unwrap_err();
        assert!(
            matches!(err, VmError::StackUnderflow),
            "expected StackUnderflow, got {:?}",
            err
        );
    }

    // ── 20. VmError::UnboundVariable ──────────────────────────────────────
    #[test]
    fn test_unbound_variable() {
        let mut prog = BytecodeProgram::new();
        prog.push(Instruction::LoadVar("missing".to_string()));
        prog.push(Instruction::Halt);
        let env = VmEnv::new();
        let err = execute(&prog, &env).unwrap_err();
        assert!(
            matches!(err, VmError::UnboundVariable(_)),
            "expected UnboundVariable, got {:?}",
            err
        );
    }

    // ── 21. VmStats.instructions_executed > 0 ─────────────────────────────
    #[test]
    fn test_stats_instructions_executed() {
        let expr = TLExpr::Constant(1.0);
        let prog = compile(&expr).expect("compile failed");
        let env = VmEnv::new();
        let (_val, stats) = execute_with_stats(&prog, &env).expect("execute failed");
        assert!(stats.instructions_executed > 0);
    }

    // ── 22. VmStats.max_stack_depth = 1 for simple push+halt ──────────────
    #[test]
    fn test_stats_max_stack_depth_single_push() {
        let mut prog = BytecodeProgram::new();
        prog.push(Instruction::PushNum(99.0));
        prog.push(Instruction::Halt);
        let env = VmEnv::new();
        let (_val, stats) = execute_with_stats(&prog, &env).expect("execute failed");
        assert_eq!(stats.max_stack_depth, 1, "single push should give depth 1");
    }

    // ── 23. TNorm(0.5, 0.5) = 0.25 ───────────────────────────────────────
    #[test]
    fn test_tnorm_product() {
        let expr = TLExpr::TNorm {
            kind: TNormKind::Product,
            left: Box::new(TLExpr::Constant(0.5)),
            right: Box::new(TLExpr::Constant(0.5)),
        };
        let result = eval(expr);
        match result {
            VmValue::Num(n) => {
                assert!((n - 0.25).abs() < 1e-10, "expected 0.25, got {}", n);
            }
            _ => panic!("expected Num, got {:?}", result),
        }
    }

    // ── 24. FuzzyNot(0.3) = 0.7 ──────────────────────────────────────────
    #[test]
    fn test_fuzzy_not() {
        let expr = TLExpr::FuzzyNot {
            kind: FuzzyNegationKind::Standard,
            expr: Box::new(TLExpr::Constant(0.3)),
        };
        let result = eval(expr);
        match result {
            VmValue::Num(n) => {
                assert!((n - 0.7).abs() < 1e-10, "expected 0.7, got {}", n);
            }
            _ => panic!("expected Num, got {:?}", result),
        }
    }

    // ── Bonus 25. TCoNorm(0.5, 0.5) = 0.75 ───────────────────────────────
    #[test]
    fn test_tconorm() {
        let expr = TLExpr::TCoNorm {
            kind: TCoNormKind::ProbabilisticSum,
            left: Box::new(TLExpr::Constant(0.5)),
            right: Box::new(TLExpr::Constant(0.5)),
        };
        let result = eval(expr);
        match result {
            VmValue::Num(n) => {
                // 0.5 + 0.5 - 0.5*0.5 = 0.75
                assert!((n - 0.75).abs() < 1e-10, "expected 0.75, got {}", n);
            }
            _ => panic!("expected Num, got {:?}", result),
        }
    }

    // ── Bonus 26. Nested arithmetic depth ─────────────────────────────────
    #[test]
    fn test_nested_arithmetic() {
        // (1 + 2) * (3 + 4) = 21
        let expr = TLExpr::mul(
            TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(2.0)),
            TLExpr::add(TLExpr::Constant(3.0), TLExpr::Constant(4.0)),
        );
        assert_eq!(eval(expr), VmValue::Num(21.0));
    }

    // ── Bonus 27. DivisionByZero error ────────────────────────────────────
    #[test]
    fn test_division_by_zero() {
        let mut prog = BytecodeProgram::new();
        prog.push(Instruction::PushNum(1.0));
        prog.push(Instruction::PushNum(0.0));
        prog.push(Instruction::Div);
        prog.push(Instruction::Halt);
        let env = VmEnv::new();
        let err = execute(&prog, &env).unwrap_err();
        assert!(
            matches!(err, VmError::DivisionByZero),
            "expected DivisionByZero, got {:?}",
            err
        );
    }

    // ── Bonus 28. Abs of negative number ──────────────────────────────────
    #[test]
    fn test_abs() {
        let expr = TLExpr::Abs(Box::new(TLExpr::Constant(-5.0)));
        assert_eq!(eval(expr), VmValue::Num(5.0));
    }

    // ── Bonus 29. Compile unsupported expr returns error ──────────────────
    #[test]
    fn test_compile_unsupported_forall() {
        use tensorlogic_ir::Term;
        let expr = TLExpr::forall("x", "D", TLExpr::pred("P", vec![Term::var("x")]));
        let err = compile(&expr).unwrap_err();
        assert!(
            matches!(err, CompileError::UnsupportedExpr(_)),
            "expected UnsupportedExpr, got {:?}",
            err
        );
    }

    // ── Bonus 30. Max depth exceeded ──────────────────────────────────────
    #[test]
    fn test_max_depth_exceeded() {
        // Build a deeply nested Add expression that exceeds depth 2.
        let inner = TLExpr::add(
            TLExpr::add(
                TLExpr::add(TLExpr::Constant(1.0), TLExpr::Constant(1.0)),
                TLExpr::Constant(1.0),
            ),
            TLExpr::Constant(1.0),
        );
        let err = compile_with_config(&inner, 1).unwrap_err();
        assert!(
            matches!(err, CompileError::MaxDepthExceeded),
            "expected MaxDepthExceeded, got {:?}",
            err
        );
    }
}
