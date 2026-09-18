//! CTL (Color Transform Language) interpreter for ACES color transforms.
//!
//! This module provides a full CTL interpreter capable of executing a subset of
//! the ACES Color Transformation Language specification. The interpreter is
//! intentionally self-contained: it calls the lexer ([`crate::ctl_lexer`]) and
//! parser ([`crate::ctl_parser`]) internally, so callers only need to supply a
//! CTL source string and input RGB values.
//!
//! # Supported language features
//!
//! - Variable declarations: `float x = 1.0;`
//! - Arithmetic: `+`, `-`, `*`, `/`
//! - Built-in math functions: `pow`, `sqrt`, `log`, `exp`, `abs`, `min`,
//!   `max`, `clamp`
//! - Comparison operators: `<`, `>`, `<=`, `>=`, `==`, `!=`
//! - Logical operators: `&&`, `||`, `!`
//! - Conditionals: `if (cond) { … } else { … }`
//! - Assignments to declared or parameter variables
//! - `return` (early exit from the main function)
//!
//! # Example
//!
//! ```rust
//! use oximedia_colormgmt::ctl_interpreter::CtlInterpreter;
//!
//! let src = r#"
//!     void main(
//!         output varying float rOut,
//!         output varying float gOut,
//!         output varying float bOut,
//!         input  varying float rIn,
//!         input  varying float gIn,
//!         input  varying float bIn
//!     ) {
//!         rOut = rIn * 2.0;
//!         gOut = gIn * 2.0;
//!         bOut = bIn * 2.0;
//!     }
//! "#;
//!
//! let mut interp = CtlInterpreter::new();
//! let result = interp.parse_and_execute(src, 0.25, 0.5, 0.1).expect("execute CTL");
//! assert!((result.r - 0.5).abs() < 1e-9);
//! assert!((result.g - 1.0).abs() < 1e-9);
//! assert!((result.b - 0.2).abs() < 1e-9);
//! ```

use std::collections::HashMap;

use crate::ctl_lexer::tokenize;
use crate::ctl_parser::{parse_main_function, CtlBinOp, CtlExpr, CtlUnOp};
use crate::error::ColorError;

// ─────────────────────────────────────────────────────────────────────────────
// Value type
// ─────────────────────────────────────────────────────────────────────────────

/// A runtime value in the CTL interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum CtlValue {
    /// A 64-bit floating-point number.
    Float(f64),
    /// A boolean.
    Bool(bool),
}

impl CtlValue {
    /// Extract the float value, or return an error.
    fn as_float(&self) -> Result<f64, ColorError> {
        match self {
            CtlValue::Float(v) => Ok(*v),
            CtlValue::Bool(b) => Err(ColorError::InvalidColor(format!(
                "expected float, got bool({b})"
            ))),
        }
    }

    /// Extract the bool value, or return an error.
    fn as_bool(&self) -> Result<bool, ColorError> {
        match self {
            CtlValue::Bool(b) => Ok(*b),
            CtlValue::Float(v) => Ok(*v != 0.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// The output of executing a CTL `main` function.
#[derive(Debug, Clone, PartialEq)]
pub struct CtlTransformResult {
    /// Output red channel.
    pub r: f64,
    /// Output green channel.
    pub g: f64,
    /// Output blue channel.
    pub b: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Early-return sentinel
// ─────────────────────────────────────────────────────────────────────────────

/// Internal sentinel used to implement `return` statements via Rust's `?`
/// operator without conflating control flow with errors.
enum EvalSignal {
    /// Normal completion with a value.
    Value(CtlValue),
    /// Early return triggered by a `return` statement.
    Return(Option<CtlValue>),
}

impl EvalSignal {
    fn into_value(self) -> CtlValue {
        match self {
            EvalSignal::Value(v) => v,
            EvalSignal::Return(Some(v)) => v,
            EvalSignal::Return(None) => CtlValue::Float(0.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpreter
// ─────────────────────────────────────────────────────────────────────────────

/// CTL interpreter that can parse and execute CTL source code.
///
/// The interpreter is reusable: call [`parse_and_execute`] multiple times with
/// different source strings and/or input values. Each call starts with a fresh
/// variable scope initialised from the input RGB parameters.
///
/// [`parse_and_execute`]: CtlInterpreter::parse_and_execute
pub struct CtlInterpreter {
    // Reserved for future use: global constants / precompiled transforms.
    _marker: (),
}

impl Default for CtlInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl CtlInterpreter {
    /// Create a new interpreter instance.
    #[must_use]
    pub fn new() -> Self {
        Self { _marker: () }
    }

    /// Parse the given CTL source, locate the `main()` function, execute it
    /// with the supplied RGB input values, and return the resulting RGB output.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::Parse`] if the CTL source cannot be tokenized or
    /// parsed, and [`ColorError::InvalidColor`] for runtime evaluation errors
    /// (type mismatches, division by zero, unknown variable, etc.).
    pub fn parse_and_execute(
        &mut self,
        ctl_source: &str,
        r: f64,
        g: f64,
        b: f64,
    ) -> Result<CtlTransformResult, ColorError> {
        // ── Lex ──────────────────────────────────────────────────────────────
        let tokens = tokenize(ctl_source)?;

        // ── Parse ─────────────────────────────────────────────────────────────
        let main_func = parse_main_function(&tokens)?;

        // ── Build variable scope ──────────────────────────────────────────────
        // Determine the input/output parameter names from the signature.
        let mut vars: HashMap<String, CtlValue> = HashMap::new();
        let mut output_names: Vec<String> = Vec::new();

        for param in &main_func.params {
            if param.is_output {
                // Output parameters start uninitialised (zero).
                vars.insert(param.name.clone(), CtlValue::Float(0.0));
                output_names.push(param.name.clone());
            } else {
                // Input parameters are bound to the caller's RGB values.
                // We use positional heuristics: first input → r, second → g, third → b.
                // We also support conventional CTL names (rIn, gIn, bIn, etc.).
                let val = input_value_for_param(&param.name, &vars, r, g, b);
                vars.insert(param.name.clone(), CtlValue::Float(val));
            }
        }

        // Provide conventional input names as fallback even if not in signature.
        vars.entry("rIn".to_string()).or_insert(CtlValue::Float(r));
        vars.entry("gIn".to_string()).or_insert(CtlValue::Float(g));
        vars.entry("bIn".to_string()).or_insert(CtlValue::Float(b));

        // ── Execute ───────────────────────────────────────────────────────────
        let mut exec = Execution { vars };
        exec.exec_block(&main_func.body)?;

        // ── Extract outputs ───────────────────────────────────────────────────
        // Try the conventional names first; fall back to first/second/third output param.
        let out_r = exec.resolve_output("rOut", &output_names, 0)?;
        let out_g = exec.resolve_output("gOut", &output_names, 1)?;
        let out_b = exec.resolve_output("bOut", &output_names, 2)?;

        Ok(CtlTransformResult {
            r: out_r,
            g: out_g,
            b: out_b,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input value binding helper
// ─────────────────────────────────────────────────────────────────────────────

/// Map an input parameter name to its caller-supplied value.
///
/// Follows the CTL convention: `rIn`/`redIn`/`R` → red, etc. Falls back to
/// positional counting using the already-populated `vars` map.
fn input_value_for_param(
    name: &str,
    vars: &HashMap<String, CtlValue>,
    r: f64,
    g: f64,
    b: f64,
) -> f64 {
    let lower = name.to_lowercase();
    if lower.starts_with('r') || lower.contains("red") {
        return r;
    }
    if lower.starts_with('g') || lower.contains("green") {
        return g;
    }
    if lower.starts_with('b') || lower.contains("blue") {
        return b;
    }
    // Positional: count how many input params have already been bound.
    let input_count = vars.values().filter(|_| true).count();
    match input_count % 3 {
        0 => r,
        1 => g,
        _ => b,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution state
// ─────────────────────────────────────────────────────────────────────────────

struct Execution {
    vars: HashMap<String, CtlValue>,
}

impl Execution {
    // ── Block / statement execution ───────────────────────────────────────────

    fn exec_block(&mut self, expr: &CtlExpr) -> Result<Option<CtlValue>, ColorError> {
        match expr {
            CtlExpr::Block(stmts) => {
                for stmt in stmts {
                    if let Some(retval) = self.exec_stmt(stmt)? {
                        return Ok(Some(retval));
                    }
                }
                Ok(None)
            }
            _ => self.exec_stmt(expr),
        }
    }

    /// Execute a statement. Returns `Some(value)` on early `return`.
    fn exec_stmt(&mut self, expr: &CtlExpr) -> Result<Option<CtlValue>, ColorError> {
        match expr {
            CtlExpr::VarDecl { name, init } => {
                let val = match init {
                    Some(init_expr) => self.eval(init_expr)?.into_value(),
                    None => CtlValue::Float(0.0),
                };
                self.vars.insert(name.clone(), val);
                Ok(None)
            }
            CtlExpr::Assign { target, value } => {
                let val = self.eval(value)?.into_value();
                self.vars.insert(target.clone(), val);
                Ok(None)
            }
            CtlExpr::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond_val = self.eval(cond)?.into_value().as_bool()?;
                if cond_val {
                    self.exec_block(then_body)
                } else if let Some(else_b) = else_body {
                    self.exec_block(else_b)
                } else {
                    Ok(None)
                }
            }
            CtlExpr::Return(val_expr) => {
                let retval = match val_expr {
                    Some(e) => Some(self.eval(e)?.into_value()),
                    None => None,
                };
                Ok(retval)
            }
            CtlExpr::Block(stmts) => {
                for stmt in stmts {
                    if let Some(retval) = self.exec_stmt(stmt)? {
                        return Ok(Some(retval));
                    }
                }
                Ok(None)
            }
            // Expression-as-statement (e.g. a bare function call)
            other => {
                self.eval(other)?;
                Ok(None)
            }
        }
    }

    // ── Expression evaluation ─────────────────────────────────────────────────

    fn eval(&mut self, expr: &CtlExpr) -> Result<EvalSignal, ColorError> {
        match expr {
            CtlExpr::FloatLit(v) => Ok(EvalSignal::Value(CtlValue::Float(*v))),
            CtlExpr::BoolLit(b) => Ok(EvalSignal::Value(CtlValue::Bool(*b))),

            CtlExpr::Var(name) => {
                let val = self.vars.get(name).ok_or_else(|| {
                    ColorError::InvalidColor(format!("undefined variable '{name}'"))
                })?;
                Ok(EvalSignal::Value(val.clone()))
            }

            CtlExpr::UnOp { op, operand } => {
                let v = self.eval(operand)?.into_value();
                let result = match op {
                    CtlUnOp::Neg => CtlValue::Float(-v.as_float()?),
                    CtlUnOp::Not => CtlValue::Bool(!v.as_bool()?),
                };
                Ok(EvalSignal::Value(result))
            }

            CtlExpr::BinOp { op, lhs, rhs } => {
                let lv = self.eval(lhs)?.into_value();
                let rv = self.eval(rhs)?.into_value();
                let result = self.eval_binop(op, &lv, &rv)?;
                Ok(EvalSignal::Value(result))
            }

            CtlExpr::Call { name, args } => {
                // Evaluate all arguments eagerly
                let mut arg_vals = Vec::with_capacity(args.len());
                for a in args {
                    arg_vals.push(self.eval(a)?.into_value());
                }
                let result = self.eval_call(name, &arg_vals)?;
                Ok(EvalSignal::Value(result))
            }

            // Assignments and declarations embedded in expressions: execute and
            // return 0.0 (they are statements, not pure expressions in CTL).
            CtlExpr::Assign { target, value } => {
                let val = self.eval(value)?.into_value();
                self.vars.insert(target.clone(), val);
                Ok(EvalSignal::Value(CtlValue::Float(0.0)))
            }
            CtlExpr::VarDecl { name, init } => {
                let val = match init {
                    Some(e) => self.eval(e)?.into_value(),
                    None => CtlValue::Float(0.0),
                };
                self.vars.insert(name.clone(), val);
                Ok(EvalSignal::Value(CtlValue::Float(0.0)))
            }
            CtlExpr::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond_val = self.eval(cond)?.into_value().as_bool()?;
                if cond_val {
                    let _ = self.exec_block(then_body)?;
                } else if let Some(else_b) = else_body {
                    let _ = self.exec_block(else_b)?;
                }
                Ok(EvalSignal::Value(CtlValue::Float(0.0)))
            }
            CtlExpr::Block(stmts) => {
                for stmt in stmts {
                    if let Some(retval) = self.exec_stmt(stmt)? {
                        return Ok(EvalSignal::Return(Some(retval)));
                    }
                }
                Ok(EvalSignal::Value(CtlValue::Float(0.0)))
            }
            CtlExpr::Return(val_expr) => {
                let retval = match val_expr {
                    Some(e) => Some(self.eval(e)?.into_value()),
                    None => None,
                };
                Ok(EvalSignal::Return(retval))
            }
        }
    }

    // ── Binary operators ──────────────────────────────────────────────────────

    fn eval_binop(
        &self,
        op: &CtlBinOp,
        lv: &CtlValue,
        rv: &CtlValue,
    ) -> Result<CtlValue, ColorError> {
        // Logical operators work on booleans
        if let CtlBinOp::And = op {
            return Ok(CtlValue::Bool(lv.as_bool()? && rv.as_bool()?));
        }
        if let CtlBinOp::Or = op {
            return Ok(CtlValue::Bool(lv.as_bool()? || rv.as_bool()?));
        }

        let l = lv.as_float()?;
        let r = rv.as_float()?;

        match op {
            CtlBinOp::Add => Ok(CtlValue::Float(l + r)),
            CtlBinOp::Sub => Ok(CtlValue::Float(l - r)),
            CtlBinOp::Mul => Ok(CtlValue::Float(l * r)),
            CtlBinOp::Div => {
                if r == 0.0 {
                    Err(ColorError::InvalidColor(
                        "division by zero in CTL expression".to_string(),
                    ))
                } else {
                    Ok(CtlValue::Float(l / r))
                }
            }
            CtlBinOp::Eq => Ok(CtlValue::Bool((l - r).abs() < f64::EPSILON * 1024.0)),
            CtlBinOp::Ne => Ok(CtlValue::Bool((l - r).abs() >= f64::EPSILON * 1024.0)),
            CtlBinOp::Lt => Ok(CtlValue::Bool(l < r)),
            CtlBinOp::Gt => Ok(CtlValue::Bool(l > r)),
            CtlBinOp::Le => Ok(CtlValue::Bool(l <= r)),
            CtlBinOp::Ge => Ok(CtlValue::Bool(l >= r)),
            CtlBinOp::And | CtlBinOp::Or => unreachable!("handled above"),
        }
    }

    // ── Built-in functions ────────────────────────────────────────────────────

    fn eval_call(&self, name: &str, args: &[CtlValue]) -> Result<CtlValue, ColorError> {
        let arity_err = |expected: usize| -> ColorError {
            ColorError::InvalidColor(format!(
                "built-in '{name}' expects {expected} argument(s), got {}",
                args.len()
            ))
        };

        match name {
            "pow" => {
                if args.len() != 2 {
                    return Err(arity_err(2));
                }
                let base = args[0].as_float()?;
                let exp = args[1].as_float()?;
                Ok(CtlValue::Float(base.powf(exp)))
            }
            "sqrt" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                let v = args[0].as_float()?;
                if v < 0.0 {
                    return Err(ColorError::InvalidColor(
                        "sqrt of negative value".to_string(),
                    ));
                }
                Ok(CtlValue::Float(v.sqrt()))
            }
            "log" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                let v = args[0].as_float()?;
                if v <= 0.0 {
                    return Err(ColorError::InvalidColor(
                        "log of non-positive value".to_string(),
                    ));
                }
                Ok(CtlValue::Float(v.ln()))
            }
            "log2" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                let v = args[0].as_float()?;
                if v <= 0.0 {
                    return Err(ColorError::InvalidColor(
                        "log2 of non-positive value".to_string(),
                    ));
                }
                Ok(CtlValue::Float(v.log2()))
            }
            "log10" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                let v = args[0].as_float()?;
                if v <= 0.0 {
                    return Err(ColorError::InvalidColor(
                        "log10 of non-positive value".to_string(),
                    ));
                }
                Ok(CtlValue::Float(v.log10()))
            }
            "exp" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.exp()))
            }
            "exp2" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.exp2()))
            }
            "abs" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.abs()))
            }
            "sign" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                let v = args[0].as_float()?;
                Ok(CtlValue::Float(if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }))
            }
            "floor" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.floor()))
            }
            "ceil" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.ceil()))
            }
            "round" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.round()))
            }
            "min" => {
                if args.len() != 2 {
                    return Err(arity_err(2));
                }
                Ok(CtlValue::Float(
                    args[0].as_float()?.min(args[1].as_float()?),
                ))
            }
            "max" => {
                if args.len() != 2 {
                    return Err(arity_err(2));
                }
                Ok(CtlValue::Float(
                    args[0].as_float()?.max(args[1].as_float()?),
                ))
            }
            "clamp" => {
                if args.len() != 3 {
                    return Err(arity_err(3));
                }
                let v = args[0].as_float()?;
                let lo = args[1].as_float()?;
                let hi = args[2].as_float()?;
                Ok(CtlValue::Float(v.clamp(lo, hi)))
            }
            "mix" | "lerp" => {
                if args.len() != 3 {
                    return Err(arity_err(3));
                }
                let a = args[0].as_float()?;
                let b = args[1].as_float()?;
                let t = args[2].as_float()?;
                Ok(CtlValue::Float(a + (b - a) * t))
            }
            "smoothstep" => {
                if args.len() != 3 {
                    return Err(arity_err(3));
                }
                let edge0 = args[0].as_float()?;
                let edge1 = args[1].as_float()?;
                let x = args[2].as_float()?;
                let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                Ok(CtlValue::Float(t * t * (3.0 - 2.0 * t)))
            }
            "sin" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.sin()))
            }
            "cos" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.cos()))
            }
            "tan" => {
                if args.len() != 1 {
                    return Err(arity_err(1));
                }
                Ok(CtlValue::Float(args[0].as_float()?.tan()))
            }
            "atan" => {
                if args.len() == 1 {
                    Ok(CtlValue::Float(args[0].as_float()?.atan()))
                } else if args.len() == 2 {
                    Ok(CtlValue::Float(
                        args[0].as_float()?.atan2(args[1].as_float()?),
                    ))
                } else {
                    Err(arity_err(1))
                }
            }
            "atan2" => {
                if args.len() != 2 {
                    return Err(arity_err(2));
                }
                Ok(CtlValue::Float(
                    args[0].as_float()?.atan2(args[1].as_float()?),
                ))
            }
            other => Err(ColorError::InvalidColor(format!(
                "unknown CTL built-in function '{other}'"
            ))),
        }
    }

    // ── Output extraction helper ──────────────────────────────────────────────

    fn resolve_output(
        &self,
        conventional: &str,
        output_names: &[String],
        positional_idx: usize,
    ) -> Result<f64, ColorError> {
        // Try the conventional name first (e.g. "rOut")
        if let Some(val) = self.vars.get(conventional) {
            return val.as_float();
        }
        // Fall back to the nth output parameter
        if let Some(name) = output_names.get(positional_idx) {
            if let Some(val) = self.vars.get(name) {
                return val.as_float();
            }
        }
        // Nothing found — default to 0.0 (permissive: some CTL snippets only
        // modify a subset of channels)
        Ok(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-built ACES CTL transform strings
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the CTL source for a simple linear exposure adjustment transform.
///
/// `exposure_stops` is the number of stops to adjust (positive = brighter,
/// negative = darker).
#[must_use]
pub fn exposure_ctl(exposure_stops: f64) -> String {
    let gain = 2_f64.powf(exposure_stops);
    format!(
        r#"
void main(
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    input  varying float rIn,
    input  varying float gIn,
    input  varying float bIn
) {{
    rOut = rIn * {gain:.10};
    gOut = gIn * {gain:.10};
    bOut = bIn * {gain:.10};
}}
"#
    )
}

/// Returns the CTL source for the ACEScg → ACES2065-1 matrix transform.
///
/// Uses the official ACES working group matrix (AP1 → AP0).
///
/// Matrix from S-2014-004 (ACES Color Space Conversion):
/// ```text
///  AP0_from_AP1 = [
///   [ 0.6954522414,  0.1406786965,  0.1638690622 ],
///   [ 0.0447945634,  0.8596711185,  0.0955343182 ],
///   [ -0.0055258826,  0.0040252103,  1.0015006723 ],
/// ]
/// ```
#[must_use]
pub fn acescg_to_aces2065_1_ctl() -> &'static str {
    r#"
// ACEScg (AP1) to ACES2065-1 (AP0) matrix transform
// S-2014-004 ACES Color Space Conversion
void main(
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    input  varying float rIn,
    input  varying float gIn,
    input  varying float bIn
) {
    rOut = rIn *  0.6954522414 + gIn * 0.1406786965 + bIn * 0.1638690622;
    gOut = rIn *  0.0447945634 + gIn * 0.8596711185 + bIn * 0.0955343182;
    bOut = rIn * -0.0055258826 + gIn * 0.0040252103 + bIn * 1.0015006723;
}
"#
}

/// Returns the CTL source for the ACES2065-1 → ACEScg (AP0 → AP1) matrix
/// transform (inverse of [`acescg_to_aces2065_1_ctl`]).
#[must_use]
pub fn aces2065_1_to_acescg_ctl() -> &'static str {
    r#"
// ACES2065-1 (AP0) to ACEScg (AP1) matrix transform
void main(
    output varying float rOut,
    output varying float gOut,
    output varying float bOut,
    input  varying float rIn,
    input  varying float gIn,
    input  varying float bIn
) {
    rOut = rIn *  1.4514393161 + gIn * -0.2365107469 + bIn * -0.2149285693;
    gOut = rIn * -0.0765537734 + gIn *  1.1762296998 + bIn * -0.0996759264;
    bOut = rIn *  0.0083161484 + gIn * -0.0060324498 + bIn *  0.9977163014;
}
"#
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn interp() -> CtlInterpreter {
        CtlInterpreter::new()
    }

    // ── Basic transform tests ─────────────────────────────────────────────────

    #[test]
    fn test_passthrough() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = rIn;
                gOut = gIn;
                bOut = bIn;
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 0.3, 0.5, 0.7)
            .expect("execute passthrough");
        assert!((result.r - 0.3).abs() < 1e-9);
        assert!((result.g - 0.5).abs() < 1e-9);
        assert!((result.b - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_exposure_double() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = rIn * 2.0;
                gOut = gIn * 2.0;
                bOut = bIn * 2.0;
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 0.25, 0.5, 0.1)
            .expect("execute exposure double");
        assert!((result.r - 0.5).abs() < 1e-9, "r: {}", result.r);
        assert!((result.g - 1.0).abs() < 1e-9, "g: {}", result.g);
        assert!((result.b - 0.2).abs() < 1e-9, "b: {}", result.b);
    }

    #[test]
    fn test_exposure_ctl_helper_one_stop() {
        let src = exposure_ctl(1.0); // 1 stop = ×2
        let result = interp()
            .parse_and_execute(&src, 0.5, 0.25, 0.125)
            .expect("execute one stop exposure");
        assert!((result.r - 1.0).abs() < 1e-9, "r: {}", result.r);
        assert!((result.g - 0.5).abs() < 1e-9, "g: {}", result.g);
        assert!((result.b - 0.25).abs() < 1e-9, "b: {}", result.b);
    }

    #[test]
    fn test_acescg_to_aces2065_1_identity_white() {
        // A pure-white ACEScg value (1,1,1) should remain close to (1,1,1) in AP0
        // (not exact but row sums should be approximately 1).
        let src = acescg_to_aces2065_1_ctl();
        let result = interp()
            .parse_and_execute(src, 1.0, 1.0, 1.0)
            .expect("execute acescg white");
        assert!((result.r - 1.0).abs() < 0.01, "r: {}", result.r);
        assert!((result.g - 1.0).abs() < 0.01, "g: {}", result.g);
        assert!((result.b - 1.0).abs() < 0.01, "b: {}", result.b);
    }

    #[test]
    fn test_acescg_to_aces2065_1_black() {
        let src = acescg_to_aces2065_1_ctl();
        let result = interp()
            .parse_and_execute(src, 0.0, 0.0, 0.0)
            .expect("execute acescg black");
        assert!((result.r).abs() < 1e-9);
        assert!((result.g).abs() < 1e-9);
        assert!((result.b).abs() < 1e-9);
    }

    #[test]
    fn test_roundtrip_acescg_aces2065_1() {
        // Apply AP1→AP0 then AP0→AP1; result should be very close to input.
        let r0 = 0.4;
        let g0 = 0.3;
        let b0 = 0.2;

        let fwd = acescg_to_aces2065_1_ctl();
        let mid = interp()
            .parse_and_execute(fwd, r0, g0, b0)
            .expect("execute forward transform");

        let inv = aces2065_1_to_acescg_ctl();
        let result = interp()
            .parse_and_execute(inv, mid.r, mid.g, mid.b)
            .expect("execute inverse transform");

        assert!((result.r - r0).abs() < 1e-6, "r: {} vs {}", result.r, r0);
        assert!((result.g - g0).abs() < 1e-6, "g: {} vs {}", result.g, g0);
        assert!((result.b - b0).abs() < 1e-6, "b: {} vs {}", result.b, b0);
    }

    // ── Built-in function tests ───────────────────────────────────────────────

    #[test]
    fn test_clamp_builtin() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = clamp(rIn, 0.0, 1.0);
                gOut = clamp(gIn, 0.0, 1.0);
                bOut = clamp(bIn, 0.0, 1.0);
            }
        "#;
        let result = interp()
            .parse_and_execute(src, -0.5, 0.5, 2.0)
            .expect("execute clamp");
        assert!((result.r - 0.0).abs() < 1e-9, "r: {}", result.r);
        assert!((result.g - 0.5).abs() < 1e-9, "g: {}", result.g);
        assert!((result.b - 1.0).abs() < 1e-9, "b: {}", result.b);
    }

    #[test]
    fn test_pow_builtin() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = pow(rIn, 2.0);
                gOut = pow(gIn, 0.5);
                bOut = pow(bIn, 1.0);
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 3.0, 9.0, 5.0)
            .expect("execute pow");
        assert!((result.r - 9.0).abs() < 1e-9, "r: {}", result.r);
        assert!((result.g - 3.0).abs() < 1e-6, "g: {}", result.g);
        assert!((result.b - 5.0).abs() < 1e-9, "b: {}", result.b);
    }

    #[test]
    fn test_sqrt_builtin() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = sqrt(rIn);
                gOut = sqrt(gIn);
                bOut = sqrt(bIn);
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 4.0, 9.0, 16.0)
            .expect("execute sqrt");
        assert!((result.r - 2.0).abs() < 1e-9);
        assert!((result.g - 3.0).abs() < 1e-9);
        assert!((result.b - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_min_max_builtin() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = min(rIn, 0.5);
                gOut = max(gIn, 0.5);
                bOut = min(bIn, max(rIn, gIn));
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 0.8, 0.2, 0.9)
            .expect("execute min/max");
        assert!((result.r - 0.5).abs() < 1e-9);
        assert!((result.g - 0.5).abs() < 1e-9);
        assert!((result.b - 0.8).abs() < 1e-9);
    }

    // ── Conditional tests ─────────────────────────────────────────────────────

    #[test]
    fn test_if_else_branch() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                if (rIn > 0.5) {
                    rOut = 1.0;
                } else {
                    rOut = 0.0;
                }
                gOut = gIn;
                bOut = bIn;
            }
        "#;
        let above = interp()
            .parse_and_execute(src, 0.8, 0.5, 0.5)
            .expect("execute if above");
        assert!((above.r - 1.0).abs() < 1e-9);

        let below = interp()
            .parse_and_execute(src, 0.2, 0.5, 0.5)
            .expect("execute if below");
        assert!((below.r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_nested_if() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                if (rIn > 0.0) {
                    if (rIn > 0.5) {
                        rOut = 1.0;
                    } else {
                        rOut = 0.5;
                    }
                } else {
                    rOut = 0.0;
                }
                gOut = gIn;
                bOut = bIn;
            }
        "#;
        let r1 = interp()
            .parse_and_execute(src, 0.8, 0.0, 0.0)
            .expect("nested if high")
            .r;
        let r2 = interp()
            .parse_and_execute(src, 0.3, 0.0, 0.0)
            .expect("nested if mid")
            .r;
        let r3 = interp()
            .parse_and_execute(src, -0.1, 0.0, 0.0)
            .expect("nested if low")
            .r;
        assert!((r1 - 1.0).abs() < 1e-9);
        assert!((r2 - 0.5).abs() < 1e-9);
        assert!((r3 - 0.0).abs() < 1e-9);
    }

    // ── Variable declaration tests ────────────────────────────────────────────

    #[test]
    fn test_local_variable() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                float gain = 3.0;
                rOut = rIn * gain;
                gOut = gIn * gain;
                bOut = bIn * gain;
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 0.1, 0.2, 0.3)
            .expect("execute local var");
        assert!((result.r - 0.3).abs() < 1e-9);
        assert!((result.g - 0.6).abs() < 1e-9);
        assert!((result.b - 0.9).abs() < 1e-9);
    }

    // ── Arithmetic tests ──────────────────────────────────────────────────────

    #[test]
    fn test_chained_arithmetic() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = (rIn + gIn) * bIn - 0.1;
                gOut = rIn * rIn + gIn * gIn;
                bOut = bIn / 2.0 + 0.25;
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 0.5, 0.5, 2.0)
            .expect("execute chained arithmetic");
        assert!((result.r - ((0.5 + 0.5) * 2.0 - 0.1)).abs() < 1e-9);
        assert!((result.g - (0.5 * 0.5 + 0.5 * 0.5)).abs() < 1e-9);
        assert!((result.b - (2.0 / 2.0 + 0.25)).abs() < 1e-9);
    }

    #[test]
    fn test_unary_negation() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = -rIn;
                gOut = -gIn;
                bOut = -bIn;
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 0.5, -0.3, 1.0)
            .expect("execute unary negation");
        assert!((result.r - (-0.5)).abs() < 1e-9);
        assert!((result.g - 0.3).abs() < 1e-9);
        assert!((result.b - (-1.0)).abs() < 1e-9);
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_division_by_zero_error() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = rIn / 0.0;
                gOut = gIn;
                bOut = bIn;
            }
        "#;
        let result = interp().parse_and_execute(src, 1.0, 0.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_undefined_variable_error() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = undefinedVar;
                gOut = gIn;
                bOut = bIn;
            }
        "#;
        let result = interp().parse_and_execute(src, 1.0, 0.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sqrt_negative_error() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = sqrt(-1.0);
                gOut = gIn;
                bOut = bIn;
            }
        "#;
        let result = interp().parse_and_execute(src, 1.0, 0.0, 0.0);
        assert!(result.is_err());
    }

    // ── Log / exp tests ───────────────────────────────────────────────────────

    #[test]
    fn test_log_exp_roundtrip() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = exp(log(rIn));
                gOut = gIn;
                bOut = bIn;
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 2.718281828, 0.0, 0.0)
            .expect("execute log/exp roundtrip");
        assert!((result.r - 2.718281828).abs() < 1e-6);
    }

    // ── Smoothstep test ───────────────────────────────────────────────────────

    #[test]
    fn test_smoothstep_builtin() {
        let src = r#"
            void main(
                output varying float rOut,
                output varying float gOut,
                output varying float bOut,
                input  varying float rIn,
                input  varying float gIn,
                input  varying float bIn
            ) {
                rOut = smoothstep(0.0, 1.0, rIn);
                gOut = smoothstep(0.0, 1.0, 0.0);
                bOut = smoothstep(0.0, 1.0, 1.0);
            }
        "#;
        let result = interp()
            .parse_and_execute(src, 0.5, 0.0, 1.0)
            .expect("execute smoothstep");
        assert!(
            (result.r - 0.5).abs() < 1e-9,
            "smoothstep(0,1,0.5) = {}",
            result.r
        );
        assert!(
            (result.g - 0.0).abs() < 1e-9,
            "smoothstep(0,1,0) = {}",
            result.g
        );
        assert!(
            (result.b - 1.0).abs() < 1e-9,
            "smoothstep(0,1,1) = {}",
            result.b
        );
    }

    // ── Default trait ─────────────────────────────────────────────────────────

    #[test]
    fn test_default_trait() {
        let mut interp: CtlInterpreter = Default::default();
        let src = r#"void main(output varying float rOut, output varying float gOut, output varying float bOut, input varying float rIn, input varying float gIn, input varying float bIn) { rOut = rIn; gOut = gIn; bOut = bIn; }"#;
        let result = interp
            .parse_and_execute(src, 0.1, 0.2, 0.3)
            .expect("execute default trait");
        assert!((result.r - 0.1).abs() < 1e-9);
    }
}
