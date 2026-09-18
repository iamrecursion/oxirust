//! Symbolic expression representation and manipulation
//!
//! This module provides the foundation for symbolic computation,
//! including expression trees, variables, and simplification rules.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Neg as StdNeg, Sub as StdSub};

// Import enum variants for local use in this module
use Pattern::{DifferenceOfSquares, PythagoreanIdentity, SumOfSquares};
use SymbolicExpression::{
    Abs, Add, Atan, Constant, Cos, Cosh, Div, Exp, Ln, Mul, Neg, Pow, Sin, Sinh, Sqrt, Sub, Tan,
    Tanh, Var,
};

/// Represents a symbolic variable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    pub name: String,
    pub index: Option<usize>, // For indexed variables like y[0], y[1]
}

impl Variable {
    /// Create a new variable
    pub fn new(name: impl Into<String>) -> Self {
        Variable {
            name: name.into(),
            index: None,
        }
    }

    /// Create an indexed variable
    pub fn indexed(name: impl Into<String>, index: usize) -> Self {
        Variable {
            name: name.into(),
            index: Some(index),
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.index {
            Some(idx) => write!(f, "{}[{}]", self.name, idx),
            None => write!(f, "{}", self.name),
        }
    }
}

/// Symbolic expression types
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpression<F: IntegrateFloat> {
    /// Constant value
    Constant(F),
    /// Variable
    Var(Variable),
    /// Addition
    Add(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Subtraction
    Sub(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Multiplication
    Mul(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Division
    Div(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Power
    Pow(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// Negation
    Neg(Box<SymbolicExpression<F>>),
    /// Sine
    Sin(Box<SymbolicExpression<F>>),
    /// Cosine
    Cos(Box<SymbolicExpression<F>>),
    /// Exponential
    Exp(Box<SymbolicExpression<F>>),
    /// Natural logarithm
    Ln(Box<SymbolicExpression<F>>),
    /// Square root
    Sqrt(Box<SymbolicExpression<F>>),
    /// Tangent
    Tan(Box<SymbolicExpression<F>>),
    /// Arctangent
    Atan(Box<SymbolicExpression<F>>),
    /// Hyperbolic sine
    Sinh(Box<SymbolicExpression<F>>),
    /// Hyperbolic cosine
    Cosh(Box<SymbolicExpression<F>>),
    /// Hyperbolic tangent
    Tanh(Box<SymbolicExpression<F>>),
    /// Absolute value
    Abs(Box<SymbolicExpression<F>>),
}

impl<F: IntegrateFloat> SymbolicExpression<F> {
    /// Create a constant expression
    pub fn constant(value: F) -> Self {
        SymbolicExpression::Constant(value)
    }

    /// Create a variable expression
    pub fn var(name: impl Into<String>) -> Self {
        SymbolicExpression::Var(Variable::new(name))
    }

    /// Create an indexed variable expression
    pub fn indexedvar(name: impl Into<String>, index: usize) -> Self {
        SymbolicExpression::Var(Variable::indexed(name, index))
    }

    /// Create a tangent expression
    pub fn tan(expr: SymbolicExpression<F>) -> Self {
        SymbolicExpression::Tan(Box::new(expr))
    }

    /// Create an arctangent expression
    pub fn atan(expr: SymbolicExpression<F>) -> Self {
        SymbolicExpression::Atan(Box::new(expr))
    }

    /// Create a hyperbolic sine expression
    pub fn sinh(expr: SymbolicExpression<F>) -> Self {
        SymbolicExpression::Sinh(Box::new(expr))
    }

    /// Create a hyperbolic cosine expression
    pub fn cosh(expr: SymbolicExpression<F>) -> Self {
        SymbolicExpression::Cosh(Box::new(expr))
    }

    /// Create a hyperbolic tangent expression
    pub fn tanh(expr: SymbolicExpression<F>) -> Self {
        SymbolicExpression::Tanh(Box::new(expr))
    }

    /// Create an absolute value expression
    pub fn abs(expr: SymbolicExpression<F>) -> Self {
        SymbolicExpression::Abs(Box::new(expr))
    }

    /// Differentiate with respect to a variable
    pub fn differentiate(&self, var: &Variable) -> SymbolicExpression<F> {
        use SymbolicExpression::*;

        match self {
            Constant(_) => Constant(F::zero()),
            Var(v) => {
                if v == var {
                    Constant(F::one())
                } else {
                    Constant(F::zero())
                }
            }
            Add(a, b) => Add(
                Box::new(a.differentiate(var)),
                Box::new(b.differentiate(var)),
            ),
            Sub(a, b) => Sub(
                Box::new(a.differentiate(var)),
                Box::new(b.differentiate(var)),
            ),
            Mul(a, b) => {
                // Product rule: (a*b)' = a'*b + a*b'
                Add(
                    Box::new(Mul(Box::new(a.differentiate(var)), b.clone())),
                    Box::new(Mul(a.clone(), Box::new(b.differentiate(var)))),
                )
            }
            Div(a, b) => {
                // Quotient rule: (a/b)' = (a'*b - a*b')/b²
                Div(
                    Box::new(Sub(
                        Box::new(Mul(Box::new(a.differentiate(var)), b.clone())),
                        Box::new(Mul(a.clone(), Box::new(b.differentiate(var)))),
                    )),
                    Box::new(Mul(b.clone(), b.clone())),
                )
            }
            Pow(a, b) => {
                // For now, handle only constant powers
                if let Constant(n) = &**b {
                    // Power rule: (a^n)' = n * a^(n-1) * a'
                    Mul(
                        Box::new(Mul(
                            Box::new(Constant(*n)),
                            Box::new(Pow(a.clone(), Box::new(Constant(*n - F::one())))),
                        )),
                        Box::new(a.differentiate(var)),
                    )
                } else {
                    // General case: a^b = exp(b*ln(a))
                    let exp_expr = Exp(Box::new(Mul(b.clone(), Box::new(Ln(a.clone())))));
                    exp_expr.differentiate(var)
                }
            }
            Neg(a) => Neg(Box::new(a.differentiate(var))),
            Sin(a) => {
                // (sin(a))' = cos(a) * a'
                Mul(Box::new(Cos(a.clone())), Box::new(a.differentiate(var)))
            }
            Cos(a) => {
                // (cos(a))' = -sin(a) * a'
                Neg(Box::new(Mul(
                    Box::new(Sin(a.clone())),
                    Box::new(a.differentiate(var)),
                )))
            }
            Exp(a) => {
                // (e^a)' = e^a * a'
                Mul(Box::new(Exp(a.clone())), Box::new(a.differentiate(var)))
            }
            Ln(a) => {
                // (ln(a))' = a'/a
                Div(Box::new(a.differentiate(var)), a.clone())
            }
            Sqrt(a) => {
                // (sqrt(a))' = a'/(2*sqrt(a))
                Div(
                    Box::new(a.differentiate(var)),
                    Box::new(Mul(
                        Box::new(Constant(
                            F::from(2.0).expect("Failed to convert constant to float"),
                        )),
                        Box::new(Sqrt(a.clone())),
                    )),
                )
            }
            Tan(a) => {
                // (tan(a))' = sec²(a) * a' = a' / cos²(a)
                Div(
                    Box::new(a.differentiate(var)),
                    Box::new(Pow(
                        Box::new(Cos(a.clone())),
                        Box::new(Constant(
                            F::from(2.0).expect("Failed to convert constant to float"),
                        )),
                    )),
                )
            }
            Atan(a) => {
                // (atan(a))' = a' / (1 + a²)
                Div(
                    Box::new(a.differentiate(var)),
                    Box::new(Add(
                        Box::new(Constant(F::one())),
                        Box::new(Pow(
                            a.clone(),
                            Box::new(Constant(
                                F::from(2.0).expect("Failed to convert constant to float"),
                            )),
                        )),
                    )),
                )
            }
            Sinh(a) => {
                // (sinh(a))' = cosh(a) * a'
                Mul(Box::new(Cosh(a.clone())), Box::new(a.differentiate(var)))
            }
            Cosh(a) => {
                // (cosh(a))' = sinh(a) * a'
                Mul(Box::new(Sinh(a.clone())), Box::new(a.differentiate(var)))
            }
            Tanh(a) => {
                // (tanh(a))' = sech²(a) * a' = a' / cosh²(a)
                Div(
                    Box::new(a.differentiate(var)),
                    Box::new(Pow(
                        Box::new(Cosh(a.clone())),
                        Box::new(Constant(
                            F::from(2.0).expect("Failed to convert constant to float"),
                        )),
                    )),
                )
            }
            Abs(a) => {
                // d/dx|a| = a'/|a| * a = sign(a) * a'
                // For symbolic differentiation, we'll use a/|a| as sign function
                Mul(
                    Box::new(Div(a.clone(), Box::new(Abs(a.clone())))),
                    Box::new(a.differentiate(var)),
                )
            }
        }
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, values: &HashMap<Variable, F>) -> IntegrateResult<F> {
        match self {
            Constant(c) => Ok(*c),
            Var(v) => values.get(v).copied().ok_or_else(|| {
                IntegrateError::ComputationError(format!("Variable {v} not found in values"))
            }),
            Add(a, b) => Ok(a.evaluate(values)? + b.evaluate(values)?),
            Sub(a, b) => Ok(a.evaluate(values)? - b.evaluate(values)?),
            Mul(a, b) => Ok(a.evaluate(values)? * b.evaluate(values)?),
            Div(a, b) => {
                let b_val = b.evaluate(values)?;
                if b_val.abs() < F::epsilon() {
                    Err(IntegrateError::ComputationError(
                        "Division by zero".to_string(),
                    ))
                } else {
                    Ok(a.evaluate(values)? / b_val)
                }
            }
            Pow(a, b) => Ok(a.evaluate(values)?.powf(b.evaluate(values)?)),
            Neg(a) => Ok(-a.evaluate(values)?),
            Sin(a) => Ok(a.evaluate(values)?.sin()),
            Cos(a) => Ok(a.evaluate(values)?.cos()),
            Exp(a) => Ok(a.evaluate(values)?.exp()),
            Ln(a) => {
                let a_val = a.evaluate(values)?;
                if a_val <= F::zero() {
                    Err(IntegrateError::ComputationError(
                        "Logarithm of non-positive value".to_string(),
                    ))
                } else {
                    Ok(a_val.ln())
                }
            }
            Sqrt(a) => {
                let a_val = a.evaluate(values)?;
                if a_val < F::zero() {
                    Err(IntegrateError::ComputationError(
                        "Square root of negative value".to_string(),
                    ))
                } else {
                    Ok(a_val.sqrt())
                }
            }
            Tan(a) => Ok(a.evaluate(values)?.tan()),
            Atan(a) => Ok(a.evaluate(values)?.atan()),
            Sinh(a) => Ok(a.evaluate(values)?.sinh()),
            Cosh(a) => Ok(a.evaluate(values)?.cosh()),
            Tanh(a) => Ok(a.evaluate(values)?.tanh()),
            Abs(a) => Ok(a.evaluate(values)?.abs()),
        }
    }

    /// Get all variables in the expression
    pub fn variables(&self) -> Vec<Variable> {
        let mut vars = Vec::new();

        match self {
            Constant(_) => {}
            Var(v) => vars.push(v.clone()),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Pow(a, b) => {
                vars.extend(a.variables());
                vars.extend(b.variables());
            }
            Neg(a) | Sin(a) | Cos(a) | Exp(a) | Ln(a) | Sqrt(a) | Tan(a) | Atan(a) | Sinh(a)
            | Cosh(a) | Tanh(a) | Abs(a) => {
                vars.extend(a.variables());
            }
        }

        // Remove duplicates
        vars.sort_by(|a, b| match (&a.name, &b.name) {
            (n1, n2) if n1 != n2 => n1.cmp(n2),
            _ => a.index.cmp(&b.index),
        });
        vars.dedup();
        vars
    }
}

/// Common mathematical patterns
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern<F: IntegrateFloat> {
    /// a^2 + b^2
    SumOfSquares(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// a^2 - b^2
    DifferenceOfSquares(Box<SymbolicExpression<F>>, Box<SymbolicExpression<F>>),
    /// sin^2(x) + cos^2(x) = 1
    PythagoreanIdentity(Box<SymbolicExpression<F>>),
    /// e^(i*x) = cos(x) + i*sin(x) (Euler's formula)
    EulerFormula(Box<SymbolicExpression<F>>),
}

/// Pattern matching for common mathematical expressions
#[allow(dead_code)]
pub fn match_pattern<F: IntegrateFloat>(expr: &SymbolicExpression<F>) -> Option<Pattern<F>> {
    match expr {
        // Match a^2 + b^2 (sum of squares)
        Add(a, b) => {
            if let (Pow(base_a, exp_a), Pow(base_b, exp_b)) = (a.as_ref(), b.as_ref()) {
                if let (Constant(n_a), Constant(n_b)) = (exp_a.as_ref(), exp_b.as_ref()) {
                    if (*n_a - F::from(2.0).expect("Failed to convert constant to float")).abs()
                        < F::epsilon()
                        && (*n_b - F::from(2.0).expect("Failed to convert constant to float")).abs()
                            < F::epsilon()
                    {
                        return Some(Pattern::SumOfSquares(base_a.clone(), base_b.clone()));
                    }
                }
            }

            // Match sin^2(x) + cos^2(x) = 1
            if let (Pow(sin_base, sin_exp), Pow(cos_base, cos_exp)) = (a.as_ref(), b.as_ref()) {
                if let (Sin(sin_arg), Cos(cos_arg), Constant(n1), Constant(n2)) = (
                    sin_base.as_ref(),
                    cos_base.as_ref(),
                    sin_exp.as_ref(),
                    cos_exp.as_ref(),
                ) {
                    if match_expressions(sin_arg, cos_arg)
                        && (*n1 - F::from(2.0).expect("Failed to convert constant to float")).abs()
                            < F::epsilon()
                        && (*n2 - F::from(2.0).expect("Failed to convert constant to float")).abs()
                            < F::epsilon()
                    {
                        return Some(Pattern::PythagoreanIdentity(sin_arg.clone()));
                    }
                }
            }
            None
        }
        // Match a^2 - b^2 (difference of squares)
        Sub(a, b) => {
            if let (Pow(base_a, exp_a), Pow(base_b, exp_b)) = (a.as_ref(), b.as_ref()) {
                if let (Constant(n_a), Constant(n_b)) = (exp_a.as_ref(), exp_b.as_ref()) {
                    if (*n_a - F::from(2.0).expect("Failed to convert constant to float")).abs()
                        < F::epsilon()
                        && (*n_b - F::from(2.0).expect("Failed to convert constant to float")).abs()
                            < F::epsilon()
                    {
                        return Some(Pattern::DifferenceOfSquares(base_a.clone(), base_b.clone()));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Helper function to check if two expressions match structurally
#[allow(dead_code)]
fn match_expressions<F: IntegrateFloat>(
    expr1: &SymbolicExpression<F>,
    expr2: &SymbolicExpression<F>,
) -> bool {
    match (expr1, expr2) {
        (Constant(a), Constant(b)) => (*a - *b).abs() < F::epsilon(),
        (Var(a), Var(b)) => a == b,
        _ => false,
    }
}

/// Apply pattern-based simplifications
#[allow(dead_code)]
pub fn pattern_simplify<F: IntegrateFloat>(expr: &SymbolicExpression<F>) -> SymbolicExpression<F> {
    if let Some(pattern) = match_pattern(expr) {
        match pattern {
            Pattern::DifferenceOfSquares(a, b) => {
                // a^2 - b^2 = (a + b)(a - b)
                Mul(Box::new(Add(a.clone(), b.clone())), Box::new(Sub(a, b)))
            }
            Pattern::PythagoreanIdentity(_) => {
                // sin^2(x) + cos^2(x) = 1
                Constant(F::one())
            }
            _ => expr.clone(),
        }
    } else {
        // Try recursive pattern simplification
        match expr {
            Add(a, b) => {
                let a_simp = pattern_simplify(a);
                let b_simp = pattern_simplify(b);
                pattern_simplify(&Add(Box::new(a_simp), Box::new(b_simp)))
            }
            Sub(a, b) => {
                let a_simp = pattern_simplify(a);
                let b_simp = pattern_simplify(b);
                pattern_simplify(&Sub(Box::new(a_simp), Box::new(b_simp)))
            }
            Mul(a, b) => {
                let a_simp = pattern_simplify(a);
                let b_simp = pattern_simplify(b);
                Mul(Box::new(a_simp), Box::new(b_simp))
            }
            _ => expr.clone(),
        }
    }
}

/// Simplify a symbolic expression
#[allow(dead_code)]
pub fn simplify<F: IntegrateFloat>(expr: &SymbolicExpression<F>) -> SymbolicExpression<F> {
    match expr {
        // Identity simplifications
        Add(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) => Constant(*x + *y),
                (Constant(x), _) if x.abs() < F::epsilon() => b_simp,
                (_, Constant(y)) if y.abs() < F::epsilon() => a_simp,
                _ => Add(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Sub(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) => Constant(*x - *y),
                (_, Constant(y)) if y.abs() < F::epsilon() => a_simp,
                _ => Sub(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Mul(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) => Constant(*x * *y),
                (Constant(x), _) if x.abs() < F::epsilon() => Constant(F::zero()),
                (_, Constant(y)) if y.abs() < F::epsilon() => Constant(F::zero()),
                (Constant(x), _) if (*x - F::one()).abs() < F::epsilon() => b_simp,
                (_, Constant(y)) if (*y - F::one()).abs() < F::epsilon() => a_simp,
                _ => Mul(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Div(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) if y.abs() > F::epsilon() => Constant(*x / *y),
                (Constant(x), _) if x.abs() < F::epsilon() => Constant(F::zero()),
                (_, Constant(y)) if (*y - F::one()).abs() < F::epsilon() => a_simp,
                _ => Div(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Neg(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) => Constant(-*x),
                Neg(inner) => (**inner).clone(),
                _ => Neg(Box::new(a_simp)),
            }
        }
        Pow(a, b) => {
            let a_simp = simplify(a);
            let b_simp = simplify(b);
            match (&a_simp, &b_simp) {
                (Constant(x), Constant(y)) => Constant(x.powf(*y)),
                (_, Constant(y)) if y.abs() < F::epsilon() => Constant(F::one()), // a^0 = 1
                (_, Constant(y)) if (*y - F::one()).abs() < F::epsilon() => a_simp, // a^1 = a
                (Constant(x), _) if x.abs() < F::epsilon() => Constant(F::zero()), // 0^b = 0 (for b > 0)
                (Constant(x), _) if (*x - F::one()).abs() < F::epsilon() => Constant(F::one()), // 1^b = 1
                _ => Pow(Box::new(a_simp), Box::new(b_simp)),
            }
        }
        Exp(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) => Constant(x.exp()),
                Ln(inner) => (**inner).clone(), // exp(ln(x)) = x
                _ => Exp(Box::new(a_simp)),
            }
        }
        Ln(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) if *x > F::zero() => Constant(x.ln()),
                Exp(inner) => (**inner).clone(), // ln(exp(x)) = x
                Constant(x) if (*x - F::one()).abs() < F::epsilon() => Constant(F::zero()), // ln(1) = 0
                _ => Ln(Box::new(a_simp)),
            }
        }
        Sin(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) => Constant(x.sin()),
                Neg(inner) => Neg(Box::new(Sin(inner.clone()))), // sin(-x) = -sin(x)
                _ => Sin(Box::new(a_simp)),
            }
        }
        Cos(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) => Constant(x.cos()),
                Neg(inner) => Cos(inner.clone()), // cos(-x) = cos(x)
                _ => Cos(Box::new(a_simp)),
            }
        }
        Tan(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) => Constant(x.tan()),
                Neg(inner) => Neg(Box::new(Tan(inner.clone()))), // tan(-x) = -tan(x)
                _ => Tan(Box::new(a_simp)),
            }
        }
        Sqrt(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) if *x >= F::zero() => Constant(x.sqrt()),
                Pow(base, exp) => {
                    if let Constant(n) = &**exp {
                        // sqrt(x^n) = x^(n/2)
                        Pow(
                            base.clone(),
                            Box::new(Constant(
                                *n / F::from(2.0).expect("Failed to convert constant to float"),
                            )),
                        )
                    } else {
                        Sqrt(Box::new(a_simp))
                    }
                }
                _ => Sqrt(Box::new(a_simp)),
            }
        }
        Abs(a) => {
            let a_simp = simplify(a);
            match &a_simp {
                Constant(x) => Constant(x.abs()),
                Neg(inner) => Abs(inner.clone()), // |−x| = |x|
                Abs(inner) => Abs(inner.clone()), // ||x|| = |x|
                _ => Abs(Box::new(a_simp)),
            }
        }
        _ => expr.clone(),
    }
}

/// Enhanced simplify that combines algebraic and pattern-based simplification
#[allow(dead_code)]
pub fn deep_simplify<F: IntegrateFloat>(expr: &SymbolicExpression<F>) -> SymbolicExpression<F> {
    // First apply algebraic simplification
    let algebraic_simplified = simplify(expr);
    // Then apply pattern-based simplification
    pattern_simplify(&algebraic_simplified)
}

// Operator overloading for easier expression building
impl<F: IntegrateFloat> StdAdd for SymbolicExpression<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        SymbolicExpression::Add(Box::new(self), Box::new(rhs))
    }
}

impl<F: IntegrateFloat> StdSub for SymbolicExpression<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        SymbolicExpression::Sub(Box::new(self), Box::new(rhs))
    }
}

impl<F: IntegrateFloat> StdMul for SymbolicExpression<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        SymbolicExpression::Mul(Box::new(self), Box::new(rhs))
    }
}

impl<F: IntegrateFloat> StdDiv for SymbolicExpression<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        SymbolicExpression::Div(Box::new(self), Box::new(rhs))
    }
}

impl<F: IntegrateFloat> StdNeg for SymbolicExpression<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        SymbolicExpression::Neg(Box::new(self))
    }
}

impl<F: IntegrateFloat> fmt::Display for SymbolicExpression<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant(c) => write!(f, "{c}"),
            Var(v) => write!(f, "{v}"),
            Add(a, b) => write!(f, "({a} + {b})"),
            Sub(a, b) => write!(f, "({a} - {b})"),
            Mul(a, b) => write!(f, "({a} * {b})"),
            Div(a, b) => write!(f, "({a} / {b})"),
            Pow(a, b) => write!(f, "({a} ^ {b})"),
            Neg(a) => write!(f, "(-{a})"),
            Sin(a) => write!(f, "sin({a})"),
            Cos(a) => write!(f, "cos({a})"),
            Exp(a) => write!(f, "exp({a})"),
            Ln(a) => write!(f, "ln({a})"),
            Sqrt(a) => write!(f, "sqrt({a})"),
            Tan(a) => write!(f, "tan({a})"),
            Atan(a) => write!(f, "atan({a})"),
            Sinh(a) => write!(f, "sinh({a})"),
            Cosh(a) => write!(f, "cosh({a})"),
            Tanh(a) => write!(f, "tanh({a})"),
            Abs(a) => write!(f, "|{a}|"),
        }
    }
}
