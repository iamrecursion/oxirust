#![allow(dead_code)]
//! Conditional step execution for workflow orchestration.
//!
//! Provides a flexible condition evaluation system that can be used to
//! gate workflow steps based on runtime values, environment variables,
//! previous step outputs, and logical combinations of sub-conditions.

use std::collections::HashMap;

/// A value that can appear in condition evaluations.
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionValue {
    /// A string value.
    Str(String),
    /// An integer value.
    Int(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// No value / null.
    Null,
}

impl ConditionValue {
    /// Try to interpret the value as a boolean.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::Int(n) => Some(*n != 0),
            Self::Str(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" => Some(true),
                "false" | "no" | "0" | "" => Some(false),
                _ => None,
            },
            Self::Null => Some(false),
            Self::Float(_) => None,
        }
    }

    /// Try to interpret the value as an i64.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(n) => Some(*n),
            Self::Float(f) => Some(*f as i64),
            Self::Str(s) => s.parse().ok(),
            Self::Bool(b) => Some(i64::from(*b)),
            Self::Null => None,
        }
    }

    /// Try to interpret the value as f64.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(n) => Some(*n as f64),
            Self::Str(s) => s.parse().ok(),
            Self::Bool(_) | Self::Null => None,
        }
    }

    /// Convert to a string representation.
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Int(n) => n.to_string(),
            Self::Float(f) => f.to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Null => "null".to_string(),
        }
    }
}

/// Comparison operator for simple value comparisons.
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOp {
    /// Equal.
    Eq,
    /// Not equal.
    Neq,
    /// Less than.
    Lt,
    /// Less than or equal.
    Lte,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Gte,
    /// String contains.
    Contains,
    /// String starts with.
    StartsWith,
    /// String ends with.
    EndsWith,
    /// Value matches a regex-like pattern (simple glob).
    Matches,
}

/// A condition that can be evaluated against a context.
#[derive(Debug, Clone)]
pub enum StepCondition {
    /// Always true.
    Always,
    /// Always false.
    Never,
    /// Compare a named variable to a literal value.
    Compare {
        /// Variable name to look up in context.
        variable: String,
        /// Comparison operator.
        op: ComparisonOp,
        /// Value to compare against.
        value: ConditionValue,
    },
    /// Check if a variable exists in the context.
    Exists {
        /// Variable name.
        variable: String,
    },
    /// Logical AND of sub-conditions.
    And(Vec<StepCondition>),
    /// Logical OR of sub-conditions.
    Or(Vec<StepCondition>),
    /// Logical NOT.
    Not(Box<StepCondition>),
    /// Check that a previous step completed with a specific status.
    StepStatus {
        /// Name of the previous step.
        step_name: String,
        /// Expected status string (e.g. "completed", "failed").
        expected_status: String,
    },
    /// Evaluate a simple expression string.
    Expression(String),
}

/// Context used to evaluate conditions.
#[derive(Debug, Clone)]
pub struct ConditionContext {
    /// Variables available for condition evaluation.
    pub variables: HashMap<String, ConditionValue>,
    /// Results from previous steps keyed by step name.
    pub step_results: HashMap<String, String>,
}

impl Default for ConditionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ConditionContext {
    /// Create an empty condition context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            step_results: HashMap::new(),
        }
    }

    /// Set a variable.
    pub fn set_variable(&mut self, name: impl Into<String>, value: ConditionValue) {
        self.variables.insert(name.into(), value);
    }

    /// Set a step result.
    pub fn set_step_result(&mut self, step_name: impl Into<String>, status: impl Into<String>) {
        self.step_results.insert(step_name.into(), status.into());
    }

    /// Get a variable by name.
    #[must_use]
    pub fn get_variable(&self, name: &str) -> Option<&ConditionValue> {
        self.variables.get(name)
    }

    /// Get a step result by step name.
    #[must_use]
    pub fn get_step_result(&self, step_name: &str) -> Option<&str> {
        self.step_results
            .get(step_name)
            .map(std::string::String::as_str)
    }

    /// Return the number of variables.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Return the number of step results.
    #[must_use]
    pub fn step_result_count(&self) -> usize {
        self.step_results.len()
    }
}

/// Evaluate a condition against the given context.
#[must_use]
pub fn evaluate(condition: &StepCondition, ctx: &ConditionContext) -> bool {
    match condition {
        StepCondition::Always => true,
        StepCondition::Never => false,
        StepCondition::Compare {
            variable,
            op,
            value,
        } => {
            let Some(actual) = ctx.get_variable(variable) else {
                return false;
            };
            compare_values(actual, op, value)
        }
        StepCondition::Exists { variable } => ctx.variables.contains_key(variable),
        StepCondition::And(conditions) => conditions.iter().all(|c| evaluate(c, ctx)),
        StepCondition::Or(conditions) => conditions.iter().any(|c| evaluate(c, ctx)),
        StepCondition::Not(inner) => !evaluate(inner, ctx),
        StepCondition::StepStatus {
            step_name,
            expected_status,
        } => ctx
            .get_step_result(step_name)
            .is_some_and(|s| s == expected_status),
        StepCondition::Expression(expr) => evaluate_expression(expr, ctx),
    }
}

/// Compare two condition values using the given operator.
#[allow(clippy::cast_precision_loss)]
fn compare_values(actual: &ConditionValue, op: &ComparisonOp, expected: &ConditionValue) -> bool {
    match op {
        ComparisonOp::Eq => actual == expected,
        ComparisonOp::Neq => actual != expected,
        ComparisonOp::Lt | ComparisonOp::Lte | ComparisonOp::Gt | ComparisonOp::Gte => {
            if let (Some(a), Some(b)) = (actual.as_float(), expected.as_float()) {
                match op {
                    ComparisonOp::Lt => a < b,
                    ComparisonOp::Lte => a <= b,
                    ComparisonOp::Gt => a > b,
                    ComparisonOp::Gte => a >= b,
                    _ => false,
                }
            } else {
                false
            }
        }
        ComparisonOp::Contains => {
            let a = actual.to_string_repr();
            let b = expected.to_string_repr();
            a.contains(&b)
        }
        ComparisonOp::StartsWith => {
            let a = actual.to_string_repr();
            let b = expected.to_string_repr();
            a.starts_with(&b)
        }
        ComparisonOp::EndsWith => {
            let a = actual.to_string_repr();
            let b = expected.to_string_repr();
            a.ends_with(&b)
        }
        ComparisonOp::Matches => {
            let text = actual.to_string_repr();
            let pattern = expected.to_string_repr();
            simple_glob_match(&pattern, &text)
        }
    }
}

/// Simple glob matching: `*` matches any sequence, `?` matches any single char.
fn simple_glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_match_recursive(&p, &t, 0, 0)
}

/// Recursive glob matcher.
fn glob_match_recursive(p: &[char], t: &[char], pi: usize, ti: usize) -> bool {
    if pi == p.len() && ti == t.len() {
        return true;
    }
    if pi == p.len() {
        return false;
    }
    if p[pi] == '*' {
        // Try matching zero or more characters
        for i in ti..=t.len() {
            if glob_match_recursive(p, t, pi + 1, i) {
                return true;
            }
        }
        false
    } else if ti < t.len() && (p[pi] == '?' || p[pi] == t[ti]) {
        glob_match_recursive(p, t, pi + 1, ti + 1)
    } else {
        false
    }
}

/// Evaluate a simple expression string.
fn evaluate_expression(expr: &str, ctx: &ConditionContext) -> bool {
    let trimmed = expr.trim();
    // Support simple "variable == value" style
    if let Some(pos) = trimmed.find("==") {
        let lhs = trimmed[..pos].trim();
        let rhs = trimmed[pos + 2..].trim();
        if let Some(val) = ctx.get_variable(lhs) {
            return val.to_string_repr() == rhs;
        }
    }
    // Support "variable != value"
    if let Some(pos) = trimmed.find("!=") {
        let lhs = trimmed[..pos].trim();
        let rhs = trimmed[pos + 2..].trim();
        if let Some(val) = ctx.get_variable(lhs) {
            return val.to_string_repr() != rhs;
        }
    }
    // If the expression is just a variable name, treat as truthy
    if let Some(val) = ctx.get_variable(trimmed) {
        return val.as_bool().unwrap_or(false);
    }
    false
}

/// Builder for constructing complex conditions fluently.
#[derive(Debug)]
pub struct ConditionBuilder {
    /// The condition being built.
    condition: StepCondition,
}

impl ConditionBuilder {
    /// Start with an always-true condition.
    #[must_use]
    pub fn always() -> Self {
        Self {
            condition: StepCondition::Always,
        }
    }

    /// Start with an always-false condition.
    #[must_use]
    pub fn never() -> Self {
        Self {
            condition: StepCondition::Never,
        }
    }

    /// Create a variable comparison condition.
    pub fn compare(variable: impl Into<String>, op: ComparisonOp, value: ConditionValue) -> Self {
        Self {
            condition: StepCondition::Compare {
                variable: variable.into(),
                op,
                value,
            },
        }
    }

    /// Create an existence check.
    pub fn exists(variable: impl Into<String>) -> Self {
        Self {
            condition: StepCondition::Exists {
                variable: variable.into(),
            },
        }
    }

    /// Create a step status check.
    pub fn step_status(step_name: impl Into<String>, expected: impl Into<String>) -> Self {
        Self {
            condition: StepCondition::StepStatus {
                step_name: step_name.into(),
                expected_status: expected.into(),
            },
        }
    }

    /// Combine with AND.
    #[must_use]
    pub fn and(self, other: ConditionBuilder) -> Self {
        Self {
            condition: StepCondition::And(vec![self.condition, other.condition]),
        }
    }

    /// Combine with OR.
    #[must_use]
    pub fn or(self, other: ConditionBuilder) -> Self {
        Self {
            condition: StepCondition::Or(vec![self.condition, other.condition]),
        }
    }

    /// Negate.
    #[must_use]
    pub fn not(self) -> Self {
        Self {
            condition: StepCondition::Not(Box::new(self.condition)),
        }
    }

    /// Build the condition.
    #[must_use]
    pub fn build(self) -> StepCondition {
        self.condition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with_vars() -> ConditionContext {
        let mut ctx = ConditionContext::new();
        ctx.set_variable("status", ConditionValue::Str("success".to_string()));
        ctx.set_variable("count", ConditionValue::Int(42));
        ctx.set_variable("ratio", ConditionValue::Float(0.95));
        ctx.set_variable("enabled", ConditionValue::Bool(true));
        ctx.set_variable("disabled", ConditionValue::Bool(false));
        ctx.set_step_result("transcode", "completed");
        ctx.set_step_result("qc", "failed");
        ctx
    }

    #[test]
    fn test_always_and_never() {
        let ctx = ConditionContext::new();
        assert!(evaluate(&StepCondition::Always, &ctx));
        assert!(!evaluate(&StepCondition::Never, &ctx));
    }

    #[test]
    fn test_compare_eq_string() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "status".to_string(),
            op: ComparisonOp::Eq,
            value: ConditionValue::Str("success".to_string()),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_compare_neq() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "status".to_string(),
            op: ComparisonOp::Neq,
            value: ConditionValue::Str("failed".to_string()),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_compare_gt_int() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "count".to_string(),
            op: ComparisonOp::Gt,
            value: ConditionValue::Int(10),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_compare_lte_float() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "ratio".to_string(),
            op: ComparisonOp::Lte,
            value: ConditionValue::Float(1.0),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_compare_contains() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "status".to_string(),
            op: ComparisonOp::Contains,
            value: ConditionValue::Str("ucc".to_string()),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_compare_starts_with() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "status".to_string(),
            op: ComparisonOp::StartsWith,
            value: ConditionValue::Str("suc".to_string()),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_compare_ends_with() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "status".to_string(),
            op: ComparisonOp::EndsWith,
            value: ConditionValue::Str("ess".to_string()),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_glob_matches() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Compare {
            variable: "status".to_string(),
            op: ComparisonOp::Matches,
            value: ConditionValue::Str("suc*".to_string()),
        };
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_exists() {
        let ctx = ctx_with_vars();
        assert!(evaluate(
            &StepCondition::Exists {
                variable: "count".to_string()
            },
            &ctx
        ));
        assert!(!evaluate(
            &StepCondition::Exists {
                variable: "nope".to_string()
            },
            &ctx
        ));
    }

    #[test]
    fn test_and_condition() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::And(vec![StepCondition::Always, StepCondition::Always]);
        assert!(evaluate(&cond, &ctx));
        let cond2 = StepCondition::And(vec![StepCondition::Always, StepCondition::Never]);
        assert!(!evaluate(&cond2, &ctx));
    }

    #[test]
    fn test_or_condition() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Or(vec![StepCondition::Never, StepCondition::Always]);
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_not_condition() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Not(Box::new(StepCondition::Never));
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_step_status() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::StepStatus {
            step_name: "transcode".to_string(),
            expected_status: "completed".to_string(),
        };
        assert!(evaluate(&cond, &ctx));

        let cond_fail = StepCondition::StepStatus {
            step_name: "qc".to_string(),
            expected_status: "completed".to_string(),
        };
        assert!(!evaluate(&cond_fail, &ctx));
    }

    #[test]
    fn test_expression_eq() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Expression("status == success".to_string());
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_expression_neq() {
        let ctx = ctx_with_vars();
        let cond = StepCondition::Expression("status != failed".to_string());
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_expression_bool_variable() {
        let ctx = ctx_with_vars();
        assert!(evaluate(
            &StepCondition::Expression("enabled".to_string()),
            &ctx
        ));
        assert!(!evaluate(
            &StepCondition::Expression("disabled".to_string()),
            &ctx
        ));
    }

    #[test]
    fn test_condition_builder() {
        let ctx = ctx_with_vars();
        let cond = ConditionBuilder::compare("count", ComparisonOp::Gte, ConditionValue::Int(40))
            .and(ConditionBuilder::step_status("transcode", "completed"))
            .build();
        assert!(evaluate(&cond, &ctx));
    }

    #[test]
    fn test_condition_value_as_bool() {
        assert_eq!(ConditionValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ConditionValue::Int(0).as_bool(), Some(false));
        assert_eq!(ConditionValue::Str("yes".to_string()).as_bool(), Some(true));
        assert_eq!(ConditionValue::Null.as_bool(), Some(false));
    }

    #[test]
    fn test_condition_value_as_int() {
        assert_eq!(ConditionValue::Int(5).as_int(), Some(5));
        assert_eq!(ConditionValue::Float(3.7).as_int(), Some(3));
        assert_eq!(ConditionValue::Str("10".to_string()).as_int(), Some(10));
        assert_eq!(ConditionValue::Bool(true).as_int(), Some(1));
    }

    #[test]
    fn test_condition_context_default() {
        let ctx = ConditionContext::default();
        assert_eq!(ctx.variable_count(), 0);
        assert_eq!(ctx.step_result_count(), 0);
    }

    #[test]
    fn test_missing_variable_compare_returns_false() {
        let ctx = ConditionContext::new();
        let cond = StepCondition::Compare {
            variable: "nonexistent".to_string(),
            op: ComparisonOp::Eq,
            value: ConditionValue::Int(1),
        };
        assert!(!evaluate(&cond, &ctx));
    }
}
