//! Conditional step execution for workflow orchestration.
//!
//! [`StepCondition`] variants cover static conditions (`Always`), success/failure
//! guards, direct field lookups, expression strings (`"field op value"`), and
//! recursive logical combinators (`And`, `Or`, `Not`).
//!
//! Use [`ConditionEvaluator::evaluate`] to test a condition against a
//! [`StepContext`] gathered at runtime.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── StepCondition ─────────────────────────────────────────────────────────────

/// A condition that gates whether a workflow step should execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepCondition {
    /// Always execute — the step is unconditional.
    Always,
    /// Execute only when the immediately preceding step succeeded.
    OnPreviousSuccess,
    /// Execute only when the immediately preceding step failed.
    OnPreviousFailure,
    /// Evaluate a simple expression of the form `"field op value"`.
    ///
    /// Supported operators: `>`, `<`, `==`, `!=`, `>=`, `<=`, `contains`.
    Expression(String),
    /// Execute when `ctx.previous_output[field] == value`.
    FieldEquals {
        /// Output field key.
        field: String,
        /// Expected value.
        value: String,
    },
    /// Execute when `ctx.previous_output[field]` contains `substring`.
    FieldContains {
        /// Output field key.
        field: String,
        /// Required substring.
        substring: String,
    },
    /// All sub-conditions must be true (short-circuit).
    And(Vec<StepCondition>),
    /// At least one sub-condition must be true (short-circuit).
    Or(Vec<StepCondition>),
    /// Negation of the inner condition.
    Not(Box<StepCondition>),
}

// ── StepContext ───────────────────────────────────────────────────────────────

/// Runtime context supplied to [`ConditionEvaluator::evaluate`].
#[derive(Debug, Clone)]
pub struct StepContext {
    /// Identifier of the step whose condition is being evaluated.
    pub step_id: String,
    /// Whether the immediately preceding step completed successfully.
    pub previous_success: bool,
    /// Key/value output produced by the preceding step.
    pub previous_output: HashMap<String, String>,
}

impl StepContext {
    /// Create a context representing a successful previous step with no output.
    #[must_use]
    pub fn new(step_id: impl Into<String>) -> Self {
        Self {
            step_id: step_id.into(),
            previous_success: true,
            previous_output: HashMap::new(),
        }
    }

    /// Override the success flag.
    #[must_use]
    pub fn with_success(mut self, success: bool) -> Self {
        self.previous_success = success;
        self
    }

    /// Insert an output key/value pair.
    #[must_use]
    pub fn with_output(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.previous_output.insert(key.into(), value.into());
        self
    }
}

// ── ConditionEvaluator ───────────────────────────────────────────────────────

/// Evaluates [`StepCondition`]s against a [`StepContext`].
pub struct ConditionEvaluator;

impl ConditionEvaluator {
    /// Evaluate `condition` using the runtime values in `ctx`.
    ///
    /// `And` and `Or` short-circuit: evaluation stops as soon as the result
    /// is determined.
    #[must_use]
    pub fn evaluate(condition: &StepCondition, ctx: &StepContext) -> bool {
        match condition {
            StepCondition::Always => true,
            StepCondition::OnPreviousSuccess => ctx.previous_success,
            StepCondition::OnPreviousFailure => !ctx.previous_success,
            StepCondition::Expression(expr) => evaluate_expression(expr, ctx),
            StepCondition::FieldEquals { field, value } => {
                ctx.previous_output.get(field).map_or(false, |v| v == value)
            }
            StepCondition::FieldContains { field, substring } => ctx
                .previous_output
                .get(field)
                .map_or(false, |v| v.contains(substring.as_str())),
            StepCondition::And(conditions) => conditions.iter().all(|c| Self::evaluate(c, ctx)),
            StepCondition::Or(conditions) => conditions.iter().any(|c| Self::evaluate(c, ctx)),
            StepCondition::Not(inner) => !Self::evaluate(inner, ctx),
        }
    }
}

// ── Expression parser / evaluator ────────────────────────────────────────────

/// Parse and evaluate a simple expression `"field op value"` against `ctx`.
///
/// Supported `op` tokens: `>=`, `<=`, `!=`, `==`, `>`, `<`, `contains`.
/// The `field` is looked up in `ctx.previous_output`; the comparison is
/// attempted numerically first, then as a string equality/containment.
fn evaluate_expression(expr: &str, ctx: &StepContext) -> bool {
    let expr = expr.trim();

    // Ordered by length (longest first) so `>=` is matched before `>`.
    const OPS: &[&str] = &[">=", "<=", "!=", "==", ">", "<", "contains"];

    for &op in OPS {
        if let Some(op_pos) = find_op(expr, op) {
            let field = expr[..op_pos].trim();
            let rhs = expr[op_pos + op.len()..].trim();

            let lhs_raw = match ctx.previous_output.get(field) {
                Some(v) => v.as_str(),
                None => return false,
            };

            return compare_str(lhs_raw, op, rhs);
        }
    }

    // No operator found — treat as boolean field test.
    ctx.previous_output
        .get(expr)
        .map_or(false, |v| is_truthy(v))
}

/// Find the byte offset of `op` in `expr`, skipping occurrences inside longer
/// operators (e.g. avoid matching `>` inside `>=`).
fn find_op(expr: &str, op: &str) -> Option<usize> {
    // We search left-to-right and ensure the character immediately following
    // the found position is not part of a longer operator.
    let bytes = expr.as_bytes();
    let op_bytes = op.as_bytes();
    let len = op_bytes.len();

    let mut i = 0usize;
    while i + len <= bytes.len() {
        if &bytes[i..i + len] == op_bytes {
            // For single-char ops like `>` or `<`, ensure the next char is
            // not `=` (which would make it `>=` / `<=`).
            let next_ok = if len == 1 {
                bytes.get(i + 1) != Some(&b'=')
            } else {
                true
            };
            if next_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Compare `lhs` (raw string from output) to `rhs` (literal from expression)
/// using operator `op`.
fn compare_str(lhs: &str, op: &str, rhs: &str) -> bool {
    // Attempt numeric comparison first.
    if let (Ok(l), Ok(r)) = (lhs.parse::<f64>(), rhs.parse::<f64>()) {
        return match op {
            "==" => (l - r).abs() < f64::EPSILON,
            "!=" => (l - r).abs() >= f64::EPSILON,
            ">" => l > r,
            "<" => l < r,
            ">=" => l >= r,
            "<=" => l <= r,
            "contains" => lhs.contains(rhs),
            _ => false,
        };
    }

    // String comparison.
    match op {
        "==" => lhs == rhs,
        "!=" => lhs != rhs,
        "contains" => lhs.contains(rhs),
        // Lexicographic ordering for non-numeric strings.
        ">" => lhs > rhs,
        "<" => lhs < rhs,
        ">=" => lhs >= rhs,
        "<=" => lhs <= rhs,
        _ => false,
    }
}

/// Interpret a string value as boolean.
fn is_truthy(v: &str) -> bool {
    matches!(v.to_lowercase().as_str(), "true" | "yes" | "1")
}

// ── StepCondition::parse ─────────────────────────────────────────────────────

impl StepCondition {
    /// Parse a [`StepCondition`] from its string representation.
    ///
    /// Supported forms:
    /// - `"always"` → [`StepCondition::Always`]
    /// - `"on_success"` / `"on_previous_success"` → [`StepCondition::OnPreviousSuccess`]
    /// - `"on_failure"` / `"on_previous_failure"` → [`StepCondition::OnPreviousFailure`]
    /// - `"expr: <expression>"` → [`StepCondition::Expression`]
    /// - `"field_equals: <field>=<value>"` → [`StepCondition::FieldEquals`]
    /// - `"field_contains: <field>=<substring>"` → [`StepCondition::FieldContains`]
    ///
    /// # Errors
    ///
    /// Returns an error string for unrecognised or malformed input.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        if s.eq_ignore_ascii_case("always") {
            return Ok(Self::Always);
        }
        if s.eq_ignore_ascii_case("on_success") || s.eq_ignore_ascii_case("on_previous_success") {
            return Ok(Self::OnPreviousSuccess);
        }
        if s.eq_ignore_ascii_case("on_failure") || s.eq_ignore_ascii_case("on_previous_failure") {
            return Ok(Self::OnPreviousFailure);
        }

        if let Some(rest) = s.strip_prefix("expr:") {
            return Ok(Self::Expression(rest.trim().to_string()));
        }

        if let Some(rest) = s.strip_prefix("field_equals:") {
            return parse_field_value_pair(rest.trim())
                .map(|(f, v)| Self::FieldEquals { field: f, value: v });
        }

        if let Some(rest) = s.strip_prefix("field_contains:") {
            return parse_field_value_pair(rest.trim()).map(|(f, v)| Self::FieldContains {
                field: f,
                substring: v,
            });
        }

        Err(format!("Unrecognised condition string: '{s}'"))
    }
}

/// Split `"field=value"` into `(field, value)`.
fn parse_field_value_pair(s: &str) -> Result<(String, String), String> {
    if let Some(eq_pos) = s.find('=') {
        let field = s[..eq_pos].trim().to_string();
        let value = s[eq_pos + 1..].trim().to_string();
        if field.is_empty() {
            return Err("Field name must not be empty".to_string());
        }
        Ok((field, value))
    } else {
        Err(format!("Expected 'field=value' in: '{s}'"))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_success() -> StepContext {
        StepContext::new("step-a")
            .with_success(true)
            .with_output("size_mb", "250")
            .with_output("status", "ok")
            .with_output("codec", "h264")
    }

    fn ctx_failure() -> StepContext {
        StepContext::new("step-b").with_success(false)
    }

    // ── Basic variants ────────────────────────────────────────────────────────

    #[test]
    fn test_always_is_true() {
        let ctx = ctx_failure();
        assert!(ConditionEvaluator::evaluate(&StepCondition::Always, &ctx));
    }

    #[test]
    fn test_on_previous_success_true_when_succeeded() {
        assert!(ConditionEvaluator::evaluate(
            &StepCondition::OnPreviousSuccess,
            &ctx_success()
        ));
    }

    #[test]
    fn test_on_previous_success_false_when_failed() {
        assert!(!ConditionEvaluator::evaluate(
            &StepCondition::OnPreviousSuccess,
            &ctx_failure()
        ));
    }

    #[test]
    fn test_on_previous_failure_true_when_failed() {
        assert!(ConditionEvaluator::evaluate(
            &StepCondition::OnPreviousFailure,
            &ctx_failure()
        ));
    }

    #[test]
    fn test_on_previous_failure_false_when_succeeded() {
        assert!(!ConditionEvaluator::evaluate(
            &StepCondition::OnPreviousFailure,
            &ctx_success()
        ));
    }

    // ── FieldEquals / FieldContains ───────────────────────────────────────────

    #[test]
    fn test_field_equals_match() {
        let cond = StepCondition::FieldEquals {
            field: "status".into(),
            value: "ok".into(),
        };
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_field_equals_no_match() {
        let cond = StepCondition::FieldEquals {
            field: "status".into(),
            value: "fail".into(),
        };
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_field_equals_missing_field() {
        let cond = StepCondition::FieldEquals {
            field: "nonexistent".into(),
            value: "x".into(),
        };
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_field_contains_match() {
        let cond = StepCondition::FieldContains {
            field: "codec".into(),
            substring: "264".into(),
        };
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_field_contains_no_match() {
        let cond = StepCondition::FieldContains {
            field: "codec".into(),
            substring: "vp9".into(),
        };
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    // ── Expression evaluation ─────────────────────────────────────────────────

    #[test]
    fn test_expression_gt_numeric() {
        let cond = StepCondition::Expression("size_mb > 100".into());
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_expression_lte_numeric_false() {
        let cond = StepCondition::Expression("size_mb <= 100".into());
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_expression_eq_string() {
        let cond = StepCondition::Expression("status == ok".into());
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_expression_neq_string() {
        let cond = StepCondition::Expression("status != fail".into());
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_expression_contains_op() {
        let cond = StepCondition::Expression("codec contains 264".into());
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_expression_missing_field_returns_false() {
        let cond = StepCondition::Expression("no_such_field > 0".into());
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    // ── And / Or / Not ────────────────────────────────────────────────────────

    #[test]
    fn test_and_all_true() {
        let cond = StepCondition::And(vec![
            StepCondition::Always,
            StepCondition::OnPreviousSuccess,
        ]);
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_and_short_circuits_on_false() {
        let cond = StepCondition::And(vec![
            StepCondition::OnPreviousFailure, // false for success ctx
            StepCondition::Always,
        ]);
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_or_true_when_any_true() {
        let cond = StepCondition::Or(vec![
            StepCondition::OnPreviousFailure, // false
            StepCondition::OnPreviousSuccess, // true
        ]);
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_or_false_when_all_false() {
        let cond = StepCondition::Or(vec![
            StepCondition::OnPreviousFailure,
            StepCondition::OnPreviousFailure,
        ]);
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    #[test]
    fn test_not_negates() {
        let cond = StepCondition::Not(Box::new(StepCondition::OnPreviousSuccess));
        assert!(!ConditionEvaluator::evaluate(&cond, &ctx_success()));
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_failure()));
    }

    #[test]
    fn test_nested_and_or_not() {
        // (size_mb > 100 AND status == ok) OR NOT(on_failure)
        let cond = StepCondition::Or(vec![
            StepCondition::And(vec![
                StepCondition::Expression("size_mb > 100".into()),
                StepCondition::FieldEquals {
                    field: "status".into(),
                    value: "ok".into(),
                },
            ]),
            StepCondition::Not(Box::new(StepCondition::OnPreviousFailure)),
        ]);
        assert!(ConditionEvaluator::evaluate(&cond, &ctx_success()));
    }

    // ── StepCondition::parse ──────────────────────────────────────────────────

    #[test]
    fn test_parse_always() {
        let c = StepCondition::parse("always").expect("parse should succeed");
        assert!(matches!(c, StepCondition::Always));
    }

    #[test]
    fn test_parse_on_success_short_form() {
        let c = StepCondition::parse("on_success").expect("parse should succeed");
        assert!(matches!(c, StepCondition::OnPreviousSuccess));
    }

    #[test]
    fn test_parse_on_failure_long_form() {
        let c = StepCondition::parse("on_previous_failure").expect("parse should succeed");
        assert!(matches!(c, StepCondition::OnPreviousFailure));
    }

    #[test]
    fn test_parse_expr() {
        let c = StepCondition::parse("expr: size_mb > 100").expect("parse should succeed");
        assert!(matches!(c, StepCondition::Expression(_)));
    }

    #[test]
    fn test_parse_field_equals() {
        let c = StepCondition::parse("field_equals: status=ok").expect("parse should succeed");
        if let StepCondition::FieldEquals { field, value } = c {
            assert_eq!(field, "status");
            assert_eq!(value, "ok");
        } else {
            panic!("expected FieldEquals");
        }
    }

    #[test]
    fn test_parse_field_contains() {
        let c = StepCondition::parse("field_contains: codec=264").expect("parse should succeed");
        if let StepCondition::FieldContains { field, substring } = c {
            assert_eq!(field, "codec");
            assert_eq!(substring, "264");
        } else {
            panic!("expected FieldContains");
        }
    }

    #[test]
    fn test_parse_unknown_returns_err() {
        let r = StepCondition::parse("nonsense_condition");
        assert!(r.is_err());
    }
}
