//! Vectorized expression evaluation for filter and project operations.
//!
//! Provides row-at-a-time and SIMD-friendly evaluation of `Expr` trees over
//! `RecordBatch` data, along with mask application and column projection.

use std::sync::Arc;

use crate::datafusion_provider::{
    BinaryOperator, ColumnData, ColumnDef, DataType, Expr, LiteralValue, RecordBatch,
    TableProviderError, TableSchema,
};

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluate a filter expression over a `RecordBatch` and return a boolean mask.
///
/// The returned `Vec<bool>` has one entry per row; `true` means the row passes
/// the filter.
pub fn eval_filter(batch: &RecordBatch, expr: &Expr) -> Result<Vec<bool>, TableProviderError> {
    let col_data = eval_expr(batch, expr)?;
    match col_data {
        EvalResult::Boolean(mask) => Ok(mask),
        other => Err(TableProviderError::TypeError(format!(
            "Filter expression must yield a boolean column, got {:?}",
            other.type_name()
        ))),
    }
}

/// Apply a boolean mask to a `RecordBatch`, returning only rows where `mask[i] = true`.
pub fn apply_mask(batch: &RecordBatch, mask: &[bool]) -> Result<RecordBatch, TableProviderError> {
    if mask.len() != batch.num_rows {
        return Err(TableProviderError::ScanError(format!(
            "Mask length {} does not match batch row count {}",
            mask.len(),
            batch.num_rows
        )));
    }

    let filtered_cols: Vec<ColumnData> = batch
        .columns
        .iter()
        .map(|c| c.filter_by_mask(mask))
        .collect();

    let num_rows = mask.iter().filter(|&&m| m).count();

    Ok(RecordBatch {
        schema: Arc::clone(&batch.schema),
        columns: filtered_cols,
        num_rows,
    })
}

/// Evaluate a list of named projection expressions over a `RecordBatch`.
///
/// Each element of `exprs` is `(output_column_name, expression)`. The result is
/// a new `RecordBatch` containing one column per expression entry.
pub fn eval_projection(
    batch: &RecordBatch,
    exprs: &[(std::string::String, Expr)],
) -> Result<RecordBatch, TableProviderError> {
    let mut out_cols: Vec<ColumnData> = Vec::with_capacity(exprs.len());
    let mut col_defs: Vec<ColumnDef> = Vec::with_capacity(exprs.len());

    for (out_name, expr) in exprs {
        let result = eval_expr(batch, expr)?;
        let (data, dtype) = result.into_column_data(batch.num_rows);
        col_defs.push(ColumnDef {
            name: out_name.clone(),
            data_type: dtype,
            nullable: false,
        });
        out_cols.push(data);
    }

    let schema = Arc::new(TableSchema::new(col_defs));
    Ok(RecordBatch {
        schema,
        columns: out_cols,
        num_rows: batch.num_rows,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal evaluation result
// ──────────────────────────────────────────────────────────────────────────────

/// Internal evaluation result: typed columnar data produced by evaluating an
/// `Expr` over a single `RecordBatch`.
#[derive(Debug)]
enum EvalResult {
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Boolean(Vec<bool>),
    Utf8(Vec<std::string::String>),
    Null(usize),
}

impl EvalResult {
    fn type_name(&self) -> &'static str {
        match self {
            EvalResult::Int32(_) => "Int32",
            EvalResult::Int64(_) => "Int64",
            EvalResult::Float32(_) => "Float32",
            EvalResult::Float64(_) => "Float64",
            EvalResult::Boolean(_) => "Boolean",
            EvalResult::Utf8(_) => "Utf8",
            EvalResult::Null(_) => "Null",
        }
    }

    fn len(&self) -> usize {
        match self {
            EvalResult::Int32(v) => v.len(),
            EvalResult::Int64(v) => v.len(),
            EvalResult::Float32(v) => v.len(),
            EvalResult::Float64(v) => v.len(),
            EvalResult::Boolean(v) => v.len(),
            EvalResult::Utf8(v) => v.len(),
            EvalResult::Null(n) => *n,
        }
    }

    fn into_column_data(self, num_rows: usize) -> (ColumnData, DataType) {
        match self {
            EvalResult::Int32(v) => (ColumnData::Int32(v), DataType::Int32),
            EvalResult::Int64(v) => (ColumnData::Int64(v), DataType::Int64),
            EvalResult::Float32(v) => (ColumnData::Float32(v), DataType::Float32),
            EvalResult::Float64(v) => (ColumnData::Float64(v), DataType::Float64),
            EvalResult::Boolean(v) => (ColumnData::Boolean(v), DataType::Boolean),
            EvalResult::Utf8(v) => (ColumnData::Utf8(v), DataType::Utf8),
            EvalResult::Null(_) => (ColumnData::Null(num_rows), DataType::Boolean),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core evaluator
// ──────────────────────────────────────────────────────────────────────────────

fn eval_expr(batch: &RecordBatch, expr: &Expr) -> Result<EvalResult, TableProviderError> {
    match expr {
        Expr::Column(name) => {
            let idx = batch
                .schema
                .field_index(name)
                .ok_or_else(|| TableProviderError::ColumnNotFound(name.clone()))?;
            let col = &batch.columns[idx];
            Ok(column_data_to_eval_result(col))
        }

        Expr::Literal(lit) => {
            let n = batch.num_rows;
            match lit {
                LiteralValue::Int64(v) => Ok(EvalResult::Int64(vec![*v; n])),
                LiteralValue::Float64(v) => Ok(EvalResult::Float64(vec![*v; n])),
                LiteralValue::Boolean(v) => Ok(EvalResult::Boolean(vec![*v; n])),
                LiteralValue::Utf8(v) => Ok(EvalResult::Utf8(vec![v.clone(); n])),
                LiteralValue::Null => Ok(EvalResult::Null(n)),
            }
        }

        Expr::BinaryOp { left, op, right } => {
            let lhs = eval_expr(batch, left)?;
            let rhs = eval_expr(batch, right)?;
            eval_binary_op(lhs, *op, rhs)
        }

        Expr::IsNull(inner) => {
            let result = eval_expr(batch, inner)?;
            let mask = match result {
                EvalResult::Null(n) => vec![true; n],
                other => vec![false; other.len()],
            };
            Ok(EvalResult::Boolean(mask))
        }

        Expr::IsNotNull(inner) => {
            let result = eval_expr(batch, inner)?;
            let mask = match result {
                EvalResult::Null(n) => vec![false; n],
                other => vec![true; other.len()],
            };
            Ok(EvalResult::Boolean(mask))
        }

        Expr::Not(inner) => {
            let result = eval_expr(batch, inner)?;
            match result {
                EvalResult::Boolean(v) => Ok(EvalResult::Boolean(v.iter().map(|b| !b).collect())),
                other => Err(TableProviderError::TypeError(format!(
                    "NOT requires boolean operand, got {}",
                    other.type_name()
                ))),
            }
        }
    }
}

fn column_data_to_eval_result(col: &ColumnData) -> EvalResult {
    match col {
        ColumnData::Int32(v) => EvalResult::Int32(v.clone()),
        ColumnData::Int64(v) => EvalResult::Int64(v.clone()),
        ColumnData::Float32(v) => EvalResult::Float32(v.clone()),
        ColumnData::Float64(v) => EvalResult::Float64(v.clone()),
        ColumnData::Boolean(v) => EvalResult::Boolean(v.clone()),
        ColumnData::Utf8(v) => EvalResult::Utf8(v.clone()),
        ColumnData::Null(n) => EvalResult::Null(*n),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Binary operator dispatch
// ──────────────────────────────────────────────────────────────────────────────

fn eval_binary_op(
    lhs: EvalResult,
    op: BinaryOperator,
    rhs: EvalResult,
) -> Result<EvalResult, TableProviderError> {
    // AND / OR require both sides to be boolean.
    match op {
        BinaryOperator::And => {
            let (a, b) = coerce_both_boolean(lhs, rhs, "AND")?;
            return Ok(EvalResult::Boolean(
                a.iter().zip(&b).map(|(&x, &y)| x && y).collect(),
            ));
        }
        BinaryOperator::Or => {
            let (a, b) = coerce_both_boolean(lhs, rhs, "OR")?;
            return Ok(EvalResult::Boolean(
                a.iter().zip(&b).map(|(&x, &y)| x || y).collect(),
            ));
        }
        _ => {}
    }

    // Numeric comparison / arithmetic: promote both sides to f64 when possible.
    if let (Some(lv), Some(rv)) = (to_f64_vec(&lhs), to_f64_vec(&rhs)) {
        match op {
            BinaryOperator::Eq => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(&rv).map(|(a, b)| a == b).collect(),
                ));
            }
            BinaryOperator::NotEq => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(&rv).map(|(a, b)| a != b).collect(),
                ));
            }
            BinaryOperator::Lt => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(&rv).map(|(a, b)| a < b).collect(),
                ));
            }
            BinaryOperator::LtEq => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(&rv).map(|(a, b)| a <= b).collect(),
                ));
            }
            BinaryOperator::Gt => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(&rv).map(|(a, b)| a > b).collect(),
                ));
            }
            BinaryOperator::GtEq => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(&rv).map(|(a, b)| a >= b).collect(),
                ));
            }
            BinaryOperator::Plus => {
                return Ok(EvalResult::Float64(
                    lv.iter().zip(&rv).map(|(a, b)| a + b).collect(),
                ));
            }
            BinaryOperator::Minus => {
                return Ok(EvalResult::Float64(
                    lv.iter().zip(&rv).map(|(a, b)| a - b).collect(),
                ));
            }
            BinaryOperator::Multiply => {
                return Ok(EvalResult::Float64(
                    lv.iter().zip(&rv).map(|(a, b)| a * b).collect(),
                ));
            }
            BinaryOperator::Divide => {
                return Ok(EvalResult::Float64(
                    lv.iter()
                        .zip(&rv)
                        .map(|(a, b)| if *b == 0.0 { f64::NAN } else { a / b })
                        .collect(),
                ));
            }
            BinaryOperator::And | BinaryOperator::Or => unreachable!(),
        }
    }

    // String equality.
    if let (EvalResult::Utf8(lv), EvalResult::Utf8(rv)) = (&lhs, &rhs) {
        match op {
            BinaryOperator::Eq => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(rv).map(|(a, b)| a == b).collect(),
                ));
            }
            BinaryOperator::NotEq => {
                return Ok(EvalResult::Boolean(
                    lv.iter().zip(rv).map(|(a, b)| a != b).collect(),
                ));
            }
            _ => {}
        }
    }

    Err(TableProviderError::TypeError(format!(
        "Unsupported binary operation {:?} on types {} and {}",
        op,
        lhs.type_name(),
        rhs.type_name()
    )))
}

fn coerce_both_boolean(
    lhs: EvalResult,
    rhs: EvalResult,
    op_name: &str,
) -> Result<(Vec<bool>, Vec<bool>), TableProviderError> {
    let a = match lhs {
        EvalResult::Boolean(v) => v,
        other => {
            return Err(TableProviderError::TypeError(format!(
                "{op_name} requires boolean left operand, got {}",
                other.type_name()
            )))
        }
    };
    let b = match rhs {
        EvalResult::Boolean(v) => v,
        other => {
            return Err(TableProviderError::TypeError(format!(
                "{op_name} requires boolean right operand, got {}",
                other.type_name()
            )))
        }
    };
    Ok((a, b))
}

fn to_f64_vec(r: &EvalResult) -> Option<Vec<f64>> {
    match r {
        EvalResult::Int32(v) => Some(v.iter().map(|&x| x as f64).collect()),
        EvalResult::Int64(v) => Some(v.iter().map(|&x| x as f64).collect()),
        EvalResult::Float32(v) => Some(v.iter().map(|&x| x as f64).collect()),
        EvalResult::Float64(v) => Some(v.clone()),
        _ => None,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafusion_provider::{ColumnDef, DataType, LiteralValue, TableSchema};

    fn make_numeric_batch() -> RecordBatch {
        let schema = Arc::new(TableSchema::new(vec![
            ColumnDef {
                name: "x".to_string(),
                data_type: DataType::Float64,
                nullable: false,
            },
            ColumnDef {
                name: "y".to_string(),
                data_type: DataType::Float64,
                nullable: false,
            },
        ]));
        let columns = vec![
            ColumnData::Float64(vec![1.0, 5.0, 3.0, 7.0, 2.0]),
            ColumnData::Float64(vec![2.0, 1.0, 4.0, 0.0, 9.0]),
        ];
        RecordBatch::new(schema, columns)
    }

    #[test]
    fn test_eval_filter_gt() {
        let batch = make_numeric_batch();
        // x > 3.0  -->  rows 1 (5.0) and 3 (7.0)
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Column("x".to_string())),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(LiteralValue::Float64(3.0))),
        };
        let mask = eval_filter(&batch, &expr).expect("eval_filter failed");
        assert_eq!(mask, vec![false, true, false, true, false]);
    }

    #[test]
    fn test_eval_filter_and() {
        let batch = make_numeric_batch();
        // x > 2.0 AND y < 5.0  -->  row 1 (x=5, y=1) and row 3 (x=7, y=0)
        let x_gt_2 = Expr::BinaryOp {
            left: Box::new(Expr::Column("x".to_string())),
            op: BinaryOperator::Gt,
            right: Box::new(Expr::Literal(LiteralValue::Float64(2.0))),
        };
        let y_lt_5 = Expr::BinaryOp {
            left: Box::new(Expr::Column("y".to_string())),
            op: BinaryOperator::Lt,
            right: Box::new(Expr::Literal(LiteralValue::Float64(5.0))),
        };
        let combined = Expr::BinaryOp {
            left: Box::new(x_gt_2),
            op: BinaryOperator::And,
            right: Box::new(y_lt_5),
        };
        let mask = eval_filter(&batch, &combined).expect("eval_filter AND failed");
        // x values:   1,  5,  3,  7,  2
        // y values:   2,  1,  4,  0,  9
        // x > 2:      F   T   T   T   F
        // y < 5:      T   T   T   T   F
        // AND:        F   T   T   T   F
        assert_eq!(mask, vec![false, true, true, true, false]);
    }

    #[test]
    fn test_apply_mask() {
        let batch = make_numeric_batch();
        let mask = vec![false, true, false, true, false];
        let filtered = apply_mask(&batch, &mask).expect("apply_mask failed");
        assert_eq!(filtered.num_rows, 2);
        if let ColumnData::Float64(vals) = &filtered.columns[0] {
            assert!((vals[0] - 5.0).abs() < 1e-12);
            assert!((vals[1] - 7.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64 column");
        }
    }

    #[test]
    fn test_eval_projection() {
        let batch = make_numeric_batch();
        // Compute new column "z" = x + y
        let proj_expr = Expr::BinaryOp {
            left: Box::new(Expr::Column("x".to_string())),
            op: BinaryOperator::Plus,
            right: Box::new(Expr::Column("y".to_string())),
        };
        let projected = eval_projection(&batch, &[("z".to_string(), proj_expr)])
            .expect("eval_projection failed");
        assert_eq!(projected.schema.columns[0].name, "z");
        assert_eq!(projected.num_rows, 5);
        if let ColumnData::Float64(vals) = &projected.columns[0] {
            // x + y: 1+2=3, 5+1=6, 3+4=7, 7+0=7, 2+9=11
            assert!((vals[0] - 3.0).abs() < 1e-12);
            assert!((vals[1] - 6.0).abs() < 1e-12);
            assert!((vals[4] - 11.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64 column for z");
        }
    }

    #[test]
    fn test_eval_filter_eq_str() {
        let schema = Arc::new(TableSchema::new(vec![ColumnDef {
            name: "cat".to_string(),
            data_type: DataType::Utf8,
            nullable: false,
        }]));
        let cols = vec![ColumnData::Utf8(vec![
            "foo".to_string(),
            "bar".to_string(),
            "foo".to_string(),
        ])];
        let batch = RecordBatch::new(schema, cols);

        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Column("cat".to_string())),
            op: BinaryOperator::Eq,
            right: Box::new(Expr::Literal(LiteralValue::Utf8("foo".to_string()))),
        };
        let mask = eval_filter(&batch, &expr).expect("eval str filter failed");
        assert_eq!(mask, vec![true, false, true]);
    }
}
