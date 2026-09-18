//! # Clausal Normal Form (CNF) Conversion
//!
//! This module provides a simplified conversion from [`TLExpr`] to CNF
//! (a set of [`Clause`]s). Full CNF conversion would additionally require
//! skolemization and universal-quantifier elimination.

use crate::error::IrError;
use crate::expr::TLExpr;

use super::clause::Clause;
use super::literal::Literal;

/// Convert a TLExpr to Clausal Normal Form (CNF).
///
/// This is a simplified conversion that handles basic logical operators.
/// Full CNF conversion would require skolemization and quantifier elimination.
pub fn to_cnf(expr: &TLExpr) -> Result<Vec<Clause>, IrError> {
    // Simplified CNF conversion
    // For full implementation, would need:
    // 1. Eliminate implications
    // 2. Move negations inward (De Morgan's laws)
    // 3. Distribute OR over AND
    // 4. Skolemize existential quantifiers
    // 5. Drop universal quantifiers

    match expr {
        TLExpr::And(left, right) => {
            let mut clauses = to_cnf(left)?;
            clauses.extend(to_cnf(right)?);
            Ok(clauses)
        }
        TLExpr::Or(left, right) => {
            // Distribute OR over AND if needed
            let left_clauses = to_cnf(left)?;
            let right_clauses = to_cnf(right)?;

            if left_clauses.len() == 1 && right_clauses.len() == 1 {
                // Simple case: combine literals
                let mut literals = left_clauses[0].literals.clone();
                literals.extend(right_clauses[0].literals.clone());
                Ok(vec![Clause::from_literals(literals)])
            } else {
                // Complex case: would need distribution
                // For now, treat as separate clauses (approximation)
                let mut result = left_clauses;
                result.extend(right_clauses);
                Ok(result)
            }
        }
        TLExpr::Not(inner) => {
            match &**inner {
                TLExpr::Pred { .. } => {
                    // Negative literal
                    Ok(vec![Clause::unit(Literal::negative((**inner).clone()))])
                }
                _ => {
                    // Would need to push negation inward
                    Err(IrError::ConstraintViolation {
                        message: "Complex negation not supported in simplified CNF conversion"
                            .to_string(),
                    })
                }
            }
        }
        TLExpr::Pred { .. } => {
            // Positive literal
            Ok(vec![Clause::unit(Literal::positive(expr.clone()))])
        }
        _ => Err(IrError::ConstraintViolation {
            message: format!(
                "Expression type not supported in CNF conversion: {:?}",
                expr
            ),
        }),
    }
}
