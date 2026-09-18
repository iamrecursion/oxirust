//! Elimination arms for quantifiers, arithmetic, comparison, unary math, and
//! modal/temporal operators (simple passthrough with recursion).

use tensorlogic_ir::TLExpr;

use super::types::{DceStats, DeadCodeEliminator};

impl DeadCodeEliminator {
    /// Handle quantifier / arithmetic / comparison / unary-math / modal / temporal arms.
    ///
    /// Returns `Ok((new_expr, changed))` if this category handled the node, or
    /// `Err(expr)` to pass the unchanged expression to the next category.
    pub(super) fn elim_ops(
        &self,
        expr: TLExpr,
        stats: &mut DceStats,
    ) -> Result<(TLExpr, bool), TLExpr> {
        match expr {
            TLExpr::Exists { var, domain, body } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                Ok((
                    TLExpr::Exists {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                ))
            }
            TLExpr::ForAll { var, domain, body } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                Ok((
                    TLExpr::ForAll {
                        var,
                        domain,
                        body: Box::new(new_body),
                    },
                    changed,
                ))
            }
            TLExpr::SoftExists {
                var,
                domain,
                body,
                temperature,
            } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                Ok((
                    TLExpr::SoftExists {
                        var,
                        domain,
                        body: Box::new(new_body),
                        temperature,
                    },
                    changed,
                ))
            }
            TLExpr::SoftForAll {
                var,
                domain,
                body,
                temperature,
            } => {
                let (new_body, changed) = self.eliminate(*body, stats);
                Ok((
                    TLExpr::SoftForAll {
                        var,
                        domain,
                        body: Box::new(new_body),
                        temperature,
                    },
                    changed,
                ))
            }

            TLExpr::Add(l, r) => Ok(self.map_binary(TLExpr::Add, *l, *r, stats)),
            TLExpr::Sub(l, r) => Ok(self.map_binary(TLExpr::Sub, *l, *r, stats)),
            TLExpr::Mul(l, r) => Ok(self.map_binary(TLExpr::Mul, *l, *r, stats)),
            TLExpr::Div(l, r) => Ok(self.map_binary(TLExpr::Div, *l, *r, stats)),
            TLExpr::Pow(l, r) => Ok(self.map_binary(TLExpr::Pow, *l, *r, stats)),
            TLExpr::Mod(l, r) => Ok(self.map_binary(TLExpr::Mod, *l, *r, stats)),
            TLExpr::Min(l, r) => Ok(self.map_binary(TLExpr::Min, *l, *r, stats)),
            TLExpr::Max(l, r) => Ok(self.map_binary(TLExpr::Max, *l, *r, stats)),

            TLExpr::Eq(l, r) => Ok(self.map_binary(TLExpr::Eq, *l, *r, stats)),
            TLExpr::Lt(l, r) => Ok(self.map_binary(TLExpr::Lt, *l, *r, stats)),
            TLExpr::Gt(l, r) => Ok(self.map_binary(TLExpr::Gt, *l, *r, stats)),
            TLExpr::Lte(l, r) => Ok(self.map_binary(TLExpr::Lte, *l, *r, stats)),
            TLExpr::Gte(l, r) => Ok(self.map_binary(TLExpr::Gte, *l, *r, stats)),

            TLExpr::Abs(e) => Ok(self.map_unary(TLExpr::Abs, *e, stats)),
            TLExpr::Floor(e) => Ok(self.map_unary(TLExpr::Floor, *e, stats)),
            TLExpr::Ceil(e) => Ok(self.map_unary(TLExpr::Ceil, *e, stats)),
            TLExpr::Round(e) => Ok(self.map_unary(TLExpr::Round, *e, stats)),
            TLExpr::Sqrt(e) => Ok(self.map_unary(TLExpr::Sqrt, *e, stats)),
            TLExpr::Exp(e) => Ok(self.map_unary(TLExpr::Exp, *e, stats)),
            TLExpr::Log(e) => Ok(self.map_unary(TLExpr::Log, *e, stats)),
            TLExpr::Sin(e) => Ok(self.map_unary(TLExpr::Sin, *e, stats)),
            TLExpr::Cos(e) => Ok(self.map_unary(TLExpr::Cos, *e, stats)),
            TLExpr::Tan(e) => Ok(self.map_unary(TLExpr::Tan, *e, stats)),
            TLExpr::Score(e) => Ok(self.map_unary(TLExpr::Score, *e, stats)),

            TLExpr::Box(e) => Ok(self.map_unary(TLExpr::Box, *e, stats)),
            TLExpr::Diamond(e) => Ok(self.map_unary(TLExpr::Diamond, *e, stats)),
            TLExpr::Next(e) => Ok(self.map_unary(TLExpr::Next, *e, stats)),
            TLExpr::Eventually(e) => Ok(self.map_unary(TLExpr::Eventually, *e, stats)),
            TLExpr::Always(e) => Ok(self.map_unary(TLExpr::Always, *e, stats)),

            TLExpr::Until { before, after } => {
                let (nb, cb) = self.eliminate(*before, stats);
                let (na, ca) = self.eliminate(*after, stats);
                Ok((
                    TLExpr::Until {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                ))
            }
            TLExpr::Release { released, releaser } => {
                let (nr, cr) = self.eliminate(*released, stats);
                let (ne, ce) = self.eliminate(*releaser, stats);
                Ok((
                    TLExpr::Release {
                        released: Box::new(nr),
                        releaser: Box::new(ne),
                    },
                    cr || ce,
                ))
            }
            TLExpr::WeakUntil { before, after } => {
                let (nb, cb) = self.eliminate(*before, stats);
                let (na, ca) = self.eliminate(*after, stats);
                Ok((
                    TLExpr::WeakUntil {
                        before: Box::new(nb),
                        after: Box::new(na),
                    },
                    cb || ca,
                ))
            }
            TLExpr::StrongRelease { released, releaser } => {
                let (nr, cr) = self.eliminate(*released, stats);
                let (ne, ce) = self.eliminate(*releaser, stats);
                Ok((
                    TLExpr::StrongRelease {
                        released: Box::new(nr),
                        releaser: Box::new(ne),
                    },
                    cr || ce,
                ))
            }

            other => Err(other),
        }
    }
}
