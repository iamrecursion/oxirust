//! Rule-Guided Sampling Decoder.
//!
//! This module implements a decoder that biases the beam-search algorithm
//! shipped in [`tensorlogic_infer::beam_search`] to prefer token sequences
//! consistent with a user-supplied [`tensorlogic_ir::TLExpr`] logical
//! constraint.
//!
//! Two enforcement strategies coexist:
//!
//! * **Hard masking** — forbidden tokens are hit with `f64::NEG_INFINITY`
//!   logits and are consequently eliminated from the candidate pool.
//! * **Soft re-weighting** — tokens that merely *violate* the constraint
//!   (without being outright forbidden) receive a log-probability penalty
//!   of `-lambda * violation_score`.  Forbidden tokens are still fully
//!   banned under soft mode — the soft rule only applies to the SoftPenalty
//!   verdict returned by the constraint.
//!
//! ## Public surface
//!
//! * [`RuleConstraint`] — wraps a `TLExpr` and compiles a vocabulary-level
//!   allow-list via a caller-supplied token-to-symbol mapper.
//! * [`ConstraintVerdict`] — per-token classification result.
//! * [`LogitMasker`] — trait implemented by [`HardMask`] and
//!   [`SoftPenaltyMask`].
//! * [`RuleGuidedBeamSearch`] — façade that plugs a constraint + masker into
//!   [`tensorlogic_infer::beam_search::BeamSearchDecoder`].
//! * [`RuleGuidedError`] / [`RuleGuidedResult`] — error taxonomy.
//!
//! ## Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use tensorlogic_infer::beam_search::BeamSearchConfig;
//! use tensorlogic_ir::{TLExpr, Term};
//! use tensorlogic_trustformers::rule_guided_decoder::{
//!     HardMask, LogitMasker, RuleConstraint, RuleGuidedBeamSearch,
//! };
//!
//! let expr = TLExpr::Pred {
//!     name: "entity".into(),
//!     args: vec![Term::Const("Alice".into())],
//! };
//! let mapper = |tid: usize| match tid {
//!     0 => Some("entity".into()),
//!     1 => Some("Alice".into()),
//!     _ => None,
//! };
//! let constraint = RuleConstraint::compile(expr, mapper).expect("compile");
//! let mask: Arc<dyn LogitMasker> = Arc::new(HardMask::new());
//! let cfg = BeamSearchConfig {
//!     beam_width: 2,
//!     max_length: 4,
//!     vocab_size: 2,
//!     ..BeamSearchConfig::default()
//! };
//! let decoder = RuleGuidedBeamSearch::new(cfg, constraint, mask);
//! // `decoder.decode(bos, score_fn)` now returns a BeamSearchResult whose
//! // hypotheses never include tokens that violate the constraint.
//! ```

pub mod constraint;
pub mod engine;
pub mod error;
pub mod mask;

#[cfg(test)]
mod tests;

pub use constraint::{ConstraintVerdict, RuleConstraint, TokenId, TokenSymbolMapper};
pub use engine::RuleGuidedBeamSearch;
pub use error::{RuleGuidedError, RuleGuidedResult};
pub use mask::{HardMask, LogitMasker, SoftPenaltyMask};
