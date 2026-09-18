//! Model-level **speculative decoding** for language-model generation.
//!
//! This module implements the rejection-sampling acceleration scheme of
//! Leviathan et al. (2023) ["Fast Inference from Transformers via Speculative
//! Decoding"](https://arxiv.org/abs/2211.17192) and Chen et al. (2023)
//! ["Accelerating Large Language Model Decoding with Speculative
//! Sampling"](https://arxiv.org/abs/2302.01318).  In their formulation a
//! small *draft model* proposes the next `k` tokens cheaply; a larger
//! *target model* verifies them in a single parallel forward pass.  A
//! per-token Bernoulli acceptance test decides which draft tokens are kept —
//! and the first rejected position is re-sampled from an *adjusted* target
//! distribution so that the algorithm's marginal output is provably
//! indistinguishable from sampling directly from the target.
//!
//! ## Relation to `tensorlogic-infer::speculative`
//!
//! That module implements **execution-level** speculation over tensor
//! pipelines (branch prediction, prefetching, rollback).  This one targets
//! **model-level** generation: draft/target LLM pairs.  The two are
//! deliberately separate — they don't share abstractions because the
//! respective correctness contracts differ.
//!
//! ## Pseudo-code (Leviathan et al. §3)
//!
//! ```text
//! Input:  prefix x_1 .. x_t,  max_tokens T,  draft model p,  target model q
//! Output: extension y_1 .. y_L  with L ≤ T
//!
//! while  |y| < T:
//!     # 1. Draft proposes k tokens and returns full per-step distributions.
//!     (tokens, p_dists) = p.propose(prefix+y, k)
//!
//!     # 2. Target scores k+1 positions in one forward pass.
//!     q_dists = q.verify(prefix+y, tokens)                 // length k+1
//!
//!     # 3. Rejection sweep.
//!     for i in 0..k:
//!         r = U[0,1)
//!         if r < min(1, q_dists[i][tokens[i]] / p_dists[i][tokens[i]]):
//!             append tokens[i] to y                        # ACCEPT
//!         else:
//!             resample from  max(0, q_dists[i] - p_dists[i]) / Z
//!             append that token to y                       # REJECT & resample
//!             break
//!     else:
//!         # all k accepted → sample bonus token from q_dists[k]
//!         append sample(q_dists[k]) to y
//! ```
//!
//! Both the per-token acceptance rule and the adjusted resampling step are
//! exported as pure functions from [`acceptance`] so they can be unit-tested
//! in isolation.
//!
//! ## Public surface
//!
//! * [`DraftModel`], [`TargetModel`] — the two trait layers.
//! * [`DraftProposal`], [`TargetScores`] — shape-typed outputs.
//! * [`SpeculativeDecoder`], [`SpeculativeDecoderConfig`] — the engine.
//! * [`SpeculativeMetrics`] — throughput / accept-rate / speedup counters.
//! * [`MockDraftModel`], [`MockTargetModel`] — test fixtures.
//! * [`SpeculativeDecodingError`], [`SpeculativeDecodingResult`] — errors.

pub mod acceptance;
pub mod engine;
pub mod error;
pub mod metrics;
pub mod mock_models;
pub mod rng;
pub mod traits;

#[cfg(test)]
mod tests;

pub use acceptance::{
    accept, adjusted_distribution, resample_from_adjusted_target, sample_from_logprobs,
    sample_index,
};
pub use engine::{SpeculativeDecoder, SpeculativeDecoderConfig};
pub use error::{SpeculativeDecodingError, SpeculativeDecodingResult};
pub use metrics::SpeculativeMetrics;
pub use mock_models::{FixedDistDraftModel, FixedDistTargetModel, MockDraftModel, MockTargetModel};
pub use rng::SpecRng;
pub use traits::{DraftModel, DraftProposal, LogProb, TargetModel, TargetScores, TokenId};
