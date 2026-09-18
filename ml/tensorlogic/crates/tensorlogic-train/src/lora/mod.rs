//! LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
//!
//! Implements Hu et al. (2021): weight updates are decomposed as
//! `dW = B @ A` where `B in R^{d x r}`, `A in R^{r x k}`, and
//! `r << min(d, k)`, drastically reducing trainable parameter count.

pub mod adapter;
pub mod config;
pub mod error;
pub mod layer;

#[cfg(test)]
mod tests;

pub use adapter::{LayerStats, LoraAdapter, LoraAdapterSummary};
pub use config::LoraConfig;
pub use error::{LoraError, LoraResult};
pub use layer::LoraLayer;
