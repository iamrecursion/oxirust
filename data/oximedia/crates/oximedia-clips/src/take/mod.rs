//! Take management for multi-take shots.

pub mod manager;
pub mod multi_criteria;
pub mod selector;

pub use manager::TakeManager;
pub use multi_criteria::{
    rank_takes_multi_criteria, MultiCriteriaTakeSelector, TakeScoreWeights, TakeWeights,
};
pub use selector::{Take, TakeId, TakeSelector};
