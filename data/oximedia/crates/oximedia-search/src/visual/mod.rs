//! Visual similarity search.

pub mod features;
pub mod index;
pub mod search;
pub mod vp_indexed;
pub mod vp_tree;

pub use features::FeatureExtractor;
pub use index::VisualIndex;
pub use search::VisualSearch;
pub use vp_indexed::VpIndexedVisual;
pub use vp_tree::FloatVpTree;
