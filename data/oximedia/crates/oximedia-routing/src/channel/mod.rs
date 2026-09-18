//! Channel mapping and management module.

pub mod extract;
pub mod map;
pub mod split;

pub use extract::{ChannelExtractor, ChannelInfo, ChannelSelector, ExtractError};
pub use map::{
    ChannelLayout, ChannelMap, ChannelMapError, ChannelMapManager, ChannelPosition, ChannelRemapper,
};
pub use split::{ChannelCombiner, ChannelSplitter, Combine, Split, SplitError};
