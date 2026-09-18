//! Clip organization and grouping.

pub mod bin;
pub mod collection;
pub mod folder;
pub mod smart;

pub use self::bin::{Bin, BinId};
pub use collection::{Collection, CollectionId};
pub use folder::{Folder, FolderId};
pub use smart::{ClipField, Comparison, MatchMode, SmartCollection, SmartRule};
