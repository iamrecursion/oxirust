//! Professional clip management and logging system for `OxiMedia`.
//!
//! This crate provides comprehensive clip management functionality for professional
//! video editing and logging workflows, including:
//!
//! - **Clip Database**: Store and manage video clips with metadata
//! - **Subclip Creation**: Create subclips with in/out points
//! - **Clip Grouping**: Organize clips into bins, folders, and collections
//! - **Professional Logging**: Keywords, markers, ratings, and notes
//! - **Take Management**: Track multiple takes of the same shot
//! - **Proxy Association**: Link clips to proxy versions
//! - **Smart Collections**: Auto-updating collections based on criteria
//! - **Search and Filter**: Advanced search and filtering capabilities
//! - **Import/Export**: EDL, XML, CSV, JSON export
//!
//! # Example
//!
//! ```rust
//! use oximedia_clips::{ClipManager, Clip, Rating};
//! use oximedia_core::types::Rational;
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a clip manager
//! let manager = ClipManager::new(":memory:").await?;
//!
//! // Create a new clip
//! let mut clip = Clip::new(PathBuf::from("/path/to/video.mov"));
//! clip.set_name("Interview Take 1");
//! clip.set_rating(Rating::FourStars);
//! clip.add_keyword("interview");
//! clip.add_keyword("john-doe");
//!
//! // Save the clip
//! let clip_id = manager.add_clip(clip).await?;
//!
//! // Search for clips
//! let results = manager.search("interview").await?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::format_push_string)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

pub mod ai_tagging;
pub mod bin;
pub mod bin_organizer;
pub mod camera_metadata;
pub mod clip;
pub mod clip_ai_tag;
pub mod clip_audit;
pub mod clip_bin;
pub mod clip_collaboration;
pub mod clip_compare;
pub mod clip_export;
pub mod clip_face_index;
pub mod clip_favorites;
pub mod clip_fingerprint;
pub mod clip_history;
pub mod clip_merge;
pub mod clip_metadata;
pub mod clip_playlist;
pub mod clip_relations;
pub mod clip_relations_bidirectional;
pub mod clip_scene_detect;
pub mod clip_search;
pub mod clip_tag;
pub mod clip_timeline;
pub mod clip_transcode_status;
pub mod clip_versioning;
pub mod clip_waveform;
#[cfg(not(target_arch = "wasm32"))]
pub mod database;
pub mod export;
pub mod group;
#[cfg(not(target_arch = "wasm32"))]
pub mod import;
pub mod logging;
pub mod marker;
pub mod note;
pub mod offline;
pub mod proxy;
pub mod proxy_link;
pub mod proxy_meta;
pub mod rating;
pub mod rating_store;
pub mod search;
pub mod storyboard;
pub mod subclip;
pub mod sync;
pub mod take;
pub mod trim;
pub mod usage;
pub mod version;

mod error;
#[cfg(not(target_arch = "wasm32"))]
mod manager;

pub use ai_tagging::{AiTagger, AiTaggerConfig, ClipInfo, TagSource, TagSuggestion};
pub use camera_metadata::{CameraMetadata, CameraMetadataExt, CameraMetadataStore};
pub use clip::{Clip, ClipId, ClipMetadata, SubClip};
pub use clip_waveform::{ClipWaveformGenerator, WaveformData, WaveformThumbnail};
pub use error::{ClipError, ClipResult};
pub use group::{
    Bin, BinId, ClipField, Collection, CollectionId, Comparison, Folder, FolderId, MatchMode,
    SmartCollection, SmartRule,
};
pub use logging::{Favorite, Keyword, Rating};
#[cfg(not(target_arch = "wasm32"))]
pub use manager::ClipManager;
pub use marker::{Marker, MarkerId, MarkerType};
pub use note::{Annotation, Note, NoteId};
pub use proxy::{ProxyLink, ProxyQuality};
pub use proxy_meta::{ProxyManagerSpec, ProxyMetadata, ProxySpec, ProxyValidationError};
pub use take::multi_criteria::{
    rank_takes_multi_criteria, MultiCriteriaTakeSelector, TakeScoreWeights, TakeWeights,
};
pub use take::{Take, TakeId, TakeSelector};
