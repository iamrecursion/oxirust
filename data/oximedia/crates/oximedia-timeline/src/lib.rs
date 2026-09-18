//! Multi-track timeline editor for `OxiMedia`.
//!
//! `oximedia-timeline` provides a professional-grade timeline editor with support for:
//!
//! - Multi-track video and audio editing
//! - Frame-accurate editing operations
//! - Transitions and effects with keyframe animation
//! - Professional editing operations (slip, slide, roll, ripple)
//! - Multi-camera editing and nested sequences
//! - Markers, metadata, and comments
//! - EDL/XML/AAF import/export
//! - Real-time playback with caching
//!
//! # Example
//!
//! ```
//! use oximedia_timeline::{Timeline, Clip, MediaSource, Transition, TransitionType};
//! use oximedia_timeline::types::{Duration, Position};
//! use oximedia_core::Rational;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new timeline
//! let mut timeline = Timeline::new(
//!     "My Project",
//!     Rational::new(24000, 1001),  // 23.976 fps
//!     48000,                        // 48kHz audio
//! )?;
//!
//! // Add tracks
//! let video_track = timeline.add_video_track("Video 1")?;
//! let _audio_track = timeline.add_audio_track("Audio 1")?;
//!
//! // Create and insert a clip
//! let clip = Clip::new(
//!     "My Clip".to_string(),
//!     MediaSource::black(),
//!     Position::new(0),
//!     Position::new(100),
//!     Position::new(0),
//! )?;
//! let clip_id = clip.id;
//! timeline.insert_clip(video_track, clip, Position::new(0))?;
//!
//! // Add a transition
//! let transition = Transition::dissolve(Duration::new(24));
//! timeline.add_transition(clip_id, transition)?;
//!
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod adjustment_layer;
pub mod asymmetric_transition;
pub mod audio;
pub mod auto_sequence;
pub mod beat_markers;
pub mod cache;
pub mod cache_warming;
pub mod clip;
pub mod clip_metadata;
pub mod clip_sequence;
pub mod color_correction_track;
pub mod compound_clip;
pub mod compound_region;
pub mod conform;
pub mod edit;
pub mod effects;
pub mod error;
pub mod export;
pub mod export_settings;
pub mod fcpxml_export;
pub mod freeze_frame;
pub mod gap_filler;
pub mod import;
pub mod incremental_render;
pub mod interval_tree;
pub mod keyframe;
pub mod keyframe_animation;
pub mod lazy_clip;
pub mod linked_clips;
pub mod marker;
pub mod markers;
pub mod metadata;
pub mod mixer;
pub mod multi_select;
pub mod multicam;
pub mod nested;
pub mod nested_compound;
pub mod nested_timeline;
pub mod otio_export;
#[cfg(not(target_arch = "wasm32"))]
pub mod playback;
pub mod point_edit;
pub mod proxy_workflow;
pub mod razor_tool;
pub mod render;
pub mod render_queue;
pub mod renderer;
pub mod sequence;
pub mod sequence_range;
pub mod snap_grid;
pub mod speed_ramp;
pub mod timecode;
pub mod timeline;
pub mod timeline_compare;
pub mod timeline_diff;
pub mod timeline_event;
pub mod timeline_exporter;
pub mod timeline_lock;
pub mod track;
pub mod track_color;
pub mod track_group;
pub mod track_routing;
pub mod transition;
pub mod transition_engine;
pub mod types;
pub mod version_snapshot;

// Re-export commonly used items
pub use asymmetric_transition::{AsymmetricTransition, AsymmetricTransitionWrapper, EasingCurve};
pub use clip::{Clip, ClipId, MediaSource};
pub use edit::{EditMode, EditOperation};
pub use effects::{Effect, EffectId, EffectStack};
pub use error::{TimelineError, TimelineResult};
pub use keyframe::{Keyframe, KeyframeInterpolation, KeyframeValue};
pub use marker::{Marker, MarkerType};
pub use metadata::Metadata;
pub use mixer::{AudioFrame, MixResult, TrackMixParams, TrackMixer};
pub use renderer::{
    render_frame_parallel, CompositeFrame, FrameRenderSettings, PixelBuffer, RenderConfig,
    RenderedFrame, TimelineRenderer,
};
pub use timecode::{TimecodeFormat, TimecodeValue};
pub use timeline::Timeline;
pub use timeline_exporter::{
    frames_to_tc, tc_to_frames, EdlEvent, EdlExportOptions, TimelineExporter,
};
pub use track::{Track, TrackId, TrackType};
pub use transition::{Transition, TransitionAlignment, TransitionType};
pub use transition_engine::{TransitionEngine, TransitionError, TransitionInput};
pub use types::{Duration, Position, Speed};
