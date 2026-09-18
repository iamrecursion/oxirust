//! Playlist and program scheduling
//!
//! Handles time-based scheduling, frame-accurate triggering, cue points,
//! SCTE-35 marker insertion, and secondary events.

use crate::playlist::{Playlist, PlaylistFormat, PlaylistItem, PlaylistManager};
use crate::{PlayoutConfig, PlayoutError, Result};
use chrono::{DateTime, Datelike, Duration, NaiveTime, Utc, Weekday};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Enable automatic scheduling
    pub auto_schedule: bool,

    /// Schedule look-ahead time in hours
    pub lookahead_hours: u32,

    /// Enable SCTE-35 insertion
    pub scte35_enabled: bool,

    /// Default fill content when idle
    pub default_fill: Option<PathBuf>,

    /// Enable macro expansion
    pub macro_expansion: bool,

    /// Frame accuracy tolerance in frames
    pub frame_tolerance: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            auto_schedule: true,
            lookahead_hours: 24,
            scte35_enabled: true,
            default_fill: None,
            macro_expansion: true,
            frame_tolerance: 1,
        }
    }
}

/// Schedule event type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EventType {
    /// Play content item
    PlayContent,
    /// Insert graphics overlay
    Graphics,
    /// Insert subtitles
    Subtitles,
    /// SCTE-35 splice insert
    Scte35Splice,
    /// Cue point
    CuePoint,
    /// Macro execution
    Macro,
    /// Emergency alert
    EmergencyAlert,
}

/// SCTE-35 splice command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scte35Command {
    /// Splice event ID
    pub event_id: u32,

    /// Pre-roll time in milliseconds
    pub pre_roll_ms: u64,

    /// Duration in milliseconds (None for return to network)
    pub duration_ms: Option<u64>,

    /// Auto-return flag
    pub auto_return: bool,

    /// Out of network indicator
    pub out_of_network: bool,

    /// Unique program ID
    pub program_id: u16,

    /// Segmentation type
    pub segmentation_type: Option<SegmentationType>,
}

/// SCTE-35 segmentation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SegmentationType {
    /// Program start
    ProgramStart,
    /// Program end
    ProgramEnd,
    /// Chapter start
    ChapterStart,
    /// Chapter end
    ChapterEnd,
    /// Provider advertisement start
    ProviderAdStart,
    /// Provider advertisement end
    ProviderAdEnd,
    /// Distributor advertisement start
    DistributorAdStart,
    /// Distributor advertisement end
    DistributorAdEnd,
    /// Unscheduled event start
    UnscheduledEventStart,
    /// Unscheduled event end
    UnscheduledEventEnd,
}

/// Cue point definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuePoint {
    /// Unique cue point ID
    pub id: String,

    /// Cue point name
    pub name: String,

    /// Frame offset from start
    pub frame_offset: u64,

    /// Cue point type
    pub cue_type: CueType,

    /// Associated data
    pub data: HashMap<String, String>,
}

/// Cue point types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CueType {
    /// Generic marker
    Marker,
    /// Ad break opportunity
    AdBreak,
    /// Chapter marker
    Chapter,
    /// Thumbnail position
    Thumbnail,
    /// Custom cue point
    Custom(String),
}

/// Transition type between content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Transition {
    /// Hard cut (no transition)
    Cut,
    /// Dissolve/crossfade
    Dissolve { duration_frames: u32 },
    /// Fade to black
    FadeToBlack { duration_frames: u32 },
    /// Fade from black
    FadeFromBlack { duration_frames: u32 },
    /// Wipe effect
    Wipe {
        duration_frames: u32,
        direction: WipeDirection,
    },
}

/// Wipe direction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WipeDirection {
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
}

/// Scheduled event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledEvent {
    /// Unique event ID
    pub id: Uuid,

    /// Event type
    pub event_type: EventType,

    /// Scheduled time (absolute)
    pub scheduled_time: DateTime<Utc>,

    /// Frame-accurate offset
    pub frame_offset: u64,

    /// Duration in frames
    pub duration_frames: Option<u64>,

    /// Content path (for PlayContent events)
    pub content_path: Option<PathBuf>,

    /// Transition in
    pub transition_in: Transition,

    /// Transition out
    pub transition_out: Transition,

    /// SCTE-35 command (if applicable)
    pub scte35: Option<Scte35Command>,

    /// Cue points
    pub cue_points: Vec<CuePoint>,

    /// Graphics overlay data
    pub graphics_data: Option<serde_json::Value>,

    /// Subtitle data
    pub subtitle_data: Option<serde_json::Value>,

    /// Macro name (for Macro events)
    pub macro_name: Option<String>,

    /// Macro parameters
    pub macro_params: HashMap<String, String>,

    /// Priority (higher values take precedence)
    pub priority: u32,

    /// Enable/disable flag
    pub enabled: bool,

    /// Tags for categorization
    pub tags: Vec<String>,
}

impl ScheduledEvent {
    /// Create a new content playback event
    #[allow(clippy::too_many_arguments)]
    pub fn new_content(
        scheduled_time: DateTime<Utc>,
        content_path: PathBuf,
        duration_frames: Option<u64>,
        transition_in: Transition,
        transition_out: Transition,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type: EventType::PlayContent,
            scheduled_time,
            frame_offset: 0,
            duration_frames,
            content_path: Some(content_path),
            transition_in,
            transition_out,
            scte35: None,
            cue_points: Vec::new(),
            graphics_data: None,
            subtitle_data: None,
            macro_name: None,
            macro_params: HashMap::new(),
            priority: 100,
            enabled: true,
            tags: Vec::new(),
        }
    }

    /// Create a new SCTE-35 splice event
    pub fn new_scte35(scheduled_time: DateTime<Utc>, scte35: Scte35Command) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type: EventType::Scte35Splice,
            scheduled_time,
            frame_offset: 0,
            duration_frames: scte35.duration_ms.map(|ms| (ms * 25) / 1000), // Approximate for 25fps
            content_path: None,
            transition_in: Transition::Cut,
            transition_out: Transition::Cut,
            scte35: Some(scte35),
            cue_points: Vec::new(),
            graphics_data: None,
            subtitle_data: None,
            macro_name: None,
            macro_params: HashMap::new(),
            priority: 200,
            enabled: true,
            tags: Vec::new(),
        }
    }

    /// Add a cue point to this event
    pub fn add_cue_point(&mut self, cue_point: CuePoint) {
        self.cue_points.push(cue_point);
    }
}

/// Recurrence pattern for scheduled events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecurrencePattern {
    /// One-time event
    Once,
    /// Daily at specific time
    Daily { time: NaiveTime },
    /// Weekly on specific days
    Weekly { days: Vec<Weekday>, time: NaiveTime },
    /// Monthly on specific day
    Monthly { day: u32, time: NaiveTime },
    /// Custom interval
    Custom { interval_seconds: u64 },
}

/// Program schedule template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramTemplate {
    /// Template ID
    pub id: String,

    /// Template name
    pub name: String,

    /// Description
    pub description: String,

    /// Recurrence pattern
    pub recurrence: RecurrencePattern,

    /// Events in this program
    pub events: Vec<ScheduledEvent>,

    /// Program duration in frames
    pub duration_frames: u64,

    /// Enable/disable flag
    pub enabled: bool,

    /// Start date (when to begin scheduling)
    pub start_date: DateTime<Utc>,

    /// End date (when to stop scheduling)
    pub end_date: Option<DateTime<Utc>>,
}

/// Macro definition for complex operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroDefinition {
    /// Macro name
    pub name: String,

    /// Description
    pub description: String,

    /// Parameters with default values
    pub parameters: HashMap<String, String>,

    /// Events to generate
    pub events: Vec<ScheduledEvent>,

    /// Enable/disable flag
    pub enabled: bool,
}

/// Schedule execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Current playback time
    pub current_time: DateTime<Utc>,

    /// Current frame number
    pub current_frame: u64,

    /// Frame rate
    pub frame_rate: f64,

    /// Active events
    pub active_events: Vec<Uuid>,
}

/// Internal scheduler state
struct SchedulerState {
    /// Scheduled events (sorted by time)
    events: BTreeMap<DateTime<Utc>, Vec<ScheduledEvent>>,

    /// Program templates
    programs: HashMap<String, ProgramTemplate>,

    /// Macro definitions
    macros: HashMap<String, MacroDefinition>,

    /// Execution context
    context: ExecutionContext,
}

/// Playlist and program scheduler
pub struct Scheduler {
    config: SchedulerConfig,
    state: Arc<RwLock<SchedulerState>>,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        let context = ExecutionContext {
            current_time: Utc::now(),
            current_frame: 0,
            frame_rate: 25.0,
            active_events: Vec::new(),
        };

        let state = SchedulerState {
            events: BTreeMap::new(),
            programs: HashMap::new(),
            macros: HashMap::new(),
            context,
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Load a playlist from file.
    ///
    /// The playlist format is inferred from the file extension:
    ///   `.json`        → JSON
    ///   `.xml`         → XML
    ///   `.smil`        → SMIL
    ///   `.m3u8`/`.m3u` → M3U8
    ///
    /// Loaded items are converted into `ScheduledEvent::PlayContent` entries
    /// scheduled sequentially starting from `Utc::now()`.  Each item is
    /// given a duration slot proportional to its frame count (at 25 fps).
    pub async fn load_playlist(&self, path: PathBuf) -> Result<()> {
        // Determine format from extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let format = match ext.as_str() {
            "json" => PlaylistFormat::JSON,
            "xml" => PlaylistFormat::XML,
            "smil" => PlaylistFormat::SMIL,
            "m3u8" | "m3u" => PlaylistFormat::M3U8,
            other => {
                return Err(PlayoutError::Playlist(format!(
                    "Unsupported playlist format extension: '{other}'"
                )))
            }
        };

        // Parse the playlist using the existing PlaylistManager helper
        let manager = PlaylistManager::new();
        let playlist_id = manager.load_from_file(&path, format)?;
        let playlist: Playlist = manager.get_playlist(playlist_id).ok_or_else(|| {
            PlayoutError::Playlist("Failed to retrieve parsed playlist".to_string())
        })?;

        // Schedule playlist items sequentially
        let fps = {
            let state = self.state.read();
            state.context.frame_rate
        };

        let mut cursor = Utc::now();
        let mut state = self.state.write();

        for item in &playlist.items {
            if !item.enabled {
                continue;
            }

            let event = ScheduledEvent::new_content(
                cursor,
                item.path.clone(),
                item.effective_duration(),
                item.transition_in.clone(),
                item.transition_out.clone(),
            );

            state.events.entry(cursor).or_default().push(event);

            // Advance cursor by this item's duration
            if let Some(frames) = item.effective_duration() {
                let duration_ms = ((frames as f64 / fps) * 1_000.0) as i64;
                cursor += Duration::milliseconds(duration_ms);
            }
        }

        tracing::info!(
            "Loaded playlist '{}' from {:?}: {} items scheduled",
            playlist.metadata.name,
            path,
            playlist.items.len()
        );

        Ok(())
    }

    /// Add a scheduled event
    pub fn add_event(&self, event: ScheduledEvent) {
        let mut state = self.state.write();
        state
            .events
            .entry(event.scheduled_time)
            .or_default()
            .push(event);
    }

    /// Remove a scheduled event by ID
    pub fn remove_event(&self, event_id: Uuid) -> Result<()> {
        let mut state = self.state.write();
        for events in state.events.values_mut() {
            events.retain(|e| e.id != event_id);
        }
        Ok(())
    }

    /// Get events in time range
    pub fn get_events_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<ScheduledEvent> {
        let state = self.state.read();
        state
            .events
            .range(start..=end)
            .flat_map(|(_, events)| events.iter().cloned())
            .collect()
    }

    /// Get next event to execute
    pub fn get_next_event(&self) -> Option<ScheduledEvent> {
        let state = self.state.read();
        let current_time = state.context.current_time;

        state
            .events
            .range(current_time..)
            .next()
            .and_then(|(_, events)| events.first().cloned())
    }

    /// Add a program template
    pub fn add_program(&self, program: ProgramTemplate) {
        let mut state = self.state.write();
        state.programs.insert(program.id.clone(), program);
    }

    /// Remove a program template
    pub fn remove_program(&self, program_id: &str) -> Result<()> {
        let mut state = self.state.write();
        state.programs.remove(program_id);
        Ok(())
    }

    /// Generate schedule from program templates
    pub fn generate_schedule(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<()> {
        // Clone programs first to avoid borrow conflicts
        let programs: Vec<ProgramTemplate> = {
            let state = self.state.read();
            state.programs.values().cloned().collect()
        };

        let mut state = self.state.write();

        for program in &programs {
            if !program.enabled {
                continue;
            }

            if program.start_date > end {
                continue;
            }

            if let Some(end_date) = program.end_date {
                if end_date < start {
                    continue;
                }
            }

            // Generate events based on recurrence pattern
            match &program.recurrence {
                RecurrencePattern::Once => {
                    for event in &program.events {
                        let mut new_event = event.clone();
                        new_event.id = Uuid::new_v4();
                        state
                            .events
                            .entry(new_event.scheduled_time)
                            .or_default()
                            .push(new_event);
                    }
                }
                RecurrencePattern::Daily { time } => {
                    let mut current = start;
                    while current <= end {
                        let scheduled_time = current.date_naive().and_time(*time).and_utc();
                        if scheduled_time >= program.start_date
                            && program.end_date.is_none_or(|ed| scheduled_time <= ed)
                        {
                            for event in &program.events {
                                let mut new_event = event.clone();
                                new_event.id = Uuid::new_v4();
                                new_event.scheduled_time = scheduled_time;
                                state
                                    .events
                                    .entry(new_event.scheduled_time)
                                    .or_default()
                                    .push(new_event);
                            }
                        }
                        current += Duration::days(1);
                    }
                }
                RecurrencePattern::Weekly { days, time } => {
                    let mut current = start;
                    while current <= end {
                        if days.contains(&current.weekday()) {
                            let scheduled_time = current.date_naive().and_time(*time).and_utc();
                            if scheduled_time >= program.start_date
                                && program.end_date.is_none_or(|ed| scheduled_time <= ed)
                            {
                                for event in &program.events {
                                    let mut new_event = event.clone();
                                    new_event.id = Uuid::new_v4();
                                    new_event.scheduled_time = scheduled_time;
                                    state
                                        .events
                                        .entry(new_event.scheduled_time)
                                        .or_default()
                                        .push(new_event);
                                }
                            }
                        }
                        current += Duration::days(1);
                    }
                }
                RecurrencePattern::Monthly { day, time } => {
                    let mut current = start;
                    while current <= end {
                        if current.day() == *day {
                            let scheduled_time = current.date_naive().and_time(*time).and_utc();
                            if scheduled_time >= program.start_date
                                && program.end_date.is_none_or(|ed| scheduled_time <= ed)
                            {
                                for event in &program.events {
                                    let mut new_event = event.clone();
                                    new_event.id = Uuid::new_v4();
                                    new_event.scheduled_time = scheduled_time;
                                    state
                                        .events
                                        .entry(new_event.scheduled_time)
                                        .or_default()
                                        .push(new_event);
                                }
                            }
                        }
                        current += Duration::days(1);
                    }
                }
                RecurrencePattern::Custom { interval_seconds } => {
                    let mut current = program.start_date;
                    while current <= end {
                        if current >= start && program.end_date.is_none_or(|ed| current <= ed) {
                            for event in &program.events {
                                let mut new_event = event.clone();
                                new_event.id = Uuid::new_v4();
                                new_event.scheduled_time = current;
                                state
                                    .events
                                    .entry(new_event.scheduled_time)
                                    .or_default()
                                    .push(new_event);
                            }
                        }
                        current += Duration::seconds(*interval_seconds as i64);
                    }
                }
            }
        }

        Ok(())
    }

    /// Add a macro definition
    pub fn add_macro(&self, macro_def: MacroDefinition) {
        let mut state = self.state.write();
        state.macros.insert(macro_def.name.clone(), macro_def);
    }

    /// Expand a macro into events
    pub fn expand_macro(
        &self,
        macro_name: &str,
        params: HashMap<String, String>,
        base_time: DateTime<Utc>,
    ) -> Result<Vec<ScheduledEvent>> {
        let state = self.state.read();

        let macro_def = state
            .macros
            .get(macro_name)
            .ok_or_else(|| PlayoutError::Scheduler(format!("Macro not found: {macro_name}")))?;

        if !macro_def.enabled {
            return Err(PlayoutError::Scheduler(format!(
                "Macro disabled: {macro_name}"
            )));
        }

        // Merge parameters with defaults
        let mut final_params = macro_def.parameters.clone();
        for (key, value) in params {
            final_params.insert(key, value);
        }

        // Generate events from template
        let mut events = Vec::new();
        for template_event in &macro_def.events {
            let mut event = template_event.clone();
            event.id = Uuid::new_v4();
            event.scheduled_time = base_time;

            // Apply parameter substitution
            if let Some(ref mut path) = event.content_path {
                let path_str = path.to_string_lossy().to_string();
                let expanded = self.substitute_params(&path_str, &final_params);
                *path = PathBuf::from(expanded);
            }

            events.push(event);
        }

        Ok(events)
    }

    /// Substitute parameters in a string
    fn substitute_params(&self, template: &str, params: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in params {
            let placeholder = format!("${{{key}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Update execution context
    pub fn update_context(&self, current_time: DateTime<Utc>, current_frame: u64) {
        let mut state = self.state.write();
        state.context.current_time = current_time;
        state.context.current_frame = current_frame;
    }

    /// Get current execution context
    pub fn get_context(&self) -> ExecutionContext {
        self.state.read().context.clone()
    }

    /// Check if event should execute now
    pub fn should_execute(&self, event: &ScheduledEvent) -> bool {
        let state = self.state.read();
        let current_time = state.context.current_time;
        let current_frame = state.context.current_frame;

        if !event.enabled {
            return false;
        }

        // Check time-based trigger
        if event.scheduled_time > current_time {
            return false;
        }

        // Check frame-accurate trigger
        let frame_diff = current_frame.abs_diff(event.frame_offset);
        if frame_diff > self.config.frame_tolerance as u64 {
            return false;
        }

        true
    }

    /// Mark event as active
    pub fn mark_active(&self, event_id: Uuid) {
        let mut state = self.state.write();
        if !state.context.active_events.contains(&event_id) {
            state.context.active_events.push(event_id);
        }
    }

    /// Mark event as complete
    pub fn mark_complete(&self, event_id: Uuid) {
        let mut state = self.state.write();
        state.context.active_events.retain(|id| *id != event_id);
    }

    /// Get active events
    pub fn get_active_events(&self) -> Vec<Uuid> {
        self.state.read().context.active_events.clone()
    }

    /// Clear all scheduled events
    pub fn clear_schedule(&self) {
        let mut state = self.state.write();
        state.events.clear();
    }

    /// Get total number of scheduled events
    pub fn event_count(&self) -> usize {
        let state = self.state.read();
        state.events.values().map(std::vec::Vec::len).sum()
    }

    /// Insert SCTE-35 marker at specific time
    pub fn insert_scte35(&self, scheduled_time: DateTime<Utc>, command: Scte35Command) {
        if !self.config.scte35_enabled {
            return;
        }

        let event = ScheduledEvent::new_scte35(scheduled_time, command);
        self.add_event(event);
    }

    /// Process cue points for an event
    pub fn process_cue_points(&self, event: &ScheduledEvent, current_frame: u64) -> Vec<CuePoint> {
        let mut triggered_cues = Vec::new();

        for cue in &event.cue_points {
            let cue_frame = event.frame_offset + cue.frame_offset;
            let frame_diff = current_frame.abs_diff(cue_frame);

            if frame_diff <= self.config.frame_tolerance as u64 {
                triggered_cues.push(cue.clone());
            }
        }

        triggered_cues
    }

    /// Calculate frame number from time
    pub fn time_to_frame(&self, time: DateTime<Utc>, start_time: DateTime<Utc>) -> u64 {
        let state = self.state.read();
        let duration = time - start_time;
        let seconds = duration.num_milliseconds() as f64 / 1000.0;
        (seconds * state.context.frame_rate) as u64
    }

    /// Calculate time from frame number
    pub fn frame_to_time(&self, frame: u64, start_time: DateTime<Utc>) -> DateTime<Utc> {
        let state = self.state.read();
        let seconds = frame as f64 / state.context.frame_rate;
        let duration = Duration::milliseconds((seconds * 1000.0) as i64);
        start_time + duration
    }

    /// Validate schedule for conflicts
    pub fn validate_schedule(&self) -> Vec<String> {
        let state = self.state.read();
        let mut warnings = Vec::new();

        // Check for overlapping events
        let mut prev_end: Option<DateTime<Utc>> = None;

        for (time, events) in &state.events {
            for event in events {
                if let Some(prev) = prev_end {
                    if *time < prev {
                        warnings.push(format!("Event {} overlaps with previous event", event.id));
                    }
                }

                if let Some(duration) = event.duration_frames {
                    let end_time =
                        self.frame_to_time(event.frame_offset + duration, event.scheduled_time);
                    prev_end = Some(end_time);
                }
            }
        }

        warnings
    }

    /// Auto-fill gaps in schedule with default content
    pub fn auto_fill_gaps(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<()> {
        if self.config.default_fill.is_none() {
            return Ok(());
        }

        let fill_path = match self.config.default_fill.as_ref() {
            Some(p) => p.clone(),
            None => return Ok(()),
        };

        // Collect events first with cloning to avoid borrow conflicts
        let sorted_events = {
            let state = self.state.read();
            let mut events: Vec<_> = state
                .events
                .range(start..=end)
                .flat_map(|(_, events)| events.iter().cloned())
                .collect();
            events.sort_by_key(|e| e.scheduled_time);
            events
        };

        let mut state = self.state.write();
        let mut current = start;

        for event in &sorted_events {
            if event.scheduled_time > current {
                // Gap found, insert fill content
                let fill_event = ScheduledEvent::new_content(
                    current,
                    fill_path.clone(),
                    None,
                    Transition::Cut,
                    Transition::Cut,
                );

                state.events.entry(current).or_default().push(fill_event);
            }

            if let Some(duration) = event.duration_frames {
                current = self.frame_to_time(event.frame_offset + duration, event.scheduled_time);
            }
        }

        Ok(())
    }

    /// Export schedule to JSON
    pub fn export_schedule(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<String> {
        let events = self.get_events_in_range(start, end);
        serde_json::to_string_pretty(&events)
            .map_err(|e| PlayoutError::Scheduler(format!("Export failed: {e}")))
    }

    /// Import schedule from JSON
    pub fn import_schedule(&self, json: &str) -> Result<()> {
        let events: Vec<ScheduledEvent> = serde_json::from_str(json)
            .map_err(|e| PlayoutError::Scheduler(format!("Import failed: {e}")))?;

        for event in events {
            self.add_event(event);
        }

        Ok(())
    }
}

// ─── Multi-channel support ────────────────────────────────────────────────────

/// Opaque identifier for a single broadcast channel managed by
/// `MultiChannelScheduler`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId(pub u32);

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CH{}", self.0)
    }
}

/// Manages an independent `Scheduler` per broadcast channel.
///
/// Each channel has its own `SchedulerConfig` and isolated event timeline,
/// enabling multi-channel playout from a single process.
pub struct MultiChannelScheduler {
    channels: HashMap<ChannelId, (PlayoutConfig, Arc<Scheduler>)>,
}

impl MultiChannelScheduler {
    /// Create an empty multi-channel scheduler.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Add a new channel with the given playout configuration.
    ///
    /// Returns `Err` if the channel already exists.
    pub fn add_channel(&mut self, id: ChannelId, config: PlayoutConfig) -> Result<()> {
        if self.channels.contains_key(&id) {
            return Err(PlayoutError::Scheduler(format!(
                "Channel {id} already exists"
            )));
        }
        let scheduler_config = SchedulerConfig::default();
        let scheduler = Arc::new(Scheduler::new(scheduler_config));
        self.channels.insert(id, (config, scheduler));
        Ok(())
    }

    /// Remove a channel and discard its schedule.
    ///
    /// Returns `Err` if the channel does not exist.
    pub fn remove_channel(&mut self, id: ChannelId) -> Result<()> {
        self.channels
            .remove(&id)
            .ok_or_else(|| PlayoutError::Scheduler(format!("Channel {id} not found")))?;
        Ok(())
    }

    /// Schedule a playlist item on the specified channel.
    pub fn schedule(&mut self, id: ChannelId, item: PlaylistItem) -> Result<()> {
        let (_, scheduler) = self
            .channels
            .get(&id)
            .ok_or_else(|| PlayoutError::Scheduler(format!("Channel {id} not found")))?;

        let event = ScheduledEvent::new_content(
            Utc::now(),
            item.path.clone(),
            item.effective_duration(),
            item.transition_in.clone(),
            item.transition_out.clone(),
        );
        scheduler.add_event(event);
        Ok(())
    }

    /// Return the next scheduled item for the given channel, if any.
    pub fn get_current(&self, id: ChannelId) -> Option<ScheduledEvent> {
        self.channels
            .get(&id)
            .and_then(|(_, sched)| sched.get_next_event())
    }

    /// Return all currently registered channel IDs.
    pub fn channel_ids(&self) -> Vec<ChannelId> {
        let mut ids: Vec<ChannelId> = self.channels.keys().copied().collect();
        ids.sort_by_key(|c| c.0);
        ids
    }

    /// Return the total number of scheduled events across all channels.
    pub fn total_event_count(&self) -> usize {
        self.channels.values().map(|(_, s)| s.event_count()).sum()
    }

    /// Clear the schedule for a single channel.
    pub fn clear_channel(&self, id: ChannelId) -> Result<()> {
        let (_, scheduler) = self
            .channels
            .get(&id)
            .ok_or_else(|| PlayoutError::Scheduler(format!("Channel {id} not found")))?;
        scheduler.clear_schedule();
        Ok(())
    }

    /// Return a reference to the underlying `Scheduler` for a channel.
    pub fn scheduler(&self, id: ChannelId) -> Option<Arc<Scheduler>> {
        self.channels.get(&id).map(|(_, s)| Arc::clone(s))
    }
}

impl Default for MultiChannelScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = Scheduler::new(config);
        assert_eq!(scheduler.event_count(), 0);
    }

    #[test]
    fn test_add_event() {
        let scheduler = Scheduler::new(SchedulerConfig::default());
        let event = ScheduledEvent::new_content(
            Utc::now(),
            PathBuf::from("/test.mxf"),
            Some(1000),
            Transition::Cut,
            Transition::Cut,
        );
        scheduler.add_event(event);
        assert_eq!(scheduler.event_count(), 1);
    }

    #[test]
    fn test_remove_event() {
        let scheduler = Scheduler::new(SchedulerConfig::default());
        let event = ScheduledEvent::new_content(
            Utc::now(),
            PathBuf::from("/test.mxf"),
            Some(1000),
            Transition::Cut,
            Transition::Cut,
        );
        let event_id = event.id;
        scheduler.add_event(event);
        assert_eq!(scheduler.event_count(), 1);

        scheduler
            .remove_event(event_id)
            .expect("should succeed in test");
        assert_eq!(scheduler.event_count(), 0);
    }

    #[test]
    fn test_scte35_event() {
        let scte35 = Scte35Command {
            event_id: 123,
            pre_roll_ms: 5000,
            duration_ms: Some(30000),
            auto_return: true,
            out_of_network: true,
            program_id: 1,
            segmentation_type: Some(SegmentationType::ProviderAdStart),
        };

        let event = ScheduledEvent::new_scte35(Utc::now(), scte35);
        assert_eq!(event.event_type, EventType::Scte35Splice);
        assert!(event.scte35.is_some());
    }

    #[test]
    fn test_macro_expansion() {
        let scheduler = Scheduler::new(SchedulerConfig::default());

        let macro_def = MacroDefinition {
            name: "test_macro".to_string(),
            description: "Test macro".to_string(),
            parameters: HashMap::from([("path".to_string(), "/default".to_string())]),
            events: vec![ScheduledEvent::new_content(
                Utc::now(),
                PathBuf::from("${path}"),
                Some(1000),
                Transition::Cut,
                Transition::Cut,
            )],
            enabled: true,
        };

        scheduler.add_macro(macro_def);

        let params = HashMap::from([("path".to_string(), "/custom.mxf".to_string())]);
        let events = scheduler
            .expand_macro("test_macro", params, Utc::now())
            .expect("should succeed in test");

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0]
                .content_path
                .as_ref()
                .expect("should succeed in test"),
            &PathBuf::from("/custom.mxf")
        );
    }

    #[test]
    fn test_time_to_frame_conversion() {
        let scheduler = Scheduler::new(SchedulerConfig::default());
        let start_time = Utc::now();
        let one_second_later = start_time + Duration::seconds(1);

        let frame = scheduler.time_to_frame(one_second_later, start_time);
        assert_eq!(frame, 25); // 25fps

        let back_to_time = scheduler.frame_to_time(25, start_time);
        assert_eq!(back_to_time, one_second_later);
    }

    #[test]
    fn test_cue_point() {
        let cue = CuePoint {
            id: "cue1".to_string(),
            name: "Ad Break".to_string(),
            frame_offset: 1000,
            cue_type: CueType::AdBreak,
            data: HashMap::new(),
        };

        assert_eq!(cue.frame_offset, 1000);
        assert_eq!(cue.cue_type, CueType::AdBreak);
    }

    // ── MultiChannelScheduler tests ──────────────────────────────────────

    fn make_playlist_item(path: &str) -> PlaylistItem {
        PlaylistItem::new(path.to_string(), PathBuf::from(path))
    }

    #[test]
    fn test_multi_channel_add_remove() {
        let mut mcs = MultiChannelScheduler::new();
        let id = ChannelId(1);
        mcs.add_channel(id, crate::PlayoutConfig::default())
            .expect("channel operation should succeed");
        assert_eq!(mcs.channel_ids(), vec![id]);
        mcs.remove_channel(id)
            .expect("channel operation should succeed");
        assert!(mcs.channel_ids().is_empty());
    }

    #[test]
    fn test_multi_channel_duplicate_add_fails() {
        let mut mcs = MultiChannelScheduler::new();
        let id = ChannelId(2);
        mcs.add_channel(id, crate::PlayoutConfig::default())
            .expect("channel operation should succeed");
        let res = mcs.add_channel(id, crate::PlayoutConfig::default());
        assert!(res.is_err());
    }

    #[test]
    fn test_multi_channel_remove_nonexistent_fails() {
        let mut mcs = MultiChannelScheduler::new();
        let res = mcs.remove_channel(ChannelId(99));
        assert!(res.is_err());
    }

    #[test]
    fn test_multi_channel_schedule_item() {
        let mut mcs = MultiChannelScheduler::new();
        let id = ChannelId(3);
        mcs.add_channel(id, crate::PlayoutConfig::default())
            .expect("channel operation should succeed");
        mcs.schedule(id, make_playlist_item("/clip.av1"))
            .expect("channel operation should succeed");
        assert_eq!(mcs.total_event_count(), 1);
    }

    #[test]
    fn test_multi_channel_schedule_unknown_channel_fails() {
        let mut mcs = MultiChannelScheduler::new();
        let res = mcs.schedule(ChannelId(42), make_playlist_item("/clip.av1"));
        assert!(res.is_err());
    }

    #[test]
    fn test_multi_channel_channel_ids_sorted() {
        let mut mcs = MultiChannelScheduler::new();
        for n in [5u32, 1, 3] {
            mcs.add_channel(ChannelId(n), crate::PlayoutConfig::default())
                .expect("channel operation should succeed");
        }
        let ids = mcs.channel_ids();
        assert_eq!(ids, vec![ChannelId(1), ChannelId(3), ChannelId(5)]);
    }

    #[test]
    fn test_multi_channel_get_current_none_when_empty() {
        let mut mcs = MultiChannelScheduler::new();
        let id = ChannelId(7);
        mcs.add_channel(id, crate::PlayoutConfig::default())
            .expect("channel operation should succeed");
        // No events scheduled yet; get_next_event may return None
        // (depends on current time being in the future relative to any event)
        // We just verify it doesn't panic.
        let _ = mcs.get_current(id);
    }

    #[test]
    fn test_multi_channel_total_event_count_multi() {
        let mut mcs = MultiChannelScheduler::new();
        for n in 1u32..=3 {
            mcs.add_channel(ChannelId(n), crate::PlayoutConfig::default())
                .expect("channel operation should succeed");
            mcs.schedule(ChannelId(n), make_playlist_item("/a.av1"))
                .expect("channel operation should succeed");
            mcs.schedule(ChannelId(n), make_playlist_item("/b.av1"))
                .expect("channel operation should succeed");
        }
        assert_eq!(mcs.total_event_count(), 6);
    }

    #[test]
    fn test_multi_channel_clear_channel() {
        let mut mcs = MultiChannelScheduler::new();
        let id = ChannelId(10);
        mcs.add_channel(id, crate::PlayoutConfig::default())
            .expect("channel operation should succeed");
        mcs.schedule(id, make_playlist_item("/x.av1"))
            .expect("channel operation should succeed");
        assert_eq!(mcs.total_event_count(), 1);
        mcs.clear_channel(id)
            .expect("channel operation should succeed");
        assert_eq!(mcs.total_event_count(), 0);
    }

    #[test]
    fn test_multi_channel_scheduler_ref() {
        let mut mcs = MultiChannelScheduler::new();
        let id = ChannelId(11);
        mcs.add_channel(id, crate::PlayoutConfig::default())
            .expect("channel operation should succeed");
        let sched = mcs.scheduler(id);
        assert!(sched.is_some());
    }

    #[test]
    fn test_channel_id_display() {
        let id = ChannelId(42);
        assert_eq!(id.to_string(), "CH42");
    }
}
