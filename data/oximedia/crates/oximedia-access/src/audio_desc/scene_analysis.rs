//! Scene-analysis-driven automatic audio description.
//!
//! This module bridges the [`oximedia-analysis`](oximedia_analysis) crate into
//! the audio-description pipeline.  It consumes the visual analysis produced by
//! [`oximedia_analysis::Analyzer`] (scene/shot boundaries, transition types, and
//! content classification) and emits timed audio-description cues that are
//! **placed inside dialogue gaps** so that the narration never overlaps speech.
//!
//! # What is consumed from `oximedia-analysis`
//!
//! The integration reads the following fields of
//! [`oximedia_analysis::AnalysisResults`]:
//!
//! - [`scenes`](oximedia_analysis::AnalysisResults::scenes) — shot-boundary list.
//!   Each [`Scene`](oximedia_analysis::scene::Scene) records the frame at which a
//!   transition occurs (`end_frame`), the [`SceneChangeType`] (cut / fade /
//!   dissolve / wipe), and a confidence score.  Every boundary becomes a
//!   candidate audio-description trigger for the *new* shot that it introduces.
//! - [`frame_rate`](oximedia_analysis::AnalysisResults::frame_rate) — used to
//!   convert frame numbers into presentation-time milliseconds.
//! - [`content_classification`](oximedia_analysis::AnalysisResults::content_classification)
//!   — optional per-frame [`ContentType`] plus temporal-activity and
//!   spatial-complexity signals.  These are aggregated
//!   per shot to choose the dominant content label and movement/detail wording.
//!
//! # Gap placement and word budget
//!
//! Cues are placed using the existing dialogue-gap machinery in
//! [`crate::audio_desc::timing`].  Each cue:
//!
//! - never starts before the scene it describes appears on screen, and never
//!   before the hosting gap opens (`start = max(scene_start, gap_start)`);
//! - never runs past `gap_start + available_duration`, where `available_duration`
//!   already reserves [`TimingConstraints::min_gap_after_ms`] before the next
//!   line of dialogue — so the narration cannot collide with speech;
//! - respects a **reading-rate word budget**: at most
//!   `floor(window_ms * words_per_minute / 60000)` words fit in the usable
//!   window, capped further by [`SceneDescriptionConfig::max_words_per_cue`].
//!   The description text is truncated to that budget.
//!
//! At most one cue is assigned per gap (greedy, left-to-right), which guarantees
//! the emitted [`AudioDescriptionScript`] is time-ordered and free of overlaps.
//!
//! # Honest limitations (richer descriptors would improve this)
//!
//! The natural-language output here is **structured/templated wording derived
//! from analysis *labels*** (transition type + content class + movement/detail
//! magnitudes).  It does **not** name concrete on-screen objects, people, or
//! actions, because [`AnalysisResults`] does not currently surface those.  The
//! following richer descriptors exist in
//! `oximedia-analysis` as standalone analyzers but are **not** aggregated into
//! `AnalysisResults`; surfacing them there would let this module produce far more
//! specific prose:
//!
//! - `oximedia_analysis::saliency_map` — where the salient region is (would let
//!   us say "in the upper-left" / "centre frame").
//! - `oximedia_analysis::facial_analysis` / `object_tracking` — who/what is on
//!   screen (named subjects and objects).
//! - `oximedia_analysis::text_detection` — on-screen text to read out.
//! - `oximedia_analysis::shot_composition` / `color_analysis` — shot size and
//!   colour/mood wording.
//!
//! Until those are exposed through `AnalysisResults`, the module is deliberately
//! conservative and does not fabricate detail it cannot derive from the labels.

use std::collections::HashMap;

use oximedia_analysis::black::SilenceSegment;
use oximedia_analysis::content::{ContentClassification, ContentType};
use oximedia_analysis::scene::SceneChangeType;
use oximedia_analysis::AnalysisResults;
use oximedia_core::types::Rational;
use serde::{Deserialize, Serialize};

use crate::audio_desc::script::{
    AudioDescriptionEntry, AudioDescriptionScript, DescriptionCategory, EntryMetadata,
    ScriptMetadata,
};
use crate::audio_desc::timing::{DialogueSegment, Gap, TimingAnalyzer, TimingConstraints};
use crate::error::{AccessError, AccessResult};

/// Default audio-description narration rate in words per minute.
///
/// 160 wpm is a common target for audio-description narration; it is a little
/// faster than the 150 wpm prose-reading default used elsewhere because trained
/// describers speak briskly to fit tight gaps.
pub const DEFAULT_AD_WORDS_PER_MINUTE: f32 = 160.0;

/// Configuration controlling scene-analysis-driven description generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDescriptionConfig {
    /// Narration reading rate in words per minute (must be > 0).
    pub words_per_minute: f32,
    /// Hard upper bound on the number of words emitted for any single cue,
    /// regardless of how large the hosting gap is.
    pub max_words_per_cue: usize,
    /// Minimum cue duration in milliseconds; gaps with a smaller usable window
    /// are skipped for that trigger.
    pub min_description_ms: i64,
    /// Only scenes whose change confidence is greater than or equal to this
    /// threshold produce a cue (`0.0` accepts every detected boundary).
    pub min_scene_confidence: f64,
    /// Prefix each cue with transition wording ("cuts to", "fades to", ...).
    pub describe_change_type: bool,
    /// Append movement/detail wording derived from temporal-activity and
    /// spatial-complexity signals.
    pub describe_dynamics: bool,
    /// Emit an opening cue for the first shot (frame 0) with no transition.
    pub describe_opening: bool,
    /// BCP-47 language tag recorded in the generated script metadata.
    pub language: String,
}

impl Default for SceneDescriptionConfig {
    fn default() -> Self {
        Self {
            words_per_minute: DEFAULT_AD_WORDS_PER_MINUTE,
            max_words_per_cue: 24,
            min_description_ms: 800,
            min_scene_confidence: 0.0,
            describe_change_type: true,
            describe_dynamics: true,
            describe_opening: true,
            language: "en".to_string(),
        }
    }
}

impl SceneDescriptionConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the narration reading rate (words per minute).
    #[must_use]
    pub fn with_words_per_minute(mut self, wpm: f32) -> Self {
        self.words_per_minute = wpm;
        self
    }

    /// Set the hard per-cue word cap.
    #[must_use]
    pub const fn with_max_words_per_cue(mut self, max_words: usize) -> Self {
        self.max_words_per_cue = max_words;
        self
    }

    /// Set the minimum cue duration in milliseconds.
    #[must_use]
    pub const fn with_min_description_ms(mut self, ms: i64) -> Self {
        self.min_description_ms = ms;
        self
    }

    /// Set the minimum scene-change confidence required to emit a cue.
    #[must_use]
    pub const fn with_min_scene_confidence(mut self, confidence: f64) -> Self {
        self.min_scene_confidence = confidence;
        self
    }

    /// Set the BCP-47 language tag recorded in the script metadata.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AccessResult<()> {
        if !(self.words_per_minute.is_finite()) || self.words_per_minute <= 0.0 {
            return Err(AccessError::AudioDescriptionFailed(
                "words_per_minute must be a positive, finite value".to_string(),
            ));
        }
        if self.max_words_per_cue == 0 {
            return Err(AccessError::AudioDescriptionFailed(
                "max_words_per_cue must be at least 1".to_string(),
            ));
        }
        if self.min_description_ms <= 0 {
            return Err(AccessError::AudioDescriptionFailed(
                "min_description_ms must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// A candidate audio-description trigger derived from one analysed shot.
///
/// A trigger is produced for every shot boundary (and, optionally, for the
/// opening shot).  It carries the timing of the new shot together with the
/// aggregated content labels used to build the description text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneTrigger {
    /// Index of the source [`Scene`](oximedia_analysis::scene::Scene), or
    /// `usize::MAX` for the synthetic opening shot.
    pub scene_index: usize,
    /// Presentation time (ms) at which the new shot begins.
    pub trigger_time_ms: i64,
    /// Presentation time (ms) at which the new shot ends (next boundary or end
    /// of media).
    pub end_time_ms: i64,
    /// Transition that introduced this shot, or `None` for the opening shot.
    pub change_type: Option<SceneChangeType>,
    /// Scene-change confidence in `[0.0, 1.0]` (`1.0` for the opening shot).
    pub confidence: f64,
    /// Dominant content type aggregated over the shot, if classification data
    /// was available.
    pub content_type: Option<ContentType>,
    /// Mean temporal activity over the shot (`0.0` when unknown).
    pub avg_temporal_activity: f64,
    /// Mean spatial complexity over the shot (`0.0` when unknown).
    pub avg_spatial_complexity: f64,
    /// Full (untruncated) templated description for this shot.
    pub description: String,
}

impl SceneTrigger {
    /// Duration of the shot in milliseconds.
    #[must_use]
    pub const fn duration_ms(&self) -> i64 {
        self.end_time_ms - self.trigger_time_ms
    }
}

/// Generates audio-description cues from `oximedia-analysis` output.
#[derive(Debug, Clone)]
pub struct SceneAudioDescriber {
    config: SceneDescriptionConfig,
}

impl Default for SceneAudioDescriber {
    fn default() -> Self {
        Self::new(SceneDescriptionConfig::default())
    }
}

impl SceneAudioDescriber {
    /// Create a describer with the given configuration.
    #[must_use]
    pub fn new(config: SceneDescriptionConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    #[must_use]
    pub const fn config(&self) -> &SceneDescriptionConfig {
        &self.config
    }

    /// Full pipeline: derive triggers from `analysis`, find the dialogue gaps
    /// (including the lead-in before the first line of dialogue), and place one
    /// cue per gap.
    ///
    /// The returned [`AudioDescriptionScript`] is time-ordered, overlap-free, and
    /// can be fed directly to [`crate::audio_desc::AudioDescriptionGenerator`].
    pub fn generate_script(
        &self,
        analysis: &AnalysisResults,
        dialogue: &[DialogueSegment],
        constraints: &TimingConstraints,
    ) -> AccessResult<AudioDescriptionScript> {
        self.generate_script_with_duration(analysis, dialogue, constraints, None)
    }

    /// Like [`generate_script`](Self::generate_script) but also accepts an
    /// optional media duration in milliseconds.
    ///
    /// When `media_duration_ms` is provided, the region after the final line of
    /// dialogue (lead-out) is treated as an additional gap, and — when there is
    /// no dialogue at all — the whole `[0, media_duration_ms)` span becomes one
    /// gap.  Without a duration the lead-out cannot be bounded and is omitted.
    pub fn generate_script_with_duration(
        &self,
        analysis: &AnalysisResults,
        dialogue: &[DialogueSegment],
        constraints: &TimingConstraints,
        media_duration_ms: Option<i64>,
    ) -> AccessResult<AudioDescriptionScript> {
        self.config.validate()?;
        constraints.validate()?;
        let triggers = self.derive_triggers(analysis)?;
        let gaps = compute_gaps(dialogue, constraints, media_duration_ms);
        self.place_triggers(&triggers, &gaps, constraints)
    }

    /// Derive one [`SceneTrigger`] per shot boundary (and, when enabled, an
    /// opening trigger for the first shot).
    ///
    /// Frame numbers are converted to presentation-time milliseconds using
    /// `analysis.frame_rate`.
    pub fn derive_triggers(&self, analysis: &AnalysisResults) -> AccessResult<Vec<SceneTrigger>> {
        let fps = analysis.frame_rate;
        let frame_count = analysis.frame_count;
        let classification = analysis.content_classification.as_ref();

        // Build the ordered list of shot starts.  Each shot start carries the
        // transition that introduced it (None = opening shot) and the source
        // scene index.
        //
        // A `Scene` records a transition at `end_frame`: the segment
        // `[start_frame, end_frame)` is one shot and a new shot begins at
        // `end_frame`.  So the shot starts are `0` (opening) followed by every
        // `scene.end_frame`.
        struct ShotStart {
            scene_index: usize,
            start_frame: usize,
            change_type: Option<SceneChangeType>,
            confidence: f64,
        }

        let mut shot_starts: Vec<ShotStart> = Vec::with_capacity(analysis.scenes.len() + 1);
        if self.config.describe_opening {
            shot_starts.push(ShotStart {
                scene_index: usize::MAX,
                start_frame: 0,
                change_type: None,
                confidence: 1.0,
            });
        }
        for (idx, scene) in analysis.scenes.iter().enumerate() {
            if scene.confidence < self.config.min_scene_confidence {
                continue;
            }
            shot_starts.push(ShotStart {
                scene_index: idx,
                start_frame: scene.end_frame,
                change_type: Some(scene.change_type),
                confidence: scene.confidence,
            });
        }

        // Order by start frame so shot extents are well-formed even if the
        // analysis emitted scenes out of order.
        shot_starts.sort_by_key(|s| s.start_frame);

        let mut triggers = Vec::with_capacity(shot_starts.len());
        for i in 0..shot_starts.len() {
            let start_frame = shot_starts[i].start_frame;
            let end_frame = shot_starts
                .get(i + 1)
                .map_or(frame_count, |next| next.start_frame)
                .max(start_frame);

            // Skip zero-length shots (e.g. an opening boundary coinciding with
            // the first cut, or duplicate boundaries).
            if end_frame <= start_frame {
                continue;
            }

            let start_ms = frame_to_ms(start_frame, fps)?;
            let end_ms = frame_to_ms(end_frame, fps)?;

            let (content_type, temporal, spatial) =
                aggregate_content(classification, start_frame, end_frame);

            let description = self.compose_description(
                shot_starts[i].change_type,
                content_type,
                temporal,
                spatial,
            );

            triggers.push(SceneTrigger {
                scene_index: shot_starts[i].scene_index,
                trigger_time_ms: start_ms,
                end_time_ms: end_ms,
                change_type: shot_starts[i].change_type,
                confidence: shot_starts[i].confidence,
                content_type,
                avg_temporal_activity: temporal,
                avg_spatial_complexity: spatial,
                description,
            });
        }

        Ok(triggers)
    }

    /// Place already-derived triggers into already-computed gaps.
    ///
    /// This is the low-level placement primitive — useful when the caller wants
    /// to supply a custom set of gaps (for example a lead-out gap, or quiet
    /// regions taken from `oximedia_analysis`'s silence detection).
    ///
    /// Placement is greedy and left-to-right: triggers and gaps are each scanned
    /// in time order and **at most one cue is assigned to each gap**, which makes
    /// the result inherently overlap-free and time-ordered.  Triggers that have
    /// no remaining hosting gap are dropped.
    pub fn place_triggers(
        &self,
        triggers: &[SceneTrigger],
        gaps: &[Gap],
        constraints: &TimingConstraints,
    ) -> AccessResult<AudioDescriptionScript> {
        self.config.validate()?;

        let mut script = AudioDescriptionScript::with_metadata(ScriptMetadata {
            language: self.config.language.clone(),
            ..ScriptMetadata::default()
        });

        // Defensive ordering: callers may hand us unsorted slices.
        let mut ordered_triggers: Vec<&SceneTrigger> = triggers.iter().collect();
        ordered_triggers.sort_by_key(|t| t.trigger_time_ms);

        let mut ordered_gaps: Vec<&Gap> = gaps.iter().collect();
        ordered_gaps.sort_by_key(|g| g.start_time_ms);

        let min_ms = self
            .config
            .min_description_ms
            .max(constraints.min_description_ms);
        let wpm = f64::from(self.config.words_per_minute);

        let mut gap_cursor = 0usize;
        for trig in ordered_triggers {
            if gap_cursor >= ordered_gaps.len() {
                break;
            }
            // Advance through gaps until one can host this trigger.  A gap is
            // consumed (cursor advanced) whether it hosts the cue or is found
            // unusable for this and every later (later-starting) trigger.
            while gap_cursor < ordered_gaps.len() {
                let gap = ordered_gaps[gap_cursor];
                // Latest moment the narration may *end* inside this gap, leaving
                // `min_gap_after_ms` of clearance before the next dialogue.
                let usable_end = gap.start_time_ms + gap.available_duration_ms;
                // Never narrate before the scene appears, nor before the gap opens.
                let start = trig.trigger_time_ms.max(gap.start_time_ms);

                if usable_end <= start || usable_end - start < min_ms {
                    // Too small (or entirely before the scene): this gap cannot
                    // help this trigger; later triggers start no earlier, so it
                    // cannot help them either — drop it.
                    gap_cursor += 1;
                    continue;
                }

                let window_ms = usable_end - start;
                let budget_words = word_budget(window_ms, wpm).min(self.config.max_words_per_cue);
                if budget_words == 0 {
                    gap_cursor += 1;
                    continue;
                }

                let text = truncate_words(&trig.description, budget_words);
                if text.split_whitespace().next().is_none() {
                    gap_cursor += 1;
                    continue;
                }

                let used_words = text.split_whitespace().count().max(1);
                let speak_ms = words_to_ms(used_words, wpm).clamp(min_ms, window_ms);

                let entry = AudioDescriptionEntry::new(start, start + speak_ms, text)
                    .with_category(category_for(trig.content_type))
                    .with_priority(priority_for(trig.confidence))
                    .with_metadata(EntryMetadata {
                        location: scene_location(trig),
                        tags: scene_tags(trig),
                        ..EntryMetadata::default()
                    });
                script.add_entry(entry);

                // One cue per gap → guaranteed non-overlap. Move to next gap.
                gap_cursor += 1;
                break;
            }
        }

        Ok(script)
    }

    /// Build the templated description for a shot from its analysis labels.
    fn compose_description(
        &self,
        change: Option<SceneChangeType>,
        content: Option<ContentType>,
        temporal: f64,
        spatial: f64,
    ) -> String {
        let intro = if self.config.describe_change_type {
            match change {
                Some(SceneChangeType::Cut) => "The scene cuts to",
                Some(SceneChangeType::Fade) => "The scene fades to",
                Some(SceneChangeType::Dissolve) => "The image dissolves into",
                Some(SceneChangeType::Wipe) => "A wipe reveals",
                Some(SceneChangeType::Unknown) => "The scene changes to",
                None => "The scene opens on",
            }
        } else if change.is_none() {
            "The scene opens on"
        } else {
            "The scene shows"
        };

        let content_phrase = match content {
            Some(ContentType::Action) => "a fast-moving action sequence",
            Some(ContentType::Still) => "a calm, still shot",
            Some(ContentType::TalkingHead) => "a person speaking on camera",
            Some(ContentType::Sports) => "an energetic sports scene",
            Some(ContentType::Animation) => "an animated scene",
            Some(ContentType::Mixed) | None => "a new scene",
        };

        let mut sentence = format!("{intro} {content_phrase}");

        if self.config.describe_dynamics {
            if let Some(clause) = dynamics_clause(temporal, spatial) {
                sentence.push(' ');
                sentence.push_str(clause);
            }
        }

        sentence.push('.');
        sentence
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Convert a frame number to presentation-time milliseconds.
///
/// `fps = num / den`, so `time_ms = frame * 1000 * den / num`.  The arithmetic
/// is performed in `i128` to avoid intermediate overflow on long timelines.
fn frame_to_ms(frame: usize, fps: Rational) -> AccessResult<i64> {
    if fps.num <= 0 || fps.den <= 0 {
        return Err(AccessError::InvalidTiming(format!(
            "invalid frame rate {}/{}",
            fps.num, fps.den
        )));
    }
    let ms = i128::from(frame as u64) * 1000 * i128::from(fps.den) / i128::from(fps.num);
    Ok(ms as i64)
}

/// Maximum whole words that fit in `window_ms` at `wpm` words per minute.
fn word_budget(window_ms: i64, wpm: f64) -> usize {
    if window_ms <= 0 || wpm <= 0.0 {
        return 0;
    }
    let words = (window_ms as f64) * wpm / 60_000.0;
    if words < 1.0 {
        0
    } else {
        words.floor() as usize
    }
}

/// Milliseconds needed to narrate `words` words at `wpm` words per minute.
fn words_to_ms(words: usize, wpm: f64) -> i64 {
    if wpm <= 0.0 {
        return 0;
    }
    ((words as f64) * 60_000.0 / wpm).ceil() as i64
}

/// Truncate `text` to at most `max_words` whitespace-delimited words.
///
/// Surrounding/intervening whitespace is normalised to single spaces.  When the
/// text is truncated, a terminating period is appended if the result does not
/// already end with sentence punctuation.
fn truncate_words(text: &str, max_words: usize) -> String {
    if max_words == 0 {
        return String::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        return words.join(" ");
    }
    let mut out = words[..max_words].join(" ");
    if !out.ends_with(['.', '!', '?']) {
        out.push('.');
    }
    out
}

/// Optional movement/detail clause derived from temporal/spatial signals.
fn dynamics_clause(temporal: f64, spatial: f64) -> Option<&'static str> {
    if temporal > 0.6 {
        Some("with rapid movement")
    } else if spatial > 0.6 {
        Some("rich in visual detail")
    } else if temporal > 0.3 {
        Some("with gentle movement")
    } else {
        None
    }
}

/// Map a content type to an audio-description entry category.
fn category_for(content: Option<ContentType>) -> DescriptionCategory {
    match content {
        Some(ContentType::TalkingHead) => DescriptionCategory::Character,
        Some(ContentType::Action) | Some(ContentType::Sports) => DescriptionCategory::VisualEffect,
        _ => DescriptionCategory::Scene,
    }
}

/// Map a scene-change confidence in `[0.0, 1.0]` to a priority in `1..=10`.
fn priority_for(confidence: f64) -> u8 {
    let scaled = (confidence.clamp(0.0, 1.0) * 9.0).round() as i64 + 1;
    scaled.clamp(1, 10) as u8
}

/// Stable location tag describing the source shot for downstream inspection.
fn scene_location(trig: &SceneTrigger) -> Option<String> {
    if trig.scene_index == usize::MAX {
        Some("opening".to_string())
    } else {
        Some(format!("scene_{}", trig.scene_index))
    }
}

/// Build organisational tags for a cue from its trigger labels.
fn scene_tags(trig: &SceneTrigger) -> Vec<String> {
    let mut tags = vec!["scene-analysis".to_string()];
    if let Some(change) = trig.change_type {
        tags.push(
            match change {
                SceneChangeType::Cut => "cut",
                SceneChangeType::Fade => "fade",
                SceneChangeType::Dissolve => "dissolve",
                SceneChangeType::Wipe => "wipe",
                SceneChangeType::Unknown => "transition",
            }
            .to_string(),
        );
    } else {
        tags.push("opening".to_string());
    }
    if let Some(content) = trig.content_type {
        tags.push(
            match content {
                ContentType::Action => "action",
                ContentType::Still => "still",
                ContentType::TalkingHead => "talking-head",
                ContentType::Sports => "sports",
                ContentType::Animation => "animation",
                ContentType::Mixed => "mixed",
            }
            .to_string(),
        );
    }
    tags
}

/// Aggregate the per-frame content classification over `[start_frame, end_frame)`.
///
/// Returns the dominant (modal) [`ContentType`] and the mean temporal-activity
/// and spatial-complexity over the shot.  When no classification frames fall in
/// the range the content type is `None` and the means are `0.0`.
fn aggregate_content(
    classification: Option<&ContentClassification>,
    start_frame: usize,
    end_frame: usize,
) -> (Option<ContentType>, f64, f64) {
    let Some(classification) = classification else {
        return (None, 0.0, 0.0);
    };

    // `ContentType` is `Copy + Eq` but not `Hash`; tally via a fixed index map.
    let mut tally: HashMap<usize, usize> = HashMap::new();
    let mut temporal_sum = 0.0;
    let mut spatial_sum = 0.0;
    let mut count = 0usize;

    for frame in &classification.frame_types {
        if frame.frame >= start_frame && frame.frame < end_frame {
            *tally
                .entry(content_type_index(frame.content_type))
                .or_insert(0) += 1;
            temporal_sum += frame.temporal_activity;
            spatial_sum += frame.spatial_complexity;
            count += 1;
        }
    }

    if count == 0 {
        return (None, 0.0, 0.0);
    }

    let dominant_index = tally
        .iter()
        .max_by_key(|&(_, &n)| n)
        .map(|(&idx, _)| idx)
        .unwrap_or(content_type_index(ContentType::Mixed));

    let denom = count as f64;
    (
        Some(content_type_from_index(dominant_index)),
        temporal_sum / denom,
        spatial_sum / denom,
    )
}

/// Stable index for a content type (for tally maps).
const fn content_type_index(content: ContentType) -> usize {
    match content {
        ContentType::Action => 0,
        ContentType::Still => 1,
        ContentType::TalkingHead => 2,
        ContentType::Sports => 3,
        ContentType::Animation => 4,
        ContentType::Mixed => 5,
    }
}

/// Inverse of [`content_type_index`].
const fn content_type_from_index(index: usize) -> ContentType {
    match index {
        0 => ContentType::Action,
        1 => ContentType::Still,
        2 => ContentType::TalkingHead,
        3 => ContentType::Sports,
        4 => ContentType::Animation,
        _ => ContentType::Mixed,
    }
}

/// Compute the placeable dialogue gaps for `dialogue`.
///
/// Includes the between-dialogue gaps from [`TimingAnalyzer::find_gaps`], the
/// lead-in before the first line, and (when `media_duration_ms` is known) the
/// lead-out after the last line.  With no dialogue but a known duration the
/// whole media span becomes a single gap.  All candidates pass the same
/// `min_gap` / `min_description` checks as the between-dialogue gaps.
fn compute_gaps(
    dialogue: &[DialogueSegment],
    constraints: &TimingConstraints,
    media_duration_ms: Option<i64>,
) -> Vec<Gap> {
    let analyzer = TimingAnalyzer::new(constraints.clone());
    let mut gaps = analyzer.find_gaps(dialogue);

    match (dialogue.first(), dialogue.last()) {
        (Some(first), Some(last)) => {
            // Lead-in: media start (assumed 0) → first line of dialogue.
            if let Some(gap) = make_gap(0, first.start_time_ms, None, Some(first), constraints) {
                gaps.push(gap);
            }
            // Lead-out: last line of dialogue → media end (when known).
            if let Some(duration) = media_duration_ms {
                if let Some(gap) =
                    make_gap(last.end_time_ms, duration, Some(last), None, constraints)
                {
                    gaps.push(gap);
                }
            }
        }
        _ => {
            // No dialogue: the entire media span (when known) is one gap.
            if let Some(duration) = media_duration_ms {
                if let Some(gap) = make_gap(0, duration, None, None, constraints) {
                    gaps.push(gap);
                }
            }
        }
    }

    gaps.sort_by_key(|g| g.start_time_ms);
    gaps
}

/// Build placeable gaps from the quiet regions detected by `oximedia-analysis`'s
/// audio analyzer.
///
/// [`oximedia_analysis::audio::AudioAnalysis`] reports silence as **sample**
/// ranges ([`SilenceSegment`]); `sample_rate` (the rate the audio was analysed
/// at) is required to convert them to milliseconds.  The returned gaps satisfy
/// the same `min_gap` / `min_description` checks as dialogue gaps and can be fed
/// straight to [`SceneAudioDescriber::place_triggers`].
///
/// This is the transcript-free path: when no dialogue timeline is available,
/// detected silence is itself the speech-free region where audio description can
/// be spoken without overlapping speech.
///
/// # Errors
///
/// Returns [`AccessError::InvalidTiming`] when `sample_rate` is zero.
pub fn gaps_from_silence(
    silence: &[SilenceSegment],
    sample_rate: u32,
    constraints: &TimingConstraints,
) -> AccessResult<Vec<Gap>> {
    if sample_rate == 0 {
        return Err(AccessError::InvalidTiming(
            "sample_rate must be non-zero to convert silence samples to time".to_string(),
        ));
    }
    let rate = i128::from(sample_rate);
    let mut gaps = Vec::new();
    for segment in silence {
        let start_ms = (i128::from(segment.start_sample as u64) * 1000 / rate) as i64;
        let end_ms = (i128::from(segment.end_sample as u64) * 1000 / rate) as i64;
        if let Some(gap) = make_gap(start_ms, end_ms, None, None, constraints) {
            gaps.push(gap);
        }
    }
    gaps.sort_by_key(|g| g.start_time_ms);
    Ok(gaps)
}

/// Construct a [`Gap`] for `[start_ms, end_ms)` if it satisfies the timing
/// constraints, mirroring the math in [`TimingAnalyzer::find_gaps`].
fn make_gap(
    start_ms: i64,
    end_ms: i64,
    before: Option<&DialogueSegment>,
    after: Option<&DialogueSegment>,
    constraints: &TimingConstraints,
) -> Option<Gap> {
    let duration = end_ms - start_ms;
    if duration < constraints.min_gap_ms {
        return None;
    }
    let available = duration - constraints.min_gap_after_ms;
    if available < constraints.min_description_ms {
        return None;
    }
    Some(Gap {
        start_time_ms: start_ms,
        end_time_ms: end_ms,
        available_duration_ms: available,
        context_before: before.cloned(),
        context_after: after.cloned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_analysis::content::{ContentStats, FrameType};
    use oximedia_analysis::quality::QualityStats;
    use oximedia_analysis::scene::Scene;

    // -- synthetic AnalysisResults builders ---------------------------------

    fn scene(start_frame: usize, end_frame: usize, change: SceneChangeType, conf: f64) -> Scene {
        Scene {
            start_frame,
            end_frame,
            confidence: conf,
            change_type: change,
        }
    }

    fn frame_type(frame: usize, content: ContentType, temporal: f64, spatial: f64) -> FrameType {
        FrameType {
            frame,
            content_type: content,
            temporal_activity: temporal,
            spatial_complexity: spatial,
        }
    }

    fn classification(frames: Vec<FrameType>) -> ContentClassification {
        ContentClassification {
            primary_type: ContentType::Mixed,
            confidence: 0.5,
            frame_types: frames,
            stats: ContentStats {
                avg_temporal_activity: 0.0,
                avg_spatial_complexity: 0.0,
                high_motion_ratio: 0.0,
                static_ratio: 0.0,
            },
        }
    }

    fn results(
        scenes: Vec<Scene>,
        classification: Option<ContentClassification>,
        frame_count: usize,
        fps: Rational,
    ) -> AnalysisResults {
        AnalysisResults {
            frame_count,
            frame_rate: fps,
            scenes,
            black_frames: Vec::new(),
            quality_stats: QualityStats::default(),
            content_classification: classification,
            thumbnails: Vec::new(),
            motion_stats: None,
            color_analysis: None,
            audio_analysis: None,
            temporal_analysis: None,
        }
    }

    fn dialogue(start_ms: i64, end_ms: i64) -> DialogueSegment {
        DialogueSegment::new(start_ms, end_ms)
    }

    /// Constraints with generous windows so placement logic, not constraint
    /// rejection, drives the assertions.
    fn lenient_constraints() -> TimingConstraints {
        TimingConstraints {
            min_gap_ms: 1000,
            min_description_ms: 500,
            max_description_ms: 20000,
            min_gap_after_ms: 200,
            allow_extended: false,
        }
    }

    /// Assert that an entry lies strictly inside some gap window and overlaps no
    /// dialogue segment.
    fn assert_in_gap_not_in_speech(
        entry: &AudioDescriptionEntry,
        gaps: &[Gap],
        dialogue: &[DialogueSegment],
    ) {
        // Inside a gap usable window.
        let inside = gaps.iter().any(|g| {
            entry.start_time_ms >= g.start_time_ms
                && entry.end_time_ms <= g.start_time_ms + g.available_duration_ms
        });
        assert!(
            inside,
            "entry {}..{} not inside any gap usable window",
            entry.start_time_ms, entry.end_time_ms
        );
        // Never overlaps speech.
        for d in dialogue {
            let overlaps =
                entry.start_time_ms < d.end_time_ms && entry.end_time_ms > d.start_time_ms;
            assert!(
                !overlaps,
                "entry {}..{} overlaps dialogue {}..{}",
                entry.start_time_ms, entry.end_time_ms, d.start_time_ms, d.end_time_ms
            );
        }
    }

    // -- frame_to_ms --------------------------------------------------------

    #[test]
    fn test_frame_to_ms_25fps() {
        let fps = Rational::new(25, 1);
        assert_eq!(frame_to_ms(0, fps).expect("ok"), 0);
        assert_eq!(frame_to_ms(25, fps).expect("ok"), 1000);
        assert_eq!(frame_to_ms(50, fps).expect("ok"), 2000);
    }

    #[test]
    fn test_frame_to_ms_30fps() {
        let fps = Rational::new(30, 1);
        assert_eq!(frame_to_ms(30, fps).expect("ok"), 1000);
        assert_eq!(frame_to_ms(15, fps).expect("ok"), 500);
    }

    #[test]
    fn test_frame_to_ms_ntsc_no_overflow() {
        // 30000/1001 (NTSC) over a large frame index must not overflow.
        let fps = Rational::new(30000, 1001);
        let ms = frame_to_ms(1_000_000, fps).expect("ok");
        // 1_000_000 frames / 29.97 ≈ 33_366_700 ms.
        assert!((33_360_000..33_375_000).contains(&ms), "ms={ms}");
    }

    #[test]
    fn test_frame_to_ms_invalid_rate() {
        // A zero numerator is rejected (constructed by hand to bypass Rational::new).
        let fps = Rational { num: 0, den: 1 };
        assert!(frame_to_ms(10, fps).is_err());
    }

    // -- word budget --------------------------------------------------------

    #[test]
    fn test_word_budget_and_inverse() {
        // 3000 ms at 160 wpm → 8 words.
        assert_eq!(word_budget(3000, 160.0), 8);
        // <1 word fits → 0.
        assert_eq!(word_budget(100, 160.0), 0);
        assert_eq!(word_budget(0, 160.0), 0);
        // words_to_ms is the (ceil) inverse.
        assert_eq!(words_to_ms(8, 160.0), 3000);
        assert_eq!(words_to_ms(1, 160.0), 375);
    }

    #[test]
    fn test_truncate_words() {
        let text = "one two three four five six";
        assert_eq!(truncate_words(text, 3), "one two three.");
        // No truncation needed → normalised, no forced period.
        assert_eq!(truncate_words("a  b   c", 5), "a b c");
        assert_eq!(truncate_words(text, 0), "");
    }

    // -- trigger derivation -------------------------------------------------

    #[test]
    fn test_derive_triggers_times_and_content() {
        // Two cuts at frame 50 and 100; 25 fps → 2000 ms and 4000 ms.
        // Opening shot [0,50), shot [50,100), shot [100,150).
        let scenes = vec![
            scene(0, 50, SceneChangeType::Cut, 0.9),
            scene(50, 100, SceneChangeType::Fade, 0.8),
        ];
        let frames = vec![
            frame_type(10, ContentType::Still, 0.1, 0.1), // opening shot
            frame_type(60, ContentType::Action, 0.8, 0.7), // second shot
            frame_type(120, ContentType::TalkingHead, 0.2, 0.5), // third shot
        ];
        let analysis = results(
            scenes,
            Some(classification(frames)),
            150,
            Rational::new(25, 1),
        );

        let describer = SceneAudioDescriber::default();
        let triggers = describer.derive_triggers(&analysis).expect("triggers");

        // Opening + two boundaries = 3 shots.
        assert_eq!(triggers.len(), 3);

        assert_eq!(triggers[0].trigger_time_ms, 0);
        assert_eq!(triggers[0].change_type, None);
        assert_eq!(triggers[0].content_type, Some(ContentType::Still));

        assert_eq!(triggers[1].trigger_time_ms, 2000);
        assert_eq!(triggers[1].change_type, Some(SceneChangeType::Cut));
        assert_eq!(triggers[1].content_type, Some(ContentType::Action));
        assert!(triggers[1].description.contains("cuts to"));
        assert!(triggers[1].description.contains("action"));

        assert_eq!(triggers[2].trigger_time_ms, 4000);
        assert_eq!(triggers[2].change_type, Some(SceneChangeType::Fade));
        assert_eq!(triggers[2].content_type, Some(ContentType::TalkingHead));
        assert!(triggers[2].description.contains("fades to"));
    }

    #[test]
    fn test_min_confidence_filters_low_confidence_scenes() {
        let scenes = vec![
            scene(0, 40, SceneChangeType::Cut, 0.2),
            scene(40, 80, SceneChangeType::Cut, 0.9),
        ];
        let analysis = results(scenes, None, 120, Rational::new(20, 1));
        let describer = SceneAudioDescriber::new(
            SceneDescriptionConfig::default().with_min_scene_confidence(0.5), // disable opening so only real boundaries remain
        );
        // describe_opening still true by default; turn it off for a clean count.
        let mut cfg = describer.config().clone();
        cfg.describe_opening = false;
        let describer = SceneAudioDescriber::new(cfg);
        let triggers = describer.derive_triggers(&analysis).expect("triggers");
        // Only the high-confidence scene(40,80) survives; the new shot it
        // introduces begins at its cut point (end_frame = 80) → 4000 ms at 20 fps.
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].scene_index, 1);
        assert_eq!(
            triggers[0].trigger_time_ms,
            frame_to_ms(80, Rational::new(20, 1)).expect("ms")
        );
    }

    // -- gap placement core properties --------------------------------------

    #[test]
    fn test_cues_land_in_gaps_never_in_speech() {
        // Dialogue with clear gaps between lines.
        let dlg = vec![
            dialogue(0, 2000),
            dialogue(5000, 7000),
            dialogue(10000, 12000),
            dialogue(16000, 18000),
        ];
        let constraints = lenient_constraints();
        let gaps = compute_gaps(&dlg, &constraints, Some(22000));

        // Scenes whose cuts roughly align with the gaps.
        let scenes = vec![
            scene(0, 75, SceneChangeType::Cut, 0.9), // cut ≈ 3000 ms
            scene(75, 200, SceneChangeType::Dissolve, 0.7), // cut ≈ 8000 ms
            scene(200, 350, SceneChangeType::Wipe, 0.8), // cut ≈ 13000 ms
        ];
        let analysis = results(scenes, None, 500, Rational::new(25, 1));

        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script_with_duration(&analysis, &dlg, &constraints, Some(22000))
            .expect("script");

        assert!(!script.is_empty(), "expected at least one cue");
        for entry in script.entries() {
            assert_in_gap_not_in_speech(entry, &gaps, &dlg);
        }
    }

    #[test]
    fn test_word_budget_respected_per_gap() {
        let dlg = vec![dialogue(0, 1000), dialogue(2600, 4000)];
        // Gap [1000,2600): duration 1600, available 1400 ms at 160 wpm → 3 words.
        let constraints = TimingConstraints {
            min_gap_ms: 1000,
            min_description_ms: 300,
            max_description_ms: 20000,
            min_gap_after_ms: 200,
            allow_extended: false,
        };
        let gaps = compute_gaps(&dlg, &constraints, None);

        let scenes = vec![scene(0, 25, SceneChangeType::Cut, 1.0)]; // cut at 1000 ms
        let analysis = results(scenes, None, 100, Rational::new(25, 1));

        let cfg = SceneDescriptionConfig::default()
            .with_min_description_ms(300)
            .with_words_per_minute(160.0);
        let describer = SceneAudioDescriber::new(cfg);
        let script = describer
            .generate_script(&analysis, &dlg, &constraints)
            .expect("script");

        for entry in script.entries() {
            // Find hosting gap.
            let gap = gaps
                .iter()
                .find(|g| {
                    entry.start_time_ms >= g.start_time_ms
                        && entry.end_time_ms <= g.start_time_ms + g.available_duration_ms
                })
                .expect("entry must be inside a gap");
            let window = g_window(gap, entry.start_time_ms);
            let budget = word_budget(window, 160.0).min(24);
            let words = entry.text.split_whitespace().count();
            assert!(
                words <= budget && words <= 24,
                "cue has {words} words but budget was {budget}"
            );
            assert!(words >= 1, "cue must carry at least one word");
        }
    }

    fn g_window(gap: &Gap, start: i64) -> i64 {
        gap.start_time_ms + gap.available_duration_ms - start
    }

    #[test]
    fn test_entries_ordered_and_non_overlapping() {
        let dlg = vec![
            dialogue(0, 1500),
            dialogue(5000, 6500),
            dialogue(10000, 11500),
            dialogue(15000, 16500),
        ];
        let constraints = lenient_constraints();
        let scenes = vec![
            scene(0, 60, SceneChangeType::Cut, 0.9),
            scene(60, 180, SceneChangeType::Cut, 0.9),
            scene(180, 320, SceneChangeType::Cut, 0.9),
        ];
        let analysis = results(scenes, None, 450, Rational::new(25, 1));

        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script_with_duration(&analysis, &dlg, &constraints, Some(20000))
            .expect("script");

        let entries = script.entries();
        assert!(entries.len() >= 2);
        for pair in entries.windows(2) {
            assert!(
                pair[0].start_time_ms <= pair[1].start_time_ms,
                "entries must be time-ordered"
            );
            assert!(
                pair[0].end_time_ms <= pair[1].start_time_ms,
                "entries must not overlap"
            );
        }
        for e in entries {
            assert!(e.end_time_ms > e.start_time_ms, "duration must be positive");
        }
        // The whole script must validate as non-overlapping.
        script.validate().expect("script timing valid");
    }

    #[test]
    fn test_one_cue_per_gap() {
        // Many triggers but only two usable gaps → at most two cues.
        let dlg = vec![
            dialogue(0, 1000),
            dialogue(3000, 4000),
            dialogue(6000, 7000),
        ];
        let constraints = lenient_constraints();
        let gaps = compute_gaps(&dlg, &constraints, None);
        let usable_gaps = gaps.len();

        // Six rapid cuts.
        let mut scenes = Vec::new();
        let mut f = 0usize;
        for _ in 0..6 {
            scenes.push(scene(f, f + 20, SceneChangeType::Cut, 0.9));
            f += 20;
        }
        let analysis = results(scenes, None, 200, Rational::new(25, 1));

        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script(&analysis, &dlg, &constraints)
            .expect("script");
        assert!(
            script.len() <= usable_gaps,
            "emitted {} cues for {} gaps",
            script.len(),
            usable_gaps
        );
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn test_no_dialogue_no_duration_yields_empty_script() {
        let scenes = vec![scene(0, 50, SceneChangeType::Cut, 0.9)];
        let analysis = results(scenes, None, 100, Rational::new(25, 1));
        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script(&analysis, &[], &lenient_constraints())
            .expect("script");
        assert!(
            script.is_empty(),
            "no gaps can be bounded without dialogue or duration"
        );
    }

    #[test]
    fn test_no_dialogue_with_duration_uses_whole_media() {
        let scenes = vec![scene(0, 50, SceneChangeType::Cut, 0.9)];
        let analysis = results(scenes, None, 100, Rational::new(25, 1));
        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script_with_duration(&analysis, &[], &lenient_constraints(), Some(10000))
            .expect("script");
        assert!(!script.is_empty(), "whole-media gap should host a cue");
        // The single gap is [0, 10000); the cue must fit inside the usable window.
        for e in script.entries() {
            assert!(e.start_time_ms >= 0);
            assert!(e.end_time_ms <= 10000 - lenient_constraints().min_gap_after_ms);
        }
    }

    #[test]
    fn test_triggers_but_no_gaps_drops_all() {
        // Dialogue is wall-to-wall: no gap is large enough.
        let dlg = vec![dialogue(0, 2000), dialogue(2100, 4000)];
        let constraints = lenient_constraints(); // min_gap_ms = 1000 > 100 gap
        let scenes = vec![scene(0, 25, SceneChangeType::Cut, 0.9)];
        let analysis = results(scenes, None, 100, Rational::new(25, 1));
        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script(&analysis, &dlg, &constraints)
            .expect("script");
        assert!(script.is_empty(), "no usable gap → no cues");
    }

    #[test]
    fn test_lead_in_gap_hosts_opening_cue() {
        // Long lead-in before the first dialogue line at 8000 ms.
        let dlg = vec![dialogue(8000, 10000), dialogue(14000, 16000)];
        let constraints = lenient_constraints();
        let scenes = vec![scene(0, 100, SceneChangeType::Cut, 0.9)];
        let analysis = results(scenes, None, 300, Rational::new(25, 1));

        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script(&analysis, &dlg, &constraints)
            .expect("script");
        assert!(!script.is_empty());
        // The first cue should be the opening, placed in the lead-in [0,8000).
        let first = &script.entries()[0];
        assert!(first.start_time_ms >= 0 && first.end_time_ms <= 8000);
        assert!(first.text.contains("opens on"));
    }

    #[test]
    fn test_empty_analysis_no_scenes() {
        let analysis = results(Vec::new(), None, 0, Rational::new(25, 1));
        let describer = SceneAudioDescriber::default();
        let triggers = describer.derive_triggers(&analysis).expect("triggers");
        assert!(triggers.is_empty());
        let script = describer
            .generate_script_with_duration(
                &analysis,
                &[dialogue(0, 1000)],
                &lenient_constraints(),
                Some(5000),
            )
            .expect("script");
        assert!(script.is_empty(), "no scenes → no cues");
    }

    // -- round-trip into the existing generator -----------------------------

    #[test]
    fn test_output_feeds_existing_generator() {
        use crate::audio_desc::generator::{AudioDescriptionConfig, AudioDescriptionGenerator};

        let dlg = vec![
            dialogue(0, 1500),
            dialogue(6000, 7500),
            dialogue(12000, 13500),
        ];
        let constraints = lenient_constraints();
        let scenes = vec![
            scene(0, 75, SceneChangeType::Cut, 0.9),
            scene(75, 200, SceneChangeType::Fade, 0.8),
        ];
        let frames = vec![
            frame_type(100, ContentType::Action, 0.8, 0.7),
            frame_type(220, ContentType::Still, 0.05, 0.1),
        ];
        let analysis = results(
            scenes,
            Some(classification(frames)),
            320,
            Rational::new(25, 1),
        );

        let describer = SceneAudioDescriber::default();
        let script = describer
            .generate_script_with_duration(&analysis, &dlg, &constraints, Some(18000))
            .expect("script");
        assert!(!script.is_empty());

        // The generated script must be synthesizable by the existing generator.
        let generator = AudioDescriptionGenerator::new(AudioDescriptionConfig::default());
        let segments = generator.generate(&script).expect("synthesis");
        assert_eq!(segments.len(), script.len());
        for seg in &segments {
            assert!(!seg.metadata.text.is_empty());
        }
    }

    #[test]
    fn test_place_triggers_low_level_with_custom_gaps() {
        // Provide a single explicit gap and a single trigger.
        let gaps = vec![Gap {
            start_time_ms: 2000,
            end_time_ms: 6000,
            available_duration_ms: 3800,
            context_before: None,
            context_after: None,
        }];
        let trig = SceneTrigger {
            scene_index: 0,
            trigger_time_ms: 1500,
            end_time_ms: 8000,
            change_type: Some(SceneChangeType::Cut),
            confidence: 0.9,
            content_type: Some(ContentType::Action),
            avg_temporal_activity: 0.8,
            avg_spatial_complexity: 0.7,
            description: "The scene cuts to a fast-moving action sequence with rapid movement."
                .to_string(),
        };
        let constraints = lenient_constraints();
        let describer = SceneAudioDescriber::default();
        let script = describer
            .place_triggers(&[trig], &gaps, &constraints)
            .expect("script");
        assert_eq!(script.len(), 1);
        let e = &script.entries()[0];
        // Start clamped to gap open (2000), not the earlier trigger time (1500).
        assert_eq!(e.start_time_ms, 2000);
        assert!(e.end_time_ms <= 2000 + 3800);
    }

    #[test]
    fn test_config_validation() {
        assert!(SceneDescriptionConfig::default().validate().is_ok());
        assert!(SceneDescriptionConfig::default()
            .with_words_per_minute(0.0)
            .validate()
            .is_err());
        assert!(SceneDescriptionConfig::default()
            .with_max_words_per_cue(0)
            .validate()
            .is_err());
        assert!(SceneDescriptionConfig::default()
            .with_min_description_ms(0)
            .validate()
            .is_err());
    }

    #[test]
    fn test_gaps_from_silence_converts_samples_to_ms() {
        // At 48 kHz: 96000 samples = 2000 ms, 240000 samples = 5000 ms.
        let silence = vec![
            SilenceSegment {
                start_sample: 96_000,
                end_sample: 240_000,
                avg_level_db: -70.0,
            },
            // Too short to be usable (200 ms span).
            SilenceSegment {
                start_sample: 480_000,
                end_sample: 489_600,
                avg_level_db: -65.0,
            },
        ];
        let constraints = lenient_constraints();
        let gaps = gaps_from_silence(&silence, 48_000, &constraints).expect("gaps");
        // Only the first (3000 ms) silence is large enough.
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_time_ms, 2000);
        assert_eq!(gaps[0].end_time_ms, 5000);
        // available = duration - min_gap_after_ms = 3000 - 200.
        assert_eq!(gaps[0].available_duration_ms, 2800);
    }

    #[test]
    fn test_gaps_from_silence_zero_rate_errors() {
        let silence = vec![SilenceSegment {
            start_sample: 0,
            end_sample: 48_000,
            avg_level_db: -70.0,
        }];
        assert!(gaps_from_silence(&silence, 0, &lenient_constraints()).is_err());
    }

    #[test]
    fn test_place_into_silence_gaps_transcript_free() {
        // No dialogue transcript: place a scene cue into a detected silence gap.
        let silence = vec![SilenceSegment {
            start_sample: 96_000, // 2000 ms
            end_sample: 336_000,  // 7000 ms
            avg_level_db: -72.0,
        }];
        let constraints = lenient_constraints();
        let gaps = gaps_from_silence(&silence, 48_000, &constraints).expect("gaps");

        let scenes = vec![scene(0, 25, SceneChangeType::Cut, 0.9)]; // cut at 1000 ms
        let analysis = results(scenes, None, 100, Rational::new(25, 1));
        let describer = SceneAudioDescriber::default();
        let triggers = describer.derive_triggers(&analysis).expect("triggers");

        let script = describer
            .place_triggers(&triggers, &gaps, &constraints)
            .expect("script");
        assert!(!script.is_empty());
        for e in script.entries() {
            // Cue sits inside the silence usable window [2000, 2000+4800].
            assert!(e.start_time_ms >= 2000);
            assert!(e.end_time_ms <= 2000 + 4800);
        }
    }

    #[test]
    fn test_priority_and_category_helpers() {
        assert_eq!(priority_for(1.0), 10);
        assert_eq!(priority_for(0.0), 1);
        assert_eq!(priority_for(-5.0), 1);
        assert_eq!(
            category_for(Some(ContentType::TalkingHead)),
            DescriptionCategory::Character
        );
        assert_eq!(
            category_for(Some(ContentType::Action)),
            DescriptionCategory::VisualEffect
        );
        assert_eq!(
            category_for(Some(ContentType::Still)),
            DescriptionCategory::Scene
        );
        assert_eq!(category_for(None), DescriptionCategory::Scene);
    }
}
