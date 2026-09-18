//! Narrative structure for automated video assembly.
//!
//! Maps clips to narrative acts (setup, rising, climax, resolution)
//! based on content analysis and shot type matching.

#![allow(dead_code)]

/// Information about a single clip candidate.
#[derive(Debug, Clone)]
pub struct ClipInfo {
    /// Unique clip identifier.
    pub id: u64,
    /// Duration of the clip in seconds.
    pub duration_secs: f32,
    /// Shot type label (e.g., "wide", "close", "action").
    pub shot_type: String,
    /// Energy level 0.0–1.0 (motion, sound, excitement).
    pub energy: f32,
    /// Overall quality score 0.0–1.0.
    pub quality_score: f32,
}

impl ClipInfo {
    /// Create a new clip descriptor.
    pub fn new(
        id: u64,
        duration_secs: f32,
        shot_type: impl Into<String>,
        energy: f32,
        quality_score: f32,
    ) -> Self {
        Self {
            id,
            duration_secs,
            shot_type: shot_type.into(),
            energy,
            quality_score,
        }
    }
}

/// Purpose of a narrative act.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActPurpose {
    /// Introduction / context setting.
    Setup,
    /// Escalation of tension or interest.
    Rising,
    /// Peak moment.
    Climax,
    /// Wind-down and conclusion.
    Resolution,
    /// Short bridging segment between acts.
    Transition,
}

/// Hint about what shot types are preferred in an act.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotTypeHint {
    /// Wide / establishing shot.
    Wide,
    /// Medium / waist shot.
    Medium,
    /// Close-up.
    Close,
    /// B-roll / cutaway.
    Broll,
    /// Interview / talking head.
    Interview,
    /// Fast action.
    Action,
}

impl ShotTypeHint {
    /// Return 0.0–1.0 preference score for a given shot-type string.
    pub fn score_for(self, shot_type: &str) -> f32 {
        let st = shot_type.to_lowercase();
        match self {
            ShotTypeHint::Wide => {
                if st.contains("wide") || st.contains("establishing") {
                    1.0
                } else {
                    0.1
                }
            }
            ShotTypeHint::Medium => {
                if st.contains("medium") || st.contains("mid") {
                    1.0
                } else {
                    0.2
                }
            }
            ShotTypeHint::Close => {
                if st.contains("close") || st.contains("cu") {
                    1.0
                } else {
                    0.2
                }
            }
            ShotTypeHint::Broll => {
                if st.contains("broll") || st.contains("b-roll") || st.contains("cutaway") {
                    1.0
                } else {
                    0.15
                }
            }
            ShotTypeHint::Interview => {
                if st.contains("interview") || st.contains("talking") {
                    1.0
                } else {
                    0.1
                }
            }
            ShotTypeHint::Action => {
                if st.contains("action") || st.contains("fast") || st.contains("sport") {
                    1.0
                } else {
                    0.2
                }
            }
        }
    }
}

/// A single act within a narrative arc.
#[derive(Debug, Clone)]
pub struct NarrativeAct {
    /// Human-readable name (e.g., "Act 1 – Setup").
    pub name: String,
    /// The dramatic purpose of this act.
    pub purpose: ActPurpose,
    /// Relative weight controlling how much of the total duration this act gets.
    pub weight: f32,
    /// Preferred shot types for this act.
    pub shot_types: Vec<ShotTypeHint>,
}

impl NarrativeAct {
    /// Create a new narrative act.
    pub fn new(
        name: impl Into<String>,
        purpose: ActPurpose,
        weight: f32,
        shot_types: Vec<ShotTypeHint>,
    ) -> Self {
        Self {
            name: name.into(),
            purpose,
            weight,
            shot_types,
        }
    }
}

/// A complete narrative arc made up of acts.
#[derive(Debug, Clone)]
pub struct NarrativeArc {
    /// Ordered list of acts.
    pub acts: Vec<NarrativeAct>,
}

impl NarrativeArc {
    /// Create a new arc from a list of acts.
    pub fn new(acts: Vec<NarrativeAct>) -> Self {
        Self { acts }
    }

    /// Total weight across all acts (used for proportional duration calculation).
    pub fn total_weight(&self) -> f32 {
        self.acts.iter().map(|a| a.weight).sum()
    }

    /// Duration (in seconds) allocated to act at `act_idx` given the total video length.
    pub fn act_duration(&self, act_idx: usize, total_secs: u32) -> u32 {
        let total_weight = self.total_weight();
        if total_weight == 0.0 {
            return 0;
        }
        let act = match self.acts.get(act_idx) {
            Some(a) => a,
            None => return 0,
        };
        ((act.weight / total_weight) * total_secs as f32).round() as u32
    }
}

/// Template presets for different narrative styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrativeTemplate {
    /// Feature-length documentary.
    Documentary,
    /// Broadcast news segment.
    News,
    /// Sports highlight reel.
    SportHighlights,
    /// Short narrative film.
    ShortFilm,
    /// Advertisement / commercial.
    Commercial,
    /// Social-media optimised short clip.
    SocialMedia,
}

impl NarrativeTemplate {
    /// Typical target duration in seconds.
    pub fn target_duration_secs(self) -> u32 {
        match self {
            NarrativeTemplate::Documentary => 1800,
            NarrativeTemplate::News => 120,
            NarrativeTemplate::SportHighlights => 180,
            NarrativeTemplate::ShortFilm => 600,
            NarrativeTemplate::Commercial => 30,
            NarrativeTemplate::SocialMedia => 60,
        }
    }
}

/// Builds narrative arcs from templates.
pub struct NarrativeBuilder;

impl NarrativeBuilder {
    /// Construct a `NarrativeArc` from the given template.
    pub fn build_arc(template: NarrativeTemplate) -> NarrativeArc {
        match template {
            NarrativeTemplate::Documentary => NarrativeArc::new(vec![
                NarrativeAct::new(
                    "Introduction",
                    ActPurpose::Setup,
                    1.5,
                    vec![ShotTypeHint::Wide, ShotTypeHint::Interview],
                ),
                NarrativeAct::new(
                    "Development",
                    ActPurpose::Rising,
                    3.0,
                    vec![ShotTypeHint::Broll, ShotTypeHint::Interview],
                ),
                NarrativeAct::new(
                    "Climax",
                    ActPurpose::Climax,
                    2.0,
                    vec![ShotTypeHint::Close, ShotTypeHint::Action],
                ),
                NarrativeAct::new(
                    "Conclusion",
                    ActPurpose::Resolution,
                    1.0,
                    vec![ShotTypeHint::Wide, ShotTypeHint::Broll],
                ),
            ]),
            NarrativeTemplate::News => NarrativeArc::new(vec![
                NarrativeAct::new(
                    "Headline",
                    ActPurpose::Setup,
                    1.0,
                    vec![ShotTypeHint::Medium, ShotTypeHint::Interview],
                ),
                NarrativeAct::new(
                    "Report",
                    ActPurpose::Rising,
                    3.0,
                    vec![ShotTypeHint::Broll, ShotTypeHint::Interview],
                ),
                NarrativeAct::new(
                    "Summary",
                    ActPurpose::Resolution,
                    1.0,
                    vec![ShotTypeHint::Medium],
                ),
            ]),
            NarrativeTemplate::SportHighlights => NarrativeArc::new(vec![
                NarrativeAct::new("Intro", ActPurpose::Setup, 0.5, vec![ShotTypeHint::Wide]),
                NarrativeAct::new(
                    "Action",
                    ActPurpose::Rising,
                    3.0,
                    vec![ShotTypeHint::Action, ShotTypeHint::Close],
                ),
                NarrativeAct::new(
                    "Highlight",
                    ActPurpose::Climax,
                    2.0,
                    vec![ShotTypeHint::Action, ShotTypeHint::Close],
                ),
                NarrativeAct::new(
                    "Recap",
                    ActPurpose::Resolution,
                    0.5,
                    vec![ShotTypeHint::Wide],
                ),
            ]),
            NarrativeTemplate::ShortFilm => NarrativeArc::new(vec![
                NarrativeAct::new(
                    "Setup",
                    ActPurpose::Setup,
                    1.0,
                    vec![ShotTypeHint::Wide, ShotTypeHint::Medium],
                ),
                NarrativeAct::new(
                    "Confrontation",
                    ActPurpose::Rising,
                    2.0,
                    vec![ShotTypeHint::Medium, ShotTypeHint::Close],
                ),
                NarrativeAct::new(
                    "Climax",
                    ActPurpose::Climax,
                    1.5,
                    vec![ShotTypeHint::Close, ShotTypeHint::Action],
                ),
                NarrativeAct::new(
                    "Resolution",
                    ActPurpose::Resolution,
                    1.0,
                    vec![ShotTypeHint::Wide],
                ),
            ]),
            NarrativeTemplate::Commercial => NarrativeArc::new(vec![
                NarrativeAct::new(
                    "Hook",
                    ActPurpose::Setup,
                    0.5,
                    vec![ShotTypeHint::Close, ShotTypeHint::Action],
                ),
                NarrativeAct::new(
                    "Build",
                    ActPurpose::Rising,
                    1.5,
                    vec![ShotTypeHint::Medium, ShotTypeHint::Broll],
                ),
                NarrativeAct::new("Payoff", ActPurpose::Climax, 1.0, vec![ShotTypeHint::Close]),
            ]),
            NarrativeTemplate::SocialMedia => NarrativeArc::new(vec![
                NarrativeAct::new(
                    "Hook",
                    ActPurpose::Setup,
                    0.5,
                    vec![ShotTypeHint::Action, ShotTypeHint::Close],
                ),
                NarrativeAct::new(
                    "Content",
                    ActPurpose::Rising,
                    2.5,
                    vec![ShotTypeHint::Medium, ShotTypeHint::Action],
                ),
                NarrativeAct::new(
                    "CTA",
                    ActPurpose::Resolution,
                    0.5,
                    vec![ShotTypeHint::Close],
                ),
            ]),
        }
    }
}

/// Assigns clips to narrative acts based on content and shot-type matching.
pub struct SceneAssigner;

impl SceneAssigner {
    /// Assign each clip to the most appropriate act.
    ///
    /// Returns a vector of `(clip_idx, act_idx)` pairs. Uses a greedy
    /// strategy: for each clip, find the act whose shot-type hints best
    /// match the clip's `shot_type` and `energy`, then assign proportionally.
    pub fn assign(clips: &[ClipInfo], arc: &NarrativeArc) -> Vec<(usize, usize)> {
        if arc.acts.is_empty() || clips.is_empty() {
            return Vec::new();
        }

        clips
            .iter()
            .enumerate()
            .map(|(clip_idx, clip)| {
                let act_idx = arc
                    .acts
                    .iter()
                    .enumerate()
                    .map(|(act_idx, act)| {
                        // Score this act for this clip
                        let shot_score: f32 = act
                            .shot_types
                            .iter()
                            .map(|hint| hint.score_for(&clip.shot_type))
                            .sum::<f32>()
                            / act.shot_types.len().max(1) as f32;

                        // Energy preference: Climax/Rising prefer high-energy clips
                        let energy_bonus = match act.purpose {
                            ActPurpose::Climax => clip.energy,
                            ActPurpose::Rising => clip.energy * 0.5,
                            ActPurpose::Setup | ActPurpose::Resolution => (1.0 - clip.energy) * 0.3,
                            ActPurpose::Transition => 0.0,
                        };

                        let total = shot_score + energy_bonus;
                        (act_idx, total)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| idx);

                (clip_idx, act_idx)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, shot: &str, energy: f32) -> ClipInfo {
        ClipInfo::new(id, 5.0, shot, energy, 0.8)
    }

    #[test]
    fn test_template_durations() {
        assert_eq!(NarrativeTemplate::Commercial.target_duration_secs(), 30);
        assert_eq!(NarrativeTemplate::SocialMedia.target_duration_secs(), 60);
        assert_eq!(NarrativeTemplate::Documentary.target_duration_secs(), 1800);
    }

    #[test]
    fn test_build_arc_documentary() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::Documentary);
        assert_eq!(arc.acts.len(), 4);
        assert_eq!(arc.acts[0].purpose, ActPurpose::Setup);
        assert_eq!(arc.acts[2].purpose, ActPurpose::Climax);
    }

    #[test]
    fn test_build_arc_commercial() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::Commercial);
        assert_eq!(arc.acts.len(), 3);
        assert_eq!(arc.acts[2].purpose, ActPurpose::Climax);
    }

    #[test]
    fn test_total_weight() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::News);
        let w = arc.total_weight();
        assert!(w > 0.0, "Total weight must be positive");
    }

    #[test]
    fn test_act_duration_sums_approx_total() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::ShortFilm);
        let total_secs = 600u32;
        let sum: u32 = (0..arc.acts.len())
            .map(|i| arc.act_duration(i, total_secs))
            .sum();
        // Allow rounding error of ±act_count seconds
        assert!((sum as i64 - total_secs as i64).abs() <= arc.acts.len() as i64);
    }

    #[test]
    fn test_act_duration_out_of_range() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::Commercial);
        assert_eq!(arc.act_duration(99, 30), 0);
    }

    #[test]
    fn test_shot_type_hint_scoring_wide() {
        assert!((ShotTypeHint::Wide.score_for("wide") - 1.0).abs() < 1e-5);
        assert!(ShotTypeHint::Wide.score_for("action") < 0.5);
    }

    #[test]
    fn test_shot_type_hint_scoring_action() {
        assert!((ShotTypeHint::Action.score_for("action") - 1.0).abs() < 1e-5);
        assert!(ShotTypeHint::Action.score_for("interview") < 0.5);
    }

    #[test]
    fn test_scene_assigner_returns_correct_count() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::SportHighlights);
        let clips = vec![
            make_clip(1, "wide", 0.2),
            make_clip(2, "action", 0.9),
            make_clip(3, "close", 0.8),
            make_clip(4, "wide", 0.1),
        ];
        let assignments = SceneAssigner::assign(&clips, &arc);
        assert_eq!(assignments.len(), clips.len());
    }

    #[test]
    fn test_scene_assigner_empty_clips() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::SocialMedia);
        let assignments = SceneAssigner::assign(&[], &arc);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_scene_assigner_empty_arc() {
        let arc = NarrativeArc::new(vec![]);
        let clips = vec![make_clip(1, "wide", 0.5)];
        let assignments = SceneAssigner::assign(&clips, &arc);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_scene_assigner_high_energy_prefers_climax() {
        let arc = NarrativeBuilder::build_arc(NarrativeTemplate::Documentary);
        let clips = vec![make_clip(1, "action", 1.0)];
        let assignments = SceneAssigner::assign(&clips, &arc);
        // The Climax act (index 2) should be preferred for high-energy action clips
        assert!(!assignments.is_empty());
        let (_, act_idx) = assignments[0];
        let act = &arc.acts[act_idx];
        assert_eq!(act.purpose, ActPurpose::Climax);
    }
}
