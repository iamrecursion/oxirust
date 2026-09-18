//! Automated transition suggestion module.
//!
//! Analyses the context between consecutive shots and recommends an appropriate
//! transition style along with a confidence score.

#![allow(dead_code)]

/// The visual transition type to apply between two clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Hard cut with no transition effect.
    Cut,
    /// Cross-dissolve (fade one clip into the next).
    Dissolve,
    /// Fade to black and back.
    Fade,
    /// Wipe from one side.
    Wipe,
    /// Dip to a solid colour (often white or black).
    Dip,
    /// Zoom-based cross-cut.
    CrossZoom,
}

impl TransitionType {
    /// Default duration in frames for this transition type at 24 fps.
    #[must_use]
    pub fn default_duration_frames(&self) -> u64 {
        match self {
            Self::Cut => 0,
            Self::Dissolve => 24,
            Self::Fade => 36,
            Self::Wipe => 18,
            Self::Dip => 30,
            Self::CrossZoom => 12,
        }
    }

    /// Whether this transition is well-suited to action sequences.
    #[must_use]
    pub fn is_appropriate_for_action(&self) -> bool {
        matches!(self, Self::Cut | Self::CrossZoom | Self::Wipe)
    }
}

/// The type of camera shot in a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotType {
    /// Wide / establishing shot.
    Wide,
    /// Medium shot.
    Medium,
    /// Close-up.
    Close,
    /// Insert / detail shot.
    Insert,
}

/// Contextual information about the pair of shots surrounding a transition point.
#[derive(Debug, Clone)]
pub struct TransitionContext {
    /// Shot type of the outgoing clip.
    pub shot_a_type: ShotType,
    /// Shot type of the incoming clip.
    pub shot_b_type: ShotType,
    /// Whether a significant time jump occurs between the two clips.
    pub time_jump: bool,
    /// Whether the physical scene/location changes.
    pub scene_change: bool,
    /// Whether the emotional tone changes noticeably.
    pub mood_change: bool,
}

/// Suggest the most appropriate transition type given a context.
///
/// Returns `(TransitionType, confidence)` where confidence is in `[0.0, 1.0]`.
#[must_use]
pub fn suggest_transition(ctx: &TransitionContext) -> (TransitionType, f64) {
    // Scene or time jump: prefer a dissolve or fade to communicate the jump.
    if ctx.scene_change && ctx.time_jump {
        return (TransitionType::Fade, 0.90);
    }
    if ctx.scene_change {
        return (TransitionType::Dissolve, 0.85);
    }
    if ctx.time_jump {
        return (TransitionType::Dissolve, 0.80);
    }

    // Mood change: dip helps separate emotional beats.
    if ctx.mood_change {
        return (TransitionType::Dip, 0.75);
    }

    // Same shot type back-to-back (e.g., wide -> wide): wipe helps differentiate.
    if ctx.shot_a_type == ctx.shot_b_type {
        match ctx.shot_a_type {
            ShotType::Wide => return (TransitionType::Dissolve, 0.65),
            ShotType::Close | ShotType::Insert => return (TransitionType::Wipe, 0.60),
            ShotType::Medium => {}
        }
    }

    // Close to insert or vice versa: quick cross-zoom.
    if matches!(
        (ctx.shot_a_type, ctx.shot_b_type),
        (ShotType::Close, ShotType::Insert) | (ShotType::Insert, ShotType::Close)
    ) {
        return (TransitionType::CrossZoom, 0.70);
    }

    // Default: hard cut is almost always safe.
    (TransitionType::Cut, 0.55)
}

/// A complete transition plan for a sequence of clips.
#[derive(Debug, Clone)]
pub struct TransitionPlan {
    /// Ordered list of clip identifiers.
    pub clips: Vec<u64>,
    /// Transition entries: `(after_clip_index, transition_type, duration_frames)`.
    pub transitions: Vec<(usize, TransitionType, u64)>,
}

impl TransitionPlan {
    /// Create an empty transition plan.
    #[must_use]
    pub fn new() -> Self {
        Self {
            clips: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Append a clip id to the plan.
    pub fn add_clip(&mut self, clip_id: u64) {
        self.clips.push(clip_id);
    }

    /// Manually set the transition that follows clip at `after_clip` index.
    pub fn set_transition(&mut self, after_clip: usize, ttype: TransitionType, duration: u64) {
        // Replace existing entry for this index or insert new.
        if let Some(entry) = self.transitions.iter_mut().find(|e| e.0 == after_clip) {
            *entry = (after_clip, ttype, duration);
        } else {
            self.transitions.push((after_clip, ttype, duration));
        }
    }

    /// Automatically suggest transitions for each consecutive clip pair using contexts.
    ///
    /// `contexts` must have exactly `clips.len() - 1` entries (one per adjacent pair).
    pub fn auto_suggest(&mut self, contexts: &[TransitionContext]) {
        for (i, ctx) in contexts.iter().enumerate() {
            if i + 1 >= self.clips.len() {
                break;
            }
            let (ttype, _confidence) = suggest_transition(ctx);
            let duration = ttype.default_duration_frames();
            self.set_transition(i, ttype, duration);
        }
    }
}

impl Default for TransitionPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_duration_is_zero() {
        assert_eq!(TransitionType::Cut.default_duration_frames(), 0);
    }

    #[test]
    fn test_dissolve_duration_nonzero() {
        assert!(TransitionType::Dissolve.default_duration_frames() > 0);
    }

    #[test]
    fn test_action_appropriate_transitions() {
        assert!(TransitionType::Cut.is_appropriate_for_action());
        assert!(TransitionType::CrossZoom.is_appropriate_for_action());
        assert!(TransitionType::Wipe.is_appropriate_for_action());
        assert!(!TransitionType::Dissolve.is_appropriate_for_action());
        assert!(!TransitionType::Fade.is_appropriate_for_action());
    }

    #[test]
    fn test_suggest_fade_for_scene_and_time_jump() {
        let ctx = TransitionContext {
            shot_a_type: ShotType::Wide,
            shot_b_type: ShotType::Wide,
            time_jump: true,
            scene_change: true,
            mood_change: false,
        };
        let (ttype, conf) = suggest_transition(&ctx);
        assert_eq!(ttype, TransitionType::Fade);
        assert!(conf > 0.8);
    }

    #[test]
    fn test_suggest_dissolve_for_scene_change_only() {
        let ctx = TransitionContext {
            shot_a_type: ShotType::Medium,
            shot_b_type: ShotType::Wide,
            time_jump: false,
            scene_change: true,
            mood_change: false,
        };
        let (ttype, _) = suggest_transition(&ctx);
        assert_eq!(ttype, TransitionType::Dissolve);
    }

    #[test]
    fn test_suggest_dip_for_mood_change() {
        let ctx = TransitionContext {
            shot_a_type: ShotType::Close,
            shot_b_type: ShotType::Wide,
            time_jump: false,
            scene_change: false,
            mood_change: true,
        };
        let (ttype, _) = suggest_transition(&ctx);
        assert_eq!(ttype, TransitionType::Dip);
    }

    #[test]
    fn test_suggest_cut_as_default() {
        let ctx = TransitionContext {
            shot_a_type: ShotType::Wide,
            shot_b_type: ShotType::Close,
            time_jump: false,
            scene_change: false,
            mood_change: false,
        };
        let (ttype, _) = suggest_transition(&ctx);
        assert_eq!(ttype, TransitionType::Cut);
    }

    #[test]
    fn test_suggest_cross_zoom_close_to_insert() {
        let ctx = TransitionContext {
            shot_a_type: ShotType::Close,
            shot_b_type: ShotType::Insert,
            time_jump: false,
            scene_change: false,
            mood_change: false,
        };
        let (ttype, _) = suggest_transition(&ctx);
        assert_eq!(ttype, TransitionType::CrossZoom);
    }

    #[test]
    fn test_transition_plan_add_clip() {
        let mut plan = TransitionPlan::new();
        plan.add_clip(1);
        plan.add_clip(2);
        assert_eq!(plan.clips.len(), 2);
    }

    #[test]
    fn test_transition_plan_set_transition() {
        let mut plan = TransitionPlan::new();
        plan.add_clip(1);
        plan.add_clip(2);
        plan.set_transition(0, TransitionType::Dissolve, 24);
        assert_eq!(plan.transitions.len(), 1);
        assert_eq!(plan.transitions[0].1, TransitionType::Dissolve);
    }

    #[test]
    fn test_transition_plan_set_transition_overwrite() {
        let mut plan = TransitionPlan::new();
        plan.add_clip(1);
        plan.add_clip(2);
        plan.set_transition(0, TransitionType::Dissolve, 24);
        plan.set_transition(0, TransitionType::Fade, 36);
        assert_eq!(plan.transitions.len(), 1);
        assert_eq!(plan.transitions[0].1, TransitionType::Fade);
    }

    #[test]
    fn test_transition_plan_auto_suggest() {
        let mut plan = TransitionPlan::new();
        plan.add_clip(10);
        plan.add_clip(20);
        plan.add_clip(30);

        let contexts = vec![
            TransitionContext {
                shot_a_type: ShotType::Wide,
                shot_b_type: ShotType::Wide,
                time_jump: false,
                scene_change: true,
                mood_change: false,
            },
            TransitionContext {
                shot_a_type: ShotType::Wide,
                shot_b_type: ShotType::Close,
                time_jump: false,
                scene_change: false,
                mood_change: false,
            },
        ];

        plan.auto_suggest(&contexts);
        assert_eq!(plan.transitions.len(), 2);
    }

    #[test]
    fn test_transition_plan_default() {
        let plan = TransitionPlan::default();
        assert!(plan.clips.is_empty());
        assert!(plan.transitions.is_empty());
    }
}
