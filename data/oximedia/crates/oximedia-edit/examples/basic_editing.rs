//! Basic timeline editing example.
//!
//! This example demonstrates the core features of the oximedia-edit crate:
//! - Creating a timeline
//! - Adding tracks and clips
//! - Performing edit operations
//! - Applying effects and transitions
//! - Rendering the timeline

use oximedia_core::Rational;
use oximedia_edit::prelude::*;

fn main() -> EditResult<()> {
    // Create a timeline with 1ms timebase and 30fps
    let mut timeline = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));

    // Add video and audio tracks
    let video_track = timeline.add_track(TrackType::Video);
    let audio_track = timeline.add_track(TrackType::Audio);

    println!("Created timeline with {} tracks", timeline.tracks.len());

    // Create video clips
    let clip1 = Clip::new(1, ClipType::Video, 0, 5000); // 0-5 seconds
    let clip2 = Clip::new(2, ClipType::Video, 5000, 3000); // 5-8 seconds

    // Add clips to timeline
    let clip1_id = timeline.add_clip(video_track, clip1)?;
    let clip2_id = timeline.add_clip(video_track, clip2)?;

    println!("Added {} clips to timeline", timeline.clip_count());

    // Create audio clip
    let audio_clip = Clip::new(3, ClipType::Audio, 0, 8000); // 0-8 seconds
    timeline.add_clip(audio_track, audio_clip)?;

    // Add markers
    timeline.markers.add_at(0, "Start".to_string());
    timeline
        .markers
        .add_at(5000, "Transition Point".to_string());
    timeline.markers.add_at(8000, "End".to_string());

    println!("Added {} markers", timeline.markers.len());

    // Add a chapter region
    timeline.regions.add_range(0, 5000, "Intro".to_string());
    timeline.regions.add_range(5000, 8000, "Main".to_string());

    // Link video and audio clips
    timeline.links.link_video_audio(clip1_id, 3);

    // Create a timeline editor
    let mut editor = TimelineEditor::new();

    // Split clip at 2.5 seconds
    timeline.set_playhead(2500);
    let new_clips = editor.split_at_playhead(&mut timeline)?;
    println!("Split created {} new clips", new_clips.len());

    // Apply effect to a clip
    if let Some(clip) = timeline.get_clip_mut(clip1_id) {
        use oximedia_edit::{Effect, EffectType, Parameter, ParameterValue};

        // Add brightness effect
        let mut effect = Effect::new(EffectType::Brightness);
        effect.set_parameter(
            "brightness".to_string(),
            Parameter::constant(ParameterValue::Float(1.2)),
        );
        clip.effects.add(effect);

        println!("Added effect to clip");
    }

    // Add transition between clips
    use oximedia_edit::TransitionPresets;

    let transition = TransitionPresets::dissolve(1, video_track, 5000, 500, clip1_id, clip2_id);
    timeline.transitions.add(transition);

    println!("Added transition");

    // Group clips
    let group_id = timeline
        .groups
        .create_group_with_clips(vec![clip1_id, clip2_id])?;
    println!("Created group {group_id}");

    // Timeline info
    println!("\nTimeline Summary:");
    println!("  Duration: {:.2}s", timeline.duration_seconds());
    println!("  Tracks: {}", timeline.tracks.len());
    println!("  Clips: {}", timeline.clip_count());
    println!("  Markers: {}", timeline.markers.len());
    println!("  Transitions: {}", timeline.transitions.len());
    println!("  Groups: {}", timeline.groups.len());

    // Set in/out points
    timeline.in_out.set_in(1000);
    timeline.in_out.set_out(7000);

    if let Some((start, end)) = timeline.in_out.range() {
        println!("\nIn/Out Range: {start}ms - {end}ms");
    }

    // Demonstrate editing operations
    println!("\nEdit Operations:");

    // Copy clip
    timeline.selection.add(clip1_id);
    editor.copy(&timeline)?;
    println!("  Copied clip to clipboard");

    // Paste clip
    timeline.set_playhead(10000);
    let pasted = editor.paste(&mut timeline)?;
    println!("  Pasted {} clips", pasted.len());

    // Speed change
    editor.set_speed(&mut timeline, clip1_id, 2.0)?;
    println!("  Changed clip speed to 2x");

    // Reverse playback
    editor.reverse_clip(&mut timeline, clip2_id)?;
    println!("  Reversed clip playback");

    // Rendering configuration
    use oximedia_edit::{RenderConfig, RenderQuality};

    let config = RenderConfig {
        render_video: true,
        render_audio: true,
        width: 1920,
        height: 1080,
        quality: RenderQuality::High,
        ..Default::default()
    };

    println!("\nRenderer configured:");
    println!("  Resolution: {}x{}", config.width, config.height);
    println!("  Quality: {:?}", config.quality);

    // Timeline state
    println!("\nFinal Timeline State:");
    for (i, track) in timeline.tracks.iter().enumerate() {
        println!(
            "  Track {}: {:?}, {} clips",
            i,
            track.track_type,
            track.clips.len()
        );
    }

    // Get clips at playhead
    let clips_at_playhead = timeline.get_clips_at(timeline.playhead);
    println!(
        "\nClips at playhead ({}ms): {}",
        timeline.playhead,
        clips_at_playhead.len()
    );

    // Find nearest marker
    if let Some(marker) = timeline.markers.find_nearest(timeline.playhead) {
        println!("Nearest marker: '{}' at {}ms", marker.name, marker.position);
    }

    Ok(())
}
