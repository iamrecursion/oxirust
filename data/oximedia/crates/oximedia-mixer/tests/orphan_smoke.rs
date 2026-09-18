//! Smoke tests verifying that all newly-wired orphan modules compile and
//! provide their core APIs.
//!
//! One or more tests cover every 2–3 modules wired in this session:
//!
//!   automation_engine, automation_playback, bounce, buffer_pool,
//!   channel_fold, channel_folder, channel_prealloc, clip_guard,
//!   cue_monitor, fader_group, gain_automation, gain_computer,
//!   mix_scene, offline_bounce, param_smoother, plugin, solo_modes,
//!   spectrum_analyzer, stem_mixer, surround_pan, talkback, vca_group

// ── automation_engine + automation_playback ─────────────────────────────────

#[test]
fn smoke_automation_engine_read_mode() {
    use oximedia_mixer::automation_engine::{
        AutomationBreakpoint, AutomationEngine, AutomationLaneData, AutomationMode,
    };

    let mut engine = AutomationEngine::new(AutomationMode::Read);
    let mut lane = AutomationLaneData::new("gain", 0.0, 1.0);
    lane.add_breakpoint(AutomationBreakpoint::new(0, 0.0));
    lane.add_breakpoint(AutomationBreakpoint::new(100, 1.0));
    engine.add_lane(lane);

    engine.render_block(0, 101);
    let mid = engine.value_at("gain", 50);
    assert!(mid.is_some(), "should have a rendered value at offset 50");
    assert!(
        (mid.unwrap_or(0.0) - 0.5).abs() < 0.01,
        "linear midpoint should be ~0.5, got {:?}",
        mid
    );
}

#[test]
fn smoke_automation_playback_advance() {
    use oximedia_mixer::automation_playback::{AutomationLane, AutomationPlayer};

    let lane = AutomationLane::new("vol", 0.0, 1.0)
        .add_point(0.0, 0.0)
        .add_point(2.0, 1.0);

    let mut player = AutomationPlayer::new().add_lane(lane);
    player.play();
    let updates = player.advance(1.0);
    let val = updates.get("vol").copied().unwrap_or(-1.0);
    assert!(
        (val - 0.5).abs() < 0.01,
        "linear automation at t=1s should be ~0.5, got {val}"
    );
}

// ── bounce + buffer_pool ─────────────────────────────────────────────────────

#[test]
fn smoke_bounce_empty() {
    use oximedia_mixer::bounce::OfflineBouncer;
    use oximedia_mixer::{AudioMixer, MixerConfig};

    let config = MixerConfig {
        sample_rate: 48000,
        buffer_size: 128,
        ..Default::default()
    };
    let mut mixer = AudioMixer::new(config);
    let bouncer = OfflineBouncer::new(48000, 128);
    let result = bouncer
        .bounce(&mut mixer, &[], |_| true)
        .expect("empty bounce should succeed");
    assert_eq!(result.sample_count(), 0);
    assert_eq!(result.sample_rate, 48000);
}

#[test]
fn smoke_buffer_pool_checkout_return() {
    use oximedia_mixer::buffer_pool::AudioBufferPool;

    let pool = AudioBufferPool::new(256, 4);
    assert_eq!(pool.available(), 4);
    {
        let buf = pool.checkout();
        assert_eq!(buf.len(), 256);
        assert!(
            buf.iter().all(|&s| s == 0.0),
            "checked-out buffer must be zeroed"
        );
        assert_eq!(pool.available(), 3);
    }
    // Buffer returned on drop.
    assert_eq!(pool.available(), 4);
}

// ── channel_fold + channel_folder ────────────────────────────────────────────

#[test]
fn smoke_channel_fold_stereo_to_mono() {
    use oximedia_mixer::channel_fold::{fold_channels, ChannelLayout as FoldLayout};

    let stereo = vec![0.8_f32, 0.4];
    let out = fold_channels(&stereo, &FoldLayout::Stereo, &FoldLayout::Mono)
        .expect("stereo→mono fold should succeed");
    assert_eq!(out.len(), 1);
    assert!(
        (out[0] - 0.6).abs() < 1e-5,
        "expected (0.8+0.4)/2=0.6, got {}",
        out[0]
    );
}

#[test]
fn smoke_channel_folder_mid_side_roundtrip() {
    use oximedia_mixer::channel_folder::{ChannelFolder, FoldMode};

    let encoder = ChannelFolder::new(FoldMode::StereoToMidSide);
    let decoder = ChannelFolder::new(FoldMode::MidSideToStereo);

    let left = vec![0.7_f32, -0.2];
    let right = vec![0.3_f32, 0.4];
    let (mid, side) = encoder.stereo_to_mid_side(&left, &right);
    let (l_out, r_out) = decoder.mid_side_to_stereo(&mid, &side);

    for i in 0..2 {
        assert!((l_out[i] - left[i]).abs() < 1e-5, "L roundtrip[{i}] failed");
        assert!(
            (r_out[i] - right[i]).abs() < 1e-5,
            "R roundtrip[{i}] failed"
        );
    }
}

// ── channel_prealloc + clip_guard ────────────────────────────────────────────

#[test]
fn smoke_channel_prealloc_slab() {
    use oximedia_mixer::channel_prealloc::ChannelSlab;

    let mut slab = ChannelSlab::new(8, 512);
    let slot = slab.add_channel().expect("should acquire slot");
    assert_eq!(slab.channel_count(), 1);
    if let Some(buf) = slab.buffer_mut(slot) {
        buf[0] = 0.42;
    }
    assert!(
        (slab.buffer(slot).expect("buffer should exist")[0] - 0.42).abs() < 1e-5,
        "written value should be readable"
    );
    slab.remove_channel(slot);
    assert_eq!(slab.channel_count(), 0);
}

#[test]
fn smoke_clip_guard_brickwall() {
    use oximedia_mixer::clip_guard::{ClipGuard, ClipGuardConfig};

    let config = ClipGuardConfig {
        ceiling_db: 0.0,
        look_ahead_samples: 0,
        detect_clips: true,
        ..Default::default()
    };
    let mut guard = ClipGuard::new(config, 48000);

    // A sample at 2.0 should be hard-clamped to 1.0 when look-ahead is 0.
    let out = guard.process_sample(2.0);
    assert!(out <= 1.0 + 1e-4, "should be limited to 0 dBFS, got {out}");

    let det = guard.detector().expect("detector should be enabled");
    assert!(det.has_clipped(), "clip event should have been recorded");
}

// ── cue_monitor + fader_group ────────────────────────────────────────────────

#[test]
fn smoke_cue_monitor_pfl_accumulates() {
    use oximedia_mixer::cue_monitor::{CueMode, CueMonitor, CueMonitorConfig};

    let mut cue = CueMonitor::new(CueMonitorConfig::default());
    cue.enable_cue("kick".into(), CueMode::Pfl);
    cue.enable_cue("snare".into(), CueMode::Pfl);

    let frames = 64usize;
    let srcs = vec![
        (
            "kick".into(),
            (vec![0.3_f32; frames], vec![0.3_f32; frames]),
        ),
        (
            "snare".into(),
            (vec![0.2_f32; frames], vec![0.2_f32; frames]),
        ),
    ];
    let (l, _r) = cue.process_block(&srcs, frames);
    assert!(
        (l[0] - 0.5).abs() < 1e-5,
        "0.3 + 0.2 = 0.5 expected, got {}",
        l[0]
    );
    assert!(cue.metrics().any_active);
}

#[test]
fn smoke_fader_group_absolute_link() {
    use oximedia_mixer::fader_group::{FaderGroup, FaderGroupId, LinkMode};

    let mut group = FaderGroup::new(FaderGroupId(1), "Drums".into(), LinkMode::Absolute);
    group.add_member(0);
    group.add_member(1);

    let gains = group.set_master(0.7);
    assert_eq!(gains.len(), 2, "both members should receive gains");
    for (_, g) in &gains {
        assert!(
            (*g - 0.7).abs() < f32::EPSILON,
            "absolute link: all members at 0.7, got {g}"
        );
    }
}

// ── gain_automation + gain_computer ─────────────────────────────────────────

#[test]
fn smoke_gain_automation_record_write() {
    use oximedia_mixer::gain_automation::{AutomationRecorder, GainLane, RecordMode};

    let mut lane = GainLane::new();
    let mut recorder = AutomationRecorder::new(48_000, RecordMode::Write);
    recorder.record_gain(&mut lane, 0, 0.8);
    recorder.record_gain(&mut lane, 512, 0.4);

    assert_eq!(lane.point_count(), 2, "should have 2 recorded points");
    let (_, v0) = lane.get_point(0).expect("point 0 should exist");
    assert!(
        (v0 - 0.8).abs() < 1e-6,
        "first point should be 0.8, got {v0}"
    );
}

#[test]
fn smoke_gain_computer_db_conversion() {
    use oximedia_mixer::gain_computer::{db_to_linear, linear_to_db};

    let lin = db_to_linear(0.0);
    assert!((lin - 1.0).abs() < 1e-9, "0 dB → linear 1.0, got {lin}");
    let db = linear_to_db(1.0);
    assert!(db.abs() < 1e-9, "linear 1.0 → 0.0 dB, got {db}");
    let neg = db_to_linear(-200.0); // below floor
    assert!(
        (neg - 0.0).abs() < 1e-30,
        "very negative dB → 0.0, got {neg}"
    );
}

// ── mix_scene + offline_bounce ───────────────────────────────────────────────

#[test]
fn smoke_mix_scene_library() {
    use oximedia_mixer::mix_scene::{MixScene, SceneLibrary};

    let mut lib = SceneLibrary::new();
    let scene = MixScene::new("Live Balance");
    let id = lib.store(scene);
    assert_eq!(lib.len(), 1);

    let recalled = lib.get(id).expect("stored scene should be retrievable");
    assert_eq!(recalled.name, "Live Balance");
}

#[test]
fn smoke_offline_bounce_channel_params() {
    use oximedia_mixer::offline_bounce::{BounceChannelParams, BounceConfig};

    let params = BounceChannelParams {
        gain: 0.5,
        pan: 0.0,
        muted: false,
    };
    assert!((params.gain - 0.5).abs() < f32::EPSILON);
    assert!(!params.muted);

    // BounceConfig::stereo is the primary constructor.
    let cfg = BounceConfig::stereo(48000, 48000);
    assert_eq!(cfg.sample_rate, 48000);
    assert_eq!(cfg.output_channels, 2);
}

// ── param_smoother + plugin ──────────────────────────────────────────────────

#[test]
fn smoke_param_smoother_linear_ramp() {
    use oximedia_mixer::param_smoother::LinearSmoother;

    let mut s = LinearSmoother::new(0.0, 4);
    s.set_target(1.0);
    let v1 = s.next_sample();
    let v2 = s.next_sample();
    // First sample should be 0.25, second 0.5 (ramp over 4 samples)
    assert!(
        v1 > 0.0 && v1 < 1.0,
        "first ramp sample should be in (0, 1), got {v1}"
    );
    assert!(v2 > v1, "ramp should be monotonically increasing");
}

#[test]
fn smoke_plugin_host_basic() {
    use oximedia_mixer::plugin::{
        AudioPlugin, ParameterInfo, PluginCategory, PluginDescriptor, PluginHost,
    };

    struct NoOpPlugin;

    impl AudioPlugin for NoOpPlugin {
        fn descriptor(&self) -> PluginDescriptor {
            PluginDescriptor {
                id: "test.noop".into(),
                name: "NoOp".into(),
                vendor: "Test".into(),
                version: "0.1".into(),
                category: PluginCategory::Gain,
                num_inputs: 2,
                num_outputs: 2,
                has_editor: false,
            }
        }

        fn prepare(&mut self, _sample_rate: f64, _max_block_size: usize) {}

        fn process(&mut self, _inputs: &[&[f32]], outputs: &mut [&mut [f32]], num_samples: usize) {
            for ch in outputs.iter_mut() {
                for s in ch[..num_samples].iter_mut() {
                    *s = 0.0;
                }
            }
        }

        fn reset(&mut self) {}

        fn parameter_count(&self) -> usize {
            0
        }

        fn parameter_info(&self, _idx: usize) -> Option<ParameterInfo> {
            None
        }

        fn get_parameter(&self, _id: u32) -> f32 {
            0.0
        }

        fn set_parameter(&mut self, _id: u32, _value: f32) {}

        fn latency_samples(&self) -> u32 {
            0
        }
    }

    let mut host = PluginHost::new(48000.0, 512);
    let id = host.load_plugin(Box::new(NoOpPlugin));
    assert_eq!(host.instance_count(), 1);
    assert!(host.get_instance(id).is_ok());
    assert!(host.unload_plugin(id));
    assert_eq!(host.instance_count(), 0);
}

// ── solo_modes + spectrum_analyzer ───────────────────────────────────────────

#[test]
fn smoke_solo_modes_sip_muting() {
    use oximedia_mixer::channel::ChannelId;
    use oximedia_mixer::solo_modes::{SoloMode, SoloRouter};
    use uuid::Uuid;

    let mut router = SoloRouter::new(SoloMode::Sip);
    let ch0 = ChannelId(Uuid::new_v4());
    let ch1 = ChannelId(Uuid::new_v4());

    router.solo(ch0);
    assert!(router.is_soloed(ch0), "ch0 should be soloed");
    assert!(!router.is_soloed(ch1), "ch1 should not be soloed");

    // In SIP mode non-soloed channels get the dim gain (default 0.0 → full mute).
    let gain = router.main_mix_gain(ch1);
    assert!(
        gain < f32::EPSILON,
        "non-soloed ch1 should have gain 0 in SIP, got {gain}"
    );
}

#[test]
fn smoke_spectrum_analyzer_push_and_analyze() {
    use oximedia_mixer::spectrum_analyzer::{SpectrumAnalyzer, SpectrumAnalyzerConfig};

    let config = SpectrumAnalyzerConfig::default();
    let mut analyzer = SpectrumAnalyzer::new(config).expect("default config should be valid");

    // Feed a DC signal to fill the ring buffer.
    let dc_signal = vec![0.5_f32; 2048];
    analyzer.push_slice(&dc_signal);

    let frame = analyzer.analyze();
    // DC signal should have energy concentrated in bin 0.
    assert!(!frame.bins.is_empty(), "analyzed frame should contain bins");
}

// ── stem_mixer + surround_pan ─────────────────────────────────────────────────

#[test]
fn smoke_stem_mixer_add_stem() {
    use oximedia_mixer::stem_mixer::{StemCategory, StemConfig, StemMixer};

    let mut mixer = StemMixer::new(48000);
    let cfg = StemConfig {
        name: "Dialogue".into(),
        category: StemCategory::Dialog,
        ..Default::default()
    };
    let id = mixer.add_stem(cfg).expect("add_stem should succeed");
    assert_eq!(mixer.stem_count(), 1);
    assert!(mixer.stem(id).is_ok(), "added stem should be retrievable");
}

#[test]
fn smoke_surround_pan_center_gains() {
    use oximedia_mixer::surround_pan::{SurroundLayout, SurroundPanPosition, SurroundPanner};

    let panner = SurroundPanner::new(SurroundLayout::Layout51, 48000.0);
    let pos = SurroundPanPosition::center();
    let gains = panner.compute_gains(&pos);
    // 5.1 has 6 speakers; center position should produce non-zero center (index 2).
    assert_eq!(gains.len(), 6, "5.1 should have 6 speaker gains");
    assert!(
        gains[2] > 0.0,
        "center speaker should have positive gain for center position"
    );
}

// ── talkback + vca_group ─────────────────────────────────────────────────────

#[test]
fn smoke_talkback_momentary_mode() {
    use oximedia_mixer::talkback::{TalkbackConfig, TalkbackSystem};

    let mut sys = TalkbackSystem::new(TalkbackConfig::default());
    assert!(!sys.is_active());
    sys.button_event(true);
    assert!(sys.is_active());
    sys.button_event(false);
    assert!(!sys.is_active(), "momentary: deactivates on release");
}

#[test]
fn smoke_vca_group_trim_gain() {
    use oximedia_mixer::vca_group::{VcaGroup, VcaGroupId};

    let mut group = VcaGroup::new(VcaGroupId(1), "Masters");
    group.add_channel(0).expect("add first channel");
    group.add_channel(1).expect("add second channel");
    assert_eq!(group.members().len(), 2);

    // Set +6 dB trim: effective gain at unity channel = ~2.0
    group.set_trim_db(6.0).expect("6 dB is within range");
    let gain = group.effective_gain(1.0);
    assert!(
        gain > 1.5,
        "+6 dB trim should roughly double gain, got {gain}"
    );
}
