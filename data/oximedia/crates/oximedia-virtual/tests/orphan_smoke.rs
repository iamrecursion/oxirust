//! Smoke tests for orphan modules wired in 0.1.8 Wave 6 Slice E.
//!
//! Verifies that newly registered modules compile and expose the expected
//! public API surface.  Tests are intentionally lightweight — full
//! functionality is covered by in-module unit tests.

// ============================================================================
// remote_session
// ============================================================================

#[test]
fn remote_session_basic_workflow() {
    use oximedia_virtual::remote_session::{
        OperatorRole, RemoteCommand, RemoteSessionConfig, RemoteSessionServer,
    };

    let config = RemoteSessionConfig {
        max_operators: 4,
        ..RemoteSessionConfig::default()
    };
    let mut server = RemoteSessionServer::new(config);

    server
        .register_operator("director", OperatorRole::Director)
        .expect("register director");
    server
        .register_operator("viewer", OperatorRole::Observer)
        .expect("register viewer");

    assert_eq!(server.operator_count(), 2);

    // Director can submit commands
    server
        .submit_command("director", RemoteCommand::Ping { sequence: 1 })
        .expect("submit ping");

    let n = server.process_commands();
    assert_eq!(n, 1);

    let resp = server.response_log().back().expect("response present");
    assert!(resp.success, "ping should succeed");

    // Observer cannot submit commands
    let err = server.submit_command("viewer", RemoteCommand::Ping { sequence: 2 });
    assert!(err.is_err(), "observer should be rejected");
}

#[test]
fn remote_session_encode_decode_roundtrip() {
    use oximedia_virtual::remote_session::{decode_message, encode_message, RemoteCommand};

    let cmd = RemoteCommand::SetLedBrightness { brightness: 0.5 };
    let bytes = encode_message(&cmd).expect("encode");
    let back: RemoteCommand = decode_message(&bytes).expect("decode");

    // Verify the round-trip preserved the value
    match back {
        RemoteCommand::SetLedBrightness { brightness } => {
            assert!((brightness - 0.5).abs() < 1e-6);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

// ============================================================================
// dmx_scene
// ============================================================================

#[test]
fn dmx_scene_blackout_and_full_on() {
    use oximedia_virtual::dmx_scene::{DmxSnapshot, DMX_CHANNELS};

    let blackout = DmxSnapshot::blackout(1);
    assert_eq!(blackout.channels.len(), DMX_CHANNELS);
    assert!(blackout.channels.iter().all(|&v| v == 0));

    let full = DmxSnapshot::full_on(1);
    assert!(full.channels.iter().all(|&v| v == 255));
}

#[test]
fn dmx_scene_crossfade() {
    use oximedia_virtual::dmx_scene::{CrossfadeState, DmxSnapshot};
    use std::time::Duration;

    let from = DmxSnapshot::blackout(0);
    let to = DmxSnapshot::full_on(0);

    let mut fader = CrossfadeState::new(from, to, Duration::from_millis(100));
    assert!(!fader.complete);

    // Advance to end
    let at_end = fader.advance(Duration::from_millis(100));
    assert!(fader.complete);
    assert!(at_end.channels.iter().all(|&v| v >= 250));
}

// ============================================================================
// hdri_capture
// ============================================================================

#[test]
fn hdri_capture_hdr_image_reinhard() {
    use oximedia_virtual::hdri_capture::HdrImage;

    // Create a 4x4 all-white float HDR image
    let pixels = vec![1.0f32; 4 * 4 * 3];
    let hdr = HdrImage::new(pixels, 4, 4).expect("create hdr image");
    assert_eq!(hdr.width, 4);
    assert_eq!(hdr.height, 4);

    let srgb = hdr.to_srgb8_reinhard();
    assert_eq!(srgb.len(), 4 * 4 * 3);
    // All pixels should be non-zero after tone mapping white
    assert!(srgb.iter().any(|&v| v > 0));
}

// ============================================================================
// stage_visualization
// ============================================================================

#[test]
fn stage_visualization_render_topdown() {
    use oximedia_virtual::stage_visualization::{
        ProjectionMode, StageVisualization, StageVisualizationConfig, Vertex3, WireMesh,
    };

    let config = StageVisualizationConfig {
        width: 64,
        height: 64,
        projection: ProjectionMode::TopDown,
        ..StageVisualizationConfig::default()
    };

    let mut vis = StageVisualization::new(config).expect("create visualizer");
    assert_eq!(vis.mesh_count(), 0);

    let mesh = WireMesh::box_wireframe(
        "led_wall",
        [255, 128, 0, 255],
        Vertex3::new(0.0, 0.0, 0.0),
        4.0,
        2.0,
        0.1,
    );
    vis.add_mesh(mesh);
    assert_eq!(vis.mesh_count(), 1);

    let frame = vis.render().expect("render");
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.pixels.len(), 64 * 64 * 3);
}

#[test]
fn stage_visualization_invalid_resolution_rejected() {
    use oximedia_virtual::stage_visualization::{StageVisualization, StageVisualizationConfig};

    let config = StageVisualizationConfig {
        width: 0,
        height: 0,
        ..StageVisualizationConfig::default()
    };
    assert!(StageVisualization::new(config).is_err());
}
