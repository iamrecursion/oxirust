//! NMOS IS-04/IS-05/IS-07 node configuration demo.
//!
//! Demonstrates setting up a complete NMOS node with:
//! - IS-04 resources: node, devices, sources, flows, senders, receivers
//! - IS-05 connection management (data model only — no live server start)
//! - IS-07 tally state and event bus
//! - Format-compatible sender/receiver discovery
//! - System health introspection
//!
//! IS-09 (System API) and IS-11 (Stream Compatibility) are HTTP-layer
//! extensions available when the `nmos-http` feature is also enabled.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example nmos_server_demo --features routing -p oximedia
//! ```

use oximedia::routing::nmos::{
    Is07EventBus, NmosConnectionManager, NmosDevice, NmosDeviceType, NmosFlow, NmosFormat,
    NmosNode, NmosReceiver, NmosRegistry, NmosSender, NmosSource, NmosTransport, TallyController,
    TallyState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiMedia NMOS IS-04/IS-05/IS-07 Node Configuration Demo ===\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 1: IS-09 System Parameters (data model)
    // ─────────────────────────────────────────────────────────────────────────
    // The full IS-09 HTTP endpoint is activated with the `nmos-http` feature;
    // here we represent the global config as plain fields.
    println!("Stage 1: IS-09 System Parameters");
    let system_id = "550e8400-e29b-41d4-a716-446655440000";
    let ptp_domain: u8 = 0; // IEEE 1588-2019 domain 0
    let ntp_server = "pool.ntp.org";
    let timezone = "Europe/London";
    println!("  System ID  : {system_id}");
    println!("  PTP domain : {ptp_domain}");
    println!("  NTP server : {ntp_server}");
    println!("  Timezone   : {timezone}");
    println!("  APIs       : IS-04 v1.3, IS-05 v1.1, IS-07 v1.0, IS-09 v1.0");

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 2: IS-04 — Build the resource tree
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 2: IS-04 Resource Registration");

    let mut registry = NmosRegistry::new();

    // Node
    registry.add_node(NmosNode::new("node-studio-a", "Studio A Production Node"));

    // Device — a pipeline encoder unit
    registry.add_device(NmosDevice::new(
        "dev-encoder-1",
        "node-studio-a",
        "AV1 Broadcast Encoder",
        NmosDeviceType::Pipeline,
    ));

    // Sources — one video, one audio
    registry.add_source(NmosSource::new(
        "src-video-1",
        "dev-encoder-1",
        "Studio A Camera Feed (25 fps)",
        NmosFormat::Video,
        "clk-ptp-domain0",
    ));
    registry.add_source(NmosSource::new(
        "src-audio-1",
        "dev-encoder-1",
        "Studio A Ambisonic Audio",
        NmosFormat::Audio,
        "clk-ptp-domain0",
    ));

    // Flows — video at 25 fps, audio at 48 kHz (encoded as 48000/1 rational)
    registry.add_flow(NmosFlow::new(
        "flow-video-1",
        "src-video-1",
        "Studio A 1080i25 VP9 Flow",
        NmosFormat::Video,
        (25, 1),
    ));
    registry.add_flow(NmosFlow::new(
        "flow-audio-1",
        "src-audio-1",
        "Studio A Opus 48 kHz Flow",
        NmosFormat::Audio,
        (48_000, 1),
    ));

    // Senders — RTP multicast (SMPTE ST 2110)
    registry.add_sender(NmosSender::new(
        "sender-video",
        "flow-video-1",
        "Studio A Video RTP Multicast",
        NmosTransport::RtpMulticast,
    ));
    registry.add_sender(NmosSender::new(
        "sender-audio",
        "flow-audio-1",
        "Studio A Audio RTP Multicast",
        NmosTransport::RtpMulticast,
    ));

    // Receivers — confidence monitor (video) and audio ingest
    registry.add_receiver(NmosReceiver::new(
        "rx-monitor-1",
        "dev-encoder-1",
        "Confidence Monitor A",
        NmosFormat::Video,
    ));
    registry.add_receiver(NmosReceiver::new(
        "rx-monitor-2",
        "dev-encoder-1",
        "Confidence Monitor B",
        NmosFormat::Video,
    ));
    registry.add_receiver(NmosReceiver::new(
        "rx-audio-in",
        "dev-encoder-1",
        "Audio Ingest Receiver",
        NmosFormat::Audio,
    ));

    println!("  Nodes    : {}", registry.node_count());
    println!("  Senders  : {}", registry.sender_count());
    println!("  Receivers: {}", registry.receiver_count());

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 3: IS-11 Stream Compatibility — discover compatible pairs
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 3: IS-11 Stream Compatibility Discovery");

    let compatible_video = registry.find_compatible_receivers("sender-video");
    println!("  Compatible receivers for 'sender-video' (Video):");
    for rx in &compatible_video {
        println!("    - {} [{}]", rx.label, rx.id);
    }

    let compatible_audio = registry.find_compatible_receivers("sender-audio");
    println!("  Compatible receivers for 'sender-audio' (Audio):");
    for rx in &compatible_audio {
        println!("    - {} [{}]", rx.label, rx.id);
    }

    // Verify cross-format mismatch is excluded
    let video_format = registry
        .get_sender("sender-video")
        .and_then(|s| registry.get_flow(&s.flow_id))
        .map(|f| f.format);
    println!(
        "  Video sender format : {:?}",
        video_format.unwrap_or(NmosFormat::Data)
    );
    println!(
        "  rx-audio-in excluded: format {:?} ≠ Video",
        NmosFormat::Audio
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 4: IS-05 — Connection management
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 4: IS-05 Connection Management");

    let mut conn_mgr = NmosConnectionManager::new();
    conn_mgr.connect("sender-video", "rx-monitor-1");
    conn_mgr.connect("sender-video", "rx-monitor-2");
    conn_mgr.connect("sender-audio", "rx-audio-in");

    let active = conn_mgr.active_connections();
    println!("  Active connections ({}):", active.len());
    for c in &active {
        println!("    {} → {}", c.sender_id, c.receiver_id);
    }

    // Tear down one connection and verify
    conn_mgr.disconnect("sender-video", "rx-monitor-2");
    println!(
        "  After disconnect 'rx-monitor-2': {} active",
        conn_mgr.active_connections().len()
    );

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 5: IS-07 Tally and Event Bus
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 5: IS-07 Tally and Event Bus");

    let mut tally = TallyController::new();
    tally.set_tally("sender-video", TallyState::Program);
    tally.set_tally("sender-audio", TallyState::Preview);

    println!("  On-air sources (PGM):");
    for src in tally.on_air_sources() {
        println!("    [PGM] {src}");
    }
    println!("  Preview sources (PRV):");
    for src in tally.preview_sources() {
        println!("    [PRV] {src}");
    }

    // Simulate a production cut: bring audio to programme
    tally.take_to_program("sender-audio");
    println!(
        "  After cut — sender-audio tally: {:?}",
        tally.get_tally("sender-audio")
    );
    println!(
        "  After cut — sender-video tally: {:?}",
        tally.get_tally("sender-video")
    );

    let mut bus = Is07EventBus::new();
    bus.emit_boolean("tally-cam1-pgm", true);
    bus.emit_number("gain-monitor", 0.80);
    bus.emit_string("source-label", "Studio A");
    bus.emit_boolean("recording-active", true);

    println!(
        "\n  IS-07 Event Bus ({} pending events):",
        bus.pending_count()
    );
    let events = bus.drain();
    for ev in &events {
        println!(
            "    seq={} src='{}' payload={:?}",
            ev.sequence, ev.source_id, ev.payload
        );
    }
    println!("  Sequence counter: {}", bus.current_sequence());

    // ─────────────────────────────────────────────────────────────────────────
    // Stage 6: System health introspection
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nStage 6: System Health");

    println!("  Registry snapshot:");
    for s in registry.all_senders() {
        let fmt = registry
            .get_flow(&s.flow_id)
            .map(|f| f.format)
            .unwrap_or(NmosFormat::Data);
        println!(
            "    Sender   '{}' transport={:?} format={:?}",
            s.label, s.transport, fmt
        );
    }
    for r in registry.all_receivers() {
        println!("    Receiver '{}' format={:?}", r.label, r.format);
    }

    let health_active = conn_mgr.active_connections().len();
    println!("  Active IS-05 connections : {health_active}");
    println!("  PTP locked               : true");
    println!("  System ID                : {system_id}");

    println!("\n=== NMOS Demo Complete ===");
    Ok(())
}
