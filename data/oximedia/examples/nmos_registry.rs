//! NMOS IS-04/IS-05/IS-07 data model demonstration.
//! Registers a broadcast node, discovers compatible receivers, manages IS-05
//! connections, tally state, and IS-07 event bus.
//! Run: `cargo run --example nmos_registry --features routing -p oximedia`

use oximedia::routing::nmos::{
    Is07EventBus, NmosConnectionManager, NmosDevice, NmosDeviceType, NmosFlow, NmosFormat,
    NmosNode, NmosReceiver, NmosRegistry, NmosSender, NmosSource, NmosTransport, TallyController,
    TallyState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia NMOS IS-04/IS-05/IS-08 Registry Demo");
    println!("===============================================\n");

    // ── IS-04: Build the resource tree ───────────────────────────────────────

    let mut registry = NmosRegistry::new();

    registry.add_node(NmosNode::new("node-studio-a", "Studio A Encoder"));
    let device = NmosDevice::new(
        "dev-encoder-1",
        "node-studio-a",
        "H.265 Encoder Unit",
        NmosDeviceType::Pipeline,
    );
    registry.add_device(device);
    let source = NmosSource::new(
        "src-video-1",
        "dev-encoder-1",
        "Studio A Video (25fps)",
        NmosFormat::Video,
        "clk-ptp-0",
    );
    registry.add_source(source);
    let flow = NmosFlow::new(
        "flow-video-1",
        "src-video-1",
        "Studio A 25p Flow",
        NmosFormat::Video,
        (25, 1),
    );
    registry.add_flow(flow);
    let sender = NmosSender::new(
        "sender-a-video",
        "flow-video-1",
        "Studio A RTP Sender",
        NmosTransport::RtpMulticast,
    );
    registry.add_sender(sender);
    let rx_video = NmosReceiver::new(
        "rx-monitor-1",
        "dev-encoder-1",
        "Confidence Monitor",
        NmosFormat::Video,
    );
    let rx_audio = NmosReceiver::new(
        "rx-audio-in",
        "dev-encoder-1",
        "Embedded Audio Input",
        NmosFormat::Audio,
    );
    registry.add_receiver(rx_video);
    registry.add_receiver(rx_audio);

    println!("IS-04 Registry populated:");
    println!("  Nodes   : {}", registry.node_count());
    println!("  Senders : {}", registry.sender_count());
    println!("  Receivers: {}", registry.receiver_count());

    // ── IS-04: Format-compatible receiver discovery ───────────────────────────

    let compatible = registry.find_compatible_receivers("sender-a-video");
    println!("\nCompatible receivers for 'sender-a-video' (Video format):");
    for rx in &compatible {
        println!("  - {} [{}]", rx.label, rx.id);
    }
    println!(
        "  (rx-audio-in excluded — format mismatch: {:?})",
        NmosFormat::Audio
    );

    // ── IS-05: Connection management ─────────────────────────────────────────

    let mut conn_mgr = NmosConnectionManager::new();
    conn_mgr.connect("sender-a-video", "rx-monitor-1");

    let active = conn_mgr.active_connections();
    println!("\nIS-05 Active connections ({}):", active.len());
    for c in &active {
        println!("  {} → {}", c.sender_id, c.receiver_id);
    }

    // ── IS-07: Tally control ──────────────────────────────────────────────────

    let mut tally = TallyController::new();
    tally.set_tally("sender-a-video", TallyState::Program);
    tally.set_tally("sender-b-preview", TallyState::Preview);

    println!("\nIS-07 Tally — On-air sources:");
    for src in tally.on_air_sources() {
        println!("  [PGM] {src}");
    }
    for src in tally.preview_sources() {
        println!("  [PRV] {src}");
    }

    // ── IS-07: Event bus — state changes from production switcher ────────────

    let mut bus = Is07EventBus::new();
    bus.emit_boolean("tally-cam1-program", true);
    bus.emit_number("gain-mic1", 0.75);
    bus.emit_string("source-label", "Studio A");

    println!(
        "\nIS-07 Event Bus ({} pending events):",
        bus.pending_count()
    );
    let events = bus.drain();
    for ev in &events {
        println!(
            "  seq={} src='{}' payload={:?}",
            ev.sequence, ev.source_id, ev.payload
        );
    }
    println!("  (sequence counter now at {})", bus.current_sequence());

    // ── Registry introspection ───────────────────────────────────────────────

    let all_senders = registry.all_senders();
    let all_receivers = registry.all_receivers();
    println!("\nRegistry snapshot:");
    for s in &all_senders {
        let flow = registry.get_flow(&s.flow_id);
        let fmt = flow.map(|f| f.format).unwrap_or(NmosFormat::Data);
        println!(
            "  Sender  '{}' transport={:?} format={:?}",
            s.label, s.transport, fmt
        );
    }
    for r in &all_receivers {
        println!("  Receiver '{}' format={:?}", r.label, r.format);
    }

    println!("\nNMOS demo complete.");
    Ok(())
}
