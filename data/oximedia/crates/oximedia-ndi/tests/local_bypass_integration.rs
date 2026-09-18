//! Integration tests for the process-local zero-copy frame bypass path.
//!
//! These tests verify:
//! - Sender/receiver in same process can exchange frames without TCP encode/decode.
//! - Frame `Bytes` payload is shared (Arc refcount, not copied).
//! - Non-local (unregistered) paths are unaffected.
//! - Sender drop causes the bypass channel to close (unregister on Drop).
//!
//! The tests work directly with [`LocalBypassRegistry`] and the protocol types
//! rather than `NdiSender`/`NdiReceiver`, which require network ports.  This
//! keeps the tests deterministic and avoids port-binding races in CI.

use bytes::Bytes;
use oximedia_ndi::{
    local_bypass::LocalBypassRegistry,
    protocol::{NdiAudioFrame, NdiFrame, NdiVideoFrame},
    AudioFormat, VideoFormat,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn video_frame(seq: u32, data: Bytes) -> NdiFrame {
    let fmt = VideoFormat::new(320, 240, 30, 1);
    let stride = 320 * 2;
    NdiFrame::Video(NdiVideoFrame::new(seq, 1_000_000, fmt, data, stride))
}

fn audio_frame(seq: u32) -> NdiFrame {
    let fmt = AudioFormat::new(48_000, 2, 16);
    let data = Bytes::from(vec![42u8; 480 * 4]);
    NdiFrame::Audio(NdiAudioFrame::new(seq, 1_000_000, fmt, data, 480))
}

// ---------------------------------------------------------------------------
// Test 1: local bypass delivers the exact same frame bytes (no copy)
// ---------------------------------------------------------------------------

/// Verify that a frame sent via the bypass channel is received by an in-process
/// subscriber, and that the `Bytes` payload is the *exact same allocation*
/// (i.e. `.as_ptr()` matches — Arc-cloned, not memcopied).
#[tokio::test]
async fn test_local_bypass_delivers_frame() {
    let reg = LocalBypassRegistry::global();
    let source_name = "integ_test_delivers_frame";

    let tx = reg.register(source_name);
    let mut rx = reg
        .subscribe(source_name)
        .expect("subscribe must succeed after register");

    // Create a frame with known pixel data.
    let pixel_data: Bytes = Bytes::from(vec![0xABu8; 320 * 240 * 2]);
    let original_ptr = pixel_data.as_ptr();
    let frame = video_frame(1, pixel_data);

    tx.send(frame).expect("send must succeed");

    let received = rx.recv().await.expect("receiver must get frame");

    if let NdiFrame::Video(vf) = received {
        // Verify content
        assert_eq!(vf.format.width, 320);
        assert_eq!(vf.format.height, 240);
        assert_eq!(vf.header.sequence, 1);
        assert_eq!(vf.data.len(), 320 * 240 * 2);
        assert!(
            vf.data.iter().all(|&b| b == 0xAB),
            "pixel content must be preserved"
        );
        // Verify zero-copy: the data ptr is the same Arc-backed allocation.
        assert_eq!(
            vf.data.as_ptr(),
            original_ptr,
            "Bytes must share the same underlying buffer (zero-copy)"
        );
    } else {
        panic!("expected NdiFrame::Video, got something else");
    }

    reg.unregister(source_name);
}

// ---------------------------------------------------------------------------
// Test 2: no encoding occurs for local delivery
// ---------------------------------------------------------------------------

/// Verify that for local delivery the `encode`/`decode` path is NOT taken.
///
/// We confirm this by:
/// 1. Sending a frame whose pixel content we know exactly.
/// 2. Receiving it via the bypass channel.
/// 3. Asserting the bytes are *identical* (not re-serialised and re-parsed).
///
/// If encode→decode had happened the `Bytes` ptr would differ AND the
/// framing overhead (24-byte NDI header + format fields) would be stripped,
/// but the content might still match.  The ptr check is the definitive proof.
#[tokio::test]
async fn test_local_bypass_no_encoding() {
    let reg = LocalBypassRegistry::global();
    let source_name = "integ_test_no_encoding";

    let tx = reg.register(source_name);
    let mut rx = reg
        .subscribe(source_name)
        .expect("subscribe after register");

    // Encode the frame to wire bytes as a reference (to show what TCP would produce).
    let pixel_data = Bytes::from(vec![0xCDu8; 160 * 120 * 2]);
    let original_ptr = pixel_data.as_ptr();
    let fmt = VideoFormat::new(160, 120, 25, 1);
    let wire_frame = NdiVideoFrame::new(7, 2_000_000, fmt, pixel_data.clone(), 160 * 2);
    let encoded = NdiFrame::Video(wire_frame.clone())
        .encode()
        .expect("encode");

    // Send the original NdiFrame (not encoded bytes).
    tx.send(NdiFrame::Video(wire_frame)).expect("send");

    let received = rx.recv().await.expect("receive");

    if let NdiFrame::Video(vf) = received {
        // Content must match the *original* pixel data.
        assert_eq!(vf.data, pixel_data, "pixel data must be byte-identical");
        // The received Bytes must share the same allocation (zero-copy, not decoded).
        assert_eq!(
            vf.data.as_ptr(),
            original_ptr,
            "bypass must NOT re-encode/decode the payload"
        );
        // The encoded wire bytes are larger (header + format fields).
        assert!(
            encoded.len() > vf.data.len(),
            "wire encoding is larger than raw pixel data — confirms bypass skips encode/decode"
        );
    } else {
        panic!("expected Video frame");
    }

    reg.unregister(source_name);
}

// ---------------------------------------------------------------------------
// Test 3: remote (unregistered) address falls back gracefully
// ---------------------------------------------------------------------------

/// Verify that subscribing to a source that has *not* been registered returns
/// `None` and does not crash.  This is the "remote fallback unaffected" case:
/// the absence of a bypass entry is the signal to use TCP.
#[tokio::test]
async fn test_remote_fallback_unaffected() {
    let reg = LocalBypassRegistry::global();
    let unregistered_name = "integ_test_remote_source_that_does_not_exist";

    // Must not panic; simply returns None.
    let rx = reg.subscribe(unregistered_name);
    assert!(
        rx.is_none(),
        "subscribe to unregistered source must return None (TCP path is used instead)"
    );

    // Also verify that other registered sources are not perturbed.
    let other_name = "integ_test_remote_fallback_other_source";
    let _tx = reg.register(other_name);
    assert!(
        reg.subscribe(other_name).is_some(),
        "a correctly registered source must still be subscribable"
    );
    reg.unregister(other_name);
}

// ---------------------------------------------------------------------------
// Test 4: bypass channel entry is cleaned up on sender drop
// ---------------------------------------------------------------------------

/// After the bypass channel sender is dropped, the registry entry must be
/// removed, and outstanding receivers must observe channel closure.
#[tokio::test]
async fn test_local_bypass_unregister_on_drop() {
    let reg = LocalBypassRegistry::global();
    let source_name = "integ_test_unregister_on_drop";

    // Register and subscribe.
    let tx = reg.register(source_name);
    let mut rx = reg
        .subscribe(source_name)
        .expect("subscribe after register");

    assert!(
        reg.is_registered(source_name),
        "source must be registered before drop"
    );

    // Simulate NdiSender::drop: unregister then drop the sender handle.
    reg.unregister(source_name);
    drop(tx);

    assert!(
        !reg.is_registered(source_name),
        "registry entry must be removed after unregister"
    );

    // The receiver must observe channel closure.
    let result = rx.recv().await;
    assert!(
        result.is_err(),
        "receiver must see closed/error after sender drop: {:?}",
        result
    );

    // A new subscribe attempt must return None.
    assert!(
        reg.subscribe(source_name).is_none(),
        "no new subscriptions possible after unregister"
    );
}

// ---------------------------------------------------------------------------
// Test 5: multiple in-process subscribers receive the same frame
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_local_bypass_multiple_receivers_same_data() {
    let reg = LocalBypassRegistry::global();
    let source_name = "integ_test_multi_receiver";

    let tx = reg.register(source_name);
    let mut rx1 = reg.subscribe(source_name).expect("rx1");
    let mut rx2 = reg.subscribe(source_name).expect("rx2");

    let data = Bytes::from(vec![0x55u8; 64 * 64 * 2]);
    let original_ptr = data.as_ptr();
    tx.send(video_frame(42, data)).expect("send");

    let f1 = rx1.recv().await.expect("rx1 recv");
    let f2 = rx2.recv().await.expect("rx2 recv");

    let (vf1, vf2) = match (f1, f2) {
        (NdiFrame::Video(v1), NdiFrame::Video(v2)) => (v1, v2),
        _ => panic!("expected two Video frames"),
    };

    assert_eq!(vf1.header.sequence, 42);
    assert_eq!(vf2.header.sequence, 42);
    // Both share the same Arc-backed allocation.
    assert_eq!(vf1.data.as_ptr(), original_ptr);
    assert_eq!(vf2.data.as_ptr(), original_ptr);

    reg.unregister(source_name);
}

// ---------------------------------------------------------------------------
// Test 6: audio frames pass through the bypass channel correctly
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_local_bypass_audio_frame() {
    let reg = LocalBypassRegistry::global();
    let source_name = "integ_test_audio_bypass";

    let tx = reg.register(source_name);
    let mut rx = reg.subscribe(source_name).expect("subscribe");

    tx.send(audio_frame(3)).expect("send audio");

    let received = rx.recv().await.expect("receive audio");
    assert!(
        matches!(received, NdiFrame::Audio(_)),
        "bypass must relay audio frames unchanged"
    );
    if let NdiFrame::Audio(af) = received {
        assert_eq!(af.header.sequence, 3);
        assert_eq!(af.format.sample_rate, 48_000);
    }

    reg.unregister(source_name);
}
