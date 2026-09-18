//! Tests for `MamEventBus` — broadcast fan-out to multiple subscribers.

use oximedia_mam::event_bus::MamEvent;
use oximedia_mam::integration::MamEventBus;

// ── Fan-out: 3 subscribers, 1 publisher ──────────────────────────────────────

#[tokio::test]
async fn test_three_subscribers_receive_one_event() {
    let bus = MamEventBus::new(16);

    let mut rx1 = bus.subscribe();
    let mut rx2 = bus.subscribe();
    let mut rx3 = bus.subscribe();

    let event = MamEvent::AssetIngested {
        asset_id: "asset-123".to_string(),
        path: "/media/test.mp4".to_string(),
        size_bytes: 4096,
    };

    // Publish — since all 3 receivers are active this must succeed.
    let count = bus.publish(event.clone()).expect("publish should succeed");
    assert_eq!(count, 3, "expected 3 receivers");

    // Each subscriber should get the event.
    for rx in [&mut rx1, &mut rx2, &mut rx3] {
        let received = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
            .await
            .expect("recv should not timeout")
            .expect("recv should not error");
        match received {
            MamEvent::AssetIngested { asset_id, .. } => {
                assert_eq!(asset_id, "asset-123");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
}

// ── No receivers: publish returns error ──────────────────────────────────────

#[tokio::test]
async fn test_publish_with_no_receivers_returns_error() {
    let bus = MamEventBus::new(16);
    // No subscribers — receiver_count() == 0
    let event = MamEvent::AssetDeleted {
        asset_id: "gone".to_string(),
    };
    let result = bus.publish(event);
    assert!(
        result.is_err(),
        "publish with no receivers should return BroadcastError"
    );
}

// ── receiver_count reflects live subscriptions ────────────────────────────────

#[tokio::test]
async fn test_receiver_count() {
    let bus = MamEventBus::new(8);
    assert_eq!(bus.receiver_count(), 0);

    let _r1 = bus.subscribe();
    assert_eq!(bus.receiver_count(), 1);

    let _r2 = bus.subscribe();
    let _r3 = bus.subscribe();
    assert_eq!(bus.receiver_count(), 3);

    // Drop r2 — count decreases
    drop(_r2);
    // Give the runtime a tick to register the drop
    tokio::task::yield_now().await;
    assert_eq!(bus.receiver_count(), 2);
}

// ── Sender clone: FolderSync-style usage ─────────────────────────────────────

#[tokio::test]
async fn test_sender_clone_delivers_event() {
    let bus = MamEventBus::new(16);
    let mut rx = bus.subscribe();

    // Simulate FolderSync holding a clone of the sender
    let tx = bus.sender();

    let event = MamEvent::StorageWarning {
        used_bytes: 900,
        capacity_bytes: 1000,
        threshold_pct: 90.0,
    };
    tx.send(event).expect("send via cloned sender");

    let received = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
        .await
        .expect("no timeout")
        .expect("no recv error");

    assert!(matches!(received, MamEvent::StorageWarning { .. }));
}

// ── Multiple events in sequence ───────────────────────────────────────────────

#[tokio::test]
async fn test_multiple_events_in_order() {
    let bus = MamEventBus::new(32);
    let mut rx = bus.subscribe();

    for i in 0..5u64 {
        let event = MamEvent::AssetIngested {
            asset_id: format!("asset-{i}"),
            path: format!("/media/file{i}.mp4"),
            size_bytes: i * 1024,
        };
        bus.publish(event).expect("publish");
    }

    for i in 0..5u64 {
        let received = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
            .await
            .expect("no timeout")
            .expect("no recv error");
        match received {
            MamEvent::AssetIngested { asset_id, .. } => {
                assert_eq!(asset_id, format!("asset-{i}"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
