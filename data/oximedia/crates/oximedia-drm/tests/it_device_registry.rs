//! Integration tests for `oximedia_drm::device_registry`.
//!
//! Smoke-tests the public surface: device registration, deauthorization,
//! removal, per-user limits, and concurrent registration safety via
//! `std::sync::Mutex` (the registry itself is not `Send + Sync` mutable, so
//! we wrap it in a `Mutex` for the 8-thread concurrency probe — that
//! mirrors how real license servers shard a registry behind a lock).

use oximedia_drm::device_registry::{DeviceRecord, DeviceRegistry, DeviceType};
use std::sync::{Arc, Mutex};
use std::thread;

fn make_record(device_id: &str, user_id: &str, dtype: DeviceType) -> DeviceRecord {
    DeviceRecord::new(device_id, "IT-Device", dtype, user_id)
}

#[test]
fn register_then_query_returns_active_record() {
    let mut reg = DeviceRegistry::new(5);
    reg.register(make_record("dev-1", "user-A", DeviceType::Desktop))
        .expect("register should succeed");

    let record = reg.get("dev-1").expect("record exists after register");
    assert_eq!(record.device_id, "dev-1");
    assert_eq!(record.user_id, "user-A");
    assert!(
        record.is_authorized(),
        "newly-registered device is authorized"
    );
    assert!(matches!(record.device_type, DeviceType::Desktop));
}

#[test]
fn deauthorize_marks_record_inactive_but_keeps_it() {
    let mut reg = DeviceRegistry::new(5);
    reg.register(make_record("dev-2", "user-B", DeviceType::Mobile))
        .expect("register should succeed");
    assert_eq!(reg.authorized_count("user-B"), 1);

    let ok = reg.deauthorize("dev-2");
    assert!(ok, "deauthorize returns true on existing record");

    // Record still exists (history), but no longer authorized.
    let record = reg
        .get("dev-2")
        .expect("record still present after deauthorize");
    assert!(!record.is_authorized());
    assert_eq!(reg.authorized_count("user-B"), 0);
}

#[test]
fn state_transitions_register_deauthorize_remove() {
    let mut reg = DeviceRegistry::new(3);

    // 1. Register
    reg.register(make_record("dev-3", "user-C", DeviceType::Television))
        .expect("register should succeed");
    assert!(reg.get("dev-3").map(|r| r.is_authorized()).unwrap_or(false));

    // 2. Deauthorize
    assert!(reg.deauthorize("dev-3"));
    assert!(!reg.get("dev-3").map(|r| r.is_authorized()).unwrap_or(true));

    // 3. Remove
    assert!(reg.remove("dev-3"));
    assert!(reg.get("dev-3").is_none(), "after remove, record is gone");

    // 4. Re-register on removed device works
    reg.register(make_record("dev-3", "user-C", DeviceType::Television))
        .expect("re-register after remove should succeed");
    assert!(reg.get("dev-3").map(|r| r.is_authorized()).unwrap_or(false));
}

#[test]
fn deauthorize_unknown_device_returns_false() {
    let mut reg = DeviceRegistry::new(3);
    assert!(!reg.deauthorize("nope-not-here"));
    assert!(!reg.remove("nope-not-here"));
}

#[test]
fn per_user_device_limit_enforced() {
    let mut reg = DeviceRegistry::new(2);
    reg.register(make_record("d1", "limited-user", DeviceType::Desktop))
        .expect("first register ok");
    reg.register(make_record("d2", "limited-user", DeviceType::Mobile))
        .expect("second register ok");

    let third = reg.register(make_record("d3", "limited-user", DeviceType::WebBrowser));
    assert!(third.is_err(), "third register must hit the limit");

    // Deauthorize one to free a slot
    assert!(reg.deauthorize("d1"));
    let retry = reg.register(make_record("d3", "limited-user", DeviceType::WebBrowser));
    assert!(retry.is_ok(), "after deauthorize, slot is free");
}

#[test]
fn concurrent_registration_eight_threads_no_panic() {
    // 8 threads, each registering a unique device for a unique user.
    // Per-user limit is 1, so each registration is independent.
    let registry = Arc::new(Mutex::new(DeviceRegistry::new(1)));

    let mut handles = Vec::new();
    for i in 0u8..8 {
        let reg = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            let device_id = format!("dev-{}", i);
            let user_id = format!("user-{}", i);
            let record = DeviceRecord::new(
                &device_id,
                "Concurrent-Device",
                DeviceType::Desktop,
                &user_id,
            );
            let mut guard = reg.lock().expect("mutex not poisoned");
            guard
                .register(record)
                .unwrap_or_else(|e| panic!("thread {} failed: {}", i, e));
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("thread join");
    }

    let guard = registry.lock().expect("mutex not poisoned");
    assert_eq!(guard.total_count(), 8, "all 8 devices registered");
    for i in 0u8..8 {
        let user = format!("user-{}", i);
        assert_eq!(guard.authorized_count(&user), 1, "{} has 1 device", user);
    }
}
