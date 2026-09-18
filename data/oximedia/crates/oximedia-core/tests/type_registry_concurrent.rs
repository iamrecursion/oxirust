//! Concurrent-access tests for `TypeRegistry`.
//!
//! `TypeRegistry` itself uses a plain `HashMap`, so thread-safe access
//! requires an external lock (`RwLock<TypeRegistry>`).  These tests verify
//! that registration and lookup are logically correct under concurrent access
//! patterns using `Arc<RwLock<TypeRegistry>>`.

use std::sync::{Arc, Barrier, RwLock};
use std::thread;

use oximedia_core::type_registry::{TypeInfo, TypeKind, TypeRegistry};

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent writers: register non-overlapping names
// ─────────────────────────────────────────────────────────────────────────────

/// Multiple threads each register a unique type name.
/// After all threads complete, every name must be present.
#[test]
fn test_concurrent_writers_distinct_names() {
    const NUM_WRITERS: usize = 8;
    let registry = Arc::new(RwLock::new(TypeRegistry::new()));
    let barrier = Arc::new(Barrier::new(NUM_WRITERS));
    let mut handles = Vec::with_capacity(NUM_WRITERS);

    for i in 0..NUM_WRITERS {
        let reg = Arc::clone(&registry);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            // All writers start together to maximise contention
            b.wait();
            let name = format!("video_type_{i}");
            let mut guard = reg.write().expect("write lock must succeed");
            guard.register(TypeInfo::new(&name, TypeKind::VideoFrame, 3));
        }));
    }

    for h in handles {
        h.join().expect("writer thread must not panic");
    }

    let reg = registry.read().expect("read lock");
    assert_eq!(
        reg.len(),
        NUM_WRITERS,
        "all {NUM_WRITERS} types must be registered"
    );
    for i in 0..NUM_WRITERS {
        let name = format!("video_type_{i}");
        assert!(
            reg.contains(&name),
            "type \"{name}\" must be present after concurrent registration"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent readers while writes are serialised
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-populate the registry, then have many threads read concurrently.
/// All readers must see consistent data (no panics, correct results).
#[test]
fn test_concurrent_readers_see_consistent_data() {
    const NUM_READERS: usize = 12;
    const TYPES: &[(&str, TypeKind, u8)] = &[
        ("yuv420p", TypeKind::VideoFrame, 3),
        ("nv12", TypeKind::VideoFrame, 2),
        ("pcm_s16le", TypeKind::AudioBuffer, 0),
        ("srt", TypeKind::Subtitle, 0),
        ("event_stream", TypeKind::DataPacket, 0),
    ];

    // Populate the registry before spawning readers
    let registry = {
        let mut reg = TypeRegistry::new();
        for (name, kind, planes) in TYPES {
            reg.register(TypeInfo::new(name, *kind, *planes));
        }
        Arc::new(RwLock::new(reg))
    };

    let barrier = Arc::new(Barrier::new(NUM_READERS));
    let mut handles = Vec::with_capacity(NUM_READERS);

    for _ in 0..NUM_READERS {
        let reg = Arc::clone(&registry);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let guard = reg.read().expect("read lock must succeed");
            // Every reader must find all pre-registered types
            for (name, expected_kind, expected_planes) in TYPES {
                let info = guard
                    .lookup(name)
                    .unwrap_or_else(|| panic!("type \"{name}\" must be found"));
                assert_eq!(info.kind, *expected_kind);
                assert_eq!(info.planes, *expected_planes);
            }
            // Registry length must be stable across all readers
            assert_eq!(guard.len(), TYPES.len());
        }));
    }

    for h in handles {
        h.join().expect("reader thread must not panic");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Interleaved readers and writers
// ─────────────────────────────────────────────────────────────────────────────

/// Some threads register types while others look up types already registered.
/// No data race, no panic, no lost registration.
#[test]
fn test_interleaved_readers_and_writers() {
    const NUM_WRITERS: usize = 4;
    const NUM_READERS: usize = 8;

    // Pre-populate one type so readers have something to look up immediately
    let registry = {
        let mut reg = TypeRegistry::new();
        reg.register(TypeInfo::new("bootstrap", TypeKind::DataPacket, 0));
        Arc::new(RwLock::new(reg))
    };

    let barrier = Arc::new(Barrier::new(NUM_WRITERS + NUM_READERS));
    let mut handles = Vec::with_capacity(NUM_WRITERS + NUM_READERS);

    // Writer threads
    for i in 0..NUM_WRITERS {
        let reg = Arc::clone(&registry);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let name = format!("audio_type_{i}");
            let mut guard = reg.write().expect("write lock");
            guard.register(TypeInfo::new(&name, TypeKind::AudioBuffer, 0));
        }));
    }

    // Reader threads
    for _ in 0..NUM_READERS {
        let reg = Arc::clone(&registry);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let guard = reg.read().expect("read lock");
            // "bootstrap" must always be present (pre-registered)
            assert!(
                guard.contains("bootstrap"),
                "pre-registered entry must always be visible"
            );
            // len() must be at least 1 (bootstrap) and at most 1+NUM_WRITERS
            let len = guard.len();
            assert!(
                len >= 1 && len <= 1 + NUM_WRITERS,
                "len ({len}) out of expected range [1, {}]",
                1 + NUM_WRITERS
            );
        }));
    }

    for h in handles {
        h.join().expect("thread must not panic");
    }

    // After all threads complete, every writer's type must be present
    let guard = registry.read().expect("read lock");
    assert_eq!(guard.len(), 1 + NUM_WRITERS);
    for i in 0..NUM_WRITERS {
        let name = format!("audio_type_{i}");
        assert!(
            guard.contains(&name),
            "writer type \"{name}\" must be in registry"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent unregister: each thread removes its own entry
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-register N types, then have N threads each remove exactly one.
/// Final registry must be empty.
#[test]
fn test_concurrent_unregister_leaves_registry_empty() {
    const N: usize = 8;

    let registry = {
        let mut reg = TypeRegistry::new();
        for i in 0..N {
            reg.register(TypeInfo::new(
                &format!("to_remove_{i}"),
                TypeKind::Metadata,
                0,
            ));
        }
        Arc::new(RwLock::new(reg))
    };

    let barrier = Arc::new(Barrier::new(N));
    let mut handles = Vec::with_capacity(N);

    for i in 0..N {
        let reg = Arc::clone(&registry);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let name = format!("to_remove_{i}");
            let mut guard = reg.write().expect("write lock");
            let removed = guard.unregister(&name);
            assert!(
                removed.is_some(),
                "each thread must successfully remove its own entry: \"{name}\""
            );
        }));
    }

    for h in handles {
        h.join().expect("thread must not panic");
    }

    let guard = registry.read().expect("read lock");
    assert_eq!(guard.len(), 0, "registry must be empty after all removals");
}

// ─────────────────────────────────────────────────────────────────────────────
// TypeRegistry is Send + Sync (compile-time assertion via wrapping)
// ─────────────────────────────────────────────────────────────────────────────

/// This is a compile-time check: if TypeRegistry were not Send, the
/// Arc<RwLock<TypeRegistry>> used in the tests above would not compile.
/// We make the requirement explicit with a static assertion function.
#[test]
fn test_type_registry_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<TypeRegistry>();
    assert_sync::<TypeRegistry>();
    // No runtime assertion needed — the test body is the proof.
}

// ─────────────────────────────────────────────────────────────────────────────
// with_defaults under concurrent read access
// ─────────────────────────────────────────────────────────────────────────────

/// Verify `with_defaults()` result is accessible correctly from many threads.
#[test]
fn test_with_defaults_concurrent_read() {
    const NUM_READERS: usize = 10;
    let registry = Arc::new(RwLock::new(TypeRegistry::with_defaults()));
    let barrier = Arc::new(Barrier::new(NUM_READERS));
    let mut handles = Vec::with_capacity(NUM_READERS);

    for _ in 0..NUM_READERS {
        let reg = Arc::clone(&registry);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let guard = reg.read().expect("read lock");
            assert!(guard.contains("yuv420p"), "yuv420p must be in defaults");
            assert!(guard.contains("nv12"), "nv12 must be in defaults");
            assert!(guard.contains("pcm_s16le"), "pcm_s16le must be in defaults");
            assert!(guard.contains("srt"), "srt must be in defaults");
            assert!(!guard.is_empty());
        }));
    }

    for h in handles {
        h.join().expect("reader thread must not panic");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// by_kind under concurrent read
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_by_kind_concurrent_read() {
    const NUM_READERS: usize = 6;
    let registry = Arc::new(RwLock::new(TypeRegistry::with_defaults()));
    let barrier = Arc::new(Barrier::new(NUM_READERS));
    let mut handles = Vec::with_capacity(NUM_READERS);

    for _ in 0..NUM_READERS {
        let reg = Arc::clone(&registry);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let guard = reg.read().expect("read lock");
            let videos = guard.by_kind(TypeKind::VideoFrame);
            for entry in &videos {
                assert_eq!(entry.kind, TypeKind::VideoFrame);
            }
            let audio = guard.by_kind(TypeKind::AudioBuffer);
            for entry in &audio {
                assert_eq!(entry.kind, TypeKind::AudioBuffer);
            }
        }));
    }

    for h in handles {
        h.join().expect("reader thread must not panic");
    }
}
