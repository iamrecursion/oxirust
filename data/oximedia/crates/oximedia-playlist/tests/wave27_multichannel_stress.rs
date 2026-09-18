//! Wave 27 — concurrent multichannel manager stress test.
//!
//! Exercises `ChannelManager` (interior `Arc<RwLock<..>>`, all `&self`) under
//! contention: 50 channels are added, given a playlist, and started from 50
//! concurrent tokio tasks against a single shared manager. The invariant is
//! that every channel lands exactly once and no lock is poisoned.

use oximedia_playlist::multichannel::manager::{Channel, ChannelManager, OutputConfig};
use oximedia_playlist::{Playlist, PlaylistType};
use std::sync::Arc;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_50_channels_concurrent_start() {
    const CHANNEL_COUNT: u32 = 50;

    let manager = Arc::new(ChannelManager::new());

    let mut handles = Vec::with_capacity(CHANNEL_COUNT as usize);
    for i in 0..CHANNEL_COUNT {
        let manager = Arc::clone(&manager);
        handles.push(tokio::spawn(async move {
            let id = format!("ch{i}");
            let channel = Channel::new(
                id.clone(),
                format!("Channel {i}"),
                i,
                OutputConfig::Sdi { output: i },
            );
            manager
                .add_channel(channel)
                .expect("add_channel must not fail (lock not poisoned)");

            let playlist = Playlist::new(format!("pl{i}"), PlaylistType::Linear);
            manager
                .load_playlist(&id, playlist)
                .expect("load_playlist must succeed for a just-added channel");

            manager
                .start_channel(&id)
                .expect("start_channel must succeed for a loaded channel");
        }));
    }

    for handle in handles {
        handle.await.expect("channel task must not panic");
    }

    // Every channel landed exactly once.
    assert_eq!(
        manager.channel_count(),
        CHANNEL_COUNT as usize,
        "channel_count must equal the number of concurrently added channels"
    );

    let all = manager
        .get_all_channels()
        .expect("get_all_channels must not fail (lock not poisoned)");
    assert_eq!(
        all.len(),
        CHANNEL_COUNT as usize,
        "get_all_channels must return every channel"
    );

    // The set of ids is exactly {ch0..ch49} with no duplicates or drops.
    let mut ids: Vec<String> = all.iter().map(|c| c.id.clone()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(
        ids.len(),
        CHANNEL_COUNT as usize,
        "channel ids must be unique with no overwrites"
    );
    for i in 0..CHANNEL_COUNT {
        assert!(
            ids.contains(&format!("ch{i}")),
            "missing channel ch{i} after concurrent insertion"
        );
    }
}
