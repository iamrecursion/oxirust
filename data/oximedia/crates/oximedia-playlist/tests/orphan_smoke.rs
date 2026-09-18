//! Smoke tests for newly-wired orphan modules in oximedia-playlist.

#[test]
fn test_m3u_entry_new() {
    use oximedia_playlist::m3u::M3uEntry;
    let entry = M3uEntry::new("/path/to/track.mp3");
    assert_eq!(entry.path, "/path/to/track.mp3");
}

#[test]
fn test_m3u8_encryption_method() {
    use oximedia_playlist::m3u8::EncryptionMethod;
    let _ = EncryptionMethod::Aes128;
}

#[test]
fn test_playlist_archive_entry() {
    use oximedia_playlist::playlist_archive::ArchivedEntry;
    use std::time::Duration;
    let entry = ArchivedEntry::new("asset-001", Duration::from_secs(60), 0);
    assert_eq!(entry.position, 0);
}

#[test]
fn test_playlist_history_edit() {
    use oximedia_playlist::playlist_history::PlaylistEdit;
    let edit = PlaylistEdit::InsertItem {
        index: 0,
        item_id: "a1".to_string(),
    };
    let _ = edit.inverse();
}

#[test]
fn test_playlist_notify_event_kind() {
    use oximedia_playlist::playlist_notify::PlaylistEventKind;
    let _ = PlaylistEventKind::ItemAdded;
}

#[test]
fn test_playlist_rotation_strategy() {
    use oximedia_playlist::playlist_rotation::RotationStrategy;
    let _ = RotationStrategy::RoundRobin;
}

#[test]
fn test_scheduler_event_status() {
    use oximedia_playlist::scheduler::EventStatus;
    let _ = EventStatus::Pending;
}

#[test]
fn test_smart_library_item() {
    use oximedia_playlist::smart::LibraryItem;
    let item = LibraryItem::new("track-1", "My Track");
    assert_eq!(item.id, "track-1");
}
