//! Integration tests: Selection, EditSelection, and LinkedSelection for
//! multi-track timeline editing.

use oximedia_edit::selection::{
    EditSelection, LinkedSelection, Selection, SelectionItem, SelectionMode, SelectionRange,
    TimelineClipRef,
};

// ── Selection: clip selection ─────────────────────────────────────────────────

#[test]
fn test_selection_add_and_is_clip_selected() {
    let mut sel = Selection::new();
    sel.select_clip(1, SelectionMode::Add);
    assert!(sel.is_clip_selected(1));
    assert!(!sel.is_clip_selected(2));
}

#[test]
fn test_selection_add_multiple_clips() {
    let mut sel = Selection::new();
    sel.select_clip(1, SelectionMode::Add);
    sel.select_clip(2, SelectionMode::Add);
    sel.select_clip(3, SelectionMode::Add);
    assert!(sel.is_clip_selected(1));
    assert!(sel.is_clip_selected(2));
    assert!(sel.is_clip_selected(3));
    assert_eq!(sel.selected_clips().len(), 3);
}

#[test]
fn test_selection_add_no_duplicate() {
    let mut sel = Selection::new();
    sel.select_clip(5, SelectionMode::Add);
    sel.select_clip(5, SelectionMode::Add);
    assert_eq!(sel.selected_clips().len(), 1);
}

#[test]
fn test_selection_replace_clears_previous() {
    let mut sel = Selection::new();
    sel.select_clip(1, SelectionMode::Add);
    sel.select_clip(2, SelectionMode::Add);
    sel.select_clip(99, SelectionMode::Replace);
    assert!(!sel.is_clip_selected(1));
    assert!(!sel.is_clip_selected(2));
    assert!(sel.is_clip_selected(99));
}

#[test]
fn test_selection_subtract_removes_clip() {
    let mut sel = Selection::new();
    sel.select_clip(1, SelectionMode::Add);
    sel.select_clip(2, SelectionMode::Add);
    sel.select_clip(1, SelectionMode::Subtract);
    assert!(!sel.is_clip_selected(1));
    assert!(sel.is_clip_selected(2));
}

#[test]
fn test_selection_toggle_adds_then_removes() {
    let mut sel = Selection::new();
    sel.select_clip(7, SelectionMode::Toggle);
    assert!(sel.is_clip_selected(7));
    sel.select_clip(7, SelectionMode::Toggle);
    assert!(!sel.is_clip_selected(7));
}

// ── Selection: range selection ────────────────────────────────────────────────

#[test]
fn test_selection_range_add() {
    let mut sel = Selection::new();
    sel.select_range(SelectionRange::new(0, 100), SelectionMode::Add);
    assert_eq!(sel.ranges().len(), 1);
    assert_eq!(sel.selected_duration(), 100);
}

#[test]
fn test_selection_range_add_merges_overlapping() {
    let mut sel = Selection::new();
    sel.select_range(SelectionRange::new(0, 60), SelectionMode::Add);
    sel.select_range(SelectionRange::new(40, 100), SelectionMode::Add);
    assert_eq!(sel.ranges().len(), 1);
    assert_eq!(sel.ranges()[0].end, 100);
}

#[test]
fn test_selection_range_subtract_splits() {
    let mut sel = Selection::new();
    sel.select_range(SelectionRange::new(0, 100), SelectionMode::Add);
    sel.select_range(SelectionRange::new(40, 60), SelectionMode::Subtract);
    assert_eq!(sel.ranges().len(), 2);
    assert_eq!(sel.ranges()[0], SelectionRange::new(0, 40));
    assert_eq!(sel.ranges()[1], SelectionRange::new(60, 100));
}

#[test]
fn test_selection_range_replace() {
    let mut sel = Selection::new();
    sel.select_range(SelectionRange::new(0, 50), SelectionMode::Add);
    sel.select_range(SelectionRange::new(200, 300), SelectionMode::Replace);
    assert_eq!(sel.ranges().len(), 1);
    assert_eq!(sel.ranges()[0], SelectionRange::new(200, 300));
}

// ── Selection: select_all_in_range ───────────────────────────────────────────

#[test]
fn test_select_all_in_range_two_tracks() {
    // Simulate two tracks: video clips on track 0, audio on track 1.
    let clips = vec![
        TimelineClipRef::new(1, 0, 1000),    // track-0 clip
        TimelineClipRef::new(2, 0, 1000),    // track-1 clip (same time range)
        TimelineClipRef::new(3, 2000, 3000), // outside range
    ];
    let mut sel = Selection::new();
    sel.select_all_in_range(&clips, SelectionRange::new(0, 1000));
    assert!(sel.is_clip_selected(1));
    assert!(sel.is_clip_selected(2));
    assert!(!sel.is_clip_selected(3));
}

#[test]
fn test_select_all_in_range_partial_overlap_included() {
    let clips = vec![
        TimelineClipRef::new(10, 900, 1100), // overlaps [500, 1000) at its start
    ];
    let mut sel = Selection::new();
    sel.select_all_in_range(&clips, SelectionRange::new(500, 1000));
    assert!(
        sel.is_clip_selected(10),
        "partially overlapping clip must be selected"
    );
}

#[test]
fn test_select_all_in_range_empty_range_selects_nothing() {
    let clips = vec![TimelineClipRef::new(1, 0, 100)];
    let mut sel = Selection::new();
    sel.select_all_in_range(&clips, SelectionRange::new(0, 0));
    assert!(!sel.is_clip_selected(1));
}

// ── Selection: track selection ────────────────────────────────────────────────

#[test]
fn test_selection_track_add() {
    let mut sel = Selection::new();
    sel.select_track(0, SelectionMode::Add);
    sel.select_track(1, SelectionMode::Add);
    assert!(sel.is_track_selected(0));
    assert!(sel.is_track_selected(1));
}

#[test]
fn test_selection_track_subtract() {
    let mut sel = Selection::new();
    sel.select_track(0, SelectionMode::Add);
    sel.select_track(1, SelectionMode::Add);
    sel.select_track(0, SelectionMode::Subtract);
    assert!(!sel.is_track_selected(0));
    assert!(sel.is_track_selected(1));
}

#[test]
fn test_selection_track_toggle() {
    let mut sel = Selection::new();
    sel.select_track(2, SelectionMode::Toggle);
    assert!(sel.is_track_selected(2));
    sel.select_track(2, SelectionMode::Toggle);
    assert!(!sel.is_track_selected(2));
}

// ── Selection: clear and is_empty ─────────────────────────────────────────────

#[test]
fn test_selection_is_empty_initially() {
    let sel = Selection::new();
    assert!(sel.is_empty());
}

#[test]
fn test_selection_clear_removes_everything() {
    let mut sel = Selection::new();
    sel.select_clip(1, SelectionMode::Add);
    sel.select_track(0, SelectionMode::Add);
    sel.select_range(SelectionRange::new(0, 100), SelectionMode::Add);
    sel.clear();
    assert!(sel.is_empty());
}

// ── EditSelection: multi-track clip selection ─────────────────────────────────

#[test]
fn test_edit_selection_add_contains() {
    let mut sel = EditSelection::new();
    sel.add(SelectionItem::new(1, 0));
    assert!(sel.contains(1));
    assert!(!sel.contains(99));
}

#[test]
fn test_edit_selection_add_no_duplicate() {
    let mut sel = EditSelection::new();
    sel.add(SelectionItem::new(5, 0));
    sel.add(SelectionItem::new(5, 0));
    assert_eq!(sel.count(), 1);
}

#[test]
fn test_edit_selection_remove() {
    let mut sel = EditSelection::new();
    sel.add(SelectionItem::new(3, 1));
    sel.remove(3);
    assert!(!sel.contains(3));
    assert_eq!(sel.count(), 0);
}

#[test]
fn test_edit_selection_clear() {
    let mut sel = EditSelection::new();
    for i in 0..5u64 {
        sel.add(SelectionItem::new(i, i as u32));
    }
    sel.clear();
    assert_eq!(sel.count(), 0);
}

#[test]
fn test_edit_selection_tracks_unique() {
    let mut sel = EditSelection::new();
    sel.add(SelectionItem::new(1, 0));
    sel.add(SelectionItem::new(2, 0));
    sel.add(SelectionItem::new(3, 1));
    sel.add(SelectionItem::new(4, 2));
    let tracks = sel.tracks();
    assert_eq!(tracks.len(), 3);
    assert!(tracks.contains(&0));
    assert!(tracks.contains(&1));
    assert!(tracks.contains(&2));
}

#[test]
fn test_edit_selection_multitrack_video_audio_pair() {
    // Simulate selecting a linked video+audio clip pair (common NLE pattern).
    let mut sel = EditSelection::new();
    sel.add(SelectionItem::new(101, 0)); // video clip on track 0
    sel.add(SelectionItem::new(201, 1)); // audio clip on track 1
    assert_eq!(sel.count(), 2);
    assert!(sel.contains(101));
    assert!(sel.contains(201));
    let tracks = sel.tracks();
    assert_eq!(tracks.len(), 2);
}

#[test]
fn test_selection_item_same_track_true() {
    let a = SelectionItem::new(1, 0);
    let b = SelectionItem::new(2, 0);
    assert!(a.same_track(&b));
}

#[test]
fn test_selection_item_same_track_false() {
    let a = SelectionItem::new(1, 0);
    let b = SelectionItem::new(2, 1);
    assert!(!a.same_track(&b));
}

// ── LinkedSelection ───────────────────────────────────────────────────────────

#[test]
fn test_linked_selection_linked_clips_video_audio_pair() {
    let mut ls = LinkedSelection::new();
    ls.add_linked_group(vec![100, 200]); // clip 100 (video) linked to clip 200 (audio)
    let linked = ls.linked_clips(100);
    assert_eq!(linked, vec![200]);
    let linked_from_audio = ls.linked_clips(200);
    assert_eq!(linked_from_audio, vec![100]);
}

#[test]
fn test_linked_selection_unlinked_returns_empty() {
    let ls = LinkedSelection::new();
    assert!(ls.linked_clips(999).is_empty());
}

#[test]
fn test_linked_selection_multiple_groups_isolated() {
    let mut ls = LinkedSelection::new();
    ls.add_linked_group(vec![1, 2]);
    ls.add_linked_group(vec![3, 4]);
    assert!(!ls.linked_clips(1).contains(&3), "groups must be isolated");
    assert!(!ls.linked_clips(3).contains(&1), "groups must be isolated");
}

#[test]
fn test_linked_selection_self_not_in_linked_list() {
    let mut ls = LinkedSelection::new();
    ls.add_linked_group(vec![10, 11, 12]);
    let linked = ls.linked_clips(10);
    assert!(
        !linked.contains(&10),
        "clip must not appear in its own linked list"
    );
}

#[test]
fn test_linked_selection_three_way_link() {
    let mut ls = LinkedSelection::new();
    ls.add_linked_group(vec![1, 2, 3]);
    let linked = ls.linked_clips(1);
    assert!(linked.contains(&2));
    assert!(linked.contains(&3));
    assert_eq!(linked.len(), 2);
}
