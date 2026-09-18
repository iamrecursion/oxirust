//! AAF merge capability
//!
//! Combines multiple `AafFile` structures into a single composition by
//! merging their `ContentStorage` mobs, renaming colliding mob IDs,
//! and optionally creating a master composition that references all
//! top-level composition mobs from the source files.
//!
//! # Merge strategy
//!
//! 1. All mobs from every source file are collected.
//! 2. UUID collisions are resolved by replacing the colliding mob's UUID
//!    with a freshly generated one (the source clip references inside the
//!    affected mob are updated accordingly).
//! 3. An optional *merge composition mob* can be created to hold a flat
//!    video track that concatenates the first video track of each source
//!    composition mob.
//! 4. Essence data is concatenated; mob-ID references inside essence are
//!    remapped to the new UUIDs after collision resolution.

use crate::composition::{
    CompositionMob, Sequence, SequenceComponent, SourceClip, Track, TrackType,
};
use crate::dictionary::Auid;
use crate::object_model::Segment;
use crate::timeline::{EditRate, Position};
use crate::{AafError, AafFile, ContentStorage, EssenceData, Result};
use std::collections::HashMap;
use uuid::Uuid;

/// Options controlling merge behaviour
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Name of the new master composition mob (if `create_master_comp` is true)
    pub master_comp_name: String,
    /// Whether to create a single master composition that references all
    /// input composition mobs in sequence
    pub create_master_comp: bool,
    /// Edit rate for the new master composition track
    pub master_edit_rate: EditRate,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            master_comp_name: "Merged Composition".to_string(),
            create_master_comp: true,
            master_edit_rate: EditRate::PAL_25,
        }
    }
}

impl MergeOptions {
    /// Create default options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the master composition name
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.master_comp_name = name.into();
        self
    }

    /// Set whether to create a master composition
    #[must_use]
    pub fn with_master_comp(mut self, create: bool) -> Self {
        self.create_master_comp = create;
        self
    }

    /// Set the edit rate for the master composition
    #[must_use]
    pub fn with_edit_rate(mut self, rate: EditRate) -> Self {
        self.master_edit_rate = rate;
        self
    }
}

/// Merge multiple `AafFile` objects into a single combined `AafFile`.
///
/// # Errors
///
/// Returns `AafError::InvalidFile` if `files` is empty.
pub fn merge_aaf_files(files: &[AafFile], options: &MergeOptions) -> Result<AafFile> {
    if files.is_empty() {
        return Err(AafError::InvalidFile(
            "merge_aaf_files: no input files provided".to_string(),
        ));
    }

    // UUID collision map: old UUID → new UUID
    let mut uuid_remap: HashMap<Uuid, Uuid> = HashMap::new();

    // Collect all source composition mob IDs (for building master composition)
    let mut source_comp_mob_ids: Vec<Uuid> = Vec::new();

    let mut merged_storage = ContentStorage::new();
    let mut merged_essence: Vec<EssenceData> = Vec::new();

    for file in files {
        // --- Merge mobs ---
        for mob in file.master_mobs().iter().chain(file.source_mobs().iter()) {
            let new_id = if merged_storage.find_mob(&mob.mob_id()).is_some() {
                // Collision — generate fresh UUID
                let fresh = Uuid::new_v4();
                uuid_remap.insert(mob.mob_id(), fresh);
                fresh
            } else {
                mob.mob_id()
            };

            let mut remapped = (*mob).clone();
            *remapped.mob_id_mut() = new_id;
            merged_storage.add_mob(remapped);
        }

        // --- Merge composition mobs ---
        for comp_mob in file.composition_mobs() {
            let new_id = if merged_storage
                .find_composition_mob(&comp_mob.mob_id())
                .is_some()
            {
                let fresh = Uuid::new_v4();
                uuid_remap.insert(comp_mob.mob_id(), fresh);
                fresh
            } else {
                comp_mob.mob_id()
            };

            let mut remapped = comp_mob.clone();
            *remapped.mob_mut().mob_id_mut() = new_id;
            // Remap source clip references within tracks
            remap_composition_mob_refs(&mut remapped, &uuid_remap);
            source_comp_mob_ids.push(new_id);
            merged_storage.add_composition_mob(remapped);
        }

        // --- Merge essence data ---
        for essence in file.essence_data() {
            let remapped_id = uuid_remap
                .get(&essence.mob_id())
                .copied()
                .unwrap_or_else(|| essence.mob_id());
            merged_essence.push(EssenceData::new(remapped_id, essence.data().to_vec()));
        }
    }

    // --- Optionally create master composition ---
    if options.create_master_comp && !source_comp_mob_ids.is_empty() {
        let master_comp = build_master_composition(&source_comp_mob_ids, &merged_storage, options)?;
        merged_storage.add_composition_mob(master_comp);
    }

    // Build result using the header/dictionary from the first file
    let first = &files[0];
    Ok(AafFile {
        header: first.header().clone(),
        dictionary: first.dictionary().clone(),
        content_storage: merged_storage,
        essence_data: merged_essence,
    })
}

/// Update all source-clip mob references inside a `CompositionMob` using the remap table.
fn remap_composition_mob_refs(comp: &mut CompositionMob, remap: &HashMap<Uuid, Uuid>) {
    if remap.is_empty() {
        return;
    }
    for track in comp.tracks_mut() {
        if let Some(ref mut seg_box) = track.segment {
            if let Segment::Sequence(ref mut seq) = **seg_box {
                for component in &mut seq.components {
                    if let Segment::SourceClip(ref mut clip) = component.segment {
                        if let Some(&new_id) = remap.get(&clip.source_mob_id) {
                            clip.source_mob_id = new_id;
                        }
                    }
                }
            }
        }
    }
}

/// Build a master `CompositionMob` whose video track concatenates clips
/// from each source composition mob.
fn build_master_composition(
    source_ids: &[Uuid],
    storage: &ContentStorage,
    options: &MergeOptions,
) -> Result<CompositionMob> {
    let master_id = Uuid::new_v4();
    let mut master = CompositionMob::new(master_id, &options.master_comp_name);

    let mut video_seq = Sequence::new(Auid::PICTURE);

    for &src_id in source_ids {
        // Find the duration of the source composition's first picture track
        let duration = storage
            .find_composition_mob(&src_id)
            .and_then(|c| {
                c.picture_tracks()
                    .into_iter()
                    .next()
                    .and_then(|t| t.duration())
            })
            .unwrap_or(0);

        if duration > 0 {
            video_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
                duration,
                Position::zero(),
                src_id,
                1, // slot 1 = first video track by convention
            )));
        }
    }

    let mut video_track = Track::new(1, "V1", options.master_edit_rate, TrackType::Picture);
    video_track.set_sequence(video_seq);
    master.add_track(video_track);

    Ok(master)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType};
    use crate::dictionary::Auid;
    use crate::object_model::{Mob, MobType};
    use crate::timeline::{EditRate, Position};
    use crate::{AafFile, ContentStorage};
    use uuid::Uuid;

    fn make_simple_aaf(name: &str, clip_duration: i64) -> AafFile {
        let mut storage = ContentStorage::new();
        let source_id = Uuid::new_v4();
        storage.add_mob(Mob::new(
            source_id,
            "source.mov".to_string(),
            MobType::Source,
        ));

        let comp_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(comp_id, name);
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            clip_duration,
            Position::zero(),
            source_id,
            1,
        )));
        let mut track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        AafFile {
            header: crate::object_model::Header::new(),
            dictionary: crate::dictionary::Dictionary::new(),
            content_storage: storage,
            essence_data: Vec::new(),
        }
    }

    #[test]
    fn test_merge_empty_files_returns_error() {
        let options = MergeOptions::default();
        let result = merge_aaf_files(&[], &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_single_file() {
        let file = make_simple_aaf("Edit1", 100);
        let options = MergeOptions::new().with_master_comp(false);
        let merged = merge_aaf_files(&[file], &options).expect("merge should succeed");
        // 1 composition mob in + 0 master = 1
        assert_eq!(merged.composition_mobs().len(), 1);
    }

    #[test]
    fn test_merge_two_files_without_master_comp() {
        let file1 = make_simple_aaf("Edit1", 100);
        let file2 = make_simple_aaf("Edit2", 200);
        let options = MergeOptions::new().with_master_comp(false);
        let merged = merge_aaf_files(&[file1, file2], &options).expect("merge should succeed");
        assert_eq!(merged.composition_mobs().len(), 2);
    }

    #[test]
    fn test_merge_two_files_with_master_comp() {
        let file1 = make_simple_aaf("Edit1", 100);
        let file2 = make_simple_aaf("Edit2", 200);
        let options = MergeOptions::new()
            .with_master_comp(true)
            .with_name("Big Merge");
        let merged = merge_aaf_files(&[file1, file2], &options).expect("merge should succeed");
        // 2 source comps + 1 master
        assert_eq!(merged.composition_mobs().len(), 3);

        // Master comp should have a video track
        let master = merged
            .content_storage()
            .find_composition_mob_by_name("Big Merge")
            .expect("master comp should exist");
        assert!(!master.picture_tracks().is_empty());
    }

    #[test]
    fn test_merge_master_comp_duration() {
        let file1 = make_simple_aaf("A", 100);
        let file2 = make_simple_aaf("B", 50);
        let options = MergeOptions::new()
            .with_master_comp(true)
            .with_name("Combined");
        let merged = merge_aaf_files(&[file1, file2], &options).expect("merge should succeed");
        let master = merged
            .content_storage()
            .find_composition_mob_by_name("Combined")
            .expect("master comp");
        // Duration = 100 + 50 = 150
        assert_eq!(master.duration(), Some(150));
    }

    #[test]
    fn test_merge_uuid_collision_resolved() {
        // Create two files with composition mobs sharing the same UUID (forced)
        let shared_id = Uuid::new_v4();
        let make_file = |clip: i64| -> AafFile {
            let mut storage = ContentStorage::new();
            let mut comp = CompositionMob::new(shared_id, "Comp");
            let mut seq = Sequence::new(Auid::PICTURE);
            seq.add_component(SequenceComponent::Filler(Filler::new(clip)));
            let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
            track.set_sequence(seq);
            comp.add_track(track);
            storage.add_composition_mob(comp);
            AafFile {
                header: crate::object_model::Header::new(),
                dictionary: crate::dictionary::Dictionary::new(),
                content_storage: storage,
                essence_data: Vec::new(),
            }
        };

        let f1 = make_file(100);
        let f2 = make_file(200);
        let options = MergeOptions::new().with_master_comp(false);
        let merged = merge_aaf_files(&[f1, f2], &options).expect("merge");
        // Both composition mobs should survive (collision resolved)
        assert_eq!(merged.composition_mobs().len(), 2);
        // The two surviving mobs must have different IDs
        let ids: Vec<Uuid> = merged
            .composition_mobs()
            .iter()
            .map(|c| c.mob_id())
            .collect();
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn test_merge_essence_data_concatenated() {
        let mob1 = Uuid::new_v4();
        let mob2 = Uuid::new_v4();
        let mut storage = ContentStorage::new();
        let comp = CompositionMob::new(Uuid::new_v4(), "E");
        storage.add_composition_mob(comp);
        let file1 = AafFile {
            header: crate::object_model::Header::new(),
            dictionary: crate::dictionary::Dictionary::new(),
            content_storage: storage,
            essence_data: vec![EssenceData::new(mob1, vec![1, 2, 3])],
        };
        let mut storage2 = ContentStorage::new();
        let comp2 = CompositionMob::new(Uuid::new_v4(), "F");
        storage2.add_composition_mob(comp2);
        let file2 = AafFile {
            header: crate::object_model::Header::new(),
            dictionary: crate::dictionary::Dictionary::new(),
            content_storage: storage2,
            essence_data: vec![EssenceData::new(mob2, vec![4, 5, 6])],
        };
        let options = MergeOptions::new().with_master_comp(false);
        let merged = merge_aaf_files(&[file1, file2], &options).expect("merge");
        assert_eq!(merged.essence_data().len(), 2);
    }
}
