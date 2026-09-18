//! AAF Edit Session round-trip tests.
//!
//! These tests construct an in-memory AAF edit session, serialise it via
//! [`AafEditSession::write_to`] (binary CFB → KLV streams), then recover
//! it through [`AafEditSession::from_reader`] and verify structural equality.

use oximedia_aaf::composition::{
    CompositionMob, Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType,
};
use oximedia_aaf::dictionary::Auid;
use oximedia_aaf::edit_session::{AafEditSession, UnknownProperty};
use oximedia_aaf::object_model::{Mob, MobType};
use oximedia_aaf::timeline::{EditRate, Position};
use oximedia_aaf::{AafFile, EssenceData};
use std::env::temp_dir;
use std::io::Cursor;
use uuid::Uuid;

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_simple_session() -> (AafEditSession, Uuid, Uuid) {
    let source_mob_id = Uuid::new_v4();
    let comp_id = Uuid::new_v4();

    let mut file = AafFile::new();
    let cs = file.content_storage_mut_internal();
    cs.add_mob(Mob::new(
        source_mob_id,
        "source.mov".to_string(),
        MobType::Source,
    ));
    let mut comp = CompositionMob::new(comp_id, "MyEdit");
    let mut seq = Sequence::new(Auid::PICTURE);
    seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
        120,
        Position::new(10),
        source_mob_id,
        1,
    )));
    seq.add_component(SequenceComponent::Filler(Filler::new(40)));
    let mut track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
    track.set_sequence(seq);
    comp.add_track(track);
    cs.add_composition_mob(comp);

    (AafEditSession::from_file(file), source_mob_id, comp_id)
}

// `AafFile::content_storage_mut` is now publicly exposed; reuse it.
trait Internal {
    fn content_storage_mut_internal(&mut self) -> &mut oximedia_aaf::ContentStorage;
}
impl Internal for AafFile {
    fn content_storage_mut_internal(&mut self) -> &mut oximedia_aaf::ContentStorage {
        self.content_storage_mut()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn write_to_emits_cfb_signature() {
    let session = AafEditSession::from_file(AafFile::new());
    let mut buf = Cursor::new(Vec::<u8>::new());
    session.write_to(&mut buf).expect("write must succeed");
    let bytes = buf.into_inner();
    assert!(bytes.len() >= 8);
    assert_eq!(
        &bytes[..8],
        &[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1],
        "CFB signature mismatch"
    );
}

#[test]
fn round_trip_simple_composition() {
    let (session, source_id, comp_id) = build_simple_session();
    let mut buf = Cursor::new(Vec::<u8>::new());
    session.write_to(&mut buf).expect("write");
    buf.set_position(0);
    let recovered = AafEditSession::from_reader(buf).expect("read");

    let comp = recovered
        .file
        .content_storage()
        .find_composition_mob(&comp_id)
        .expect("composition mob preserved");
    assert_eq!(comp.name(), "MyEdit");
    let tracks = comp.tracks();
    assert_eq!(tracks.len(), 1);
    let track = &tracks[0];
    assert_eq!(track.track_id, 1);
    assert_eq!(track.name, "V1");
    assert_eq!(track.edit_rate, EditRate::PAL_25);
    assert_eq!(track.track_type, TrackType::Picture);
    let seq = track.sequence.as_ref().expect("sequence preserved");
    assert_eq!(seq.components.len(), 2);

    match &seq.components[0] {
        SequenceComponent::SourceClip(c) => {
            assert_eq!(c.length, 120);
            assert_eq!(c.start_time, Position::new(10));
            assert_eq!(c.source_mob_id, source_id);
            assert_eq!(c.source_mob_slot_id, 1);
        }
        _ => panic!("expected SourceClip"),
    }
    match &seq.components[1] {
        SequenceComponent::Filler(f) => assert_eq!(f.length, 40),
        _ => panic!("expected Filler"),
    }

    let sources: Vec<_> = recovered.file.source_mobs();
    assert!(
        sources.iter().any(|m| m.mob_id == source_id),
        "source mob preserved"
    );
}

#[test]
fn round_trip_preserves_unknown_properties() {
    let (mut session, _, _) = build_simple_session();
    session.preserve_property(
        "Header",
        UnknownProperty::new(0x1234, vec![0xDE, 0xAD, 0xBE, 0xEF]),
    );
    session.preserve_property("Mob/abc", UnknownProperty::new(0x42, vec![0xFF; 100]));

    let mut buf = Cursor::new(Vec::<u8>::new());
    session.write_to(&mut buf).expect("write");
    buf.set_position(0);
    let recovered = AafEditSession::from_reader(buf).expect("read");
    let hdr_props = recovered.preserved_properties("Header");
    assert_eq!(hdr_props.len(), 1);
    assert_eq!(hdr_props[0].tag, 0x1234);
    assert_eq!(hdr_props[0].value, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    let mob_props = recovered.preserved_properties("Mob/abc");
    assert_eq!(mob_props.len(), 1);
    assert_eq!(mob_props[0].tag, 0x42);
    assert_eq!(mob_props[0].value.len(), 100);
    assert!(mob_props[0].value.iter().all(|&b| b == 0xFF));
}

#[test]
fn round_trip_large_content_crosses_regular_fat() {
    // Build a composition whose serialised ContentStorage stream exceeds
    // the 4 KiB mini-stream cutoff and is therefore routed through the
    // regular-FAT sector chain instead of the mini-stream.  This exercises
    // multi-sector FAT chains on the read path.
    let mut file = AafFile::new();
    let cs = file.content_storage_mut_internal();
    for i in 0..200 {
        let id = Uuid::new_v4();
        let mut comp = CompositionMob::new(id, format!("LargeEdit{i:03}"));
        let mut seq = Sequence::new(Auid::PICTURE);
        // Make each track a bit chunky so the cumulative content stream
        // crosses the 4 KiB cutoff.
        for j in 0..10 {
            seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
                10 + j,
                Position::zero(),
                Uuid::new_v4(),
                1,
            )));
        }
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        cs.add_composition_mob(comp);
    }
    let session = AafEditSession::from_file(file);
    let mut buf = Cursor::new(Vec::<u8>::new());
    session.write_to(&mut buf).expect("write");
    let len = buf.get_ref().len();
    // Sanity: the on-disk file must be substantial enough that one of the
    // streams certainly used a regular-FAT chain (> 4 KiB stream + sector
    // overhead).
    assert!(len > 8 * 1024, "expected CFB > 8 KiB, got {len}");
    buf.set_position(0);
    let recovered = AafEditSession::from_reader(buf).expect("read");
    assert_eq!(recovered.file.composition_mobs().len(), 200);
}

#[test]
fn round_trip_many_mobs_crosses_sector_boundary() {
    let mut file = AafFile::new();
    let cs = file.content_storage_mut_internal();
    // 20 mobs guarantees > 1 KiB of content storage, crossing a single
    // 512-byte CFB sector.
    let mut ids = Vec::with_capacity(20);
    for i in 0..20 {
        let id = Uuid::new_v4();
        ids.push(id);
        let mut comp = CompositionMob::new(id, format!("Edit{i:02}"));
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100 + i as i64,
            Position::zero(),
            Uuid::new_v4(),
            1,
        )));
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        cs.add_composition_mob(comp);
    }
    let session = AafEditSession::from_file(file);
    let mut buf = Cursor::new(Vec::<u8>::new());
    session.write_to(&mut buf).expect("write");
    buf.set_position(0);
    let recovered = AafEditSession::from_reader(buf).expect("read");
    for id in &ids {
        let found = recovered.file.content_storage().find_composition_mob(id);
        assert!(found.is_some(), "mob {id} survived sector boundary");
    }
}

#[test]
fn mini_stream_path_for_small_payload() {
    // An empty session yields a tiny ContentStorage stream which must
    // therefore traverse the mini-stream FAT chain.
    let session = AafEditSession::from_file(AafFile::new());
    let mut buf = Cursor::new(Vec::<u8>::new());
    session.write_to(&mut buf).expect("write");
    buf.set_position(0);
    let recovered = AafEditSession::from_reader(buf).expect("read mini stream");
    assert_eq!(recovered.file.composition_mobs().len(), 0);
}

#[test]
fn round_trip_via_temp_file() {
    let (session, _, comp_id) = build_simple_session();
    let path = temp_dir().join("oximedia_aaf_round_trip_test.aaf");
    let mut sess_mut = session;
    sess_mut.save_to(&path).expect("save to temp file");
    let recovered = AafEditSession::open(&path).expect("open back");
    assert!(recovered
        .file
        .content_storage()
        .find_composition_mob(&comp_id)
        .is_some());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn round_trip_preserves_essence_data() {
    let mut file = AafFile::new();
    let mob_id = Uuid::new_v4();
    file.essence_data_mut_internal()
        .push(EssenceData::new(mob_id, vec![0xAA, 0xBB, 0xCC, 0xDD]));
    let session = AafEditSession::from_file(file);
    let mut buf = Cursor::new(Vec::<u8>::new());
    session.write_to(&mut buf).expect("write");
    buf.set_position(0);
    let recovered = AafEditSession::from_reader(buf).expect("read");
    let essences = recovered.file.essence_data();
    assert_eq!(essences.len(), 1);
    assert_eq!(essences[0].mob_id(), mob_id);
    assert_eq!(essences[0].data(), &[0xAA, 0xBB, 0xCC, 0xDD]);
}

trait EssenceMut {
    fn essence_data_mut_internal(&mut self) -> &mut Vec<EssenceData>;
}
impl EssenceMut for AafFile {
    fn essence_data_mut_internal(&mut self) -> &mut Vec<EssenceData> {
        self.essence_data_mut()
    }
}
