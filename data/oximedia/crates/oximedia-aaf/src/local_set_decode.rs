//! Local-Set decoder — symmetric counterpart to [`crate::local_set_encode`].
//!
//! Walks the `(AUID, Vec<u8>)` pairs produced by the encoder and reconstructs
//! the in-memory object model.  Unknown AUIDs are returned through the
//! [`DecodedHeader::unknown`] / `DecodedMob::unknown` / etc. fields so the
//! edit-session round-trip preserves extension properties verbatim.

use crate::composition::{
    CompositionMob, Filler, Sequence, SequenceComponent, SourceClip, Track, Transition,
};
use crate::dictionary::Auid;
use crate::edit_session::UnknownProperty;
use crate::klv;
use crate::local_set_encode::{
    byte_to_composition_usage, byte_to_fade_type, byte_to_mob_type, byte_to_track_type,
    dec_edit_rate, dec_i64, dec_position, dec_str, dec_u32, dec_uuid, prop, TAG_EFFECT, TAG_FILLER,
    TAG_SOURCE_CLIP, TAG_TRANSITION, UNKNOWN_PROP_AUID,
};
use crate::object_model::{Header, Mob, MobType};
use crate::timeline::EditRate;
use crate::{ContentStorage, EssenceData, Result};
use std::collections::HashMap;

/// Decoded header plus any unrecognised KLV pairs found in the stream.
#[derive(Debug, Default)]
pub struct DecodedHeader {
    pub header: Header,
    pub unknown: Vec<(Auid, Vec<u8>)>,
}

/// Decoded composition / non-composition mob with its raw extras.
#[derive(Debug)]
pub enum DecodedMob {
    Composition(Box<CompositionMob>),
    Other(Box<Mob>),
}

/// Decode a Header from its Local-Set stream.
pub fn decode_header(bytes: &[u8]) -> Result<DecodedHeader> {
    let entries = klv::decode_local_set(bytes)?;
    let mut hdr = Header::new();
    let mut unknown = Vec::new();
    for (key, value) in entries {
        if key == prop::BYTE_ORDER && value.len() == 2 {
            hdr.byte_order = u16::from_le_bytes([value[0], value[1]]);
        } else if key == prop::LAST_MODIFIED && value.len() == 8 {
            hdr.last_modified = u64::from_le_bytes([
                value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
            ]);
        } else if key == prop::MAJOR_VERSION && value.len() == 2 {
            hdr.major_version = u16::from_le_bytes([value[0], value[1]]);
        } else if key == prop::MINOR_VERSION && value.len() == 2 {
            hdr.minor_version = u16::from_le_bytes([value[0], value[1]]);
        } else if key == prop::OBJECT_MODEL_VERSION && value.len() == 4 {
            hdr.object_model_version = u32::from_le_bytes([value[0], value[1], value[2], value[3]]);
        } else if key == prop::OPERATIONAL_PATTERN && value.len() == 16 {
            let mut a = [0u8; 16];
            a.copy_from_slice(&value);
            hdr.operational_pattern = Auid::from_bytes(&a);
        } else if key == prop::ESSENCE_CONTAINERS && value.len() >= 4 {
            let count = u32::from_le_bytes([value[0], value[1], value[2], value[3]]) as usize;
            let mut containers = Vec::with_capacity(count);
            let mut idx = 4;
            for _ in 0..count {
                if value.len() < idx + 16 {
                    break;
                }
                let mut a = [0u8; 16];
                a.copy_from_slice(&value[idx..idx + 16]);
                containers.push(Auid::from_bytes(&a));
                idx += 16;
            }
            hdr.essence_containers = containers;
        } else {
            unknown.push((key, value));
        }
    }
    Ok(DecodedHeader {
        header: hdr,
        unknown,
    })
}

/// Decode a single SequenceComponent local-set into a [`SequenceComponent`].
pub fn decode_sequence_component(bytes: &[u8]) -> Result<SequenceComponent> {
    let entries = klv::decode_local_set(bytes)?;
    let mut tag: u8 = 0;
    let mut length: i64 = 0;
    let mut start_time = crate::timeline::Position::zero();
    let mut cut_point = crate::timeline::Position::zero();
    let mut source_mob_id = uuid::Uuid::nil();
    let mut source_slot_id: u32 = 0;
    let mut operation_id = Auid::null();
    for (key, value) in entries {
        if key == prop::COMPONENT_TAG && !value.is_empty() {
            tag = value[0];
        } else if key == prop::COMPONENT_LENGTH {
            length = dec_i64(&value);
        } else if key == prop::COMPONENT_START_TIME {
            start_time = dec_position(&value);
        } else if key == prop::COMPONENT_CUT_POINT {
            cut_point = dec_position(&value);
        } else if key == prop::COMPONENT_SOURCE_MOB_ID {
            source_mob_id = dec_uuid(&value);
        } else if key == prop::COMPONENT_SOURCE_SLOT_ID {
            source_slot_id = dec_u32(&value);
        } else if key == prop::SLOT_DATA_DEF && value.len() == 16 {
            let mut a = [0u8; 16];
            a.copy_from_slice(&value);
            operation_id = Auid::from_bytes(&a);
        }
    }
    let comp = match tag {
        TAG_SOURCE_CLIP => SequenceComponent::SourceClip(SourceClip::new(
            length,
            start_time,
            source_mob_id,
            source_slot_id,
        )),
        TAG_FILLER => SequenceComponent::Filler(Filler::new(length)),
        TAG_TRANSITION => SequenceComponent::Transition(Transition::new(length, cut_point)),
        TAG_EFFECT => {
            let mut effect = crate::composition::Effect::new(operation_id);
            if length > 0 {
                effect.set_length(length);
            }
            SequenceComponent::Effect(effect)
        }
        _ => SequenceComponent::Filler(Filler::new(length)),
    };
    Ok(comp)
}

/// Decode a Sequence local-set.
pub fn decode_sequence(bytes: &[u8]) -> Result<Sequence> {
    let entries = klv::decode_local_set(bytes)?;
    let mut data_def = Auid::null();
    let mut components = Vec::new();
    for (key, value) in entries {
        if key == prop::SEQ_DATA_DEFINITION && value.len() == 16 {
            let mut a = [0u8; 16];
            a.copy_from_slice(&value);
            data_def = Auid::from_bytes(&a);
        } else if key == prop::SEQ_COMPONENTS {
            components.push(decode_sequence_component(&value)?);
        }
    }
    let mut seq = Sequence::new(data_def);
    for c in components {
        seq.add_component(c);
    }
    Ok(seq)
}

/// Decode a Track (mob-slot) local-set.
pub fn decode_track(bytes: &[u8]) -> Result<Track> {
    let entries = klv::decode_local_set(bytes)?;
    let mut track_id: u32 = 0;
    let mut name = String::new();
    let mut edit_rate = EditRate::default();
    let mut origin = crate::timeline::Position::zero();
    let mut track_type = crate::composition::TrackType::Unknown;
    let mut physical_track_number: Option<u32> = None;
    let mut sequence: Option<Sequence> = None;
    for (key, value) in entries {
        if key == prop::SLOT_ID {
            track_id = dec_u32(&value);
        } else if key == prop::SLOT_NAME {
            name = dec_str(&value);
        } else if key == prop::SLOT_EDIT_RATE {
            edit_rate = dec_edit_rate(&value);
        } else if key == prop::SLOT_ORIGIN {
            origin = dec_position(&value);
        } else if key == prop::SLOT_TRACK_TYPE && !value.is_empty() {
            track_type = byte_to_track_type(value[0]);
        } else if key == prop::SLOT_PHYSICAL_TRACK_NUMBER {
            physical_track_number = Some(dec_u32(&value));
        } else if key == prop::SLOT_SEQUENCE {
            sequence = Some(decode_sequence(&value)?);
        }
    }
    let mut track = Track::new(track_id, name, edit_rate, track_type);
    track.origin = origin;
    track.physical_track_number = physical_track_number;
    if let Some(seq) = sequence {
        track.set_sequence(seq);
    }
    Ok(track)
}

/// Decode a single Mob local-set, dispatching on the embedded mob-type byte.
pub fn decode_mob(bytes: &[u8]) -> Result<DecodedMob> {
    let entries = klv::decode_local_set(bytes)?;
    let mut mob_type = MobType::Composition;
    let mut mob_id = uuid::Uuid::nil();
    let mut name = String::new();
    let mut tracks: Vec<Track> = Vec::new();
    let mut usage: Option<crate::composition::UsageCode> = None;
    let mut default_fade: Option<(i64, crate::composition::FadeType)> = None;
    for (key, value) in entries {
        if key == prop::MOB_TYPE && !value.is_empty() {
            mob_type = byte_to_mob_type(value[0]);
        } else if key == prop::MOB_ID {
            mob_id = dec_uuid(&value);
        } else if key == prop::MOB_NAME {
            name = dec_str(&value);
        } else if key == prop::MOB_SLOTS {
            tracks.push(decode_track(&value)?);
        } else if key == prop::MOB_USAGE_CODE && !value.is_empty() {
            usage = Some(byte_to_composition_usage(value[0]));
        } else if key == prop::MOB_DEFAULT_FADE && value.len() == 9 {
            let len = i64::from_le_bytes([
                value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
            ]);
            default_fade = Some((len, byte_to_fade_type(value[8])));
        }
    }
    match mob_type {
        MobType::Composition => {
            let mut comp = CompositionMob::new(mob_id, name);
            for t in tracks {
                comp.add_track(t);
            }
            if let Some(u) = usage {
                comp.set_usage_code(u);
            }
            if let Some((l, ft)) = default_fade {
                comp.set_default_fade(l, ft);
            }
            Ok(DecodedMob::Composition(Box::new(comp)))
        }
        _ => {
            let mut mob = Mob::new(mob_id, name, mob_type);
            for t in tracks {
                mob.add_slot(t.into_mob_slot());
            }
            Ok(DecodedMob::Other(Box::new(mob)))
        }
    }
}

/// Decode the ContentStorage local-set into a `ContentStorage`.
pub fn decode_content_storage(bytes: &[u8]) -> Result<ContentStorage> {
    let entries = klv::decode_local_set(bytes)?;
    let mut storage = ContentStorage::new();
    for (key, value) in entries {
        if key == prop::MOBS {
            let decoded = decode_mob(&value)?;
            match decoded {
                DecodedMob::Composition(c) => storage.add_composition_mob(*c),
                DecodedMob::Other(m) => storage.add_mob(*m),
            }
        }
    }
    Ok(storage)
}

/// Decode the EssenceData stream into a `Vec<EssenceData>`.
pub fn decode_essence_data_list(bytes: &[u8]) -> Result<Vec<EssenceData>> {
    let entries = klv::decode_local_set(bytes)?;
    let mut out = Vec::new();
    for (key, value) in entries {
        if key != prop::ESSENCE_DATA {
            continue;
        }
        let inner = klv::decode_local_set(&value)?;
        let mut mob_id = uuid::Uuid::nil();
        let mut data = Vec::new();
        for (ikey, ivalue) in inner {
            if ikey == prop::ESSENCE_MOB_ID {
                mob_id = dec_uuid(&ivalue);
            } else if ikey == prop::ESSENCE_DATA_BYTES {
                data = ivalue;
            }
        }
        out.push(EssenceData::new(mob_id, data));
    }
    Ok(out)
}

/// Decode an UnknownProperty bag previously encoded by
/// [`crate::local_set_encode::encode_unknown_properties`].
pub fn decode_unknown_properties(bytes: &[u8]) -> Result<HashMap<String, Vec<UnknownProperty>>> {
    let entries = klv::decode_local_set(bytes)?;
    let mut out: HashMap<String, Vec<UnknownProperty>> = HashMap::new();
    for (key, value) in entries {
        if key != UNKNOWN_PROP_AUID {
            continue;
        }
        if value.len() < 2 {
            continue;
        }
        let name_len = u16::from_le_bytes([value[0], value[1]]) as usize;
        if value.len() < 2 + name_len + 4 {
            continue;
        }
        let name = String::from_utf8_lossy(&value[2..2 + name_len]).into_owned();
        let off = 2 + name_len;
        let prop_count =
            u32::from_le_bytes([value[off], value[off + 1], value[off + 2], value[off + 3]])
                as usize;
        let mut cursor = off + 4;
        let mut props = Vec::with_capacity(prop_count);
        for _ in 0..prop_count {
            if cursor + 6 > value.len() {
                break;
            }
            let tag = u16::from_le_bytes([value[cursor], value[cursor + 1]]);
            let vlen = u32::from_le_bytes([
                value[cursor + 2],
                value[cursor + 3],
                value[cursor + 4],
                value[cursor + 5],
            ]) as usize;
            cursor += 6;
            if cursor + vlen > value.len() {
                break;
            }
            let payload = value[cursor..cursor + vlen].to_vec();
            cursor += vlen;
            props.push(UnknownProperty::new(tag, payload));
        }
        out.insert(name, props);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType};
    use crate::dictionary::Auid;
    use crate::klv;
    use crate::local_set_encode::{
        encode_mob_local_set, encode_unknown_properties, LocalSetEncode,
    };
    use crate::timeline::{EditRate, Position};
    use uuid::Uuid;

    #[test]
    fn header_round_trip() {
        let mut hdr = Header::new();
        hdr.major_version = 2;
        hdr.minor_version = 3;
        hdr.last_modified = 1_234_567_890;
        hdr.object_model_version = 5;
        hdr.operational_pattern = Auid::CLASS_COMPOSITION_MOB;
        hdr.essence_containers = vec![Auid::CLASS_HEADER, Auid::CLASS_SEQUENCE];
        let bytes = klv::encode_local_set(&hdr.encode_local_set());
        let DecodedHeader {
            header: hdr2,
            unknown,
        } = decode_header(&bytes).expect("decode");
        assert_eq!(hdr.major_version, hdr2.major_version);
        assert_eq!(hdr.minor_version, hdr2.minor_version);
        assert_eq!(hdr.last_modified, hdr2.last_modified);
        assert_eq!(hdr.object_model_version, hdr2.object_model_version);
        assert_eq!(hdr.operational_pattern, hdr2.operational_pattern);
        assert_eq!(hdr.essence_containers, hdr2.essence_containers);
        assert!(unknown.is_empty());
    }

    #[test]
    fn header_preserves_unknown_klvs() {
        let hdr = Header::new();
        let mut entries = hdr.encode_local_set();
        // Inject an unknown KLV
        let custom_key = Auid::from_bytes(&[
            0xAA, 0xFA, 0xFE, 0xED, 0x00, 0x00, 0x00, 0x01, 0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01,
            0x01, 0x01,
        ]);
        entries.push((custom_key, vec![1, 2, 3, 4, 5]));
        let bytes = klv::encode_local_set(&entries);
        let DecodedHeader { unknown, .. } = decode_header(&bytes).expect("decode");
        assert!(unknown
            .iter()
            .any(|(k, v)| *k == custom_key && v == &[1, 2, 3, 4, 5]));
    }

    #[test]
    fn track_round_trip() {
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            120,
            Position::new(10),
            Uuid::new_v4(),
            2,
        )));
        seq.add_component(SequenceComponent::Filler(Filler::new(30)));
        let mut track = Track::new(7, "V1", EditRate::FILM_24, TrackType::Picture);
        track.set_sequence(seq);
        track.origin = Position::new(5);
        track.physical_track_number = Some(3);

        let bytes = klv::encode_local_set(&track.encode_local_set());
        let recovered = decode_track(&bytes).expect("decode");
        assert_eq!(recovered.track_id, 7);
        assert_eq!(recovered.name, "V1");
        assert_eq!(recovered.edit_rate, EditRate::FILM_24);
        assert_eq!(recovered.origin, Position::new(5));
        assert_eq!(recovered.physical_track_number, Some(3));
        assert_eq!(recovered.track_type, TrackType::Picture);
        let sequence = recovered.sequence.expect("must have sequence");
        assert_eq!(sequence.components.len(), 2);
    }

    #[test]
    fn mob_round_trip() {
        let id = Uuid::new_v4();
        let mob = Mob::new(id, "source.mov".to_string(), MobType::Source);
        let bytes = klv::encode_local_set(&encode_mob_local_set(&mob));
        let recovered = decode_mob(&bytes).expect("decode");
        match recovered {
            DecodedMob::Other(m) => {
                assert_eq!(m.mob_id, id);
                assert_eq!(m.name, "source.mov");
                assert_eq!(m.mob_type, MobType::Source);
            }
            _ => panic!("expected Other mob"),
        }
    }

    #[test]
    fn composition_round_trip() {
        let id = Uuid::new_v4();
        let mut comp = CompositionMob::new(id, "MyEdit");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100,
            Position::new(0),
            Uuid::new_v4(),
            1,
        )));
        let mut track = Track::new(1, "Video", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);

        let bytes = klv::encode_local_set(&comp.encode_local_set());
        let recovered = decode_mob(&bytes).expect("decode");
        match recovered {
            DecodedMob::Composition(c) => {
                assert_eq!(c.mob_id(), id);
                assert_eq!(c.name(), "MyEdit");
                let tracks = c.tracks();
                assert_eq!(tracks.len(), 1);
            }
            _ => panic!("expected Composition"),
        }
    }

    #[test]
    fn unknown_property_bag_round_trip() {
        let mut bag = HashMap::new();
        bag.insert(
            "Header".to_string(),
            vec![
                UnknownProperty::new(0x1234, vec![0xDE, 0xAD, 0xBE, 0xEF]),
                UnknownProperty::new(0x5678, vec![1, 2, 3]),
            ],
        );
        bag.insert(
            "Mob/abc".to_string(),
            vec![UnknownProperty::new(0xFEED, vec![9, 9, 9, 9, 9])],
        );
        let entries = encode_unknown_properties(&bag);
        let bytes = klv::encode_local_set(&entries);
        let decoded = decode_unknown_properties(&bytes).expect("decode");
        assert_eq!(decoded.len(), bag.len());
        for (k, v) in &bag {
            let recovered = decoded.get(k).expect("key present");
            assert_eq!(recovered.len(), v.len());
            for (a, b) in v.iter().zip(recovered.iter()) {
                assert_eq!(a.tag, b.tag);
                assert_eq!(a.value, b.value);
            }
        }
    }
}
