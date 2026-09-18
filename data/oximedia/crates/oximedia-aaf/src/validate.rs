//! AAF validation pass
//!
//! Performs structural integrity checks on an AAF `ContentStorage`:
//! - All mob source-clip references must resolve to known mobs
//! - Composition mobs must have at least one track
//! - Required properties (mob name) must be present
//! - No orphan tracks (tracks with no parent mob)
//!
//! Returns a `ValidationReport` with categorised `errors` and `warnings`.

use crate::composition::SequenceComponent;
use crate::ContentStorage;
use uuid::Uuid;

/// Severity level of a validation issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Hard error — the AAF is malformed
    Error,
    /// Warning — the AAF may still function but has potential issues
    Warning,
}

/// A single validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity
    pub severity: IssueSeverity,
    /// Human-readable message
    pub message: String,
    /// Mob UUID context (if applicable)
    pub mob_id: Option<Uuid>,
}

impl ValidationIssue {
    fn error(message: impl Into<String>, mob_id: Option<Uuid>) -> Self {
        Self {
            severity: IssueSeverity::Error,
            message: message.into(),
            mob_id,
        }
    }

    fn warning(message: impl Into<String>, mob_id: Option<Uuid>) -> Self {
        Self {
            severity: IssueSeverity::Warning,
            message: message.into(),
            mob_id,
        }
    }
}

/// Report produced by `AafValidator::validate`
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Hard errors
    pub errors: Vec<ValidationIssue>,
    /// Warnings
    pub warnings: Vec<ValidationIssue>,
}

impl ValidationReport {
    fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Whether the validation passed with no errors
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Total number of issues (errors + warnings)
    #[must_use]
    pub fn issue_count(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }

    fn add_error(&mut self, msg: impl Into<String>, mob_id: Option<Uuid>) {
        self.errors.push(ValidationIssue::error(msg, mob_id));
    }

    fn add_warning(&mut self, msg: impl Into<String>, mob_id: Option<Uuid>) {
        self.warnings.push(ValidationIssue::warning(msg, mob_id));
    }
}

/// Validator for AAF ContentStorage
pub struct AafValidator;

impl AafValidator {
    /// Run a full validation pass on the content storage.
    ///
    /// Checks performed:
    /// 1. Composition mobs must have a non-empty name
    /// 2. Composition mobs must have at least one track
    /// 3. All source-clip mob references must resolve
    /// 4. Mobs with no tracks are warned about
    /// 5. Empty compositions (all tracks have no components) are warned about
    #[must_use]
    pub fn validate(storage: &ContentStorage) -> ValidationReport {
        let mut report = ValidationReport::new();

        Self::validate_composition_mobs(storage, &mut report);
        Self::validate_mob_references(storage, &mut report);
        Self::validate_master_mobs(storage, &mut report);

        report
    }

    fn validate_composition_mobs(storage: &ContentStorage, report: &mut ValidationReport) {
        for comp_mob in storage.composition_mobs() {
            let mob_id = comp_mob.mob_id();

            // Check name is not empty
            if comp_mob.name().trim().is_empty() {
                report.add_error(
                    "CompositionMob has an empty name — required property missing",
                    Some(mob_id),
                );
            }

            let tracks = comp_mob.tracks();

            // Check has tracks
            if tracks.is_empty() {
                report.add_warning(
                    format!("CompositionMob '{}' has no tracks", comp_mob.name()),
                    Some(mob_id),
                );
                continue;
            }

            // Check for tracks with no sequence
            let mut has_content = false;
            for track in &tracks {
                match &track.sequence {
                    None => {
                        report.add_warning(
                            format!(
                                "Track '{}' (id={}) in mob '{}' has no sequence",
                                track.name,
                                track.track_id,
                                comp_mob.name()
                            ),
                            Some(mob_id),
                        );
                    }
                    Some(seq) => {
                        if !seq.components.is_empty() {
                            has_content = true;
                        }
                    }
                }
            }

            if !has_content {
                report.add_warning(
                    format!(
                        "CompositionMob '{}' has tracks but all sequences are empty",
                        comp_mob.name()
                    ),
                    Some(mob_id),
                );
            }
        }
    }

    fn validate_mob_references(storage: &ContentStorage, report: &mut ValidationReport) {
        for comp_mob in storage.composition_mobs() {
            let mob_id = comp_mob.mob_id();

            for track in comp_mob.tracks() {
                let Some(ref sequence) = track.sequence else {
                    continue;
                };

                for component in &sequence.components {
                    let SequenceComponent::SourceClip(clip) = component else {
                        continue;
                    };

                    let ref_id = clip.source_mob_id;

                    // A null UUID is a special "no reference" case — skip
                    if ref_id == Uuid::nil() {
                        continue;
                    }

                    // Reference must resolve in either composition mobs or generic mobs
                    let resolves_in_comp = storage.find_composition_mob(&ref_id).is_some();
                    let resolves_in_mob = storage.find_mob(&ref_id).is_some();

                    if !resolves_in_comp && !resolves_in_mob {
                        report.add_error(
                            format!(
                                "SourceClip in mob '{}' track '{}' references unknown mob {}",
                                comp_mob.name(),
                                track.name,
                                ref_id
                            ),
                            Some(mob_id),
                        );
                    }
                }
            }
        }
    }

    fn validate_master_mobs(storage: &ContentStorage, report: &mut ValidationReport) {
        for mob in storage.master_mobs() {
            if mob.name().trim().is_empty() {
                report.add_warning("MasterMob has an empty name", Some(mob.mob_id()));
            }

            if mob.slots().is_empty() {
                report.add_warning(
                    format!("MasterMob '{}' has no slots (orphan mob)", mob.name()),
                    Some(mob.mob_id()),
                );
            }
        }

        for mob in storage.source_mobs() {
            if mob.slots().is_empty() {
                report.add_warning(
                    format!("SourceMob '{}' has no slots (orphan mob)", mob.name()),
                    Some(mob.mob_id()),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Sequence, SequenceComponent, SourceClip, Track, TrackType,
    };
    use crate::dictionary::Auid;
    use crate::object_model::{Mob, MobType};
    use crate::timeline::{EditRate, Position};
    use crate::ContentStorage;
    use uuid::Uuid;

    fn make_valid_storage() -> ContentStorage {
        let mut storage = ContentStorage::new();
        let source_mob_id = Uuid::new_v4();

        // Register a source mob so references resolve
        storage.add_mob(Mob::new(
            source_mob_id,
            "SourceFile.mov".to_string(),
            MobType::Source,
        ));

        let mut comp = CompositionMob::new(Uuid::new_v4(), "Valid Composition");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100,
            Position::zero(),
            source_mob_id,
            1,
        )));
        let mut track = Track::new(1, "Video", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        storage
    }

    #[test]
    fn test_valid_storage_passes() {
        let storage = make_valid_storage();
        let report = AafValidator::validate(&storage);
        assert!(
            report.is_valid(),
            "Expected no errors, got: {:?}",
            report.errors
        );
    }

    #[test]
    fn test_empty_name_is_error() {
        let mut storage = ContentStorage::new();
        let comp = CompositionMob::new(Uuid::new_v4(), ""); // empty name
        storage.add_composition_mob(comp);
        let report = AafValidator::validate(&storage);
        assert!(!report.is_valid());
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.message.contains("empty name")),
            "Expected an empty-name error"
        );
    }

    #[test]
    fn test_no_tracks_is_warning() {
        let mut storage = ContentStorage::new();
        let comp = CompositionMob::new(Uuid::new_v4(), "Empty Comp");
        storage.add_composition_mob(comp);
        let report = AafValidator::validate(&storage);
        // no tracks → warning
        assert!(report.is_valid()); // warnings only, no errors
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_unresolved_mob_ref_is_error() {
        let mut storage = ContentStorage::new();
        let bogus_source = Uuid::new_v4(); // not registered

        let mut comp = CompositionMob::new(Uuid::new_v4(), "Broken Comp");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            50,
            Position::zero(),
            bogus_source,
            1,
        )));
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        let report = AafValidator::validate(&storage);
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|e| e.message.contains("unknown mob")));
    }

    #[test]
    fn test_nil_uuid_ref_not_error() {
        let mut storage = ContentStorage::new();
        let mut comp = CompositionMob::new(Uuid::new_v4(), "NilRef Comp");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            50,
            Position::zero(),
            Uuid::nil(), // nil UUID = intentional "no ref"
            1,
        )));
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        let report = AafValidator::validate(&storage);
        // Nil UUID should not produce an error
        assert!(
            report.is_valid(),
            "nil UUID references should not be errors"
        );
    }

    #[test]
    fn test_empty_sequences_warning() {
        let mut storage = ContentStorage::new();
        let mut comp = CompositionMob::new(Uuid::new_v4(), "EmptyTracks");
        let seq = Sequence::new(Auid::PICTURE); // empty sequence
        let mut track = Track::new(1, "V", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        storage.add_composition_mob(comp);

        let report = AafValidator::validate(&storage);
        assert!(report.is_valid());
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_issue_count() {
        let mut storage = ContentStorage::new();
        // Add a composition with empty name (error) and no tracks (warning)
        let comp = CompositionMob::new(Uuid::new_v4(), "");
        storage.add_composition_mob(comp);
        let report = AafValidator::validate(&storage);
        assert!(report.issue_count() >= 2);
    }

    #[test]
    fn test_orphan_master_mob_warning() {
        let mut storage = ContentStorage::new();
        // Master mob with no slots
        storage.add_mob(Mob::new(
            Uuid::new_v4(),
            "OrphanMaster".to_string(),
            MobType::Master,
        ));
        let report = AafValidator::validate(&storage);
        assert!(report.is_valid()); // only a warning
        assert!(report.warnings.iter().any(|w| w.message.contains("orphan")));
    }
}
