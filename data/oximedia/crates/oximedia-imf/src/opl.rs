//! Output Profile List (OPL) - SMPTE ST 2067-8
//!
//! The OPL defines output requirements and device constraints for playing
//! an IMF composition.

use crate::{ImfError, ImfResult};
use chrono::{DateTime, Utc};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use uuid::Uuid;

/// Device constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConstraint {
    constraint_type: String,
    value: String,
    min_value: Option<String>,
    max_value: Option<String>,
}

impl DeviceConstraint {
    /// Create a new device constraint
    pub fn new(constraint_type: String, value: String) -> Self {
        Self {
            constraint_type,
            value,
            min_value: None,
            max_value: None,
        }
    }

    /// Get constraint type
    pub fn constraint_type(&self) -> &str {
        &self.constraint_type
    }

    /// Get value
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Get minimum value
    pub fn min_value(&self) -> Option<&str> {
        self.min_value.as_deref()
    }

    /// Set minimum value
    pub fn set_min_value(&mut self, value: String) {
        self.min_value = Some(value);
    }

    /// Get maximum value
    pub fn max_value(&self) -> Option<&str> {
        self.max_value.as_deref()
    }

    /// Set maximum value
    pub fn set_max_value(&mut self, value: String) {
        self.max_value = Some(value);
    }

    /// Builder pattern: with range
    pub fn with_range(mut self, min: String, max: String) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }
}

/// Device constraints collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConstraints {
    constraints: Vec<DeviceConstraint>,
}

impl DeviceConstraints {
    /// Create a new device constraints collection
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Get constraints
    pub fn constraints(&self) -> &[DeviceConstraint] {
        &self.constraints
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: DeviceConstraint) {
        self.constraints.push(constraint);
    }

    /// Find constraint by type
    pub fn find_constraint(&self, constraint_type: &str) -> Option<&DeviceConstraint> {
        self.constraints
            .iter()
            .find(|c| c.constraint_type == constraint_type)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get constraint count
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Create common video constraints
    pub fn video_constraints() -> Self {
        let mut constraints = Self::new();

        // Maximum resolution
        constraints.add_constraint(DeviceConstraint::new(
            "MaxResolution".to_string(),
            "3840x2160".to_string(),
        ));

        // Maximum frame rate
        constraints.add_constraint(DeviceConstraint::new(
            "MaxFrameRate".to_string(),
            "60".to_string(),
        ));

        // Color depth
        constraints.add_constraint(DeviceConstraint::new(
            "ColorDepth".to_string(),
            "10".to_string(),
        ));

        constraints
    }

    /// Create common audio constraints
    pub fn audio_constraints() -> Self {
        let mut constraints = Self::new();

        // Sample rate
        constraints.add_constraint(DeviceConstraint::new(
            "SampleRate".to_string(),
            "48000".to_string(),
        ));

        // Bit depth
        constraints.add_constraint(DeviceConstraint::new(
            "BitDepth".to_string(),
            "24".to_string(),
        ));

        // Maximum channels
        constraints.add_constraint(DeviceConstraint::new(
            "MaxChannels".to_string(),
            "16".to_string(),
        ));

        constraints
    }
}

impl Default for DeviceConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// Output profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputProfile {
    id: Uuid,
    annotation: Option<String>,
    profile_name: String,
    profile_version: String,
    device_constraints: DeviceConstraints,
    composition_playlist_id: Option<Uuid>,
    properties: HashMap<String, String>,
}

impl OutputProfile {
    /// Create a new output profile
    pub fn new(id: Uuid, profile_name: String, profile_version: String) -> Self {
        Self {
            id,
            annotation: None,
            profile_name,
            profile_version,
            device_constraints: DeviceConstraints::new(),
            composition_playlist_id: None,
            properties: HashMap::new(),
        }
    }

    /// Get profile ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get annotation
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_deref()
    }

    /// Set annotation
    pub fn set_annotation(&mut self, annotation: String) {
        self.annotation = Some(annotation);
    }

    /// Get profile name
    pub fn profile_name(&self) -> &str {
        &self.profile_name
    }

    /// Get profile version
    pub fn profile_version(&self) -> &str {
        &self.profile_version
    }

    /// Get device constraints
    pub fn device_constraints(&self) -> &DeviceConstraints {
        &self.device_constraints
    }

    /// Get mutable device constraints
    pub fn device_constraints_mut(&mut self) -> &mut DeviceConstraints {
        &mut self.device_constraints
    }

    /// Get composition playlist ID
    pub fn composition_playlist_id(&self) -> Option<Uuid> {
        self.composition_playlist_id
    }

    /// Set composition playlist ID
    pub fn set_composition_playlist_id(&mut self, id: Uuid) {
        self.composition_playlist_id = Some(id);
    }

    /// Get properties
    pub fn properties(&self) -> &HashMap<String, String> {
        &self.properties
    }

    /// Get property value
    pub fn get_property(&self, key: &str) -> Option<&str> {
        self.properties.get(key).map(String::as_str)
    }

    /// Set property
    pub fn set_property(&mut self, key: String, value: String) {
        self.properties.insert(key, value);
    }

    /// Builder pattern: with annotation
    pub fn with_annotation(mut self, annotation: String) -> Self {
        self.annotation = Some(annotation);
        self
    }

    /// Builder pattern: with CPL ID
    pub fn with_cpl_id(mut self, id: Uuid) -> Self {
        self.composition_playlist_id = Some(id);
        self
    }
}

/// Output Profile List (OPL) - SMPTE ST 2067-8
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputProfileList {
    id: Uuid,
    annotation: Option<String>,
    issue_date: DateTime<Utc>,
    issuer: Option<String>,
    creator: Option<String>,
    profiles: Vec<OutputProfile>,
}

impl OutputProfileList {
    /// Create a new output profile list
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            annotation: None,
            issue_date: Utc::now(),
            issuer: None,
            creator: None,
            profiles: Vec::new(),
        }
    }

    /// Get OPL ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get annotation
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_deref()
    }

    /// Set annotation
    pub fn set_annotation(&mut self, annotation: String) {
        self.annotation = Some(annotation);
    }

    /// Get issue date
    pub fn issue_date(&self) -> DateTime<Utc> {
        self.issue_date
    }

    /// Set issue date
    pub fn set_issue_date(&mut self, date: DateTime<Utc>) {
        self.issue_date = date;
    }

    /// Get issuer
    pub fn issuer(&self) -> Option<&str> {
        self.issuer.as_deref()
    }

    /// Set issuer
    pub fn set_issuer(&mut self, issuer: String) {
        self.issuer = Some(issuer);
    }

    /// Get creator
    pub fn creator(&self) -> Option<&str> {
        self.creator.as_deref()
    }

    /// Set creator
    pub fn set_creator(&mut self, creator: String) {
        self.creator = Some(creator);
    }

    /// Get profiles
    pub fn profiles(&self) -> &[OutputProfile] {
        &self.profiles
    }

    /// Add a profile
    pub fn add_profile(&mut self, profile: OutputProfile) {
        self.profiles.push(profile);
    }

    /// Find a profile by ID
    pub fn find_profile(&self, id: Uuid) -> Option<&OutputProfile> {
        self.profiles.iter().find(|p| p.id == id)
    }

    /// Remove a profile by ID
    pub fn remove_profile(&mut self, id: Uuid) -> Option<OutputProfile> {
        if let Some(pos) = self.profiles.iter().position(|p| p.id == id) {
            Some(self.profiles.remove(pos))
        } else {
            None
        }
    }

    /// Find profiles by CPL ID
    pub fn find_profiles_for_cpl(&self, cpl_id: Uuid) -> Vec<&OutputProfile> {
        self.profiles
            .iter()
            .filter(|p| p.composition_playlist_id == Some(cpl_id))
            .collect()
    }

    /// Parse OPL from XML
    pub fn from_xml<R: BufRead>(reader: R) -> ImfResult<Self> {
        OplParser::parse(reader)
    }

    /// Write OPL to XML
    pub fn to_xml<W: Write>(&self, writer: W) -> ImfResult<()> {
        OplWriter::write(self, writer)
    }
}

/// OPL XML parser
struct OplParser;

impl OplParser {
    #[allow(clippy::too_many_lines)]
    fn parse<R: BufRead>(reader: R) -> ImfResult<OutputProfileList> {
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut text_buffer = String::new();

        // OPL fields
        let mut id: Option<Uuid> = None;
        let mut annotation: Option<String> = None;
        let mut issue_date: Option<DateTime<Utc>> = None;
        let mut issuer: Option<String> = None;
        let mut creator: Option<String> = None;
        let mut profiles: Vec<OutputProfile> = Vec::new();

        // State for parsing profiles
        let mut in_profile = false;
        let mut current_profile_id: Option<Uuid> = None;
        let mut current_profile_name: Option<String> = None;
        let mut current_profile_version: Option<String> = None;

        loop {
            match xml_reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let element_name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    text_buffer.clear();

                    if element_name == "OutputProfile" {
                        in_profile = true;
                        current_profile_id = None;
                        current_profile_name = None;
                        current_profile_version = None;
                    }
                }
                Ok(Event::Text(e)) => {
                    text_buffer = String::from_utf8_lossy(e.as_ref()).to_string();
                }
                Ok(Event::End(e)) => {
                    let element_name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if in_profile {
                        match element_name.as_str() {
                            "Id" => {
                                current_profile_id = Some(Self::parse_uuid(&text_buffer)?);
                            }
                            "ProfileName" => {
                                current_profile_name = Some(text_buffer.clone());
                            }
                            "ProfileVersion" => {
                                current_profile_version = Some(text_buffer.clone());
                            }
                            "OutputProfile" => {
                                // Build profile
                                let profile_id = current_profile_id.take().ok_or_else(|| {
                                    ImfError::MissingElement("Profile Id".to_string())
                                })?;
                                let profile_name =
                                    current_profile_name.take().ok_or_else(|| {
                                        ImfError::MissingElement("ProfileName".to_string())
                                    })?;
                                let profile_version = current_profile_version
                                    .take()
                                    .unwrap_or_else(|| "1.0".to_string());

                                let profile =
                                    OutputProfile::new(profile_id, profile_name, profile_version);
                                profiles.push(profile);
                                in_profile = false;
                            }
                            _ => {}
                        }
                    } else {
                        // Top-level elements
                        match element_name.as_str() {
                            "Id" => id = Some(Self::parse_uuid(&text_buffer)?),
                            "Annotation" => annotation = Some(text_buffer.clone()),
                            "IssueDate" => {
                                issue_date = Some(
                                    DateTime::parse_from_rfc3339(&text_buffer)
                                        .map_err(|e| {
                                            ImfError::InvalidStructure(format!(
                                                "Invalid IssueDate: {e}"
                                            ))
                                        })?
                                        .with_timezone(&Utc),
                                );
                            }
                            "Issuer" => issuer = Some(text_buffer.clone()),
                            "Creator" => creator = Some(text_buffer.clone()),
                            _ => {}
                        }
                    }

                    text_buffer.clear();
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(ImfError::XmlError(format!("XML parse error: {e}"))),
                _ => {}
            }
            buf.clear();
        }

        // Build OPL
        let id = id.ok_or_else(|| ImfError::MissingElement("Id".to_string()))?;

        let mut opl = OutputProfileList::new(id);
        opl.annotation = annotation;
        opl.issue_date = issue_date.unwrap_or_else(Utc::now);
        opl.issuer = issuer;
        opl.creator = creator;
        opl.profiles = profiles;

        Ok(opl)
    }

    fn parse_uuid(s: &str) -> ImfResult<Uuid> {
        // Handle URN format: urn:uuid:xxxxx
        let uuid_str = s.trim().strip_prefix("urn:uuid:").unwrap_or(s);
        Uuid::parse_str(uuid_str).map_err(|e| ImfError::InvalidUuid(e.to_string()))
    }
}

/// OPL XML writer
struct OplWriter;

impl OplWriter {
    fn write<W: Write>(opl: &OutputProfileList, writer: W) -> ImfResult<()> {
        let mut xml_writer = Writer::new_with_indent(writer, b' ', 2);

        // XML declaration
        xml_writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        // Root element
        let mut root = BytesStart::new("OutputProfileList");
        root.push_attribute(("xmlns", "http://www.smpte-ra.org/schemas/2067-8/2013"));
        xml_writer
            .write_event(Event::Start(root))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        // Write fields
        Self::write_element(&mut xml_writer, "Id", &format!("urn:uuid:{}", opl.id))?;

        if let Some(ref annotation) = opl.annotation {
            Self::write_element(&mut xml_writer, "Annotation", annotation)?;
        }

        Self::write_element(&mut xml_writer, "IssueDate", &opl.issue_date.to_rfc3339())?;

        if let Some(ref issuer) = opl.issuer {
            Self::write_element(&mut xml_writer, "Issuer", issuer)?;
        }

        if let Some(ref creator) = opl.creator {
            Self::write_element(&mut xml_writer, "Creator", creator)?;
        }

        // Profiles
        Self::write_profiles(&mut xml_writer, &opl.profiles)?;

        // Close root
        xml_writer
            .write_event(Event::End(BytesEnd::new("OutputProfileList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }

    fn write_element<W: Write>(writer: &mut Writer<W>, name: &str, content: &str) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new(name)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::Text(BytesText::new(content)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::End(BytesEnd::new(name)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        Ok(())
    }

    fn write_profiles<W: Write>(
        writer: &mut Writer<W>,
        profiles: &[OutputProfile],
    ) -> ImfResult<()> {
        for profile in profiles {
            Self::write_profile(writer, profile)?;
        }
        Ok(())
    }

    fn write_profile<W: Write>(writer: &mut Writer<W>, profile: &OutputProfile) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("OutputProfile")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Self::write_element(writer, "Id", &format!("urn:uuid:{}", profile.id))?;

        if let Some(ref annotation) = profile.annotation {
            Self::write_element(writer, "Annotation", annotation)?;
        }

        Self::write_element(writer, "ProfileName", &profile.profile_name)?;
        Self::write_element(writer, "ProfileVersion", &profile.profile_version)?;

        if let Some(cpl_id) = profile.composition_playlist_id {
            Self::write_element(
                writer,
                "CompositionPlaylistId",
                &format!("urn:uuid:{cpl_id}"),
            )?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("OutputProfile")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_constraint() {
        let constraint =
            DeviceConstraint::new("MaxResolution".to_string(), "3840x2160".to_string())
                .with_range("1920x1080".to_string(), "3840x2160".to_string());

        assert_eq!(constraint.constraint_type(), "MaxResolution");
        assert_eq!(constraint.value(), "3840x2160");
        assert_eq!(constraint.min_value(), Some("1920x1080"));
        assert_eq!(constraint.max_value(), Some("3840x2160"));
    }

    #[test]
    fn test_device_constraints() {
        let mut constraints = DeviceConstraints::new();
        constraints.add_constraint(DeviceConstraint::new(
            "MaxFrameRate".to_string(),
            "60".to_string(),
        ));

        assert_eq!(constraints.len(), 1);
        assert!(constraints.find_constraint("MaxFrameRate").is_some());
        assert!(constraints.find_constraint("MinFrameRate").is_none());
    }

    #[test]
    fn test_output_profile() {
        let mut profile = OutputProfile::new(
            Uuid::new_v4(),
            "IMF Main Profile".to_string(),
            "1.0".to_string(),
        );

        profile.set_annotation("Test profile".to_string());

        let cpl_id = Uuid::new_v4();
        profile.set_composition_playlist_id(cpl_id);

        assert_eq!(profile.profile_name(), "IMF Main Profile");
        assert_eq!(profile.profile_version(), "1.0");
        assert_eq!(profile.composition_playlist_id(), Some(cpl_id));
    }

    #[test]
    fn test_output_profile_list() {
        let mut opl = OutputProfileList::new(Uuid::new_v4());
        opl.set_creator("OxiMedia".to_string());
        opl.set_issuer("Test Studio".to_string());

        let profile = OutputProfile::new(
            Uuid::new_v4(),
            "Test Profile".to_string(),
            "1.0".to_string(),
        );
        let profile_id = profile.id();

        opl.add_profile(profile);

        assert_eq!(opl.creator(), Some("OxiMedia"));
        assert_eq!(opl.profiles().len(), 1);
        assert!(opl.find_profile(profile_id).is_some());
    }

    #[test]
    fn test_video_constraints() {
        let constraints = DeviceConstraints::video_constraints();
        assert!(!constraints.is_empty());
        assert!(constraints.find_constraint("MaxResolution").is_some());
        assert!(constraints.find_constraint("MaxFrameRate").is_some());
    }

    #[test]
    fn test_audio_constraints() {
        let constraints = DeviceConstraints::audio_constraints();
        assert!(!constraints.is_empty());
        assert!(constraints.find_constraint("SampleRate").is_some());
        assert!(constraints.find_constraint("BitDepth").is_some());
    }
}
