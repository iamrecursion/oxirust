//! METS (Metadata Encoding and Transmission Standard) support
//!
//! METS is a standard for encoding descriptive, administrative, and structural metadata
//! for digital library objects.

use crate::Result;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// METS file section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetsFile {
    /// File ID
    pub id: String,
    /// File location
    pub location: PathBuf,
    /// MIME type
    pub mime_type: Option<String>,
    /// File size
    pub size: Option<u64>,
    /// Checksums
    pub checksums: Vec<(String, String)>,
}

/// METS structural map division
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetsDiv {
    /// Division ID
    pub id: String,
    /// Label
    pub label: Option<String>,
    /// Type
    pub type_: Option<String>,
    /// Order (1-based logical sequence number)
    pub order: Option<u32>,
    /// File pointers
    pub file_pointers: Vec<String>,
    /// Sub-divisions
    pub subdivisions: Vec<MetsDiv>,
}

impl MetsDiv {
    /// Create a new division with the given ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
            type_: None,
            order: None,
            file_pointers: Vec::new(),
            subdivisions: Vec::new(),
        }
    }

    /// Set the label for this division.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the type for this division.
    #[must_use]
    pub fn with_type(mut self, type_: impl Into<String>) -> Self {
        self.type_ = Some(type_.into());
        self
    }

    /// Set the order (logical sequence) for this division.
    #[must_use]
    pub const fn with_order(mut self, order: u32) -> Self {
        self.order = Some(order);
        self
    }

    /// Add a file pointer to this division.
    #[must_use]
    pub fn with_file_ptr(mut self, file_id: impl Into<String>) -> Self {
        self.file_pointers.push(file_id.into());
        self
    }

    /// Add a sub-division to this division.
    #[must_use]
    pub fn with_subdivision(mut self, div: MetsDiv) -> Self {
        self.subdivisions.push(div);
        self
    }
}

/// METS document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetsDocument {
    /// Object ID
    pub object_id: String,
    /// Title
    pub title: Option<String>,
    /// Files
    pub files: Vec<MetsFile>,
    /// Structural map
    pub struct_map: Option<MetsDiv>,
}

impl Default for MetsDocument {
    fn default() -> Self {
        Self::new("default-object")
    }
}

impl MetsDocument {
    /// Create a new METS document
    #[must_use]
    pub fn new(object_id: &str) -> Self {
        Self {
            object_id: object_id.to_string(),
            title: None,
            files: Vec::new(),
            struct_map: None,
        }
    }

    /// Set the title
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Add a file
    #[must_use]
    pub fn with_file(mut self, file: MetsFile) -> Self {
        self.files.push(file);
        self
    }

    /// Set structural map
    #[must_use]
    pub fn with_struct_map(mut self, struct_map: MetsDiv) -> Self {
        self.struct_map = Some(struct_map);
        self
    }

    /// Convert to XML
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails
    pub fn to_xml(&self) -> Result<String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<mets xmlns=\"http://www.loc.gov/METS/\" ");
        xml.push_str("xmlns:xlink=\"http://www.w3.org/1999/xlink\" ");
        xml.push_str(&format!("OBJID=\"{}\">\n", escape_xml(&self.object_id)));

        // Header
        xml.push_str("  <metsHdr>\n");
        if let Some(ref title) = self.title {
            xml.push_str(&format!(
                "    <altRecordID>{}</altRecordID>\n",
                escape_xml(title)
            ));
        }
        xml.push_str("  </metsHdr>\n");

        // File section
        if !self.files.is_empty() {
            xml.push_str("  <fileSec>\n");
            xml.push_str("    <fileGrp>\n");
            for file in &self.files {
                xml.push_str(&format!("      <file ID=\"{}\"", escape_xml(&file.id)));
                if let Some(ref mime) = file.mime_type {
                    xml.push_str(&format!(" MIMETYPE=\"{}\"", escape_xml(mime)));
                }
                if let Some(size) = file.size {
                    xml.push_str(&format!(" SIZE=\"{size}\""));
                }
                xml.push_str(">\n");

                for (algo, value) in &file.checksums {
                    xml.push_str(&format!(
                        "        <checksum CHECKSUMTYPE=\"{}\">{}</checksum>\n",
                        escape_xml(algo),
                        escape_xml(value)
                    ));
                }

                xml.push_str(&format!(
                    "        <FLocat LOCTYPE=\"URL\" xlink:href=\"{}\"/>\n",
                    escape_xml(&file.location.to_string_lossy())
                ));
                xml.push_str("      </file>\n");
            }
            xml.push_str("    </fileGrp>\n");
            xml.push_str("  </fileSec>\n");
        }

        // Structural map
        if let Some(ref div) = self.struct_map {
            xml.push_str("  <structMap>\n");
            xml.push_str(&render_div(div, 2)?);
            xml.push_str("  </structMap>\n");
        }

        xml.push_str("</mets>\n");
        Ok(xml)
    }

    /// Convert to XML using zero-copy streaming via `quick-xml` (preferred for large documents).
    ///
    /// Produces an identical document to `to_xml()` but avoids repeated string
    /// allocations by writing directly to an internal `Vec<u8>` buffer via the
    /// `quick-xml::Writer` streaming API.
    ///
    /// # Errors
    ///
    /// Returns an error if XML serialization fails.
    pub fn to_xml_streaming(&self) -> Result<String> {
        let buf = Vec::with_capacity(4096);
        let mut writer = Writer::new(Cursor::new(buf));

        // XML declaration
        writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;

        // <mets …>
        let mut mets_start = BytesStart::new("mets");
        mets_start.push_attribute(("xmlns", "http://www.loc.gov/METS/"));
        mets_start.push_attribute(("xmlns:xlink", "http://www.w3.org/1999/xlink"));
        mets_start.push_attribute(("OBJID", self.object_id.as_str()));
        writer.write_event(Event::Start(mets_start))?;

        // <metsHdr>
        writer.write_event(Event::Start(BytesStart::new("metsHdr")))?;
        if let Some(ref title) = self.title {
            writer.write_event(Event::Start(BytesStart::new("altRecordID")))?;
            writer.write_event(Event::Text(BytesText::new(title)))?;
            writer.write_event(Event::End(BytesEnd::new("altRecordID")))?;
        }
        writer.write_event(Event::End(BytesEnd::new("metsHdr")))?;

        // <fileSec> / <fileGrp>
        if !self.files.is_empty() {
            writer.write_event(Event::Start(BytesStart::new("fileSec")))?;
            writer.write_event(Event::Start(BytesStart::new("fileGrp")))?;
            for file in &self.files {
                let mut file_elem = BytesStart::new("file");
                file_elem.push_attribute(("ID", file.id.as_str()));
                if let Some(ref mime) = file.mime_type {
                    file_elem.push_attribute(("MIMETYPE", mime.as_str()));
                }
                if let Some(size) = file.size {
                    file_elem.push_attribute(("SIZE", size.to_string().as_str()));
                }
                writer.write_event(Event::Start(file_elem))?;

                for (algo, value) in &file.checksums {
                    let mut ck = BytesStart::new("checksum");
                    ck.push_attribute(("CHECKSUMTYPE", algo.as_str()));
                    writer.write_event(Event::Start(ck))?;
                    writer.write_event(Event::Text(BytesText::new(value)))?;
                    writer.write_event(Event::End(BytesEnd::new("checksum")))?;
                }

                let mut flocat = BytesStart::new("FLocat");
                flocat.push_attribute(("LOCTYPE", "URL"));
                flocat.push_attribute(("xlink:href", file.location.to_string_lossy().as_ref()));
                writer.write_event(Event::Empty(flocat))?;

                writer.write_event(Event::End(BytesEnd::new("file")))?;
            }
            writer.write_event(Event::End(BytesEnd::new("fileGrp")))?;
            writer.write_event(Event::End(BytesEnd::new("fileSec")))?;
        }

        // <structMap>
        if let Some(ref div) = self.struct_map {
            writer.write_event(Event::Start(BytesStart::new("structMap")))?;
            write_div_streaming(&mut writer, div)?;
            writer.write_event(Event::End(BytesEnd::new("structMap")))?;
        }

        writer.write_event(Event::End(BytesEnd::new("mets")))?;

        let result = writer.into_inner().into_inner();
        String::from_utf8(result)
            .map_err(|e| crate::Error::Metadata(format!("UTF-8 encoding error: {e}")))
    }

    /// Save to file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written
    pub fn save(&self, path: &Path) -> Result<()> {
        let xml = self.to_xml_streaming()?;
        fs::write(path, xml)?;
        Ok(())
    }
}

/// Write a METS `<div>` element tree using the quick-xml streaming API.
fn write_div_streaming<W: std::io::Write>(writer: &mut Writer<W>, div: &MetsDiv) -> Result<()> {
    let mut elem = BytesStart::new("div");
    elem.push_attribute(("ID", div.id.as_str()));
    if let Some(ref label) = div.label {
        elem.push_attribute(("LABEL", label.as_str()));
    }
    if let Some(ref type_) = div.type_ {
        elem.push_attribute(("TYPE", type_.as_str()));
    }
    if let Some(order) = div.order {
        elem.push_attribute(("ORDER", order.to_string().as_str()));
    }
    writer.write_event(Event::Start(elem))?;

    for fptr in &div.file_pointers {
        let mut fp = BytesStart::new("fptr");
        fp.push_attribute(("FILEID", fptr.as_str()));
        writer.write_event(Event::Empty(fp))?;
    }

    for subdiv in &div.subdivisions {
        write_div_streaming(writer, subdiv)?;
    }

    writer.write_event(Event::End(BytesEnd::new("div")))?;
    Ok(())
}

fn render_div(div: &MetsDiv, indent: usize) -> Result<String> {
    let mut xml = String::new();
    let spaces = " ".repeat(indent);

    xml.push_str(&format!("{}<div ID=\"{}\"", spaces, escape_xml(&div.id)));
    if let Some(ref label) = div.label {
        xml.push_str(&format!(" LABEL=\"{}\"", escape_xml(label)));
    }
    if let Some(ref type_) = div.type_ {
        xml.push_str(&format!(" TYPE=\"{}\"", escape_xml(type_)));
    }
    if let Some(order) = div.order {
        xml.push_str(&format!(" ORDER=\"{order}\""));
    }
    xml.push_str(">\n");

    for fptr in &div.file_pointers {
        xml.push_str(&format!(
            "{}  <fptr FILEID=\"{}\"/>\n",
            spaces,
            escape_xml(fptr)
        ));
    }

    for subdiv in &div.subdivisions {
        xml.push_str(&render_div(subdiv, indent + 2)?);
    }

    xml.push_str(&format!("{spaces}</div>\n"));
    Ok(xml)
}

/// METS builder
pub struct MetsBuilder {
    document: MetsDocument,
}

impl MetsBuilder {
    /// Create a new METS builder
    #[must_use]
    pub fn new(object_id: &str) -> Self {
        Self {
            document: MetsDocument::new(object_id),
        }
    }

    /// Set the title
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.document = self.document.with_title(title);
        self
    }

    /// Add a file
    #[must_use]
    pub fn add_file(mut self, id: &str, location: PathBuf, mime_type: Option<String>) -> Self {
        self.document = self.document.with_file(MetsFile {
            id: id.to_string(),
            location,
            mime_type,
            size: None,
            checksums: Vec::new(),
        });
        self
    }

    /// Set structural map
    #[must_use]
    pub fn with_struct_map(mut self, div: MetsDiv) -> Self {
        self.document = self.document.with_struct_map(div);
        self
    }

    /// Build the METS document
    #[must_use]
    pub fn build(self) -> MetsDocument {
        self.document
    }

    /// Build a complex structural map automatically from the registered files.
    ///
    /// Groups files by their MIME type family (video, audio, image, other)
    /// and creates a hierarchical `<div TYPE="ComplexObject">` with sub-divisions
    /// per group.
    #[must_use]
    pub fn build_with_auto_struct_map(mut self) -> MetsDocument {
        if !self.document.files.is_empty() {
            let root_div = build_auto_structural_map(&self.document.files);
            self.document = self.document.with_struct_map(root_div);
        }
        self.document
    }
}

/// Build a hierarchical structural map from a list of METS files.
///
/// Groups files into logical divisions:
/// - Video (`video/*`)
/// - Audio (`audio/*`)
/// - Image (`image/*`)
/// - Document (`application/*`, `text/*`)
/// - Other
fn build_auto_structural_map(files: &[MetsFile]) -> MetsDiv {
    use std::collections::HashMap;

    let mut groups: HashMap<&'static str, Vec<&MetsFile>> = HashMap::new();

    for file in files {
        let group = match file.mime_type.as_deref() {
            Some(m) if m.starts_with("video/") => "Video",
            Some(m) if m.starts_with("audio/") => "Audio",
            Some(m) if m.starts_with("image/") => "Image",
            Some(m) if m.starts_with("application/") || m.starts_with("text/") => "Document",
            _ => "Other",
        };
        groups.entry(group).or_default().push(file);
    }

    let mut root = MetsDiv::new("div-root")
        .with_type("ComplexObject")
        .with_label("Complete Digital Object");

    // Order groups for deterministic output
    let group_order = ["Video", "Audio", "Image", "Document", "Other"];
    let mut order_counter = 1u32;
    for group_name in &group_order {
        if let Some(group_files) = groups.get(group_name) {
            let div_id = format!("div-{}", group_name.to_lowercase());
            let mut group_div = MetsDiv::new(div_id)
                .with_type(*group_name)
                .with_label(*group_name)
                .with_order(order_counter);
            order_counter += 1;

            let mut file_order = 1u32;
            for file in group_files {
                let leaf_div = MetsDiv::new(format!("div-file-{}", file.id))
                    .with_type("File")
                    .with_label(file.location.to_string_lossy().as_ref())
                    .with_order(file_order)
                    .with_file_ptr(file.id.clone());
                file_order += 1;
                group_div = group_div.with_subdivision(leaf_div);
            }
            root = root.with_subdivision(group_div);
        }
    }
    root
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mets_document_creation() {
        let doc = MetsDocument::new("obj-001").with_title("Test Object");

        assert_eq!(doc.object_id, "obj-001");
        assert_eq!(doc.title, Some("Test Object".to_string()));
    }

    #[test]
    fn test_mets_with_files() {
        let doc = MetsBuilder::new("obj-001")
            .add_file(
                "file-1",
                PathBuf::from("video.mkv"),
                Some("video/x-matroska".to_string()),
            )
            .build();

        assert_eq!(doc.files.len(), 1);
        assert_eq!(doc.files[0].id, "file-1");
    }

    #[test]
    fn test_mets_xml() {
        let doc = MetsDocument::new("obj-001").with_title("Test");

        let xml = doc.to_xml().expect("operation should succeed");
        assert!(xml.contains("<mets"));
        assert!(xml.contains("obj-001"));
    }

    #[test]
    fn test_mets_structural_map_generation() {
        let doc = MetsBuilder::new("obj-struct-001")
            .with_title("Multi-file Object")
            .add_file(
                "vid-1",
                PathBuf::from("video.mkv"),
                Some("video/x-matroska".to_string()),
            )
            .add_file(
                "aud-1",
                PathBuf::from("audio.flac"),
                Some("audio/flac".to_string()),
            )
            .add_file(
                "img-1",
                PathBuf::from("thumbnail.png"),
                Some("image/png".to_string()),
            )
            .build_with_auto_struct_map();

        assert!(doc.struct_map.is_some(), "structural map must be set");
        let root = doc.struct_map.as_ref().expect("root div");
        assert_eq!(root.type_.as_deref(), Some("ComplexObject"));
        assert!(
            root.subdivisions.len() >= 3,
            "should have Video, Audio, Image divisions"
        );

        let types: Vec<_> = root
            .subdivisions
            .iter()
            .filter_map(|d| d.type_.as_deref())
            .collect();
        assert!(types.contains(&"Video"), "Video group missing");
        assert!(types.contains(&"Audio"), "Audio group missing");
        assert!(types.contains(&"Image"), "Image group missing");
    }

    #[test]
    fn test_mets_structural_map_with_order() {
        let div = MetsDiv::new("d1")
            .with_type("Video")
            .with_order(1)
            .with_file_ptr("file-1");
        assert_eq!(div.order, Some(1));
        assert_eq!(div.file_pointers.len(), 1);
    }

    #[test]
    fn test_mets_xml_streaming_matches_content() {
        let doc = MetsBuilder::new("stream-001")
            .with_title("Streaming Test")
            .add_file(
                "f1",
                PathBuf::from("test.mkv"),
                Some("video/x-matroska".to_string()),
            )
            .build();

        let xml = doc
            .to_xml_streaming()
            .expect("streaming XML should succeed");
        assert!(xml.contains("stream-001"), "OBJID must be present");
        assert!(xml.contains("Streaming Test"), "title must be present");
        assert!(xml.contains("test.mkv"), "file location must be present");
        assert!(
            xml.contains("video/x-matroska"),
            "mime type must be present"
        );
    }

    #[test]
    fn test_mets_structural_map_nested_subdivisions() {
        let leaf1 = MetsDiv::new("leaf-1")
            .with_type("Scene")
            .with_order(1)
            .with_file_ptr("f1");
        let leaf2 = MetsDiv::new("leaf-2")
            .with_type("Scene")
            .with_order(2)
            .with_file_ptr("f2");
        let chapter = MetsDiv::new("chapter-1")
            .with_type("Chapter")
            .with_label("Chapter 1")
            .with_order(1)
            .with_subdivision(leaf1)
            .with_subdivision(leaf2);
        let root = MetsDiv::new("root")
            .with_type("ComplexObject")
            .with_subdivision(chapter);

        let doc = MetsDocument::new("nested-obj").with_struct_map(root);
        let xml = doc.to_xml().expect("xml generation");
        assert!(xml.contains("chapter-1"));
        assert!(xml.contains("leaf-1"));
        assert!(xml.contains("leaf-2"));
    }

    #[test]
    fn test_mets_save_and_reload() {
        let temp_dir = tempfile::TempDir::new().expect("temp dir");
        let out_path = temp_dir.path().join("mets.xml");

        let doc = MetsBuilder::new("save-001")
            .with_title("Save Test")
            .add_file("f-1", PathBuf::from("video.mkv"), None)
            .build();

        doc.save(&out_path).expect("save should succeed");
        assert!(out_path.exists(), "saved file must exist");

        let content = std::fs::read_to_string(&out_path).expect("read back");
        assert!(content.contains("save-001"), "OBJID must round-trip");
        assert!(content.contains("Save Test"), "title must round-trip");
    }
}
