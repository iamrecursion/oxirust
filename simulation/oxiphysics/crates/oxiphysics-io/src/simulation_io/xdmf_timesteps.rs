// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Reader for the time steps of an XDMF file.
//!
//! The XML is tokenised with the pure-Rust `oxixml-xml` pull parser, so the
//! result does not depend on how the file is laid out: attributes may be
//! written in any order, in single or double quotes and spread over several
//! lines, tags may share a line, and comments, CDATA sections and the
//! entities declared in the `DOCTYPE` (`<!ENTITY HeavyData "run.h5">`, as in
//! many files written by Xdmf2 tools) are handled.

use std::collections::HashMap;

use oxixml_xml::XmlVersion;
use oxixml_xml::escape::resolve_predefined_entity;
use oxixml_xml::events::{BytesRef, BytesStart, Event};
use oxixml_xml::reader::Reader;

use crate::{Error, Result};

/// Parse an XDMF document and return the `(time, data_file)` pair of every
/// time step.
///
/// A time step is a *uniform* `<Grid>` (a grid without a `GridType` attribute
/// is uniform, as the XDMF specification says; `GridType="Collection"`,
/// `"Tree"` and `"Subset"` grids only group other grids).  Its time is the
/// `Value` of a `<Time>` child element, or else the grid's own `Time`
/// attribute, or else the time of the nearest enclosing grid (leaf grids of a
/// spatial collection usually carry no time of their own); a uniform grid
/// without any time is not a time step and is skipped.  Its data file is the
/// file part of the text of the first heavy-data `DataItem` inside the grid,
/// i.e. `data.h5` for `<DataItem Format="HDF">data.h5:/pressure/0</DataItem>`
/// (`Format="Binary"` items name the file directly); it is the empty string
/// when the grid keeps its data inline (`Format="XML"`).  The pairs are
/// returned in document order.
///
/// This function is lenient: if the document is malformed the steps that were
/// completed before the first error are returned.  Use
/// [`try_read_xdmf_timesteps`] to get the error instead.
pub fn read_xdmf_timesteps(xml: &str) -> Vec<(f64, String)> {
    let (scan, _error) = scan_xdmf(xml);
    scan.steps()
}

/// Like [`read_xdmf_timesteps`], but fails on a malformed document instead of
/// returning the steps found before the error.
///
/// # Errors
///
/// [`Error::Parse`] when the XML is not well formed (including a document that
/// ends inside an element or uses an entity that is not declared) or a grid
/// attribute cannot be decoded.
pub fn try_read_xdmf_timesteps(xml: &str) -> Result<Vec<(f64, String)>> {
    let (scan, error) = scan_xdmf(xml);
    match error {
        Some(error) => Err(error),
        None => Ok(scan.steps()),
    }
}

/// The file part of an XDMF heavy-data reference `file.h5:/group/dataset`.
///
/// The dataset path of a reference is absolute inside the file, so the file
/// name ends at the first `:/`; a Windows drive prefix (`C:/data/run.h5:/x`,
/// `C:\data\run.h5:/x`) is not mistaken for that separator.  A reference with
/// a relative dataset name (`run.h5:group/x`) is split at its last `:`, and a
/// text without a separator is taken as the file name as it is.
fn hdf_file_name(reference: &str) -> &str {
    let bytes = reference.as_bytes();
    let drive_letter = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    let search_from = if drive_letter { 2 } else { 0 };
    if let Some(offset) = reference[search_from..].find(":/") {
        return &reference[..search_from + offset];
    }
    match reference.rfind(':') {
        Some(index) if !(drive_letter && index == 1) => &reference[..index],
        _ => reference,
    }
}

/// A finite time written as text.
fn parse_time(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok().filter(|t| t.is_finite())
}

/// One `<Grid>` element of the document.
struct GridNode {
    /// Index of the enclosing grid, if any.
    parent: Option<usize>,
    /// Whether the grid is a uniform grid (a leaf that holds data).
    uniform: bool,
    /// Time from a `<Time Value="..."/>` child.
    time_element: Option<f64>,
    /// Time from a `Time="..."` attribute of the grid itself.
    time_attribute: Option<f64>,
    /// File of the first heavy-data `DataItem`.
    file: Option<String>,
    /// Whether the closing tag has been seen.
    closed: bool,
}

/// An element that is currently open.
enum Element {
    Grid,
    /// A `DataItem`; `capturing` when its text is the file reference of the
    /// enclosing grid.
    DataItem {
        capturing: bool,
    },
    Other,
}

/// The text of the heavy-data `DataItem` being read.
struct Capture {
    text: String,
    /// `true` for `Format="HDF"` (the text is `file:/dataset`), `false` for
    /// `Format="Binary"` (the text is the file name).
    hdf: bool,
}

#[derive(Default)]
struct XdmfScan {
    grids: Vec<GridNode>,
    /// Indices into `grids` of the grids that are open, innermost last.
    open_grids: Vec<usize>,
    stack: Vec<Element>,
    capture: Option<Capture>,
    /// Entities declared in the internal subset of the `DOCTYPE`.
    entities: HashMap<String, String>,
}

impl XdmfScan {
    /// Resolve an entity name: the predefined XML entities, then those declared
    /// in the `DOCTYPE`.
    fn resolve_entity(&self, name: &str) -> Option<&str> {
        resolve_predefined_entity(name).or_else(|| self.entities.get(name).map(String::as_str))
    }

    /// The attributes of a start tag as decoded `(name, value)` pairs.
    fn attributes(&self, element: &BytesStart<'_>) -> Result<Vec<(String, String)>> {
        let mut pairs = Vec::new();
        for attribute in element.attributes() {
            let attribute = attribute
                .map_err(|err| Error::Parse(format!("XDMF: malformed XML attribute: {err}")))?;
            let value = attribute
                .normalized_value_with(XmlVersion::Implicit1_0, 1, |name| self.resolve_entity(name))
                .map_err(|err| Error::Parse(format!("XDMF: invalid attribute value: {err}")))?;
            pairs.push((
                String::from_utf8_lossy(attribute.key.as_ref()).into_owned(),
                value.into_owned(),
            ));
        }
        Ok(pairs)
    }

    fn start(&mut self, element: &BytesStart<'_>) -> Result<()> {
        let kind = match element.local_name().as_ref() {
            b"Grid" => {
                self.open_grid(element)?;
                Element::Grid
            }
            b"Time" => {
                self.read_time(element)?;
                Element::Other
            }
            b"DataItem" => Element::DataItem {
                capturing: self.begin_data_item(element)?,
            },
            _ => Element::Other,
        };
        self.stack.push(kind);
        Ok(())
    }

    fn end(&mut self) {
        match self.stack.pop() {
            Some(Element::Grid) => {
                if let Some(index) = self.open_grids.pop() {
                    self.grids[index].closed = true;
                }
            }
            Some(Element::DataItem { capturing: true }) => self.finish_data_item(),
            _ => {}
        }
    }

    fn open_grid(&mut self, element: &BytesStart<'_>) -> Result<()> {
        let attributes = self.attributes(element)?;
        let attribute = |key: &str| {
            attributes
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
        };
        let uniform = attribute("GridType").is_none_or(|kind| kind.eq_ignore_ascii_case("Uniform"));
        let time_attribute = attribute("Time").and_then(parse_time);
        self.grids.push(GridNode {
            parent: self.open_grids.last().copied(),
            uniform,
            time_element: None,
            time_attribute,
            file: None,
            closed: false,
        });
        self.open_grids.push(self.grids.len() - 1);
        Ok(())
    }

    /// A `<Time Value="..."/>` element directly inside a grid.
    fn read_time(&mut self, element: &BytesStart<'_>) -> Result<()> {
        if !matches!(self.stack.last(), Some(Element::Grid)) {
            return Ok(());
        }
        let attributes = self.attributes(element)?;
        let time = attributes
            .iter()
            .find(|(name, _)| name == "Value")
            .and_then(|(_, value)| parse_time(value));
        if let (Some(time), Some(&index)) = (time, self.open_grids.last()) {
            self.grids[index].time_element = Some(time);
        }
        Ok(())
    }

    /// Start of a `DataItem`; returns whether its text has to be captured.
    fn begin_data_item(&mut self, element: &BytesStart<'_>) -> Result<bool> {
        let Some(&grid) = self.open_grids.last() else {
            return Ok(false);
        };
        if !self.grids[grid].uniform || self.grids[grid].file.is_some() || self.capture.is_some() {
            return Ok(false);
        }
        let attributes = self.attributes(element)?;
        let format = attributes
            .iter()
            .find(|(name, _)| name == "Format")
            .map_or("XML", |(_, value)| value.as_str());
        let hdf = format.eq_ignore_ascii_case("HDF") || format.eq_ignore_ascii_case("HDF5");
        if !hdf && !format.eq_ignore_ascii_case("Binary") {
            return Ok(false);
        }
        self.capture = Some(Capture {
            text: String::new(),
            hdf,
        });
        Ok(true)
    }

    fn finish_data_item(&mut self) {
        let (Some(capture), Some(&grid)) = (self.capture.take(), self.open_grids.last()) else {
            return;
        };
        let text = capture.text.trim();
        let file = if capture.hdf {
            hdf_file_name(text)
        } else {
            text
        };
        if !file.is_empty() {
            self.grids[grid].file = Some(file.to_string());
        }
    }

    fn push_text(&mut self, text: &str) {
        if let Some(capture) = self.capture.as_mut() {
            capture.text.push_str(text);
        }
    }

    /// An entity or character reference in character data.
    fn reference(&mut self, reference: &BytesRef<'_>) -> Result<()> {
        if self.capture.is_none() {
            return Ok(());
        }
        if reference.is_char_ref() {
            let character = reference
                .resolve_char_ref()
                .map_err(|err| Error::Parse(format!("XDMF: invalid character reference: {err}")))?;
            if let Some(character) = character {
                self.push_text(character.encode_utf8(&mut [0u8; 4]));
            }
            return Ok(());
        }
        let name = reference
            .decode()
            .map_err(|err| Error::Parse(format!("XDMF: invalid entity name: {err}")))?;
        let replacement = self
            .resolve_entity(&name)
            .ok_or_else(|| Error::Parse(format!("XDMF: undeclared entity '&{name};'")))?
            .to_string();
        self.push_text(&replacement);
        Ok(())
    }

    /// Record the `<!ENTITY name "value">` declarations of a `DOCTYPE`.
    fn declare_entities(&mut self, doctype: &str) {
        let mut rest = doctype;
        while let Some(position) = rest.find("<!ENTITY") {
            rest = &rest[position + "<!ENTITY".len()..];
            let declaration = rest.trim_start();
            let Some(name_end) = declaration.find(char::is_whitespace) else {
                break;
            };
            let name = &declaration[..name_end];
            let value_part = declaration[name_end..].trim_start();
            let Some(quote) = value_part
                .chars()
                .next()
                .filter(|c| *c == '"' || *c == '\'')
            else {
                // A parameter entity (`%`) or an external one (`SYSTEM`): not a literal.
                continue;
            };
            let body = &value_part[1..];
            let Some(end) = body.find(quote) else {
                break;
            };
            if !name.is_empty() && !name.starts_with('%') {
                self.entities
                    .entry(name.to_string())
                    .or_insert_with(|| body[..end].to_string());
            }
            rest = &body[end + 1..];
        }
    }

    /// The time steps: closed uniform grids that have a time of their own or
    /// inherit one from an enclosing grid.
    fn steps(&self) -> Vec<(f64, String)> {
        self.grids
            .iter()
            .enumerate()
            .filter(|(_, grid)| grid.uniform && grid.closed)
            .filter_map(|(index, grid)| {
                let time = self.time_of(index)?;
                Some((time, grid.file.clone().unwrap_or_default()))
            })
            .collect()
    }

    fn time_of(&self, mut index: usize) -> Option<f64> {
        loop {
            let grid = &self.grids[index];
            if let Some(time) = grid.time_element.or(grid.time_attribute) {
                return Some(time);
            }
            index = grid.parent?;
        }
    }
}

/// Run the pull parser over `xml`; returns what was collected and the first
/// error, if any.
fn scan_xdmf(xml: &str) -> (XdmfScan, Option<Error>) {
    let mut reader = Reader::from_str(xml);
    let mut scan = XdmfScan::default();
    loop {
        let event = match reader.read_event() {
            Ok(event) => event,
            Err(err) => {
                let error = Error::Parse(format!(
                    "XDMF XML error at byte {}: {err}",
                    reader.error_position()
                ));
                return (scan, Some(error));
            }
        };
        let outcome = match event {
            Event::Eof => {
                if scan.stack.is_empty() {
                    break;
                }
                Err(Error::Parse(format!(
                    "XDMF: unexpected end of document, {} element(s) still open",
                    scan.stack.len()
                )))
            }
            Event::Start(element) => scan.start(&element),
            Event::Empty(element) => scan.start(&element).map(|()| scan.end()),
            Event::End(_) => {
                scan.end();
                Ok(())
            }
            Event::Text(text) => text
                .decode()
                .map(|text| scan.push_text(&text))
                .map_err(|err| Error::Parse(format!("XDMF: invalid text encoding: {err}"))),
            Event::CData(text) => text
                .decode()
                .map(|text| scan.push_text(&text))
                .map_err(|err| Error::Parse(format!("XDMF: invalid text encoding: {err}"))),
            Event::GeneralRef(reference) => scan.reference(&reference),
            Event::DocType(doctype) => doctype
                .decode()
                .map(|text| scan.declare_entities(&text))
                .map_err(|err| Error::Parse(format!("XDMF: invalid DOCTYPE encoding: {err}"))),
            _ => Ok(()),
        };
        if let Err(error) = outcome {
            return (scan, Some(error));
        }
    }
    (scan, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiphysics_core::math::Vec3;

    fn step(time: f64, file: &str) -> (f64, String) {
        (time, file.to_string())
    }

    #[test]
    fn hdf_reference_names_the_file_before_the_dataset_path() {
        assert_eq!(hdf_file_name("data.h5:/geometry"), "data.h5");
        assert_eq!(
            hdf_file_name("out/run 1/data.h5:/fields/u"),
            "out/run 1/data.h5"
        );
        assert_eq!(hdf_file_name("C:/x/f.h5:/coords"), "C:/x/f.h5");
        assert_eq!(hdf_file_name("C:\\x\\f.h5:/coords"), "C:\\x\\f.h5");
        assert_eq!(hdf_file_name("f.h5:/a:b"), "f.h5");
        assert_eq!(hdf_file_name("rel/f.h5:group/x"), "rel/f.h5");
        assert_eq!(hdf_file_name("C:\\x\\f.h5:group"), "C:\\x\\f.h5");
        assert_eq!(hdf_file_name("C:\\f.h5"), "C:\\f.h5");
        assert_eq!(hdf_file_name("f.h5"), "f.h5");
        assert_eq!(hdf_file_name("f.h5:"), "f.h5");
        assert_eq!(hdf_file_name(""), "");
    }

    #[test]
    fn a_single_grid_with_an_hdf_data_item() {
        let xml = r#"<Grid Name="step_0" GridType="Uniform"><Time Value="0.5"/><Geometry GeometryType="XYZ"><DataItem Format="HDF" Dimensions="4 3">data.h5:/xyz</DataItem></Geometry></Grid>"#;
        assert_eq!(read_xdmf_timesteps(xml), vec![step(0.5, "data.h5")]);
        assert_eq!(
            try_read_xdmf_timesteps(xml).expect("well formed"),
            vec![step(0.5, "data.h5")]
        );
    }

    #[test]
    fn time_and_file_are_found_in_a_realistic_multi_line_temporal_collection() {
        let xml = r#"<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
      <Grid
          Name="mesh"
          GridType="Uniform">
        <Time
            Value="0.0" />
        <Topology TopologyType="Triangle" NumberOfElements="2">
          <DataItem Format="HDF" Dimensions="2 3">
            run.h5:/mesh/topology
          </DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Format="HDF" Dimensions="4 3">run.h5:/mesh/geometry</DataItem>
        </Geometry>
      </Grid>
      <Grid Name="mesh" GridType="Uniform">
        <Time Value="0.25"></Time>
        <Attribute Name="p" AttributeType="Scalar" Center="Node">
          <DataItem Format="HDF" Dimensions="4">run.h5:/p/1</DataItem>
        </Attribute>
      </Grid>
      <Grid Name="mesh" GridType='Uniform'>
        <Time Value='1.5e-1' TimeType='Single'/>
        <Attribute Name="p"><DataItem Format="HDF">run2.h5:/p/2</DataItem></Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>"#;
        assert_eq!(
            read_xdmf_timesteps(xml),
            vec![
                step(0.0, "run.h5"),
                step(0.25, "run.h5"),
                step(0.15, "run2.h5")
            ]
        );
    }

    #[test]
    fn inline_xml_data_gives_an_empty_file_name() {
        let xml = r#"<Xdmf><Domain><Grid GridType="Uniform"><Time Value="2"/>
            <Geometry><DataItem Format="XML" Dimensions="1 3">0 0 0</DataItem></Geometry>
            </Grid></Domain></Xdmf>"#;
        assert_eq!(read_xdmf_timesteps(xml), vec![step(2.0, "")]);
    }

    #[test]
    fn binary_data_items_name_their_file_directly() {
        let xml = r#"<Grid><Time Value="1"/><DataItem Format="Binary" Dimensions="9">geometry.bin</DataItem></Grid>"#;
        assert_eq!(read_xdmf_timesteps(xml), vec![step(1.0, "geometry.bin")]);
    }

    #[test]
    fn a_grid_without_grid_type_is_uniform_and_collections_are_not_steps() {
        let xml = r#"<Domain>
            <Grid GridType="Collection" CollectionType="Temporal">
              <Grid Name="a"><Time Value="1"/></Grid>
            </Grid>
            <Grid GridType="Tree"><Time Value="9"/><Grid Name="leaf"><Time Value="3"/></Grid></Grid>
            </Domain>"#;
        assert_eq!(read_xdmf_timesteps(xml), vec![step(1.0, ""), step(3.0, "")]);
    }

    #[test]
    fn grids_without_a_time_are_skipped_and_leaves_inherit_the_time_of_their_collection() {
        let xml = r#"<Domain>
            <Grid Name="static" GridType="Uniform"/>
            <Grid GridType="Collection" CollectionType="Temporal">
              <Grid GridType="Collection" CollectionType="Spatial">
                <Time Value="0.1"/>
                <Grid GridType="Uniform"><DataItem Format="HDF">part0.h5:/x</DataItem></Grid>
                <Grid GridType="Uniform"><DataItem Format="HDF">part1.h5:/x</DataItem></Grid>
              </Grid>
              <Grid GridType="Collection" CollectionType="Spatial">
                <Grid GridType="Uniform"><Time Value="0.2"/><DataItem Format="HDF">part0.h5:/y</DataItem></Grid>
              </Grid>
            </Grid>
            </Domain>"#;
        assert_eq!(
            read_xdmf_timesteps(xml),
            vec![
                step(0.1, "part0.h5"),
                step(0.1, "part1.h5"),
                step(0.2, "part0.h5")
            ]
        );
    }

    #[test]
    fn the_time_element_wins_over_the_time_attribute_whatever_their_order() {
        let before = r#"<Grid Time="9"><Time Value="1"/></Grid>"#;
        assert_eq!(read_xdmf_timesteps(before), vec![step(1.0, "")]);
        let only_attribute = r#"<Grid Time="9"/>"#;
        assert_eq!(read_xdmf_timesteps(only_attribute), vec![step(9.0, "")]);
        // A <Time> that follows the data is still found.
        let after = r#"<Grid><DataItem Format="HDF">a.h5:/x</DataItem><Time Value="4"/></Grid>"#;
        assert_eq!(read_xdmf_timesteps(after), vec![step(4.0, "a.h5")]);
    }

    #[test]
    fn only_the_first_heavy_data_item_of_a_grid_gives_the_file() {
        let xml = r#"<Grid><Time Value="1"/>
            <Topology><DataItem Format="HDF">first.h5:/t</DataItem></Topology>
            <Geometry><DataItem Format="HDF">second.h5:/g</DataItem></Geometry></Grid>"#;
        assert_eq!(read_xdmf_timesteps(xml), vec![step(1.0, "first.h5")]);
    }

    #[test]
    fn comments_cdata_and_entities_are_handled() {
        let xml = r#"<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [
  <!ENTITY HeavyData "Output.h5">
  <!ENTITY T1 '0.75'>
]>
<Xdmf>
  <!-- <Grid><Time Value="99"/></Grid> -->
  <Grid Name="a &amp; b"><Time Value="&T1;"/>
    <DataItem Format="HDF">&HeavyData;:/XYZ</DataItem>
  </Grid>
  <Grid><Time Value="2"/><DataItem Format="HDF"><![CDATA[odd name.h5]]>:/data</DataItem></Grid>
</Xdmf>"#;
        assert_eq!(
            read_xdmf_timesteps(xml),
            vec![step(0.75, "Output.h5"), step(2.0, "odd name.h5")]
        );
    }

    #[test]
    fn a_malformed_document_gives_the_completed_steps_leniently_and_an_error_strictly() {
        // The second grid is never closed.
        let xml = r#"<Xdmf><Grid><Time Value="1"/></Grid><Grid><Time Value="2"/>"#;
        assert_eq!(read_xdmf_timesteps(xml), vec![step(1.0, "")]);
        assert!(matches!(try_read_xdmf_timesteps(xml), Err(Error::Parse(_))));

        let mismatched = r#"<Xdmf><Grid><Time Value="1"/></Grid><Grid></Xdmf>"#;
        assert_eq!(read_xdmf_timesteps(mismatched), vec![step(1.0, "")]);
        assert!(try_read_xdmf_timesteps(mismatched).is_err());

        let undeclared =
            r#"<Grid><Time Value="1"/><DataItem Format="HDF">&Nope;:/x</DataItem></Grid>"#;
        let error = try_read_xdmf_timesteps(undeclared).expect_err("undeclared entity");
        assert!(error.to_string().contains("Nope"), "{error}");
    }

    #[test]
    fn empty_and_non_xdmf_input_has_no_steps() {
        assert!(read_xdmf_timesteps("").is_empty());
        assert!(read_xdmf_timesteps("plain text").is_empty());
        assert!(read_xdmf_timesteps("<Xdmf/>").is_empty());
        assert_eq!(try_read_xdmf_timesteps("").expect("empty is fine"), vec![]);
    }

    #[test]
    fn unusable_times_are_ignored() {
        let xml = r#"<Domain><Grid><Time Value="soon"/></Grid><Grid><Time Value="NaN"/></Grid>
            <Grid><Time Value="inf"/></Grid><Grid><Time Value=" 5 "/></Grid></Domain>"#;
        assert_eq!(read_xdmf_timesteps(xml), vec![step(5.0, "")]);
    }

    #[test]
    fn times_written_by_write_xdmf_temporal_are_read_back_exactly() {
        let frame = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0)];
        let times = [
            0.0,
            0.1,
            0.30000000000000004,
            1e-9,
            12345.678901234567,
            2.5e17,
        ];
        let timesteps: Vec<(f64, &[Vec3])> = times.iter().map(|&t| (t, &frame[..])).collect();
        let mut buffer = Vec::new();
        crate::xdmf::write_xdmf_temporal(&mut buffer, &timesteps).expect("write");
        let xml = String::from_utf8(buffer).expect("utf-8");
        let steps = try_read_xdmf_timesteps(&xml).expect("own output parses");
        assert_eq!(steps.len(), times.len());
        for ((time, file), expected) in steps.iter().zip(times) {
            assert_eq!(time.to_bits(), expected.to_bits());
            // The particle positions are written inline.
            assert_eq!(file, "");
        }
    }

    #[test]
    fn xdmf_with_hdf_references_from_write_xdmf_with_attributes_has_no_time() {
        // A static mesh file: grids without a time are not time steps.
        let path =
            std::env::temp_dir().join(format!("oxiphysics_xdmf_steps_{}.xmf", std::process::id()));
        crate::xdmf::write_xdmf_with_attributes(
            path.to_str().expect("utf-8 path"),
            "mesh.h5",
            4,
            2,
            "Triangle",
            &[],
        )
        .expect("write");
        let xml = std::fs::read_to_string(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(read_xdmf_timesteps(&xml).is_empty());
        assert!(xml.contains("mesh.h5:/coordinates"));
    }
}
