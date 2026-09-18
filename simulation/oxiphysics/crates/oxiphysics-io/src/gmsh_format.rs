// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Gmsh mesh format (.msh) version 2.2 ASCII support.
//!
//! Implements reading and writing of the Gmsh 2.2 ASCII `.msh` format,
//! including node coordinates, element connectivity, and physical groups.
//!
//! The reader validates its input and reports every problem as
//! [`Error::Parse`]: files in another MSH version (notably
//! the 4.x layout that current Gmsh writes by default) or in the binary
//! encoding are rejected up front, and malformed sections never panic.

use std::collections::HashMap;
use std::ops::Range;

use crate::{Error, Result};

// ── Data types ────────────────────────────────────────────────────────────────

/// A node in a Gmsh mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct GmshNode {
    /// 1-based node number as stored in the file.
    pub id: usize,
    /// Node coordinates `[x, y, z]`.
    pub coords: [f64; 3],
}

/// Gmsh element types (subset used in practice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmshElementType {
    /// 2-node line.
    Line2 = 1,
    /// 3-node triangle.
    Triangle3 = 2,
    /// 4-node quadrilateral.
    Quad4 = 3,
    /// 4-node tetrahedron.
    Tet4 = 4,
    /// 8-node hexahedron.
    Hex8 = 5,
    /// 1-node point.
    Point1 = 15,
}

impl GmshElementType {
    /// Try to construct from the integer type code.
    pub fn from_code(code: usize) -> Option<Self> {
        match code {
            1 => Some(Self::Line2),
            2 => Some(Self::Triangle3),
            3 => Some(Self::Quad4),
            4 => Some(Self::Tet4),
            5 => Some(Self::Hex8),
            15 => Some(Self::Point1),
            _ => None,
        }
    }

    /// Number of nodes for this element type.
    pub fn node_count(self) -> usize {
        match self {
            Self::Point1 => 1,
            Self::Line2 => 2,
            Self::Triangle3 => 3,
            Self::Quad4 => 4,
            Self::Tet4 => 4,
            Self::Hex8 => 8,
        }
    }
}

/// A single Gmsh mesh element.
#[derive(Debug, Clone, PartialEq)]
pub struct GmshElement {
    /// 1-based element number.
    pub id: usize,
    /// Element type code.
    pub element_type: GmshElementType,
    /// Physical group tag (first tag in the tag list, 0 if absent).
    pub physical_tag: usize,
    /// Geometric entity tag (second tag, 0 if absent).
    pub geometric_tag: usize,
    /// 1-based node connectivity.
    pub nodes: Vec<usize>,
}

/// A physical group: a named set of elements sharing a tag.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalGroup {
    /// Spatial dimension (0–3).
    pub dimension: usize,
    /// Integer tag matching `GmshElement::physical_tag`.
    pub tag: usize,
    /// Human-readable name.
    pub name: String,
}

/// A complete Gmsh 2.2 mesh.
#[derive(Debug, Clone, Default)]
pub struct GmshMesh {
    /// Node list (1-based IDs preserved).
    pub nodes: Vec<GmshNode>,
    /// Element list.
    pub elements: Vec<GmshElement>,
    /// Physical groups declared in `$PhysicalNames`.
    pub physical_groups: Vec<PhysicalGroup>,
}

impl GmshMesh {
    /// Create an empty Gmsh mesh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the node with the given 1-based ID, or `None`.
    pub fn node_by_id(&self, id: usize) -> Option<&GmshNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Count of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Count of elements.
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }
}

/// Generic FEM mesh connectivity.
#[derive(Debug, Clone, Default)]
pub struct FemMesh {
    /// Node coordinates `[x, y, z]` in node-index order (0-based).
    pub coords: Vec<[f64; 3]>,
    /// Element connectivity as lists of 0-based node indices.
    pub connectivity: Vec<Vec<usize>>,
}

// ── Parsing ───────────────────────────────────────────────────────────────────

/// Build the error for a problem found on (1-based) line `line_no`.
fn parse_error(line_no: usize, message: impl std::fmt::Display) -> Error {
    Error::Parse(format!("Gmsh MSH line {line_no}: {message}"))
}

/// The sections `read_gmsh_v2` interprets; every other section is skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SectionKind {
    MeshFormat,
    PhysicalNames,
    Nodes,
    Elements,
}

impl SectionKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "MeshFormat" => Some(Self::MeshFormat),
            "PhysicalNames" => Some(Self::PhysicalNames),
            "Nodes" => Some(Self::Nodes),
            "Elements" => Some(Self::Elements),
            _ => None,
        }
    }
}

/// One `$Name` ... `$EndName` block located inside the line array.
struct Section {
    /// 1-based line number of the opening `$Name` line.
    start_line: usize,
    /// Indices (into the line array) of the lines between the markers.
    body: Range<usize>,
    /// Index of the first line after the `$EndName` marker.
    next: usize,
}

/// Locate the block opened by the `$name` marker at `lines[start]`.
fn read_section(lines: &[&str], start: usize, name: &str) -> Result<Section> {
    let end_marker = format!("$End{name}");
    let body_start = start + 1;
    match lines
        .get(body_start..)
        .and_then(|rest| rest.iter().position(|line| line.trim() == end_marker))
    {
        Some(offset) => Ok(Section {
            start_line: start + 1,
            body: body_start..body_start + offset,
            next: body_start + offset + 1,
        }),
        None => Err(parse_error(
            start + 1,
            format!("section ${name} is not terminated by {end_marker}"),
        )),
    }
}

/// The non-blank, trimmed lines of `body` together with their 1-based line numbers.
fn records<'a>(
    lines: &'a [&'a str],
    body: Range<usize>,
) -> impl Iterator<Item = (usize, &'a str)> + 'a {
    body.filter_map(move |index| {
        let text = lines.get(index)?.trim();
        (!text.is_empty()).then_some((index + 1, text))
    })
}

/// Parse the leading record-count line shared by `$PhysicalNames`, `$Nodes`
/// and `$Elements`.
fn read_count<'a>(
    section_name: &str,
    section: &Section,
    recs: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<usize> {
    let (line_no, text) = recs.next().ok_or_else(|| {
        parse_error(
            section.start_line,
            format!("section ${section_name} lacks its record-count line"),
        )
    })?;
    text.parse::<usize>().map_err(|_| {
        parse_error(
            line_no,
            format!("expected the {section_name} record count, found '{text}'"),
        )
    })
}

/// Fail unless exactly `declared` records were read.
fn check_declared_count(
    section_name: &str,
    section: &Section,
    declared: usize,
    found: usize,
) -> Result<()> {
    if declared == found {
        Ok(())
    } else {
        Err(parse_error(
            section.start_line,
            format!("section ${section_name} declares {declared} records but contains {found}"),
        ))
    }
}

/// Accept only the ASCII 2.x flavour of the format.
fn check_mesh_format(lines: &[&str], section: &Section) -> Result<()> {
    let mut recs = records(lines, section.body.clone());
    let (line_no, text) = recs
        .next()
        .ok_or_else(|| parse_error(section.start_line, "section $MeshFormat is empty"))?;
    let mut fields = text.split_whitespace();
    let version = fields.next().unwrap_or_default();
    let file_type = fields.next().ok_or_else(|| {
        parse_error(
            line_no,
            format!("$MeshFormat line '{text}' lacks the file-type field"),
        )
    })?;
    if !version.starts_with("2.") {
        return Err(parse_error(
            line_no,
            format!(
                "unsupported MSH version '{version}': only the 2.x ASCII format (e.g. 2.2) can be read; \
                 re-export the mesh with `Mesh.MshFileVersion = 2.2` (gmsh -format msh2)"
            ),
        ));
    }
    match file_type.parse::<u32>() {
        Ok(0) => {}
        Ok(_) => {
            return Err(parse_error(
                line_no,
                format!(
                    "binary MSH files (file-type {file_type}) are not supported; only ASCII (0)"
                ),
            ));
        }
        Err(_) => {
            return Err(parse_error(
                line_no,
                format!("invalid $MeshFormat file-type '{file_type}'"),
            ));
        }
    }
    if let Some(data_size) = fields.next()
        && data_size.parse::<usize>().is_err()
    {
        return Err(parse_error(
            line_no,
            format!("invalid $MeshFormat data-size '{data_size}'"),
        ));
    }
    Ok(())
}

/// Split `text` into its first whitespace-delimited token and the rest.
fn split_token(text: &str) -> Option<(&str, &str)> {
    let text = text.trim_start();
    let end = text.find(char::is_whitespace).unwrap_or(text.len());
    if end == 0 {
        return None;
    }
    Some((&text[..end], text[end..].trim_start()))
}

fn read_physical_names(lines: &[&str], section: &Section, mesh: &mut GmshMesh) -> Result<()> {
    let mut recs = records(lines, section.body.clone());
    let declared = read_count("PhysicalNames", section, &mut recs)?;
    let mut found = 0usize;
    for (line_no, text) in recs {
        let malformed = || {
            parse_error(
                line_no,
                format!("expected `<dimension> <tag> \"<name>\"`, found '{text}'"),
            )
        };
        let (dim_text, rest) = split_token(text).ok_or_else(malformed)?;
        let (tag_text, name_text) = split_token(rest).ok_or_else(malformed)?;
        let dimension = dim_text.parse::<usize>().map_err(|_| malformed())?;
        let tag = tag_text.parse::<usize>().map_err(|_| malformed())?;
        if name_text.is_empty() {
            return Err(malformed());
        }
        mesh.physical_groups.push(PhysicalGroup {
            dimension,
            tag,
            name: name_text.trim_matches('"').to_string(),
        });
        found += 1;
    }
    check_declared_count("PhysicalNames", section, declared, found)
}

fn read_nodes(lines: &[&str], section: &Section, mesh: &mut GmshMesh) -> Result<()> {
    let mut recs = records(lines, section.body.clone());
    let declared = read_count("Nodes", section, &mut recs)?;
    mesh.nodes.reserve(section.body.len());
    let mut found = 0usize;
    for (line_no, text) in recs {
        let fields: Vec<&str> = text.split_whitespace().collect();
        let [id_text, x_text, y_text, z_text] = fields.as_slice() else {
            return Err(parse_error(
                line_no,
                format!(
                    "expected `<node-number> <x> <y> <z>` (4 fields), found {} fields",
                    fields.len()
                ),
            ));
        };
        let id = id_text
            .parse::<usize>()
            .map_err(|_| parse_error(line_no, format!("invalid node number '{id_text}'")))?;
        let mut coords = [0.0_f64; 3];
        for (slot, coord_text) in coords.iter_mut().zip([x_text, y_text, z_text]) {
            *slot = coord_text.parse::<f64>().map_err(|_| {
                parse_error(
                    line_no,
                    format!("invalid coordinate '{coord_text}' for node {id}"),
                )
            })?;
        }
        mesh.nodes.push(GmshNode { id, coords });
        found += 1;
    }
    check_declared_count("Nodes", section, declared, found)
}

fn read_elements(lines: &[&str], section: &Section, mesh: &mut GmshMesh) -> Result<()> {
    let mut recs = records(lines, section.body.clone());
    let declared = read_count("Elements", section, &mut recs)?;
    mesh.elements.reserve(section.body.len());
    let mut found = 0usize;
    for (line_no, text) in recs {
        found += 1;
        let fields: Vec<&str> = text.split_whitespace().collect();
        let parse_field = |index: usize, what: &str| -> Result<usize> {
            let field = fields.get(index).copied().ok_or_else(|| {
                parse_error(line_no, format!("element line is missing its {what}"))
            })?;
            field
                .parse::<usize>()
                .map_err(|_| parse_error(line_no, format!("invalid {what} '{field}'")))
        };
        let id = parse_field(0, "element number")?;
        let type_code = parse_field(1, "element type")?;
        let n_tags = parse_field(2, "tag count")?;
        // `n_tags` comes straight from the file: never use it unchecked as an index.
        let tag_start = 3usize;
        let node_start = tag_start.checked_add(n_tags).ok_or_else(|| {
            parse_error(
                line_no,
                format!("element {id}: tag count {n_tags} is out of range"),
            )
        })?;
        let Some(node_fields) = fields.get(node_start..) else {
            return Err(parse_error(
                line_no,
                format!(
                    "element {id} declares {n_tags} tags but the line has only {} fields after \
                     the element header",
                    fields.len().saturating_sub(tag_start)
                ),
            ));
        };
        // Element types this module cannot represent are skipped, not guessed at.
        let Some(element_type) = GmshElementType::from_code(type_code) else {
            continue;
        };
        let physical_tag = if n_tags >= 1 {
            parse_field(tag_start, "physical tag")?
        } else {
            0
        };
        let geometric_tag = if n_tags >= 2 {
            parse_field(tag_start + 1, "elementary tag")?
        } else {
            0
        };
        let nodes = node_fields
            .iter()
            .map(|field| {
                field.parse::<usize>().map_err(|_| {
                    parse_error(
                        line_no,
                        format!("element {id}: invalid node number '{field}'"),
                    )
                })
            })
            .collect::<Result<Vec<usize>>>()?;
        if nodes.len() != element_type.node_count() {
            return Err(parse_error(
                line_no,
                format!(
                    "element {id} of type {type_code} ({element_type:?}) needs {} nodes but has {}",
                    element_type.node_count(),
                    nodes.len()
                ),
            ));
        }
        mesh.elements.push(GmshElement {
            id,
            element_type,
            physical_tag,
            geometric_tag,
            nodes,
        });
    }
    check_declared_count("Elements", section, declared, found)
}

/// Parse a Gmsh 2.x ASCII `.msh` file from a string.
///
/// Sections recognised: `$MeshFormat`, `$PhysicalNames`, `$Nodes`, `$Elements`.
/// Any other section (`$NodeData`, `$Periodic`, `$Comments`, ...) is skipped.
/// Elements whose type has no [`GmshElementType`] (for example second-order
/// elements) are skipped as well.
///
/// The first section must be `$MeshFormat` and must announce a `2.x` version
/// in the ASCII encoding (file-type `0`).
///
/// # Errors
///
/// Returns [`Error::Parse`] if
/// - `$MeshFormat` is missing or announces another version (for example the
///   `4.1` files current Gmsh writes by default) or the binary encoding;
/// - a section is not terminated, or its declared record count does not match
///   the number of records it contains;
/// - a record is malformed: a field fails to parse, a node line does not have
///   exactly four fields, or an element declares more tags than its line holds;
/// - an element does not list exactly the number of nodes its type requires.
pub fn read_gmsh_v2(src: &str) -> Result<GmshMesh> {
    let src = src.strip_prefix('\u{feff}').unwrap_or(src);
    let lines: Vec<&str> = src.lines().collect();
    let mut mesh = GmshMesh::new();
    let mut have_format = false;
    let mut i = 0;

    while i < lines.len() {
        let text = lines[i].trim();
        let Some(name) = text.strip_prefix('$') else {
            i += 1;
            continue;
        };
        if name.starts_with("End") {
            i += 1;
            continue;
        }
        let Some(kind) = SectionKind::from_name(name) else {
            // Unknown section: jump over its body so that marker-like text inside
            // (for example in `$Comments`) is not mistaken for a real section.
            let end_marker = format!("$End{name}");
            i = lines
                .get(i + 1..)
                .and_then(|rest| rest.iter().position(|line| line.trim() == end_marker))
                .map_or(i + 1, |offset| i + 1 + offset + 1);
            continue;
        };
        let section = read_section(&lines, i, name)?;
        i = section.next;
        if kind != SectionKind::MeshFormat && !have_format {
            return Err(parse_error(
                section.start_line,
                format!("section ${name} appears before $MeshFormat"),
            ));
        }
        match kind {
            SectionKind::MeshFormat => {
                check_mesh_format(&lines, &section)?;
                have_format = true;
            }
            SectionKind::PhysicalNames => read_physical_names(&lines, &section, &mut mesh)?,
            SectionKind::Nodes => read_nodes(&lines, &section, &mut mesh)?,
            SectionKind::Elements => read_elements(&lines, &section, &mut mesh)?,
        }
    }

    if !have_format {
        return Err(Error::Parse(
            "Gmsh MSH: missing $MeshFormat section, cannot tell which format version this is"
                .to_string(),
        ));
    }
    Ok(mesh)
}

/// Serialise a `GmshMesh` to Gmsh 2.2 ASCII format.
pub fn write_gmsh_v2(mesh: &GmshMesh) -> String {
    let mut out = String::new();

    // Header
    out.push_str("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n");

    // Physical names
    if !mesh.physical_groups.is_empty() {
        out.push_str("$PhysicalNames\n");
        out.push_str(&format!("{}\n", mesh.physical_groups.len()));
        for pg in &mesh.physical_groups {
            out.push_str(&format!("{} {} \"{}\"\n", pg.dimension, pg.tag, pg.name));
        }
        out.push_str("$EndPhysicalNames\n");
    }

    // Nodes
    out.push_str("$Nodes\n");
    out.push_str(&format!("{}\n", mesh.nodes.len()));
    for node in &mesh.nodes {
        out.push_str(&format!(
            "{} {} {} {}\n",
            node.id, node.coords[0], node.coords[1], node.coords[2]
        ));
    }
    out.push_str("$EndNodes\n");

    // Elements
    out.push_str("$Elements\n");
    out.push_str(&format!("{}\n", mesh.elements.len()));
    for el in &mesh.elements {
        let type_code = el.element_type as usize;
        // Always write 2 tags: physical + geometric
        let node_str: Vec<String> = el.nodes.iter().map(|n| n.to_string()).collect();
        out.push_str(&format!(
            "{} {} 2 {} {} {}\n",
            el.id,
            type_code,
            el.physical_tag,
            el.geometric_tag,
            node_str.join(" ")
        ));
    }
    out.push_str("$EndElements\n");

    out
}

/// Collect the 1-based node IDs belonging to a specific physical group tag.
///
/// An element belongs to the group if `element.physical_tag == tag`.
/// All node IDs referenced by matching elements are returned (deduplicated,
/// sorted).
pub fn physical_group_nodes(mesh: &GmshMesh, tag: usize) -> Vec<usize> {
    let mut node_set: std::collections::BTreeSet<usize> = Default::default();
    for el in &mesh.elements {
        if el.physical_tag == tag {
            for &nid in &el.nodes {
                node_set.insert(nid);
            }
        }
    }
    node_set.into_iter().collect()
}

/// Convert a `GmshMesh` to a generic `FemMesh`.
///
/// Nodes are re-indexed to 0-based.  Only elements with a known element type
/// are included.
pub fn convert_to_fem_mesh(mesh: &GmshMesh) -> FemMesh {
    // Build a map from 1-based Gmsh ID → 0-based FEM index
    let mut id_to_idx: HashMap<usize, usize> = HashMap::new();
    let mut coords: Vec<[f64; 3]> = Vec::with_capacity(mesh.nodes.len());

    for (idx, node) in mesh.nodes.iter().enumerate() {
        id_to_idx.insert(node.id, idx);
        coords.push(node.coords);
    }

    let mut connectivity: Vec<Vec<usize>> = Vec::with_capacity(mesh.elements.len());
    for el in &mesh.elements {
        let zero_based: Vec<usize> = el
            .nodes
            .iter()
            .filter_map(|nid| id_to_idx.get(nid).copied())
            .collect();
        connectivity.push(zero_based);
    }

    FemMesh {
        coords,
        connectivity,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_msh_src() -> &'static str {
        "$MeshFormat\n\
         2.2 0 8\n\
         $EndMeshFormat\n\
         $PhysicalNames\n\
         1\n\
         2 1 \"Surface\"\n\
         $EndPhysicalNames\n\
         $Nodes\n\
         3\n\
         1 0.0 0.0 0.0\n\
         2 1.0 0.0 0.0\n\
         3 0.0 1.0 0.0\n\
         $EndNodes\n\
         $Elements\n\
         1\n\
         1 2 2 1 0 1 2 3\n\
         $EndElements\n"
    }

    // --- read_gmsh_v2 ---

    #[test]
    fn test_read_node_count() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.node_count(), 3);
    }

    #[test]
    fn test_read_node_coords() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let n1 = mesh.node_by_id(1).unwrap();
        assert_eq!(n1.coords, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_read_node_id_preserved() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert!(mesh.node_by_id(2).is_some());
        assert!(mesh.node_by_id(99).is_none());
    }

    #[test]
    fn test_read_element_count() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.element_count(), 1);
    }

    #[test]
    fn test_read_element_type() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.elements[0].element_type, GmshElementType::Triangle3);
    }

    #[test]
    fn test_read_element_physical_tag() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.elements[0].physical_tag, 1);
    }

    #[test]
    fn test_read_element_nodes() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.elements[0].nodes, vec![1, 2, 3]);
    }

    #[test]
    fn test_read_physical_group_count() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.physical_groups.len(), 1);
    }

    #[test]
    fn test_read_physical_group_name() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.physical_groups[0].name, "Surface");
    }

    #[test]
    fn test_read_physical_group_dimension() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        assert_eq!(mesh.physical_groups[0].dimension, 2);
    }

    #[test]
    fn test_read_empty_string() {
        // An empty input has no $MeshFormat, so it cannot be identified as MSH 2.x.
        let err = read_gmsh_v2("").expect_err("empty input is not a Gmsh file");
        assert!(err.to_string().contains("$MeshFormat"), "{err}");
    }

    #[test]
    fn test_read_format_only_gives_empty_mesh() {
        let mesh = read_gmsh_v2("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n").unwrap();
        assert_eq!(mesh.node_count(), 0);
        assert_eq!(mesh.element_count(), 0);
        assert!(mesh.physical_groups.is_empty());
    }

    // --- GmshElementType ---

    #[test]
    fn test_element_type_from_code_triangle() {
        assert_eq!(
            GmshElementType::from_code(2),
            Some(GmshElementType::Triangle3)
        );
    }

    #[test]
    fn test_element_type_from_code_unknown() {
        assert!(GmshElementType::from_code(999).is_none());
    }

    #[test]
    fn test_element_type_node_count_tet() {
        assert_eq!(GmshElementType::Tet4.node_count(), 4);
    }

    #[test]
    fn test_element_type_node_count_triangle() {
        assert_eq!(GmshElementType::Triangle3.node_count(), 3);
    }

    // --- write_gmsh_v2 ---

    #[test]
    fn test_write_contains_mesh_format() {
        let mesh = GmshMesh::new();
        let out = write_gmsh_v2(&mesh);
        assert!(out.contains("$MeshFormat"));
        assert!(out.contains("2.2 0 8"));
    }

    #[test]
    fn test_write_contains_nodes_section() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let out = write_gmsh_v2(&mesh);
        assert!(out.contains("$Nodes"));
        assert!(out.contains("$EndNodes"));
    }

    #[test]
    fn test_write_contains_elements_section() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let out = write_gmsh_v2(&mesh);
        assert!(out.contains("$Elements"));
        assert!(out.contains("$EndElements"));
    }

    // --- Roundtrip ---

    #[test]
    fn test_roundtrip_node_count() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let out = write_gmsh_v2(&mesh);
        let mesh2 = read_gmsh_v2(&out).unwrap();
        assert_eq!(mesh2.node_count(), mesh.node_count());
    }

    #[test]
    fn test_roundtrip_element_count() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let out = write_gmsh_v2(&mesh);
        let mesh2 = read_gmsh_v2(&out).unwrap();
        assert_eq!(mesh2.element_count(), mesh.element_count());
    }

    #[test]
    fn test_roundtrip_node_coords() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let out = write_gmsh_v2(&mesh);
        let mesh2 = read_gmsh_v2(&out).unwrap();
        let n2 = mesh2.node_by_id(2).unwrap();
        assert!((n2.coords[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_roundtrip_physical_group_name() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let out = write_gmsh_v2(&mesh);
        let mesh2 = read_gmsh_v2(&out).unwrap();
        assert_eq!(mesh2.physical_groups[0].name, "Surface");
    }

    #[test]
    fn test_roundtrip_element_type() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let out = write_gmsh_v2(&mesh);
        let mesh2 = read_gmsh_v2(&out).unwrap();
        assert_eq!(mesh2.elements[0].element_type, GmshElementType::Triangle3);
    }

    // --- physical_group_nodes ---

    #[test]
    fn test_physical_group_nodes_correct() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let nodes = physical_group_nodes(&mesh, 1);
        assert_eq!(nodes, vec![1, 2, 3]);
    }

    #[test]
    fn test_physical_group_nodes_missing_tag() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let nodes = physical_group_nodes(&mesh, 999);
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_physical_group_nodes_deduplicated() {
        // Two elements sharing node 1
        let src = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
                   $Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 1 1 0\n$EndNodes\n\
                   $Elements\n2\n\
                   1 2 2 1 0 1 2 3\n\
                   2 2 2 1 0 2 3 4\n\
                   $EndElements\n";
        let mesh = read_gmsh_v2(src).unwrap();
        let nodes = physical_group_nodes(&mesh, 1);
        // Nodes 1,2,3,4 — no duplicates
        assert_eq!(nodes.len(), 4);
    }

    // --- convert_to_fem_mesh ---

    #[test]
    fn test_fem_mesh_node_count() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let fem = convert_to_fem_mesh(&mesh);
        assert_eq!(fem.coords.len(), 3);
    }

    #[test]
    fn test_fem_mesh_connectivity_count() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let fem = convert_to_fem_mesh(&mesh);
        assert_eq!(fem.connectivity.len(), 1);
    }

    #[test]
    fn test_fem_mesh_connectivity_zero_based() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let fem = convert_to_fem_mesh(&mesh);
        // 1-based [1,2,3] → 0-based [0,1,2]
        assert_eq!(fem.connectivity[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_fem_mesh_coords_correct() {
        let mesh = read_gmsh_v2(triangle_msh_src()).unwrap();
        let fem = convert_to_fem_mesh(&mesh);
        // Node 2 has coords [1,0,0]
        assert!((fem.coords[1][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fem_mesh_empty() {
        let mesh = GmshMesh::new();
        let fem = convert_to_fem_mesh(&mesh);
        assert!(fem.coords.is_empty());
        assert!(fem.connectivity.is_empty());
    }

    // --- reader robustness (malformed and foreign-version input) ---

    /// A syntactically valid MSH 2.2 header followed by `sections`.
    fn msh2(sections: &str) -> String {
        format!("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n{sections}")
    }

    /// The message of the `Error::Parse` a reader call must have produced.
    fn parse_message(result: Result<GmshMesh>) -> String {
        match result {
            Err(Error::Parse(message)) => message,
            Err(other) => panic!("expected Error::Parse, got {other:?}"),
            Ok(mesh) => panic!(
                "expected an error but parsed {} nodes and {} elements",
                mesh.node_count(),
                mesh.element_count()
            ),
        }
    }

    const TRIANGLE_NODES: &str = "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n";

    #[test]
    fn gmsh_v2_bad_ntags_returns_err() {
        // 99 tags on a six-field line: used to slice `parts[102..]` and panic.
        let src = msh2(&format!(
            "{TRIANGLE_NODES}$Elements\n1\n1 2 99 1 2 3\n$EndElements\n"
        ));
        let message = parse_message(read_gmsh_v2(&src));
        assert!(message.contains("99 tags"), "unexpected message: {message}");
        assert!(
            message.contains("line 12"),
            "should locate the line: {message}"
        );

        // A tag count so large that `3 + n_tags` overflows usize.
        let src = msh2(&format!(
            "{TRIANGLE_NODES}$Elements\n1\n1 2 {} 1 2 3\n$EndElements\n",
            usize::MAX
        ));
        let message = parse_message(read_gmsh_v2(&src));
        assert!(
            message.contains("out of range"),
            "unexpected message: {message}"
        );

        // Tags consume the whole line, leaving no node list at all.
        let src = msh2(&format!(
            "{TRIANGLE_NODES}$Elements\n1\n1 2 3 7 8 9\n$EndElements\n"
        ));
        let message = parse_message(read_gmsh_v2(&src));
        assert!(message.contains("needs 3 nodes but has 0"), "{message}");
    }

    #[test]
    fn gmsh_v4_element_line_in_v2_section_returns_err() {
        // The MSH 4.1 element record `1 1 2 3` reads as type 1 with two tags in
        // the 2.x layout: node_start = 5 > 4 fields.
        let src = msh2("$Elements\n1\n1 1 2 3\n$EndElements\n");
        let message = parse_message(read_gmsh_v2(&src));
        assert!(message.contains("declares 2 tags"), "{message}");
    }

    #[test]
    fn gmsh_v41_input_returns_err() {
        // What `gmsh -3 model.geo` writes by default for a single triangle.
        let v41 = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
                   $Entities\n0 0 1 0\n1 0 0 0 1 1 0 1 1 3 -1 -2 -3\n$EndEntities\n\
                   $Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n\
                   $Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n";
        let message = parse_message(read_gmsh_v2(v41));
        assert!(
            message.contains("4.1"),
            "should name the version: {message}"
        );
        assert!(message.contains("unsupported"), "{message}");
        assert!(
            message.contains("line 2"),
            "should locate the version line: {message}"
        );

        for version in ["4.0", "3.0", "1.0", "20.1", "2", ""] {
            let src = format!("$MeshFormat\n{version} 0 8\n$EndMeshFormat\n");
            assert!(
                read_gmsh_v2(&src).is_err(),
                "version '{version}' must be rejected"
            );
        }
    }

    #[test]
    fn gmsh_binary_input_returns_err() {
        let src = "$MeshFormat\n2.2 1 8\n$EndMeshFormat\n";
        let message = parse_message(read_gmsh_v2(src));
        assert!(message.contains("binary"), "{message}");
    }

    #[test]
    fn gmsh_mesh_format_must_be_complete() {
        for header in ["2.2", "2.2 x 8", "2.2 0 eight", "2.2 -1 8"] {
            let src = format!("$MeshFormat\n{header}\n$EndMeshFormat\n");
            assert!(read_gmsh_v2(&src).is_err(), "'{header}' must be rejected");
        }
        let empty_format = "$MeshFormat\n$EndMeshFormat\n";
        assert!(read_gmsh_v2(empty_format).is_err());
    }

    #[test]
    fn gmsh_missing_mesh_format_returns_err() {
        let without_format = format!("{TRIANGLE_NODES}$Elements\n0\n$EndElements\n");
        let message = parse_message(read_gmsh_v2(&without_format));
        assert!(message.contains("before $MeshFormat"), "{message}");

        let message = parse_message(read_gmsh_v2("hello world\nthis is not a mesh\n"));
        assert!(message.contains("missing $MeshFormat"), "{message}");
    }

    #[test]
    fn gmsh_v20_and_v21_headers_are_accepted() {
        for version in ["2.0", "2.1", "2.2"] {
            let src = format!(
                "$MeshFormat\n{version} 0 8\n$EndMeshFormat\n{TRIANGLE_NODES}\
                 $Elements\n1\n1 2 2 1 0 1 2 3\n$EndElements\n"
            );
            let mesh = read_gmsh_v2(&src).unwrap();
            assert_eq!(mesh.node_count(), 3, "version {version}");
            assert_eq!(mesh.element_count(), 1, "version {version}");
        }
    }

    #[test]
    fn gmsh_wrong_node_count_returns_err() {
        for (type_code, nodes, expected) in [
            (2, "1 2", 3),
            (2, "1 2 3 1", 3),
            (3, "1 2 3", 4),
            (4, "1 2 3", 4),
            (5, "1 2 3 4", 8),
            (1, "1", 2),
            (15, "1 2", 1),
        ] {
            let src = msh2(&format!(
                "{TRIANGLE_NODES}$Elements\n1\n1 {type_code} 2 1 0 {nodes}\n$EndElements\n"
            ));
            let message = parse_message(read_gmsh_v2(&src));
            assert!(
                message.contains(&format!("needs {expected} nodes")),
                "type {type_code} with nodes '{nodes}': {message}"
            );
        }
    }

    #[test]
    fn gmsh_malformed_records_return_err() {
        let cases: [(&str, &str); 14] = [
            ("$Nodes\n1\n1 0 0\n$EndNodes\n", "4 fields"),
            ("$Nodes\n1\n1 0 0 0 0\n$EndNodes\n", "4 fields"),
            (
                "$Nodes\n1\n1 0 abc 0\n$EndNodes\n",
                "invalid coordinate 'abc'",
            ),
            ("$Nodes\n1\nx 0 0 0\n$EndNodes\n", "invalid node number 'x'"),
            ("$Nodes\nmany\n$EndNodes\n", "record count"),
            ("$Nodes\n$EndNodes\n", "record-count line"),
            (
                "$Elements\n1\na 2 2 1 0 1 2 3\n$EndElements\n",
                "invalid element number",
            ),
            (
                "$Elements\n1\n1 b 2 1 0 1 2 3\n$EndElements\n",
                "invalid element type",
            ),
            ("$Elements\n1\n1 2\n$EndElements\n", "missing its tag count"),
            (
                "$Elements\n1\n1 2 2 x 0 1 2 3\n$EndElements\n",
                "invalid physical tag",
            ),
            (
                "$Elements\n1\n1 2 2 1 y 1 2 3\n$EndElements\n",
                "invalid elementary tag",
            ),
            (
                "$Elements\n1\n1 2 2 1 0 1 2 z\n$EndElements\n",
                "invalid node number 'z'",
            ),
            (
                "$PhysicalNames\n1\n2 x \"s\"\n$EndPhysicalNames\n",
                "expected `<dimension>",
            ),
            (
                "$PhysicalNames\n1\n2 1\n$EndPhysicalNames\n",
                "expected `<dimension>",
            ),
        ];
        for (sections, expected) in cases {
            let message = parse_message(read_gmsh_v2(&msh2(sections)));
            assert!(
                message.contains(expected),
                "input {sections:?}: expected '{expected}' in '{message}'"
            );
        }
    }

    #[test]
    fn gmsh_declared_counts_must_match() {
        let cases = [
            "$Nodes\n3\n1 0 0 0\n2 1 0 0\n$EndNodes\n",
            "$Nodes\n1\n1 0 0 0\n2 1 0 0\n$EndNodes\n",
            "$Elements\n2\n1 2 2 1 0 1 2 3\n$EndElements\n",
            "$Elements\n0\n1 2 2 1 0 1 2 3\n$EndElements\n",
            "$PhysicalNames\n2\n2 1 \"a\"\n$EndPhysicalNames\n",
        ];
        for sections in cases {
            let message = parse_message(read_gmsh_v2(&msh2(sections)));
            assert!(
                message.contains("declares"),
                "input {sections:?}: {message}"
            );
        }
    }

    #[test]
    fn gmsh_unterminated_section_returns_err() {
        for sections in [
            "$Nodes\n1\n1 0 0 0\n",
            "$Elements\n1\n1 2 2 1 0 1 2 3\n",
            "$PhysicalNames\n1\n2 1 \"a\"\n",
        ] {
            let message = parse_message(read_gmsh_v2(&msh2(sections)));
            assert!(
                message.contains("not terminated"),
                "input {sections:?}: {message}"
            );
        }
        let unterminated_format = "$MeshFormat\n2.2 0 8\n";
        assert!(read_gmsh_v2(unterminated_format).is_err());
    }

    #[test]
    fn gmsh_element_tag_variants() {
        let src = msh2(&format!(
            "{TRIANGLE_NODES}$Elements\n3\n\
             1 2 0 1 2 3\n\
             2 2 1 5 1 2 3\n\
             3 2 4 7 9 2 1 1 2 3\n\
             $EndElements\n"
        ));
        let mesh = read_gmsh_v2(&src).unwrap();
        assert_eq!(mesh.element_count(), 3);
        assert_eq!(
            (
                mesh.elements[0].physical_tag,
                mesh.elements[0].geometric_tag
            ),
            (0, 0)
        );
        assert_eq!(
            (
                mesh.elements[1].physical_tag,
                mesh.elements[1].geometric_tag
            ),
            (5, 0)
        );
        // Partitioned meshes carry extra tags after the first two.
        assert_eq!(
            (
                mesh.elements[2].physical_tag,
                mesh.elements[2].geometric_tag
            ),
            (7, 9)
        );
        for element in &mesh.elements {
            assert_eq!(element.nodes, vec![1, 2, 3]);
        }
    }

    #[test]
    fn gmsh_unknown_sections_and_element_types_are_skipped() {
        let src = msh2(&format!(
            "$Comments\n$Nodes\n99\n$EndNodes\n$EndComments\n\
             {TRIANGLE_NODES}\
             $Elements\n2\n\
             1 2 2 1 0 1 2 3\n\
             2 9 2 1 0 1 2 3 4 5 6\n\
             $EndElements\n\
             $NodeData\n1\n\"scalar\"\n1\n0.0\n3\n0\n1\n3\n1 0.5\n2 0.5\n3 0.5\n$EndNodeData\n"
        ));
        let mesh = read_gmsh_v2(&src).unwrap();
        // The commented-out $Nodes block must not leak into the mesh.
        assert_eq!(mesh.node_count(), 3);
        // The second-order triangle (type 9) is not representable and is skipped.
        assert_eq!(mesh.element_count(), 1);
        assert_eq!(mesh.elements[0].element_type, GmshElementType::Triangle3);
    }

    #[test]
    fn gmsh_crlf_line_endings_and_bom_are_accepted() {
        let unix = triangle_msh_src();
        let windows = unix.replace('\n', "\r\n");
        let with_bom = format!("\u{feff}{windows}");
        for text in [windows, with_bom] {
            let mesh = read_gmsh_v2(&text).unwrap();
            assert_eq!(mesh.node_count(), 3);
            assert_eq!(mesh.elements[0].nodes, vec![1, 2, 3]);
            assert_eq!(mesh.physical_groups[0].name, "Surface");
        }
    }

    #[test]
    fn gmsh_physical_name_keeps_inner_spacing() {
        let src =
            msh2("$PhysicalNames\n2\n2 1 \"Left  wall\"\n3   2    \"Fluid\"\n$EndPhysicalNames\n");
        let mesh = read_gmsh_v2(&src).unwrap();
        assert_eq!(mesh.physical_groups.len(), 2);
        assert_eq!(mesh.physical_groups[0].name, "Left  wall");
        assert_eq!(mesh.physical_groups[0].dimension, 2);
        assert_eq!(mesh.physical_groups[1].name, "Fluid");
        assert_eq!(
            (
                mesh.physical_groups[1].dimension,
                mesh.physical_groups[1].tag
            ),
            (3, 2)
        );
    }

    #[test]
    fn gmsh_roundtrip_every_element_type() {
        let mut mesh = GmshMesh::new();
        for id in 1..=8usize {
            mesh.nodes.push(GmshNode {
                id,
                coords: [id as f64, 0.5 * id as f64, -(id as f64)],
            });
        }
        for (index, element_type) in [
            GmshElementType::Point1,
            GmshElementType::Line2,
            GmshElementType::Triangle3,
            GmshElementType::Quad4,
            GmshElementType::Tet4,
            GmshElementType::Hex8,
        ]
        .into_iter()
        .enumerate()
        {
            mesh.elements.push(GmshElement {
                id: index + 1,
                element_type,
                physical_tag: index + 10,
                geometric_tag: index + 20,
                nodes: (1..=element_type.node_count()).collect(),
            });
        }
        let text = write_gmsh_v2(&mesh);
        let reread = read_gmsh_v2(&text).unwrap();
        assert_eq!(reread.nodes, mesh.nodes);
        assert_eq!(reread.elements, mesh.elements);
    }

    #[test]
    fn gmsh_roundtrip_empty_mesh() {
        let text = write_gmsh_v2(&GmshMesh::new());
        let mesh = read_gmsh_v2(&text).unwrap();
        assert_eq!(mesh.node_count(), 0);
        assert_eq!(mesh.element_count(), 0);
    }

    #[test]
    fn gmsh_truncation_and_token_mutation_never_panic() {
        let doc = triangle_msh_src();
        assert!(read_gmsh_v2(doc).is_ok());
        // Every prefix cut at a char boundary (this also truncates sections).
        for (cut, _) in doc.char_indices() {
            let _ = read_gmsh_v2(&doc[..cut]);
        }
        // Every whitespace-separated token replaced by hostile text.
        let junk = [
            "",
            "-1",
            "x",
            "99999999999999999999999",
            "1e400",
            "\u{65e5}",
            "$Nodes",
        ];
        let lines: Vec<&str> = doc.lines().collect();
        for (line_index, line) in lines.iter().enumerate() {
            let tokens: Vec<&str> = line.split(' ').collect();
            for token_index in 0..tokens.len() {
                for replacement in junk {
                    let mut mutated_tokens = tokens.clone();
                    mutated_tokens[token_index] = replacement;
                    let mut mutated_lines: Vec<String> =
                        lines.iter().map(|l| l.to_string()).collect();
                    mutated_lines[line_index] = mutated_tokens.join(" ");
                    let _ = read_gmsh_v2(&mutated_lines.join("\n"));
                }
            }
        }
    }
}
