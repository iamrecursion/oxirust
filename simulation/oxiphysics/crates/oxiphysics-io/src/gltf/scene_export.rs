// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! glTF 2.0 JSON (`.gltf`) export of a [`GltfScene`].
//!
//! [`scene_to_gltf_json`] and [`scene_to_gltf_external`] turn a scene into a
//! complete glTF document: the node graph with its transforms, one mesh entry
//! per [`GltfMesh`](super::GltfMesh), and a single binary buffer that holds all
//! vertex and index data, described by buffer views and accessors.  The result
//! is meant to render: every index in the document resolves, `POSITION`
//! accessors carry the `min`/`max` bounds the specification requires, and no
//! empty array (which the schema forbids) is emitted.

use std::collections::HashSet;

use super::functions::{base64_encode, escape_json};
use super::types::{GltfNode, GltfPrimitive, GltfScene};

/// `bufferView.target` of vertex attribute data (`ARRAY_BUFFER`).
const TARGET_ARRAY_BUFFER: u32 = 34962;
/// `bufferView.target` of index data (`ELEMENT_ARRAY_BUFFER`).
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;
/// `accessor.componentType` of 32-bit floats.
const COMPONENT_FLOAT: u32 = 5126;
/// `accessor.componentType` of unsigned 32-bit integers.
const COMPONENT_UNSIGNED_INT: u32 = 5125;
/// External buffer URI used by [`scene_to_gltf_json`] when buffers are not embedded.
const DEFAULT_BUFFER_URI: &str = "buffer.bin";

/// Where the document says its binary buffer lives.
enum BufferUri<'a> {
    /// Embed the payload as a base64 `data:` URI.
    Embedded,
    /// Refer to a separate file at this relative URI.
    External(&'a str),
}

/// Serialise `scene` as a glTF 2.0 JSON document (`.gltf`).
///
/// With `embed_buffers == true` the binary payload is embedded as a base64
/// `data:` URI, so the returned text is self-contained.  With `false` the
/// document refers to an external `buffer.bin` that **this function does not
/// produce**: use [`scene_to_gltf_external`] to get the document together with
/// the bytes to store in that file.
///
/// The document contains
/// - `nodes`: every node of `scene` with its name, mesh, children and
///   translation/rotation/scale.  Child indices that point outside the scene,
///   at the node itself, or that repeat are dropped, and a mesh index that
///   does not resolve is omitted, so every reference in the output is valid.
///   If the scene has meshes but no nodes at all, one root node per mesh is
///   created so the geometry is actually instantiated;
/// - `scenes[0].nodes`: the root nodes, i.e. those that are nobody's child;
/// - `meshes`: one entry per [`GltfMesh`](super::GltfMesh) holding all of its
///   primitives, so `node.mesh` indices equal the indices into `scene.meshes`;
/// - `accessors`/`bufferViews`/`buffers`: `POSITION` (with `min`/`max`),
///   `NORMAL` and `TEXCOORD_0` when their length equals the vertex count, and
///   `u32` triangle indices.
///
/// The scene's materials, cameras, lights and animations are not part of this
/// document.
///
/// Anything glTF cannot represent is left out instead of producing an invalid
/// file: primitives without positions (a mesh left with none is dropped and the
/// mesh indices of later meshes shift down), index lists that are empty or
/// shorter than a triangle, a trailing partial triangle, and non-finite
/// numbers (written as `0`).  [`validate_scene`](super::validate_scene) reports
/// such problems up front.
pub fn scene_to_gltf_json(scene: &GltfScene, embed_buffers: bool) -> String {
    let uri = if embed_buffers {
        BufferUri::Embedded
    } else {
        BufferUri::External(DEFAULT_BUFFER_URI)
    };
    build_document(scene, uri).0
}

/// Serialise `scene` as a `.gltf` document plus the separate binary buffer it
/// refers to.
///
/// Returns `(json, buffer)`.  `json` references the buffer as `buffer_uri`, a
/// relative URI that must be properly percent-encoded; store `buffer` in the
/// file that URI names, next to the `.gltf` file.  The content of the document
/// is the same as for [`scene_to_gltf_json`].  A scene without any geometry has
/// an empty `buffer` and its document declares no buffer at all.
pub fn scene_to_gltf_external(scene: &GltfScene, buffer_uri: &str) -> (String, Vec<u8>) {
    build_document(scene, BufferUri::External(buffer_uri))
}

/// A finite JSON number for `v`; JSON has no NaN or infinity, so those become `0`.
///
/// The `f32` is widened to `f64` first, so the text parses back to exactly the
/// value stored in the binary buffer.  That matters for accessor `min`/`max`,
/// which readers compare against the data (`0.1f32` is written as
/// `0.10000000149011612`, not `0.1`, which would be a different double).
fn json_f32(v: f32) -> String {
    json_f64(f64::from(v))
}

/// A finite JSON number for `v`; JSON has no NaN or infinity, so those become `0`.
fn json_f64(v: f64) -> String {
    if v.is_finite() {
        v.to_string()
    } else {
        "0".to_string()
    }
}

fn f32_list(values: &[f32]) -> String {
    let parts: Vec<String> = values.iter().map(|&v| json_f32(v)).collect();
    parts.join(", ")
}

fn f64_list(values: &[f64]) -> String {
    let parts: Vec<String> = values.iter().map(|&v| json_f64(v)).collect();
    parts.join(", ")
}

fn index_list(indices: &[usize]) -> String {
    let parts: Vec<String> = indices.iter().map(usize::to_string).collect();
    parts.join(", ")
}

/// Little-endian bytes of a list of 3-component `f32` vectors.
fn vec3_bytes(values: &[[f32; 3]]) -> Vec<u8> {
    values
        .iter()
        .flatten()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

/// Little-endian bytes of a list of 2-component `f32` vectors.
fn vec2_bytes(values: &[[f32; 2]]) -> Vec<u8> {
    values
        .iter()
        .flatten()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

/// Collects the single binary buffer together with its views and accessors.
#[derive(Default)]
struct BufferBuilder {
    data: Vec<u8>,
    views: Vec<String>,
    accessors: Vec<String>,
}

impl BufferBuilder {
    /// Append `bytes` as a new, 4-byte aligned buffer view; returns its index.
    fn push_view(&mut self, bytes: &[u8], target: u32) -> usize {
        while !self.data.len().is_multiple_of(4) {
            self.data.push(0);
        }
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        let index = self.views.len();
        self.views.push(format!(
            "{{ \"buffer\": 0, \"byteOffset\": {offset}, \"byteLength\": {}, \"target\": {target} }}",
            bytes.len()
        ));
        index
    }

    /// Describe `count` elements of `kind` stored in buffer view `view`.
    fn push_accessor(
        &mut self,
        view: usize,
        component_type: u32,
        count: usize,
        kind: &str,
        bounds: Option<([f32; 3], [f32; 3])>,
    ) -> usize {
        let mut fields = vec![
            format!("\"bufferView\": {view}"),
            format!("\"componentType\": {component_type}"),
            format!("\"count\": {count}"),
            format!("\"type\": \"{kind}\""),
        ];
        if let Some((min, max)) = bounds {
            fields.push(format!("\"min\": [{}]", f32_list(&min)));
            fields.push(format!("\"max\": [{}]", f32_list(&max)));
        }
        let index = self.accessors.len();
        self.accessors.push(format!("{{ {} }}", fields.join(", ")));
        index
    }

    /// Store a per-vertex attribute and return its accessor index.
    fn push_attribute(&mut self, bytes: &[u8], count: usize, kind: &str) -> usize {
        let view = self.push_view(bytes, TARGET_ARRAY_BUFFER);
        self.push_accessor(view, COMPONENT_FLOAT, count, kind, None)
    }

    /// Store `primitive` and return its JSON, or `None` if it has no positions.
    fn add_primitive(&mut self, primitive: &GltfPrimitive) -> Option<String> {
        let vertex_count = primitive.positions.len();
        if vertex_count == 0 {
            return None;
        }
        let bounds = primitive.bounding_box();
        let position_view = self.push_view(&vec3_bytes(&primitive.positions), TARGET_ARRAY_BUFFER);
        let position_accessor = self.push_accessor(
            position_view,
            COMPONENT_FLOAT,
            vertex_count,
            "VEC3",
            Some(bounds),
        );
        let mut attributes = vec![format!("\"POSITION\": {position_accessor}")];
        if primitive.normals.len() == vertex_count {
            let accessor =
                self.push_attribute(&vec3_bytes(&primitive.normals), vertex_count, "VEC3");
            attributes.push(format!("\"NORMAL\": {accessor}"));
        }
        if primitive.texcoords.len() == vertex_count {
            let accessor =
                self.push_attribute(&vec2_bytes(&primitive.texcoords), vertex_count, "VEC2");
            attributes.push(format!("\"TEXCOORD_0\": {accessor}"));
        }
        let mut fields = vec![format!("\"attributes\": {{ {} }}", attributes.join(", "))];
        // Triangle lists only: a trailing partial triangle cannot be drawn.  An
        // empty index list is written as non-indexed geometry, because glTF
        // forbids zero-length buffer views and accessors.
        let index_count = primitive.indices.len() / 3 * 3;
        if index_count > 0 {
            let bytes: Vec<u8> = primitive
                .indices
                .iter()
                .take(index_count)
                .flat_map(|index| index.to_le_bytes())
                .collect();
            let view = self.push_view(&bytes, TARGET_ELEMENT_ARRAY_BUFFER);
            let accessor =
                self.push_accessor(view, COMPONENT_UNSIGNED_INT, index_count, "SCALAR", None);
            fields.push(format!("\"indices\": {accessor}"));
        }
        Some(format!("{{ {} }}", fields.join(", ")))
    }
}

/// JSON of one node; `mesh` and `children` are already validated indices.
fn node_json(node: &GltfNode, mesh: Option<usize>, children: &[usize]) -> String {
    let mut fields = vec![format!("\"name\": \"{}\"", escape_json(&node.name))];
    if let Some(mesh) = mesh {
        fields.push(format!("\"mesh\": {mesh}"));
    }
    if !children.is_empty() {
        fields.push(format!("\"children\": [{}]", index_list(children)));
    }
    fields.push(format!(
        "\"translation\": [{}]",
        f64_list(&node.translation)
    ));
    fields.push(format!("\"rotation\": [{}]", f64_list(&node.rotation)));
    fields.push(format!("\"scale\": [{}]", f64_list(&node.scale)));
    format!("{{ {} }}", fields.join(", "))
}

/// The scene's nodes as JSON, plus the indices of the root nodes.
///
/// `mesh_remap[i]` is the exported index of `scene.meshes[i]`, or `None` if
/// that mesh was dropped.
fn describe_nodes(scene: &GltfScene, mesh_remap: &[Option<usize>]) -> (Vec<String>, Vec<usize>) {
    let node_count = scene.nodes.len();
    let mut has_parent = vec![false; node_count];
    let mut entries = Vec::with_capacity(node_count);
    for (index, node) in scene.nodes.iter().enumerate() {
        let mut seen = HashSet::new();
        let children: Vec<usize> = node
            .children
            .iter()
            .copied()
            .filter(|&child| child < node_count && child != index && seen.insert(child))
            .collect();
        for &child in &children {
            has_parent[child] = true;
        }
        let mesh = node
            .mesh
            .and_then(|mesh| mesh_remap.get(mesh).copied().flatten());
        entries.push(node_json(node, mesh, &children));
    }
    let roots = (0..node_count).filter(|&i| !has_parent[i]).collect();
    (entries, roots)
}

/// One root node per exported mesh, for scenes that define meshes but no nodes.
fn synthesize_nodes(scene: &GltfScene, mesh_remap: &[Option<usize>]) -> (Vec<String>, Vec<usize>) {
    let entries: Vec<String> = scene
        .meshes
        .iter()
        .zip(mesh_remap)
        .filter_map(|(mesh, exported)| {
            exported.map(|exported| {
                format!(
                    "{{ \"name\": \"{}\", \"mesh\": {exported} }}",
                    escape_json(&mesh.name)
                )
            })
        })
        .collect();
    let roots = (0..entries.len()).collect();
    (entries, roots)
}

/// `"key": [ item, item ]`, or nothing for an empty list (the schema requires
/// at least one element in every top-level array).
fn array_member(key: &str, items: &[String]) -> Option<String> {
    if items.is_empty() {
        None
    } else {
        Some(format!("\"{key}\": [\n    {}\n  ]", items.join(",\n    ")))
    }
}

fn build_document(scene: &GltfScene, uri: BufferUri<'_>) -> (String, Vec<u8>) {
    let mut buffer = BufferBuilder::default();

    // One glTF mesh per `GltfMesh`, holding all of its primitives.
    let mut mesh_entries: Vec<String> = Vec::new();
    let mut mesh_remap: Vec<Option<usize>> = Vec::with_capacity(scene.meshes.len());
    for mesh in &scene.meshes {
        let primitives: Vec<String> = mesh
            .primitives
            .iter()
            .filter_map(|primitive| buffer.add_primitive(primitive))
            .collect();
        if primitives.is_empty() {
            mesh_remap.push(None);
        } else {
            mesh_remap.push(Some(mesh_entries.len()));
            mesh_entries.push(format!(
                "{{ \"name\": \"{}\", \"primitives\": [{}] }}",
                escape_json(&mesh.name),
                primitives.join(", ")
            ));
        }
    }

    let (node_entries, roots) = if scene.nodes.is_empty() {
        synthesize_nodes(scene, &mesh_remap)
    } else {
        describe_nodes(scene, &mesh_remap)
    };

    let scene_entry = if roots.is_empty() {
        "{}".to_string()
    } else {
        format!("{{ \"nodes\": [{}] }}", index_list(&roots))
    };
    let mut members = vec![
        "\"asset\": { \"version\": \"2.0\", \"generator\": \"OxiPhysics\" }".to_string(),
        "\"scene\": 0".to_string(),
        format!("\"scenes\": [{scene_entry}]"),
    ];
    members.extend(array_member("nodes", &node_entries));
    members.extend(array_member("meshes", &mesh_entries));
    members.extend(array_member("accessors", &buffer.accessors));
    members.extend(array_member("bufferViews", &buffer.views));
    if !buffer.data.is_empty() {
        let uri_text = match uri {
            BufferUri::Embedded => format!(
                "data:application/octet-stream;base64,{}",
                base64_encode(&buffer.data)
            ),
            BufferUri::External(external) => escape_json(external),
        };
        members.push(format!(
            "\"buffers\": [{{ \"uri\": \"{uri_text}\", \"byteLength\": {} }}]",
            buffer.data.len()
        ));
    }
    let body: Vec<String> = members.iter().map(|member| format!("  {member}")).collect();
    (format!("{{\n{}\n}}", body.join(",\n")), buffer.data)
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;
    use crate::gltf::{GltfMesh, GltfNode, GltfPrimitive};

    // ---- helpers -------------------------------------------------------------

    fn parse(json: &str) -> Value {
        serde_json::from_str(json).expect("the exporter must emit valid JSON")
    }

    /// The numbers of a JSON array as `f64` (so `1` and `1.0` compare equal).
    fn floats(value: &Value) -> Vec<f64> {
        value
            .as_array()
            .expect("a JSON array")
            .iter()
            .map(|v| v.as_f64().expect("a JSON number"))
            .collect()
    }

    fn as_usize(value: &Value) -> usize {
        usize::try_from(value.as_u64().expect("a non-negative integer")).expect("fits usize")
    }

    fn list<'a>(doc: &'a Value, key: &str) -> &'a [Value] {
        doc.get(key)
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
    }

    /// Decode base64 (standard alphabet, optional `=` padding).
    fn base64_decode(text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut accumulator = 0u32;
        let mut bits = 0u32;
        for byte in text.bytes() {
            let value = match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => break,
                other => panic!("not a base64 character: {other}"),
            };
            accumulator = (accumulator << 6) | u32::from(value);
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((accumulator >> bits) & 0xFF) as u8);
                accumulator &= (1 << bits) - 1;
            }
        }
        out
    }

    /// Bytes of the buffer view behind `accessor`, from the accessor's own offset on.
    fn accessor_bytes<'a>(doc: &Value, buffer: &'a [u8], accessor: usize) -> &'a [u8] {
        let acc = &list(doc, "accessors")[accessor];
        let view = &list(doc, "bufferViews")[as_usize(&acc["bufferView"])];
        let start = as_usize(&view["byteOffset"]) + acc.get("byteOffset").map_or(0, as_usize);
        let end = as_usize(&view["byteOffset"]) + as_usize(&view["byteLength"]);
        &buffer[start..end]
    }

    fn accessor_f32(doc: &Value, buffer: &[u8], accessor: usize) -> Vec<f32> {
        accessor_bytes(doc, buffer, accessor)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn accessor_u32(doc: &Value, buffer: &[u8], accessor: usize) -> Vec<u32> {
        accessor_bytes(doc, buffer, accessor)
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Check every cross-reference and byte range of a glTF document.
    fn assert_consistent(doc: &Value, buffer: &[u8]) {
        assert_eq!(doc["asset"]["version"], "2.0");
        assert_eq!(doc["scene"], 0);
        assert!(doc["scenes"].as_array().is_some_and(|s| s.len() == 1));
        for (key, value) in doc.as_object().expect("a JSON object") {
            if let Some(items) = value.as_array() {
                assert!(!items.is_empty(), "'{key}' must not be an empty array");
            }
        }

        let buffers = list(doc, "buffers");
        let views = list(doc, "bufferViews");
        let accessors = list(doc, "accessors");
        match buffers.first() {
            Some(entry) => {
                assert_eq!(buffers.len(), 1);
                assert_eq!(as_usize(&entry["byteLength"]), buffer.len());
                assert!(!buffer.is_empty());
                assert!(entry["uri"].is_string());
            }
            None => {
                assert!(buffer.is_empty());
                assert!(views.is_empty() && accessors.is_empty());
            }
        }

        for view in views {
            let offset = as_usize(&view["byteOffset"]);
            let length = as_usize(&view["byteLength"]);
            assert_eq!(as_usize(&view["buffer"]), 0);
            assert!(length >= 1, "zero-length bufferView");
            assert_eq!(offset % 4, 0, "bufferView offset must be 4-byte aligned");
            assert!(
                offset + length <= buffer.len(),
                "bufferView leaves the buffer"
            );
            assert!(matches!(as_usize(&view["target"]), 34962 | 34963));
        }

        for (index, accessor) in accessors.iter().enumerate() {
            let view = &views[as_usize(&accessor["bufferView"])];
            let count = as_usize(&accessor["count"]);
            assert!(count >= 1, "accessor {index} has no elements");
            let width = match accessor["type"].as_str().expect("accessor type") {
                "SCALAR" => 1,
                "VEC2" => 2,
                "VEC3" => 3,
                other => panic!("unexpected accessor type {other}"),
            };
            assert_eq!(
                count * width * 4,
                as_usize(&view["byteLength"]),
                "accessor {index} must cover its bufferView exactly"
            );
            match as_usize(&accessor["componentType"]) {
                5126 => assert_eq!(as_usize(&view["target"]), 34962),
                5125 => {
                    assert_eq!(width, 1);
                    assert_eq!(as_usize(&view["target"]), 34963);
                }
                other => panic!("unexpected componentType {other}"),
            }
        }

        let meshes = list(doc, "meshes");
        for mesh in meshes {
            let primitives = mesh["primitives"].as_array().expect("primitives");
            assert!(!primitives.is_empty());
            for primitive in primitives {
                let attributes = primitive["attributes"].as_object().expect("attributes");
                let position = &accessors[as_usize(&attributes["POSITION"])];
                let vertex_count = as_usize(&position["count"]);
                assert_eq!(position["type"], "VEC3");
                // The specification requires min/max on POSITION, and they must be tight.
                let values = accessor_f32(doc, buffer, as_usize(&attributes["POSITION"]));
                for axis in 0..3 {
                    let min = position["min"][axis].as_f64().expect("min");
                    let max = position["max"][axis].as_f64().expect("max");
                    let column: Vec<f64> =
                        values.chunks_exact(3).map(|v| f64::from(v[axis])).collect();
                    let lowest = column.iter().copied().fold(f64::INFINITY, f64::min);
                    let highest = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    if column.iter().all(|v| v.is_finite()) {
                        assert_eq!(min, lowest);
                        assert_eq!(max, highest);
                    }
                    assert!(min <= max);
                }
                for (name, accessor) in attributes {
                    let accessor = &accessors[as_usize(accessor)];
                    assert_eq!(
                        as_usize(&accessor["count"]),
                        vertex_count,
                        "attribute {name} must have one element per vertex"
                    );
                }
                if let Some(indices) = primitive.get("indices") {
                    let accessor = &accessors[as_usize(indices)];
                    assert_eq!(as_usize(&accessor["componentType"]), 5125);
                    assert_eq!(as_usize(&accessor["count"]) % 3, 0);
                }
            }
        }

        let nodes = list(doc, "nodes");
        let mut parents = vec![0usize; nodes.len()];
        for node in nodes {
            assert!(node["name"].is_string());
            if let Some(mesh) = node.get("mesh") {
                assert!(as_usize(mesh) < meshes.len(), "dangling node.mesh");
            }
            if let Some(children) = node.get("children") {
                let children = children.as_array().expect("children");
                assert!(!children.is_empty());
                let mut unique = HashSet::new();
                for child in children {
                    let child = as_usize(child);
                    assert!(child < nodes.len(), "dangling child index");
                    assert!(unique.insert(child), "duplicate child index");
                    parents[child] += 1;
                }
            }
            // Transforms are optional in glTF; when present they need the right arity.
            for (key, arity) in [("translation", 3), ("rotation", 4), ("scale", 3)] {
                if let Some(transform) = node.get(key) {
                    assert_eq!(transform.as_array().map(Vec::len), Some(arity), "{key}");
                }
            }
        }
        assert!(parents.iter().all(|&p| p <= 1), "a node has two parents");
        let roots: Vec<usize> = (0..nodes.len()).filter(|&i| parents[i] == 0).collect();
        let listed: Vec<usize> = doc["scenes"][0]
            .get("nodes")
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |n| n.iter().map(as_usize).collect());
        assert_eq!(
            listed, roots,
            "scenes[0].nodes must be exactly the root nodes"
        );
    }

    fn triangle(offset: f32) -> GltfPrimitive {
        GltfPrimitive {
            positions: vec![
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset, 1.0, -2.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            texcoords: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
        }
    }

    fn quad() -> GltfPrimitive {
        GltfPrimitive {
            positions: vec![
                [-1.0, -1.0, 0.5],
                [1.0, -1.0, 0.5],
                [1.0, 1.0, 0.5],
                [-1.0, 1.0, 0.5],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            texcoords: vec![],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    fn node(name: &str, mesh: Option<usize>, children: &[usize]) -> GltfNode {
        GltfNode {
            name: name.to_string(),
            mesh,
            children: children.to_vec(),
            ..GltfNode::default()
        }
    }

    /// Two meshes (2 + 1 primitives) under a small hierarchy with two roots.
    fn hierarchical_scene() -> GltfScene {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "body".into(),
            primitives: vec![quad(), triangle(3.0)],
        });
        scene.add_mesh(GltfMesh {
            name: "wheel".into(),
            primitives: vec![triangle(-4.0)],
        });
        let mut root = node("root", None, &[1, 2]);
        root.translation = [1.0, 2.0, 3.0];
        let mut body = node("body", Some(0), &[3]);
        body.rotation = [0.0, 0.5, 0.0, (0.75f64).sqrt()];
        let mut wheel = node("wheel", Some(1), &[]);
        wheel.scale = [2.0, 2.0, 2.0];
        scene.add_node(root);
        scene.add_node(body);
        scene.add_node(wheel);
        scene.add_node(node("marker", None, &[]));
        scene.add_node(node("second root", Some(0), &[]));
        scene
    }

    // ---- node graph -------------------------------------------------------------

    #[test]
    fn scene_json_emits_node_graph_and_one_mesh_per_gltf_mesh() {
        let scene = hierarchical_scene();
        let (json, buffer) = scene_to_gltf_external(&scene, "model.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);

        // scenes[0].nodes lists exactly the nodes that are nobody's child.
        let roots: Vec<usize> = doc["scenes"][0]["nodes"]
            .as_array()
            .expect("scene nodes")
            .iter()
            .map(as_usize)
            .collect();
        assert_eq!(roots, vec![0, 4]);

        // Nodes keep name, mesh, children and transform.
        let nodes = list(&doc, "nodes");
        assert_eq!(nodes.len(), scene.nodes.len());
        for node in nodes {
            assert_eq!(node["translation"].as_array().map(Vec::len), Some(3));
            assert_eq!(node["rotation"].as_array().map(Vec::len), Some(4));
            assert_eq!(node["scale"].as_array().map(Vec::len), Some(3));
        }
        assert_eq!(nodes[0]["name"], "root");
        assert_eq!(nodes[0]["children"], serde_json::json!([1, 2]));
        assert_eq!(floats(&nodes[0]["translation"]), vec![1.0, 2.0, 3.0]);
        assert!(nodes[0].get("mesh").is_none());
        assert_eq!(nodes[1]["mesh"], 0);
        assert_eq!(nodes[1]["children"], serde_json::json!([3]));
        assert_eq!(floats(&nodes[1]["rotation"])[1], 0.5);
        assert_eq!(nodes[2]["mesh"], 1);
        assert_eq!(floats(&nodes[2]["scale"]), vec![2.0, 2.0, 2.0]);
        assert_eq!(nodes[4]["mesh"], 0);

        // meshes.len() == scene.meshes.len(); every primitive of a mesh lives in
        // that mesh's entry, so node.mesh indexes scene.meshes directly.
        let meshes = list(&doc, "meshes");
        assert_eq!(meshes.len(), scene.meshes.len());
        assert_eq!(meshes[0]["name"], "body");
        assert_eq!(meshes[0]["primitives"].as_array().map(Vec::len), Some(2));
        assert_eq!(meshes[1]["name"], "wheel");
        assert_eq!(meshes[1]["primitives"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn scene_json_buffer_round_trips_the_vertex_data() {
        let scene = hierarchical_scene();
        let (json, buffer) = scene_to_gltf_external(&scene, "model.bin");
        let doc = parse(&json);
        assert_eq!(doc["buffers"][0]["uri"], "model.bin");
        assert_eq!(as_usize(&doc["buffers"][0]["byteLength"]), buffer.len());

        // Walk every primitive and compare the decoded accessors with the input.
        for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
            let primitives = list(&doc, "meshes")[mesh_index]["primitives"]
                .as_array()
                .expect("primitives");
            assert_eq!(primitives.len(), mesh.primitives.len());
            for (primitive, source) in primitives.iter().zip(&mesh.primitives) {
                let attributes = &primitive["attributes"];
                let positions = accessor_f32(&doc, &buffer, as_usize(&attributes["POSITION"]));
                let expected: Vec<f32> = source.positions.iter().flatten().copied().collect();
                assert_eq!(positions, expected);
                let normals = accessor_f32(&doc, &buffer, as_usize(&attributes["NORMAL"]));
                let expected: Vec<f32> = source.normals.iter().flatten().copied().collect();
                assert_eq!(normals, expected);
                let indices = accessor_u32(&doc, &buffer, as_usize(&primitive["indices"]));
                assert_eq!(indices, source.indices);
                if source.texcoords.is_empty() {
                    assert!(attributes.get("TEXCOORD_0").is_none());
                } else {
                    let uvs = accessor_f32(&doc, &buffer, as_usize(&attributes["TEXCOORD_0"]));
                    let expected: Vec<f32> = source.texcoords.iter().flatten().copied().collect();
                    assert_eq!(uvs, expected);
                }
            }
        }
    }

    #[test]
    fn position_accessor_has_tight_min_max() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "m".into(),
            primitives: vec![GltfPrimitive {
                positions: vec![[-1.5, 2.0, 7.0], [3.0, -4.0, 0.25], [0.0, 0.5, -9.0]],
                normals: vec![],
                texcoords: vec![],
                indices: vec![0, 1, 2],
            }],
        });
        let (json, _) = scene_to_gltf_external(&scene, "m.bin");
        let doc = parse(&json);
        let accessor = &list(&doc, "accessors")[0];
        assert_eq!(accessor["type"], "VEC3");
        assert_eq!(accessor["componentType"], 5126);
        assert_eq!(accessor["count"], 3);
        assert_eq!(floats(&accessor["min"]), vec![-1.5, -4.0, -9.0]);
        assert_eq!(floats(&accessor["max"]), vec![3.0, 2.0, 7.0]);
    }

    #[test]
    fn position_bounds_equal_the_buffer_values_exactly() {
        // 0.1, 1/3 and 1e-7 have no exact short decimal form as f32.
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "inexact".into(),
            primitives: vec![GltfPrimitive {
                positions: vec![
                    [0.1, -1.0 / 3.0, 1e-7],
                    [-0.7, 123_456.79, 3.0e-5],
                    [0.3, 0.2, -2.5e6],
                ],
                normals: vec![],
                texcoords: vec![],
                indices: vec![0, 1, 2],
            }],
        });
        let (json, buffer) = scene_to_gltf_external(&scene, "i.bin");
        // `assert_consistent` compares min/max with the decoded data bit for bit.
        assert_consistent(&parse(&json), &buffer);
        let doc = parse(&json);
        let accessor = &list(&doc, "accessors")[0];
        assert_eq!(floats(&accessor["min"])[0], f64::from(-0.7f32));
        assert_eq!(floats(&accessor["max"])[0], f64::from(0.3f32));
        assert_eq!(floats(&accessor["max"])[1], f64::from(123_456.79f32));
    }

    // ---- optional attributes -------------------------------------------------------

    #[test]
    fn normal_and_texcoord_accessors_only_when_lengths_match() {
        let mut scene = GltfScene::new();
        let mut short_normals = triangle(0.0);
        short_normals.normals.pop();
        let mut short_uvs = triangle(1.0);
        short_uvs.texcoords.truncate(1);
        let no_extras = GltfPrimitive {
            normals: vec![],
            texcoords: vec![],
            ..triangle(2.0)
        };
        scene.add_mesh(GltfMesh {
            name: "attrs".into(),
            primitives: vec![triangle(-1.0), short_normals, short_uvs, no_extras],
        });
        let (json, buffer) = scene_to_gltf_external(&scene, "a.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        let primitives = list(&doc, "meshes")[0]["primitives"]
            .as_array()
            .expect("primitives");
        let has =
            |primitive: usize, name: &str| primitives[primitive]["attributes"].get(name).is_some();
        assert!(has(0, "NORMAL") && has(0, "TEXCOORD_0"));
        assert!(!has(1, "NORMAL") && has(1, "TEXCOORD_0"));
        assert!(has(2, "NORMAL") && !has(2, "TEXCOORD_0"));
        assert!(!has(3, "NORMAL") && !has(3, "TEXCOORD_0"));
    }

    #[test]
    fn empty_index_list_skips_the_index_buffer_view() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "soup".into(),
            primitives: vec![GltfPrimitive {
                indices: vec![],
                ..triangle(0.0)
            }],
        });
        let (json, buffer) = scene_to_gltf_external(&scene, "soup.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        assert!(
            list(&doc, "meshes")[0]["primitives"][0]
                .get("indices")
                .is_none()
        );
        for view in list(&doc, "bufferViews") {
            assert_ne!(
                view["target"], 34963,
                "no element-array view without indices"
            );
            assert!(as_usize(&view["byteLength"]) > 0);
        }
        // positions + normals + uvs only
        assert_eq!(list(&doc, "bufferViews").len(), 3);
    }

    #[test]
    fn trailing_partial_triangle_is_not_exported() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "four".into(),
            primitives: vec![GltfPrimitive {
                indices: vec![0, 1, 2, 1],
                ..triangle(0.0)
            }],
        });
        let (json, buffer) = scene_to_gltf_external(&scene, "p.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        let indices = as_usize(&list(&doc, "meshes")[0]["primitives"][0]["indices"]);
        assert_eq!(list(&doc, "accessors")[indices]["count"], 3);
        assert_eq!(accessor_u32(&doc, &buffer, indices), vec![0, 1, 2]);

        // Fewer than three indices: nothing drawable is left, so no index accessor.
        let mut tiny = GltfScene::new();
        tiny.add_mesh(GltfMesh {
            name: "two".into(),
            primitives: vec![GltfPrimitive {
                indices: vec![0, 1],
                ..triangle(0.0)
            }],
        });
        let (json, buffer) = scene_to_gltf_external(&tiny, "t.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        assert!(
            list(&doc, "meshes")[0]["primitives"][0]
                .get("indices")
                .is_none()
        );
    }

    // ---- robustness ---------------------------------------------------------------

    #[test]
    fn names_are_escaped_so_the_json_stays_valid() {
        let hostile = "a\"b\\c\nd\te\u{1}f\u{1f}\u{7f}é日\u{1F600}</script>";
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: hostile.to_string(),
            primitives: vec![triangle(0.0)],
        });
        scene.add_node(node(hostile, Some(0), &[]));
        let (json, buffer) = scene_to_gltf_external(&scene, "weird \"name\".bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        assert_eq!(doc["meshes"][0]["name"], hostile);
        assert_eq!(doc["nodes"][0]["name"], hostile);
        assert_eq!(doc["buffers"][0]["uri"], "weird \"name\".bin");

        // Synthesized nodes reuse the (escaped) mesh name as well.
        let mut node_less = GltfScene::new();
        node_less.add_mesh(GltfMesh {
            name: hostile.to_string(),
            primitives: vec![triangle(0.0)],
        });
        let doc = parse(&scene_to_gltf_json(&node_less, true));
        assert_eq!(doc["nodes"][0]["name"], hostile);
    }

    #[test]
    fn escape_json_handles_every_control_character() {
        assert_eq!(escape_json("plain"), "plain");
        assert_eq!(escape_json("\"\\"), "\\\"\\\\");
        assert_eq!(escape_json("\n\r\t\u{8}\u{c}"), "\\n\\r\\t\\b\\f");
        assert_eq!(escape_json("\u{0}\u{1f}"), "\\u0000\\u001f");
        for code in 0u32..0x20 {
            let text = char::from_u32(code).expect("control character").to_string();
            let quoted = format!("\"{}\"", escape_json(&text));
            let parsed: String = serde_json::from_str(&quoted).expect("valid JSON string");
            assert_eq!(parsed, text, "U+{code:04X} must survive a round trip");
        }
    }

    #[test]
    fn invalid_references_are_dropped_instead_of_dangling() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "only".into(),
            primitives: vec![triangle(0.0)],
        });
        // Out-of-range mesh; out-of-range, self-referencing and repeated children.
        scene.add_node(node("a", Some(99), &[1, 1, 0, 7, 2]));
        scene.add_node(node("b", Some(0), &[]));
        scene.add_node(node("c", None, &[]));
        let (json, buffer) = scene_to_gltf_external(&scene, "r.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        let nodes = list(&doc, "nodes");
        assert!(nodes[0].get("mesh").is_none());
        assert_eq!(nodes[0]["children"], serde_json::json!([1, 2]));
        assert_eq!(nodes[1]["mesh"], 0);
        assert_eq!(doc["scenes"][0]["nodes"], serde_json::json!([0]));
    }

    #[test]
    fn meshes_without_drawable_primitives_are_dropped_and_indices_shift() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "empty".into(),
            primitives: vec![GltfPrimitive {
                positions: vec![],
                normals: vec![],
                texcoords: vec![],
                indices: vec![],
            }],
        });
        scene.add_mesh(GltfMesh {
            name: "no primitives".into(),
            primitives: vec![],
        });
        scene.add_mesh(GltfMesh {
            name: "real".into(),
            primitives: vec![
                GltfPrimitive {
                    positions: vec![],
                    normals: vec![],
                    texcoords: vec![],
                    indices: vec![],
                },
                triangle(0.0),
            ],
        });
        scene.add_node(node("n0", Some(0), &[]));
        scene.add_node(node("n1", Some(1), &[]));
        scene.add_node(node("n2", Some(2), &[]));
        let (json, buffer) = scene_to_gltf_external(&scene, "d.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        let meshes = list(&doc, "meshes");
        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0]["name"], "real");
        assert_eq!(meshes[0]["primitives"].as_array().map(Vec::len), Some(1));
        let nodes = list(&doc, "nodes");
        assert!(nodes[0].get("mesh").is_none());
        assert!(nodes[1].get("mesh").is_none());
        assert_eq!(nodes[2]["mesh"], 0);
    }

    #[test]
    fn non_finite_numbers_never_break_the_json() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "nan".into(),
            primitives: vec![GltfPrimitive {
                positions: vec![
                    [f32::NAN, 0.0, 0.0],
                    [f32::INFINITY, 1.0, 0.0],
                    [f32::NEG_INFINITY, 2.0, 0.0],
                ],
                normals: vec![],
                texcoords: vec![],
                indices: vec![0, 1, 2],
            }],
        });
        let mut bad_node = node("bad", Some(0), &[]);
        bad_node.translation = [f64::NAN, f64::INFINITY, 1.0];
        scene.add_node(bad_node);
        let (json, buffer) = scene_to_gltf_external(&scene, "n.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        assert_eq!(floats(&doc["nodes"][0]["translation"]), vec![0.0, 0.0, 1.0]);
        for bound in ["min", "max"] {
            for value in doc["accessors"][0][bound].as_array().expect("bound") {
                assert!(value.as_f64().is_some_and(f64::is_finite));
            }
        }
    }

    // ---- scene shapes ----------------------------------------------------------------

    #[test]
    fn empty_scene_is_a_valid_document_without_empty_arrays() {
        let (json, buffer) = scene_to_gltf_external(&GltfScene::new(), "unused.bin");
        assert!(buffer.is_empty());
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        assert!(doc.get("buffers").is_none());
        assert!(doc.get("meshes").is_none());
        assert_eq!(doc["scenes"], serde_json::json!([{}]));
        assert_eq!(json, scene_to_gltf_json(&GltfScene::new(), true));
        assert_eq!(json, scene_to_gltf_json(&GltfScene::new(), false));
    }

    #[test]
    fn node_only_scene_has_roots_but_no_buffer() {
        let mut scene = GltfScene::new();
        scene.add_node(node("parent", None, &[1]));
        scene.add_node(node("child", None, &[]));
        let (json, buffer) = scene_to_gltf_external(&scene, "x.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        assert!(buffer.is_empty());
        assert_eq!(doc["scenes"][0]["nodes"], serde_json::json!([0]));
    }

    #[test]
    fn meshes_without_nodes_get_one_root_node_each() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "first".into(),
            primitives: vec![triangle(0.0)],
        });
        scene.add_mesh(GltfMesh {
            name: "second".into(),
            primitives: vec![quad()],
        });
        let (json, buffer) = scene_to_gltf_external(&scene, "m.bin");
        let doc = parse(&json);
        assert_consistent(&doc, &buffer);
        assert_eq!(doc["scenes"][0]["nodes"], serde_json::json!([0, 1]));
        assert_eq!(doc["nodes"][0]["name"], "first");
        assert_eq!(doc["nodes"][0]["mesh"], 0);
        assert_eq!(doc["nodes"][1]["name"], "second");
        assert_eq!(doc["nodes"][1]["mesh"], 1);
    }

    // ---- buffer delivery -----------------------------------------------------------------

    #[test]
    fn embedded_buffer_decodes_to_the_external_bytes() {
        let scene = hierarchical_scene();
        let (external_json, external_buffer) = scene_to_gltf_external(&scene, "buffer.bin");
        let embedded = parse(&scene_to_gltf_json(&scene, true));
        let uri = embedded["buffers"][0]["uri"].as_str().expect("uri");
        let payload = uri
            .strip_prefix("data:application/octet-stream;base64,")
            .expect("a base64 data URI");
        assert_eq!(base64_decode(payload), external_buffer);
        assert_eq!(
            as_usize(&embedded["buffers"][0]["byteLength"]),
            external_buffer.len()
        );
        assert_consistent(&embedded, &external_buffer);

        // Everything but the buffer URI is identical between the two modes.
        let mut external = parse(&external_json);
        let mut embedded = embedded;
        external["buffers"][0]["uri"] = Value::Null;
        embedded["buffers"][0]["uri"] = Value::Null;
        assert_eq!(external, embedded);
    }

    #[test]
    fn non_embedded_json_references_buffer_bin() {
        let scene = hierarchical_scene();
        let doc = parse(&scene_to_gltf_json(&scene, false));
        assert_eq!(doc["buffers"][0]["uri"], "buffer.bin");
        let (_, buffer) = scene_to_gltf_external(&scene, "buffer.bin");
        assert_eq!(as_usize(&doc["buffers"][0]["byteLength"]), buffer.len());
    }

    // ---- behaviour carried over from the original exporter tests ------------------------

    #[test]
    fn test_scene_to_gltf_json_empty_scene() {
        let scene = GltfScene::new();
        let json = scene_to_gltf_json(&scene, false);
        assert!(
            json.contains("\"version\": \"2.0\""),
            "should contain asset version"
        );
        assert!(json.contains("OxiPhysics"), "should contain generator");
    }

    #[test]
    fn test_scene_to_gltf_json_with_mesh() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "triangle".into(),
            primitives: vec![GltfPrimitive {
                positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                normals: vec![[0.0, 0.0, 1.0]; 3],
                texcoords: vec![],
                indices: vec![0, 1, 2],
            }],
        });
        let json = scene_to_gltf_json(&scene, false);
        assert!(json.contains("triangle"), "should contain mesh name");
        assert!(json.contains("VEC3"), "should contain VEC3 accessor type");
        assert!(json.contains("SCALAR"), "should contain SCALAR for indices");
        // A node-less scene still instantiates its mesh.
        let doc = parse(&json);
        assert_eq!(doc["scenes"][0]["nodes"], serde_json::json!([0]));
    }

    #[test]
    fn test_scene_to_gltf_json_embed_buffer() {
        let mut scene = GltfScene::new();
        scene.add_mesh(GltfMesh {
            name: "cube".into(),
            primitives: vec![GltfPrimitive {
                positions: vec![[0.0, 0.0, 0.0]],
                normals: vec![[0.0, 1.0, 0.0]],
                texcoords: vec![],
                indices: vec![0],
            }],
        });
        let json = scene_to_gltf_json(&scene, true);
        assert!(
            json.contains("data:application/octet-stream;base64,"),
            "embedded buffer should use data URI"
        );
        let doc = parse(&json);
        let (_, buffer) = scene_to_gltf_external(&scene, "unused.bin");
        assert_consistent(&doc, &buffer);
    }
}
