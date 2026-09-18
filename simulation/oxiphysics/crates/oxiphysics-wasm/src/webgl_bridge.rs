// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! WebGL-compatible render data extraction for browser-based visualisation.
//!
//! This module provides:
//! - Procedural triangle mesh generation for each collider shape (sphere, box,
//!   capsule, cylinder, plane).
//! - Per-frame instance data arrays (positions, quaternions, scales, colours)
//!   extracted directly from a [`WasmPhysicsEngine`], ready for GPU-instanced
//!   rendering via `gl.drawArraysInstanced` / `gl.drawElementsInstanced`.
//! - Column-major 4×4 matrix helpers (perspective projection, look-at view,
//!   and per-body model matrix from position + quaternion + scale).
//!
//! ## WebGL workflow
//!
//! ```no_run
//! use oxiphysics_wasm::{WasmPhysicsEngine, webgl_bridge::{WebGlBridge, WebGlCamera}};
//!
//! # fn main() -> Result<(), oxiphysics_wasm::Error> {
//! let mut engine = WasmPhysicsEngine::new(0.0, -9.81, 0.0);
//! let b = engine.add_dynamic_body(1.0, 0.0, 5.0, 0.0)?;
//! engine.add_sphere_collider(b, 0.5)?;
//! engine.step(1.0 / 60.0)?;
//!
//! // Extract per-frame instance data (positions, rotations, …)
//! let frame = WebGlBridge::extract_render_frame(&engine);
//! assert_eq!(frame.body_count as usize, frame.positions().len() / 3);
//! # Ok(())
//! # }
//! ```

use wasm_bindgen::prelude::*;

use crate::engine::WasmPhysicsEngine;

// ============================================================================
// WebGlMesh — interleaved vertex + index buffer
// ============================================================================

/// Triangle mesh with interleaved vertex data suitable for a WebGL VBO.
///
/// Each vertex is stored as 8 consecutive `f32` values:
/// `[pos_x, pos_y, pos_z, nrm_x, nrm_y, nrm_z, uv_u, uv_v]`
///
/// Indices reference into that vertex array and describe triangles.
///
/// Every generated mesh winds its triangles counter-clockwise seen from the
/// side its vertex normals point to (the outside of a closed shape), which is
/// WebGL's default front face, so `gl.enable(gl.CULL_FACE)` keeps the visible
/// surface.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebGlMesh {
    /// Interleaved vertex data: `[pos(3) | normal(3) | uv(2)]` per vertex.
    pub vertex_count: u32,
    /// Number of triangles.
    pub triangle_count: u32,
    vertices: Vec<f32>,
    indices: Vec<u32>,
}

#[wasm_bindgen]
impl WebGlMesh {
    /// Return the interleaved vertex data as a `Vec<f32>`.
    pub fn vertices(&self) -> Vec<f32> {
        self.vertices.clone()
    }

    /// Return the triangle index list as a `Vec<u32>`.
    pub fn indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    /// Return the interleaved vertex data as a `js_sys::Float32Array`.
    pub fn vertices_typed(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.vertices.as_slice())
    }

    /// Return the triangle index list as a `js_sys::Uint32Array`.
    pub fn indices_typed(&self) -> js_sys::Uint32Array {
        js_sys::Uint32Array::from(self.indices.as_slice())
    }
}

impl WebGlMesh {
    fn new(vertices: Vec<f32>, indices: Vec<u32>) -> Self {
        let vertex_count = (vertices.len() / 8) as u32;
        let triangle_count = (indices.len() / 3) as u32;
        Self {
            vertices,
            indices,
            vertex_count,
            triangle_count,
        }
    }

    fn push_vertex(
        buf: &mut Vec<f32>,
        px: f32,
        py: f32,
        pz: f32,
        nx: f32,
        ny: f32,
        nz: f32,
        u: f32,
        v: f32,
    ) {
        buf.extend_from_slice(&[px, py, pz, nx, ny, nz, u, v]);
    }

    /// Return the vertex data slice (Rust-only).
    pub fn vertices_slice(&self) -> &[f32] {
        &self.vertices
    }

    /// Return the index slice (Rust-only).
    pub fn indices_slice(&self) -> &[u32] {
        &self.indices
    }
}

// ============================================================================
// WebGlBridge — geometry generators + render-frame extractor
// ============================================================================

/// Utility that generates procedural meshes and extracts per-frame render data.
#[wasm_bindgen]
pub struct WebGlBridge;

#[wasm_bindgen]
impl WebGlBridge {
    /// Generate a UV-sphere mesh with the given tessellation.
    ///
    /// - `stacks` — latitude bands (minimum 2).
    /// - `slices` — longitude segments (minimum 3).
    pub fn sphere_mesh(stacks: u32, slices: u32) -> WebGlMesh {
        Self::sphere_mesh_inner(stacks, slices)
    }

    /// Generate a unit-cube mesh (half-extent 0.5, 6 faces, 24 vertices).
    pub fn box_mesh() -> WebGlMesh {
        Self::box_mesh_inner()
    }

    /// Generate a capsule mesh with unit radius and height.
    pub fn capsule_mesh(stacks: u32, slices: u32) -> WebGlMesh {
        Self::capsule_mesh_inner(stacks, slices)
    }

    /// Generate a closed cylinder mesh (flat caps + smooth sides).
    pub fn cylinder_mesh(slices: u32) -> WebGlMesh {
        Self::cylinder_mesh_inner(slices)
    }

    /// Generate a flat square mesh centred at the origin, normal = +Y.
    ///
    /// `half_size` is the half-extent along X and Z.
    pub fn plane_mesh(half_size: f32) -> WebGlMesh {
        Self::plane_mesh_inner(half_size)
    }

    /// Extract per-frame instance arrays from a live `WasmPhysicsEngine`.
    pub fn extract_render_frame(engine: &WasmPhysicsEngine) -> WebGlRenderFrame {
        let handles = engine.get_all_body_handles();
        let raw = engine.get_all_transforms();
        let n = handles.len();
        debug_assert_eq!(raw.len(), n * 7, "transform stride mismatch");

        let mut positions = Vec::with_capacity(n * 3);
        let mut rotations = Vec::with_capacity(n * 4);
        let mut scales = Vec::with_capacity(n * 3);
        let mut colors = Vec::with_capacity(n * 4);
        let mut matrices = Vec::with_capacity(n * 16);

        for i in 0..n {
            let base = i * 7;
            let (px, py, pz) = (raw[base] as f32, raw[base + 1] as f32, raw[base + 2] as f32);
            let (qx, qy, qz, qw) = (
                raw[base + 3] as f32,
                raw[base + 4] as f32,
                raw[base + 5] as f32,
                raw[base + 6] as f32,
            );
            positions.extend_from_slice(&[px, py, pz]);
            rotations.extend_from_slice(&[qx, qy, qz, qw]);
            scales.extend_from_slice(&[1.0_f32, 1.0, 1.0]);
            colors.extend_from_slice(&[0.2_f32, 0.5, 0.9, 1.0]);

            let m = Self::model_matrix_inner([px, py, pz], [qx, qy, qz, qw], [1.0, 1.0, 1.0]);
            matrices.extend_from_slice(&m);
        }

        WebGlRenderFrame {
            handles,
            positions,
            rotations,
            scales,
            colors,
            model_matrices: matrices,
            body_count: n as u32,
        }
    }

    /// Build a column-major 4×4 model matrix from flat Vec inputs.
    ///
    /// `pos_flat` — `[px, py, pz]` (3 elements)
    /// `rot_flat` — `[qx, qy, qz, qw]` (4 elements)
    /// `scale_flat` — `[sx, sy, sz]` (3 elements)
    ///
    /// Returns a flat `Vec<f32>` of 16 elements.
    pub fn model_matrix_js(
        pos_flat: Vec<f32>,
        rot_flat: Vec<f32>,
        scale_flat: Vec<f32>,
    ) -> Vec<f32> {
        let pos = if pos_flat.len() >= 3 {
            [pos_flat[0], pos_flat[1], pos_flat[2]]
        } else {
            [0.0; 3]
        };
        let rot = if rot_flat.len() >= 4 {
            [rot_flat[0], rot_flat[1], rot_flat[2], rot_flat[3]]
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };
        let scale = if scale_flat.len() >= 3 {
            [scale_flat[0], scale_flat[1], scale_flat[2]]
        } else {
            [1.0; 3]
        };
        Self::model_matrix_inner(pos, rot, scale).to_vec()
    }

    /// Build a column-major perspective projection matrix as `Vec<f32>` (16 elements).
    pub fn perspective_matrix_js(fov_y: f32, aspect: f32, near: f32, far: f32) -> Vec<f32> {
        Self::perspective_matrix_inner(fov_y, aspect, near, far).to_vec()
    }

    /// Build a column-major look-at view matrix as `Vec<f32>` (16 elements).
    ///
    /// `eye_flat` — `[ex, ey, ez]`, `center_flat` — `[cx, cy, cz]`, `up_flat` — `[ux, uy, uz]`.
    pub fn look_at_matrix_js(
        eye_flat: Vec<f32>,
        center_flat: Vec<f32>,
        up_flat: Vec<f32>,
    ) -> Vec<f32> {
        let eye = if eye_flat.len() >= 3 {
            [eye_flat[0], eye_flat[1], eye_flat[2]]
        } else {
            [0.0; 3]
        };
        let center = if center_flat.len() >= 3 {
            [center_flat[0], center_flat[1], center_flat[2]]
        } else {
            [0.0; 3]
        };
        let up = if up_flat.len() >= 3 {
            [up_flat[0], up_flat[1], up_flat[2]]
        } else {
            [0.0, 1.0, 0.0]
        };
        Self::look_at_matrix_inner(eye, center, up).to_vec()
    }

    /// Multiply two column-major 4×4 matrices `a × b`, both as flat `Vec<f32>` (16 elements).
    pub fn mat4_mul_js(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
        let a_arr: [f32; 16] = a.try_into().unwrap_or([0.0; 16]);
        let b_arr: [f32; 16] = b.try_into().unwrap_or([0.0; 16]);
        Self::mat4_mul_inner(&a_arr, &b_arr).to_vec()
    }
}

impl WebGlBridge {
    // -----------------------------------------------------------------------
    // Sphere (UV sphere, unit radius, centred at origin)
    // -----------------------------------------------------------------------

    /// Generate a UV-sphere mesh (Rust-only; use `sphere_mesh` from JS).
    pub fn sphere_mesh_inner(stacks: u32, slices: u32) -> WebGlMesh {
        let stacks = stacks.max(2);
        let slices = slices.max(3);
        let mut verts: Vec<f32> = Vec::new();
        let mut idxs: Vec<u32> = Vec::new();

        for i in 0..=stacks {
            let v = i as f32 / stacks as f32;
            let theta = v * std::f32::consts::PI;
            let sin_t = theta.sin();
            let cos_t = theta.cos();

            for j in 0..=slices {
                let u = j as f32 / slices as f32;
                let phi = u * 2.0 * std::f32::consts::PI;
                let nx = sin_t * phi.cos();
                let ny = cos_t;
                let nz = sin_t * phi.sin();
                WebGlMesh::push_vertex(&mut verts, nx, ny, nz, nx, ny, nz, u, v);
            }
        }

        // Rows run from the north pole (+Y) down and the azimuth grows from +X
        // towards +Z, so `a -> a + 1` points east and `a -> b` points south.
        // `east x south` is the outward normal, hence `(a, a + 1, b)` and
        // `(b, a + 1, b + 1)` are counter-clockwise seen from outside.
        let row = slices + 1;
        for i in 0..stacks {
            for j in 0..slices {
                let a = i * row + j;
                let b = a + row;
                idxs.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
            }
        }
        WebGlMesh::new(verts, idxs)
    }

    // -----------------------------------------------------------------------
    // Box (axis-aligned unit cube, half-extent = 0.5)
    // -----------------------------------------------------------------------

    /// Generate a unit-cube mesh (Rust-only).
    pub fn box_mesh_inner() -> WebGlMesh {
        type FaceData = ([f32; 3], [[f32; 3]; 4], [[f32; 2]; 4]);
        #[rustfmt::skip]
        let face_data: &[FaceData] = &[
            ([0.,0.,1.],  [[-0.5,-0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5]],  [[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),
            ([0.,0.,-1.], [[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5]],[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),
            ([-1.,0.,0.], [[-0.5,-0.5,-0.5],[-0.5,-0.5,0.5],[-0.5,0.5,0.5],[-0.5,0.5,-0.5]],[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),
            ([1.,0.,0.],  [[0.5,-0.5,0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[0.5,0.5,0.5]],   [[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),
            ([0.,1.,0.],  [[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5]],  [[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),
            ([0.,-1.,0.], [[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,-0.5,0.5],[-0.5,-0.5,0.5]],[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),
        ];

        let mut verts: Vec<f32> = Vec::with_capacity(24 * 8);
        let mut idxs: Vec<u32> = Vec::with_capacity(36);

        for (fi, (normal, corners, uvs)) in face_data.iter().enumerate() {
            let base = (fi * 4) as u32;
            for k in 0..4 {
                WebGlMesh::push_vertex(
                    &mut verts,
                    corners[k][0],
                    corners[k][1],
                    corners[k][2],
                    normal[0],
                    normal[1],
                    normal[2],
                    uvs[k][0],
                    uvs[k][1],
                );
            }
            idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        WebGlMesh::new(verts, idxs)
    }

    // -----------------------------------------------------------------------
    // Capsule
    // -----------------------------------------------------------------------

    /// Generate a capsule mesh (Rust-only).
    ///
    /// `stacks` is raised to at least 4 and rounded down to an even number
    /// `s`, `slices` is raised to at least 3.  The vertex grid has `s + 2`
    /// rows of `slices + 1` vertices: rows `0..=s / 2` are the upper
    /// hemisphere, rows `s / 2 + 1..=s + 1` the lower one, and the strip
    /// between the two equator rows forms the cylindrical side.  The mesh
    /// therefore has `s + 1` strips of `slices` quads, i.e.
    /// `2 * slices * (s + 1)` triangles.
    pub fn capsule_mesh_inner(stacks: u32, slices: u32) -> WebGlMesh {
        let stacks = stacks.max(4) & !1;
        let slices = slices.max(3);
        let half = 0.5_f32;
        let hemi = stacks / 2;
        let mut verts: Vec<f32> = Vec::new();
        let mut idxs: Vec<u32> = Vec::new();
        let row = slices + 1;

        for i in 0..=hemi {
            let v = i as f32 / stacks as f32;
            let theta = v * std::f32::consts::PI;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            for j in 0..=slices {
                let u = j as f32 / slices as f32;
                let phi = u * 2.0 * std::f32::consts::PI;
                let nx = sin_t * phi.cos();
                let ny = cos_t;
                let nz = sin_t * phi.sin();
                WebGlMesh::push_vertex(&mut verts, nx, ny + half, nz, nx, ny, nz, u, v * 2.0);
            }
        }

        for i in hemi..=stacks {
            let v = i as f32 / stacks as f32;
            let theta = v * std::f32::consts::PI;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            for j in 0..=slices {
                let u = j as f32 / slices as f32;
                let phi = u * 2.0 * std::f32::consts::PI;
                let nx = sin_t * phi.cos();
                let ny = cos_t;
                let nz = sin_t * phi.sin();
                WebGlMesh::push_vertex(&mut verts, nx, ny - half, nz, nx, ny, nz, u, v * 2.0 - 1.0);
            }
        }

        // The two hemisphere loops above emit `hemi + 1` rows each, i.e.
        // `stacks + 2` rows in total.  A strip joins two consecutive rows, so
        // there are `stacks + 1` strips; the one between rows `hemi` (equator
        // of the upper hemisphere) and `hemi + 1` (equator of the lower one)
        // is the cylindrical side.
        //
        // Every row is laid out as in the sphere (north to south, azimuth from
        // +X towards +Z), so the same counter-clockwise quad split applies.
        let total_rows = stacks + 2;
        for i in 0..total_rows - 1 {
            for j in 0..slices {
                let a = i * row + j;
                let b = a + row;
                idxs.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
            }
        }
        WebGlMesh::new(verts, idxs)
    }

    // -----------------------------------------------------------------------
    // Cylinder
    // -----------------------------------------------------------------------

    /// Generate a closed cylinder mesh (Rust-only).
    pub fn cylinder_mesh_inner(slices: u32) -> WebGlMesh {
        let slices = slices.max(3);
        let mut verts: Vec<f32> = Vec::new();
        let mut idxs: Vec<u32> = Vec::new();

        let side_base = 0u32;
        for j in 0..=slices {
            let u = j as f32 / slices as f32;
            let phi = u * 2.0 * std::f32::consts::PI;
            let (cos_p, sin_p) = (phi.cos(), phi.sin());
            let nx = cos_p;
            let nz = sin_p;
            WebGlMesh::push_vertex(
                &mut verts,
                0.5 * cos_p,
                -0.5,
                0.5 * sin_p,
                nx,
                0.0,
                nz,
                u,
                0.0,
            );
            WebGlMesh::push_vertex(
                &mut verts,
                0.5 * cos_p,
                0.5,
                0.5 * sin_p,
                nx,
                0.0,
                nz,
                u,
                1.0,
            );
        }
        for j in 0..slices {
            let a = side_base + j * 2;
            idxs.extend_from_slice(&[a, a + 1, a + 2, a + 2, a + 1, a + 3]);
        }

        let top_base = (verts.len() / 8) as u32;
        WebGlMesh::push_vertex(&mut verts, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5);
        for j in 0..=slices {
            let phi = (j as f32 / slices as f32) * 2.0 * std::f32::consts::PI;
            let (cos_p, sin_p) = (phi.cos(), phi.sin());
            WebGlMesh::push_vertex(
                &mut verts,
                0.5 * cos_p,
                0.5,
                0.5 * sin_p,
                0.0,
                1.0,
                0.0,
                0.5 + 0.5 * cos_p,
                0.5 + 0.5 * sin_p,
            );
        }
        // The ring runs from +X towards +Z, i.e. clockwise seen from +Y, so the
        // top cap visits it backwards to be counter-clockwise from outside.
        let centre_t = top_base;
        for j in 0..slices {
            idxs.extend_from_slice(&[centre_t, top_base + 2 + j, top_base + 1 + j]);
        }

        let bot_base = (verts.len() / 8) as u32;
        WebGlMesh::push_vertex(&mut verts, 0.0, -0.5, 0.0, 0.0, -1.0, 0.0, 0.5, 0.5);
        for j in 0..=slices {
            let phi = (j as f32 / slices as f32) * 2.0 * std::f32::consts::PI;
            let (cos_p, sin_p) = (phi.cos(), phi.sin());
            WebGlMesh::push_vertex(
                &mut verts,
                0.5 * cos_p,
                -0.5,
                0.5 * sin_p,
                0.0,
                -1.0,
                0.0,
                0.5 + 0.5 * cos_p,
                0.5 - 0.5 * sin_p,
            );
        }
        // Seen from below (-Y) the same ring is counter-clockwise.
        let centre_b = bot_base;
        for j in 0..slices {
            idxs.extend_from_slice(&[centre_b, bot_base + 1 + j, bot_base + 2 + j]);
        }

        WebGlMesh::new(verts, idxs)
    }

    // -----------------------------------------------------------------------
    // Plane
    // -----------------------------------------------------------------------

    /// Generate a flat square mesh (Rust-only).
    pub fn plane_mesh_inner(half_size: f32) -> WebGlMesh {
        let h = half_size.max(0.001);
        let mut verts: Vec<f32> = Vec::new();
        let corners: &[[f32; 3]] = &[[-h, 0.0, -h], [h, 0.0, -h], [h, 0.0, h], [-h, 0.0, h]];
        let uvs: &[[f32; 2]] = &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for (c, uv) in corners.iter().zip(uvs.iter()) {
            WebGlMesh::push_vertex(&mut verts, c[0], c[1], c[2], 0.0, 1.0, 0.0, uv[0], uv[1]);
        }
        // Counter-clockwise seen from +Y, matching the +Y vertex normals.
        let idxs = vec![0u32, 2, 1, 0, 3, 2];
        WebGlMesh::new(verts, idxs)
    }

    // -----------------------------------------------------------------------
    // Matrix math helpers (Rust-only; use *_js variants from JS)
    // -----------------------------------------------------------------------

    /// Build a column-major 4×4 model matrix (Rust-only).
    pub fn model_matrix_inner(pos: [f32; 3], rot: [f32; 4], scale: [f32; 3]) -> [f32; 16] {
        let (qx, qy, qz, qw) = (rot[0], rot[1], rot[2], rot[3]);
        let (sx, sy, sz) = (scale[0], scale[1], scale[2]);

        let x2 = qx + qx;
        let y2 = qy + qy;
        let z2 = qz + qz;
        let xx = qx * x2;
        let xy = qx * y2;
        let xz = qx * z2;
        let yy = qy * y2;
        let yz = qy * z2;
        let zz = qz * z2;
        let wx = qw * x2;
        let wy = qw * y2;
        let wz = qw * z2;

        let r00 = 1.0 - (yy + zz);
        let r01 = xy - wz;
        let r02 = xz + wy;
        let r10 = xy + wz;
        let r11 = 1.0 - (xx + zz);
        let r12 = yz - wx;
        let r20 = xz - wy;
        let r21 = yz + wx;
        let r22 = 1.0 - (xx + yy);

        [
            r00 * sx,
            r10 * sx,
            r20 * sx,
            0.0,
            r01 * sy,
            r11 * sy,
            r21 * sy,
            0.0,
            r02 * sz,
            r12 * sz,
            r22 * sz,
            0.0,
            pos[0],
            pos[1],
            pos[2],
            1.0,
        ]
    }

    /// Public alias used by tests and render frame extraction.
    pub fn model_matrix(pos: [f32; 3], rot: [f32; 4], scale: [f32; 3]) -> [f32; 16] {
        Self::model_matrix_inner(pos, rot, scale)
    }

    /// Build a column-major perspective projection matrix (Rust-only).
    pub fn perspective_matrix_inner(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
        let f = 1.0 / (fov_y * 0.5).tan();
        let nf = 1.0 / (near - far);
        [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            (far + near) * nf,
            -1.0,
            0.0,
            0.0,
            2.0 * far * near * nf,
            0.0,
        ]
    }

    /// Public alias for perspective matrix.
    pub fn perspective_matrix(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
        Self::perspective_matrix_inner(fov_y, aspect, near, far)
    }

    /// Build a column-major look-at view matrix (Rust-only).
    pub fn look_at_matrix_inner(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
        let f = normalize(sub(center, eye));
        let r = normalize(cross(f, up));
        let u = cross(r, f);
        [
            r[0],
            u[0],
            -f[0],
            0.0,
            r[1],
            u[1],
            -f[1],
            0.0,
            r[2],
            u[2],
            -f[2],
            0.0,
            -dot(r, eye),
            -dot(u, eye),
            dot(f, eye),
            1.0,
        ]
    }

    /// Public alias for look-at matrix.
    pub fn look_at_matrix(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
        Self::look_at_matrix_inner(eye, center, up)
    }

    /// Multiply two column-major 4×4 matrices: `a × b` (Rust-only).
    pub fn mat4_mul_inner(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
        let mut m = [0.0_f32; 16];
        for col in 0..4usize {
            for row in 0..4usize {
                let mut s = 0.0_f32;
                for k in 0..4usize {
                    s += a[k * 4 + row] * b[col * 4 + k];
                }
                m[col * 4 + row] = s;
            }
        }
        m
    }

    /// Public alias for mat4 multiply.
    pub fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
        Self::mat4_mul_inner(a, b)
    }
}

// ============================================================================
// WebGlRenderFrame — per-frame extracted instance data
// ============================================================================

/// Per-frame render data extracted from the physics engine.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebGlRenderFrame {
    /// Body handles (one per body).
    handles: Vec<u32>,
    /// Flat position array: `[x₀,y₀,z₀, x₁,y₁,z₁, …]`.
    positions: Vec<f32>,
    /// Flat quaternion array: `[qx₀,qy₀,qz₀,qw₀, …]`.
    rotations: Vec<f32>,
    /// Flat scale array: `[sx₀,sy₀,sz₀, …]`.
    scales: Vec<f32>,
    /// Flat colour array: `[r₀,g₀,b₀,a₀, …]`.
    colors: Vec<f32>,
    /// Flat model-matrix array: 16 values per body, column-major.
    model_matrices: Vec<f32>,
    /// Number of bodies in this frame.
    pub body_count: u32,
}

#[wasm_bindgen]
impl WebGlRenderFrame {
    /// Return the body handles as a `Vec<u32>`.
    pub fn handles(&self) -> Vec<u32> {
        self.handles.clone()
    }

    /// Return the flat positions array as a `Vec<f32>`.
    pub fn positions(&self) -> Vec<f32> {
        self.positions.clone()
    }

    /// Return the flat rotations array as a `Vec<f32>`.
    pub fn rotations(&self) -> Vec<f32> {
        self.rotations.clone()
    }

    /// Return the flat scales array as a `Vec<f32>`.
    pub fn scales(&self) -> Vec<f32> {
        self.scales.clone()
    }

    /// Return the flat colors array as a `Vec<f32>`.
    pub fn colors(&self) -> Vec<f32> {
        self.colors.clone()
    }

    /// Return the flat model-matrices array as a `Vec<f32>`.
    pub fn model_matrices(&self) -> Vec<f32> {
        self.model_matrices.clone()
    }

    /// Return positions as a `js_sys::Float32Array`.
    pub fn positions_typed(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.positions.as_slice())
    }

    /// Return rotations as a `js_sys::Float32Array`.
    pub fn rotations_typed(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.rotations.as_slice())
    }

    /// Return model matrices as a `js_sys::Float32Array`.
    pub fn model_matrices_typed(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.model_matrices.as_slice())
    }

    /// Override the colour of a specific body (by index).
    pub fn set_color(&mut self, i: u32, r: f32, g: f32, b: f32, a: f32) {
        let base = (i as usize) * 4;
        if base + 3 < self.colors.len() {
            self.colors[base] = r;
            self.colors[base + 1] = g;
            self.colors[base + 2] = b;
            self.colors[base + 3] = a;
        }
    }

    /// Override the scale of a specific body (by index) and recompute its model matrix.
    pub fn set_scale(&mut self, i: u32, sx: f32, sy: f32, sz: f32) {
        let idx = i as usize;
        let base = idx * 3;
        if base + 2 < self.scales.len() {
            self.scales[base] = sx;
            self.scales[base + 1] = sy;
            self.scales[base + 2] = sz;
            let pos = self.position_inner(idx);
            let rot = self.rotation_inner(idx);
            let m = WebGlBridge::model_matrix_inner(pos, rot, [sx, sy, sz]);
            let mb = idx * 16;
            if mb + 16 <= self.model_matrices.len() {
                self.model_matrices[mb..mb + 16].copy_from_slice(&m);
            }
        }
    }
}

impl WebGlRenderFrame {
    /// Return the position of body at index `i` as `[x, y, z]` (Rust-only).
    pub fn position_inner(&self, i: usize) -> [f32; 3] {
        let b = i * 3;
        [
            self.positions[b],
            self.positions[b + 1],
            self.positions[b + 2],
        ]
    }

    /// Return the rotation of body at index `i` as `[qx, qy, qz, qw]` (Rust-only).
    pub fn rotation_inner(&self, i: usize) -> [f32; 4] {
        let b = i * 4;
        [
            self.rotations[b],
            self.rotations[b + 1],
            self.rotations[b + 2],
            self.rotations[b + 3],
        ]
    }

    /// Return the 4×4 model matrix of body at index `i` (Rust-only).
    pub fn model_matrix_slice(&self, i: usize) -> &[f32] {
        let b = i * 16;
        &self.model_matrices[b..b + 16]
    }
}

// ============================================================================
// WebGlCamera — camera with view and projection matrices
// ============================================================================

/// Camera state with precomputed view and projection matrices (column-major).
///
/// The `[f32; 16]` matrix fields are private to avoid wasm-bindgen
/// incompatibility; use the provided accessor methods.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebGlCamera {
    view_matrix: [f32; 16],
    projection_matrix: [f32; 16],
    eye_x: f32,
    eye_y: f32,
    eye_z: f32,
}

#[wasm_bindgen]
impl WebGlCamera {
    /// Construct a camera from look-at parameters and a perspective projection.
    ///
    /// `eye_flat`, `center_flat`, `up_flat` are `[x, y, z]` (3 elements each).
    pub fn new_js(
        eye_flat: Vec<f32>,
        center_flat: Vec<f32>,
        up_flat: Vec<f32>,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let eye = if eye_flat.len() >= 3 {
            [eye_flat[0], eye_flat[1], eye_flat[2]]
        } else {
            [0.0; 3]
        };
        let center = if center_flat.len() >= 3 {
            [center_flat[0], center_flat[1], center_flat[2]]
        } else {
            [0.0; 3]
        };
        let up = if up_flat.len() >= 3 {
            [up_flat[0], up_flat[1], up_flat[2]]
        } else {
            [0.0, 1.0, 0.0]
        };
        Self {
            view_matrix: WebGlBridge::look_at_matrix_inner(eye, center, up),
            projection_matrix: WebGlBridge::perspective_matrix_inner(fov_y, aspect, near, far),
            eye_x: eye[0],
            eye_y: eye[1],
            eye_z: eye[2],
        }
    }

    /// Return the view matrix as a `Vec<f32>` of 16 elements (JS-compatible).
    pub fn view_matrix_js(&self) -> Vec<f32> {
        self.view_matrix.to_vec()
    }

    /// Return the projection matrix as a `Vec<f32>` of 16 elements (JS-compatible).
    pub fn projection_matrix_js(&self) -> Vec<f32> {
        self.projection_matrix.to_vec()
    }

    /// Return the eye position as `Vec<f32>` of `[ex, ey, ez]`.
    pub fn eye_js(&self) -> Vec<f32> {
        vec![self.eye_x, self.eye_y, self.eye_z]
    }

    /// Compute the combined view-projection matrix (`projection × view`) as `Vec<f32>`.
    pub fn view_projection_js(&self) -> Vec<f32> {
        WebGlBridge::mat4_mul_inner(&self.projection_matrix, &self.view_matrix).to_vec()
    }

    /// Update the view matrix with a new eye position and target.
    ///
    /// `eye_flat`, `center_flat`, `up_flat` are `[x, y, z]` (3 elements each).
    pub fn set_look_at_js(&mut self, eye_flat: Vec<f32>, center_flat: Vec<f32>, up_flat: Vec<f32>) {
        let eye = if eye_flat.len() >= 3 {
            [eye_flat[0], eye_flat[1], eye_flat[2]]
        } else {
            [self.eye_x, self.eye_y, self.eye_z]
        };
        let center = if center_flat.len() >= 3 {
            [center_flat[0], center_flat[1], center_flat[2]]
        } else {
            [0.0; 3]
        };
        let up = if up_flat.len() >= 3 {
            [up_flat[0], up_flat[1], up_flat[2]]
        } else {
            [0.0, 1.0, 0.0]
        };
        self.eye_x = eye[0];
        self.eye_y = eye[1];
        self.eye_z = eye[2];
        self.view_matrix = WebGlBridge::look_at_matrix_inner(eye, center, up);
    }

    /// Update the projection (e.g. on window resize).
    pub fn set_perspective(&mut self, fov_y: f32, aspect: f32, near: f32, far: f32) {
        self.projection_matrix = WebGlBridge::perspective_matrix_inner(fov_y, aspect, near, far);
    }
}

impl WebGlCamera {
    /// Construct a camera from typed array arguments (Rust-only).
    pub fn look_at(
        eye: [f32; 3],
        center: [f32; 3],
        up: [f32; 3],
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            view_matrix: WebGlBridge::look_at_matrix_inner(eye, center, up),
            projection_matrix: WebGlBridge::perspective_matrix_inner(fov_y, aspect, near, far),
            eye_x: eye[0],
            eye_y: eye[1],
            eye_z: eye[2],
        }
    }

    /// Return the view-projection matrix as `[f32; 16]` (Rust-only).
    pub fn view_projection(&self) -> [f32; 16] {
        WebGlBridge::mat4_mul_inner(&self.projection_matrix, &self.view_matrix)
    }

    /// Update the view matrix (Rust-only).
    pub fn set_look_at(&mut self, eye: [f32; 3], center: [f32; 3], up: [f32; 3]) {
        self.eye_x = eye[0];
        self.eye_y = eye[1];
        self.eye_z = eye[2];
        self.view_matrix = WebGlBridge::look_at_matrix_inner(eye, center, up);
    }

    /// Return eye position as `[f32; 3]` (Rust-only).
    pub fn eye(&self) -> [f32; 3] {
        [self.eye_x, self.eye_y, self.eye_z]
    }
}

// ============================================================================
// Internal vector helpers
// ============================================================================

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-7 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_mesh_vertex_count() {
        let m = WebGlBridge::sphere_mesh_inner(8, 8);
        assert_eq!(m.vertices.len(), m.vertex_count as usize * 8);
        assert_eq!(m.indices.len(), m.triangle_count as usize * 3);
        assert!(m.triangle_count > 0);
    }

    #[test]
    fn box_mesh_correct() {
        let m = WebGlBridge::box_mesh_inner();
        assert_eq!(m.vertex_count, 24);
        assert_eq!(m.triangle_count, 12);
    }

    #[test]
    fn capsule_mesh_valid() {
        let m = WebGlBridge::capsule_mesh_inner(8, 8);
        assert_eq!(m.vertices.len(), m.vertex_count as usize * 8);
        assert!(m.triangle_count > 0);
    }

    #[test]
    fn cylinder_mesh_valid() {
        let m = WebGlBridge::cylinder_mesh_inner(8);
        assert_eq!(m.vertices.len(), m.vertex_count as usize * 8);
        assert!(m.triangle_count > 0);
    }

    #[test]
    fn plane_mesh_valid() {
        let m = WebGlBridge::plane_mesh_inner(10.0);
        assert_eq!(m.vertex_count, 4);
        assert_eq!(m.triangle_count, 2);
    }

    #[test]
    fn model_matrix_identity_quat() {
        let m =
            WebGlBridge::model_matrix_inner([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
        assert!((m[0] - 1.0).abs() < 1e-6);
        assert!((m[5] - 1.0).abs() < 1e-6);
        assert!((m[10] - 1.0).abs() < 1e-6);
        assert!((m[15] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn model_matrix_translation() {
        let m =
            WebGlBridge::model_matrix_inner([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]);
        assert!((m[12] - 1.0).abs() < 1e-6);
        assert!((m[13] - 2.0).abs() < 1e-6);
        assert!((m[14] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn perspective_matrix_valid() {
        let p = WebGlBridge::perspective_matrix_inner(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            1000.0,
        );
        assert!((p[11] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn look_at_matrix_orthogonal() {
        let v =
            WebGlBridge::look_at_matrix_inner([0.0, 3.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let c0 = [v[0], v[1], v[2]];
        let c1 = [v[4], v[5], v[6]];
        assert!((dot(c0, c0) - 1.0).abs() < 1e-5);
        assert!((dot(c1, c1) - 1.0).abs() < 1e-5);
        assert!(dot(c0, c1).abs() < 1e-5);
    }

    #[test]
    fn extract_render_frame_counts() {
        let mut engine = WasmPhysicsEngine::new(0.0, -9.81, 0.0);
        for i in 0..5 {
            let b = engine
                .add_dynamic_body(1.0, i as f64, 0.0, 0.0)
                .expect("add dynamic body");
            engine
                .add_sphere_collider(b, 0.5)
                .expect("add sphere collider");
        }
        let frame = WebGlBridge::extract_render_frame(&engine);
        assert_eq!(frame.body_count, 5);
        assert_eq!(frame.positions.len(), 15);
        assert_eq!(frame.rotations.len(), 20);
        assert_eq!(frame.model_matrices.len(), 80);
    }

    #[test]
    fn camera_view_projection_valid() {
        let cam = WebGlCamera::look_at(
            [0.0, 3.0, 10.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            1000.0,
        );
        let vp = cam.view_projection();
        assert!(vp.iter().any(|&v| v.abs() > 1e-6));
    }

    // -----------------------------------------------------------------------
    // Index-buffer validity (F5420)
    // -----------------------------------------------------------------------

    /// Panics unless `mesh` is a well-formed indexed triangle list.
    ///
    /// The index list must describe whole triangles, every index must address
    /// an existing vertex (WebGL rejects a `drawElements` call that reads past
    /// the vertex buffer) and every vertex must be referenced by at least one
    /// triangle, so a dangling index and a silently dropped strip are both
    /// caught.
    fn assert_indices_valid(mesh: &WebGlMesh, what: &str) {
        let vertex_count = mesh.vertex_count;
        assert!(vertex_count > 0, "{what}: the mesh has no vertices");
        assert_eq!(
            mesh.vertices.len(),
            vertex_count as usize * 8,
            "{what}: vertex buffer length"
        );
        assert_eq!(
            mesh.indices.len() % 3,
            0,
            "{what}: {} indices do not form whole triangles",
            mesh.indices.len()
        );
        assert_eq!(
            mesh.indices.len(),
            mesh.triangle_count as usize * 3,
            "{what}: triangle_count disagrees with the index buffer"
        );
        let mut referenced = vec![false; vertex_count as usize];
        for &ix in &mesh.indices {
            assert!(
                ix < vertex_count,
                "{what}: index {ix} out of range for {vertex_count} vertices"
            );
            referenced[ix as usize] = true;
        }
        let unused = referenced.iter().filter(|&&used| !used).count();
        assert_eq!(unused, 0, "{what}: {unused} vertices are never referenced");
    }

    /// Interleaved attributes `[pos(3) | normal(3) | uv(2)]` of vertex `k`.
    fn vertex(mesh: &WebGlMesh, k: usize) -> [f32; 8] {
        let base = k * 8;
        let mut v = [0.0_f32; 8];
        v.copy_from_slice(&mesh.vertices[base..base + 8]);
        v
    }

    /// Position of vertex `k` in double precision.
    fn position(mesh: &WebGlMesh, k: usize) -> [f64; 3] {
        let v = vertex(mesh, k);
        [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
    }

    /// Total area of all triangles of `mesh`.
    fn surface_area(mesh: &WebGlMesh) -> f64 {
        mesh.indices
            .chunks_exact(3)
            .map(|tri| {
                let a = position(mesh, tri[0] as usize);
                let b = position(mesh, tri[1] as usize);
                let c = position(mesh, tri[2] as usize);
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let cross = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
            })
            .sum()
    }

    #[test]
    fn every_generated_mesh_has_valid_indices() {
        for (stacks, slices) in [(0, 0), (2, 3), (8, 8), (16, 32)] {
            assert_indices_valid(
                &WebGlBridge::sphere_mesh_inner(stacks, slices),
                &format!("sphere({stacks}, {slices})"),
            );
        }
        assert_indices_valid(&WebGlBridge::box_mesh_inner(), "box");
        for slices in [0, 3, 8, 32] {
            assert_indices_valid(
                &WebGlBridge::cylinder_mesh_inner(slices),
                &format!("cylinder({slices})"),
            );
        }
        assert_indices_valid(&WebGlBridge::plane_mesh_inner(10.0), "plane");
        assert_indices_valid(&WebGlBridge::plane_mesh_inner(-1.0), "plane (clamped size)");
    }

    #[test]
    fn capsule_mesh_indices_stay_inside_the_vertex_buffer() {
        // Stacks are rounded down to an even number and raised to at least 4,
        // so 7 stacks give 6 and 0..=3 give 4.
        for (stacks, slices) in [
            (4, 3),
            (4, 8),
            (7, 8),
            (8, 8),
            (8, 16),
            (33, 5),
            (0, 0),
            (1, 1),
        ] {
            let mesh = WebGlBridge::capsule_mesh_inner(stacks, slices);
            assert_indices_valid(&mesh, &format!("capsule({stacks}, {slices})"));
        }
    }

    #[test]
    fn capsule_mesh_has_one_strip_between_every_pair_of_rows() {
        let slices = 6;
        for (requested, effective) in [(0, 4), (3, 4), (4, 4), (7, 6), (8, 8), (9, 8)] {
            let mesh = WebGlBridge::capsule_mesh_inner(requested, slices);
            // Two hemispheres of `effective / 2 + 1` rows each.
            let rows = effective + 2;
            assert_eq!(
                mesh.vertex_count,
                rows * (slices + 1),
                "vertices for {requested} stacks"
            );
            // `rows - 1` strips of `slices` quads, two triangles per quad.
            assert_eq!(
                mesh.triangle_count,
                2 * slices * (rows - 1),
                "triangles for {requested} stacks"
            );
            let highest = mesh.indices_slice().iter().copied().max();
            assert_eq!(
                highest,
                Some(mesh.vertex_count - 1),
                "the last vertex row is used for {requested} stacks"
            );
        }
    }

    #[test]
    fn capsule_mesh_side_strip_joins_the_two_equators() {
        let (stacks, slices) = (8_u32, 12_u32);
        let mesh = WebGlBridge::capsule_mesh_inner(stacks, slices);
        let row = (slices + 1) as usize;
        let hemi = (stacks / 2) as usize;

        // Equator of the upper hemisphere (row `hemi`) and of the lower one
        // (row `hemi + 1`): same azimuth and radius, 1 apart in height.
        for j in 0..row {
            let upper = position(&mesh, hemi * row + j);
            let lower = position(&mesh, (hemi + 1) * row + j);
            assert!((upper[1] - 0.5).abs() < 1e-6, "upper equator height");
            assert!((lower[1] + 0.5).abs() < 1e-6, "lower equator height");
            assert!((upper[0] - lower[0]).abs() < 1e-6, "same x");
            assert!((upper[2] - lower[2]).abs() < 1e-6, "same z");
            let radius = upper[0].hypot(upper[2]);
            assert!((radius - 1.0).abs() < 1e-6, "equator radius {radius}");
        }

        // The strip between those rows is the cylindrical side: it is emitted
        // after `hemi` hemisphere strips and only touches the two equators.
        let per_strip = slices as usize * 6;
        let side = &mesh.indices_slice()[hemi * per_strip..(hemi + 1) * per_strip];
        for &ix in side {
            let r = ix as usize / row;
            assert!(r == hemi || r == hemi + 1, "side vertex in row {r}");
        }
        for tri in side.chunks_exact(3) {
            let ys: Vec<f64> = tri
                .iter()
                .map(|&k| position(&mesh, k as usize)[1])
                .collect();
            let top = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let bottom = ys.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                (top - 0.5).abs() < 1e-6 && (bottom + 0.5).abs() < 1e-6,
                "side triangle spans y in [{bottom}, {top}]"
            );
        }
    }

    #[test]
    fn capsule_mesh_surface_matches_a_capsule() {
        // Two unit hemispheres plus a cylinder of height 1:
        // 4 pi r^2 + 2 pi r h.  A dropped strip (the cylinder) or a strip that
        // joins the wrong rows would miss this by far more than the 1 %
        // tessellation error.
        let exact = 4.0 * std::f64::consts::PI + 2.0 * std::f64::consts::PI;
        let mesh = WebGlBridge::capsule_mesh_inner(32, 64);
        let area = surface_area(&mesh);
        assert!(
            (area - exact).abs() / exact < 0.01,
            "capsule area {area} should be close to {exact}"
        );

        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..mesh.vertex_count as usize {
            let y = position(&mesh, k)[1];
            low = low.min(y);
            high = high.max(y);
        }
        assert!((low + 1.5).abs() < 1e-6 && (high - 1.5).abs() < 1e-6);
    }

    #[test]
    fn sphere_mesh_surface_matches_a_sphere() {
        let exact = 4.0 * std::f64::consts::PI;
        let area = surface_area(&WebGlBridge::sphere_mesh_inner(32, 64));
        assert!(
            (area - exact).abs() / exact < 0.01,
            "sphere area {area} should be close to {exact}"
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn index_validator_rejects_the_old_capsule_index_buffer() {
        // Rebuild the buffer the generator used to emit: one strip more than
        // there are pairs of rows, which reads one vertex row past the end.
        let (stacks, slices) = (4_u32, 3_u32);
        let good = WebGlBridge::capsule_mesh_inner(stacks, slices);
        let row = slices + 1;
        let mut indices = good.indices_slice().to_vec();
        for j in 0..slices {
            let a = (stacks + 1) * row + j;
            let b = a + row;
            indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
        }
        let broken = WebGlMesh::new(good.vertices_slice().to_vec(), indices);
        assert_indices_valid(&broken, "pre-fix capsule");
    }

    // -----------------------------------------------------------------------
    // Triangle winding: counter-clockwise seen from outside (G008)
    // -----------------------------------------------------------------------

    fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Shading normal stored at vertex `k`, in double precision.
    fn stored_normal(mesh: &WebGlMesh, k: usize) -> [f64; 3] {
        let v = vertex(mesh, k);
        [f64::from(v[3]), f64::from(v[4]), f64::from(v[5])]
    }

    /// Panics unless every triangle of `mesh` with a real orientation is wound
    /// counter-clockwise seen from outside.
    ///
    /// A triangle passes when its geometric normal `(p1 - p0) x (p2 - p0)`
    /// points the same way as the mean of the shading normals stored at its
    /// vertices *and* away from the interior, `outward(centroid)` being a
    /// direction that leaves the solid at the triangle centroid.  Collapsed
    /// triangles (the pole fans of a sphere, whose vertices only differ by the
    /// `f32` rounding of `sin(pi)`) have no orientation and are skipped: a
    /// triangle counts as collapsed when its doubled area is below `1e-5` times
    /// its squared longest edge.  At least half of the triangles must be real,
    /// so a degenerate mesh cannot pass.
    fn assert_outward_winding(
        mesh: &WebGlMesh,
        what: &str,
        outward: impl Fn([f64; 3]) -> [f64; 3],
    ) {
        let mut checked = 0_usize;
        for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|k| position(mesh, k as usize));
            let (ab, bc, ca) = (sub3(b, a), sub3(c, b), sub3(a, c));
            let geometric = cross3(ab, sub3(c, a));
            let doubled_area = dot3(geometric, geometric).sqrt();
            let longest_edge_sq = dot3(ab, ab).max(dot3(bc, bc)).max(dot3(ca, ca));
            if doubled_area < 1e-5 * longest_edge_sq {
                continue;
            }
            let normals = [tri[0], tri[1], tri[2]].map(|k| stored_normal(mesh, k as usize));
            let shading: [f64; 3] =
                std::array::from_fn(|i| normals[0][i] + normals[1][i] + normals[2][i]);
            let centroid: [f64; 3] = std::array::from_fn(|i| (a[i] + b[i] + c[i]) / 3.0);
            assert!(
                dot3(geometric, shading) > 0.0,
                "{what}: triangle {t} is wound clockwise seen from outside, its normal \
                 {geometric:?} opposes the shading normal {shading:?}"
            );
            assert!(
                dot3(geometric, outward(centroid)) > 0.0,
                "{what}: triangle {t} is wound clockwise seen from outside, its normal \
                 {geometric:?} points into the solid at {centroid:?}"
            );
            checked += 1;
        }
        assert!(
            checked * 2 >= mesh.triangle_count as usize,
            "{what}: only {checked} of {} triangles have an orientation",
            mesh.triangle_count
        );
    }

    /// Volume enclosed by a closed mesh from the divergence theorem,
    /// `sum(p0 . (p1 x p2)) / 6`; positive for counter-clockwise (outward)
    /// winding and negative for clockwise winding.
    fn signed_volume(mesh: &WebGlMesh) -> f64 {
        mesh.indices
            .chunks_exact(3)
            .map(|tri| {
                let [a, b, c] = [tri[0], tri[1], tri[2]].map(|k| position(mesh, k as usize));
                dot3(a, cross3(b, c)) / 6.0
            })
            .sum()
    }

    /// Direction that leaves a capsule of half height 0.5 at point `p`: away
    /// from the nearest point of its axis segment.
    fn capsule_outward(p: [f64; 3]) -> [f64; 3] {
        [p[0], p[1] - p[1].clamp(-0.5, 0.5), p[2]]
    }

    /// Direction that leaves the cylinder (radius 0.5, height 1) at `p`: along
    /// the axis on the caps, radially on the side.
    fn cylinder_outward(p: [f64; 3]) -> [f64; 3] {
        if p[1].abs() > 0.4 {
            [0.0, p[1].signum(), 0.0]
        } else {
            [p[0], 0.0, p[2]]
        }
    }

    #[test]
    fn sphere_triangles_are_wound_counter_clockwise_from_outside() {
        for (stacks, slices) in [(2, 3), (5, 7), (8, 8), (16, 32)] {
            let mesh = WebGlBridge::sphere_mesh_inner(stacks, slices);
            assert_outward_winding(&mesh, &format!("sphere({stacks}, {slices})"), |p| p);
        }
    }

    #[test]
    fn capsule_triangles_are_wound_counter_clockwise_from_outside() {
        for (stacks, slices) in [(4, 3), (7, 5), (8, 8), (16, 32)] {
            let mesh = WebGlBridge::capsule_mesh_inner(stacks, slices);
            assert_outward_winding(
                &mesh,
                &format!("capsule({stacks}, {slices})"),
                capsule_outward,
            );
        }
    }

    #[test]
    fn cylinder_box_and_plane_triangles_are_wound_counter_clockwise_from_outside() {
        for slices in [3, 8, 32] {
            let mesh = WebGlBridge::cylinder_mesh_inner(slices);
            assert_outward_winding(&mesh, &format!("cylinder({slices})"), cylinder_outward);
        }
        assert_outward_winding(&WebGlBridge::box_mesh_inner(), "box", |p| p);
        // The plane is not closed: it faces +Y, like its vertex normals.
        assert_outward_winding(&WebGlBridge::plane_mesh_inner(2.5), "plane", |_| {
            [0.0, 1.0, 0.0]
        });
    }

    #[test]
    fn closed_meshes_enclose_a_positive_volume() {
        // A clockwise mesh encloses the negative of its volume, so this pins the
        // winding of the whole surface at once.
        let pi = std::f64::consts::PI;
        let sphere = signed_volume(&WebGlBridge::sphere_mesh_inner(32, 64));
        assert!(
            (sphere - 4.0 / 3.0 * pi).abs() / (4.0 / 3.0 * pi) < 0.01,
            "sphere volume {sphere}"
        );
        // Unit hemispheres plus a cylinder of radius 1 and height 1.
        let capsule_exact = 4.0 / 3.0 * pi + pi;
        let capsule = signed_volume(&WebGlBridge::capsule_mesh_inner(32, 64));
        assert!(
            (capsule - capsule_exact).abs() / capsule_exact < 0.01,
            "capsule volume {capsule}"
        );
        // Radius 0.5, height 1.
        let cylinder_exact = pi * 0.25;
        let cylinder = signed_volume(&WebGlBridge::cylinder_mesh_inner(64));
        assert!(
            (cylinder - cylinder_exact).abs() / cylinder_exact < 0.01,
            "cylinder volume {cylinder}"
        );
        let cube = signed_volume(&WebGlBridge::box_mesh_inner());
        assert!((cube - 1.0).abs() < 1e-6, "box volume {cube}");
    }

    #[test]
    #[should_panic(expected = "wound clockwise")]
    fn winding_validator_rejects_the_old_clockwise_sphere() {
        // Rebuild the index buffer the generator used to emit, which split every
        // quad into `(a, b, a + 1)` and `(b, b + 1, a + 1)`.
        let (stacks, slices) = (6_u32, 8_u32);
        let good = WebGlBridge::sphere_mesh_inner(stacks, slices);
        let row = slices + 1;
        let mut indices = Vec::new();
        for i in 0..stacks {
            for j in 0..slices {
                let a = i * row + j;
                let b = a + row;
                indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
            }
        }
        let old = WebGlMesh::new(good.vertices_slice().to_vec(), indices);
        assert_outward_winding(&old, "pre-fix sphere", |p| p);
    }

    #[test]
    fn the_old_clockwise_sphere_enclosed_a_negative_volume() {
        let (stacks, slices) = (16_u32, 32_u32);
        let good = WebGlBridge::sphere_mesh_inner(stacks, slices);
        let row = slices + 1;
        let mut indices = Vec::new();
        for i in 0..stacks {
            for j in 0..slices {
                let a = i * row + j;
                let b = a + row;
                indices.extend_from_slice(&[a, b, a + 1, b, b + 1, a + 1]);
            }
        }
        let old = WebGlMesh::new(good.vertices_slice().to_vec(), indices);
        assert!(signed_volume(&old) < -4.0, "volume {}", signed_volume(&old));
        assert!(
            signed_volume(&good) > 4.0,
            "volume {}",
            signed_volume(&good)
        );
    }
}
