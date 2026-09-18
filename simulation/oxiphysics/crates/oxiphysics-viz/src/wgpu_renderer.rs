// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU/wgpu rendering backend for OxiPhysics visualisations.
//!
//! This module provides a feature-gated GPU renderer that uses the
//! [`oxiphysics_gpu::compute::WgpuBackend`] from
//! `oxiphysics-gpu` as its device abstraction.
//!
//! When the `wgpu-backend` feature on `oxiphysics-gpu` is active and a
//! compatible GPU adapter is found, rendering is performed on the GPU.
//! Otherwise [`GpuRenderer`] falls back to the software rasteriser from
//! [`mesh_render`](crate::mesh_render).
//!
//! # Feature flag
//!
//! The GPU renderer is gated behind the `wgpu-renderer` feature on this crate
//! (planned for v0.2.0):
//!
//! ```toml
//! [dependencies]
//! oxiphysics-viz = { features = ["wgpu-renderer"] }
//! ```
//!
//! ## Architecture
//!
//! ```text
//!  GpuRenderer
//!   ├── WgpuBackend (from oxiphysics-gpu)
//!   │     ├── depth_buffer   : WgpuBufferHandle   (f32, w×h)
//!   │     ├── color_buffer   : WgpuBufferHandle   (f32 RGBA, w×h×4)
//!   │     └── scratch        : WgpuBufferHandle   (staging)
//!   ├── ShaderRegistry
//!   │     ├── "vertex_transform"   WGSL vertex + Phong lighting shader
//!   │     ├── "particle_splat"     WGSL particle splatting shader
//!   │     └── "fullscreen_blit"    WGSL tone-map + gamma correction
//!   └── render_target: GpuRenderTarget
//! ```
//!
//! ## Render pipeline
//!
//! 1. **Geometry upload** — mesh vertices / indices written to GPU buffers.
//! 2. **Vertex transform dispatch** — project vertices using MVP matrix,
//!    compute per-vertex Phong diffuse + specular.
//! 3. **Rasterise** — a software-style dispatch fills color_buffer with
//!    Phong-shaded fragments; depth-test is performed against depth_buffer.
//! 4. **Post-process** — optional bloom, ACES tone-map, gamma correction.
//! 5. **Present** — colour buffer downloaded to an [`HdrFramebuffer`].
//!
//! ## Usage
//!
//! ```
//! use oxiphysics_viz::wgpu_renderer::{GpuRenderer, GpuRendererConfig, GpuLight};
//!
//! let config = GpuRendererConfig {
//!     width: 1280, height: 720,
//!     ..GpuRendererConfig::default()
//! };
//! let mut renderer = GpuRenderer::new(config);
//!
//! // Upload a simple triangle
//! let verts = vec![[0.0f32, 1.0, 0.0, 1.0],  // xyzw
//!                  [-1.0, -1.0, 0.0, 1.0],
//!                  [ 1.0, -1.0, 0.0, 1.0]];
//! let indices = vec![0u32, 1, 2];
//! let mesh_id = renderer.upload_mesh(&verts, &indices);
//!
//! // Render one frame
//! let identity = [1.0f32,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.];
//! renderer.begin_frame();
//! renderer.draw_mesh(mesh_id, &identity);
//! renderer.end_frame();
//!
//! // Download result to HDR framebuffer
//! let fb = renderer.to_hdr_framebuffer();
//! assert_eq!(fb.width, 1280);
//! ```

use crate::hdr_framebuffer::{HdrFramebuffer, ToneMapMethod, ToneMapper};
use oxiphysics_gpu::compute::{WgpuBackend, WgpuBufferHandle};

// ── WGSL shaders ─────────────────────────────────────────────────────────────

/// WGSL source: vertex transform + Phong shading.
///
/// Inputs: vertex buffer (`array<vec4<f32>>` = xyzw positions),
///         index buffer (`array<u32>`),
///         MVP matrix uniform.
/// Output: writes to the colour and depth compute buffers (storage).
pub const WGSL_VERTEX_PHONG: &str = r#"
struct Uniforms {
    mvp:        mat4x4<f32>,
    mv:         mat4x4<f32>,   // model-view (for normals)
    light_dir:  vec3<f32>,
    _pad:       f32,
    light_col:  vec3<f32>,
    ambient:    f32,
    diffuse_col: vec3<f32>,
    specular:   f32,
    shininess:  f32,
    width:      u32,
    height:     u32,
    _pad2:      u32,
}

@group(0) @binding(0) var<storage, read>       vertices : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       normals  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       indices  : array<u32>;
@group(0) @binding(3) var<storage, read_write> color_buf: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> depth_buf: array<f32>;
@group(0) @binding(5) var<uniform>             uniforms : Uniforms;

@compute @workgroup_size(64, 1, 1)
fn vertex_phong(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri_idx = gid.x;
    let n_tris  = arrayLength(&indices) / 3u;
    if tri_idx >= n_tris { return; }

    let i0 = indices[tri_idx * 3u];
    let i1 = indices[tri_idx * 3u + 1u];
    let i2 = indices[tri_idx * 3u + 2u];

    let p0 = uniforms.mvp * vertices[i0];
    let p1 = uniforms.mvp * vertices[i1];
    let p2 = uniforms.mvp * vertices[i2];

    // Flat shading: face normal from cross-product in view space
    let e0 = (uniforms.mv * vertices[i1] - uniforms.mv * vertices[i0]).xyz;
    let e1 = (uniforms.mv * vertices[i2] - uniforms.mv * vertices[i0]).xyz;
    let n  = normalize(cross(e0, e1));

    let diff  = max(dot(n, -normalize(uniforms.light_dir)), 0.0);
    let color = vec4<f32>(
        uniforms.diffuse_col * (uniforms.ambient + diff * uniforms.light_col),
        1.0
    );

    // Rasterise triangle into color/depth buffer (simple bounding-box fill)
    let w = f32(uniforms.width);
    let h = f32(uniforms.height);

    // NDC → pixel (perspective divide)
    let px0 = vec2<f32>((p0.x/p0.w * 0.5 + 0.5) * w, (1.0 - p0.y/p0.w * 0.5 - 0.5) * h);
    let px1 = vec2<f32>((p1.x/p1.w * 0.5 + 0.5) * w, (1.0 - p1.y/p1.w * 0.5 - 0.5) * h);
    let px2 = vec2<f32>((p2.x/p2.w * 0.5 + 0.5) * w, (1.0 - p2.y/p2.w * 0.5 - 0.5) * h);

    let min_x = max(i32(min(min(px0.x, px1.x), px2.x)), 0);
    let max_x = min(i32(max(max(px0.x, px1.x), px2.x)), i32(uniforms.width) - 1);
    let min_y = max(i32(min(min(px0.y, px1.y), px2.y)), 0);
    let max_y = min(i32(max(max(px0.y, px1.y), px2.y)), i32(uniforms.height) - 1);

    // Barycentric rasterisation
    for (var py = min_y; py <= max_y; py++) {
        for (var px = min_x; px <= max_x; px++) {
            let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
            // Edge functions
            let w0 = (px1.y-px2.y)*(p.x-px2.x) + (px2.x-px1.x)*(p.y-px2.y);
            let w1 = (px2.y-px0.y)*(p.x-px0.x) + (px0.x-px2.x)*(p.y-px0.y);
            let w2 = (px0.y-px1.y)*(p.x-px1.x) + (px1.x-px0.x)*(p.y-px1.y);
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let area_inv = 1.0 / (w0 + w1 + w2);
                let bc0 = w0 * area_inv;
                let bc1 = w1 * area_inv;
                let bc2 = w2 * area_inv;
                let z = bc0 * p0.z/p0.w + bc1 * p1.z/p1.w + bc2 * p2.z/p2.w;
                let pixel_idx = u32(py) * uniforms.width + u32(px);
                if z < depth_buf[pixel_idx] {
                    depth_buf[pixel_idx]  = z;
                    color_buf[pixel_idx]  = color;
                }
            }
        }
    }
}
"#;

/// WGSL source: particle splatting (screen-space circles).
pub const WGSL_PARTICLE_SPLAT: &str = r#"
struct ParticleUniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
    radius_px: f32,
    width: u32,
    height: u32,
    _pad: u32,
}
@group(0) @binding(0) var<storage, read>       positions  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> color_buf  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> depth_buf  : array<f32>;
@group(0) @binding(3) var<uniform>             uniforms   : ParticleUniforms;

@compute @workgroup_size(64, 1, 1)
fn particle_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pi = gid.x;
    if pi >= arrayLength(&positions) { return; }
    let clip = uniforms.mvp * positions[pi];
    if clip.w <= 0.0 { return; }
    let ndcx = clip.x / clip.w;
    let ndcy = clip.y / clip.w;
    let cx = (ndcx * 0.5 + 0.5) * f32(uniforms.width);
    let cy = (1.0 - ndcy * 0.5 - 0.5) * f32(uniforms.height);
    let z  = clip.z / clip.w;
    let r  = uniforms.radius_px;
    let x0 = max(i32(cx - r), 0);
    let x1 = min(i32(cx + r), i32(uniforms.width) - 1);
    let y0 = max(i32(cy - r), 0);
    let y1 = min(i32(cy + r), i32(uniforms.height) - 1);
    for (var sy = y0; sy <= y1; sy++) {
        for (var sx = x0; sx <= x1; sx++) {
            let dx = f32(sx) + 0.5 - cx;
            let dy = f32(sy) + 0.5 - cy;
            if dx*dx + dy*dy <= r*r {
                let idx = u32(sy) * uniforms.width + u32(sx);
                if z < depth_buf[idx] {
                    depth_buf[idx] = z;
                    color_buf[idx] = uniforms.color;
                }
            }
        }
    }
}
"#;

/// WGSL source: fullscreen blit with ACES tone-map and gamma correction.
pub const WGSL_FULLSCREEN_BLIT: &str = r#"
@group(0) @binding(0) var<storage, read>       hdr_in : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> ldr_out: array<vec4<f32>>;

fn aces(x: f32) -> f32 {
    return clamp((x*(2.51*x+0.03)) / (x*(2.43*x+0.59)+0.14), 0.0, 1.0);
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { return v * 12.92; }
    return 1.055 * pow(v, 1.0/2.4) - 0.055;
}

@compute @workgroup_size(256, 1, 1)
fn fullscreen_blit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&hdr_in) { return; }
    let h = hdr_in[idx];
    ldr_out[idx] = vec4<f32>(
        linear_to_srgb(aces(h.x)),
        linear_to_srgb(aces(h.y)),
        linear_to_srgb(aces(h.z)),
        h.w
    );
}
"#;

// ── GpuRendererConfig ─────────────────────────────────────────────────────────

/// Configuration for the GPU renderer.
#[derive(Debug, Clone)]
pub struct GpuRendererConfig {
    /// Framebuffer width in pixels.
    ///
    /// On a live renderer a new value takes effect at the next
    /// [`GpuRenderer::begin_frame`]; [`GpuRenderer::resize`] applies it at once.
    pub width: u32,
    /// Framebuffer height in pixels.
    ///
    /// On a live renderer a new value takes effect at the next
    /// [`GpuRenderer::begin_frame`]; [`GpuRenderer::resize`] applies it at once.
    pub height: u32,
    /// Field-of-view in degrees (vertical).
    pub fov_deg: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
    /// Whether to apply ACES tone-mapping in the blit pass.
    pub tone_map: bool,
    /// HDR exposure multiplier.
    pub exposure: f32,
    /// Ambient light intensity [0, 1].
    pub ambient: f32,
}

impl Default for GpuRendererConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            fov_deg: 60.0,
            near: 0.1,
            far: 1000.0,
            tone_map: true,
            exposure: 1.0,
            ambient: 0.1,
        }
    }
}

// ── GpuLight ─────────────────────────────────────────────────────────────────

/// A directional light for Phong shading.
#[derive(Debug, Clone, Copy)]
pub struct GpuLight {
    /// Direction *from* the light source (will be normalised).
    pub direction: [f32; 3],
    /// Light colour (linear RGB, each component [0, ∞]).
    pub colour: [f32; 3],
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            direction: [-0.5, -1.0, -0.5],
            colour: [1.0, 1.0, 1.0],
        }
    }
}

// ── GpuMesh ───────────────────────────────────────────────────────────────────

/// Uploaded GPU mesh (opaque handle + metadata).
#[derive(Debug, Clone, Copy)]
pub struct GpuMeshHandle {
    /// Index into the internal mesh registry.
    pub id: usize,
    /// Number of triangles.
    pub n_triangles: usize,
}

/// Internal mesh entry — GPU path stores buffer handles; CPU path stores raw data.
enum MeshEntry {
    /// Mesh has been uploaded to the GPU backend.
    Gpu {
        /// Vertex buffer handle (`vec4<f32>` positions, xyzw).
        vertex_buf: WgpuBufferHandle,
        /// Normal buffer handle (`vec4<f32>`, xyz normal + w unused).
        normal_buf: WgpuBufferHandle,
        /// Index buffer handle (u32 triangle indices).
        index_buf: WgpuBufferHandle,
        n_triangles: usize,
    },
    /// No GPU is available; mesh data is kept in host memory for software rasterisation.
    Cpu {
        /// Vertex positions as `[x, y, z, w]`.
        positions: Vec<[f32; 4]>,
        /// Triangle indices (triples).
        indices: Vec<u32>,
        /// Per-vertex normals `[nx, ny, nz, 0]`.
        normals: Vec<[f32; 4]>,
    },
}

// ── GpuRenderTarget ───────────────────────────────────────────────────────────

/// GPU-side render target (colour + depth buffers).
struct GpuRenderTarget {
    color_buf: WgpuBufferHandle,
    depth_buf: WgpuBufferHandle,
}

// ── GpuRenderer ───────────────────────────────────────────────────────────────

/// GPU-accelerated renderer using [`WgpuBackend`].
///
/// When no GPU is available the renderer falls back to a software path using
/// the internal CPU colour buffer.
///
/// See [module-level documentation](self) for pipeline details and examples.
///
/// # Framebuffer size
///
/// The colour and depth buffers are allocated for `config.width` x
/// `config.height` pixels.  `config` is public, but a new size written to it
/// only takes effect at the next [`begin_frame`](Self::begin_frame),
/// [`clear`](Self::clear) or [`resize`](Self::resize) (which reallocates the
/// buffers); drawing and read-back always use the size the buffers were
/// allocated for, so they can never index outside them.
pub struct GpuRenderer {
    /// Configuration.
    pub config: GpuRendererConfig,
    /// Active directional light.
    pub light: GpuLight,
    /// Camera view-projection matrix (column-major, 4×4).
    pub view_proj: [f32; 16],
    /// Camera view matrix (for shading).
    pub view: [f32; 16],

    /// GPU backend (None → CPU fallback).
    backend: Option<WgpuBackend>,
    /// Render target (GPU-side colour + depth).
    target: Option<GpuRenderTarget>,
    /// Uploaded meshes.
    meshes: Vec<MeshEntry>,
    /// CPU-side fallback colour buffer (f32 RGBA, w×h×4).
    cpu_color: Vec<[f32; 4]>,
    /// CPU-side fallback depth buffer.
    cpu_depth: Vec<f32>,
    /// `(width, height)` in pixels that `cpu_color`, `cpu_depth` and the device
    /// target were allocated for.
    target_extent: (usize, usize),
    /// Total frames rendered.
    pub frame_count: u64,
}

/// Number of pixels of a `width` x `height` framebuffer.
fn pixel_count(width: usize, height: usize) -> usize {
    width.saturating_mul(height)
}

impl GpuRenderer {
    /// Create a new renderer.  Attempts GPU initialisation; falls back to CPU.
    pub fn new(config: GpuRendererConfig) -> Self {
        let target_extent = (config.width as usize, config.height as usize);
        let n = pixel_count(target_extent.0, target_extent.1);
        let (backend, target) = match WgpuBackend::try_new() {
            Ok(mut b) => {
                // Register shaders
                b.register_shader("vertex_phong", WGSL_VERTEX_PHONG);
                b.register_shader("particle_splat", WGSL_PARTICLE_SPLAT);
                b.register_shader("fullscreen_blit", WGSL_FULLSCREEN_BLIT);

                // Allocate render target buffers
                let color_buf = b.create_buffer(n * 4); // 4 f64 per pixel (RGBA cast to f64 slots)
                let depth_buf = b.create_buffer(n);
                let tgt = GpuRenderTarget {
                    color_buf,
                    depth_buf,
                };
                (Some(b), Some(tgt))
            }
            Err(_) => (None, None),
        };

        let view_proj = identity4();
        let view = identity4();

        Self {
            config: config.clone(),
            light: GpuLight::default(),
            view_proj,
            view,
            backend,
            target,
            meshes: Vec::new(),
            cpu_color: vec![[0.0; 4]; n],
            cpu_depth: vec![1.0; n],
            target_extent,
            frame_count: 0,
        }
    }

    /// True if a real GPU backend is active.
    pub fn has_gpu(&self) -> bool {
        self.backend.is_some()
    }

    // ── Render target ─────────────────────────────────────────────────────────

    /// Resize the framebuffer to `width` x `height` pixels.
    ///
    /// Updates `config`, reallocates the CPU colour and depth buffers (and the
    /// device buffers when a GPU backend is active) and clears them, so the
    /// previous contents are lost.  A zero width or height is allowed and
    /// renders nothing.
    ///
    /// The projection built by [`look_at`](Self::look_at) bakes in the aspect
    /// ratio of the size it was called with: call `look_at` again after a
    /// resize that changes the aspect ratio.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.sync_render_target();
    }

    /// Apply the size in `config` to the render target when it differs from
    /// the size the buffers were allocated for.
    fn sync_render_target(&mut self) {
        let wanted = (self.config.width as usize, self.config.height as usize);
        let n = pixel_count(wanted.0, wanted.1);
        if self.target_extent != wanted || self.cpu_color.len() != n || self.cpu_depth.len() != n {
            self.allocate_target(wanted.0, wanted.1);
        }
    }

    /// Replace the CPU buffers (and the device target, if any) by cleared ones
    /// for `width` x `height` pixels.
    fn allocate_target(&mut self, width: usize, height: usize) {
        let n = pixel_count(width, height);
        self.cpu_color = vec![[0.0; 4]; n];
        self.cpu_depth = vec![1.0; n];
        self.target_extent = (width, height);
        if let Some(b) = &mut self.backend {
            let color_buf = b.create_buffer(n * 4);
            let depth_buf = b.create_buffer(n);
            self.target = Some(GpuRenderTarget {
                color_buf,
                depth_buf,
            });
        }
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    /// Set the camera using a look-at transform.
    ///
    /// The projection uses the aspect ratio of `config.width` x
    /// `config.height` (1 when either is zero).
    pub fn look_at(&mut self, eye: [f32; 3], target: [f32; 3], up: [f32; 3]) {
        self.view = look_at_mat(eye, target, up);
        let aspect = if self.config.width > 0 && self.config.height > 0 {
            self.config.width as f32 / self.config.height as f32
        } else {
            1.0
        };
        let proj = perspective_mat(
            self.config.fov_deg.to_radians(),
            aspect,
            self.config.near,
            self.config.far,
        );
        self.view_proj = mat4_mul(proj, self.view);
    }

    // ── Mesh upload ───────────────────────────────────────────────────────────

    /// Upload a mesh to GPU memory.
    ///
    /// `vertices` — flat `[x, y, z, w=1]` positions (one vec4 per vertex).
    /// `indices`  — triangle index triples.
    ///
    /// Returns a [`GpuMeshHandle`] for subsequent draw calls.
    pub fn upload_mesh(&mut self, vertices: &[[f32; 4]], indices: &[u32]) -> GpuMeshHandle {
        let n_tri = indices.len() / 3;

        // Compute flat normals (one per vertex, averaged from incident triangles)
        let normals = compute_vertex_normals(vertices, indices);

        let id = self.meshes.len();

        if let Some(b) = &mut self.backend {
            let vert_f64: Vec<f64> = vertices
                .iter()
                .flat_map(|v| v.iter().map(|&x| x as f64))
                .collect();
            let norm_f64: Vec<f64> = normals
                .iter()
                .flat_map(|n| n.iter().map(|&x| x as f64))
                .collect();
            let idx_f64: Vec<f64> = indices.iter().map(|&i| i as f64).collect();

            let vb = b.create_buffer(vert_f64.len());
            let nb = b.create_buffer(norm_f64.len());
            let ib = b.create_buffer(idx_f64.len());
            b.write_buffer(vb, &vert_f64);
            b.write_buffer(nb, &norm_f64);
            b.write_buffer(ib, &idx_f64);

            self.meshes.push(MeshEntry::Gpu {
                vertex_buf: vb,
                normal_buf: nb,
                index_buf: ib,
                n_triangles: n_tri,
            });
        } else {
            // CPU fallback: keep mesh data in host memory for software rasterisation.
            self.meshes.push(MeshEntry::Cpu {
                positions: vertices.to_vec(),
                indices: indices.to_vec(),
                normals,
            });
        }

        GpuMeshHandle {
            id,
            n_triangles: n_tri,
        }
    }

    // ── Frame lifecycle ───────────────────────────────────────────────────────

    /// Begin a new frame: apply a changed `config` size to the render target
    /// (see [`resize`](Self::resize)) and clear the colour and depth buffers.
    pub fn begin_frame(&mut self) {
        self.sync_render_target();
        let n = pixel_count(self.target_extent.0, self.target_extent.1);
        self.cpu_color.fill([0.0; 4]);
        self.cpu_depth.fill(1.0_f32);

        if let (Some(b), Some(tgt)) = (&mut self.backend, &self.target) {
            let zeros = vec![0.0_f64; n * 4];
            let ones = vec![1.0_f64; n];
            b.write_buffer(tgt.color_buf, &zeros);
            b.write_buffer(tgt.depth_buf, &ones);
        }
    }

    /// Submit a draw call for mesh `handle` with model matrix `model` (column-major).
    pub fn draw_mesh(&mut self, handle: GpuMeshHandle, model: &[f32; 16]) {
        if handle.id >= self.meshes.len() {
            return;
        }

        let mvp = mat4_mul(self.view_proj, *model);
        let mv = mat4_mul(self.view, *model);
        // The size the buffers were allocated for, not the (mutable) config.
        let (w, h) = self.target_extent;

        match &self.meshes[handle.id] {
            MeshEntry::Gpu {
                vertex_buf,
                normal_buf,
                index_buf,
                n_triangles,
            } => {
                if let (Some(b), Some(tgt)) = (&mut self.backend, &self.target) {
                    let n_tri_wg = (*n_triangles as u32).div_ceil(64);

                    // Pack MVP + light into a small uniform buffer (f64 slots)
                    let mut uni = Vec::with_capacity(48);
                    for v in &mvp {
                        uni.push(*v as f64);
                    } // 16 f64 (MVP)
                    for v in &mv {
                        uni.push(*v as f64);
                    } // 16 f64 (MV)
                    let ld = normalize3f(self.light.direction);
                    uni.push(ld[0] as f64);
                    uni.push(ld[1] as f64);
                    uni.push(ld[2] as f64);
                    uni.push(0.0); // pad
                    uni.push(self.light.colour[0] as f64);
                    uni.push(self.light.colour[1] as f64);
                    uni.push(self.light.colour[2] as f64);
                    uni.push(self.config.ambient as f64);
                    uni.push(0.8); // diffuse_col R
                    uni.push(0.8); // diffuse_col G
                    uni.push(0.8); // diffuse_col B
                    uni.push(0.5); // specular
                    uni.push(32.0); // shininess
                    uni.push(w as f64);
                    uni.push(h as f64);
                    uni.push(0.0); // pad2

                    let ub = b.create_buffer(uni.len());
                    b.write_buffer(ub, &uni);

                    b.dispatch(
                        "vertex_phong",
                        &[
                            *vertex_buf,
                            *normal_buf,
                            *index_buf,
                            tgt.color_buf,
                            tgt.depth_buf,
                            ub,
                        ],
                        n_tri_wg,
                    );
                }
            }
            MeshEntry::Cpu {
                positions,
                indices,
                normals,
            } => {
                // Software rasterisation path — no GPU required.
                let positions = positions.clone();
                let indices = indices.clone();
                let normals = normals.clone();
                draw_mesh_cpu(
                    &mut self.cpu_color,
                    &mut self.cpu_depth,
                    w,
                    h,
                    &positions,
                    &indices,
                    &normals,
                    MeshRenderParams {
                        mvp: &mvp,
                        mv: &mv,
                        light: self.light,
                        ambient: self.config.ambient,
                    },
                );
            }
        }
        let _ = (mvp, mv, w, h);
    }

    /// Submit a particle draw call.
    ///
    /// `positions` — flat `[x, y, z, w=1]` particle positions.
    /// `radius_px` — screen-space radius in pixels.
    /// `colour`    — particle colour (linear RGBA).
    ///
    /// With a GPU backend the `particle_splat` kernel runs on the device;
    /// without one the same splatting is performed on the CPU colour and depth
    /// buffers (see [`WGSL_PARTICLE_SPLAT`]), depth-tested against the meshes
    /// drawn so far in the frame.
    pub fn draw_particles(&mut self, positions: &[[f32; 4]], radius_px: f32, colour: [f32; 4]) {
        if positions.is_empty() {
            return;
        }
        let n = positions.len();

        if let (Some(b), Some(tgt)) = (&mut self.backend, &self.target) {
            let pos_f64: Vec<f64> = positions
                .iter()
                .flat_map(|p| p.iter().map(|&x| x as f64))
                .collect();
            let pb = b.create_buffer(pos_f64.len());
            b.write_buffer(pb, &pos_f64);

            let mut uni: Vec<f64> = Vec::with_capacity(24);
            for v in &self.view_proj {
                uni.push(*v as f64);
            }
            uni.push(colour[0] as f64);
            uni.push(colour[1] as f64);
            uni.push(colour[2] as f64);
            uni.push(colour[3] as f64);
            uni.push(radius_px as f64);
            uni.push(self.target_extent.0 as f64);
            uni.push(self.target_extent.1 as f64);
            uni.push(0.0);

            let ub = b.create_buffer(uni.len());
            b.write_buffer(ub, &uni);

            let wg = (n as u32).div_ceil(64);
            b.dispatch(
                "particle_splat",
                &[pb, tgt.color_buf, tgt.depth_buf, ub],
                wg,
            );
        } else {
            // Software path — no GPU required.
            draw_particles_cpu(
                &mut self.cpu_color,
                &mut self.cpu_depth,
                self.target_extent.0,
                self.target_extent.1,
                &self.view_proj,
                positions,
                radius_px,
                colour,
            );
        }
    }

    /// Finalise the frame: apply tone-mapping blit and download result.
    pub fn end_frame(&mut self) {
        let n = pixel_count(self.target_extent.0, self.target_extent.1);

        if let (Some(b), Some(tgt)) = (&mut self.backend, &self.target) {
            if self.config.tone_map {
                let ldr_buf = b.create_buffer(n * 4);
                let wg = (n as u32).div_ceil(256);
                b.dispatch("fullscreen_blit", &[tgt.color_buf, ldr_buf], wg);

                // Download LDR result
                let raw = b.read_buffer(ldr_buf);
                for i in 0..n {
                    self.cpu_color[i] = [
                        raw[i * 4] as f32,
                        raw[i * 4 + 1] as f32,
                        raw[i * 4 + 2] as f32,
                        raw[i * 4 + 3] as f32,
                    ];
                }
            } else {
                let raw = b.read_buffer(tgt.color_buf);
                for i in 0..n {
                    self.cpu_color[i] = [
                        raw[i * 4] as f32,
                        raw[i * 4 + 1] as f32,
                        raw[i * 4 + 2] as f32,
                        raw[i * 4 + 3] as f32,
                    ];
                }
            }
        }

        self.frame_count += 1;
    }

    /// Convert the current frame to an [`HdrFramebuffer`].
    ///
    /// The framebuffer has the size the buffers were allocated for, which a
    /// pending change of `config.width` / `config.height` does not affect
    /// before the next [`begin_frame`](Self::begin_frame) or
    /// [`resize`](Self::resize).
    pub fn to_hdr_framebuffer(&self) -> HdrFramebuffer {
        let (w, h) = self.target_extent;
        let mut fb = HdrFramebuffer::new(w, h);
        for (dst, src) in fb.pixels.iter_mut().zip(&self.cpu_color) {
            *dst = *src;
        }
        fb
    }

    /// Convert the current frame directly to an 8-bit RGBA byte buffer.
    pub fn to_ldr_bytes(&self) -> Vec<u8> {
        let tone = ToneMapper {
            method: ToneMapMethod::Aces,
            exposure: self.config.exposure,
            gamma: 2.2,
        };
        let mut out = Vec::with_capacity(self.cpu_color.len() * 4);
        for &px in &self.cpu_color {
            let [r, g, b, a] = tone.map(px);
            out.push((r.clamp(0.0, 1.0) * 255.0) as u8);
            out.push((g.clamp(0.0, 1.0) * 255.0) as u8);
            out.push((b.clamp(0.0, 1.0) * 255.0) as u8);
            out.push((a.clamp(0.0, 1.0) * 255.0) as u8);
        }
        out
    }

    /// Clear the colour and depth buffers to a background colour.
    ///
    /// Like [`begin_frame`](Self::begin_frame) this first applies a changed
    /// `config` size to the render target.
    pub fn clear(&mut self, colour: [f32; 4]) {
        self.sync_render_target();
        self.cpu_color.fill(colour);
        self.cpu_depth.fill(1.0);
    }

    /// Number of uploaded meshes.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/// Column-major identity 4×4 matrix.
fn identity4() -> [f32; 16] {
    [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ]
}

/// Multiply two 4×4 column-major matrices: C = A × B.
fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut c = [0.0_f32; 16];
    for row in 0..4 {
        for col in 0..4 {
            for k in 0..4 {
                c[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
            }
        }
    }
    c
}

/// Build a look-at view matrix (right-handed, column-major).
///
/// The rows are the camera axes `s` (right), `u` (up) and `-f` (backwards), so
/// the translation column holds `-s . eye`, `-u . eye` and `-(-f) . eye`: the
/// eye maps to the view-space origin and the target to `(0, 0, -|target - eye|)`.
fn look_at_mat(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize3f([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
    let s = normalize3f(cross3f(f, up));
    let u = cross3f(s, f);
    [
        s[0],
        u[0],
        -f[0],
        0.,
        s[1],
        u[1],
        -f[1],
        0.,
        s[2],
        u[2],
        -f[2],
        0.,
        -dot3f(s, eye),
        -dot3f(u, eye),
        dot3f(f, eye),
        1.,
    ]
}

/// Perspective projection matrix (column-major, right-handed, −1..1 Z range).
fn perspective_mat(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        (far + near) * nf,
        -1.,
        0.,
        0.,
        2. * far * near * nf,
        0.,
    ]
}

fn normalize3f(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3f(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ── CPU software rasteriser ───────────────────────────────────────────────────

/// Multiply a 4×4 column-major matrix by a vec4.
fn mat4_vec4_mul(m: &[f32; 16], v: [f32; 4]) -> [f32; 4] {
    [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
        m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
    ]
}

/// Per-frame render parameters that do not belong to the geometry buffers.
struct MeshRenderParams<'a> {
    /// Combined model-view-projection matrix (column-major).
    mvp: &'a [f32; 16],
    /// Model-view matrix for lighting in view space (column-major).
    mv: &'a [f32; 16],
    /// Active directional light.
    light: GpuLight,
    /// Ambient intensity \[0, 1\].
    ambient: f32,
}

/// Software rasteriser: rasterise mesh triangles into `color_buf` / `depth_buf`
/// using Phong shading and a barycentric scan-line fill.
///
/// # Arguments
/// * `color_buf` – linear RGBA framebuffer `[f32; 4]` per pixel, width × height.
/// * `depth_buf` – depth buffer `f32` per pixel (1.0 = far, 0.0 = near).
/// * `w`, `h`    – framebuffer dimensions.
/// * `positions` – per-vertex `[x, y, z, w]`.
/// * `indices`   – triangle index triples.
/// * `normals`   – per-vertex `[nx, ny, nz, 0]` in model space.
/// * `rp`        – per-frame render parameters (matrices, light, ambient).
///
/// Nothing is drawn when `w * h` is zero or larger than either buffer, and
/// triangles with an index outside `positions` or `normals` are skipped, so
/// inconsistent arguments can never index out of bounds.
fn draw_mesh_cpu(
    color_buf: &mut [[f32; 4]],
    depth_buf: &mut [f32],
    w: usize,
    h: usize,
    positions: &[[f32; 4]],
    indices: &[u32],
    normals: &[[f32; 4]],
    rp: MeshRenderParams<'_>,
) {
    let pixels = w.checked_mul(h).unwrap_or(0);
    if pixels == 0 || color_buf.len() < pixels || depth_buf.len() < pixels {
        return;
    }
    let mvp = rp.mvp;
    let mv = rp.mv;
    let light = rp.light;
    let ambient = rp.ambient;
    let n_tri = indices.len() / 3;
    let vertex_count = positions.len().min(normals.len());
    let wf = w as f32;
    let hf = h as f32;
    let light_dir_view = {
        let ld = normalize3f(light.direction);
        // Transform light direction into view space (ignore translation: w=0).
        let lv = mat4_vec4_mul(mv, [ld[0], ld[1], ld[2], 0.0]);
        normalize3f([lv[0], lv[1], lv[2]])
    };

    for t in 0..n_tri {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
            continue;
        }

        // Project each vertex into clip space then NDC → screen.
        let clip = [
            mat4_vec4_mul(mvp, positions[i0]),
            mat4_vec4_mul(mvp, positions[i1]),
            mat4_vec4_mul(mvp, positions[i2]),
        ];
        // Perspective divide
        let ndc: [[f32; 3]; 3] = std::array::from_fn(|k| {
            let w_inv = if clip[k][3].abs() > 1e-8 {
                1.0 / clip[k][3]
            } else {
                0.0
            };
            [clip[k][0] * w_inv, clip[k][1] * w_inv, clip[k][2] * w_inv]
        });
        // Viewport transform: NDC [-1,1] → pixel coords [0, w/h)
        let screen: [[f32; 3]; 3] = std::array::from_fn(|k| {
            [
                (ndc[k][0] * 0.5 + 0.5) * wf,
                (1.0 - (ndc[k][1] * 0.5 + 0.5)) * hf,
                ndc[k][2],
            ]
        });

        // Flat normal in view space (average of vertex normals for this triangle)
        let flat_nv = {
            let sum_n: [f32; 3] = [
                normals[i0][0] + normals[i1][0] + normals[i2][0],
                normals[i0][1] + normals[i1][1] + normals[i2][1],
                normals[i0][2] + normals[i1][2] + normals[i2][2],
            ];
            let n_view = mat4_vec4_mul(mv, [sum_n[0], sum_n[1], sum_n[2], 0.0]);
            normalize3f([n_view[0], n_view[1], n_view[2]])
        };
        let diffuse = dot3f(flat_nv, light_dir_view).max(0.0);
        let shading = ambient + diffuse * (1.0 - ambient);
        let base_col = [
            (light.colour[0] * shading).clamp(0.0, 1.0),
            (light.colour[1] * shading).clamp(0.0, 1.0),
            (light.colour[2] * shading).clamp(0.0, 1.0),
            1.0_f32,
        ];

        // Bounding box of the projected triangle, clamped to viewport
        let min_x = screen
            .iter()
            .map(|s| s[0])
            .fold(f32::INFINITY, f32::min)
            .max(0.0) as usize;
        let max_x = (screen
            .iter()
            .map(|s| s[0])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil() as usize)
            .min(w.saturating_sub(1));
        let min_y = screen
            .iter()
            .map(|s| s[1])
            .fold(f32::INFINITY, f32::min)
            .max(0.0) as usize;
        let max_y = (screen
            .iter()
            .map(|s| s[1])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil() as usize)
            .min(h.saturating_sub(1));

        // Precompute edge vectors for barycentric test
        let [x0, y0, _] = screen[0];
        let [x1, y1, _] = screen[1];
        let [x2, y2, _] = screen[2];
        let denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if denom.abs() < 1e-8 {
            continue; // degenerate
        }
        let inv_denom = 1.0 / denom;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let fx = px as f32 + 0.5;
                let fy = py as f32 + 0.5;
                // Barycentric coordinates
                let lam0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * inv_denom;
                let lam1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * inv_denom;
                let lam2 = 1.0 - lam0 - lam1;
                if lam0 < 0.0 || lam1 < 0.0 || lam2 < 0.0 {
                    continue; // outside triangle
                }
                // Interpolated depth
                let depth = lam0 * screen[0][2] + lam1 * screen[1][2] + lam2 * screen[2][2];
                let pidx = py * w + px;
                if depth < depth_buf[pidx] {
                    depth_buf[pidx] = depth;
                    color_buf[pidx] = base_col;
                }
            }
        }
    }
}

/// Software port of [`WGSL_PARTICLE_SPLAT`]: splat every particle as a
/// screen-space disc into `color_buf` / `depth_buf`.
///
/// A particle is projected with `view_proj` (column-major) and skipped when it
/// lies behind the camera (`clip.w <= 0`) or projects to a non-finite position.
/// Otherwise every pixel whose centre lies within `radius_px` of the projected
/// position receives `colour` when the particle's raw NDC depth `clip.z /
/// clip.w` is smaller than the stored depth (the same depth convention as
/// [`draw_mesh_cpu`]; the buffer is cleared to 1.0).  A radius that is not a
/// positive finite number draws nothing.
///
/// # Arguments
/// * `color_buf` – linear RGBA framebuffer `[f32; 4]` per pixel, width × height.
/// * `depth_buf` – depth buffer `f32` per pixel.
/// * `w`, `h`    – framebuffer dimensions.
/// * `view_proj` – camera view-projection matrix.
/// * `positions` – particle positions `[x, y, z, w]`.
/// * `radius_px` – screen-space radius in pixels.
/// * `colour`    – particle colour (linear RGBA), written without blending.
fn draw_particles_cpu(
    color_buf: &mut [[f32; 4]],
    depth_buf: &mut [f32],
    w: usize,
    h: usize,
    view_proj: &[f32; 16],
    positions: &[[f32; 4]],
    radius_px: f32,
    colour: [f32; 4],
) {
    if w == 0 || h == 0 || !radius_px.is_finite() || radius_px <= 0.0 {
        return;
    }
    let wf = w as f32;
    let hf = h as f32;
    let r2 = radius_px * radius_px;
    let last_col = i64::try_from(w).unwrap_or(i64::MAX) - 1;
    let last_row = i64::try_from(h).unwrap_or(i64::MAX) - 1;

    for p in positions {
        let clip = mat4_vec4_mul(view_proj, *p);
        // Behind the camera (or no meaningful depth at all).
        if clip[3].is_nan() || clip[3] <= 0.0 {
            continue;
        }
        let ndc_x = clip[0] / clip[3];
        let ndc_y = clip[1] / clip[3];
        let z = clip[2] / clip[3];
        let cx = (ndc_x * 0.5 + 0.5) * wf;
        let cy = (0.5 - ndc_y * 0.5) * hf;
        if !(cx.is_finite() && cy.is_finite() && z.is_finite()) {
            continue;
        }

        // Pixel bounding box of the disc, clamped to the viewport.
        let x_lo = ((cx - radius_px).floor() as i64).max(0);
        let x_hi = ((cx + radius_px).floor() as i64).min(last_col);
        let y_lo = ((cy - radius_px).floor() as i64).max(0);
        let y_hi = ((cy + radius_px).floor() as i64).min(last_row);
        for py in y_lo..=y_hi {
            for px in x_lo..=x_hi {
                let dx = px as f32 + 0.5 - cx;
                let dy = py as f32 + 0.5 - cy;
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let idx = py as usize * w + px as usize;
                if let (Some(depth), Some(pixel)) = (depth_buf.get_mut(idx), color_buf.get_mut(idx))
                    && z < *depth
                {
                    *depth = z;
                    *pixel = colour;
                }
            }
        }
    }
}

/// Compute per-vertex normals by averaging face normals of incident triangles.
fn compute_vertex_normals(vertices: &[[f32; 4]], indices: &[u32]) -> Vec<[f32; 4]> {
    let n = vertices.len();
    let mut normals = vec![[0.0_f32; 4]; n];
    let triangles = indices.len() / 3;
    for t in 0..triangles {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];
        let e0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e1 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let n3 = cross3f(e0, e1);
        for &i in &[i0, i1, i2] {
            normals[i][0] += n3[0];
            normals[i][1] += n3[1];
            normals[i][2] += n3[2];
        }
    }
    for n3 in &mut normals {
        let nn = normalize3f([n3[0], n3[1], n3[2]]);
        n3[0] = nn[0];
        n3[1] = nn[1];
        n3[2] = nn[2];
        n3[3] = 0.0;
    }
    normals
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_triangle() -> (Vec<[f32; 4]>, Vec<u32>) {
        let verts = vec![
            [0.0_f32, 1.0, -2.0, 1.0],
            [-1.0, -1.0, -2.0, 1.0],
            [1.0, -1.0, -2.0, 1.0],
        ];
        let idx = vec![0u32, 1, 2];
        (verts, idx)
    }

    #[test]
    fn test_renderer_construction() {
        let r = GpuRenderer::new(GpuRendererConfig::default());
        assert_eq!(r.config.width, 800);
        assert_eq!(r.config.height, 600);
        assert_eq!(r.frame_count, 0);
    }

    #[test]
    fn test_upload_mesh_and_frame() {
        let mut r = GpuRenderer::new(GpuRendererConfig::default());
        let (v, i) = simple_triangle();
        let h = r.upload_mesh(&v, &i);
        assert_eq!(h.n_triangles, 1);
        assert_eq!(r.mesh_count(), 1);

        r.look_at([0., 0., 0.], [0., 0., -1.], [0., 1., 0.]);
        r.begin_frame();
        r.draw_mesh(h, &identity4());
        r.end_frame();
        assert_eq!(r.frame_count, 1);
    }

    #[test]
    fn test_to_ldr_bytes() {
        let mut r = GpuRenderer::new(GpuRendererConfig {
            width: 4,
            height: 4,
            ..Default::default()
        });
        r.begin_frame();
        r.end_frame();
        let bytes = r.to_ldr_bytes();
        assert_eq!(bytes.len(), 4 * 4 * 4); // w*h*4 channels
    }

    #[test]
    fn test_to_hdr_framebuffer() {
        let r = GpuRenderer::new(GpuRendererConfig {
            width: 8,
            height: 8,
            ..Default::default()
        });
        let fb = r.to_hdr_framebuffer();
        assert_eq!(fb.width, 8);
        assert_eq!(fb.height, 8);
    }

    #[test]
    fn test_mat4_mul_identity() {
        let id = identity4();
        let result = mat4_mul(id, id);
        for i in 0..16 {
            assert!(
                (result[i] - id[i]).abs() < 1e-6,
                "result[{}]={} id[{}]={}",
                i,
                result[i],
                i,
                id[i]
            );
        }
    }

    #[test]
    fn test_look_at_sets_view() {
        let mut r = GpuRenderer::new(GpuRendererConfig::default());
        r.look_at([0., 0., 5.], [0., 0., 0.], [0., 1., 0.]);
        // The camera looks down view-space -z, so the origin it looks at, 5
        // units away, is translated to z = -5.
        assert!(
            (r.view[14] + 5.0).abs() < 0.01,
            "look-at should translate the target to z = -5, got view[14]={}",
            r.view[14]
        );
    }

    #[test]
    fn test_vertex_normals_triangle() {
        let verts = vec![[0.0_f32, 0., 0., 1.], [1., 0., 0., 1.], [0., 1., 0., 1.]];
        let idx = vec![0u32, 1, 2];
        let n = compute_vertex_normals(&verts, &idx);
        assert_eq!(n.len(), 3);
        // All normals should point in +Z
        for ni in &n {
            assert!(ni[2] > 0.9, "expected +Z normal, got {:?}", ni);
        }
    }

    // --- F3: MeshEntry::Cpu variant ---

    #[test]
    fn test_upload_mesh_cpu_variant_stored() {
        // Force CPU path by checking a no-backend renderer
        // GpuRenderer::new falls back to CPU when WgpuBackend::try_new() fails.
        // In a test environment there is no GPU, so backend == None.
        let mut r = GpuRenderer::new(GpuRendererConfig {
            width: 8,
            height: 8,
            ..Default::default()
        });
        let (v, i) = simple_triangle();
        let h = r.upload_mesh(&v, &i);
        assert_eq!(h.n_triangles, 1);
        assert_eq!(r.mesh_count(), 1);
        // Verify the stored entry is the Cpu variant
        match &r.meshes[0] {
            MeshEntry::Cpu {
                positions,
                indices,
                normals,
            } => {
                assert_eq!(positions.len(), v.len());
                assert_eq!(indices.len(), i.len());
                assert_eq!(normals.len(), v.len());
            }
            MeshEntry::Gpu { .. } => {
                // In GPU-available environments this is fine — skip the Cpu check.
            }
        }
    }

    #[test]
    fn test_render_cpu_produces_pixels() {
        let mut r = GpuRenderer::new(GpuRendererConfig {
            width: 16,
            height: 16,
            ..Default::default()
        });
        // Only meaningful on CPU fallback
        if r.backend.is_some() {
            return;
        }
        let verts = vec![
            [-1.0_f32, -1.0, 0.5, 1.0],
            [1.0, -1.0, 0.5, 1.0],
            [0.0, 1.0, 0.5, 1.0],
        ];
        let idx = vec![0u32, 1, 2];
        let h = r.upload_mesh(&verts, &idx);

        // Set up an orthographic-ish view that keeps the triangle on screen
        let identity = identity4();
        r.view_proj = identity;
        r.view = identity;

        r.begin_frame();
        r.draw_mesh(h, &identity);
        r.end_frame();

        // At least some pixels should be non-black
        let bytes = r.to_ldr_bytes();
        let non_zero = bytes.iter().any(|&b| b > 0);
        assert!(
            non_zero,
            "CPU rasteriser should produce at least one non-black pixel"
        );
    }

    // --- draw_particles: CPU path ---

    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.25, 1.0, 1.0];
    const SPLAT: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

    /// A renderer forced onto the software path, whatever the build offers.
    fn cpu_renderer(width: u32, height: u32) -> GpuRenderer {
        let mut r = GpuRenderer::new(GpuRendererConfig {
            width,
            height,
            fov_deg: 90.0,
            ..Default::default()
        });
        r.backend = None;
        r.target = None;
        r
    }

    /// Pixels of the CPU colour buffer that differ from the cleared value.
    fn painted_pixels(r: &GpuRenderer) -> Vec<(usize, usize)> {
        let w = r.config.width as usize;
        r.cpu_color
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != [0.0; 4])
            .map(|(i, _)| (i % w, i / w))
            .collect()
    }

    #[test]
    fn test_draw_particles_cpu_paints_the_projected_disc() {
        // Identity view-projection: the origin projects to (7.5, 7.5), the
        // centre of pixel (7, 7) of a 15 x 15 framebuffer.
        let mut r = cpu_renderer(15, 15);
        r.begin_frame();
        r.draw_particles(&[[0.0, 0.0, 0.0, 1.0]], 3.0, SPLAT);
        r.end_frame();
        let fb = r.to_hdr_framebuffer();
        assert_eq!(*fb.pixel(7, 7), SPLAT);
        let mut painted = 0;
        for y in 0..15_i32 {
            for x in 0..15_i32 {
                let inside = (x - 7).pow(2) + (y - 7).pow(2) <= 9;
                let is_splat = *fb.pixel(x as usize, y as usize) == SPLAT;
                assert_eq!(is_splat, inside, "pixel ({x}, {y})");
                painted += i32::from(is_splat);
            }
        }
        // Lattice points of the radius-3 disc.
        assert_eq!(painted, 29);
        assert_eq!(r.frame_count, 1);
    }

    #[test]
    fn test_draw_particles_cpu_paints_every_particle_of_a_call() {
        let mut r = cpu_renderer(16, 16);
        r.begin_frame();
        r.draw_particles(
            &[
                [-0.5, 0.5, 0.0, 1.0],
                [0.5, 0.5, 0.0, 1.0],
                [0.0, -0.5, 0.0, 1.0],
            ],
            1.5,
            RED,
        );
        // NDC (-0.5, 0.5) -> pixel (4, 4); (0.5, 0.5) -> (12, 4); (0, -0.5) -> (8, 12).
        for (cx, cy) in [(4_usize, 4_usize), (12, 4), (8, 12)] {
            assert_eq!(
                r.cpu_color[(cy - 1) * 16 + cx - 1],
                RED,
                "disc at ({cx}, {cy})"
            );
            assert_eq!(r.cpu_color[cy * 16 + cx], RED, "disc at ({cx}, {cy})");
        }
        // Each radius-1.5 disc about a pixel corner covers the four pixels around it.
        assert_eq!(painted_pixels(&r).len(), 3 * 4);
    }

    #[test]
    fn test_draw_particles_cpu_nearer_particle_wins_in_either_order() {
        let far = ([0.0, 0.0, 0.6, 1.0], RED);
        let near = ([0.0, 0.0, -0.3, 1.0], BLUE);
        for order in [[far, near], [near, far]] {
            let mut r = cpu_renderer(15, 15);
            r.begin_frame();
            for (position, colour) in order {
                r.draw_particles(&[position], 3.0, colour);
            }
            r.end_frame();
            assert_eq!(*r.to_hdr_framebuffer().pixel(7, 7), BLUE);
            // The near particle's disc is the only one that shows.
            assert_eq!(painted_pixels(&r).len(), 29);
        }
    }

    #[test]
    fn test_draw_particles_cpu_is_depth_tested_against_meshes() {
        let mut r = cpu_renderer(16, 16);
        let verts = vec![
            [-1.0_f32, -1.0, 0.5, 1.0],
            [1.0, -1.0, 0.5, 1.0],
            [0.0, 1.0, 0.5, 1.0],
        ];
        let mesh = r.upload_mesh(&verts, &[0, 1, 2]);
        r.begin_frame();
        r.draw_mesh(mesh, &identity4());
        let centre = 8 * 16 + 8;
        let mesh_pixel = r.cpu_color[centre];
        assert_ne!(mesh_pixel, [0.0; 4], "the triangle covers the centre");
        // Behind the triangle (depth 0.8 > 0.5): hidden.
        r.draw_particles(&[[0.0, 0.0, 0.8, 1.0]], 2.0, BLUE);
        assert_eq!(r.cpu_color[centre], mesh_pixel);
        // In front of it (depth 0.2 < 0.5): visible.
        r.draw_particles(&[[0.0, 0.0, 0.2, 1.0]], 2.0, BLUE);
        assert_eq!(r.cpu_color[centre], BLUE);
    }

    #[test]
    fn test_draw_particles_cpu_projects_with_the_camera_and_flips_y() {
        let mut r = cpu_renderer(16, 16);
        r.look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        r.begin_frame();
        // 90 degree field of view: at z = -2 the half-extent is 2, so (1, 1)
        // is NDC (0.5, 0.5), i.e. pixel coordinates (12, 4) with +y up.
        r.draw_particles(&[[1.0, 1.0, -2.0, 1.0]], 1.0, RED);
        let mut painted = painted_pixels(&r);
        painted.sort_unstable();
        assert_eq!(painted, vec![(11, 3), (11, 4), (12, 3), (12, 4)]);
    }

    #[test]
    fn test_draw_particles_cpu_skips_particles_behind_the_camera() {
        let mut r = cpu_renderer(16, 16);
        r.look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        r.begin_frame();
        r.draw_particles(&[[0.0, 0.0, 5.0, 1.0]], 4.0, RED);
        assert!(
            painted_pixels(&r).is_empty(),
            "a particle behind the camera"
        );
        r.draw_particles(&[[0.0, 0.0, -5.0, 1.0]], 4.0, RED);
        assert!(!painted_pixels(&r).is_empty(), "a particle in front");
    }

    #[test]
    fn test_draw_particles_cpu_clips_at_the_viewport_edges() {
        // Centred on the top-left corner of a 10 x 10 viewport (NDC -1, +1).
        let mut r = cpu_renderer(10, 10);
        r.begin_frame();
        r.draw_particles(&[[-1.0, 1.0, 0.0, 1.0]], 2.0, RED);
        let mut painted = painted_pixels(&r);
        painted.sort_unstable();
        assert_eq!(painted, vec![(0, 0), (0, 1), (1, 0)]);

        // Centred just beyond the bottom-right corner: only the nearest pixel
        // centre is inside the disc.
        let mut r = cpu_renderer(10, 10);
        r.begin_frame();
        r.draw_particles(&[[1.0, -1.0, 0.0, 1.0]], 0.8, RED);
        assert_eq!(painted_pixels(&r), vec![(9, 9)]);
    }

    #[test]
    fn test_draw_particles_cpu_ignores_invalid_input() {
        let mut r = cpu_renderer(12, 12);
        r.begin_frame();
        let position = [[0.0, 0.0, 0.0, 1.0]];
        for radius in [0.0, -2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            r.draw_particles(&position, radius, RED);
        }
        // Particles at non-finite or enormous positions, and with w <= 0.
        r.draw_particles(
            &[
                [f32::NAN, 0.0, 0.0, 1.0],
                [0.0, f32::INFINITY, 0.0, 1.0],
                [0.0, 0.0, f32::NAN, 1.0],
                [3.0e38, 0.0, 0.0, 1.0],
                [1.0e30, -1.0e30, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, f32::NAN],
            ],
            3.0,
            RED,
        );
        assert!(painted_pixels(&r).is_empty());
        r.draw_particles(&[], 3.0, RED);
        assert!(painted_pixels(&r).is_empty());
    }

    #[test]
    fn test_draw_particles_cpu_survives_a_huge_radius_and_a_resized_config() {
        let mut r = cpu_renderer(6, 5);
        r.begin_frame();
        r.draw_particles(&[[0.0, 0.0, 0.0, 1.0]], 1.0e9, RED);
        assert_eq!(painted_pixels(&r).len(), 30, "the disc covers the viewport");
        // `config` is public: a size written to it takes effect at the next
        // `begin_frame`, which reallocates the buffers, so the splat must cover
        // the whole new viewport and must not index past the buffers.
        r.config.width = 40;
        r.config.height = 40;
        r.begin_frame();
        assert_eq!(r.cpu_color.len(), 40 * 40);
        r.draw_particles(&[[0.0, 0.0, 0.0, 1.0]], 100.0, RED);
        assert_eq!(r.cpu_color.iter().filter(|c| **c == RED).count(), 40 * 40);
    }

    // --- framebuffer resize ---

    /// A triangle covering half of the normalised device square.
    fn half_screen_triangle() -> (Vec<[f32; 4]>, Vec<u32>) {
        (
            vec![
                [-1.0_f32, -1.0, 0.5, 1.0],
                [1.0, -1.0, 0.5, 1.0],
                [0.0, 1.0, 0.5, 1.0],
            ],
            vec![0, 1, 2],
        )
    }

    /// Pixels of `fb` that were painted (the buffer is cleared to zero).
    fn painted_count(fb: &HdrFramebuffer) -> usize {
        fb.pixels
            .iter()
            .filter(|p| p[0] > 0.0 || p[1] > 0.0 || p[2] > 0.0)
            .count()
    }

    #[test]
    fn test_resize_reallocates_the_framebuffer() {
        let mut r = cpu_renderer(8, 8);
        r.resize(64, 32);
        assert_eq!((r.config.width, r.config.height), (64, 32));
        let fb = r.to_hdr_framebuffer();
        assert_eq!((fb.width, fb.height), (64, 32));
        assert_eq!(fb.pixels.len(), 64 * 32);
        assert_eq!(r.to_ldr_bytes().len(), 64 * 32 * 4);

        let (verts, idx) = half_screen_triangle();
        let mesh = r.upload_mesh(&verts, &idx);
        let identity = identity4();
        r.view_proj = identity;
        r.view = identity;
        r.begin_frame();
        r.draw_mesh(mesh, &identity);
        r.end_frame();
        let painted = painted_count(&r.to_hdr_framebuffer());
        // The triangle covers half of the viewport, whatever its size.
        assert!(
            painted > 64 * 32 * 2 / 5 && painted < 64 * 32 * 3 / 5,
            "{painted} of {} pixels painted",
            64 * 32
        );
    }

    #[test]
    fn test_draw_mesh_after_a_direct_config_resize_stays_in_bounds() {
        // Regression: `draw_mesh_cpu` read the framebuffer size from the public
        // `config` while the buffers kept their construction-time size, so a
        // larger `config` indexed past both of them.
        let mut r = cpu_renderer(8, 8);
        let (verts, idx) = half_screen_triangle();
        let mesh = r.upload_mesh(&verts, &idx);
        let identity = identity4();
        r.view_proj = identity;
        r.view = identity;
        r.begin_frame();
        r.draw_mesh(mesh, &identity);
        assert!(painted_count(&r.to_hdr_framebuffer()) > 0);

        r.config.width = 24;
        r.config.height = 16;
        // Until the next frame begins drawing keeps using the allocated 8 x 8
        // target, and reading it back reports that size.
        r.draw_mesh(mesh, &identity);
        r.draw_particles(&[[0.0, 0.0, 0.0, 1.0]], 2.0, RED);
        let fb = r.to_hdr_framebuffer();
        assert_eq!((fb.width, fb.height), (8, 8));
        assert_eq!(r.to_ldr_bytes().len(), 8 * 8 * 4);

        // The next frame applies the new size.
        r.begin_frame();
        assert_eq!(painted_count(&r.to_hdr_framebuffer()), 0, "cleared");
        r.draw_mesh(mesh, &identity);
        r.end_frame();
        let fb = r.to_hdr_framebuffer();
        assert_eq!((fb.width, fb.height), (24, 16));
        let painted = painted_count(&fb);
        assert!(
            painted > 24 * 16 * 2 / 5 && painted < 24 * 16 * 3 / 5,
            "{painted} of {} pixels painted",
            24 * 16
        );
    }

    #[test]
    fn test_shrinking_the_framebuffer_stays_in_bounds() {
        let mut r = cpu_renderer(16, 16);
        let (verts, idx) = half_screen_triangle();
        let mesh = r.upload_mesh(&verts, &idx);
        let identity = identity4();
        r.view_proj = identity;
        r.view = identity;
        r.begin_frame();
        r.draw_mesh(mesh, &identity);
        r.config.width = 4;
        r.config.height = 4;
        r.begin_frame();
        r.draw_mesh(mesh, &identity);
        r.draw_particles(&[[0.0, 0.0, 0.0, 1.0]], 1.5, RED);
        let fb = r.to_hdr_framebuffer();
        assert_eq!((fb.width, fb.height), (4, 4));
        assert!(painted_count(&fb) > 0);
    }

    #[test]
    fn test_clear_applies_a_pending_config_size() {
        let mut r = cpu_renderer(4, 4);
        r.config.width = 6;
        r.config.height = 3;
        r.clear([0.25, 0.5, 0.75, 1.0]);
        let fb = r.to_hdr_framebuffer();
        assert_eq!((fb.width, fb.height), (6, 3));
        assert!(fb.pixels.iter().all(|p| *p == [0.25, 0.5, 0.75, 1.0]));
    }

    #[test]
    fn test_zero_sized_framebuffer_renders_nothing() {
        let mut r = cpu_renderer(8, 8);
        let (verts, idx) = half_screen_triangle();
        let mesh = r.upload_mesh(&verts, &idx);
        let identity = identity4();
        r.view_proj = identity;
        r.view = identity;
        for (width, height) in [(0, 0), (0, 5), (5, 0)] {
            r.resize(width, height);
            r.begin_frame();
            r.draw_mesh(mesh, &identity);
            r.draw_particles(&[[0.0, 0.0, 0.0, 1.0]], 3.0, RED);
            r.end_frame();
            let fb = r.to_hdr_framebuffer();
            assert_eq!((fb.width, fb.height), (width as usize, height as usize));
            assert!(fb.pixels.is_empty());
            assert!(r.to_ldr_bytes().is_empty());
            // A zero extent must not poison the projection with an infinite
            // or NaN aspect ratio.
            r.look_at([0.0, 0.0, 3.0], [0.0; 3], [0.0, 1.0, 0.0]);
            assert!(
                r.view_proj.iter().all(|m| m.is_finite()),
                "{width}x{height}"
            );
        }
        // The renderer is usable again at a real size.
        r.view_proj = identity;
        r.view = identity;
        r.resize(8, 8);
        r.begin_frame();
        r.draw_mesh(mesh, &identity);
        assert!(painted_count(&r.to_hdr_framebuffer()) > 0);
    }

    #[test]
    fn test_look_at_maps_the_target_straight_ahead_of_an_offset_eye() {
        // Regression: the z translation of the view matrix was `-dot(f, eye)`
        // instead of `+dot(f, eye)`, so every eye away from the origin along the
        // viewing direction put the scene behind the camera.
        let mut r = cpu_renderer(16, 16);
        let eye = [1.0, 2.0, 5.0];
        let target = [1.0, 2.0, -1.0];
        r.look_at(eye, target, [0.0, 1.0, 0.0]);
        let at_eye = mat4_vec4_mul(&r.view, [eye[0], eye[1], eye[2], 1.0]);
        assert!(at_eye[..3].iter().all(|c| c.abs() < 1e-5), "{at_eye:?}");
        // The target is 6 units straight ahead, i.e. down view-space -z.
        let ahead = mat4_vec4_mul(&r.view, [target[0], target[1], target[2], 1.0]);
        assert!(ahead[0].abs() < 1e-5 && ahead[1].abs() < 1e-5, "{ahead:?}");
        assert!((ahead[2] + 6.0).abs() < 1e-5, "{ahead:?}");
        // After the projection it sits in front of the camera, at the screen centre.
        let clip = mat4_vec4_mul(&r.view_proj, [target[0], target[1], target[2], 1.0]);
        assert!(clip[3] > 0.0, "target behind the camera: {clip:?}");
        assert!((clip[0] / clip[3]).abs() < 1e-5 && (clip[1] / clip[3]).abs() < 1e-5);
        assert!((-1.0..1.0).contains(&(clip[2] / clip[3])), "{clip:?}");

        // An oblique viewing direction: the target is |target - eye| ahead.
        r.look_at([4.0, 3.0, 2.0], [0.0; 3], [0.0, 1.0, 0.0]);
        let origin = mat4_vec4_mul(&r.view, [0.0, 0.0, 0.0, 1.0]);
        let distance = (16.0_f32 + 9.0 + 4.0).sqrt();
        assert!(
            origin[0].abs() < 1e-4 && origin[1].abs() < 1e-4,
            "{origin:?}"
        );
        assert!((origin[2] + distance).abs() < 1e-4, "{origin:?}");
    }

    #[test]
    fn test_draw_mesh_with_an_offset_camera_paints_the_visible_triangle() {
        // End to end: a camera 3 units in front of the triangle sees it.
        let mut r = cpu_renderer(16, 16);
        let (verts, idx) = half_screen_triangle();
        let mesh = r.upload_mesh(&verts, &idx);
        r.look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        r.begin_frame();
        r.draw_mesh(mesh, &identity4());
        let fb = r.to_hdr_framebuffer();
        assert!(painted_count(&fb) > 0, "the triangle must be visible");
        // The triangle is 2.5 units from the eye and the field of view is 90
        // degrees, so it spans NDC (+-0.4, +-0.4): rows 4.8 to 11.2 of the 16
        // pixel rows, with the apex (y = +1) on top and the base (y = -1) below.
        let row_count = |y: usize| (0..16).filter(|&x| fb.pixels[y * 16 + x][0] > 0.0).count();
        for y in (0..=4).chain(11..16) {
            assert_eq!(row_count(y), 0, "row {y} lies outside the triangle");
        }
        assert!(
            row_count(7) > 0 && row_count(7) < row_count(10),
            "the triangle widens towards its base: {} then {}",
            row_count(7),
            row_count(10)
        );
    }

    #[test]
    fn test_look_at_aspect_follows_the_resized_config() {
        let eye = [0.0, 1.0, 4.0];
        let mut wide = cpu_renderer(64, 32);
        wide.look_at(eye, [0.0; 3], [0.0, 1.0, 0.0]);
        let mut resized = cpu_renderer(8, 8);
        resized.resize(64, 32);
        resized.look_at(eye, [0.0; 3], [0.0, 1.0, 0.0]);
        assert_eq!(wide.view_proj, resized.view_proj);
        let mut square = cpu_renderer(8, 8);
        square.look_at(eye, [0.0; 3], [0.0, 1.0, 0.0]);
        assert_ne!(square.view_proj, resized.view_proj);
    }

    #[test]
    fn test_resize_reallocates_the_device_target() {
        // A stub backend stands in for a device: its buffers must follow the size.
        let mut r = cpu_renderer(4, 4);
        r.backend = Some(WgpuBackend::new_stub());
        r.resize(8, 6);
        let n = 8 * 6;
        let (color_buf, depth_buf) = match &r.target {
            Some(tgt) => (tgt.color_buf, tgt.depth_buf),
            None => panic!("resize must allocate the device target"),
        };
        if let Some(b) = &r.backend {
            assert_eq!(b.read_buffer(color_buf).len(), n * 4);
            assert_eq!(b.read_buffer(depth_buf).len(), n);
        }
        r.begin_frame();
        if let Some(b) = &r.backend {
            assert!(b.read_buffer(depth_buf).iter().all(|d| *d == 1.0));
            assert!(b.read_buffer(color_buf).iter().all(|c| *c == 0.0));
        }
        assert_eq!(r.cpu_color.len(), n);
        assert_eq!(r.cpu_depth.len(), n);
    }

    #[test]
    fn test_draw_mesh_cpu_never_indexes_outside_its_buffers() {
        let (verts, idx) = half_screen_triangle();
        let normals = compute_vertex_normals(&verts, &idx);
        let identity = identity4();
        let params = || MeshRenderParams {
            mvp: &identity,
            mv: &identity,
            light: GpuLight::default(),
            ambient: 0.1,
        };
        let painted = |color: &[[f32; 4]]| color.iter().filter(|c| **c != [0.0; 4]).count();

        // Exact-size buffers draw.
        let mut color = vec![[0.0_f32; 4]; 64];
        let mut depth = vec![1.0_f32; 64];
        draw_mesh_cpu(
            &mut color,
            &mut depth,
            8,
            8,
            &verts,
            &idx,
            &normals,
            params(),
        );
        assert!(painted(&color) > 8, "{} pixels", painted(&color));

        // Buffers smaller than `w * h` (the stale-buffer case) draw nothing.
        let mut color = vec![[0.0_f32; 4]; 16];
        let mut depth = vec![1.0_f32; 16];
        draw_mesh_cpu(
            &mut color,
            &mut depth,
            8,
            8,
            &verts,
            &idx,
            &normals,
            params(),
        );
        assert_eq!(painted(&color), 0);
        assert!(depth.iter().all(|d| *d == 1.0));
        // ... also when only the depth buffer is short.
        let mut color = vec![[0.0_f32; 4]; 64];
        let mut depth = vec![1.0_f32; 63];
        draw_mesh_cpu(
            &mut color,
            &mut depth,
            8,
            8,
            &verts,
            &idx,
            &normals,
            params(),
        );
        assert_eq!(painted(&color), 0);

        // Empty or overflowing extents draw nothing.
        for (w, h) in [(0, 8), (8, 0), (usize::MAX, 2), (2, usize::MAX)] {
            let mut color = vec![[0.0_f32; 4]; 64];
            let mut depth = vec![1.0_f32; 64];
            draw_mesh_cpu(
                &mut color,
                &mut depth,
                w,
                h,
                &verts,
                &idx,
                &normals,
                params(),
            );
            assert_eq!(painted(&color), 0, "{w}x{h}");
        }

        // Fewer normals than positions: the triangles that need the missing
        // normals are skipped instead of indexing past `normals`.
        let mut color = vec![[0.0_f32; 4]; 64];
        let mut depth = vec![1.0_f32; 64];
        draw_mesh_cpu(
            &mut color,
            &mut depth,
            8,
            8,
            &verts,
            &idx,
            &normals[..2],
            params(),
        );
        assert_eq!(painted(&color), 0);
    }
}
