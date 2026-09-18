//! 3D wireframe preview of stage layout for virtual production.
//!
//! Provides a lightweight pure-Rust renderer that projects 3D stage geometry
//! (LED walls, cameras, lights, talent marks) into a 2D orthographic or
//! perspective view and rasterises the wireframe into an RGB image buffer.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// A vertex in 3D world space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vertex3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vertex3 {
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Translate by offset.
    #[must_use]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }

    /// Scale uniformly from origin.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

/// An edge connecting two vertex indices in a mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
}

impl Edge {
    #[must_use]
    pub const fn new(a: usize, b: usize) -> Self {
        Self { a, b }
    }
}

/// A wireframe mesh: vertices + edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMesh {
    pub vertices: Vec<Vertex3>,
    pub edges: Vec<Edge>,
    /// RGBA colour for this mesh.
    pub color: [u8; 4],
    /// Display label.
    pub label: String,
}

impl WireMesh {
    /// Create an empty mesh.
    #[must_use]
    pub fn new(label: &str, color: [u8; 4]) -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            color,
            label: label.to_string(),
        }
    }

    /// Build a rectangular box wireframe (8 vertices, 12 edges).
    #[must_use]
    pub fn box_wireframe(
        label: &str,
        color: [u8; 4],
        center: Vertex3,
        width: f64,
        height: f64,
        depth: f64,
    ) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;
        let hd = depth / 2.0;
        let cx = center.x;
        let cy = center.y;
        let cz = center.z;

        let vertices = vec![
            Vertex3::new(cx - hw, cy - hh, cz - hd), // 0 front-bottom-left
            Vertex3::new(cx + hw, cy - hh, cz - hd), // 1 front-bottom-right
            Vertex3::new(cx + hw, cy + hh, cz - hd), // 2 front-top-right
            Vertex3::new(cx - hw, cy + hh, cz - hd), // 3 front-top-left
            Vertex3::new(cx - hw, cy - hh, cz + hd), // 4 back-bottom-left
            Vertex3::new(cx + hw, cy - hh, cz + hd), // 5 back-bottom-right
            Vertex3::new(cx + hw, cy + hh, cz + hd), // 6 back-top-right
            Vertex3::new(cx - hw, cy + hh, cz + hd), // 7 back-top-left
        ];

        let edges = vec![
            // Front face
            Edge::new(0, 1),
            Edge::new(1, 2),
            Edge::new(2, 3),
            Edge::new(3, 0),
            // Back face
            Edge::new(4, 5),
            Edge::new(5, 6),
            Edge::new(6, 7),
            Edge::new(7, 4),
            // Connections
            Edge::new(0, 4),
            Edge::new(1, 5),
            Edge::new(2, 6),
            Edge::new(3, 7),
        ];

        Self {
            vertices,
            edges,
            color,
            label: label.to_string(),
        }
    }

    /// Build a frustum wireframe for a camera.
    #[must_use]
    pub fn camera_frustum(
        label: &str,
        color: [u8; 4],
        position: Vertex3,
        look_dist: f64,
        hfov_deg: f64,
        vfov_deg: f64,
    ) -> Self {
        let hh = (hfov_deg.to_radians() / 2.0).tan() * look_dist;
        let vh = (vfov_deg.to_radians() / 2.0).tan() * look_dist;
        let far_z = position.z - look_dist;

        let vertices = vec![
            position,                                              // 0: apex
            Vertex3::new(position.x - hh, position.y + vh, far_z), // 1: tl
            Vertex3::new(position.x + hh, position.y + vh, far_z), // 2: tr
            Vertex3::new(position.x + hh, position.y - vh, far_z), // 3: br
            Vertex3::new(position.x - hh, position.y - vh, far_z), // 4: bl
        ];

        let edges = vec![
            Edge::new(0, 1),
            Edge::new(0, 2),
            Edge::new(0, 3),
            Edge::new(0, 4),
            Edge::new(1, 2),
            Edge::new(2, 3),
            Edge::new(3, 4),
            Edge::new(4, 1),
        ];

        Self {
            vertices,
            edges,
            color,
            label: label.to_string(),
        }
    }

    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

/// Projection mode for the stage view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMode {
    /// Top-down orthographic (looking from +Y toward -Y).
    TopDown,
    /// Front orthographic (looking from +Z toward -Z).
    FrontView,
    /// Side orthographic (looking from +X toward -X).
    SideView,
    /// Simple perspective projection.
    Perspective,
}

/// Stage visualization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageVisualizationConfig {
    /// Output image width.
    pub width: usize,
    /// Output image height.
    pub height: usize,
    /// Projection mode.
    pub projection: ProjectionMode,
    /// World units per pixel (for orthographic modes).
    pub units_per_pixel: f64,
    /// Camera position for perspective mode.
    pub eye: [f64; 3],
    /// Perspective field of view in degrees.
    pub fov_deg: f64,
    /// Background colour (RGB).
    pub background: [u8; 3],
    /// Whether to draw grid lines.
    pub draw_grid: bool,
    /// Grid spacing in world units.
    pub grid_spacing: f64,
}

impl Default for StageVisualizationConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            projection: ProjectionMode::TopDown,
            units_per_pixel: 0.02, // 2 cm per pixel → 10m total view
            eye: [0.0, 10.0, 0.0],
            fov_deg: 60.0,
            background: [20, 20, 20],
            draw_grid: true,
            grid_spacing: 1.0,
        }
    }
}

/// Rendered frame from the stage visualizer.
#[derive(Debug, Clone)]
pub struct StageFrame {
    /// RGB pixel data (row-major).
    pub pixels: Vec<u8>,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
}

impl StageFrame {
    fn new(width: usize, height: usize, background: [u8; 3]) -> Self {
        let mut pixels = Vec::with_capacity(width * height * 3);
        for _ in 0..(width * height) {
            pixels.push(background[0]);
            pixels.push(background[1]);
            pixels.push(background[2]);
        }
        Self {
            pixels,
            width,
            height,
        }
    }

    /// Set pixel, ignoring out-of-bounds.
    fn set_pixel(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let idx = (y as usize * self.width + x as usize) * 3;
        self.pixels[idx] = color[0];
        self.pixels[idx + 1] = color[1];
        self.pixels[idx + 2] = color[2];
    }

    /// Get pixel value.
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 3;
        Some([self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]])
    }

    /// Draw a line using Bresenham's algorithm.
    fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
        let mut cx = x0;
        let mut cy = y0;
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: i32 = if x0 < x1 { 1 } else { -1 };
        let sy: i32 = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            self.set_pixel(cx, cy, color);
            if cx == x1 && cy == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                cx += sx;
            }
            if e2 <= dx {
                err += dx;
                cy += sy;
            }
        }
    }
}

/// Main stage visualizer.
pub struct StageVisualization {
    config: StageVisualizationConfig,
    meshes: Vec<WireMesh>,
}

impl StageVisualization {
    /// Create a new stage visualizer.
    pub fn new(config: StageVisualizationConfig) -> Result<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(VirtualProductionError::InvalidConfig(
                "Stage visualization resolution must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            config,
            meshes: Vec::new(),
        })
    }

    /// Add a wire mesh to the scene.
    pub fn add_mesh(&mut self, mesh: WireMesh) {
        self.meshes.push(mesh);
    }

    /// Clear all meshes.
    pub fn clear(&mut self) {
        self.meshes.clear();
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &StageVisualizationConfig {
        &self.config
    }

    /// Number of meshes.
    #[must_use]
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Render the current scene to an RGB image.
    pub fn render(&self) -> Result<StageFrame> {
        let w = self.config.width;
        let h = self.config.height;
        let mut frame = StageFrame::new(w, h, self.config.background);

        // Draw grid
        if self.config.draw_grid {
            self.draw_grid(&mut frame);
        }

        // Project and draw each mesh
        for mesh in &self.meshes {
            self.draw_mesh(&mut frame, mesh);
        }

        Ok(frame)
    }

    /// Project a 3D point to 2D pixel coordinates.
    fn project(&self, v: &Vertex3) -> (f64, f64) {
        let w = self.config.width as f64;
        let h = self.config.height as f64;
        let upp = self.config.units_per_pixel;

        match self.config.projection {
            ProjectionMode::TopDown => {
                // XZ plane, origin at image center
                let px = v.x / upp + w / 2.0;
                let py = -v.z / upp + h / 2.0;
                (px, py)
            }
            ProjectionMode::FrontView => {
                let px = v.x / upp + w / 2.0;
                let py = -v.y / upp + h / 2.0;
                (px, py)
            }
            ProjectionMode::SideView => {
                let px = v.z / upp + w / 2.0;
                let py = -v.y / upp + h / 2.0;
                (px, py)
            }
            ProjectionMode::Perspective => {
                let eye = &self.config.eye;
                let dx = v.x - eye[0];
                let dy = v.y - eye[1];
                let dz = v.z - eye[2];
                let depth = -dz;
                if depth <= 0.001 {
                    return (-1.0, -1.0); // behind camera
                }
                let fov_rad = self.config.fov_deg.to_radians();
                let f = 1.0 / (fov_rad / 2.0).tan();
                let px = dx / depth * f * w / 2.0 + w / 2.0;
                let py = -dy / depth * f * h / 2.0 + h / 2.0;
                (px, py)
            }
        }
    }

    /// Draw a wire mesh into a frame.
    fn draw_mesh(&self, frame: &mut StageFrame, mesh: &WireMesh) {
        let color = [mesh.color[0], mesh.color[1], mesh.color[2]];

        for edge in &mesh.edges {
            if edge.a >= mesh.vertices.len() || edge.b >= mesh.vertices.len() {
                continue;
            }
            let va = &mesh.vertices[edge.a];
            let vb = &mesh.vertices[edge.b];
            let (x0, y0) = self.project(va);
            let (x1, y1) = self.project(vb);
            frame.draw_line(x0 as i32, y0 as i32, x1 as i32, y1 as i32, color);
        }
    }

    /// Draw a grid floor in the current projection.
    fn draw_grid(&self, frame: &mut StageFrame) {
        let color = [50u8, 50, 50];
        let step = self.config.grid_spacing;
        let half_range = 10.0 * step;

        let grid_range = 20;
        for i in -grid_range..=grid_range {
            let coord = i as f64 * step;

            // Line parallel to X axis (varying Z)
            let va = Vertex3::new(-half_range, 0.0, coord);
            let vb = Vertex3::new(half_range, 0.0, coord);
            let (x0, y0) = self.project(&va);
            let (x1, y1) = self.project(&vb);
            frame.draw_line(x0 as i32, y0 as i32, x1 as i32, y1 as i32, color);

            // Line parallel to Z axis (varying X)
            let vc = Vertex3::new(coord, 0.0, -half_range);
            let vd = Vertex3::new(coord, 0.0, half_range);
            let (x2, y2) = self.project(&vc);
            let (x3, y3) = self.project(&vd);
            frame.draw_line(x2 as i32, y2 as i32, x3 as i32, y3 as i32, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config_small() -> StageVisualizationConfig {
        StageVisualizationConfig {
            width: 128,
            height: 128,
            projection: ProjectionMode::TopDown,
            units_per_pixel: 0.1,
            eye: [0.0, 10.0, 0.0],
            fov_deg: 60.0,
            background: [0, 0, 0],
            draw_grid: false,
            grid_spacing: 1.0,
        }
    }

    #[test]
    fn test_stage_visualization_creation() {
        let config = default_config_small();
        let vis = StageVisualization::new(config);
        assert!(vis.is_ok());
    }

    #[test]
    fn test_stage_visualization_zero_resolution_fails() {
        let mut config = default_config_small();
        config.width = 0;
        let vis = StageVisualization::new(config);
        assert!(vis.is_err());
    }

    #[test]
    fn test_add_mesh() {
        let config = default_config_small();
        let mut vis = StageVisualization::new(config).expect("should create");
        let mesh = WireMesh::new("test", [255, 0, 0, 255]);
        vis.add_mesh(mesh);
        assert_eq!(vis.mesh_count(), 1);
    }

    #[test]
    fn test_clear_meshes() {
        let config = default_config_small();
        let mut vis = StageVisualization::new(config).expect("should create");
        vis.add_mesh(WireMesh::new("a", [255, 0, 0, 255]));
        vis.add_mesh(WireMesh::new("b", [0, 255, 0, 255]));
        assert_eq!(vis.mesh_count(), 2);
        vis.clear();
        assert_eq!(vis.mesh_count(), 0);
    }

    #[test]
    fn test_render_empty_scene() {
        let config = default_config_small();
        let vis = StageVisualization::new(config).expect("should create");
        let frame = vis.render();
        assert!(frame.is_ok());
        let f = frame.expect("ok");
        assert_eq!(f.pixels.len(), 128 * 128 * 3);
    }

    #[test]
    fn test_render_with_box_mesh() {
        let config = default_config_small();
        let mut vis = StageVisualization::new(config).expect("should create");

        let mesh = WireMesh::box_wireframe(
            "LED Wall",
            [0, 200, 255, 255],
            Vertex3::new(0.0, 0.0, 0.0),
            4.0,
            2.0,
            0.1,
        );
        vis.add_mesh(mesh);

        let frame = vis.render().expect("should render");
        assert_eq!(frame.width, 128);
        assert_eq!(frame.height, 128);

        // Some non-black pixels should exist from the wireframe
        let has_colored = frame
            .pixels
            .chunks_exact(3)
            .any(|c| c[0] > 0 || c[1] > 0 || c[2] > 0);
        assert!(has_colored, "wireframe should produce non-black pixels");
    }

    #[test]
    fn test_render_with_camera_frustum() {
        let config = default_config_small();
        let mut vis = StageVisualization::new(config).expect("should create");

        let mesh = WireMesh::camera_frustum(
            "Main Camera",
            [255, 200, 0, 255],
            Vertex3::new(0.0, 0.0, 3.0),
            3.0,
            70.0,
            42.0,
        );
        vis.add_mesh(mesh);

        let frame = vis.render().expect("should render");
        assert!(frame.pixels.len() > 0);
    }

    #[test]
    fn test_box_wireframe_vertex_count() {
        let mesh = WireMesh::box_wireframe(
            "box",
            [255, 255, 255, 255],
            Vertex3::new(0.0, 0.0, 0.0),
            1.0,
            1.0,
            1.0,
        );
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.edge_count(), 12);
    }

    #[test]
    fn test_camera_frustum_vertex_count() {
        let mesh = WireMesh::camera_frustum(
            "cam",
            [255, 255, 255, 255],
            Vertex3::new(0.0, 0.0, 0.0),
            2.0,
            60.0,
            40.0,
        );
        assert_eq!(mesh.vertex_count(), 5);
        assert_eq!(mesh.edge_count(), 8);
    }

    #[test]
    fn test_render_front_view() {
        let config = StageVisualizationConfig {
            width: 64,
            height: 64,
            projection: ProjectionMode::FrontView,
            ..default_config_small()
        };
        let mut vis = StageVisualization::new(config).expect("should create");
        vis.add_mesh(WireMesh::box_wireframe(
            "wall",
            [255, 0, 0, 255],
            Vertex3::new(0.0, 0.0, 0.0),
            2.0,
            2.0,
            0.1,
        ));
        let frame = vis.render().expect("ok");
        assert_eq!(frame.pixels.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_render_side_view() {
        let config = StageVisualizationConfig {
            width: 64,
            height: 64,
            projection: ProjectionMode::SideView,
            ..default_config_small()
        };
        let vis = StageVisualization::new(config).expect("should create");
        let frame = vis.render().expect("ok");
        assert_eq!(frame.pixels.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_render_perspective() {
        let config = StageVisualizationConfig {
            width: 64,
            height: 64,
            projection: ProjectionMode::Perspective,
            ..default_config_small()
        };
        let mut vis = StageVisualization::new(config).expect("should create");
        vis.add_mesh(WireMesh::box_wireframe(
            "box",
            [0, 255, 0, 255],
            Vertex3::new(0.0, 0.0, -3.0),
            1.0,
            1.0,
            1.0,
        ));
        let frame = vis.render().expect("ok");
        assert_eq!(frame.pixels.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_render_with_grid() {
        let config = StageVisualizationConfig {
            width: 64,
            height: 64,
            draw_grid: true,
            grid_spacing: 1.0,
            background: [0, 0, 0],
            ..default_config_small()
        };
        let vis = StageVisualization::new(config).expect("should create");
        let frame = vis.render().expect("ok");
        let has_grid = frame
            .pixels
            .chunks_exact(3)
            .any(|c| c[0] > 0 || c[1] > 0 || c[2] > 0);
        assert!(has_grid, "grid should produce visible pixels");
    }

    #[test]
    fn test_vertex3_translate() {
        let v = Vertex3::new(1.0, 2.0, 3.0);
        let t = v.translate(1.0, -1.0, 0.5);
        assert!((t.x - 2.0).abs() < 1e-9);
        assert!((t.y - 1.0).abs() < 1e-9);
        assert!((t.z - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_vertex3_scale() {
        let v = Vertex3::new(2.0, 4.0, 6.0);
        let s = v.scale(0.5);
        assert!((s.x - 1.0).abs() < 1e-9);
        assert!((s.y - 2.0).abs() < 1e-9);
        assert!((s.z - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stage_frame_get_set_pixel() {
        let mut frame = StageFrame::new(10, 10, [0, 0, 0]);
        frame.set_pixel(5, 5, [255, 128, 0]);
        let px = frame.get_pixel(5, 5);
        assert_eq!(px, Some([255, 128, 0]));
    }

    #[test]
    fn test_stage_frame_out_of_bounds() {
        let frame = StageFrame::new(10, 10, [0, 0, 0]);
        assert!(frame.get_pixel(10, 10).is_none());
    }
}
