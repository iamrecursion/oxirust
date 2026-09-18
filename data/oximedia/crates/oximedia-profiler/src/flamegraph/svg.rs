//! SVG flame graph rendering.

use super::generate::{FlameGraphData, FlameNode};

/// SVG renderer for flame graphs.
#[derive(Debug)]
pub struct SvgRenderer {
    width: u32,
    height: u32,
    frame_height: u32,
}

impl SvgRenderer {
    /// Create a new SVG renderer.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frame_height: 16,
        }
    }

    /// Render flame graph to SVG.
    pub fn render(&self, data: &FlameGraphData) -> String {
        let mut svg = String::new();

        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.width, self.height
        ));

        svg.push_str(
            r#"<style>
            .frame { stroke: white; stroke-width: 1; }
            .frame:hover { stroke: black; stroke-width: 2; }
            text { font-family: monospace; font-size: 12px; }
        </style>"#,
        );

        self.render_node(&mut svg, &data.root, 0, 0, self.width, data.total_samples);

        svg.push_str("</svg>");
        svg
    }

    /// Render a single node.
    #[allow(clippy::too_many_arguments)]
    fn render_node(
        &self,
        svg: &mut String,
        node: &FlameNode,
        x: u32,
        y: u32,
        width: u32,
        total: u64,
    ) {
        if node.value == 0 || node.name == "root" {
            let mut x_offset = x;
            for child in &node.children {
                let child_width = (width as f64 * (child.value as f64 / total as f64)) as u32;
                self.render_node(svg, child, x_offset, y, child_width, total);
                x_offset += child_width;
            }
            return;
        }

        let percentage = (node.value as f64 / total as f64) * 100.0;
        let color = self.get_color(percentage);

        svg.push_str(&format!(
            r#"<rect class="frame" x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
            x, y, width, self.frame_height, color
        ));

        if width > 20 {
            let text_x = x + 2;
            let text_y = y + 12;
            svg.push_str(&format!(
                r#"<text x="{}" y="{}">{}</text>"#,
                text_x, text_y, node.name
            ));
        }

        let mut x_offset = x;
        for child in &node.children {
            let child_width = (width as f64 * (child.value as f64 / node.value as f64)) as u32;
            self.render_node(
                svg,
                child,
                x_offset,
                y + self.frame_height,
                child_width,
                total,
            );
            x_offset += child_width;
        }
    }

    /// Get color for a percentage.
    fn get_color(&self, percentage: f64) -> String {
        let hue = (1.0 - (percentage / 100.0)) * 60.0;
        format!("hsl({}, 70%, 50%)", hue as i32)
    }
}

impl Default for SvgRenderer {
    fn default() -> Self {
        Self::new(1200, 800)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_renderer() {
        let renderer = SvgRenderer::new(1200, 800);
        assert_eq!(renderer.width, 1200);
        assert_eq!(renderer.height, 800);
    }

    #[test]
    fn test_svg_generation() {
        let renderer = SvgRenderer::new(100, 100);
        let node = FlameNode {
            name: "test".to_string(),
            value: 10,
            children: Vec::new(),
        };
        let data = FlameGraphData {
            root: node,
            total_samples: 10,
        };

        let svg = renderer.render(&data);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_color_generation() {
        let renderer = SvgRenderer::default();
        let color1 = renderer.get_color(10.0);
        let color2 = renderer.get_color(90.0);

        assert!(color1.starts_with("hsl("));
        assert!(color2.starts_with("hsl("));
        assert_ne!(color1, color2);
    }
}
