//! Interactive HTML flame graph rendering.

use super::generate::FlameGraphData;
use super::svg::SvgRenderer;

/// Interactive HTML renderer for flame graphs.
#[derive(Debug)]
pub struct InteractiveRenderer {
    svg_renderer: SvgRenderer,
}

impl InteractiveRenderer {
    /// Create a new interactive renderer.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            svg_renderer: SvgRenderer::new(width, height),
        }
    }

    /// Render flame graph to interactive HTML.
    pub fn render(&self, data: &FlameGraphData) -> String {
        let svg = self.svg_renderer.render(data);

        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>OxiMedia Flame Graph</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { margin: 0; padding: 20px; font-family: sans-serif; }\n");
        html.push_str("#info { padding: 10px; background: #f0f0f0; margin-bottom: 10px; }\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str("<h1>OxiMedia Flame Graph</h1>\n");
        html.push_str(&format!(
            "<div id=\"info\">Total Samples: {}</div>\n",
            data.total_samples
        ));

        html.push_str("<div id=\"flamegraph\">\n");
        html.push_str(&svg);
        html.push_str("</div>\n");

        html.push_str("<script>\n");
        html.push_str(
            r#"
            const frames = document.querySelectorAll('.frame');
            const info = document.getElementById('info');

            frames.forEach(frame => {
                frame.addEventListener('mouseenter', (e) => {
                    const rect = e.target;
                    const text = rect.nextElementSibling?.textContent || 'Unknown';
                    info.textContent = `Function: ${text}`;
                });

                frame.addEventListener('mouseleave', () => {
                    info.textContent = `Total Samples: "#,
        );
        html.push_str(&data.total_samples.to_string());
        html.push_str(
            r#"`;
                });
            });
        "#,
        );
        html.push_str("</script>\n");

        html.push_str("</body>\n</html>");

        html
    }
}

impl Default for InteractiveRenderer {
    fn default() -> Self {
        Self::new(1200, 800)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flamegraph::generate::FlameNode;

    #[test]
    fn test_interactive_renderer() {
        let renderer = InteractiveRenderer::new(1200, 800);
        let node = FlameNode {
            name: "test".to_string(),
            value: 10,
            children: Vec::new(),
        };
        let data = FlameGraphData {
            root: node,
            total_samples: 10,
        };

        let html = renderer.render(&data);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("OxiMedia Flame Graph"));
        assert!(html.contains("<script>"));
    }

    #[test]
    fn test_html_structure() {
        let renderer = InteractiveRenderer::default();
        let data = FlameGraphData {
            root: FlameNode {
                name: "root".to_string(),
                value: 0,
                children: Vec::new(),
            },
            total_samples: 0,
        };

        let html = renderer.render(&data);
        assert!(html.contains("<html>"));
        assert!(html.contains("</html>"));
        assert!(html.contains("<body>"));
        assert!(html.contains("</body>"));
    }
}
