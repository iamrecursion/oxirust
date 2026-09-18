//! Plotly backend for visualization
//!
//! This module provides an adapter for rendering visualizations using the plotly crate.

use plotly::{
    common::{ColorScale, ColorScalePalette, DashType, Line, Marker, Mode, Title},
    layout::{Annotation, Axis, Layout},
    Bar, HeatMap, Histogram, Plot, Scatter,
};
use std::error::Error;
use std::path::Path;

use crate::visualization::{
    ColorMap, PlotType, VisualizationData, VisualizationMetadata, VisualizationOptions,
};

use super::PlottingBackend;

/// A struct for rendering visualizations using plotly
pub struct PlotlyBackend;

impl Default for PlotlyBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PlotlyBackend {
    /// Create a new plotly backend
    pub fn new() -> Self {
        Self
    }

    /// Map the scirs2-metrics color map to a plotly color scale
    fn map_color_scheme(&self, colormap: &ColorMap) -> ColorScale {
        match colormap {
            ColorMap::BlueRed => ColorScale::Palette(ColorScalePalette::RdBu),
            ColorMap::GreenRed => ColorScale::Palette(ColorScalePalette::Greens),
            ColorMap::Grayscale => ColorScale::Palette(ColorScalePalette::Greys),
            ColorMap::Viridis => ColorScale::Palette(ColorScalePalette::Viridis),
            ColorMap::Magma => ColorScale::Palette(ColorScalePalette::Hot),
        }
    }

    /// Add line traces to a plot
    fn add_line_traces(
        &self,
        plot: &mut Plot,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
    ) -> Result<(), Box<dyn Error>> {
        let default_name = vec!["Series 1".to_string()];
        let series_names = data.series_names.as_ref().unwrap_or(&default_name);

        let trace = Scatter::new(data.x.clone(), data.y.clone())
            .mode(Mode::Lines)
            .name(&series_names[0]);

        plot.add_trace(trace);

        // Add additional series from the series HashMap
        if !data.series.is_empty() {
            for (name, series_data) in &data.series {
                // If we have x data for this series, use it; otherwise reuse main x
                let x_data = if let Some(x_series) = data.series.get(&format!("{}_x", name)) {
                    x_series.clone()
                } else {
                    data.x.clone()
                };

                let trace = Scatter::new(x_data, series_data.clone())
                    .mode(Mode::Lines)
                    .name(name);

                plot.add_trace(trace);
            }
        }

        Ok(())
    }

    /// Add scatter traces to a plot
    fn add_scatter_traces(
        &self,
        plot: &mut Plot,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
    ) -> Result<(), Box<dyn Error>> {
        let default_name = vec!["Series 1".to_string()];
        let series_names = data.series_names.as_ref().unwrap_or(&default_name);

        let trace = Scatter::new(data.x.clone(), data.y.clone())
            .mode(Mode::Markers)
            .name(&series_names[0]);

        plot.add_trace(trace);

        // Add additional series from the series HashMap
        if !data.series.is_empty() {
            for (name, series_data) in &data.series {
                // If we have x data for this series, use it; otherwise reuse main x
                let x_data = if let Some(x_series) = data.series.get(&format!("{}_x", name)) {
                    x_series.clone()
                } else {
                    data.x.clone()
                };

                let trace = Scatter::new(x_data, series_data.clone())
                    .mode(Mode::Markers)
                    .name(name);

                plot.add_trace(trace);
            }
        }

        Ok(())
    }

    /// Add bar traces to a plot
    fn add_bar_traces(
        &self,
        plot: &mut Plot,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
    ) -> Result<(), Box<dyn Error>> {
        let default_name = vec!["Series 1".to_string()];
        let series_names = data.series_names.as_ref().unwrap_or(&default_name);

        let trace = Bar::new(data.x.clone(), data.y.clone()).name(&series_names[0]);

        plot.add_trace(trace);

        // Add additional series from the series HashMap
        if !data.series.is_empty() {
            for (name, series_data) in &data.series {
                // If we have x data for this series, use it; otherwise reuse main x
                let x_data = if let Some(x_series) = data.series.get(&format!("{}_x", name)) {
                    x_series.clone()
                } else {
                    data.x.clone()
                };

                let trace = Bar::new(x_data, series_data.clone()).name(name);

                plot.add_trace(trace);
            }
        }

        Ok(())
    }

    /// Add heatmap traces to a plot
    fn add_heatmap_traces(
        &self,
        plot: &mut Plot,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(z_data) = &data.z {
            let colorscale = if let Some(ref color_map) = options.color_map {
                self.map_color_scheme(color_map)
            } else {
                ColorScale::Palette(ColorScalePalette::Viridis)
            };

            let trace = HeatMap::new_z(z_data.clone()).color_scale(colorscale);

            plot.add_trace(trace);
        }

        Ok(())
    }

    /// Add histogram traces to a plot
    fn add_histogram_traces(
        &self,
        plot: &mut Plot,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
    ) -> Result<(), Box<dyn Error>> {
        let default_name = vec!["Series 1".to_string()];
        let series_names = data.series_names.as_ref().unwrap_or(&default_name);

        let trace = Histogram::new(data.x.clone()).name(&series_names[0]);

        plot.add_trace(trace);

        Ok(())
    }
}

impl PlottingBackend for PlotlyBackend {
    fn save_to_file(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn Error>> {
        let mut plot = Plot::new();

        match metadata.plot_type {
            PlotType::Line => self.add_line_traces(&mut plot, data, metadata)?,
            PlotType::Scatter => self.add_scatter_traces(&mut plot, data, metadata)?,
            PlotType::Bar => self.add_bar_traces(&mut plot, data, metadata)?,
            PlotType::Heatmap => self.add_heatmap_traces(&mut plot, data, metadata, options)?,
            PlotType::Histogram => self.add_histogram_traces(&mut plot, data, metadata)?,
        }

        // Set layout options
        let layout = Layout::new()
            .title(Title::with_text(&metadata.title))
            .x_axis(Axis::new().title(Title::with_text(&metadata.x_label)))
            .y_axis(Axis::new().title(Title::with_text(&metadata.y_label)))
            .width(options.width)
            .height(options.height)
            .show_legend(options.show_legend);

        plot.set_layout(layout);

        // Save to file
        match path.as_ref().extension().and_then(|e| e.to_str()) {
            Some("html") => {
                plot.write_html(path);
                Ok(())
            }
            Some("json") => {
                let json_data = plot.to_json();
                std::fs::write(path, json_data)?;
                Ok(())
            }
            _ => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Unsupported file extension for plotly output. Only .html and .json are supported.",
            ))),
        }
    }

    fn render_svg(
        &self,
        self_data: &VisualizationData,
        _metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        // Plotly doesn't directly support SVG generation in the Rust crate
        // We'll return an error indicating this limitation
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "SVG rendering is not directly supported by the Plotly Rust crate. Use HTML output and convert externally if needed.",
        )))
    }

    fn render_png(
        &self,
        self_data: &VisualizationData,
        _metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        // Plotly doesn't directly support PNG generation in the Rust crate
        // We'll return an error indicating this limitation
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "PNG rendering is not directly supported by the Plotly Rust crate. Use HTML output and convert externally if needed.",
        )))
    }
}
