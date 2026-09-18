# Progress Visualization for Long-Running Operations

This document outlines a detailed implementation plan for enhancing the logging and diagnostics module with advanced progress visualization capabilities.

## Overview

Scientific computing often involves long-running operations where users need to understand progress, estimated completion time, and resource utilization. The enhanced progress visualization system will provide rich, interactive terminal-based progress indicators along with statistical information.

## Architecture

```
scirs2-core/src/logging/
├── mod.rs                  # Module exports
├── logger.rs               # Existing logger implementation
├── levels.rs               # Log levels definition
├── progress/               # Enhanced progress tracking
│   ├── mod.rs
│   ├── tracker.rs          # Core progress tracking
│   ├── formats.rs          # Progress visualization formats
│   ├── statistics.rs       # Statistical calculations
│   ├── adaptive.rs         # Adaptive update rate
│   └── renderer.rs         # Terminal rendering
└── formatters/             # Log formatters
    ├── mod.rs
    ├── text.rs             # Text formatter
    ├── json.rs             # JSON formatter
    └── progress.rs         # Progress log formatter
```

## Core Components

### 1. Enhanced Progress Tracker

```rust
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

/// Progress visualization style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProgressStyle {
    /// Simple percentage text (e.g., "45%")
    Percentage,
    /// Basic ASCII progress bar
    Bar,
    /// Unicode block characters for smoother bar
    BlockBar,
    /// Spinner with percentage
    Spinner,
    /// Detailed bar with additional statistics
    DetailedBar,
}

/// Progress tracker configuration
#[derive(Debug, Clone)]
pub struct ProgressConfig {
    /// The style of progress visualization
    pub style: ProgressStyle,
    /// Width of the progress bar in characters
    pub width: usize,
    /// Whether to show ETA
    pub show_eta: bool,
    /// Whether to show detailed statistics
    pub show_statistics: bool,
    /// Whether to show transfer rate (items/second)
    pub show_speed: bool,
    /// Whether to use adaptive update rate
    pub adaptive_rate: bool,
    /// Minimum update interval
    pub min_update_interval: Duration,
    /// Maximum update interval
    pub max_update_interval: Duration,
    /// The template for progress display
    pub template: Option<String>,
    /// Custom symbols for the progress bar
    pub symbols: Option<ProgressSymbols>,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            style: ProgressStyle::BlockBar,
            width: 40,
            show_eta: true,
            show_statistics: true,
            show_speed: true,
            adaptive_rate: true,
            min_update_interval: Duration::from_millis(100),
            max_update_interval: Duration::from_secs(1),
            template: None,
            symbols: None,
        }
    }
}

/// Custom symbols for progress visualization
#[derive(Debug, Clone)]
pub struct ProgressSymbols {
    /// Start of progress bar
    pub start: String,
    /// End of progress bar
    pub end: String,
    /// Filled section of progress bar
    pub fill: String,
    /// Empty section of progress bar
    pub empty: String,
    /// Spinner frames
    pub spinner: Vec<String>,
}

impl Default for ProgressSymbols {
    fn default() -> Self {
        Self {
            start: "[".to_string(),
            end: "]".to_string(),
            fill: "=".to_string(),
            empty: " ".to_string(),
            spinner: vec!["-".to_string(), "\\".to_string(), "|".to_string(), "/".to_string()],
        }
    }
}

impl ProgressSymbols {
    /// Create symbols for block-style progress bar
    pub fn blocks() -> Self {
        Self {
            start: "│".to_string(),
            end: "│".to_string(),
            fill: "█".to_string(),
            empty: " ".to_string(),
            spinner: vec!["⠋".to_string(), "⠙".to_string(), "⠹".to_string(), "⠸".to_string(), 
                          "⠼".to_string(), "⠴".to_string(), "⠦".to_string(), "⠧".to_string(), 
                          "⠇".to_string(), "⠏".to_string()],
        }
    }
}

/// Progress tracking statistics
#[derive(Debug, Clone)]
pub struct ProgressStats {
    /// Number of items processed
    pub processed: u64,
    /// Total number of items to process
    pub total: u64,
    /// Percentage complete (0-100)
    pub percentage: f64,
    /// Items per second
    pub items_per_second: f64,
    /// Estimated time remaining
    pub eta: Duration,
    /// Time elapsed
    pub elapsed: Duration,
    /// A record of recent processing speeds
    pub recent_speeds: VecDeque<f64>,
    /// Highest observed items per second
    pub max_speed: f64,
    /// Time of last update
    pub last_update: Instant,
    /// Number of updates
    pub update_count: u64,
}

impl ProgressStats {
    /// Create new progress statistics
    fn new(total: u64) -> Self {
        Self {
            processed: 0,
            total,
            percentage: 0.0,
            items_per_second: 0.0,
            eta: Duration::from_secs(0),
            elapsed: Duration::from_secs(0),
            recent_speeds: VecDeque::with_capacity(20),
            max_speed: 0.0,
            last_update: Instant::now(),
            update_count: 0,
        }
    }
    
    /// Update statistics based on current processed count
    fn update(&mut self, processed: u64, now: Instant) {
        let old_processed = self.processed;
        self.processed = processed;
        
        // Calculate percentage
        if self.total > 0 {
            self.percentage = (self.processed as f64 / self.total as f64) * 100.0;
        } else {
            self.percentage = 0.0;
        }
        
        // Calculate elapsed time
        let time_diff = now.duration_since(self.last_update);
        self.elapsed += time_diff;
        
        // Calculate processing speed
        let items_diff = self.processed - old_processed;
        if items_diff > 0 && !time_diff.is_zero() {
            let speed = items_diff as f64 / time_diff.as_secs_f64();
            self.recent_speeds.push_back(speed);
            
            // Keep only the last 20 speed measurements
            if self.recent_speeds.len() > 20 {
                self.recent_speeds.pop_front();
            }
            
            // Calculate average speed from recent measurements
            let avg_speed: f64 = self.recent_speeds.iter().sum::<f64>() / self.recent_speeds.len() as f64;
            self.items_per_second = avg_speed;
            self.max_speed = self.max_speed.max(avg_speed);
        }
        
        // Calculate ETA
        if self.items_per_second > 0.0 && self.processed < self.total {
            let remaining_items = self.total - self.processed;
            let remaining_seconds = remaining_items as f64 / self.items_per_second;
            self.eta = Duration::from_secs_f64(remaining_seconds);
        } else {
            self.eta = Duration::from_secs(0);
        }
        
        self.last_update = now;
        self.update_count += 1;
    }
}

/// Enhanced progress tracker
pub struct EnhancedProgressTracker {
    /// The progress description
    pub description: String,
    /// Progress configuration
    pub config: ProgressConfig,
    /// Progress statistics
    stats: Arc<Mutex<ProgressStats>>,
    /// Start time
    start_time: Instant,
    /// Is the progress tracking active?
    active: bool,
    /// Should the progress bar be hidden?
    hidden: bool,
    /// Renderer for the progress bar
    renderer: ProgressRenderer,
}

impl EnhancedProgressTracker {
    /// Create a new progress tracker
    pub fn new(description: &str, total: u64) -> Self {
        let config = ProgressConfig::default();
        let stats = Arc::new(Mutex::new(ProgressStats::new(total)));
        let renderer = ProgressRenderer::new();
        
        Self {
            description: description.to_string(),
            config,
            stats,
            start_time: Instant::now(),
            active: false,
            hidden: false,
            renderer,
        }
    }
    
    /// Configure the progress tracker
    pub fn with_config(mut self, config: ProgressConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Use a specific progress style
    pub fn with_style(mut self, style: ProgressStyle) -> Self {
        self.config.style = style;
        self
    }
    
    /// Set custom symbols
    pub fn with_symbols(mut self, symbols: ProgressSymbols) -> Self {
        self.config.symbols = Some(symbols);
        self
    }
    
    /// Show or hide ETA
    pub fn with_eta(mut self, show: bool) -> Self {
        self.config.show_eta = show;
        self
    }
    
    /// Show or hide statistics
    pub fn with_statistics(mut self, show: bool) -> Self {
        self.config.show_statistics = show;
        self
    }
    
    /// Start tracking progress
    pub fn start(&mut self) {
        self.active = true;
        self.start_time = Instant::now();
        
        // Initialize terminal if needed
        if !self.hidden {
            self.renderer.init();
            self.render();
        }
    }
    
    /// Update progress with current count
    pub fn update(&mut self, processed: u64) {
        if !self.active {
            return;
        }
        
        let now = Instant::now();
        
        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.update(processed, now);
        }
        
        // Determine if we should render an update
        let should_render = if self.config.adaptive_rate {
            self.should_update_adaptive()
        } else {
            self.should_update_fixed()
        };
        
        if should_render && !self.hidden {
            self.render();
        }
    }
    
    /// Increment progress by a specified amount
    pub fn increment(&mut self, amount: u64) {
        if !self.active {
            return;
        }
        
        let processed = {
            let stats = self.stats.lock().unwrap();
            stats.processed + amount
        };
        
        self.update(processed);
    }
    
    /// Finish progress tracking
    pub fn finish(&mut self) {
        if !self.active {
            return;
        }
        
        self.active = false;
        
        // Set processed to total
        {
            let mut stats = self.stats.lock().unwrap();
            stats.processed = stats.total;
            stats.percentage = 100.0;
            stats.eta = Duration::from_secs(0);
            stats.update(stats.total, Instant::now());
        }
        
        // Final render
        if !self.hidden {
            self.render();
            self.renderer.finalize();
        }
    }
    
    /// Hide the progress bar (useful for non-interactive environments)
    pub fn hide(&mut self) {
        self.hidden = true;
    }
    
    /// Show the progress bar
    pub fn show(&mut self) {
        self.hidden = false;
        
        if self.active {
            self.render();
        }
    }
    
    /// Get the current progress statistics
    pub fn stats(&self) -> ProgressStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Determine if we should update based on fixed interval
    fn should_update_fixed(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        let elapsed = stats.last_update.elapsed();
        elapsed >= self.config.min_update_interval
    }
    
    /// Determine if we should update based on adaptive interval
    fn should_update_adaptive(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        let elapsed = stats.last_update.elapsed();
        
        // Always update if we've exceeded the maximum interval
        if elapsed >= self.config.max_update_interval {
            return true;
        }
        
        // Always update if we've exceeded the minimum interval and progress has changed significantly
        if elapsed >= self.config.min_update_interval {
            // Calculate how much progress has been made
            let progress_ratio = stats.processed as f64 / stats.total as f64;
            
            // Adaptive update logic:
            // - Update more frequently at the beginning and end
            // - Update less frequently in the middle
            let position_factor = 4.0 * progress_ratio * (1.0 - progress_ratio);
            let threshold = self.config.min_update_interval.as_secs_f64() 
                + position_factor * (self.config.max_update_interval.as_secs_f64() - self.config.min_update_interval.as_secs_f64());
            
            elapsed.as_secs_f64() >= threshold
        } else {
            false
        }
    }
    
    /// Render the progress bar
    fn render(&self) {
        if self.hidden {
            return;
        }
        
        let stats = self.stats.lock().unwrap();
        let template = self.config.template.clone();
        let symbols = self.config.symbols.clone().unwrap_or_default();
        
        match self.config.style {
            ProgressStyle::Percentage => {
                self.renderer.render_percentage(&self.description, &stats);
            },
            ProgressStyle::Bar => {
                self.renderer.render_bar(
                    &self.description, 
                    &stats, 
                    self.config.width,
                    self.config.show_eta,
                    &symbols,
                );
            },
            ProgressStyle::BlockBar => {
                let block_symbols = ProgressSymbols::blocks();
                self.renderer.render_bar(
                    &self.description, 
                    &stats, 
                    self.config.width,
                    self.config.show_eta,
                    &block_symbols,
                );
            },
            ProgressStyle::Spinner => {
                self.renderer.render_spinner(
                    &self.description, 
                    &stats, 
                    &symbols,
                    self.config.show_eta,
                );
            },
            ProgressStyle::DetailedBar => {
                self.renderer.render_detailed_bar(
                    &self.description, 
                    &stats, 
                    self.config.width,
                    self.config.show_eta,
                    self.config.show_statistics,
                    self.config.show_speed,
                    &symbols,
                );
            },
        }
    }
}

/// Progress bar renderer
struct ProgressRenderer {
    // Terminal handling state
    last_length: usize,
    spinner_index: usize,
}

impl ProgressRenderer {
    /// Create a new renderer
    fn new() -> Self {
        Self {
            last_length: 0,
            spinner_index: 0,
        }
    }
    
    /// Initialize terminal for progress rendering
    fn init(&mut self) {
        // Nothing needed for simple terminal output
        // In a real implementation, we might set up terminal modes
    }
    
    /// Finalize terminal after progress rendering
    fn finalize(&mut self) {
        // Print a newline to ensure next output starts on a fresh line
        println!();
    }
    
    /// Render percentage-only progress
    fn render_percentage(&self, description: &str, stats: &ProgressStats) {
        let output = format!("{}: {:.1}%", description, stats.percentage);
        self.print_progress(&output);
    }
    
    /// Render basic progress bar
    fn render_bar(
        &self, 
        description: &str, 
        stats: &ProgressStats, 
        width: usize,
        show_eta: bool,
        symbols: &ProgressSymbols,
    ) {
        let percentage = stats.percentage;
        let filled_width = ((percentage / 100.0) * width as f64) as usize;
        let empty_width = width.saturating_sub(filled_width);
        
        let progress_bar = format!(
            "{}{:fill$}{:empty$}{}",
            symbols.start,
            symbols.fill,
            symbols.empty,
            symbols.end,
            fill = filled_width,
            empty = empty_width,
        );
        
        let mut output = format!("{}: {} {:.1}%", description, progress_bar, percentage);
        
        if show_eta && stats.processed < stats.total {
            output.push_str(&format!(" ETA: {}", format_duration(&stats.eta)));
        }
        
        self.print_progress(&output);
    }
    
    /// Render spinner progress
    fn render_spinner(
        &mut self, 
        description: &str, 
        stats: &ProgressStats, 
        symbols: &ProgressSymbols,
        show_eta: bool,
    ) {
        self.spinner_index = (self.spinner_index + 1) % symbols.spinner.len();
        let spinner = &symbols.spinner[self.spinner_index];
        
        let mut output = format!("{} {} {}/{} ({:.1}%)", 
            spinner, description, stats.processed, stats.total, stats.percentage);
        
        if show_eta && stats.processed < stats.total {
            output.push_str(&format!(" ETA: {}", format_duration(&stats.eta)));
        }
        
        self.print_progress(&output);
    }
    
    /// Render detailed progress bar with statistics
    fn render_detailed_bar(
        &self, 
        description: &str, 
        stats: &ProgressStats, 
        width: usize,
        show_eta: bool,
        show_statistics: bool,
        show_speed: bool,
        symbols: &ProgressSymbols,
    ) {
        let percentage = stats.percentage;
        let filled_width = ((percentage / 100.0) * width as f64) as usize;
        let empty_width = width.saturating_sub(filled_width);
        
        let progress_bar = format!(
            "{}{:fill$}{:empty$}{}",
            symbols.start,
            symbols.fill,
            symbols.empty,
            symbols.end,
            fill = filled_width,
            empty = empty_width,
        );
        
        let mut output = format!("{}: {} {:.1}% ({}/{})", 
            description, progress_bar, percentage, stats.processed, stats.total);
        
        if show_speed {
            output.push_str(&format!(" [{:.1} it/s]", stats.items_per_second));
        }
        
        if show_eta && stats.processed < stats.total {
            output.push_str(&format!(" ETA: {}", format_duration(&stats.eta)));
        }
        
        if show_statistics {
            output.push_str(&format!(" | Elapsed: {}", format_duration(&stats.elapsed)));
            
            if stats.max_speed > 0.0 {
                output.push_str(&format!(" | Max: {:.1} it/s", stats.max_speed));
            }
        }
        
        self.print_progress(&output);
    }
    
    /// Print progress with carriage return
    fn print_progress(&self, output: &str) {
        // Clear previous output
        if self.last_length > 0 {
            print!("\r{}", " ".repeat(self.last_length));
        }
        
        // Print new output
        print!("\r{}", output);
        let _ = std::io::stdout().flush();
    }
}

/// Format duration in human-readable format
fn format_duration(duration: &Duration) -> String {
    let total_secs = duration.as_secs();
    
    if total_secs < 60 {
        return format!("{}s", total_secs);
    }
    
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    
    if mins < 60 {
        return format!("{}m {}s", mins, secs);
    }
    
    let hours = mins / 60;
    let mins = mins % 60;
    
    format!("{}h {}m {}s", hours, mins, secs)
}
```

### 2. Progress Visualization Builder

```rust
/// Builder for creating progress visualizations
pub struct ProgressBuilder {
    description: String,
    total: u64,
    style: ProgressStyle,
    width: usize,
    show_eta: bool,
    show_statistics: bool,
    show_speed: bool,
    adaptive_rate: bool,
    min_update_interval: Duration,
    max_update_interval: Duration,
    template: Option<String>,
    symbols: Option<ProgressSymbols>,
    hidden: bool,
}

impl ProgressBuilder {
    /// Create a new progress builder
    pub fn new(description: &str, total: u64) -> Self {
        Self {
            description: description.to_string(),
            total,
            style: ProgressStyle::BlockBar,
            width: 40,
            show_eta: true,
            show_statistics: true,
            show_speed: true,
            adaptive_rate: true,
            min_update_interval: Duration::from_millis(100),
            max_update_interval: Duration::from_secs(1),
            template: None,
            symbols: None,
            hidden: false,
        }
    }
    
    /// Set the progress style
    pub fn style(mut self, style: ProgressStyle) -> Self {
        self.style = style;
        self
    }
    
    /// Set the progress bar width
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }
    
    /// Show or hide ETA
    pub fn show_eta(mut self, show: bool) -> Self {
        self.show_eta = show;
        self
    }
    
    /// Show or hide statistics
    pub fn show_statistics(mut self, show: bool) -> Self {
        self.show_statistics = show;
        self
    }
    
    /// Show or hide speed
    pub fn show_speed(mut self, show: bool) -> Self {
        self.show_speed = show;
        self
    }
    
    /// Enable or disable adaptive update rate
    pub fn adaptive_rate(mut self, enable: bool) -> Self {
        self.adaptive_rate = enable;
        self
    }
    
    /// Set the minimum update interval
    pub fn min_update_interval(mut self, interval: Duration) -> Self {
        self.min_update_interval = interval;
        self
    }
    
    /// Set the maximum update interval
    pub fn max_update_interval(mut self, interval: Duration) -> Self {
        self.max_update_interval = interval;
        self
    }
    
    /// Set a custom template
    pub fn template(mut self, template: &str) -> Self {
        self.template = Some(template.to_string());
        self
    }
    
    /// Set custom symbols
    pub fn symbols(mut self, symbols: ProgressSymbols) -> Self {
        self.symbols = Some(symbols);
        self
    }
    
    /// Hide the progress bar
    pub fn hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }
    
    /// Build the progress tracker
    pub fn build(self) -> EnhancedProgressTracker {
        let config = ProgressConfig {
            style: self.style,
            width: self.width,
            show_eta: self.show_eta,
            show_statistics: self.show_statistics,
            show_speed: self.show_speed,
            adaptive_rate: self.adaptive_rate,
            min_update_interval: self.min_update_interval,
            max_update_interval: self.max_update_interval,
            template: self.template,
            symbols: self.symbols,
        };
        
        let mut tracker = EnhancedProgressTracker::new(&self.description, self.total)
            .with_config(config);
            
        if self.hidden {
            tracker.hide();
        }
        
        tracker
    }
}
```

### 3. Multi-Progress Tracking

```rust
/// Manager for multiple progress bars
pub struct MultiProgress {
    trackers: Vec<EnhancedProgressTracker>,
}

impl MultiProgress {
    /// Create a new multi-progress manager
    pub fn new() -> Self {
        Self {
            trackers: Vec::new(),
        }
    }
    
    /// Add a progress tracker
    pub fn add(&mut self, tracker: EnhancedProgressTracker) -> usize {
        let id = self.trackers.len();
        self.trackers.push(tracker);
        id
    }
    
    /// Get a progress tracker by ID
    pub fn get(&mut self, id: usize) -> Option<&mut EnhancedProgressTracker> {
        self.trackers.get_mut(id)
    }
    
    /// Start all progress trackers
    pub fn start_all(&mut self) {
        for tracker in &mut self.trackers {
            tracker.start();
        }
    }
    
    /// Update a specific tracker
    pub fn update(&mut self, id: usize, processed: u64) {
        if let Some(tracker) = self.trackers.get_mut(id) {
            tracker.update(processed);
        }
    }
    
    /// Increment a specific tracker
    pub fn increment(&mut self, id: usize, amount: u64) {
        if let Some(tracker) = self.trackers.get_mut(id) {
            tracker.increment(amount);
        }
    }
    
    /// Finish a specific tracker
    pub fn finish(&mut self, id: usize) {
        if let Some(tracker) = self.trackers.get_mut(id) {
            tracker.finish();
        }
    }
    
    /// Finish all trackers
    pub fn finish_all(&mut self) {
        for tracker in &mut self.trackers {
            tracker.finish();
        }
    }
}
```

### 4. Integration with Existing Logger

```rust
use scirs2_core::logging::{Logger, LogLevel};
use scirs2_core::logging::progress::{EnhancedProgressTracker, ProgressBuilder, ProgressStyle};

impl Logger {
    /// Track progress of a long-running operation
    pub fn track_progress(&self, description: &str, total: u64) -> EnhancedProgressTracker {
        let builder = ProgressBuilder::new(description, total)
            .style(ProgressStyle::DetailedBar)
            .show_statistics(true);
            
        let mut tracker = builder.build();
        tracker.start();
        tracker
    }
    
    /// Log a message with progress update
    pub fn info_with_progress(&self, message: &str, progress: &mut EnhancedProgressTracker, update: u64) {
        self.info(message);
        progress.update(update);
    }
    
    /// Execute an operation with progress tracking
    pub fn with_progress<F, R>(&self, description: &str, total: u64, operation: F) -> R
    where
        F: FnOnce(&mut EnhancedProgressTracker) -> R,
    {
        let mut progress = self.track_progress(description, total);
        let result = operation(&mut progress);
        progress.finish();
        result
    }
}
```

## Usage Examples

### Basic Progress Tracking

```rust
use scirs2_core::logging::progress::{ProgressBuilder, ProgressStyle};
use std::time::Duration;
use std::thread::sleep;

fn process_items() {
    let total_items = 100;
    
    // Create and configure a progress tracker
    let mut progress = ProgressBuilder::new("Processing items", total_items)
        .style(ProgressStyle::DetailedBar)
        .show_eta(true)
        .show_statistics(true)
        .build();
    
    // Start tracking
    progress.start();
    
    // Process items
    for i in 0..total_items {
        // Simulate processing
        sleep(Duration::from_millis(50));
        
        // Update progress
        progress.update(i + 1);
    }
    
    // Finish tracking
    progress.finish();
}
```

### Different Progress Styles

```rust
use scirs2_core::logging::progress::{ProgressBuilder, ProgressStyle, ProgressSymbols};
use std::time::Duration;
use std::thread::sleep;

fn demonstrate_progress_styles() {
    let total = 100;
    let styles = [
        ProgressStyle::Percentage,
        ProgressStyle::Bar,
        ProgressStyle::BlockBar,
        ProgressStyle::Spinner,
        ProgressStyle::DetailedBar,
    ];
    
    for style in &styles {
        println!("\nDemonstrating style: {:?}", style);
        
        let mut progress = ProgressBuilder::new("Processing", total)
            .style(*style)
            .build();
            
        progress.start();
        
        for i in 0..total {
            sleep(Duration::from_millis(20));
            progress.update(i + 1);
        }
        
        progress.finish();
    }
    
    // Custom symbols example
    println!("\nDemonstrating custom symbols");
    
    let custom_symbols = ProgressSymbols {
        start: "【".to_string(),
        end: "】".to_string(),
        fill: "■".to_string(),
        empty: "□".to_string(),
        spinner: vec!["◐".to_string(), "◓".to_string(), "◑".to_string(), "◒".to_string()],
    };
    
    let mut progress = ProgressBuilder::new("Custom symbols", total)
        .style(ProgressStyle::Bar)
        .symbols(custom_symbols)
        .build();
        
    progress.start();
    
    for i in 0..total {
        sleep(Duration::from_millis(20));
        progress.update(i + 1);
    }
    
    progress.finish();
}
```

### Multiple Progress Bars

```rust
use scirs2_core::logging::progress::{ProgressBuilder, ProgressStyle, MultiProgress};
use std::time::Duration;
use std::thread;

fn multi_progress_example() {
    let mut multi = MultiProgress::new();
    
    // Add multiple progress trackers
    let task1_id = multi.add(
        ProgressBuilder::new("Task 1", 100)
            .style(ProgressStyle::Bar)
            .build()
    );
    
    let task2_id = multi.add(
        ProgressBuilder::new("Task 2", 75)
            .style(ProgressStyle::DetailedBar)
            .build()
    );
    
    let task3_id = multi.add(
        ProgressBuilder::new("Task 3", 50)
            .style(ProgressStyle::Spinner)
            .build()
    );
    
    // Start all trackers
    multi.start_all();
    
    // Spawn threads to update each tracker
    let handle1 = thread::spawn(move || {
        for i in 0..100 {
            thread::sleep(Duration::from_millis(100));
            multi.update(task1_id, i + 1);
        }
        multi.finish(task1_id);
    });
    
    let handle2 = thread::spawn(move || {
        for i in 0..75 {
            thread::sleep(Duration::from_millis(150));
            multi.update(task2_id, i + 1);
        }
        multi.finish(task2_id);
    });
    
    let handle3 = thread::spawn(move || {
        for i in 0..50 {
            thread::sleep(Duration::from_millis(200));
            multi.update(task3_id, i + 1);
        }
        multi.finish(task3_id);
    });
    
    // Wait for all threads to complete
    handle1.join().unwrap();
    handle2.join().unwrap();
    handle3.join().unwrap();
}
```

### Integration with Logger

```rust
use scirs2_core::logging::{Logger, LogLevel};
use std::time::Duration;
use std::thread::sleep;

fn process_with_logging() {
    let logger = Logger::new("processor")
        .with_field("operation", "data_processing")
        .with_field("batch", "batch_123");
    
    logger.info("Starting data processing");
    
    // Using the logger's progress tracking
    logger.with_progress("Processing data", 100, |progress| {
        for i in 0..100 {
            // Simulate processing
            sleep(Duration::from_millis(50));
            
            // Log important steps
            if (i + 1) % 25 == 0 {
                logger.info_with_progress(
                    &format!("Completed processing batch {}/4", (i + 1) / 25),
                    progress,
                    i + 1
                );
            } else {
                // Just update progress without logging
                progress.update(i + 1);
            }
        }
        
        logger.info("Data processing completed successfully");
    });
}
```

### Dynamic Progress Example

```rust
use scirs2_core::logging::progress::{ProgressBuilder, ProgressStyle};
use std::time::Duration;
use std::thread::sleep;
use rand::Rng;

fn dynamic_progress_example() {
    // Create a progress tracker with unknown total
    let mut progress = ProgressBuilder::new("Downloading data", 0)
        .style(ProgressStyle::Spinner)
        .build();
    
    progress.start();
    
    let mut rng = rand::thread_rng();
    let mut downloaded = 0;
    let mut total_size = 0;
    
    // Simulate discovering the total size after starting
    sleep(Duration::from_secs(1));
    total_size = rng.gen_range(100..200);
    
    // Update the total
    {
        let mut stats = progress.stats.lock().unwrap();
        stats.total = total_size;
    }
    
    // Continue with normal progress updates
    while downloaded < total_size {
        // Simulate variable download speeds
        let chunk_size = rng.gen_range(1..5);
        downloaded += chunk_size;
        downloaded = downloaded.min(total_size);
        
        sleep(Duration::from_millis(100));
        progress.update(downloaded);
    }
    
    progress.finish();
}
```

### Advanced Integration Example

```rust
use scirs2_core::logging::progress::{ProgressBuilder, ProgressStyle};
use scirs2_core::logging::Logger;
use scirs2_core::profiling::Timer;
use scirs2_core::memory::metrics::MemoryMetrics;
use std::time::Duration;
use std::thread::sleep;

fn integrated_example() {
    let logger = Logger::new("integrated_example");
    let _timer = Timer::start("process_large_dataset");
    let _memory_tracker = MemoryMetrics::track_scope("dataset_processing", None);
    
    logger.info("Starting large dataset processing");
    
    // Create a progress tracker
    let mut progress = ProgressBuilder::new("Processing large dataset", 1000)
        .style(ProgressStyle::DetailedBar)
        .show_statistics(true)
        .show_speed(true)
        .build();
    
    progress.start();
    
    // Process in chunks
    for chunk_id in 0..10 {
        logger.info(&format!("Processing chunk {}/10", chunk_id + 1));
        
        let chunk_timer = Timer::start(&format!("process_chunk_{}", chunk_id));
        
        // Process items in the chunk
        for item_id in 0..100 {
            // Simulate processing
            let global_item_id = chunk_id * 100 + item_id;
            
            // Simulate variable processing time
            let processing_time = if item_id % 20 == 0 {
                // Occasional slow item
                200
            } else {
                // Normal items
                10
            };
            
            sleep(Duration::from_millis(processing_time));
            
            // Update progress
            progress.update(global_item_id + 1);
        }
        
        chunk_timer.stop();
        
        // Log chunk completion with timing
        if let Ok(profiler) = Timer::global_profiler().lock() {
            if let Some((_, _, avg, _)) = profiler.get_timing_stats(&format!("process_chunk_{}", chunk_id)) {
                logger.info(&format!("Completed chunk {}/10 in {:.2} ms", 
                    chunk_id + 1, avg.as_secs_f64() * 1000.0));
            }
        }
    }
    
    progress.finish();
    
    // Log final results
    logger.info("Dataset processing complete");
    
    // Log memory usage
    let peak_memory = MemoryMetrics::peak_usage();
    logger.info(&format!("Peak memory usage: {} bytes", peak_memory));
}
```

## Benefits of Enhanced Progress Visualization

1. **Improved User Experience**: Rich, interactive progress indicators provide better feedback
2. **Detailed Statistics**: Users can see processing speed, estimated completion time, and other metrics
3. **Multiple Visualization Styles**: Different styles for different contexts (CLI, log files, etc.)
4. **Multi-Progress Support**: Track multiple operations simultaneously
5. **Integration with Logging**: Seamless integration with the existing logging system
6. **Adaptive Updates**: Intelligent update rate to balance responsiveness and performance
7. **Customization**: Flexible customization of appearance and behavior

## Integration with Other Core Features

This enhanced progress tracking system integrates well with other core features:

1. **Profiling**: Track timing of operations along with progress
2. **Memory Management**: Monitor memory usage during long-running operations
3. **GPU Acceleration**: Track progress of GPU operations
4. **Logging**: Seamless integration with structured logging

## Next Steps

1. Implement the base progress visualization system
2. Add different visualization styles (bar, spinner, etc.)
3. Create the multi-progress tracking capability
4. Integrate with the existing logging system
5. Add adaptive update rate logic
6. Implement terminal-specific enhancements for better display
7. Create examples showing integration with other core features