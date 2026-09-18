//! Stream widgets (chat, donations, etc.).

/// Widget configuration.
#[derive(Debug, Clone)]
pub struct Widget {
    /// Widget type
    pub widget_type: WidgetType,
    /// Configuration
    pub config: WidgetConfig,
}

/// Widget type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidgetType {
    /// Chat widget
    Chat,
    /// Donation widget
    Donation,
    /// Follower list
    FollowerList,
    /// Event list
    EventList,
    /// Goal tracker
    GoalTracker,
}

/// Widget configuration.
#[derive(Debug, Clone, Default)]
pub struct WidgetConfig {
    /// Position (x, y)
    pub position: (i32, i32),
    /// Size (width, height)
    pub size: (u32, u32),
    /// Font size
    pub font_size: u32,
}

impl Widget {
    /// Create a new widget.
    #[must_use]
    pub fn new(widget_type: WidgetType, config: WidgetConfig) -> Self {
        Self {
            widget_type,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_creation() {
        let widget = Widget::new(WidgetType::Chat, WidgetConfig::default());
        assert_eq!(widget.widget_type, WidgetType::Chat);
    }
}
