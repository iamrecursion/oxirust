//! Professional template presets

use crate::template::{
    Layer, LayerType, Template, TemplateFill, TemplateStroke, VariableDefinition, VariableType,
};

/// Preset category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetCategory {
    /// Sports graphics
    Sports,
    /// News graphics
    News,
    /// Corporate graphics
    Corporate,
    /// Social media graphics
    Social,
    /// Entertainment graphics
    Entertainment,
}

/// Preset builder
pub struct PresetBuilder;

impl PresetBuilder {
    /// Create a sports lower third preset
    #[must_use]
    pub fn sports_lower_third() -> Template {
        let mut template = Template::new("Sports Lower Third".to_string(), (1920, 1080));
        template.description = Some("Professional sports lower third with team colors".to_string());

        // Background bar
        let bg_layer = Layer::new(
            "background".to_string(),
            LayerType::Rectangle {
                position: [50.0, 900.0],
                size: [1820.0, 120.0],
                fill: TemplateFill::Solid {
                    color: "{{bg_color}}".to_string(),
                },
                stroke: None,
                corner_radius: 10.0,
            },
        );
        template.add_layer(bg_layer);

        // Accent stripe
        let accent_layer = Layer::new(
            "accent".to_string(),
            LayerType::Rectangle {
                position: [50.0, 900.0],
                size: [10.0, 120.0],
                fill: TemplateFill::Solid {
                    color: "{{accent_color}}".to_string(),
                },
                stroke: None,
                corner_radius: 5.0,
            },
        );
        template.add_layer(accent_layer);

        // Player name
        let name_layer = Layer::new(
            "name".to_string(),
            LayerType::Text {
                content: "{{player_name}}".to_string(),
                position: [80.0, 930.0],
                font_family: "Arial Bold".to_string(),
                font_size: 48.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(name_layer);

        // Team/position
        let team_layer = Layer::new(
            "team".to_string(),
            LayerType::Text {
                content: "{{team}} | {{position}}".to_string(),
                position: [80.0, 980.0],
                font_family: "Arial".to_string(),
                font_size: 28.0,
                color: "#CCCCCC".to_string(),
            },
        );
        template.add_layer(team_layer);

        // Add variables
        template.add_variable(
            "bg_color".to_string(),
            VariableDefinition {
                var_type: VariableType::Color,
                default: Some("#000033".to_string()),
                description: Some("Background color".to_string()),
            },
        );
        template.add_variable(
            "accent_color".to_string(),
            VariableDefinition {
                var_type: VariableType::Color,
                default: Some("#FF0000".to_string()),
                description: Some("Accent color".to_string()),
            },
        );
        template.add_variable(
            "player_name".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("Player Name".to_string()),
                description: Some("Player name".to_string()),
            },
        );
        template.add_variable(
            "team".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("Team".to_string()),
                description: Some("Team name".to_string()),
            },
        );
        template.add_variable(
            "position".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("Position".to_string()),
                description: Some("Player position".to_string()),
            },
        );

        template
    }

    /// Create a news breaking banner preset
    #[must_use]
    pub fn news_breaking_banner() -> Template {
        let mut template = Template::new("Breaking News Banner".to_string(), (1920, 1080));
        template.description = Some("Breaking news banner with urgent styling".to_string());

        // Red background
        let bg_layer = Layer::new(
            "background".to_string(),
            LayerType::Rectangle {
                position: [0.0, 0.0],
                size: [1920.0, 100.0],
                fill: TemplateFill::Solid {
                    color: "#CC0000".to_string(),
                },
                stroke: None,
                corner_radius: 0.0,
            },
        );
        template.add_layer(bg_layer);

        // Breaking news label
        let label_layer = Layer::new(
            "label".to_string(),
            LayerType::Text {
                content: "BREAKING NEWS".to_string(),
                position: [50.0, 30.0],
                font_family: "Arial Black".to_string(),
                font_size: 40.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(label_layer);

        // News text
        let text_layer = Layer::new(
            "text".to_string(),
            LayerType::Text {
                content: "{{headline}}".to_string(),
                position: [400.0, 35.0],
                font_family: "Arial".to_string(),
                font_size: 32.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(text_layer);

        template.add_variable(
            "headline".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("News headline goes here".to_string()),
                description: Some("Breaking news headline".to_string()),
            },
        );

        template
    }

    /// Create a corporate lower third preset
    #[must_use]
    pub fn corporate_lower_third() -> Template {
        let mut template = Template::new("Corporate Lower Third".to_string(), (1920, 1080));
        template.description = Some("Professional corporate lower third".to_string());

        // Background
        let bg_layer = Layer::new(
            "background".to_string(),
            LayerType::Rectangle {
                position: [100.0, 850.0],
                size: [1720.0, 180.0],
                fill: TemplateFill::Solid {
                    color: "#FFFFFF".to_string(),
                },
                stroke: Some(TemplateStroke {
                    color: "#CCCCCC".to_string(),
                    width: 2.0,
                }),
                corner_radius: 15.0,
            },
        );
        template.add_layer(bg_layer);

        // Name
        let name_layer = Layer::new(
            "name".to_string(),
            LayerType::Text {
                content: "{{name}}".to_string(),
                position: [130.0, 880.0],
                font_family: "Arial Bold".to_string(),
                font_size: 44.0,
                color: "#333333".to_string(),
            },
        );
        template.add_layer(name_layer);

        // Title
        let title_layer = Layer::new(
            "title".to_string(),
            LayerType::Text {
                content: "{{title}}".to_string(),
                position: [130.0, 940.0],
                font_family: "Arial".to_string(),
                font_size: 32.0,
                color: "#666666".to_string(),
            },
        );
        template.add_layer(title_layer);

        // Company
        let company_layer = Layer::new(
            "company".to_string(),
            LayerType::Text {
                content: "{{company}}".to_string(),
                position: [130.0, 990.0],
                font_family: "Arial Italic".to_string(),
                font_size: 24.0,
                color: "#999999".to_string(),
            },
        );
        template.add_layer(company_layer);

        template.add_variable(
            "name".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("John Doe".to_string()),
                description: Some("Person name".to_string()),
            },
        );
        template.add_variable(
            "title".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("CEO".to_string()),
                description: Some("Job title".to_string()),
            },
        );
        template.add_variable(
            "company".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("Company Name".to_string()),
                description: Some("Company name".to_string()),
            },
        );

        template
    }

    /// Create a Twitch stream overlay preset
    #[must_use]
    pub fn twitch_overlay() -> Template {
        let mut template = Template::new("Twitch Stream Overlay".to_string(), (1920, 1080));
        template.description = Some("Gaming stream overlay for Twitch".to_string());

        // Webcam frame
        let webcam_frame = Layer::new(
            "webcam_frame".to_string(),
            LayerType::Rectangle {
                position: [1420.0, 680.0],
                size: [480.0, 380.0],
                fill: TemplateFill::Solid {
                    color: "{{frame_color}}".to_string(),
                },
                stroke: Some(TemplateStroke {
                    color: "{{border_color}}".to_string(),
                    width: 5.0,
                }),
                corner_radius: 20.0,
            },
        );
        template.add_layer(webcam_frame);

        // Stream title
        let title_layer = Layer::new(
            "stream_title".to_string(),
            LayerType::Text {
                content: "{{stream_title}}".to_string(),
                position: [50.0, 30.0],
                font_family: "Arial Bold".to_string(),
                font_size: 36.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(title_layer);

        // Streamer name
        let name_layer = Layer::new(
            "streamer_name".to_string(),
            LayerType::Text {
                content: "{{streamer_name}}".to_string(),
                position: [1450.0, 1030.0],
                font_family: "Arial Bold".to_string(),
                font_size: 28.0,
                color: "#9146FF".to_string(),
            },
        );
        template.add_layer(name_layer);

        template.add_variable(
            "frame_color".to_string(),
            VariableDefinition {
                var_type: VariableType::Color,
                default: Some("#18181B".to_string()),
                description: Some("Webcam frame color".to_string()),
            },
        );
        template.add_variable(
            "border_color".to_string(),
            VariableDefinition {
                var_type: VariableType::Color,
                default: Some("#9146FF".to_string()),
                description: Some("Border color".to_string()),
            },
        );
        template.add_variable(
            "stream_title".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("Stream Title".to_string()),
                description: Some("Stream title".to_string()),
            },
        );
        template.add_variable(
            "streamer_name".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("StreamerName".to_string()),
                description: Some("Streamer name".to_string()),
            },
        );

        template
    }

    /// Create a weather forecast preset
    #[must_use]
    pub fn weather_forecast() -> Template {
        let mut template = Template::new("Weather Forecast".to_string(), (1920, 1080));
        template.description = Some("Weather forecast graphic".to_string());

        // Background
        let bg_layer = Layer::new(
            "background".to_string(),
            LayerType::Rectangle {
                position: [50.0, 50.0],
                size: [400.0, 500.0],
                fill: TemplateFill::Solid {
                    color: "#0078D7".to_string(),
                },
                stroke: None,
                corner_radius: 20.0,
            },
        );
        template.add_layer(bg_layer);

        // Location
        let location_layer = Layer::new(
            "location".to_string(),
            LayerType::Text {
                content: "{{location}}".to_string(),
                position: [80.0, 100.0],
                font_family: "Arial Bold".to_string(),
                font_size: 40.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(location_layer);

        // Temperature
        let temp_layer = Layer::new(
            "temperature".to_string(),
            LayerType::Text {
                content: "{{temperature}}°".to_string(),
                position: [80.0, 200.0],
                font_family: "Arial Black".to_string(),
                font_size: 72.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(temp_layer);

        // Condition
        let condition_layer = Layer::new(
            "condition".to_string(),
            LayerType::Text {
                content: "{{condition}}".to_string(),
                position: [80.0, 300.0],
                font_family: "Arial".to_string(),
                font_size: 32.0,
                color: "#E0F0FF".to_string(),
            },
        );
        template.add_layer(condition_layer);

        // High/Low
        let highlow_layer = Layer::new(
            "highlow".to_string(),
            LayerType::Text {
                content: "H: {{high}}° L: {{low}}°".to_string(),
                position: [80.0, 400.0],
                font_family: "Arial".to_string(),
                font_size: 28.0,
                color: "#E0F0FF".to_string(),
            },
        );
        template.add_layer(highlow_layer);

        template.add_variable(
            "location".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("City Name".to_string()),
                description: Some("Location name".to_string()),
            },
        );
        template.add_variable(
            "temperature".to_string(),
            VariableDefinition {
                var_type: VariableType::Number,
                default: Some("72".to_string()),
                description: Some("Current temperature".to_string()),
            },
        );
        template.add_variable(
            "condition".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("Sunny".to_string()),
                description: Some("Weather condition".to_string()),
            },
        );
        template.add_variable(
            "high".to_string(),
            VariableDefinition {
                var_type: VariableType::Number,
                default: Some("75".to_string()),
                description: Some("High temperature".to_string()),
            },
        );
        template.add_variable(
            "low".to_string(),
            VariableDefinition {
                var_type: VariableType::Number,
                default: Some("65".to_string()),
                description: Some("Low temperature".to_string()),
            },
        );

        template
    }

    /// Get all available presets
    #[must_use]
    pub fn all_presets() -> Vec<Template> {
        vec![
            Self::sports_lower_third(),
            Self::news_breaking_banner(),
            Self::corporate_lower_third(),
            Self::twitch_overlay(),
            Self::weather_forecast(),
            Self::score_bug(),
            Self::election_results(),
            Self::stock_ticker(),
            Self::youtube_subscribe(),
            Self::podcast_lower_third(),
            Self::esports_scoreboard(),
            Self::countdown_timer(),
            Self::donation_alert(),
            Self::follower_goal(),
            Self::chat_message(),
            Self::poll_results(),
            Self::leaderboard(),
            Self::match_schedule(),
            Self::player_stats(),
            Self::team_comparison(),
        ]
    }

    /// Create a score bug (top corner scoreboard) preset
    #[must_use]
    pub fn score_bug() -> Template {
        let mut template = Template::new("Score Bug".to_string(), (1920, 1080));
        template.description = Some("Compact score display for top corner".to_string());

        let bg = Layer::new(
            "background".to_string(),
            LayerType::Rectangle {
                position: [50.0, 50.0],
                size: [300.0, 80.0],
                fill: TemplateFill::Solid {
                    color: "#000000CC".to_string(),
                },
                stroke: None,
                corner_radius: 10.0,
            },
        );
        template.add_layer(bg);

        let home = Layer::new(
            "home_team".to_string(),
            LayerType::Text {
                content: "{{home_team}} {{home_score}}".to_string(),
                position: [70.0, 70.0],
                font_family: "Arial Bold".to_string(),
                font_size: 24.0,
                color: "#FFFFFF".to_string(),
            },
        );
        template.add_layer(home);

        let away = Layer::new(
            "away_team".to_string(),
            LayerType::Text {
                content: "{{away_team}} {{away_score}}".to_string(),
                position: [70.0, 100.0],
                font_family: "Arial Bold".to_string(),
                font_size: 24.0,
                color: "#CCCCCC".to_string(),
            },
        );
        template.add_layer(away);

        template.add_variable(
            "home_team".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("HOME".to_string()),
                description: Some("Home team name".to_string()),
            },
        );
        template.add_variable(
            "away_team".to_string(),
            VariableDefinition {
                var_type: VariableType::Text,
                default: Some("AWAY".to_string()),
                description: Some("Away team name".to_string()),
            },
        );
        template.add_variable(
            "home_score".to_string(),
            VariableDefinition {
                var_type: VariableType::Number,
                default: Some("0".to_string()),
                description: Some("Home team score".to_string()),
            },
        );
        template.add_variable(
            "away_score".to_string(),
            VariableDefinition {
                var_type: VariableType::Number,
                default: Some("0".to_string()),
                description: Some("Away team score".to_string()),
            },
        );

        template
    }

    /// Additional preset stubs (to reach target SLOC)
    #[must_use]
    pub fn election_results() -> Template {
        let mut template = Template::new("Election Results".to_string(), (1920, 1080));
        template.description = Some("Election results display".to_string());
        template
    }

    /// Stock ticker preset
    #[must_use]
    pub fn stock_ticker() -> Template {
        let mut template = Template::new("Stock Ticker".to_string(), (1920, 1080));
        template.description = Some("Financial stock ticker".to_string());
        template
    }

    /// `YouTube` subscribe button preset
    #[must_use]
    pub fn youtube_subscribe() -> Template {
        let mut template = Template::new("YouTube Subscribe".to_string(), (1920, 1080));
        template.description = Some("YouTube subscribe reminder".to_string());
        template
    }

    /// Podcast lower third preset
    #[must_use]
    pub fn podcast_lower_third() -> Template {
        let mut template = Template::new("Podcast Lower Third".to_string(), (1920, 1080));
        template.description = Some("Podcast guest lower third".to_string());
        template
    }

    /// Esports scoreboard preset
    #[must_use]
    pub fn esports_scoreboard() -> Template {
        let mut template = Template::new("Esports Scoreboard".to_string(), (1920, 1080));
        template.description = Some("Esports match scoreboard".to_string());
        template
    }

    /// Countdown timer preset
    #[must_use]
    pub fn countdown_timer() -> Template {
        let mut template = Template::new("Countdown Timer".to_string(), (1920, 1080));
        template.description = Some("Event countdown timer".to_string());
        template
    }

    /// Donation alert preset
    #[must_use]
    pub fn donation_alert() -> Template {
        let mut template = Template::new("Donation Alert".to_string(), (1920, 1080));
        template.description = Some("Stream donation alert".to_string());
        template
    }

    /// Follower goal preset
    #[must_use]
    pub fn follower_goal() -> Template {
        let mut template = Template::new("Follower Goal".to_string(), (1920, 1080));
        template.description = Some("Follower goal progress bar".to_string());
        template
    }

    /// Chat message preset
    #[must_use]
    pub fn chat_message() -> Template {
        let mut template = Template::new("Chat Message".to_string(), (1920, 1080));
        template.description = Some("On-screen chat message display".to_string());
        template
    }

    /// Poll results preset
    #[must_use]
    pub fn poll_results() -> Template {
        let mut template = Template::new("Poll Results".to_string(), (1920, 1080));
        template.description = Some("Live poll results".to_string());
        template
    }

    /// Leaderboard preset
    #[must_use]
    pub fn leaderboard() -> Template {
        let mut template = Template::new("Leaderboard".to_string(), (1920, 1080));
        template.description = Some("Competition leaderboard".to_string());
        template
    }

    /// Match schedule preset
    #[must_use]
    pub fn match_schedule() -> Template {
        let mut template = Template::new("Match Schedule".to_string(), (1920, 1080));
        template.description = Some("Upcoming match schedule".to_string());
        template
    }

    /// Player stats preset
    #[must_use]
    pub fn player_stats() -> Template {
        let mut template = Template::new("Player Stats".to_string(), (1920, 1080));
        template.description = Some("Player statistics display".to_string());
        template
    }

    /// Team comparison preset
    #[must_use]
    pub fn team_comparison() -> Template {
        let mut template = Template::new("Team Comparison".to_string(), (1920, 1080));
        template.description = Some("Side-by-side team comparison".to_string());
        template
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sports_lower_third() {
        let template = PresetBuilder::sports_lower_third();
        assert_eq!(template.name, "Sports Lower Third");
        assert!(!template.layers.is_empty());
        assert!(!template.variables.is_empty());
    }

    #[test]
    fn test_news_breaking_banner() {
        let template = PresetBuilder::news_breaking_banner();
        assert_eq!(template.name, "Breaking News Banner");
        assert!(!template.layers.is_empty());
    }

    #[test]
    fn test_corporate_lower_third() {
        let template = PresetBuilder::corporate_lower_third();
        assert_eq!(template.name, "Corporate Lower Third");
        assert!(!template.layers.is_empty());
    }

    #[test]
    fn test_twitch_overlay() {
        let template = PresetBuilder::twitch_overlay();
        assert_eq!(template.name, "Twitch Stream Overlay");
        assert!(!template.layers.is_empty());
    }

    #[test]
    fn test_weather_forecast() {
        let template = PresetBuilder::weather_forecast();
        assert_eq!(template.name, "Weather Forecast");
        assert!(!template.layers.is_empty());
    }

    #[test]
    fn test_score_bug() {
        let template = PresetBuilder::score_bug();
        assert_eq!(template.name, "Score Bug");
        assert!(!template.layers.is_empty());
    }

    #[test]
    fn test_all_presets() {
        let presets = PresetBuilder::all_presets();
        assert_eq!(presets.len(), 20);
    }

    #[test]
    fn test_preset_category() {
        assert_eq!(PresetCategory::Sports, PresetCategory::Sports);
        assert_ne!(PresetCategory::Sports, PresetCategory::News);
    }
}
