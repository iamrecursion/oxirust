//! Weather widget graphics for broadcast overlays.
//!
//! Provides weather display widgets including current conditions,
//! forecast panels, temperature displays, and animated weather icons.

#![allow(dead_code)]

/// Weather condition type
#[derive(Debug, Clone, PartialEq)]
pub enum WeatherCondition {
    Sunny,
    PartlyCloudy,
    Cloudy,
    Rainy,
    Thunderstorm,
    Snowy,
    Foggy,
    Windy,
    Hail,
    Tornado,
}

impl WeatherCondition {
    /// Returns a short display name for the condition
    pub fn display_name(&self) -> &'static str {
        match self {
            WeatherCondition::Sunny => "Sunny",
            WeatherCondition::PartlyCloudy => "Partly Cloudy",
            WeatherCondition::Cloudy => "Cloudy",
            WeatherCondition::Rainy => "Rainy",
            WeatherCondition::Thunderstorm => "Thunderstorm",
            WeatherCondition::Snowy => "Snowy",
            WeatherCondition::Foggy => "Foggy",
            WeatherCondition::Windy => "Windy",
            WeatherCondition::Hail => "Hail",
            WeatherCondition::Tornado => "Tornado",
        }
    }

    /// Returns the icon code for this condition
    pub fn icon_code(&self) -> &'static str {
        match self {
            WeatherCondition::Sunny => "sun",
            WeatherCondition::PartlyCloudy => "cloud-sun",
            WeatherCondition::Cloudy => "cloud",
            WeatherCondition::Rainy => "cloud-rain",
            WeatherCondition::Thunderstorm => "cloud-lightning",
            WeatherCondition::Snowy => "cloud-snow",
            WeatherCondition::Foggy => "fog",
            WeatherCondition::Windy => "wind",
            WeatherCondition::Hail => "cloud-hail",
            WeatherCondition::Tornado => "tornado",
        }
    }

    /// Returns whether the condition involves precipitation
    pub fn has_precipitation(&self) -> bool {
        matches!(
            self,
            WeatherCondition::Rainy
                | WeatherCondition::Thunderstorm
                | WeatherCondition::Snowy
                | WeatherCondition::Hail
        )
    }
}

/// Temperature unit
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureUnit {
    Celsius,
    Fahrenheit,
    Kelvin,
}

impl TemperatureUnit {
    /// Returns the unit symbol
    pub fn symbol(&self) -> &'static str {
        match self {
            TemperatureUnit::Celsius => "°C",
            TemperatureUnit::Fahrenheit => "°F",
            TemperatureUnit::Kelvin => "K",
        }
    }
}

/// Temperature value with unit
#[derive(Debug, Clone, Copy)]
pub struct Temperature {
    /// Value in the specified unit
    pub value: f64,
    /// Unit of measurement
    pub unit: TemperatureUnit,
}

impl Temperature {
    /// Creates a new temperature value in Celsius
    pub fn celsius(value: f64) -> Self {
        Self {
            value,
            unit: TemperatureUnit::Celsius,
        }
    }

    /// Creates a new temperature value in Fahrenheit
    pub fn fahrenheit(value: f64) -> Self {
        Self {
            value,
            unit: TemperatureUnit::Fahrenheit,
        }
    }

    /// Converts to Celsius
    pub fn to_celsius(self) -> f64 {
        match self.unit {
            TemperatureUnit::Celsius => self.value,
            TemperatureUnit::Fahrenheit => (self.value - 32.0) * 5.0 / 9.0,
            TemperatureUnit::Kelvin => self.value - 273.15,
        }
    }

    /// Converts to Fahrenheit
    pub fn to_fahrenheit(self) -> f64 {
        match self.unit {
            TemperatureUnit::Celsius => self.value * 9.0 / 5.0 + 32.0,
            TemperatureUnit::Fahrenheit => self.value,
            TemperatureUnit::Kelvin => (self.value - 273.15) * 9.0 / 5.0 + 32.0,
        }
    }

    /// Formats the temperature for display
    pub fn display(&self) -> String {
        format!("{:.0}{}", self.value, self.unit.symbol())
    }
}

/// Wind direction as compass bearing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindDirection {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl WindDirection {
    /// Returns the compass abbreviation
    pub fn abbreviation(&self) -> &'static str {
        match self {
            WindDirection::North => "N",
            WindDirection::NorthEast => "NE",
            WindDirection::East => "E",
            WindDirection::SouthEast => "SE",
            WindDirection::South => "S",
            WindDirection::SouthWest => "SW",
            WindDirection::West => "W",
            WindDirection::NorthWest => "NW",
        }
    }

    /// Returns the bearing in degrees
    pub fn degrees(&self) -> f64 {
        match self {
            WindDirection::North => 0.0,
            WindDirection::NorthEast => 45.0,
            WindDirection::East => 90.0,
            WindDirection::SouthEast => 135.0,
            WindDirection::South => 180.0,
            WindDirection::SouthWest => 225.0,
            WindDirection::West => 270.0,
            WindDirection::NorthWest => 315.0,
        }
    }
}

/// Current weather data
#[derive(Debug, Clone)]
pub struct CurrentWeather {
    /// Location name
    pub location: String,
    /// Current temperature
    pub temperature: Temperature,
    /// Feels-like temperature
    pub feels_like: Temperature,
    /// Weather condition
    pub condition: WeatherCondition,
    /// Humidity percentage (0-100)
    pub humidity: u8,
    /// Wind speed in km/h
    pub wind_speed: f64,
    /// Wind direction
    pub wind_direction: WindDirection,
    /// Visibility in km
    pub visibility: f64,
    /// Barometric pressure in hPa
    pub pressure: f64,
    /// UV index (0-11+)
    pub uv_index: u8,
}

impl CurrentWeather {
    /// Creates a new current weather instance
    pub fn new(
        location: impl Into<String>,
        temperature: Temperature,
        condition: WeatherCondition,
    ) -> Self {
        Self {
            location: location.into(),
            temperature,
            feels_like: temperature,
            condition,
            humidity: 50,
            wind_speed: 0.0,
            wind_direction: WindDirection::North,
            visibility: 10.0,
            pressure: 1013.25,
            uv_index: 0,
        }
    }

    /// Returns whether conditions are considered severe
    pub fn is_severe(&self) -> bool {
        matches!(
            self.condition,
            WeatherCondition::Thunderstorm | WeatherCondition::Tornado | WeatherCondition::Hail
        ) || self.wind_speed > 80.0
    }
}

/// A single day forecast entry
#[derive(Debug, Clone)]
pub struct ForecastDay {
    /// Day label (e.g., "Mon", "Tuesday")
    pub label: String,
    /// High temperature
    pub high: Temperature,
    /// Low temperature
    pub low: Temperature,
    /// Expected condition
    pub condition: WeatherCondition,
    /// Precipitation probability (0-100)
    pub precip_probability: u8,
}

impl ForecastDay {
    /// Creates a new forecast day
    pub fn new(
        label: impl Into<String>,
        high: Temperature,
        low: Temperature,
        condition: WeatherCondition,
    ) -> Self {
        Self {
            label: label.into(),
            high,
            low,
            condition,
            precip_probability: 0,
        }
    }

    /// Returns the temperature range as a string
    pub fn temp_range(&self) -> String {
        format!("{} / {}", self.high.display(), self.low.display())
    }
}

/// Weather widget layout style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeatherWidgetLayout {
    /// Compact single-line display
    Compact,
    /// Standard current conditions panel
    Standard,
    /// Extended with 5-day forecast
    Extended,
    /// Full-screen weather map style
    FullScreen,
}

/// Weather widget configuration
#[derive(Debug, Clone)]
pub struct WeatherWidgetConfig {
    /// Layout style
    pub layout: WeatherWidgetLayout,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Background color (RGBA)
    pub background_color: [u8; 4],
    /// Text color (RGBA)
    pub text_color: [u8; 4],
    /// Accent color for highlights
    pub accent_color: [u8; 4],
    /// Show animated icons
    pub animated_icons: bool,
    /// Temperature display unit
    pub temperature_unit: TemperatureUnit,
    /// Show UV index
    pub show_uv_index: bool,
    /// Show wind information
    pub show_wind: bool,
}

impl Default for WeatherWidgetConfig {
    fn default() -> Self {
        Self {
            layout: WeatherWidgetLayout::Standard,
            width: 400,
            height: 200,
            background_color: [0, 0, 0, 200],
            text_color: [255, 255, 255, 255],
            accent_color: [0, 150, 255, 255],
            animated_icons: true,
            temperature_unit: TemperatureUnit::Celsius,
            show_uv_index: true,
            show_wind: true,
        }
    }
}

/// Weather widget for broadcast overlays
#[derive(Debug)]
pub struct WeatherWidget {
    /// Widget configuration
    pub config: WeatherWidgetConfig,
    /// Current weather data
    pub current: Option<CurrentWeather>,
    /// Forecast data (up to 7 days)
    pub forecast: Vec<ForecastDay>,
    /// Animation frame counter
    frame: u64,
}

impl WeatherWidget {
    /// Creates a new weather widget with default config
    pub fn new() -> Self {
        Self {
            config: WeatherWidgetConfig::default(),
            current: None,
            forecast: Vec::new(),
            frame: 0,
        }
    }

    /// Creates a weather widget with custom config
    pub fn with_config(config: WeatherWidgetConfig) -> Self {
        Self {
            config,
            current: None,
            forecast: Vec::new(),
            frame: 0,
        }
    }

    /// Sets the current weather data
    pub fn set_current(&mut self, weather: CurrentWeather) {
        self.current = Some(weather);
    }

    /// Adds a forecast day (maximum 7 days)
    pub fn add_forecast_day(&mut self, day: ForecastDay) -> bool {
        if self.forecast.len() >= 7 {
            return false;
        }
        self.forecast.push(day);
        true
    }

    /// Clears all forecast days
    pub fn clear_forecast(&mut self) {
        self.forecast.clear();
    }

    /// Advances the animation by one frame
    pub fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    /// Returns the current animation frame
    pub fn current_frame(&self) -> u64 {
        self.frame
    }

    /// Returns whether the widget has data to display
    pub fn has_data(&self) -> bool {
        self.current.is_some()
    }

    /// Returns the number of forecast days
    pub fn forecast_count(&self) -> usize {
        self.forecast.len()
    }

    /// Returns whether a severe weather alert should be shown
    pub fn show_severe_alert(&self) -> bool {
        self.current.as_ref().is_some_and(CurrentWeather::is_severe)
    }

    /// Gets the display dimensions as (width, height)
    pub fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

impl Default for WeatherWidget {
    fn default() -> Self {
        Self::new()
    }
}

/// Animated weather icon state
#[derive(Debug, Clone)]
pub struct WeatherIconAnimation {
    /// Condition being animated
    pub condition: WeatherCondition,
    /// Current animation phase (0.0 to 1.0)
    pub phase: f64,
    /// Animation speed multiplier
    pub speed: f64,
}

impl WeatherIconAnimation {
    /// Creates a new weather icon animation
    pub fn new(condition: WeatherCondition) -> Self {
        Self {
            condition,
            phase: 0.0,
            speed: 1.0,
        }
    }

    /// Advances the animation by the given delta time in seconds
    pub fn update(&mut self, delta: f64) {
        self.phase = (self.phase + delta * self.speed * 0.5) % 1.0;
    }

    /// Returns the sun rotation angle in degrees
    pub fn sun_rotation(&self) -> f64 {
        self.phase * 360.0
    }

    /// Returns the rain drop offset
    pub fn rain_offset(&self) -> f64 {
        self.phase * 20.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weather_condition_display_name() {
        assert_eq!(WeatherCondition::Sunny.display_name(), "Sunny");
        assert_eq!(
            WeatherCondition::Thunderstorm.display_name(),
            "Thunderstorm"
        );
        assert_eq!(
            WeatherCondition::PartlyCloudy.display_name(),
            "Partly Cloudy"
        );
    }

    #[test]
    fn test_weather_condition_icon_code() {
        assert_eq!(WeatherCondition::Sunny.icon_code(), "sun");
        assert_eq!(WeatherCondition::Rainy.icon_code(), "cloud-rain");
        assert_eq!(WeatherCondition::Snowy.icon_code(), "cloud-snow");
    }

    #[test]
    fn test_weather_condition_has_precipitation() {
        assert!(WeatherCondition::Rainy.has_precipitation());
        assert!(WeatherCondition::Snowy.has_precipitation());
        assert!(WeatherCondition::Thunderstorm.has_precipitation());
        assert!(WeatherCondition::Hail.has_precipitation());
        assert!(!WeatherCondition::Sunny.has_precipitation());
        assert!(!WeatherCondition::Cloudy.has_precipitation());
        assert!(!WeatherCondition::Windy.has_precipitation());
    }

    #[test]
    fn test_temperature_celsius_conversion() {
        let t = Temperature::celsius(100.0);
        assert!((t.to_celsius() - 100.0).abs() < 0.001);
        assert!((t.to_fahrenheit() - 212.0).abs() < 0.001);
    }

    #[test]
    fn test_temperature_fahrenheit_conversion() {
        let t = Temperature::fahrenheit(32.0);
        assert!((t.to_celsius() - 0.0).abs() < 0.001);
        assert!((t.to_fahrenheit() - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_temperature_display() {
        let t = Temperature::celsius(25.5);
        let s = t.display();
        assert!(s.contains("°C"));
        assert!(s.contains("26") || s.contains("25"));
    }

    #[test]
    fn test_temperature_unit_symbol() {
        assert_eq!(TemperatureUnit::Celsius.symbol(), "°C");
        assert_eq!(TemperatureUnit::Fahrenheit.symbol(), "°F");
        assert_eq!(TemperatureUnit::Kelvin.symbol(), "K");
    }

    #[test]
    fn test_wind_direction_abbreviation() {
        assert_eq!(WindDirection::North.abbreviation(), "N");
        assert_eq!(WindDirection::SouthWest.abbreviation(), "SW");
        assert_eq!(WindDirection::NorthEast.abbreviation(), "NE");
    }

    #[test]
    fn test_wind_direction_degrees() {
        assert!((WindDirection::North.degrees() - 0.0).abs() < 0.001);
        assert!((WindDirection::East.degrees() - 90.0).abs() < 0.001);
        assert!((WindDirection::South.degrees() - 180.0).abs() < 0.001);
        assert!((WindDirection::West.degrees() - 270.0).abs() < 0.001);
    }

    #[test]
    fn test_current_weather_new() {
        let t = Temperature::celsius(22.0);
        let w = CurrentWeather::new("London", t, WeatherCondition::PartlyCloudy);
        assert_eq!(w.location, "London");
        assert_eq!(w.humidity, 50);
        assert!(!w.is_severe());
    }

    #[test]
    fn test_current_weather_severe() {
        let t = Temperature::celsius(15.0);
        let mut w = CurrentWeather::new("Test", t, WeatherCondition::Tornado);
        assert!(w.is_severe());
        w.condition = WeatherCondition::Cloudy;
        w.wind_speed = 90.0;
        assert!(w.is_severe());
        w.wind_speed = 50.0;
        assert!(!w.is_severe());
    }

    #[test]
    fn test_forecast_day_new() {
        let high = Temperature::celsius(28.0);
        let low = Temperature::celsius(15.0);
        let day = ForecastDay::new("Mon", high, low, WeatherCondition::Sunny);
        assert_eq!(day.label, "Mon");
        assert_eq!(day.precip_probability, 0);
    }

    #[test]
    fn test_forecast_day_temp_range() {
        let high = Temperature::celsius(30.0);
        let low = Temperature::celsius(20.0);
        let day = ForecastDay::new("Tue", high, low, WeatherCondition::Cloudy);
        let range = day.temp_range();
        assert!(range.contains("°C"));
        assert!(range.contains('/'));
    }

    #[test]
    fn test_weather_widget_new() {
        let w = WeatherWidget::new();
        assert!(!w.has_data());
        assert_eq!(w.forecast_count(), 0);
        assert_eq!(w.current_frame(), 0);
    }

    #[test]
    fn test_weather_widget_set_current() {
        let mut w = WeatherWidget::new();
        let t = Temperature::celsius(20.0);
        let weather = CurrentWeather::new("Paris", t, WeatherCondition::Sunny);
        w.set_current(weather);
        assert!(w.has_data());
        assert!(!w.show_severe_alert());
    }

    #[test]
    fn test_weather_widget_forecast_limit() {
        let mut w = WeatherWidget::new();
        let high = Temperature::celsius(25.0);
        let low = Temperature::celsius(15.0);
        for i in 0..8 {
            let day = ForecastDay::new(format!("Day{i}"), high, low, WeatherCondition::Sunny);
            let added = w.add_forecast_day(day);
            if i < 7 {
                assert!(added, "Day {i} should be added");
            } else {
                assert!(!added, "Day {i} should NOT be added (limit reached)");
            }
        }
        assert_eq!(w.forecast_count(), 7);
    }

    #[test]
    fn test_weather_widget_tick() {
        let mut w = WeatherWidget::new();
        w.tick();
        w.tick();
        w.tick();
        assert_eq!(w.current_frame(), 3);
    }

    #[test]
    fn test_weather_widget_clear_forecast() {
        let mut w = WeatherWidget::new();
        let high = Temperature::celsius(25.0);
        let low = Temperature::celsius(15.0);
        w.add_forecast_day(ForecastDay::new("Mon", high, low, WeatherCondition::Sunny));
        assert_eq!(w.forecast_count(), 1);
        w.clear_forecast();
        assert_eq!(w.forecast_count(), 0);
    }

    #[test]
    fn test_weather_widget_dimensions() {
        let w = WeatherWidget::new();
        let (width, height) = w.dimensions();
        assert_eq!(width, 400);
        assert_eq!(height, 200);
    }

    #[test]
    fn test_weather_icon_animation() {
        let mut anim = WeatherIconAnimation::new(WeatherCondition::Sunny);
        assert!((anim.phase - 0.0).abs() < 0.001);
        anim.update(1.0);
        assert!(anim.phase > 0.0);
        let rot = anim.sun_rotation();
        assert!(rot >= 0.0 && rot < 360.0);
    }

    #[test]
    fn test_weather_icon_animation_rain_offset() {
        let mut anim = WeatherIconAnimation::new(WeatherCondition::Rainy);
        anim.update(0.5);
        let offset = anim.rain_offset();
        assert!(offset >= 0.0 && offset <= 20.0);
    }

    #[test]
    fn test_weather_config_default() {
        let config = WeatherWidgetConfig::default();
        assert_eq!(config.layout, WeatherWidgetLayout::Standard);
        assert!(config.animated_icons);
        assert_eq!(config.temperature_unit, TemperatureUnit::Celsius);
    }
}
