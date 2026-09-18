//! Sports scoreboard graphics for broadcast overlays.
//!
//! Provides configurable scoreboard renderers for various sports,
//! including a game clock, team scores, and period/quarter tracking.

/// Supported sport types for scoreboard formatting.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SportType {
    /// Association football / soccer.
    Soccer,
    /// American or Canadian football.
    Football,
    /// Basketball.
    Basketball,
    /// Tennis.
    Tennis,
    /// Ice hockey.
    Hockey,
    /// Generic / custom sport.
    Generic,
}

impl SportType {
    /// Score display format string for this sport type.
    #[allow(dead_code)]
    pub fn score_format(&self) -> &str {
        match self {
            Self::Soccer => "SCORE",
            Self::Football => "SCORE",
            Self::Basketball => "SCORE",
            Self::Tennis => "SETS",
            Self::Hockey => "SCORE",
            Self::Generic => "SCORE",
        }
    }

    /// Returns true if the clock counts up (e.g. soccer).
    #[allow(dead_code)]
    pub fn clock_counts_up(&self) -> bool {
        matches!(self, Self::Soccer)
    }

    /// Returns the typical number of periods.
    #[allow(dead_code)]
    pub fn periods(&self) -> u32 {
        match self {
            Self::Soccer => 2,
            Self::Football => 4,
            Self::Basketball => 4,
            Self::Tennis => 3,
            Self::Hockey => 3,
            Self::Generic => 2,
        }
    }
}

/// Score and identity for a single team.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TeamScore {
    /// Team name.
    pub name: String,
    /// Current score.
    pub score: u32,
    /// Team color as RGBA.
    pub color: [u8; 4],
}

impl TeamScore {
    /// Create a new team score entry.
    #[allow(dead_code)]
    pub fn new(name: impl Into<String>, score: u32, color: [u8; 4]) -> Self {
        Self {
            name: name.into(),
            score,
            color,
        }
    }

    /// Increment score by one.
    #[allow(dead_code)]
    pub fn add_point(&mut self) {
        self.score += 1;
    }
}

/// Game clock with period tracking.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GameClock {
    /// Minutes component.
    pub minutes: u32,
    /// Seconds component.
    pub seconds: u32,
    /// Whether the clock is currently running.
    pub is_running: bool,
    /// Current period / quarter / half.
    pub period: u32,
}

impl GameClock {
    /// Create a new game clock.
    #[allow(dead_code)]
    pub fn new(minutes: u32, seconds: u32, period: u32) -> Self {
        Self {
            minutes,
            seconds,
            is_running: false,
            period,
        }
    }

    /// Start the clock.
    #[allow(dead_code)]
    pub fn start(&mut self) {
        self.is_running = true;
    }

    /// Stop the clock.
    #[allow(dead_code)]
    pub fn stop(&mut self) {
        self.is_running = false;
    }

    /// Tick the clock down by one second.
    ///
    /// Returns `true` if the period has ended (clock reached 00:00).
    #[allow(dead_code)]
    pub fn tick(&mut self) -> bool {
        if !self.is_running {
            return false;
        }

        if self.seconds > 0 {
            self.seconds -= 1;
            false
        } else if self.minutes > 0 {
            self.minutes -= 1;
            self.seconds = 59;
            false
        } else {
            // Period ended
            self.is_running = false;
            true
        }
    }

    /// Format the clock as "MM:SS".
    #[allow(dead_code)]
    pub fn format(&self) -> String {
        format!("{:02}:{:02}", self.minutes, self.seconds)
    }

    /// Total seconds remaining.
    #[allow(dead_code)]
    pub fn total_seconds(&self) -> u32 {
        self.minutes * 60 + self.seconds
    }
}

impl Default for GameClock {
    fn default() -> Self {
        Self::new(15, 0, 1)
    }
}

/// Complete scoreboard configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScoreboardConfig {
    /// Sport type.
    pub sport: SportType,
    /// Home team score data.
    pub home: TeamScore,
    /// Away team score data.
    pub away: TeamScore,
    /// Game clock.
    pub clock: GameClock,
    /// Whether to display the period number.
    pub show_period: bool,
}

impl ScoreboardConfig {
    /// Create a new scoreboard configuration.
    #[allow(dead_code)]
    pub fn new(
        sport: SportType,
        home: TeamScore,
        away: TeamScore,
        clock: GameClock,
        show_period: bool,
    ) -> Self {
        Self {
            sport,
            home,
            away,
            clock,
            show_period,
        }
    }
}

/// Renderer for the scoreboard overlay.
pub struct ScoreboardRenderer;

impl ScoreboardRenderer {
    /// Render the scoreboard as an RGBA overlay strip at the top of the frame.
    ///
    /// Returns a `Vec<u8>` of RGBA data with length `width * bar_height * 4`.
    /// The bar height is fixed at 10% of common 1080p height (~108px) or
    /// parameterised as `width / 18` for a proportional strip.
    #[allow(dead_code)]
    pub fn render(config: &ScoreboardConfig, width: u32, height: u32) -> Vec<u8> {
        let bar_height = (height as f32 * 0.08).max(40.0) as u32;
        let total_pixels = (width * bar_height) as usize;
        let mut data = vec![0u8; total_pixels * 4];

        // Fill background (dark translucent bar)
        let bg = [20u8, 20, 20, 230];
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = bg[0];
            chunk[1] = bg[1];
            chunk[2] = bg[2];
            chunk[3] = bg[3];
        }

        // Draw home team color stripe on left third
        let third = (width / 3) as usize;
        for row in 0..(bar_height as usize) {
            for col in 0..third {
                let stripe_progress = col as f32 / third as f32;
                if stripe_progress < 0.3 {
                    let idx = (row * width as usize + col) * 4;
                    if idx + 3 < data.len() {
                        data[idx] = config.home.color[0];
                        data[idx + 1] = config.home.color[1];
                        data[idx + 2] = config.home.color[2];
                        data[idx + 3] = 180;
                    }
                }
            }
        }

        // Draw away team color stripe on right third
        let two_thirds = (width * 2 / 3) as usize;
        for row in 0..(bar_height as usize) {
            for col in two_thirds..(width as usize) {
                let stripe_progress = (col - two_thirds) as f32 / third as f32;
                if stripe_progress > 0.7 {
                    let idx = (row * width as usize + col) * 4;
                    if idx + 3 < data.len() {
                        data[idx] = config.away.color[0];
                        data[idx + 1] = config.away.color[1];
                        data[idx + 2] = config.away.color[2];
                        data[idx + 3] = 180;
                    }
                }
            }
        }

        data
    }
}

/// Update events that can be applied to a scoreboard.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ScoreboardUpdate {
    /// Update home team score.
    HomeScore(u32),
    /// Update away team score.
    AwayScore(u32),
    /// Replace clock entirely.
    ClockUpdate(GameClock),
    /// Change the current period.
    PeriodChange(u32),
}

/// A live scoreboard that can receive updates.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Scoreboard {
    /// Current configuration.
    pub config: ScoreboardConfig,
}

impl Scoreboard {
    /// Create a new scoreboard.
    #[allow(dead_code)]
    pub fn new(config: ScoreboardConfig) -> Self {
        Self { config }
    }

    /// Apply an update to the scoreboard state.
    #[allow(dead_code)]
    pub fn apply_update(&mut self, update: ScoreboardUpdate) {
        match update {
            ScoreboardUpdate::HomeScore(s) => {
                self.config.home.score = s;
            }
            ScoreboardUpdate::AwayScore(s) => {
                self.config.away.score = s;
            }
            ScoreboardUpdate::ClockUpdate(clock) => {
                self.config.clock = clock;
            }
            ScoreboardUpdate::PeriodChange(p) => {
                self.config.clock.period = p;
            }
        }
    }

    /// Tick the game clock.
    ///
    /// Returns `true` if the period ended.
    #[allow(dead_code)]
    pub fn tick_clock(&mut self) -> bool {
        self.config.clock.tick()
    }
}

// ---------------------------------------------------------------------------
// Real-time data binding — live score update via DataSource trait
// ---------------------------------------------------------------------------

/// A snapshot of live game state emitted by a data source.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LiveGameState {
    /// Home team score.
    pub home_score: u32,
    /// Away team score.
    pub away_score: u32,
    /// Clock minutes remaining.
    pub clock_minutes: u32,
    /// Clock seconds remaining.
    pub clock_seconds: u32,
    /// Whether the clock is running.
    pub clock_running: bool,
    /// Current period.
    pub period: u32,
}

impl LiveGameState {
    /// Create a new live game state.
    #[allow(dead_code)]
    pub fn new(
        home_score: u32,
        away_score: u32,
        clock_minutes: u32,
        clock_seconds: u32,
        clock_running: bool,
        period: u32,
    ) -> Self {
        Self {
            home_score,
            away_score,
            clock_minutes,
            clock_seconds,
            clock_running,
            period,
        }
    }
}

/// Trait for a real-time score data source.
///
/// Implement this trait to connect live scoring data to a [`LiveScoreboard`].
/// The `poll` method is called each frame to check for updates.
#[allow(dead_code)]
pub trait ScoreDataSource: Send + Sync {
    /// Poll for the latest game state.
    ///
    /// Returns `Some(state)` when new data is available, `None` if no change.
    fn poll(&mut self) -> Option<LiveGameState>;

    /// Returns `true` if the data source is connected and healthy.
    fn is_connected(&self) -> bool;
}

/// A static in-memory data source for testing and development.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StaticDataSource {
    state: Option<LiveGameState>,
    connected: bool,
}

impl StaticDataSource {
    /// Create a connected static source with the given initial state.
    #[allow(dead_code)]
    pub fn new(state: LiveGameState) -> Self {
        Self {
            state: Some(state),
            connected: true,
        }
    }

    /// Create a disconnected static source (simulates a lost feed).
    #[allow(dead_code)]
    pub fn disconnected() -> Self {
        Self {
            state: None,
            connected: false,
        }
    }

    /// Push a new state to be emitted on the next `poll`.
    #[allow(dead_code)]
    pub fn push(&mut self, state: LiveGameState) {
        self.state = Some(state);
    }
}

impl ScoreDataSource for StaticDataSource {
    fn poll(&mut self) -> Option<LiveGameState> {
        self.state.take()
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

/// A scoreboard with live data binding to a [`ScoreDataSource`].
pub struct LiveScoreboard {
    /// Current scoreboard state.
    pub scoreboard: Scoreboard,
    /// Number of updates received so far.
    pub update_count: u64,
}

impl LiveScoreboard {
    /// Create a new live scoreboard wrapping the given configuration.
    #[allow(dead_code)]
    pub fn new(config: ScoreboardConfig) -> Self {
        Self {
            scoreboard: Scoreboard::new(config),
            update_count: 0,
        }
    }

    /// Poll the data source and apply any available state update.
    ///
    /// Returns `true` if the scoreboard was updated this call.
    #[allow(dead_code)]
    pub fn sync(&mut self, source: &mut dyn ScoreDataSource) -> bool {
        if !source.is_connected() {
            return false;
        }
        if let Some(state) = source.poll() {
            self.apply_live_state(&state);
            self.update_count += 1;
            true
        } else {
            false
        }
    }

    /// Apply a raw `LiveGameState` snapshot to the internal scoreboard.
    #[allow(dead_code)]
    pub fn apply_live_state(&mut self, state: &LiveGameState) {
        self.scoreboard
            .apply_update(ScoreboardUpdate::HomeScore(state.home_score));
        self.scoreboard
            .apply_update(ScoreboardUpdate::AwayScore(state.away_score));
        self.scoreboard
            .apply_update(ScoreboardUpdate::PeriodChange(state.period));
        let clock = GameClock {
            minutes: state.clock_minutes,
            seconds: state.clock_seconds,
            is_running: state.clock_running,
            period: state.period,
        };
        self.scoreboard
            .apply_update(ScoreboardUpdate::ClockUpdate(clock));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> ScoreboardConfig {
        ScoreboardConfig::new(
            SportType::Basketball,
            TeamScore::new("HOME", 0, [200, 0, 0, 255]),
            TeamScore::new("AWAY", 0, [0, 0, 200, 255]),
            GameClock::new(12, 0, 1),
            true,
        )
    }

    #[test]
    fn test_sport_type_score_format() {
        assert_eq!(SportType::Soccer.score_format(), "SCORE");
        assert_eq!(SportType::Tennis.score_format(), "SETS");
    }

    #[test]
    fn test_sport_type_periods() {
        assert_eq!(SportType::Football.periods(), 4);
        assert_eq!(SportType::Soccer.periods(), 2);
        assert_eq!(SportType::Hockey.periods(), 3);
    }

    #[test]
    fn test_team_score_add_point() {
        let mut team = TeamScore::new("Team A", 0, [255, 0, 0, 255]);
        team.add_point();
        team.add_point();
        assert_eq!(team.score, 2);
    }

    #[test]
    fn test_game_clock_format() {
        let clock = GameClock::new(12, 34, 1);
        assert_eq!(clock.format(), "12:34");
    }

    #[test]
    fn test_game_clock_tick_counts_down() {
        let mut clock = GameClock::new(0, 3, 1);
        clock.start();
        assert!(!clock.tick()); // 0:02
        assert!(!clock.tick()); // 0:01
        assert!(!clock.tick()); // 0:00
        let period_ended = clock.tick(); // period ends
        assert!(period_ended);
    }

    #[test]
    fn test_game_clock_tick_minutes_rollover() {
        let mut clock = GameClock::new(1, 0, 1);
        clock.start();
        let ended = clock.tick();
        assert!(!ended);
        assert_eq!(clock.minutes, 0);
        assert_eq!(clock.seconds, 59);
    }

    #[test]
    fn test_game_clock_not_running_does_not_tick() {
        let mut clock = GameClock::new(5, 0, 1);
        // is_running defaults to false
        let ended = clock.tick();
        assert!(!ended);
        assert_eq!(clock.minutes, 5); // unchanged
    }

    #[test]
    fn test_game_clock_total_seconds() {
        let clock = GameClock::new(2, 30, 1);
        assert_eq!(clock.total_seconds(), 150);
    }

    #[test]
    fn test_scoreboard_render_size() {
        let config = make_config();
        let data = ScoreboardRenderer::render(&config, 1920, 1080);
        let bar_height = (1080_f32 * 0.08).max(40.0) as u32;
        assert_eq!(data.len(), (1920 * bar_height * 4) as usize);
    }

    #[test]
    fn test_scoreboard_render_non_empty() {
        let config = make_config();
        let data = ScoreboardRenderer::render(&config, 320, 240);
        let has_non_zero = data.iter().any(|&b| b > 0);
        assert!(has_non_zero);
    }

    #[test]
    fn test_scoreboard_apply_update_home_score() {
        let mut sb = Scoreboard::new(make_config());
        sb.apply_update(ScoreboardUpdate::HomeScore(3));
        assert_eq!(sb.config.home.score, 3);
    }

    #[test]
    fn test_scoreboard_apply_update_away_score() {
        let mut sb = Scoreboard::new(make_config());
        sb.apply_update(ScoreboardUpdate::AwayScore(5));
        assert_eq!(sb.config.away.score, 5);
    }

    #[test]
    fn test_scoreboard_apply_update_period() {
        let mut sb = Scoreboard::new(make_config());
        sb.apply_update(ScoreboardUpdate::PeriodChange(2));
        assert_eq!(sb.config.clock.period, 2);
    }

    #[test]
    fn test_scoreboard_apply_update_clock() {
        let mut sb = Scoreboard::new(make_config());
        sb.apply_update(ScoreboardUpdate::ClockUpdate(GameClock::new(5, 30, 2)));
        assert_eq!(sb.config.clock.minutes, 5);
        assert_eq!(sb.config.clock.seconds, 30);
    }

    // --- LiveScoreboard / data binding tests ---

    fn make_live_state() -> LiveGameState {
        LiveGameState::new(3, 1, 10, 45, true, 2)
    }

    #[test]
    fn test_static_data_source_poll_returns_state() {
        let mut src = StaticDataSource::new(make_live_state());
        assert!(src.is_connected());
        let state = src.poll();
        assert!(state.is_some());
        let s = state.expect("state should be present");
        assert_eq!(s.home_score, 3);
        assert_eq!(s.away_score, 1);
    }

    #[test]
    fn test_static_data_source_poll_returns_none_after_first() {
        let mut src = StaticDataSource::new(make_live_state());
        let _ = src.poll(); // consume
        assert!(src.poll().is_none());
    }

    #[test]
    fn test_static_data_source_disconnected() {
        let src = StaticDataSource::disconnected();
        assert!(!src.is_connected());
    }

    #[test]
    fn test_live_scoreboard_sync_updates_state() {
        let mut lb = LiveScoreboard::new(make_config());
        let mut src = StaticDataSource::new(make_live_state());
        let updated = lb.sync(&mut src);
        assert!(updated);
        assert_eq!(lb.scoreboard.config.home.score, 3);
        assert_eq!(lb.scoreboard.config.away.score, 1);
        assert_eq!(lb.scoreboard.config.clock.period, 2);
        assert_eq!(lb.update_count, 1);
    }

    #[test]
    fn test_live_scoreboard_sync_no_update_when_no_data() {
        let mut lb = LiveScoreboard::new(make_config());
        let mut src = StaticDataSource::new(make_live_state());
        lb.sync(&mut src); // consume first
        let updated = lb.sync(&mut src); // no data left
        assert!(!updated);
        assert_eq!(lb.update_count, 1);
    }

    #[test]
    fn test_live_scoreboard_sync_disconnected_source() {
        let mut lb = LiveScoreboard::new(make_config());
        let mut src = StaticDataSource::disconnected();
        let updated = lb.sync(&mut src);
        assert!(!updated);
    }

    #[test]
    fn test_live_game_state_new() {
        let s = LiveGameState::new(5, 2, 8, 30, false, 3);
        assert_eq!(s.home_score, 5);
        assert_eq!(s.away_score, 2);
        assert_eq!(s.clock_minutes, 8);
        assert!(!s.clock_running);
    }

    #[test]
    fn test_static_data_source_push() {
        let mut src = StaticDataSource::new(make_live_state());
        let _ = src.poll(); // drain
        src.push(LiveGameState::new(10, 7, 5, 0, true, 4));
        let state = src.poll().expect("pushed state should be present");
        assert_eq!(state.home_score, 10);
    }
}
