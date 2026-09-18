#![allow(dead_code)]
//! Game metadata and registry for `OxiMedia` gaming crate.
//!
//! Provides structured metadata about games, including genre classification
//! and a lightweight in-memory registry for lookup.

/// Primary genre classification for a game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Genre {
    /// First-person or third-person shooter
    Shooter,
    /// Real-time strategy
    Strategy,
    /// Multiplayer online battle arena
    Moba,
    /// Battle royale
    BattleRoyale,
    /// Sports simulation
    Sports,
    /// Racing
    Racing,
    /// Role-playing game
    Rpg,
    /// Fighting game
    Fighting,
    /// Puzzle
    Puzzle,
    /// Sandbox / open world
    Sandbox,
}

impl Genre {
    /// Short human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Genre::Shooter => "shooter",
            Genre::Strategy => "strategy",
            Genre::Moba => "moba",
            Genre::BattleRoyale => "battle_royale",
            Genre::Sports => "sports",
            Genre::Racing => "racing",
            Genre::Rpg => "rpg",
            Genre::Fighting => "fighting",
            Genre::Puzzle => "puzzle",
            Genre::Sandbox => "sandbox",
        }
    }

    /// Return `true` for genres that are commonly played competitively (esports).
    #[must_use]
    pub fn is_competitive(self) -> bool {
        matches!(
            self,
            Genre::Shooter
                | Genre::Moba
                | Genre::BattleRoyale
                | Genre::Fighting
                | Genre::Strategy
                | Genre::Sports
        )
    }
}

/// Metadata describing a game title.
#[derive(Debug, Clone)]
pub struct GameMetadata {
    /// Canonical title of the game.
    pub title: String,
    /// Primary genre.
    pub genre: Genre,
    /// Developer / studio name.
    pub developer: String,
    /// Release year (four-digit).
    pub release_year: u16,
    /// Maximum number of concurrent players (0 = single-player only).
    pub max_players: u32,
    /// Whether the game has an active ranked / tournament scene.
    pub has_ranked_mode: bool,
    /// Optional Twitch / `YouTube` category name for the game.
    pub stream_category: Option<String>,
}

impl GameMetadata {
    /// Create a new `GameMetadata`.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        genre: Genre,
        developer: impl Into<String>,
        release_year: u16,
        max_players: u32,
    ) -> Self {
        Self {
            title: title.into(),
            genre,
            developer: developer.into(),
            release_year,
            max_players,
            has_ranked_mode: false,
            stream_category: None,
        }
    }

    /// Enable ranked mode flag.
    #[must_use]
    pub fn with_ranked(mut self) -> Self {
        self.has_ranked_mode = true;
        self
    }

    /// Set the stream category label.
    #[must_use]
    pub fn with_stream_category(mut self, cat: impl Into<String>) -> Self {
        self.stream_category = Some(cat.into());
        self
    }

    /// Return `true` if the game qualifies as an esports title.
    ///
    /// Criteria: competitive genre AND has a ranked mode AND at least 2 players.
    #[must_use]
    pub fn is_esports(&self) -> bool {
        self.genre.is_competitive() && self.has_ranked_mode && self.max_players >= 2
    }
}

/// In-memory registry of known game titles.
#[derive(Debug, Default)]
pub struct GameRegistry {
    entries: Vec<GameMetadata>,
}

impl GameRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new game.
    pub fn register(&mut self, metadata: GameMetadata) {
        self.entries.push(metadata);
    }

    /// Total number of registered games.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Find all games with the given genre.
    #[must_use]
    pub fn find_by_genre(&self, genre: Genre) -> Vec<&GameMetadata> {
        self.entries.iter().filter(|g| g.genre == genre).collect()
    }

    /// Find a game by exact title (case-insensitive).
    #[must_use]
    pub fn find_by_title(&self, title: &str) -> Option<&GameMetadata> {
        let lower = title.to_lowercase();
        self.entries
            .iter()
            .find(|g| g.title.to_lowercase() == lower)
    }

    /// All esports-eligible titles in the registry.
    #[must_use]
    pub fn esports_titles(&self) -> Vec<&GameMetadata> {
        self.entries.iter().filter(|g| g.is_esports()).collect()
    }

    /// Return all registered games.
    #[must_use]
    pub fn all(&self) -> &[GameMetadata] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fps() -> GameMetadata {
        GameMetadata::new("ShooterGame", Genre::Shooter, "DevA", 2020, 64)
            .with_ranked()
            .with_stream_category("ShooterGame")
    }

    fn sample_puzzle() -> GameMetadata {
        GameMetadata::new("PuzzleGame", Genre::Puzzle, "DevB", 2018, 1)
    }

    fn sample_moba() -> GameMetadata {
        GameMetadata::new("MobaGame", Genre::Moba, "DevC", 2015, 10).with_ranked()
    }

    #[test]
    fn test_genre_labels() {
        assert_eq!(Genre::Shooter.label(), "shooter");
        assert_eq!(Genre::Moba.label(), "moba");
        assert_eq!(Genre::BattleRoyale.label(), "battle_royale");
        assert_eq!(Genre::Rpg.label(), "rpg");
        assert_eq!(Genre::Fighting.label(), "fighting");
    }

    #[test]
    fn test_genre_competitive_classification() {
        assert!(Genre::Shooter.is_competitive());
        assert!(Genre::Moba.is_competitive());
        assert!(Genre::BattleRoyale.is_competitive());
        assert!(Genre::Fighting.is_competitive());
        assert!(Genre::Strategy.is_competitive());
        assert!(Genre::Sports.is_competitive());

        assert!(!Genre::Puzzle.is_competitive());
        assert!(!Genre::Rpg.is_competitive());
        assert!(!Genre::Sandbox.is_competitive());
        assert!(!Genre::Racing.is_competitive());
    }

    #[test]
    fn test_game_metadata_new() {
        let g = sample_fps();
        assert_eq!(g.title, "ShooterGame");
        assert_eq!(g.genre, Genre::Shooter);
        assert_eq!(g.developer, "DevA");
        assert_eq!(g.release_year, 2020);
        assert_eq!(g.max_players, 64);
        assert!(g.has_ranked_mode);
    }

    #[test]
    fn test_is_esports_true() {
        let g = sample_fps();
        assert!(g.is_esports());
    }

    #[test]
    fn test_is_esports_false_no_ranked() {
        let g = GameMetadata::new("CasualShooter", Genre::Shooter, "DevX", 2022, 10);
        assert!(!g.is_esports());
    }

    #[test]
    fn test_is_esports_false_single_player() {
        // Puzzle: non-competitive genre
        let g = sample_puzzle();
        assert!(!g.is_esports());
    }

    #[test]
    fn test_is_esports_false_non_competitive_genre() {
        let g = GameMetadata::new("AdventureRpg", Genre::Rpg, "DevY", 2019, 4).with_ranked();
        assert!(!g.is_esports());
    }

    #[test]
    fn test_stream_category() {
        let g = sample_fps();
        assert_eq!(g.stream_category.as_deref(), Some("ShooterGame"));
    }

    #[test]
    fn test_registry_register_and_count() {
        let mut reg = GameRegistry::new();
        assert_eq!(reg.count(), 0);
        reg.register(sample_fps());
        reg.register(sample_puzzle());
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_registry_find_by_genre() {
        let mut reg = GameRegistry::new();
        reg.register(sample_fps());
        reg.register(sample_moba());
        reg.register(sample_puzzle());

        let shooters = reg.find_by_genre(Genre::Shooter);
        assert_eq!(shooters.len(), 1);
        assert_eq!(shooters[0].title, "ShooterGame");

        let mobas = reg.find_by_genre(Genre::Moba);
        assert_eq!(mobas.len(), 1);

        let rpgs = reg.find_by_genre(Genre::Rpg);
        assert!(rpgs.is_empty());
    }

    #[test]
    fn test_registry_find_by_title_case_insensitive() {
        let mut reg = GameRegistry::new();
        reg.register(sample_fps());

        assert!(reg.find_by_title("shootergame").is_some());
        assert!(reg.find_by_title("SHOOTERGAME").is_some());
        assert!(reg.find_by_title("UnknownGame").is_none());
    }

    #[test]
    fn test_registry_esports_titles() {
        let mut reg = GameRegistry::new();
        reg.register(sample_fps()); // esports
        reg.register(sample_moba()); // esports
        reg.register(sample_puzzle()); // not esports

        let esports = reg.esports_titles();
        assert_eq!(esports.len(), 2);
    }

    #[test]
    fn test_registry_all() {
        let mut reg = GameRegistry::new();
        reg.register(sample_fps());
        reg.register(sample_puzzle());
        assert_eq!(reg.all().len(), 2);
    }
}
