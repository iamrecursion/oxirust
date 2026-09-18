//! Gaming achievement system for `oximedia-gaming`.
//!
//! Defines achievement tiers, individual achievements, player progress,
//! and a catalog for looking up and summarising achievements.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

// ---------------------------------------------------------------------------
// AchievementTier
// ---------------------------------------------------------------------------

/// Rarity / reward tier of a gaming achievement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AchievementTier {
    /// Entry-level achievement.
    Bronze,
    /// Mid-range achievement.
    Silver,
    /// High-value achievement.
    Gold,
    /// Near-maximum prestige.
    Platinum,
    /// Top-tier, rarest achievement.
    Diamond,
}

impl AchievementTier {
    /// Point value awarded when this tier is unlocked.
    pub fn points(self) -> u32 {
        match self {
            Self::Bronze => 10,
            Self::Silver => 25,
            Self::Gold => 50,
            Self::Platinum => 100,
            Self::Diamond => 250,
        }
    }
}

// ---------------------------------------------------------------------------
// Achievement
// ---------------------------------------------------------------------------

/// A single achievement definition.
#[derive(Debug, Clone)]
pub struct Achievement {
    /// Unique identifier (e.g. `"first_kill"`).
    pub id: String,
    /// Display name shown to the player.
    pub name: String,
    /// Human-readable description of how to unlock this achievement.
    pub description: String,
    /// Tier / reward level.
    pub tier: AchievementTier,
    /// When `true` the name and description are hidden until unlocked.
    pub secret: bool,
}

impl Achievement {
    /// Create a new achievement.
    pub fn new(
        id: &str,
        name: &str,
        description: &str,
        tier: AchievementTier,
        secret: bool,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            tier,
            secret,
        }
    }

    /// Points awarded when this achievement is unlocked.
    pub fn display_points(&self) -> u32 {
        self.tier.points()
    }
}

// ---------------------------------------------------------------------------
// PlayerAchievements
// ---------------------------------------------------------------------------

/// Tracks which achievements a specific player has unlocked and when.
///
/// Each entry in `unlocked` is `(achievement_id, unix_timestamp_ms)`.
#[derive(Debug, Clone)]
pub struct PlayerAchievements {
    /// Player identifier.
    pub player_id: String,
    /// Unlocked achievements together with the unlock timestamp in milliseconds.
    pub unlocked: Vec<(String, u64)>,
}

impl PlayerAchievements {
    /// Create a new, empty progress tracker for `player_id`.
    pub fn new(player_id: &str) -> Self {
        Self {
            player_id: player_id.to_string(),
            unlocked: Vec::new(),
        }
    }

    /// Attempt to unlock an achievement.
    ///
    /// Returns `true` when the achievement was newly unlocked, `false` when it
    /// had already been unlocked previously.
    pub fn unlock(&mut self, id: &str, now_ms: u64) -> bool {
        if self.is_unlocked(id) {
            return false;
        }
        self.unlocked.push((id.to_string(), now_ms));
        true
    }

    /// Returns `true` when the player has already unlocked the achievement with
    /// the given `id`.
    pub fn is_unlocked(&self, id: &str) -> bool {
        self.unlocked.iter().any(|(uid, _)| uid == id)
    }

    /// Sum of points for all unlocked achievements, looked up from `catalog`.
    pub fn total_points(&self, catalog: &[Achievement]) -> u32 {
        self.unlocked
            .iter()
            .filter_map(|(uid, _)| catalog.iter().find(|a| &a.id == uid))
            .map(Achievement::display_points)
            .sum()
    }

    /// Number of achievements the player has unlocked.
    pub fn unlock_count(&self) -> usize {
        self.unlocked.len()
    }
}

// ---------------------------------------------------------------------------
// AchievementCatalog
// ---------------------------------------------------------------------------

/// A collection of achievement definitions that can be queried.
#[derive(Debug, Clone, Default)]
pub struct AchievementCatalog {
    /// All achievements in the catalog.
    pub achievements: Vec<Achievement>,
}

impl AchievementCatalog {
    /// Create a catalog pre-populated with `achievements`.
    pub fn new(achievements: Vec<Achievement>) -> Self {
        Self { achievements }
    }

    /// Find an achievement by its unique `id`.
    pub fn find(&self, id: &str) -> Option<&Achievement> {
        self.achievements.iter().find(|a| a.id == id)
    }

    /// All achievements belonging to the given `tier`.
    pub fn by_tier(&self, tier: &AchievementTier) -> Vec<&Achievement> {
        self.achievements
            .iter()
            .filter(|a| &a.tier == tier)
            .collect()
    }

    /// Sum of points across every achievement in the catalog.
    pub fn total_possible_points(&self) -> u32 {
        self.achievements
            .iter()
            .map(Achievement::display_points)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_catalog() -> AchievementCatalog {
        AchievementCatalog::new(vec![
            Achievement::new(
                "first_kill",
                "First Blood",
                "Get your first kill",
                AchievementTier::Bronze,
                false,
            ),
            Achievement::new(
                "ten_kills",
                "On a Roll",
                "Get 10 kills",
                AchievementTier::Silver,
                false,
            ),
            Achievement::new(
                "hundred_kills",
                "Sharpshooter",
                "Get 100 kills",
                AchievementTier::Gold,
                false,
            ),
            Achievement::new(
                "secret_boss",
                "???",
                "Defeat the hidden boss",
                AchievementTier::Diamond,
                true,
            ),
            Achievement::new(
                "pacifist",
                "Pacifist",
                "Win without killing",
                AchievementTier::Platinum,
                false,
            ),
        ])
    }

    #[test]
    fn test_tier_points_bronze() {
        assert_eq!(AchievementTier::Bronze.points(), 10);
    }

    #[test]
    fn test_tier_points_silver() {
        assert_eq!(AchievementTier::Silver.points(), 25);
    }

    #[test]
    fn test_tier_points_gold() {
        assert_eq!(AchievementTier::Gold.points(), 50);
    }

    #[test]
    fn test_tier_points_platinum() {
        assert_eq!(AchievementTier::Platinum.points(), 100);
    }

    #[test]
    fn test_tier_points_diamond() {
        assert_eq!(AchievementTier::Diamond.points(), 250);
    }

    #[test]
    fn test_achievement_display_points() {
        let a = Achievement::new("x", "X", "desc", AchievementTier::Gold, false);
        assert_eq!(a.display_points(), 50);
    }

    #[test]
    fn test_achievement_secret_flag() {
        let a = Achievement::new("hidden", "???", "secret", AchievementTier::Diamond, true);
        assert!(a.secret);
    }

    #[test]
    fn test_player_unlock_new() {
        let mut p = PlayerAchievements::new("player1");
        assert!(p.unlock("first_kill", 1_000));
        assert_eq!(p.unlock_count(), 1);
    }

    #[test]
    fn test_player_unlock_duplicate() {
        let mut p = PlayerAchievements::new("player1");
        p.unlock("first_kill", 1_000);
        let second = p.unlock("first_kill", 2_000);
        assert!(!second);
        assert_eq!(p.unlock_count(), 1);
    }

    #[test]
    fn test_player_is_unlocked_true() {
        let mut p = PlayerAchievements::new("p");
        p.unlock("x", 100);
        assert!(p.is_unlocked("x"));
    }

    #[test]
    fn test_player_is_unlocked_false() {
        let p = PlayerAchievements::new("p");
        assert!(!p.is_unlocked("x"));
    }

    #[test]
    fn test_player_total_points() {
        let cat = sample_catalog();
        let mut p = PlayerAchievements::new("p");
        p.unlock("first_kill", 100); // 10 pts
        p.unlock("ten_kills", 200); // 25 pts
        assert_eq!(p.total_points(&cat.achievements), 35);
    }

    #[test]
    fn test_catalog_find_existing() {
        let cat = sample_catalog();
        assert!(cat.find("ten_kills").is_some());
    }

    #[test]
    fn test_catalog_find_missing() {
        let cat = sample_catalog();
        assert!(cat.find("does_not_exist").is_none());
    }

    #[test]
    fn test_catalog_by_tier() {
        let cat = sample_catalog();
        let golds = cat.by_tier(&AchievementTier::Gold);
        assert_eq!(golds.len(), 1);
        assert_eq!(golds[0].id, "hundred_kills");
    }

    #[test]
    fn test_catalog_total_possible_points() {
        let cat = sample_catalog();
        // Bronze(10) + Silver(25) + Gold(50) + Diamond(250) + Platinum(100) = 435
        assert_eq!(cat.total_possible_points(), 435);
    }
}
