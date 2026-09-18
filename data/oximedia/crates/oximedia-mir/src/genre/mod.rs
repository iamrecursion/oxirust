//! Genre classification, hierarchy, similarity, and blend detection.
//!
//! This module provides:
//!
//! - **Genre classification** from audio features (rule-based and Naive Bayes).
//! - **Genre hierarchy** with top-level, sub-genre, and micro-genre tiers.
//! - **Genre family grouping** (e.g., Rock -> {Hard Rock, Punk Rock, ...}).
//! - **Genre similarity scoring** using feature-space distance.
//! - **Genre blend detection** for songs that span multiple genres.

pub mod classify;
pub mod features;
pub mod genre_enum;

pub use classify::{GenreClassifier, StreamingGenreClassifier};
pub use features::GenreFeatures;
pub use genre_enum::{classify_genre, Genre};

use std::collections::HashMap;

// ── Genre Family Grouping ───────────────────────────────────────────────────

/// A family of related sub-genres under a single top-level genre.
#[derive(Debug, Clone)]
pub struct GenreFamily {
    /// The parent top-level genre name.
    pub parent: &'static str,
    /// Sub-genres belonging to this family.
    pub members: &'static [&'static str],
}

/// Return the genre family (sub-genre list) for a given top-level genre.
///
/// Returns `None` if the genre name is not recognised.
#[must_use]
pub fn genre_family(genre: &str) -> Option<GenreFamily> {
    let lower = genre.to_ascii_lowercase();
    match lower.as_str() {
        "rock" => Some(GenreFamily {
            parent: "Rock",
            members: &[
                "Hard Rock",
                "Punk Rock",
                "Indie Rock",
                "Alternative Rock",
                "Classic Rock",
                "Progressive Rock",
                "Grunge",
                "Post-Rock",
            ],
        }),
        "electronic" => Some(GenreFamily {
            parent: "Electronic",
            members: &[
                "House",
                "Techno",
                "Drum & Bass",
                "Trance",
                "Dubstep",
                "Ambient Electronic",
                "Synthwave",
                "IDM",
            ],
        }),
        "jazz" => Some(GenreFamily {
            parent: "Jazz",
            members: &[
                "Bebop",
                "Smooth Jazz",
                "Free Jazz",
                "Jazz Fusion",
                "Swing",
                "Cool Jazz",
                "Latin Jazz",
                "Modal Jazz",
            ],
        }),
        "classical" => Some(GenreFamily {
            parent: "Classical",
            members: &[
                "Baroque",
                "Romantic",
                "Contemporary Classical",
                "Minimalist",
                "Chamber Music",
                "Orchestral",
                "Opera",
                "Renaissance",
            ],
        }),
        "metal" => Some(GenreFamily {
            parent: "Metal",
            members: &[
                "Heavy Metal",
                "Death Metal",
                "Black Metal",
                "Doom Metal",
                "Thrash Metal",
                "Power Metal",
                "Nu-Metal",
                "Progressive Metal",
            ],
        }),
        "hip-hop" | "hiphop" => Some(GenreFamily {
            parent: "Hip-Hop",
            members: &[
                "Trap",
                "Boom Bap",
                "Lo-Fi Hip-Hop",
                "Cloud Rap",
                "Old School",
                "Conscious Hip-Hop",
                "Drill",
                "Crunk",
            ],
        }),
        "pop" => Some(GenreFamily {
            parent: "Pop",
            members: &[
                "Dance Pop",
                "Synth Pop",
                "Indie Pop",
                "K-Pop",
                "Acoustic Pop",
                "Electropop",
                "Teen Pop",
                "Art Pop",
            ],
        }),
        "country" => Some(GenreFamily {
            parent: "Country",
            members: &[
                "Traditional Country",
                "Alt-Country",
                "Country Rock",
                "Bluegrass",
                "Outlaw Country",
                "Nashville Sound",
            ],
        }),
        "folk" => Some(GenreFamily {
            parent: "Folk",
            members: &[
                "Singer-Songwriter",
                "Celtic",
                "Indie Folk",
                "Americana",
                "Neofolk",
                "Contemporary Folk",
            ],
        }),
        "rnb" | "r&b" => Some(GenreFamily {
            parent: "R&B",
            members: &[
                "Contemporary R&B",
                "Neo-Soul",
                "Quiet Storm",
                "New Jack Swing",
                "Alternative R&B",
                "Funk",
            ],
        }),
        "latin" => Some(GenreFamily {
            parent: "Latin",
            members: &[
                "Salsa",
                "Reggaeton",
                "Bossa Nova",
                "Cumbia",
                "Bachata",
                "Latin Pop",
            ],
        }),
        "ambient" => Some(GenreFamily {
            parent: "Ambient",
            members: &[
                "Dark Ambient",
                "Space Ambient",
                "Drone",
                "New Age",
                "Ambient Dub",
                "Field Recordings",
            ],
        }),
        "blues" => Some(GenreFamily {
            parent: "Blues",
            members: &[
                "Delta Blues",
                "Chicago Blues",
                "Electric Blues",
                "Blues Rock",
                "Jump Blues",
                "Acoustic Blues",
            ],
        }),
        "reggae" => Some(GenreFamily {
            parent: "Reggae",
            members: &[
                "Roots Reggae",
                "Dub",
                "Dancehall",
                "Ska",
                "Rocksteady",
                "Lovers Rock",
            ],
        }),
        "world" => Some(GenreFamily {
            parent: "World",
            members: &[
                "Afrobeat",
                "Flamenco",
                "Fado",
                "Qawwali",
                "Gamelan",
                "Celtic Traditional",
            ],
        }),
        "soundtrack" => Some(GenreFamily {
            parent: "Soundtrack",
            members: &[
                "Film Score",
                "TV Score",
                "Video Game OST",
                "Musical Theatre",
                "Trailer Music",
                "Incidental Music",
            ],
        }),
        _ => None,
    }
}

/// Return all known top-level genre names.
#[must_use]
pub fn all_top_level_genres() -> &'static [&'static str] {
    &[
        "Rock",
        "Electronic",
        "Jazz",
        "Classical",
        "Metal",
        "Hip-Hop",
        "Pop",
        "Country",
        "Folk",
        "R&B",
        "Latin",
        "Ambient",
        "Blues",
        "Reggae",
        "World",
        "Soundtrack",
    ]
}

// ── Genre Hierarchy ─────────────────────────────────────────────────────────

/// Tier within the genre hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenreTier {
    /// Broadest grouping (e.g., "Rock").
    TopLevel,
    /// Sub-genre (e.g., "Indie Rock").
    SubGenre,
    /// Micro-genre (e.g., "Math Rock").
    MicroGenre,
}

/// A node in the genre hierarchy tree.
#[derive(Debug, Clone)]
pub struct GenreNode {
    /// Name of this genre.
    pub name: String,
    /// Tier within the hierarchy.
    pub tier: GenreTier,
    /// Parent genre name (`None` for top-level).
    pub parent: Option<String>,
    /// Child genre names.
    pub children: Vec<String>,
}

/// Build a full genre hierarchy for a given top-level genre.
///
/// Returns the top-level node plus all sub-genre and micro-genre nodes.
/// Returns an empty `Vec` if the genre is not recognised.
#[must_use]
pub fn genre_hierarchy(top_level: &str) -> Vec<GenreNode> {
    let family = match genre_family(top_level) {
        Some(f) => f,
        None => return Vec::new(),
    };

    let mut nodes = Vec::new();

    // Top-level node
    let children: Vec<String> = family.members.iter().map(|s| (*s).to_string()).collect();
    nodes.push(GenreNode {
        name: family.parent.to_string(),
        tier: GenreTier::TopLevel,
        parent: None,
        children: children.clone(),
    });

    // Sub-genre nodes with two micro-genre children each
    for &sub in family.members {
        let micro_a = format!("{sub} (Experimental)");
        let micro_b = format!("{sub} (Traditional)");
        nodes.push(GenreNode {
            name: sub.to_string(),
            tier: GenreTier::SubGenre,
            parent: Some(family.parent.to_string()),
            children: vec![micro_a.clone(), micro_b.clone()],
        });
        // Micro-genre nodes (leaf)
        nodes.push(GenreNode {
            name: micro_a,
            tier: GenreTier::MicroGenre,
            parent: Some(sub.to_string()),
            children: Vec::new(),
        });
        nodes.push(GenreNode {
            name: micro_b,
            tier: GenreTier::MicroGenre,
            parent: Some(sub.to_string()),
            children: Vec::new(),
        });
    }

    nodes
}

// ── Genre Similarity Scoring ────────────────────────────────────────────────

/// Feature profile used for computing genre similarity.
///
/// Each field is normalised to approximately [0, 1].
#[derive(Debug, Clone)]
pub struct GenreProfile {
    /// Typical spectral centroid (normalised 0-1).
    pub centroid: f32,
    /// Typical zero-crossing rate.
    pub zcr: f32,
    /// Typical tempo (normalised: BPM / 200).
    pub tempo: f32,
    /// Typical energy level.
    pub energy: f32,
    /// Spectral flatness.
    pub flatness: f32,
}

/// Return a canonical feature profile for a top-level genre.
///
/// Returns `None` if the genre is not recognised.
#[must_use]
pub fn genre_profile(genre: &str) -> Option<GenreProfile> {
    let lower = genre.to_ascii_lowercase();
    match lower.as_str() {
        "rock" => Some(GenreProfile {
            centroid: 0.25,
            zcr: 0.12,
            tempo: 0.65,
            energy: 0.75,
            flatness: 0.15,
        }),
        "electronic" => Some(GenreProfile {
            centroid: 0.55,
            zcr: 0.20,
            tempo: 0.70,
            energy: 0.80,
            flatness: 0.60,
        }),
        "jazz" => Some(GenreProfile {
            centroid: 0.18,
            zcr: 0.06,
            tempo: 0.50,
            energy: 0.40,
            flatness: 0.35,
        }),
        "classical" => Some(GenreProfile {
            centroid: 0.15,
            zcr: 0.04,
            tempo: 0.40,
            energy: 0.30,
            flatness: 0.40,
        }),
        "metal" => Some(GenreProfile {
            centroid: 0.38,
            zcr: 0.25,
            tempo: 0.85,
            energy: 0.95,
            flatness: 0.50,
        }),
        "hip-hop" | "hiphop" => Some(GenreProfile {
            centroid: 0.20,
            zcr: 0.15,
            tempo: 0.45,
            energy: 0.70,
            flatness: 0.25,
        }),
        "pop" => Some(GenreProfile {
            centroid: 0.22,
            zcr: 0.10,
            tempo: 0.58,
            energy: 0.55,
            flatness: 0.20,
        }),
        "country" => Some(GenreProfile {
            centroid: 0.17,
            zcr: 0.07,
            tempo: 0.55,
            energy: 0.45,
            flatness: 0.18,
        }),
        "folk" => Some(GenreProfile {
            centroid: 0.14,
            zcr: 0.05,
            tempo: 0.38,
            energy: 0.28,
            flatness: 0.12,
        }),
        "rnb" | "r&b" => Some(GenreProfile {
            centroid: 0.21,
            zcr: 0.08,
            tempo: 0.48,
            energy: 0.60,
            flatness: 0.22,
        }),
        "latin" => Some(GenreProfile {
            centroid: 0.24,
            zcr: 0.10,
            tempo: 0.60,
            energy: 0.65,
            flatness: 0.20,
        }),
        "ambient" => Some(GenreProfile {
            centroid: 0.08,
            zcr: 0.02,
            tempo: 0.30,
            energy: 0.15,
            flatness: 0.55,
        }),
        "blues" => Some(GenreProfile {
            centroid: 0.20,
            zcr: 0.08,
            tempo: 0.42,
            energy: 0.50,
            flatness: 0.15,
        }),
        "reggae" => Some(GenreProfile {
            centroid: 0.18,
            zcr: 0.06,
            tempo: 0.40,
            energy: 0.50,
            flatness: 0.18,
        }),
        "world" => Some(GenreProfile {
            centroid: 0.20,
            zcr: 0.09,
            tempo: 0.50,
            energy: 0.55,
            flatness: 0.25,
        }),
        "soundtrack" => Some(GenreProfile {
            centroid: 0.22,
            zcr: 0.06,
            tempo: 0.50,
            energy: 0.45,
            flatness: 0.30,
        }),
        _ => None,
    }
}

/// Compute similarity between two genres using Euclidean distance in feature space.
///
/// Returns a value in [0.0, 1.0] where 1.0 means identical profiles.
/// Returns `None` if either genre is not recognised.
#[must_use]
pub fn genre_similarity(genre_a: &str, genre_b: &str) -> Option<f32> {
    let pa = genre_profile(genre_a)?;
    let pb = genre_profile(genre_b)?;
    Some(profile_similarity(&pa, &pb))
}

/// Compute similarity between two profiles.
///
/// Uses inverse exponential of Euclidean distance so the result is in [0, 1].
#[must_use]
pub fn profile_similarity(a: &GenreProfile, b: &GenreProfile) -> f32 {
    let dc = a.centroid - b.centroid;
    let dz = a.zcr - b.zcr;
    let dt = a.tempo - b.tempo;
    let de = a.energy - b.energy;
    let df = a.flatness - b.flatness;
    let dist_sq = dc * dc + dz * dz + dt * dt + de * de + df * df;
    (-dist_sq.sqrt() * 4.0).exp()
}

/// Return the top-N most similar genres to the given genre.
///
/// Results are sorted by descending similarity (the query genre itself is excluded).
#[must_use]
pub fn most_similar_genres(genre: &str, n: usize) -> Vec<(String, f32)> {
    let target = match genre_profile(genre) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let lower = genre.to_ascii_lowercase();
    let mut scored: Vec<(String, f32)> = all_top_level_genres()
        .iter()
        .filter(|g| g.to_ascii_lowercase() != lower)
        .filter_map(|g| {
            genre_profile(g).map(|p| ((*g).to_string(), profile_similarity(&target, &p)))
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
}

// ── Genre Blend Detection ───────────────────────────────────────────────────

/// A detected genre blend with per-genre confidence scores.
#[derive(Debug, Clone)]
pub struct GenreBlend {
    /// Per-genre confidence scores summing to approximately 1.0.
    pub scores: HashMap<String, f32>,
    /// Whether the track is a clear blend of 2+ genres (top two scores both > threshold).
    pub is_blended: bool,
    /// The primary genre.
    pub primary: String,
    /// The secondary genre (if blended).
    pub secondary: Option<String>,
}

/// Detect genre blending from a set of per-genre confidence scores.
///
/// A track is considered blended if the second-highest score is at least
/// `blend_threshold` fraction of the highest score.
///
/// # Arguments
/// * `scores` - genre name to confidence mapping
/// * `blend_threshold` - minimum ratio of secondary to primary (e.g., 0.5)
#[must_use]
pub fn detect_genre_blend(scores: &HashMap<String, f32>, blend_threshold: f32) -> GenreBlend {
    let mut sorted: Vec<(String, f32)> = scores.iter().map(|(k, &v)| (k.clone(), v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let primary = sorted
        .first()
        .map_or_else(|| "Unknown".to_string(), |(g, _)| g.clone());
    let primary_score = sorted.first().map_or(0.0, |(_, s)| *s);

    let secondary_entry = sorted.get(1);
    let secondary_score = secondary_entry.map_or(0.0, |(_, s)| *s);

    let is_blended = primary_score > 0.0 && (secondary_score / primary_score) >= blend_threshold;

    let secondary = if is_blended {
        secondary_entry.map(|(g, _)| g.clone())
    } else {
        None
    };

    GenreBlend {
        scores: scores.clone(),
        is_blended,
        primary,
        secondary,
    }
}

/// Detect genre blend from a raw audio signal by running classification first.
///
/// # Errors
///
/// Returns error if genre classification fails.
pub fn detect_blend_from_signal(
    signal: &[f32],
    sample_rate: f32,
    blend_threshold: f32,
) -> crate::MirResult<GenreBlend> {
    let classifier = GenreClassifier::new(sample_rate);
    let result = classifier.classify(signal)?;
    Ok(detect_genre_blend(&result.genres, blend_threshold))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genre_family_rock() {
        let family = genre_family("rock");
        assert!(family.is_some());
        let f = family.expect("rock family");
        assert_eq!(f.parent, "Rock");
        assert!(f.members.len() >= 6);
        assert!(f.members.contains(&"Hard Rock"));
        assert!(f.members.contains(&"Punk Rock"));
    }

    #[test]
    fn test_genre_family_unknown() {
        assert!(genre_family("nonexistent").is_none());
    }

    #[test]
    fn test_genre_family_case_insensitive() {
        assert!(genre_family("ELECTRONIC").is_some());
        assert!(genre_family("Jazz").is_some());
    }

    #[test]
    fn test_all_top_level_genres_count() {
        let genres = all_top_level_genres();
        assert!(
            genres.len() >= 15,
            "expected at least 15 top-level genres, got {}",
            genres.len()
        );
    }

    #[test]
    fn test_genre_hierarchy_rock() {
        let nodes = genre_hierarchy("rock");
        assert!(!nodes.is_empty());
        // First node is top-level
        assert_eq!(nodes[0].tier, GenreTier::TopLevel);
        assert_eq!(nodes[0].name, "Rock");
        assert!(nodes[0].parent.is_none());
        // Should have sub-genre nodes
        let sub_count = nodes
            .iter()
            .filter(|n| n.tier == GenreTier::SubGenre)
            .count();
        assert!(sub_count >= 6);
    }

    #[test]
    fn test_genre_hierarchy_unknown() {
        let nodes = genre_hierarchy("xyz");
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_genre_hierarchy_micro_genres_exist() {
        let nodes = genre_hierarchy("jazz");
        let micro_count = nodes
            .iter()
            .filter(|n| n.tier == GenreTier::MicroGenre)
            .count();
        assert!(micro_count > 0, "should have micro-genre nodes");
    }

    #[test]
    fn test_genre_profile_rock() {
        let profile = genre_profile("rock");
        assert!(profile.is_some());
        let p = profile.expect("rock profile");
        assert!(p.energy > 0.5);
    }

    #[test]
    fn test_genre_profile_unknown() {
        assert!(genre_profile("xyzzy").is_none());
    }

    #[test]
    fn test_genre_similarity_same_genre() {
        let sim = genre_similarity("rock", "rock");
        assert!(sim.is_some());
        let s = sim.expect("similarity");
        assert!(
            (s - 1.0).abs() < 1e-6,
            "same genre similarity should be 1.0, got {s}"
        );
    }

    #[test]
    fn test_genre_similarity_different_genres() {
        let sim = genre_similarity("rock", "ambient");
        assert!(sim.is_some());
        let s = sim.expect("similarity");
        assert!(s < 0.8, "rock vs ambient should be dissimilar, got {s}");
    }

    #[test]
    fn test_genre_similarity_symmetry() {
        let ab = genre_similarity("jazz", "classical").expect("ab");
        let ba = genre_similarity("classical", "jazz").expect("ba");
        assert!((ab - ba).abs() < 1e-6, "similarity must be symmetric");
    }

    #[test]
    fn test_genre_similarity_unknown() {
        assert!(genre_similarity("rock", "nonexistent").is_none());
    }

    #[test]
    fn test_most_similar_genres() {
        let result = most_similar_genres("rock", 3);
        assert_eq!(result.len(), 3);
        // Scores should be in descending order
        assert!(result[0].1 >= result[1].1);
        assert!(result[1].1 >= result[2].1);
    }

    #[test]
    fn test_most_similar_genres_unknown() {
        let result = most_similar_genres("unknown_genre", 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_profile_similarity_range() {
        let a = GenreProfile {
            centroid: 0.0,
            zcr: 0.0,
            tempo: 0.0,
            energy: 0.0,
            flatness: 0.0,
        };
        let b = GenreProfile {
            centroid: 1.0,
            zcr: 1.0,
            tempo: 1.0,
            energy: 1.0,
            flatness: 1.0,
        };
        let sim = profile_similarity(&a, &b);
        assert!(sim >= 0.0 && sim <= 1.0, "similarity out of range: {sim}");
    }

    #[test]
    fn test_detect_genre_blend_blended() {
        let mut scores = HashMap::new();
        scores.insert("Rock".to_string(), 0.40);
        scores.insert("Electronic".to_string(), 0.35);
        scores.insert("Pop".to_string(), 0.15);
        scores.insert("Jazz".to_string(), 0.10);

        let blend = detect_genre_blend(&scores, 0.5);
        assert!(
            blend.is_blended,
            "should detect blend when second is >= 50% of first"
        );
        assert_eq!(blend.primary, "Rock");
        assert_eq!(blend.secondary, Some("Electronic".to_string()));
    }

    #[test]
    fn test_detect_genre_blend_not_blended() {
        let mut scores = HashMap::new();
        scores.insert("Classical".to_string(), 0.90);
        scores.insert("Jazz".to_string(), 0.05);
        scores.insert("Pop".to_string(), 0.05);

        let blend = detect_genre_blend(&scores, 0.5);
        assert!(
            !blend.is_blended,
            "should not be blended when second is far below first"
        );
        assert_eq!(blend.primary, "Classical");
        assert!(blend.secondary.is_none());
    }

    #[test]
    fn test_detect_genre_blend_empty() {
        let scores = HashMap::new();
        let blend = detect_genre_blend(&scores, 0.5);
        assert!(!blend.is_blended);
        assert_eq!(blend.primary, "Unknown");
    }

    #[test]
    fn test_genre_family_all_top_levels_have_families() {
        for genre in all_top_level_genres() {
            let family = genre_family(genre);
            assert!(
                family.is_some(),
                "top-level genre '{genre}' should have a family"
            );
        }
    }

    #[test]
    fn test_genre_hierarchy_has_three_tiers() {
        let nodes = genre_hierarchy("electronic");
        let tiers: std::collections::HashSet<GenreTier> = nodes.iter().map(|n| n.tier).collect();
        assert!(tiers.contains(&GenreTier::TopLevel));
        assert!(tiers.contains(&GenreTier::SubGenre));
        assert!(tiers.contains(&GenreTier::MicroGenre));
    }

    #[test]
    fn test_genre_node_parent_child_consistency() {
        let nodes = genre_hierarchy("metal");
        // Every sub-genre should reference the top-level as parent
        for node in &nodes {
            if node.tier == GenreTier::SubGenre {
                assert_eq!(node.parent, Some("Metal".to_string()));
            }
        }
    }

    #[test]
    fn test_rock_metal_more_similar_than_rock_ambient() {
        let rock_metal = genre_similarity("rock", "metal").expect("rock-metal");
        let rock_ambient = genre_similarity("rock", "ambient").expect("rock-ambient");
        assert!(
            rock_metal > rock_ambient,
            "Rock-Metal ({rock_metal}) should be more similar than Rock-Ambient ({rock_ambient})"
        );
    }
}
