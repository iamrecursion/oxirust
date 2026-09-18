//! Advanced Interactive Tutor for Special Functions
//!
//! This example provides an adaptive tutoring system for learning special functions.
//! Split from the original 2541-line file to comply with the 2000-line policy.
//!
//! Features:
//! - Adaptive difficulty based on performance
//! - Interactive problem-solving sessions
//! - Achievement tracking
//!
//! Run with: cargo run --example advanced_interactive_tutor

#![allow(clippy::all)]

use scirs2_core::Complex64;
use scirs2_special::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io::{self, Write};

#[derive(Debug, Clone)]
struct UserProfile {
    name: String,
    level: u32,
    experience_points: u32,
    achievements: Vec<String>,
    mastery_scores: HashMap<String, f64>,
    current_streak: u32,
}

impl UserProfile {
    fn new(name: String) -> Self {
        Self {
            name,
            level: 1,
            experience_points: 0,
            achievements: Vec::new(),
            mastery_scores: HashMap::new(),
            current_streak: 0,
        }
    }

    fn add_experience(&mut self, points: u32) {
        self.experience_points += points;
        let new_level = (self.experience_points / 100) + 1;
        if new_level > self.level {
            self.level = new_level;
            println!("\nCongratulations! You've reached level {}!", self.level);
        }
    }

    fn update_mastery(&mut self, topic: &str, score: f64) {
        let current = self.mastery_scores.get(topic).unwrap_or(&0.0);
        let new_mastery = (current * 0.7) + (score * 0.3);
        self.mastery_scores.insert(topic.to_string(), new_mastery);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Welcome to the Advanced Interactive Special Functions Tutor!\n");

    print!("Please enter your name: ");
    io::stdout().flush()?;
    let mut name = String::new();
    io::stdin().read_line(&mut name)?;

    let mut profile = UserProfile::new(name.trim().to_string());

    println!(
        "\nGreat! Let's start your learning journey, {}!",
        profile.name
    );

    loop {
        display_menu(&profile);

        print!("\nChoose an option: ");
        io::stdout().flush()?;

        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;

        match choice.trim() {
            "1" => gamma_lesson(&mut profile)?,
            "2" => bessel_lesson(&mut profile)?,
            "3" => error_function_lesson(&mut profile)?,
            "4" => practice_problems(&mut profile)?,
            "5" => show_progress(&profile),
            "q" | "Q" => {
                println!("\nFinal Statistics:");
                show_progress(&profile);
                println!("\nThank you for learning with us!");
                break;
            }
            _ => println!("Invalid choice. Please try again."),
        }
    }

    Ok(())
}

fn display_menu(profile: &UserProfile) {
    println!("\n{:=^60}", "");
    println!(
        "{:^60}",
        format!("Level {} - {} XP", profile.level, profile.experience_points)
    );
    println!("{:=^60}", "");
    println!("1. Gamma Function Lesson");
    println!("2. Bessel Function Lesson");
    println!("3. Error Function Lesson");
    println!("4. Practice Problems");
    println!("5. Show Progress");
    println!("q. Quit");
}

fn gamma_lesson(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Gamma Function Lesson ===");
    println!("\nThe gamma function extends the factorial to complex numbers.");
    println!("Definition: Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt");
    println!("\nKey fact: Γ(n) = (n-1)! for positive integers n");

    let mut score = 0.0;
    let total_questions = 3;

    println!("\n--- Question 1 ---");
    println!("What is Γ(5)?");
    print!("Your answer: ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    match answer.trim().parse::<f64>() {
        Ok(val) if (val - 24.0).abs() < 0.1 => {
            println!("Correct! Γ(5) = 4! = 24");
            score += 1.0;
        }
        _ => {
            println!("Incorrect. Γ(5) = 4! = 24");
            println!("Remember: Γ(n) = (n-1)!");
        }
    }

    println!("\n--- Question 2 ---");
    println!("Which of these is true?");
    println!("a) Γ(z+1) = (z+1)·Γ(z)");
    println!("b) Γ(z+1) = z·Γ(z)");
    println!("c) Γ(z) = z·Γ(z-1)");
    print!("Your answer (a/b/c): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "b" {
        println!("Correct! The functional equation is Γ(z+1) = z·Γ(z)");
        score += 1.0;
    } else {
        println!("Incorrect. The correct answer is b) Γ(z+1) = z·Γ(z)");
        println!("This is the fundamental functional equation of the gamma function.");
    }

    println!("\n--- Question 3 ---");
    println!("Calculate Γ(1/2) using the fact that Γ(1/2) = √π");
    println!("What is the numerical value? (Enter to 2 decimal places)");
    print!("Your answer: ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    let sqrt_pi = PI.sqrt();
    match answer.trim().parse::<f64>() {
        Ok(val) if (val - sqrt_pi).abs() < 0.05 => {
            println!("Correct! Γ(1/2) = √π ≈ {:.2}", sqrt_pi);
            score += 1.0;
        }
        _ => {
            println!("Incorrect. Γ(1/2) = √π ≈ {:.2}", sqrt_pi);
        }
    }

    let percentage = (score / total_questions as f64) * 100.0;
    println!(
        "\nLesson complete! Score: {}/{} ({:.0}%)",
        score, total_questions, percentage
    );

    let xp = (score * 50.0) as u32;
    profile.add_experience(xp);
    profile.update_mastery("Gamma Functions", score / total_questions as f64);

    println!("Earned {} XP!", xp);

    Ok(())
}

fn bessel_lesson(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Bessel Function Lesson ===");
    println!("\nBessel functions are solutions to Bessel's differential equation:");
    println!("x²y'' + xy' + (x² - ν²)y = 0");
    println!("\nApplications: vibrating membranes, heat conduction, wave propagation");

    let mut score = 0.0;
    let total_questions = 3;

    println!("\n--- Question 1 ---");
    println!("For a circular drumhead, which function describes the radial vibration?");
    println!("a) J_n(x) - Bessel of first kind");
    println!("b) Y_n(x) - Bessel of second kind");
    println!("c) I_n(x) - Modified Bessel of first kind");
    print!("Your answer (a/b/c): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "a" {
        println!("Correct! J_n is regular at the origin, unlike Y_n.");
        score += 1.0;
    } else {
        println!("Incorrect. J_n(x) is used because it's regular at x=0.");
        println!("Y_n(x) is singular at the origin.");
    }

    println!("\n--- Question 2 ---");
    println!("The zeros of J_0(x) are important in physics. The first zero is:");
    if let Ok(first_zero) = j0_zeros::<f64>(1) {
        println!("Approximately what value?");
        println!("a) 1.8");
        println!("b) 2.4");
        println!("c) 3.1");
        println!("d) 3.8");
        print!("Your answer (a/b/c/d): ");
        io::stdout().flush()?;

        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;

        if answer.trim().to_lowercase() == "b" {
            println!("Correct! The first zero is approximately {:.6}", first_zero);
            score += 1.0;
        } else {
            println!(
                "Incorrect. The first zero of J₀ is approximately {:.2}",
                first_zero
            );
        }
    }

    println!("\n--- Question 3 ---");
    println!("Which relationship is correct for modified Bessel functions?");
    println!("a) I_n(-x) = I_n(x)");
    println!("b) I_n(-x) = (-1)^n I_n(x)");
    println!("c) I_n(x) = i^(-n) J_n(ix)");
    print!("Your answer (a/b/c): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "c" {
        println!("Correct! Modified Bessel functions are related to regular Bessel functions.");
        score += 1.0;
    } else {
        println!("Incorrect. The correct relationship is I_n(x) = i^(-n) J_n(ix)");
    }

    let percentage = (score / total_questions as f64) * 100.0;
    println!(
        "\nLesson complete! Score: {}/{} ({:.0}%)",
        score, total_questions, percentage
    );

    let xp = (score * 50.0) as u32;
    profile.add_experience(xp);
    profile.update_mastery("Bessel Functions", score / total_questions as f64);

    println!("Earned {} XP!", xp);

    Ok(())
}

fn error_function_lesson(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Error Function Lesson ===");
    println!("\nThe error function is defined as:");
    println!("erf(x) = (2/√π) ∫₀^x e^(-t²) dt");
    println!("\nUsed extensively in probability theory and statistics.");

    let mut score = 0.0;
    let total_questions = 3;

    println!("\n--- Question 1 ---");
    println!("As x → ∞, erf(x) approaches:");
    println!("a) 0");
    println!("b) 1");
    println!("c) π");
    println!("d) ∞");
    print!("Your answer (a/b/c/d): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "b" {
        println!("Correct! erf(∞) = 1");
        score += 1.0;
    } else {
        println!("Incorrect. erf(x) → 1 as x → ∞");
    }

    println!("\n--- Question 2 ---");
    println!("The error function is:");
    println!("a) Even: erf(-x) = erf(x)");
    println!("b) Odd: erf(-x) = -erf(x)");
    println!("c) Neither even nor odd");
    print!("Your answer (a/b/c): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "b" {
        println!("Correct! The error function is an odd function.");
        score += 1.0;
    } else {
        println!("Incorrect. erf is an odd function: erf(-x) = -erf(x)");
    }

    println!("\n--- Question 3 ---");
    println!("What is the relationship between erf and erfc?");
    println!("a) erfc(x) = 1 - erf(x)");
    println!("b) erfc(x) = erf(1-x)");
    println!("c) erfc(x) = -erf(x)");
    print!("Your answer (a/b/c): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "a" {
        println!("Correct! erfc is the complementary error function.");
        score += 1.0;
    } else {
        println!("Incorrect. erfc(x) = 1 - erf(x) by definition.");
    }

    let percentage = (score / total_questions as f64) * 100.0;
    println!(
        "\nLesson complete! Score: {}/{} ({:.0}%)",
        score, total_questions, percentage
    );

    let xp = (score * 50.0) as u32;
    profile.add_experience(xp);
    profile.update_mastery("Error Functions", score / total_questions as f64);

    println!("Earned {} XP!", xp);

    Ok(())
}

fn practice_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Practice Problems ===");
    println!("Choose difficulty:");
    println!("1. Beginner");
    println!("2. Intermediate");
    println!("3. Advanced");
    print!("Your choice: ");
    io::stdout().flush()?;

    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;

    match choice.trim() {
        "1" => beginner_problems(profile)?,
        "2" => intermediate_problems(profile)?,
        "3" => advanced_problems(profile)?,
        _ => println!("Invalid choice."),
    }

    Ok(())
}

fn beginner_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Beginner Problem ---");
    println!("Compute Γ(3)");
    print!("Your answer: ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    match answer.trim().parse::<f64>() {
        Ok(val) if (val - 2.0).abs() < 0.1 => {
            println!("Correct! Γ(3) = 2! = 2");
            profile.add_experience(20);
        }
        _ => {
            println!("Incorrect. Γ(3) = 2! = 2");
            println!("Hint: Γ(n) = (n-1)! for positive integers");
        }
    }

    Ok(())
}

fn intermediate_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Intermediate Problem ---");
    println!("If J₀(x) = 0, x is a zero of the Bessel function.");
    println!("The first three zeros of J₀ are approximately: 2.40, 5.52, 8.65");
    println!("\nThese zeros appear in the resonant frequencies of circular membranes.");
    println!("True or False: The zeros are equally spaced?");
    print!("Your answer (true/false): ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    if answer.trim().to_lowercase() == "false" {
        println!("Correct! The zeros are NOT equally spaced.");
        println!("Though they become approximately equally spaced as x → ∞");
        profile.add_experience(35);
    } else {
        println!("Incorrect. The zeros are not equally spaced,");
        println!("though the spacing approaches π for large x.");
    }

    Ok(())
}

fn advanced_problems(profile: &mut UserProfile) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Advanced Problem ---");
    println!("Verify the identity: Γ(1/2) = √π");
    println!("Using numerical computation:");

    let computed = gamma(0.5);
    let expected = PI.sqrt();

    println!("Γ(1/2) computed = {:.10}", computed);
    println!("√π = {:.10}", expected);
    println!("Difference = {:.2e}", (computed - expected).abs());

    println!("\nNow compute Γ(3/2) using Γ(z+1) = z·Γ(z)");
    println!("Expected value: (1/2)·√π");
    print!("Your computed answer: ");
    io::stdout().flush()?;

    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;

    let expected_val = 0.5 * expected;
    match answer.trim().parse::<f64>() {
        Ok(val) if (val - expected_val).abs() < 0.01 => {
            println!("Correct! Γ(3/2) = {:.6}", expected_val);
            profile.add_experience(50);
        }
        _ => {
            println!(
                "Incorrect. Γ(3/2) = (1/2)·Γ(1/2) = (1/2)·√π ≈ {:.6}",
                expected_val
            );
        }
    }

    Ok(())
}

fn show_progress(profile: &UserProfile) {
    println!("\n{:=^60}", " Your Progress ");
    println!("Name: {}", profile.name);
    println!("Level: {}", profile.level);
    println!("Experience Points: {}", profile.experience_points);
    println!("Current Streak: {}", profile.current_streak);

    if !profile.mastery_scores.is_empty() {
        println!("\nMastery Levels:");
        for (topic, score) in &profile.mastery_scores {
            let percentage = (score * 100.0) as u32;
            let bars = percentage / 10;
            let bar_string = "█".repeat(bars as usize);
            println!("  {:20} [{}] {}%", topic, bar_string, percentage);
        }
    }

    if !profile.achievements.is_empty() {
        println!("\nAchievements:");
        for achievement in &profile.achievements {
            println!("  - {}", achievement);
        }
    }

    println!("{:=^60}", "");
}
