//! Template rendering engine with parsed-AST caching.
//!
//! Each unique template string is parsed once into a [`Vec<Segment>`] that
//! represents its structure.  Subsequent calls to [`TemplateEngine::render`]
//! with the same template string skip re-parsing and walk the already-built
//! segment list directly, paying only the cost of variable look-ups and string
//! concatenation.

use crate::error::Result;
use crate::template::functions::TemplateFunctions;
use crate::template::TemplateContext;
use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Segment AST
// ---------------------------------------------------------------------------

/// A single parsed unit of a template.
///
/// The three forms that the template engine recognises are:
///
/// * `{variable}` → [`Segment::Variable`]
/// * `{function(args)}` → [`Segment::Function`]
/// * `{if cond}…{else}…{endif}` → [`Segment::Conditional`]
///
/// Everything else is captured as [`Segment::Literal`].
#[derive(Debug, Clone)]
enum Segment {
    /// Raw text that is emitted verbatim.
    Literal(String),
    /// `{name}` — replaced by the context value of `name`.
    Variable(String),
    /// `{name(args_raw)}` — forwarded to [`TemplateFunctions::call`].
    Function {
        name: String,
        /// Raw argument string exactly as it appears inside the parentheses.
        args_raw: String,
    },
    /// `{if condition}then_segments…{else}else_segments…{endif}`
    Conditional {
        condition: String,
        then_segs: Vec<Segment>,
        else_segs: Vec<Segment>,
    },
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Lightweight scanner that converts a raw template string into [`Segment`]s.
///
/// The scanner operates in a single forward pass over the character stream.
/// It does **not** use the pre-compiled regexes from [`TemplateEngine`]; those
/// are kept on the engine solely for the legacy substitution helpers that are
/// no longer called from the hot path (they remain available to other callers).
struct Parser {
    src: Vec<char>,
    pos: usize,
}

impl Parser {
    fn new(template: &str) -> Self {
        Self {
            src: template.chars().collect(),
            pos: 0,
        }
    }

    fn remaining(&self) -> &[char] {
        &self.src[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.src.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    /// Skip over a literal string (asserts it is present).
    fn skip_str(&mut self, s: &str) {
        for _ in s.chars() {
            self.advance();
        }
    }

    /// Attempt to consume `s` at the current position. Returns true on match.
    fn try_consume_str(&mut self, s: &str) -> bool {
        let needle: Vec<char> = s.chars().collect();
        let matched = self.remaining().starts_with(&needle);
        if matched {
            self.skip_str(s);
        }
        matched
    }

    /// Parse the top-level segment list, stopping at `stop_marker` if provided.
    ///
    /// `stop_marker` is used when parsing nested bodies of `{if …}` blocks;
    /// the caller is responsible for consuming the marker after this returns.
    fn parse_segments(&mut self, stop_markers: &[&str]) -> Vec<Segment> {
        let mut segs: Vec<Segment> = Vec::new();
        let mut literal_buf = String::new();

        'outer: loop {
            if self.pos >= self.src.len() {
                break;
            }

            // Check stop markers first
            for marker in stop_markers {
                let needle: Vec<char> = marker.chars().collect();
                if self.remaining().starts_with(&needle) {
                    break 'outer;
                }
            }

            // Opening brace?
            if self.peek() == Some('{') {
                // Save accumulated literal
                if !literal_buf.is_empty() {
                    segs.push(Segment::Literal(std::mem::take(&mut literal_buf)));
                }

                // Peek ahead to decide which form this is
                if let Some(seg) = self.try_parse_tag() {
                    segs.push(seg);
                    continue;
                } else {
                    // Not a recognised tag — treat the `{` as literal text
                    if let Some(c) = self.advance() {
                        literal_buf.push(c);
                    }
                    continue;
                }
            }

            // Ordinary character
            if let Some(c) = self.advance() {
                literal_buf.push(c);
            }
        }

        if !literal_buf.is_empty() {
            segs.push(Segment::Literal(literal_buf));
        }

        segs
    }

    /// Try to parse one tag starting at the current `{`.  Returns `None` if
    /// the content is not a recognised tag (caller should treat `{` as literal).
    fn try_parse_tag(&mut self) -> Option<Segment> {
        // We need a speculative look-ahead without side effects.  Build a
        // temporary string of what is inside the braces.
        let start = self.pos;

        // Confirm opening brace
        if self.peek() != Some('{') {
            return None;
        }

        // Find the matching closing brace for this `{`.  We look for the first
        // `}` that is not nested (templates do not support nested `{}`).
        let close_offset = self.src[self.pos + 1..].iter().position(|&c| c == '}')?;
        let inner_start = self.pos + 1; // skip the opening `{`
        let inner_end = inner_start + close_offset;
        let inner: String = self.src[inner_start..inner_end].iter().collect();

        // ---- {if condition} ------------------------------------------------
        if let Some(condition_raw) = inner.strip_prefix("if ") {
            let condition = condition_raw.trim().to_string();
            if condition.is_empty() {
                return None;
            }
            // Consume `{if condition}`
            self.pos = inner_end + 1; // +1 for the closing `}`

            // Parse then-body until `{else}` or `{endif}`
            let then_segs = self.parse_segments(&["{else}", "{endif}"]);

            let else_segs = if self.try_consume_str("{else}") {
                self.parse_segments(&["{endif}"])
            } else {
                Vec::new()
            };

            // Consume `{endif}`
            self.try_consume_str("{endif}");

            return Some(Segment::Conditional {
                condition,
                then_segs,
                else_segs,
            });
        }

        // ---- {function(args)} ---------------------------------------------
        // inner looks like: `funcname(args_raw)` — look for `(` and trailing `)`
        // The closing brace comes AFTER the `)`, so inner must end with `)`.
        if inner.ends_with(')') {
            if let Some(paren_pos) = inner.find('(') {
                let func_name = &inner[..paren_pos];
                // Validate function name
                if !func_name.is_empty()
                    && func_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    && func_name
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic() || c == '_')
                {
                    let args_raw = inner[paren_pos + 1..inner.len() - 1].to_string();
                    // Consume the whole `{funcname(args_raw)}`
                    self.pos = inner_end + 1;
                    return Some(Segment::Function {
                        name: func_name.to_string(),
                        args_raw,
                    });
                }
            }
        }

        // ---- {variable} ---------------------------------------------------
        // inner must be a valid identifier: [a-zA-Z_][a-zA-Z0-9_]*
        let inner_trimmed = inner.trim();
        if !inner_trimmed.is_empty()
            && inner_trimmed
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_')
            && inner_trimmed
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
        {
            // Make sure it's not an `if`/`else`/`endif` keyword
            if inner_trimmed != "else" && inner_trimmed != "endif" {
                self.pos = inner_end + 1;
                return Some(Segment::Variable(inner_trimmed.to_string()));
            }
        }

        // Not recognised — rewind and return None
        self.pos = start;
        None
    }
}

/// Parse a raw template string into a segment list.
fn parse_template(template: &str) -> Vec<Segment> {
    let mut parser = Parser::new(template);
    parser.parse_segments(&[])
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Evaluate a pre-parsed segment list against a context, appending the result
/// into `out`.
fn eval_segments(
    segs: &[Segment],
    context: &TemplateContext,
    functions: &TemplateFunctions,
    out: &mut String,
) -> Result<()> {
    for seg in segs {
        match seg {
            Segment::Literal(s) => out.push_str(s),

            Segment::Variable(name) => {
                if let Some(val) = context.get(name) {
                    out.push_str(val);
                } else {
                    tracing::warn!("Variable not found: {}", name);
                    // Leave the placeholder in the output (matches existing behaviour)
                    out.push('{');
                    out.push_str(name);
                    out.push('}');
                }
            }

            Segment::Function { name, args_raw } => {
                let output = functions.call(name, args_raw, context)?;
                out.push_str(&output);
            }

            Segment::Conditional {
                condition,
                then_segs,
                else_segs,
            } => {
                if evaluate_condition(condition, context) {
                    eval_segments(then_segs, context, functions, out)?;
                } else {
                    eval_segments(else_segs, context, functions, out)?;
                }
            }
        }
    }
    Ok(())
}

/// Evaluate a condition expression against the context.
///
/// Supports the same forms as the original engine:
/// * `!varname` — negation
/// * `varname==value` — equality
/// * `varname!=value` — inequality
/// * `varname` — truth check (non-empty, non-`"false"`, non-`"0"`)
fn evaluate_condition(condition: &str, context: &TemplateContext) -> bool {
    let condition = condition.trim();

    if let Some(var) = condition.strip_prefix('!') {
        let val = context
            .get(var.trim())
            .map_or("", std::string::String::as_str);
        return val.is_empty() || val == "false" || val == "0";
    }

    if let Some(eq_pos) = condition.find("==") {
        let var_name = condition[..eq_pos].trim();
        let expected = condition[eq_pos + 2..].trim().trim_matches('"');
        let actual = context
            .get(var_name)
            .map_or("", std::string::String::as_str);
        return actual == expected;
    }

    if let Some(ne_pos) = condition.find("!=") {
        let var_name = condition[..ne_pos].trim();
        let expected = condition[ne_pos + 2..].trim().trim_matches('"');
        let actual = context
            .get(var_name)
            .map_or("", std::string::String::as_str);
        return actual != expected;
    }

    let val = context
        .get(condition)
        .map_or("", std::string::String::as_str);
    !val.is_empty() && val != "false" && val != "0"
}

// ---------------------------------------------------------------------------
// TemplateEngine
// ---------------------------------------------------------------------------

/// Template engine for processing templates.
///
/// Regex patterns are compiled once at construction time and reused across
/// all calls to [`render`](Self::render), avoiding repeated compilation cost.
///
/// Additionally, each unique template string is **parsed once** into an AST of
/// `Segment`s stored in `parsed_cache`.  Subsequent `render` calls for the
/// same template string bypass re-parsing entirely and walk the cached segment
/// list directly.
pub struct TemplateEngine {
    functions: TemplateFunctions,
    /// Pre-compiled pattern for simple `{variable}` substitution (kept for
    /// internal use / legacy helpers).
    var_re: Regex,
    /// Pre-compiled pattern for `{function(args)}` call recognition (kept for
    /// internal use / legacy helpers).
    func_re: Regex,
    /// Cache mapping raw template strings to their parsed segment ASTs.
    parsed_cache: Mutex<HashMap<String, Arc<Vec<Segment>>>>,
}

impl TemplateEngine {
    /// Create a new template engine
    #[must_use]
    pub fn new() -> Self {
        let var_re = Regex::new(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
            .unwrap_or_else(|_| unreachable!("var_re is a compile-time constant and always valid"));
        let func_re = Regex::new(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)\}").unwrap_or_else(|_| {
            unreachable!("func_re is a compile-time constant and always valid")
        });
        Self {
            functions: TemplateFunctions::new(),
            var_re,
            func_re,
            parsed_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Return (or build and cache) the parsed segment list for `template`.
    ///
    /// Uses a `Mutex` for interior mutability; poison-safe via
    /// `unwrap_or_else(|e| e.into_inner())`.
    fn parse_template_to_segments(&self, template: &str) -> Arc<Vec<Segment>> {
        let mut cache = self.parsed_cache.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(arc) = cache.get(template) {
            return Arc::clone(arc);
        }

        let segs = Arc::new(parse_template(template));
        cache.insert(template.to_string(), Arc::clone(&segs));
        segs
    }

    /// Render a template with context.
    ///
    /// The template string is parsed into segments on the first call and
    /// the result is cached; subsequent calls with the same `template` skip
    /// the parse phase entirely.
    ///
    /// # Arguments
    ///
    /// * `template` - Template string
    /// * `context` - Template context
    ///
    /// # Errors
    ///
    /// Returns an error if a function call inside the template fails.
    pub fn render(&self, template: &str, context: &TemplateContext) -> Result<String> {
        let segs = self.parse_template_to_segments(template);
        let mut out = String::with_capacity(template.len());
        eval_segments(&segs, context, &self.functions, &mut out)?;
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Legacy helpers (kept for completeness; not called from `render`)
    // -----------------------------------------------------------------------

    /// Substitute `{variable}` placeholders using the pre-compiled regex.
    #[allow(dead_code)]
    fn substitute_variables(&self, template: &str, context: &TemplateContext) -> String {
        let mut result = template.to_string();

        for cap in self.var_re.captures_iter(template) {
            if let Some(var_name) = cap.get(1) {
                let var_name_str = var_name.as_str();
                if let Some(value) = context.get(var_name_str) {
                    let pattern = format!("{{{var_name_str}}}");
                    result = result.replace(&pattern, value);
                } else {
                    tracing::warn!("Variable not found: {}", var_name_str);
                }
            }
        }

        result
    }

    #[allow(dead_code)]
    fn process_functions(&self, template: &str, context: &TemplateContext) -> Result<String> {
        let mut result = template.to_string();

        for cap in self.func_re.captures_iter(template) {
            if let (Some(func_name), Some(args)) = (cap.get(1), cap.get(2)) {
                let func_name_str = func_name.as_str();
                let args_str = args.as_str();

                let output = self.functions.call(func_name_str, args_str, context)?;

                let pattern = format!("{{{func_name_str}({args_str})}}");
                result = result.replace(&pattern, &output);
            }
        }

        Ok(result)
    }

    #[allow(dead_code, clippy::unnecessary_wraps)]
    fn process_conditionals(template: &str, context: &TemplateContext) -> Result<String> {
        let mut result = template.to_string();

        while let Some(if_start) = result.find("{if ") {
            let Some(endif_offset) = result[if_start..].find("{endif}") else {
                break;
            };
            let endif_pos = if_start + endif_offset;

            let cond_start = if_start + 4;
            let Some(cond_offset) = result[cond_start..].find('}') else {
                break;
            };
            let cond_end = cond_start + cond_offset;
            let condition = result[cond_start..cond_end].trim().to_string();

            let content_start = cond_end + 1;
            let block_content = &result[content_start..endif_pos];

            let (true_content, false_content) = if let Some(else_pos) = block_content.find("{else}")
            {
                (&block_content[..else_pos], &block_content[else_pos + 6..])
            } else {
                (block_content, "")
            };

            let condition_true = evaluate_condition(&condition, context);

            let replacement = if condition_true {
                true_content
            } else {
                false_content
            };
            let full_block = &result[if_start..endif_pos + 7];
            let result_new = result.replacen(full_block, replacement, 1);
            result = result_new;
        }

        Ok(result)
    }

    /// Expose the cache length for testing purposes.
    #[cfg(test)]
    fn cache_len(&self) -> usize {
        self.parsed_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- original tests (preserved) ----------------------------------------

    #[test]
    fn test_engine_creation() {
        let engine = TemplateEngine::new();
        let _ = engine;
    }

    #[test]
    fn test_engine_has_cached_regexes() {
        let engine1 = TemplateEngine::new();
        let engine2 = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set("key".to_string(), "val".to_string());
        let r1 = engine1
            .render("{key}", &ctx)
            .expect("render 1 must succeed");
        let r2 = engine2
            .render("{key}", &ctx)
            .expect("render 2 must succeed");
        assert_eq!(r1, r2);
        assert_eq!(r1, "val");
    }

    #[test]
    fn test_simple_variable_substitution() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("filename".to_string(), "test.mp4".to_string());

        let result = engine.render("Output: {filename}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "Output: test.mp4");
    }

    #[test]
    fn test_multiple_variables() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("name".to_string(), "video".to_string());
        context.set("ext".to_string(), "mp4".to_string());

        let result = engine.render("{name}.{ext}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "video.mp4");
    }

    #[test]
    fn test_missing_variable() {
        let engine = TemplateEngine::new();
        let context = TemplateContext::new();

        let result = engine.render("{missing}", &context);
        assert!(result.is_ok());
        // Missing variables are left as-is (with braces)
        assert_eq!(result.expect("result should be valid"), "{missing}");
    }

    #[test]
    fn test_conditional_true() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("enabled".to_string(), "true".to_string());

        let result = engine.render("{if enabled}yes{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "yes");
    }

    #[test]
    fn test_conditional_false_missing() {
        let engine = TemplateEngine::new();
        let context = TemplateContext::new();

        let result = engine.render("{if enabled}yes{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "");
    }

    #[test]
    fn test_conditional_else() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("enabled".to_string(), "false".to_string());

        let result = engine.render("{if enabled}yes{else}no{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "no");
    }

    #[test]
    fn test_conditional_negation() {
        let engine = TemplateEngine::new();
        let context = TemplateContext::new();

        let result = engine.render("{if !missing}not set{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "not set");
    }

    #[test]
    fn test_conditional_equality() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("codec".to_string(), "h264".to_string());

        let result = engine.render("{if codec==h264}mp4{else}other{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "mp4");
    }

    #[test]
    fn test_conditional_inequality() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("codec".to_string(), "vp9".to_string());

        let result = engine.render("{if codec!=h264}not-h264{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "not-h264");
    }

    #[test]
    fn test_conditional_zero_is_falsy() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("count".to_string(), "0".to_string());

        let result = engine.render("{if count}has items{else}empty{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "empty");
    }

    // ---- new cache tests ---------------------------------------------------

    /// Basic variable substitution (using double-brace `{{…}}` is not the syntax;
    /// the engine uses single-brace `{…}`).
    #[test]
    fn test_render_basic_variable() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set("name".to_string(), "World".to_string());

        let result = engine
            .render("Hello {name}", &ctx)
            .expect("render must succeed");
        assert_eq!(result, "Hello World");
    }

    /// Rendering the same template twice must result in exactly one cache entry.
    #[test]
    fn test_render_cache_hit_single_entry() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set("x".to_string(), "1".to_string());

        engine.render("{x}", &ctx).expect("first render");
        engine.render("{x}", &ctx).expect("second render");

        assert_eq!(
            engine.cache_len(),
            1,
            "same template must occupy 1 cache slot"
        );
    }

    /// Rendering two distinct templates must produce two cache entries.
    #[test]
    fn test_render_different_templates_two_entries() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set("a".to_string(), "A".to_string());
        ctx.set("b".to_string(), "B".to_string());

        engine.render("{a}", &ctx).expect("render a");
        engine.render("{b}", &ctx).expect("render b");

        assert_eq!(
            engine.cache_len(),
            2,
            "two different templates → 2 cache slots"
        );
    }

    /// The cache is keyed on the template string, so the same template with
    /// different contexts must still yield different outputs.
    #[test]
    fn test_render_same_template_different_context() {
        let engine = TemplateEngine::new();

        let mut ctx1 = TemplateContext::new();
        ctx1.set("x".to_string(), "1".to_string());

        let mut ctx2 = TemplateContext::new();
        ctx2.set("x".to_string(), "2".to_string());

        let r1 = engine.render("{x}", &ctx1).expect("render ctx1");
        let r2 = engine.render("{x}", &ctx2).expect("render ctx2");

        assert_ne!(r1, r2, "different contexts must produce different output");
        assert_eq!(r1, "1");
        assert_eq!(r2, "2");
        // Still only one cache entry (template key is the same)
        assert_eq!(engine.cache_len(), 1);
    }

    /// Function segment: `{uppercase(name)}` must produce the uppercased value.
    #[test]
    fn test_render_function_segment() {
        let engine = TemplateEngine::new();
        let mut ctx = TemplateContext::new();
        ctx.set("name".to_string(), "hello".to_string());

        let result = engine
            .render("{uppercase(name)}", &ctx)
            .expect("function render must succeed");
        assert_eq!(result, "HELLO");
    }

    /// Multiple threads rendering the same template concurrently must not panic.
    #[test]
    fn test_render_concurrent_safe() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(TemplateEngine::new());
        let template = "{x} and {y}";

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    let mut ctx = TemplateContext::new();
                    ctx.set("x".to_string(), i.to_string());
                    ctx.set("y".to_string(), (i * 2).to_string());
                    for _ in 0..10 {
                        eng.render(template, &ctx)
                            .expect("concurrent render must not fail");
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread must not panic");
        }

        // The template was the same string in all threads → 1 cache entry.
        assert_eq!(engine.cache_len(), 1);
    }
}
