use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tensorlogic_ir::{TLExpr, Term};

use super::types::{
    AggregateFunction, BindExpr, FilterCondition, GraphPattern, PatternElement, QueryType,
    SelectElement, SparqlQuery, TriplePattern,
};

/// SPARQL query parser and compiler
pub struct SparqlCompiler {
    /// Map of predicate IRIs to TensorLogic predicate names
    predicate_mapping: HashMap<String, String>,
}

impl SparqlCompiler {
    pub fn new() -> Self {
        SparqlCompiler {
            predicate_mapping: HashMap::new(),
        }
    }

    /// Add a mapping from IRI to predicate name
    ///
    /// Example: map `"http://example.org/knows"` to `"knows"`
    pub fn add_predicate_mapping(&mut self, iri: String, predicate_name: String) {
        self.predicate_mapping.insert(iri, predicate_name);
    }

    /// Parse a SPARQL query (SELECT, ASK, DESCRIBE, or CONSTRUCT)
    ///
    /// Supports SPARQL 1.1 syntax including:
    /// ```sparql
    /// # SELECT query
    /// SELECT DISTINCT ?x ?y WHERE {
    ///   ?x <http://example.org/knows> ?y .
    ///   OPTIONAL { ?x <http://example.org/age> ?age }
    ///   FILTER(?age > 18)
    /// } LIMIT 10
    ///
    /// # ASK query
    /// ASK WHERE {
    ///   ?x <http://example.org/knows> ?y .
    /// }
    ///
    /// # DESCRIBE query
    /// DESCRIBE ?x WHERE {
    ///   ?x <http://example.org/knows> ?y .
    /// }
    ///
    /// # CONSTRUCT query
    /// CONSTRUCT { ?x <http://example.org/friend> ?y }
    /// WHERE { ?x <http://example.org/knows> ?y }
    /// ```
    ///
    /// Note: This is a simplified parser for demonstration.
    /// For production, use a dedicated SPARQL parser.
    pub fn parse_query(&self, sparql: &str) -> Result<SparqlQuery> {
        // Normalize the query by collapsing whitespace and removing newlines within clauses
        let normalized = sparql
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        // Determine query type
        let query_type = self.parse_query_type(&normalized)?;

        // Parse WHERE clause
        let where_pattern = self.parse_where_clause(&normalized)?;

        // Parse GROUP BY and HAVING
        let group_by = self.parse_group_by(&normalized);
        let having = self.parse_having(&normalized)?;

        // Parse solution modifiers
        let limit = self.parse_limit(&normalized);
        let offset = self.parse_offset(&normalized);
        let order_by = self.parse_order_by(&normalized);

        Ok(SparqlQuery {
            query_type,
            where_pattern,
            group_by,
            having,
            limit,
            offset,
            order_by,
        })
    }

    /// Parse GROUP BY clause
    fn parse_group_by(&self, normalized: &str) -> Vec<String> {
        let mut group_by = Vec::new();

        if let Some(group_pos) = normalized.find("GROUP BY") {
            // Find the end of GROUP BY (next clause or end of query)
            let remaining = &normalized[group_pos + 8..];
            let end_pos = remaining
                .find("HAVING")
                .or_else(|| remaining.find("ORDER BY"))
                .or_else(|| remaining.find("LIMIT"))
                .or_else(|| remaining.find("OFFSET"))
                .unwrap_or(remaining.len());

            let group_part = remaining[..end_pos].trim();
            for token in group_part.split_whitespace() {
                if let Some(var_name) = token.strip_prefix('?') {
                    group_by.push(var_name.to_string());
                }
            }
        }

        group_by
    }

    /// Parse HAVING clause
    fn parse_having(&self, normalized: &str) -> Result<Vec<FilterCondition>> {
        let mut conditions = Vec::new();

        if let Some(having_pos) = normalized.find("HAVING") {
            // Find the end of HAVING (next clause or end of query)
            let remaining = &normalized[having_pos + 6..];
            let end_pos = remaining
                .find("ORDER BY")
                .or_else(|| remaining.find("LIMIT"))
                .or_else(|| remaining.find("OFFSET"))
                .unwrap_or(remaining.len());

            let having_part = remaining[..end_pos].trim();

            // Parse conditions similar to FILTER
            if !having_part.is_empty() {
                if let Some(filter) = self.parse_filter(&format!("FILTER{}", having_part))? {
                    conditions.push(filter);
                }
            }
        }

        Ok(conditions)
    }

    /// Parse an aggregate function
    fn parse_aggregate(&self, text: &str) -> Option<(AggregateFunction, String)> {
        let text = text.trim();

        // Check for AS alias
        let (func_part, alias) = if let Some(as_pos) = text.to_uppercase().find(" AS ") {
            let alias_start = as_pos + 4;
            let alias = text[alias_start..]
                .trim()
                .trim_matches(|c| c == '?' || c == ')')
                .to_string();
            (text[..as_pos].trim(), Some(alias))
        } else {
            (text, None)
        };

        // Parse aggregate function
        let upper = func_part.to_uppercase();

        if upper.starts_with("COUNT(") {
            let inner = func_part[6..].trim_end_matches(')').trim();
            let distinct = inner.to_uppercase().starts_with("DISTINCT");
            let var_part = if distinct { inner[8..].trim() } else { inner };
            let variable = if var_part == "*" {
                None
            } else {
                Some(var_part.trim_start_matches('?').to_string())
            };
            return Some((
                AggregateFunction::Count { variable, distinct },
                alias.unwrap_or_else(|| "count".to_string()),
            ));
        }

        if upper.starts_with("SUM(") {
            let inner = func_part[4..].trim_end_matches(')').trim();
            let distinct = inner.to_uppercase().starts_with("DISTINCT");
            let var_part = if distinct { inner[8..].trim() } else { inner };
            let variable = var_part.trim_start_matches('?').to_string();
            return Some((
                AggregateFunction::Sum { variable, distinct },
                alias.unwrap_or_else(|| "sum".to_string()),
            ));
        }

        if upper.starts_with("AVG(") {
            let inner = func_part[4..].trim_end_matches(')').trim();
            let distinct = inner.to_uppercase().starts_with("DISTINCT");
            let var_part = if distinct { inner[8..].trim() } else { inner };
            let variable = var_part.trim_start_matches('?').to_string();
            return Some((
                AggregateFunction::Avg { variable, distinct },
                alias.unwrap_or_else(|| "avg".to_string()),
            ));
        }

        if upper.starts_with("MIN(") {
            let inner = func_part[4..].trim_end_matches(')').trim();
            let variable = inner.trim_start_matches('?').to_string();
            return Some((
                AggregateFunction::Min { variable },
                alias.unwrap_or_else(|| "min".to_string()),
            ));
        }

        if upper.starts_with("MAX(") {
            let inner = func_part[4..].trim_end_matches(')').trim();
            let variable = inner.trim_start_matches('?').to_string();
            return Some((
                AggregateFunction::Max { variable },
                alias.unwrap_or_else(|| "max".to_string()),
            ));
        }

        if upper.starts_with("GROUP_CONCAT(") {
            let inner = func_part[13..].trim_end_matches(')').trim();
            let distinct = inner.to_uppercase().starts_with("DISTINCT");
            let var_part = if distinct { inner[8..].trim() } else { inner };
            // Check for SEPARATOR
            let (variable, separator) =
                if let Some(sep_pos) = var_part.to_uppercase().find("; SEPARATOR") {
                    let var = var_part[..sep_pos]
                        .trim()
                        .trim_start_matches('?')
                        .to_string();
                    let sep_start = var_part.find('=').map(|p| p + 1).unwrap_or(sep_pos);
                    let sep = var_part[sep_start..].trim().trim_matches('"').to_string();
                    (var, Some(sep))
                } else {
                    (var_part.trim_start_matches('?').to_string(), None)
                };
            return Some((
                AggregateFunction::GroupConcat {
                    variable,
                    separator,
                    distinct,
                },
                alias.unwrap_or_else(|| "group_concat".to_string()),
            ));
        }

        if upper.starts_with("SAMPLE(") {
            let inner = func_part[7..].trim_end_matches(')').trim();
            let variable = inner.trim_start_matches('?').to_string();
            return Some((
                AggregateFunction::Sample { variable },
                alias.unwrap_or_else(|| "sample".to_string()),
            ));
        }

        None
    }

    /// Parse the query type (SELECT, ASK, DESCRIBE, CONSTRUCT)
    fn parse_query_type(&self, normalized: &str) -> Result<QueryType> {
        if normalized.contains("ASK") {
            Ok(QueryType::Ask)
        } else if let Some(describe_pos) = normalized.find("DESCRIBE") {
            // Parse DESCRIBE resources
            let where_pos = normalized.find("WHERE").unwrap_or(normalized.len());
            let describe_part = normalized[describe_pos + 8..where_pos].trim();
            let mut resources = Vec::new();

            for token in describe_part.split_whitespace() {
                if token.starts_with('?') || token.starts_with('<') {
                    resources.push(
                        token
                            .trim_matches(|c| c == '?' || c == '<' || c == '>')
                            .to_string(),
                    );
                }
            }

            Ok(QueryType::Describe { resources })
        } else if normalized.contains("CONSTRUCT") {
            // Parse CONSTRUCT template
            let template = self.parse_construct_template(normalized)?;
            Ok(QueryType::Construct { template })
        } else if let Some(select_pos) = normalized.find("SELECT") {
            // Parse SELECT variables and aggregates
            let where_pos = normalized.find("WHERE").unwrap_or(normalized.len());
            let select_part = normalized[select_pos + 6..where_pos].trim();

            let distinct = select_part.starts_with("DISTINCT");
            let vars_part = if distinct {
                &select_part[8..]
            } else {
                select_part
            };

            let mut select_vars = Vec::new();
            let mut projections = Vec::new();

            // Split on commas or parentheses to handle aggregates
            let mut current_token = String::new();
            let mut paren_depth = 0;

            for c in vars_part.chars() {
                match c {
                    '(' => {
                        paren_depth += 1;
                        current_token.push(c);
                    }
                    ')' => {
                        paren_depth -= 1;
                        current_token.push(c);
                    }
                    ' ' | ',' if paren_depth == 0 => {
                        if !current_token.trim().is_empty() {
                            let token = current_token.trim();
                            // Strip outer parentheses for aggregate expressions
                            let token = if token.starts_with('(') && token.ends_with(')') {
                                &token[1..token.len() - 1]
                            } else {
                                token
                            };
                            if let Some((agg_func, alias)) = self.parse_aggregate(token) {
                                projections.push(SelectElement::Aggregate {
                                    function: agg_func,
                                    alias: Some(alias.clone()),
                                });
                                select_vars.push(alias);
                            } else if let Some(var_name) = token.strip_prefix('?') {
                                projections.push(SelectElement::Variable(var_name.to_string()));
                                select_vars.push(var_name.to_string());
                            } else if token == "*" {
                                projections.push(SelectElement::Variable("*".to_string()));
                                select_vars.push("*".to_string());
                            }
                        }
                        current_token.clear();
                    }
                    _ => current_token.push(c),
                }
            }

            // Handle the last token
            if !current_token.trim().is_empty() {
                let token = current_token.trim();
                // Strip outer parentheses for aggregate expressions
                let token = if token.starts_with('(') && token.ends_with(')') {
                    &token[1..token.len() - 1]
                } else {
                    token
                };
                if let Some((agg_func, alias)) = self.parse_aggregate(token) {
                    projections.push(SelectElement::Aggregate {
                        function: agg_func,
                        alias: Some(alias.clone()),
                    });
                    select_vars.push(alias);
                } else if let Some(var_name) = token.strip_prefix('?') {
                    projections.push(SelectElement::Variable(var_name.to_string()));
                    select_vars.push(var_name.to_string());
                } else if token == "*" {
                    projections.push(SelectElement::Variable("*".to_string()));
                    select_vars.push("*".to_string());
                }
            }

            Ok(QueryType::Select {
                projections,
                select_vars,
                distinct,
            })
        } else {
            Err(anyhow!("Unable to determine query type"))
        }
    }

    /// Parse CONSTRUCT template patterns
    fn parse_construct_template(&self, normalized: &str) -> Result<Vec<TriplePattern>> {
        let construct_pos = normalized
            .find("CONSTRUCT")
            .ok_or_else(|| anyhow!("No CONSTRUCT found"))?;
        let where_pos = normalized.find("WHERE").unwrap_or(normalized.len());

        // Find template content between { and }
        let template_start = normalized[construct_pos..where_pos]
            .find('{')
            .ok_or_else(|| anyhow!("No opening brace in CONSTRUCT template"))?;
        let template_end = normalized[construct_pos..where_pos]
            .rfind('}')
            .ok_or_else(|| anyhow!("No closing brace in CONSTRUCT template"))?;

        let template_content =
            &normalized[construct_pos + template_start + 1..construct_pos + template_end];

        let mut patterns = Vec::new();
        for statement in self.split_sparql_statements(template_content) {
            if let Some(pattern) = self.parse_triple_pattern(statement)? {
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Parse WHERE clause into graph patterns
    fn parse_where_clause(&self, normalized: &str) -> Result<GraphPattern> {
        // Find WHERE clause content (between { and })
        if let Some(where_start) = normalized.find("WHERE") {
            if let Some(brace_start) = normalized[where_start..].find('{') {
                let content_start = where_start + brace_start + 1;

                // Find matching closing brace
                let closing_brace = self.find_matching_brace(&normalized[content_start..])?;
                let where_content = &normalized[content_start..content_start + closing_brace];

                return self.parse_graph_pattern(where_content);
            }
        }

        Err(anyhow!("No WHERE clause found"))
    }

    /// Parse a graph pattern (handles OPTIONAL, UNION, FILTER)
    fn parse_graph_pattern(&self, content: &str) -> Result<GraphPattern> {
        let content = content.trim();

        if content.is_empty() {
            return Err(anyhow!("Empty graph pattern"));
        }

        // Check for UNION (top-level only)
        if let Some(union_pos) = content.find("UNION") {
            // Ensure it's not inside braces
            let before_union = &content[..union_pos];
            let open_braces = before_union.matches('{').count();
            let close_braces = before_union.matches('}').count();

            if open_braces == close_braces {
                // UNION is at top level
                let left_part = before_union.trim();
                let right_part = content[union_pos + 5..].trim();

                let left_pattern = self.parse_graph_pattern(left_part)?;
                let right_pattern = self.parse_graph_pattern(right_part)?;

                return Ok(GraphPattern::Union(
                    Box::new(left_pattern),
                    Box::new(right_pattern),
                ));
            }
        }

        // Parse statements using split_sparql_statements
        let mut patterns = Vec::new();
        let statements = self.split_sparql_statements(content);

        for statement in statements {
            let statement = statement.trim();

            if statement.is_empty() {
                continue;
            }

            // Check for OPTIONAL
            if statement.starts_with("OPTIONAL") {
                // Find the content in braces
                if let Some(brace_start_pos) = statement.find('{') {
                    let content_start = brace_start_pos + 1;
                    if let Ok(closing_offset) =
                        self.find_matching_brace(&statement[content_start..])
                    {
                        let optional_content =
                            &statement[content_start..content_start + closing_offset];
                        let inner_pattern = self.parse_graph_pattern(optional_content)?;
                        patterns.push(GraphPattern::Optional(Box::new(inner_pattern)));
                        continue;
                    }
                }
            }

            // Check for FILTER
            if statement.starts_with("FILTER") {
                if let Some(filter) = self.parse_filter(statement)? {
                    patterns.push(GraphPattern::Filter(filter));
                }
                continue;
            }

            // Check for BIND
            if statement.starts_with("BIND") {
                patterns.push(self.parse_bind(statement)?);
                continue;
            }

            // Check for VALUES
            if statement.starts_with("VALUES") {
                patterns.push(self.parse_values(statement)?);
                continue;
            }

            // Check for nested braces (subgraph pattern)
            if statement.starts_with('{') && statement.ends_with('}') {
                let inner = &statement[1..statement.len() - 1];
                let inner_pattern = self.parse_graph_pattern(inner)?;
                patterns.push(inner_pattern);
                continue;
            }

            // Parse as triple pattern
            if let Some(pattern) = self.parse_triple_pattern(statement)? {
                patterns.push(GraphPattern::Triple(pattern));
            }
        }

        if patterns.is_empty() {
            Err(anyhow!("Empty graph pattern in content: {}", content))
        } else if patterns.len() == 1 {
            Ok(patterns
                .into_iter()
                .next()
                .expect("patterns is non-empty, len==1 guaranteed"))
        } else {
            Ok(GraphPattern::Group(patterns))
        }
    }

    /// Find matching closing brace
    fn find_matching_brace(&self, content: &str) -> Result<usize> {
        let mut depth = 1;
        let chars: Vec<char> = content.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(i);
                    }
                }
                _ => {}
            }
        }

        Err(anyhow!("No matching closing brace found"))
    }

    /// Parse LIMIT modifier
    fn parse_limit(&self, normalized: &str) -> Option<usize> {
        if let Some(limit_pos) = normalized.find("LIMIT") {
            let after_limit = &normalized[limit_pos + 5..].trim();
            if let Some(num_str) = after_limit.split_whitespace().next() {
                return num_str.parse().ok();
            }
        }
        None
    }

    /// Parse OFFSET modifier
    fn parse_offset(&self, normalized: &str) -> Option<usize> {
        if let Some(offset_pos) = normalized.find("OFFSET") {
            let after_offset = &normalized[offset_pos + 6..].trim();
            if let Some(num_str) = after_offset.split_whitespace().next() {
                return num_str.parse().ok();
            }
        }
        None
    }

    /// Parse ORDER BY modifier
    fn parse_order_by(&self, normalized: &str) -> Vec<String> {
        if let Some(order_pos) = normalized.find("ORDER BY") {
            let after_order = &normalized[order_pos + 8..];

            // Find the end of ORDER BY clause (either LIMIT, OFFSET, or end of string)
            let limit_offset = after_order.find("LIMIT").unwrap_or(after_order.len());
            let offset_offset = after_order.find("OFFSET").unwrap_or(after_order.len());
            let end_offset = limit_offset.min(offset_offset);

            let order_part = after_order[..end_offset].trim();
            return order_part
                .split_whitespace()
                .filter_map(|s| s.strip_prefix('?').map(|v| v.to_string()))
                .collect();
        }
        Vec::new()
    }

    /// Split SPARQL WHERE content into statements, respecting URI boundaries,
    /// brace blocks (VALUES/OPTIONAL bodies), and parenthesised expressions.
    ///
    /// Only splits on '.' that appear at brace-depth == 0 and paren-depth == 0
    /// and outside of '<…>' URI delimiters or '"…"' string literals.
    fn split_sparql_statements<'a>(&self, content: &'a str) -> Vec<&'a str> {
        let mut statements = Vec::new();
        let mut current_start = 0;
        let mut inside_uri = false;
        let mut inside_string = false;
        let mut brace_depth: usize = 0;
        let mut paren_depth: usize = 0;

        for (byte_pos, ch) in content.char_indices() {
            match ch {
                '<' if !inside_string && brace_depth == 0 && paren_depth == 0 => {
                    inside_uri = true;
                }
                '>' if !inside_string && inside_uri => {
                    inside_uri = false;
                }
                '"' if !inside_uri => {
                    inside_string = !inside_string;
                }
                '{' if !inside_uri && !inside_string => {
                    brace_depth = brace_depth.saturating_add(1);
                }
                '}' if !inside_uri && !inside_string => {
                    brace_depth = brace_depth.saturating_sub(1);
                }
                '(' if !inside_uri && !inside_string => {
                    paren_depth = paren_depth.saturating_add(1);
                }
                ')' if !inside_uri && !inside_string => {
                    paren_depth = paren_depth.saturating_sub(1);
                }
                '.' if !inside_uri && !inside_string && brace_depth == 0 && paren_depth == 0 => {
                    let stmt = &content[current_start..byte_pos];
                    if !stmt.trim().is_empty() {
                        statements.push(stmt);
                    }
                    current_start = byte_pos + ch.len_utf8();
                }
                _ => {}
            }
        }

        // Add the last statement if there's anything left
        if current_start < content.len() {
            let statement = &content[current_start..];
            if !statement.trim().is_empty() {
                statements.push(statement);
            }
        }

        statements
    }

    /// Parse a triple pattern
    fn parse_triple_pattern(&self, line: &str) -> Result<Option<TriplePattern>> {
        // Remove trailing dot and split by whitespace
        let line = line.trim_end_matches('.').trim();
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Ok(None);
        }

        let subject = self.parse_pattern_element(parts[0])?;
        let predicate = self.parse_pattern_element(parts[1])?;
        let object = self.parse_pattern_element(parts[2])?;

        Ok(Some(TriplePattern {
            subject,
            predicate,
            object,
        }))
    }

    /// Parse a pattern element (variable or constant)
    fn parse_pattern_element(&self, s: &str) -> Result<PatternElement> {
        if let Some(var_name) = s.strip_prefix('?') {
            Ok(PatternElement::Variable(var_name.to_string()))
        } else if let Some(iri) = s.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
            Ok(PatternElement::Constant(iri.to_string()))
        } else if let Some(literal) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            Ok(PatternElement::Constant(literal.to_string()))
        } else {
            Ok(PatternElement::Constant(s.to_string()))
        }
    }

    /// Parse a FILTER clause
    fn parse_filter(&self, line: &str) -> Result<Option<FilterCondition>> {
        let filter_content = line
            .strip_prefix("FILTER")
            .and_then(|s| s.trim().strip_prefix('('))
            .and_then(|s| s.trim().strip_suffix(')'))
            .map(|s| s.trim());

        if let Some(content) = filter_content {
            // Check for built-in functions
            if content.starts_with("BOUND(") {
                if let Some(var_end) = content.find(')') {
                    let var = &content[6..var_end].trim_start_matches('?');
                    return Ok(Some(FilterCondition::Bound(var.to_string())));
                }
            } else if content.starts_with("isIRI(") || content.starts_with("isURI(") {
                // Both isIRI and isURI have the same length (6 characters including parenthesis)
                let start_pos = 6;
                if let Some(var_end) = content.find(')') {
                    let var = &content[start_pos..var_end].trim_start_matches('?');
                    return Ok(Some(FilterCondition::IsIri(var.to_string())));
                }
            } else if content.starts_with("isLiteral(") {
                if let Some(var_end) = content.find(')') {
                    let var = &content[10..var_end].trim_start_matches('?');
                    return Ok(Some(FilterCondition::IsLiteral(var.to_string())));
                }
            } else if content.starts_with("regex(") {
                // regex(?var, "pattern")
                if let Some(comma_pos) = content.find(',') {
                    let var = content[6..comma_pos].trim().trim_start_matches('?');
                    let pattern_part = content[comma_pos + 1..]
                        .trim()
                        .trim_end_matches(')')
                        .trim_matches('"');
                    return Ok(Some(FilterCondition::Regex(
                        var.to_string(),
                        pattern_part.to_string(),
                    )));
                }
            }

            // Check for comparison operators
            if content.contains(">=") {
                let parts: Vec<&str> = content.split(">=").map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::GreaterOrEqual(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains("<=") {
                let parts: Vec<&str> = content.split("<=").map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::LessOrEqual(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains(">") && !content.contains(">=") {
                let parts: Vec<&str> = content.split('>').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::GreaterThan(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains("<") && !content.contains("<=") {
                let parts: Vec<&str> = content.split('<').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::LessThan(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains("!=") {
                let parts: Vec<&str> = content.split("!=").map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::NotEquals(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains("=")
                && !content.contains("!=")
                && !content.contains(">=")
                && !content.contains("<=")
            {
                let parts: Vec<&str> = content.split('=').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::Equals(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            }
        }

        Ok(None)
    }

    /// Parse a BIND clause: `BIND ( <element> AS ?<var> )`
    fn parse_bind(&self, stmt: &str) -> Result<GraphPattern> {
        // Strip "BIND" prefix then the outer parentheses
        let inner = stmt.trim_start_matches("BIND").trim();
        let inner = inner
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| anyhow!("malformed BIND clause: {}", stmt))?
            .trim();

        // Split on " AS " (case-insensitive by trying both forms)
        let (lhs, rhs) = inner
            .split_once(" AS ")
            .or_else(|| inner.split_once(" as "))
            .ok_or_else(|| anyhow!("BIND missing AS: {}", stmt))?;

        let lhs = lhs.trim();
        let rhs = rhs.trim().trim_start_matches('?');
        let elem = self.parse_pattern_element(lhs)?;
        Ok(GraphPattern::Bind(BindExpr::Term(elem), rhs.to_string()))
    }

    /// Parse a VALUES clause: single or multi-variable form.
    fn parse_values(&self, stmt: &str) -> Result<GraphPattern> {
        let rest = stmt.trim_start_matches("VALUES").trim();

        // Find the opening brace
        let brace_start = rest
            .find('{')
            .ok_or_else(|| anyhow!("VALUES missing '{{': {}", stmt))?;
        let var_part = rest[..brace_start].trim();
        let body = rest[brace_start + 1..].trim().trim_end_matches('}').trim();

        // Parse variable list
        let vars: Vec<String> = if var_part.starts_with('(') {
            var_part
                .trim_start_matches('(')
                .trim_end_matches(')')
                .split_whitespace()
                .map(|v| v.trim_start_matches('?').to_string())
                .collect()
        } else {
            vec![var_part.trim_start_matches('?').to_string()]
        };

        let n = vars.len();

        // Parse rows
        let rows: Result<Vec<Vec<PatternElement>>> = if n == 1 {
            // Single-var: bare terms separated by whitespace
            body.split_whitespace()
                .map(|t| Ok(vec![self.parse_pattern_element(t)?]))
                .collect()
        } else {
            // Multi-var: groups of ( t1 t2 … )
            let mut rows = Vec::new();
            let mut remaining = body;
            while let Some(start) = remaining.find('(') {
                let end = remaining[start..]
                    .find(')')
                    .map(|i| start + i)
                    .ok_or_else(|| anyhow!("VALUES row missing ')'"))?;
                let row_str = remaining[start + 1..end].trim();
                let row: Result<Vec<PatternElement>> = row_str
                    .split_whitespace()
                    .map(|t| self.parse_pattern_element(t))
                    .collect();
                let row = row?;
                if row.len() != n {
                    return Err(anyhow!(
                        "VALUES row has {} terms but {} vars declared",
                        row.len(),
                        n
                    ));
                }
                rows.push(row);
                remaining = &remaining[end + 1..];
            }
            Ok(rows)
        };

        Ok(GraphPattern::Values(vars, rows?))
    }

    /// Compile a SPARQL query to TensorLogic expression
    ///
    /// Converts SPARQL patterns to TLExpr predicates and filters to constraints.
    /// Supports all query types (SELECT, ASK, DESCRIBE, CONSTRUCT) and advanced
    /// patterns (OPTIONAL, UNION).
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::sparql::SparqlCompiler;
    ///
    /// let mut compiler = SparqlCompiler::new();
    /// compiler.add_predicate_mapping(
    ///     "http://example.org/knows".to_string(),
    ///     "knows".to_string()
    /// );
    ///
    /// // SELECT query
    /// let query = r#"
    ///     SELECT ?x ?y WHERE {
    ///       ?x <http://example.org/knows> ?y .
    ///     }
    /// "#;
    ///
    /// let sparql_query = compiler.parse_query(query).expect("unwrap");
    /// let tl_expr = compiler.compile_to_tensorlogic(&sparql_query).expect("unwrap");
    ///
    /// // ASK query
    /// let ask_query = r#"
    ///     ASK WHERE {
    ///       ?x <http://example.org/knows> ?y .
    ///     }
    /// "#;
    ///
    /// let sparql_ask = compiler.parse_query(ask_query).expect("unwrap");
    /// let ask_expr = compiler.compile_to_tensorlogic(&sparql_ask).expect("unwrap");
    /// ```
    pub fn compile_to_tensorlogic(&self, query: &SparqlQuery) -> Result<TLExpr> {
        // Compile WHERE clause pattern
        let where_expr = self.compile_graph_pattern(&query.where_pattern)?;

        // For ASK queries, wrap in EXISTS quantifier
        match &query.query_type {
            QueryType::Ask => {
                // ASK is essentially EXISTS over all variables in the pattern
                Ok(where_expr) // The pattern itself represents existence
            }
            QueryType::Select { select_vars, .. } => {
                // For SELECT, the expression is the WHERE clause
                // Variable projection happens at execution time
                if select_vars.is_empty() || select_vars.contains(&"*".to_string()) {
                    Ok(where_expr)
                } else {
                    // Could add quantifiers for non-selected variables here
                    Ok(where_expr)
                }
            }
            QueryType::Describe { .. } => {
                // DESCRIBE returns all triples about specified resources
                Ok(where_expr)
            }
            QueryType::Construct { template: _ } => {
                // CONSTRUCT applies template pattern after WHERE clause matches
                // For now, we return the WHERE clause; template application
                // would happen at execution time
                Ok(where_expr)
            }
        }
    }

    /// Compile a graph pattern to TLExpr
    fn compile_graph_pattern(&self, pattern: &GraphPattern) -> Result<TLExpr> {
        match pattern {
            GraphPattern::Triple(triple) => self.compile_triple_pattern(triple),

            GraphPattern::Group(patterns) => {
                if patterns.is_empty() {
                    return Err(anyhow!("Empty pattern group"));
                }

                let mut exprs: Vec<TLExpr> = Vec::new();
                for p in patterns {
                    exprs.push(self.compile_graph_pattern(p)?);
                }

                // Combine with AND
                Ok(exprs
                    .into_iter()
                    .reduce(TLExpr::and)
                    .expect("exprs is non-empty, checked above"))
            }

            GraphPattern::Optional(inner) => {
                // OPTIONAL in SPARQL is like left-outer join
                // In logic, we can represent as: pattern OR TRUE
                // This ensures the outer pattern succeeds even if inner fails
                let inner_expr = self.compile_graph_pattern(inner)?;

                // Use OR with a trivially true expression
                // This gives "optional" semantics - the pattern can match or not
                Ok(TLExpr::or(inner_expr.clone(), TLExpr::pred("true", vec![])))
            }

            GraphPattern::Union(left, right) => {
                // UNION is disjunction
                let left_expr = self.compile_graph_pattern(left)?;
                let right_expr = self.compile_graph_pattern(right)?;
                Ok(TLExpr::or(left_expr, right_expr))
            }

            GraphPattern::Filter(filter_cond) => self.compile_filter_condition(filter_cond),

            GraphPattern::Bind(expr, var) => {
                // Lower BIND to an equality predicate: equals(var, term)
                let term = match expr {
                    BindExpr::Term(PatternElement::Variable(v)) => Term::var(v),
                    BindExpr::Term(PatternElement::Constant(c)) => Term::constant(c),
                };
                Ok(TLExpr::pred("equals", vec![Term::var(var), term]))
            }

            GraphPattern::Values(vars, rows) => {
                // Lower VALUES to disjunction of equality conjunctions.
                // Each row becomes an AND of equals(var, term), rows are OR-ed.
                if rows.is_empty() {
                    return Err(anyhow!("VALUES clause has no rows"));
                }
                let row_exprs: Result<Vec<TLExpr>> = rows
                    .iter()
                    .map(|row| {
                        let eq_exprs: Result<Vec<TLExpr>> = vars
                            .iter()
                            .zip(row.iter())
                            .map(|(var, elem)| {
                                let term = match elem {
                                    PatternElement::Variable(v) => Term::var(v),
                                    PatternElement::Constant(c) => Term::constant(c),
                                };
                                Ok(TLExpr::pred("equals", vec![Term::var(var), term]))
                            })
                            .collect();
                        let eq_exprs = eq_exprs?;
                        eq_exprs
                            .into_iter()
                            .reduce(TLExpr::and)
                            .ok_or_else(|| anyhow!("VALUES row is empty"))
                    })
                    .collect();
                row_exprs?
                    .into_iter()
                    .reduce(TLExpr::or)
                    .ok_or_else(|| anyhow!("VALUES clause produced no expressions"))
            }
        }
    }

    /// Compile a triple pattern to TLExpr
    fn compile_triple_pattern(&self, pattern: &TriplePattern) -> Result<TLExpr> {
        let pred_name = match &pattern.predicate {
            PatternElement::Constant(iri) => {
                // Try to map IRI to predicate name
                self.predicate_mapping
                    .get(iri)
                    .cloned()
                    .unwrap_or_else(|| Self::iri_to_name(iri))
            }
            PatternElement::Variable(v) => {
                return Err(anyhow!("Variable predicates not supported: ?{}", v));
            }
        };

        let subj_term = match &pattern.subject {
            PatternElement::Variable(v) => Term::var(v),
            PatternElement::Constant(c) => Term::constant(c),
        };

        let obj_term = match &pattern.object {
            PatternElement::Variable(v) => Term::var(v),
            PatternElement::Constant(c) => Term::constant(c),
        };

        Ok(TLExpr::pred(&pred_name, vec![subj_term, obj_term]))
    }

    /// Compile a filter condition to TLExpr
    fn compile_filter_condition(&self, filter: &FilterCondition) -> Result<TLExpr> {
        let expr = match filter {
            FilterCondition::Equals(var, val) => {
                TLExpr::pred("equals", vec![Term::var(var), Term::constant(val)])
            }
            FilterCondition::NotEquals(var, val) => TLExpr::negate(TLExpr::pred(
                "equals",
                vec![Term::var(var), Term::constant(val)],
            )),
            FilterCondition::GreaterThan(var, val) => {
                TLExpr::pred("greaterThan", vec![Term::var(var), Term::constant(val)])
            }
            FilterCondition::LessThan(var, val) => {
                TLExpr::pred("lessThan", vec![Term::var(var), Term::constant(val)])
            }
            FilterCondition::GreaterOrEqual(var, val) => {
                TLExpr::pred("greaterOrEqual", vec![Term::var(var), Term::constant(val)])
            }
            FilterCondition::LessOrEqual(var, val) => {
                TLExpr::pred("lessOrEqual", vec![Term::var(var), Term::constant(val)])
            }
            FilterCondition::Regex(var, pattern) => {
                TLExpr::pred("matches", vec![Term::var(var), Term::constant(pattern)])
            }
            FilterCondition::Bound(var) => TLExpr::pred("bound", vec![Term::var(var)]),
            FilterCondition::IsIri(var) => TLExpr::pred("isIri", vec![Term::var(var)]),
            FilterCondition::IsLiteral(var) => TLExpr::pred("isLiteral", vec![Term::var(var)]),
        };

        Ok(expr)
    }

    /// Extract local name from IRI
    fn iri_to_name(iri: &str) -> String {
        iri.split(['/', '#']).next_back().unwrap_or(iri).to_string()
    }
}

impl Default for SparqlCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bind_variable() {
        let compiler = SparqlCompiler::new();
        let result = compiler
            .parse_bind("BIND ( ?z AS ?y )")
            .expect("parse_bind should succeed");
        assert_eq!(
            result,
            GraphPattern::Bind(
                BindExpr::Term(PatternElement::Variable("z".to_string())),
                "y".to_string()
            )
        );
    }

    #[test]
    fn test_parse_bind_constant() {
        let compiler = SparqlCompiler::new();
        let result = compiler
            .parse_bind(r#"BIND ( "hello" AS ?greeting )"#)
            .expect("parse_bind should succeed");
        assert_eq!(
            result,
            GraphPattern::Bind(
                BindExpr::Term(PatternElement::Constant("hello".to_string())),
                "greeting".to_string()
            )
        );
    }

    #[test]
    fn test_parse_values_single_var() {
        let compiler = SparqlCompiler::new();
        let result = compiler
            .parse_values("VALUES ?x { 1 2 3 }")
            .expect("parse_values should succeed");
        assert_eq!(
            result,
            GraphPattern::Values(
                vec!["x".to_string()],
                vec![
                    vec![PatternElement::Constant("1".to_string())],
                    vec![PatternElement::Constant("2".to_string())],
                    vec![PatternElement::Constant("3".to_string())],
                ]
            )
        );
    }

    #[test]
    fn test_parse_values_multi_var() {
        let compiler = SparqlCompiler::new();
        let result = compiler
            .parse_values(r#"VALUES (?x ?y) { (1 "a") (2 "b") }"#)
            .expect("parse_values should succeed");
        assert_eq!(
            result,
            GraphPattern::Values(
                vec!["x".to_string(), "y".to_string()],
                vec![
                    vec![
                        PatternElement::Constant("1".to_string()),
                        PatternElement::Constant("a".to_string()),
                    ],
                    vec![
                        PatternElement::Constant("2".to_string()),
                        PatternElement::Constant("b".to_string()),
                    ],
                ]
            )
        );
    }

    #[test]
    fn test_filter_bound() {
        let compiler = SparqlCompiler::new();
        let filter = compiler.parse_filter("FILTER(BOUND(?x))").expect("unwrap");

        match filter {
            Some(FilterCondition::Bound(var)) => {
                assert_eq!(var, "x");
            }
            _ => panic!("Expected BOUND filter"),
        }
    }

    #[test]
    fn test_filter_is_iri() {
        let compiler = SparqlCompiler::new();
        let filter = compiler.parse_filter("FILTER(isIRI(?x))").expect("unwrap");

        match filter {
            Some(FilterCondition::IsIri(var)) => {
                assert_eq!(var, "x");
            }
            _ => panic!("Expected isIRI filter"),
        }
    }

    #[test]
    fn test_filter_regex() {
        let compiler = SparqlCompiler::new();
        let filter = compiler
            .parse_filter(r#"FILTER(regex(?name, "^John"))"#)
            .expect("unwrap");

        match filter {
            Some(FilterCondition::Regex(var, pattern)) => {
                assert_eq!(var, "name");
                assert_eq!(pattern, "^John");
            }
            _ => panic!("Expected regex filter"),
        }
    }

    #[test]
    fn test_iri_to_name() {
        assert_eq!(
            SparqlCompiler::iri_to_name("http://example.org/knows"),
            "knows"
        );
        assert_eq!(
            SparqlCompiler::iri_to_name("http://xmlns.com/foaf/0.1#Person"),
            "Person"
        );
        assert_eq!(SparqlCompiler::iri_to_name("simple"), "simple");
    }
}
