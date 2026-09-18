//! The XSL property value expression language of
//! [XSL 1.1 §5.9](https://www.w3.org/TR/xsl11/#d0e5060) and the core function
//! library of [§5.10](https://www.w3.org/TR/xsl11/#d0e6262).
//!
//! Expressions are parsed into an [`Expr`] tree and evaluated against an
//! [`EvalContext`] that supplies the *property context*: the current font size,
//! the reference value for percentages and the values needed by the property
//! value functions (`from-parent()`, `inherited-property-value()`,
//! `from-nearest-specified-value()`, …).

use crate::error::{FoError, Result};
use crate::property::datatype::{Color, Length, Unit};
use crate::property::lexer::{Operator, Token, tokenize};
use crate::property::value::Value;

/// A parsed property value expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    /// A number with an optional unit name.
    Numeric(f64, Option<String>),
    /// A percentage.
    Percent(f64),
    /// A hexadecimal colour literal.
    Color(String),
    /// A quoted string.
    Literal(String),
    /// A keyword or enumeration token.
    Name(String),
    /// A function call.
    Call(String, Vec<Expr>),
    /// Unary negation.
    Negate(Box<Expr>),
    /// A binary operation.
    Binary(Operator, Box<Expr>, Box<Expr>),
}

/// The result of evaluating an expression, before it is converted into the
/// datatype the property expects.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Eval {
    /// An absolute numeric: `value` expressed in points raised to `power`.
    /// A unit-less number has a power of zero.
    Numeric {
        /// The magnitude.
        value: f64,
        /// The unit power.
        power: i32,
    },
    /// A percentage that no conversion was available for.
    Percent(f64),
    /// An enumeration token.
    Token(String),
    /// A string literal.
    Literal(String),
    /// A colour.
    Color(Color),
    /// A complete property value, produced by the property value functions.
    Value(Value),
}

impl Eval {
    /// A dimensionless number.
    pub(crate) fn number(value: f64) -> Self {
        Self::Numeric { value, power: 0 }
    }

    /// A length in points.
    pub(crate) fn length(points: f64) -> Self {
        Self::Numeric {
            value: points,
            power: 1,
        }
    }
}

/// Everything an expression needs from the formatting object it is evaluated
/// on.
///
/// The method names are the function names of XSL 1.1 §5.10, so several of them
/// start with `from_` without being constructors.
#[allow(
    clippy::wrong_self_convention,
    reason = "method names mirror the XSL core function library"
)]
pub(crate) trait EvalContext {
    /// The name of the property whose value is being computed, for diagnostics
    /// and for the no-argument form of the property value functions.
    fn property_name(&self) -> &str;

    /// The font size in force, in points; resolves the `em` unit.
    fn font_size(&self) -> f64;

    /// Device resolution used to resolve the `px` unit.
    fn pixels_per_inch(&self) -> f64;

    /// The reference value a percentage refers to, when refinement knows it.
    fn percent_reference(&self) -> Option<Length>;

    /// `from-parent()`.
    fn from_parent(&self, property: Option<&str>) -> Result<Value>;

    /// `inherited-property-value()`.
    fn inherited_property_value(&self, property: Option<&str>) -> Result<Value>;

    /// `from-nearest-specified-value()`.
    fn from_nearest_specified_value(&self, property: Option<&str>) -> Result<Value>;

    /// `from-page-master-region()`.
    fn from_page_master_region(&self, property: Option<&str>) -> Result<Value>;

    /// `from-table-column()`.
    fn from_table_column(&self, property: Option<&str>) -> Result<Value>;

    /// `merge-property-values()`.
    fn merge_property_values(&self, property: Option<&str>) -> Result<Value>;

    /// `label-end()`.
    fn label_end(&self) -> Result<Value>;

    /// `body-start()`.
    fn body_start(&self) -> Result<Value>;

    /// `system-font()`.
    fn system_font(&self, font: &str, property: Option<&str>) -> Result<Value>;

    /// `system-color()`.
    fn system_color(&self, name: &str) -> Result<Color>;
}

// --------------------------------------------------------------------- parse

/// How deeply a property expression may nest before the parser gives up.
///
/// The grammar is parsed by recursive descent, so `((((…))))` costs a run of
/// stack frames per level. A stylesheet is input like any other, and an
/// unbounded descent turns a kilobyte of parentheses into a stack overflow,
/// which aborts the process rather than raising a [`FoError`]. No property
/// value in XSL 1.1 nests anywhere near this deep.
const MAX_DEPTH: usize = 128;

struct Parser<'a> {
    property: &'a str,
    tokens: Vec<Token>,
    position: usize,
    /// How many nested productions are currently on the stack.
    depth: usize,
}

impl<'a> Parser<'a> {
    /// Runs `descend` one level deeper, refusing to go past [`MAX_DEPTH`].
    fn nested<T>(&mut self, descend: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.depth >= MAX_DEPTH {
            return Err(FoError::expression(
                self.property,
                format!("the expression nests more than {MAX_DEPTH} levels deep"),
            ));
        }
        self.depth += 1;
        let result = descend(self);
        self.depth -= 1;
        result
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        if token.is_some() {
            self.position += 1;
        }
        token
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        if self.peek() == Some(expected) {
            self.position += 1;
            Ok(())
        } else {
            Err(FoError::expression(
                self.property,
                format!("expected {expected:?}, found {:?}", self.peek()),
            ))
        }
    }

    /// `Expr ::= AdditiveExpr`
    fn expr(&mut self) -> Result<Expr> {
        self.nested(Self::additive)
    }

    fn additive(&mut self) -> Result<Expr> {
        let mut left = self.multiplicative()?;
        while let Some(Token::Operator(op @ (Operator::Add | Operator::Subtract))) = self.peek() {
            let op = *op;
            self.position += 1;
            let right = self.multiplicative()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn multiplicative(&mut self) -> Result<Expr> {
        let mut left = self.unary()?;
        while let Some(Token::Operator(
            op @ (Operator::Multiply | Operator::Divide | Operator::Modulo),
        )) = self.peek()
        {
            let op = *op;
            self.position += 1;
            let right = self.unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr> {
        // A `- - - …` or `+ + + …` chain recurses on itself, so it is bounded
        // by the same counter.
        if self.peek() == Some(&Token::Operator(Operator::Subtract)) {
            self.position += 1;
            let inner = self.nested(Self::unary)?;
            return Ok(Expr::Negate(Box::new(inner)));
        }
        if self.peek() == Some(&Token::Operator(Operator::Add)) {
            self.position += 1;
            return self.nested(Self::unary);
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Expr> {
        match self.next() {
            Some(Token::LeftParen) => {
                let inner = self.expr()?;
                self.expect(&Token::RightParen)?;
                Ok(inner)
            }
            Some(Token::Numeric(value, unit)) => Ok(Expr::Numeric(value, unit)),
            Some(Token::Percent(value)) => Ok(Expr::Percent(value)),
            Some(Token::Color(text)) => Ok(Expr::Color(text)),
            Some(Token::Literal(text)) => Ok(Expr::Literal(text)),
            Some(Token::Name(text)) => Ok(Expr::Name(text)),
            Some(Token::Function(name)) => {
                self.expect(&Token::LeftParen)?;
                let mut arguments = Vec::new();
                if self.peek() != Some(&Token::RightParen) {
                    loop {
                        arguments.push(self.expr()?);
                        if self.peek() == Some(&Token::Comma) {
                            self.position += 1;
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Token::RightParen)?;
                Ok(Expr::Call(name, arguments))
            }
            other => Err(FoError::expression(
                self.property,
                format!("unexpected {other:?}"),
            )),
        }
    }
}

/// Parses a complete property value expression.
pub(crate) fn parse(property: &str, input: &str) -> Result<Expr> {
    let tokens = tokenize(property, input)?;
    if tokens.is_empty() {
        return Err(FoError::expression(property, "empty expression"));
    }
    let mut parser = Parser {
        property,
        tokens,
        position: 0,
        depth: 0,
    };
    let expr = parser.expr()?;
    if parser.position != parser.tokens.len() {
        return Err(FoError::expression(
            property,
            format!("trailing {:?}", parser.tokens[parser.position]),
        ));
    }
    Ok(expr)
}

// ------------------------------------------------------------------ evaluate

/// Evaluates a parsed expression against a property context.
pub(crate) fn evaluate(expr: &Expr, context: &dyn EvalContext) -> Result<Eval> {
    match expr {
        Expr::Numeric(value, unit) => match unit {
            None => Ok(Eval::number(*value)),
            Some(name) => {
                let unit = Unit::parse(name).ok_or_else(|| {
                    FoError::expression(
                        context.property_name(),
                        format!("`{name}` is not a unit of measure"),
                    )
                })?;
                Ok(Eval::length(
                    Length::from_unit_with(
                        *value,
                        unit,
                        context.font_size(),
                        context.pixels_per_inch(),
                    )
                    .points(),
                ))
            }
        },
        Expr::Percent(value) => match context.percent_reference() {
            Some(reference) => Ok(Eval::length(reference.points() * value / 100.0)),
            None => Ok(Eval::Percent(*value)),
        },
        Expr::Color(text) => Color::from_hex(text).map(Eval::Color).ok_or_else(|| {
            FoError::expression(
                context.property_name(),
                format!("`{text}` is not an RGB colour"),
            )
        }),
        Expr::Literal(text) => Ok(Eval::Literal(text.clone())),
        Expr::Name(text) => Ok(Eval::Token(text.clone())),
        Expr::Negate(inner) => match evaluate(inner, context)? {
            Eval::Numeric { value, power } => Ok(Eval::Numeric {
                value: -value,
                power,
            }),
            Eval::Percent(value) => Ok(Eval::Percent(-value)),
            other => Err(FoError::expression(
                context.property_name(),
                format!("cannot negate {other:?}"),
            )),
        },
        Expr::Binary(op, left, right) => {
            let left = evaluate(left, context)?;
            let right = evaluate(right, context)?;
            apply(*op, left, right, context)
        }
        Expr::Call(name, arguments) => call(name, arguments, context),
    }
}

fn numeric(value: &Eval, context: &dyn EvalContext) -> Result<(f64, i32)> {
    match value {
        Eval::Numeric { value, power } => Ok((*value, *power)),
        // A property value function may yield a scalar, which then takes part
        // in arithmetic like any other numeric (XSL 1.1 §5.10.4).
        Eval::Value(Value::Length(length)) => Ok((length.points(), 1)),
        Eval::Value(Value::Number(number)) => Ok((*number, 0)),
        Eval::Value(Value::Integer(integer)) => Ok((*integer as f64, 0)),
        Eval::Value(Value::Percent(percent)) | Eval::Percent(percent) => {
            match context.percent_reference() {
                Some(reference) => Ok((reference.points() * percent / 100.0, 1)),
                None => Err(FoError::expression(
                    context.property_name(),
                    "a percentage cannot take part in arithmetic here: \
                     the reference value is not known during refinement",
                )),
            }
        }
        other => Err(FoError::expression(
            context.property_name(),
            format!("{other:?} is not a numeric"),
        )),
    }
}

fn apply(op: Operator, left: Eval, right: Eval, context: &dyn EvalContext) -> Result<Eval> {
    let (left_value, left_power) = numeric(&left, context)?;
    let (right_value, right_power) = numeric(&right, context)?;
    match op {
        Operator::Add | Operator::Subtract | Operator::Modulo => {
            if left_power != right_power {
                return Err(FoError::expression(
                    context.property_name(),
                    format!(
                        "`{}` requires operands of the same unit power, got {left_power} and {right_power}",
                        operator_name(op)
                    ),
                ));
            }
            let value = match op {
                Operator::Add => left_value + right_value,
                Operator::Subtract => left_value - right_value,
                _ => left_value % right_value,
            };
            Ok(Eval::Numeric {
                value,
                power: left_power,
            })
        }
        Operator::Multiply => Ok(Eval::Numeric {
            value: left_value * right_value,
            power: left_power + right_power,
        }),
        Operator::Divide => Ok(Eval::Numeric {
            value: left_value / right_value,
            power: left_power - right_power,
        }),
    }
}

fn operator_name(op: Operator) -> &'static str {
    match op {
        Operator::Add => "+",
        Operator::Subtract => "-",
        Operator::Multiply => "*",
        Operator::Divide => "div",
        Operator::Modulo => "mod",
    }
}

fn argument_name(arguments: &[Expr]) -> Option<&str> {
    match arguments.first() {
        Some(Expr::Name(name)) => Some(name),
        Some(Expr::Literal(name)) => Some(name),
        _ => None,
    }
}

fn zero_power(value: &Eval, context: &dyn EvalContext, function: &str) -> Result<f64> {
    let (value, power) = numeric(value, context)?;
    if power != 0 {
        return Err(FoError::expression(
            context.property_name(),
            format!("the argument of `{function}()` must have unit power zero"),
        ));
    }
    Ok(value)
}

fn same_power(
    left: &Eval,
    right: &Eval,
    context: &dyn EvalContext,
    f: &str,
) -> Result<(f64, f64, i32)> {
    let (left_value, left_power) = numeric(left, context)?;
    let (right_value, right_power) = numeric(right, context)?;
    if left_power != right_power {
        return Err(FoError::expression(
            context.property_name(),
            format!("the arguments of `{f}()` must have the same unit power"),
        ));
    }
    Ok((left_value, right_value, left_power))
}

fn arity(name: &str, arguments: &[Expr], expected: usize, context: &dyn EvalContext) -> Result<()> {
    if arguments.len() == expected {
        Ok(())
    } else {
        Err(FoError::expression(
            context.property_name(),
            format!(
                "`{name}()` takes {expected} argument(s), got {}",
                arguments.len()
            ),
        ))
    }
}

fn channel(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn call(name: &str, arguments: &[Expr], context: &dyn EvalContext) -> Result<Eval> {
    match name {
        // 5.10.1 Number functions.
        "floor" | "ceiling" | "round" | "abs" => {
            arity(name, arguments, 1, context)?;
            let argument = evaluate(&arguments[0], context)?;
            if name == "abs" {
                let (value, power) = numeric(&argument, context)?;
                return Ok(Eval::Numeric {
                    value: value.abs(),
                    power,
                });
            }
            let value = zero_power(&argument, context, name)?;
            let value = match name {
                "floor" => value.floor(),
                "ceiling" => value.ceil(),
                _ => (value + 0.5).floor(),
            };
            Ok(Eval::number(value))
        }
        "min" | "max" => {
            arity(name, arguments, 2, context)?;
            let left = evaluate(&arguments[0], context)?;
            let right = evaluate(&arguments[1], context)?;
            let (left, right, power) = same_power(&left, &right, context, name)?;
            let value = if name == "min" {
                left.min(right)
            } else {
                left.max(right)
            };
            Ok(Eval::Numeric { value, power })
        }
        // 5.10.2 Colour functions.
        "rgb" => {
            arity(name, arguments, 3, context)?;
            let mut channels = [0u8; 3];
            for (slot, argument) in channels.iter_mut().zip(arguments) {
                let value = evaluate(argument, context)?;
                *slot = channel(zero_power(&value, context, "rgb")?);
            }
            Ok(Eval::Color(Color::rgb(
                channels[0],
                channels[1],
                channels[2],
            )))
        }
        "rgb-icc" => {
            if arguments.len() < 4 {
                return Err(FoError::expression(
                    context.property_name(),
                    "`rgb-icc()` takes at least four arguments",
                ));
            }
            let mut fallback = [0u8; 3];
            for (slot, argument) in fallback.iter_mut().zip(&arguments[..3]) {
                let value = evaluate(argument, context)?;
                *slot = channel(zero_power(&value, context, "rgb-icc")?);
            }
            let profile = argument_name(&arguments[3..]).ok_or_else(|| {
                FoError::expression(
                    context.property_name(),
                    "the fourth argument of `rgb-icc()` must be a colour profile name",
                )
            })?;
            let mut components = Vec::new();
            for argument in &arguments[4..] {
                let value = evaluate(argument, context)?;
                components.push(zero_power(&value, context, "rgb-icc")?);
            }
            Ok(Eval::Color(Color::RgbIcc {
                red: fallback[0],
                green: fallback[1],
                blue: fallback[2],
                profile: profile.to_owned(),
                components,
            }))
        }
        "system-color" => {
            arity(name, arguments, 1, context)?;
            let color = argument_name(arguments).ok_or_else(|| {
                FoError::expression(
                    context.property_name(),
                    "`system-color()` takes a colour name",
                )
            })?;
            context.system_color(color).map(Eval::Color)
        }
        // 5.10.3 Font functions.
        "system-font" => {
            if arguments.is_empty() || arguments.len() > 2 {
                return Err(FoError::expression(
                    context.property_name(),
                    "`system-font()` takes one or two arguments",
                ));
            }
            let font = argument_name(arguments).ok_or_else(|| {
                FoError::expression(context.property_name(), "`system-font()` takes a font name")
            })?;
            let property = argument_name(&arguments[1..]);
            context.system_font(font, property).map(Eval::Value)
        }
        // 5.10.4 Property value functions.
        "from-parent" => context
            .from_parent(argument_name(arguments))
            .map(Eval::Value),
        "inherited-property-value" => context
            .inherited_property_value(argument_name(arguments))
            .map(Eval::Value),
        "from-nearest-specified-value" => context
            .from_nearest_specified_value(argument_name(arguments))
            .map(Eval::Value),
        "from-page-master-region" => context
            .from_page_master_region(argument_name(arguments))
            .map(Eval::Value),
        "from-table-column" => context
            .from_table_column(argument_name(arguments))
            .map(Eval::Value),
        "merge-property-values" => context
            .merge_property_values(argument_name(arguments))
            .map(Eval::Value),
        "label-end" => {
            arity(name, arguments, 0, context)?;
            context.label_end().map(Eval::Value)
        }
        "body-start" => {
            arity(name, arguments, 0, context)?;
            context.body_start().map(Eval::Value)
        }
        "proportional-column-width" => {
            arity(name, arguments, 1, context)?;
            let value = evaluate(&arguments[0], context)?;
            let share = zero_power(&value, context, "proportional-column-width")?;
            Ok(Eval::Value(Value::ProportionalColumnWidth(share)))
        }
        other => Err(FoError::expression(
            context.property_name(),
            format!("unknown function `{other}()`"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Bare {
        font_size: f64,
        reference: Option<Length>,
    }

    impl EvalContext for Bare {
        fn property_name(&self) -> &str {
            "test"
        }
        fn font_size(&self) -> f64 {
            self.font_size
        }
        fn pixels_per_inch(&self) -> f64 {
            72.0
        }
        fn percent_reference(&self) -> Option<Length> {
            self.reference
        }
        fn from_parent(&self, _: Option<&str>) -> Result<Value> {
            Ok(Value::Length(Length::from_points(3.0)))
        }
        fn inherited_property_value(&self, _: Option<&str>) -> Result<Value> {
            Ok(Value::Length(Length::from_points(5.0)))
        }
        fn from_nearest_specified_value(&self, _: Option<&str>) -> Result<Value> {
            Ok(Value::Absent)
        }
        fn from_page_master_region(&self, _: Option<&str>) -> Result<Value> {
            Ok(Value::Absent)
        }
        fn from_table_column(&self, _: Option<&str>) -> Result<Value> {
            Ok(Value::Absent)
        }
        fn merge_property_values(&self, _: Option<&str>) -> Result<Value> {
            Ok(Value::Absent)
        }
        fn label_end(&self) -> Result<Value> {
            Ok(Value::Absent)
        }
        fn body_start(&self) -> Result<Value> {
            Ok(Value::Absent)
        }
        fn system_font(&self, _: &str, _: Option<&str>) -> Result<Value> {
            Ok(Value::Absent)
        }
        fn system_color(&self, name: &str) -> Result<Color> {
            Ok(Color::System(name.to_owned()))
        }
    }

    fn eval(input: &str) -> Eval {
        let context = Bare {
            font_size: 10.0,
            reference: None,
        };
        let expr = parse("test", input).expect("parses");
        evaluate(&expr, &context).expect("evaluates")
    }

    /// A property value comes from a document, so a parenthesis chain has to
    /// come back as an error rather than as a stack overflow, which would abort
    /// the process instead of failing the property.
    #[test]
    fn deeply_nested_parentheses_are_rejected_instead_of_overflowing() {
        for levels in [700_usize, 10_000, 100_000] {
            let input = format!("{}1pt{}", "(".repeat(levels), ")".repeat(levels));
            let error = parse("test", &input).expect_err("a chain that deep must be refused");
            assert!(
                error.to_string().contains("nests more than"),
                "unexpected error: {error}"
            );
        }
        assert!(parse("test", &format!("{}1pt", "-".repeat(100_000))).is_err());
        let arguments = format!("{}1pt{}", "round(".repeat(1000), ")".repeat(1000));
        assert!(parse("test", &arguments).is_err());
    }

    #[test]
    fn nesting_just_inside_the_bound_still_parses() {
        let levels = MAX_DEPTH - 8;
        let input = format!("{}1pt{}", "(".repeat(levels), ")".repeat(levels));
        assert_eq!(eval(&input), Eval::length(1.0));
    }

    #[test]
    fn operator_precedence_matches_the_grammar() {
        assert_eq!(eval("2*3 + 4 div 5"), Eval::number(6.8));
        assert_eq!(eval("2 + 3 * 4"), Eval::number(14.0));
        assert_eq!(eval("(2 + 3) * 4"), Eval::number(20.0));
    }

    #[test]
    fn unit_arithmetic_tracks_the_unit_power() {
        assert_eq!(eval("10pt + 2pt"), Eval::length(12.0));
        assert_eq!(eval("2 * 3pt"), Eval::length(6.0));
        assert_eq!(
            eval("3pt * 4pt"),
            Eval::Numeric {
                value: 12.0,
                power: 2
            }
        );
        assert_eq!(eval("1in div 1pt"), Eval::number(72.0));
        assert!(parse("test", "1pt + 1").is_ok());
        let context = Bare {
            font_size: 10.0,
            reference: None,
        };
        let expr = parse("test", "1pt + 1").expect("parses");
        assert!(evaluate(&expr, &context).is_err(), "mixed unit powers");
    }

    #[test]
    fn em_resolves_against_the_current_font_size() {
        assert_eq!(eval("1.5em"), Eval::length(15.0));
    }

    #[test]
    fn mod_follows_the_java_semantics_of_the_note() {
        assert_eq!(eval("5 mod 2"), Eval::number(1.0));
        assert_eq!(eval("5 mod -2"), Eval::number(1.0));
        assert_eq!(eval("-5 mod 2"), Eval::number(-1.0));
        assert_eq!(eval("-5 mod -2"), Eval::number(-1.0));
    }

    #[test]
    fn number_functions() {
        assert_eq!(eval("floor(1.7)"), Eval::number(1.0));
        assert_eq!(eval("ceiling(1.2)"), Eval::number(2.0));
        assert_eq!(eval("round(1.5)"), Eval::number(2.0));
        assert_eq!(eval("round(-1.5)"), Eval::number(-1.0));
        assert_eq!(eval("abs(-3pt)"), Eval::length(3.0));
        assert_eq!(eval("min(3pt, 4pt)"), Eval::length(3.0));
        assert_eq!(eval("max(3pt, 4pt)"), Eval::length(4.0));
        assert_eq!(eval("floor(1.4in div 1.0in) * 1.0in"), Eval::length(72.0));
    }

    #[test]
    fn colour_functions() {
        assert_eq!(eval("rgb(255, 0, 0)"), Eval::Color(Color::rgb(255, 0, 0)));
        assert_eq!(eval("#00ff00"), Eval::Color(Color::rgb(0, 255, 0)));
    }

    #[test]
    fn percentages_resolve_when_a_reference_is_available() {
        let context = Bare {
            font_size: 10.0,
            reference: Some(Length::from_points(200.0)),
        };
        let expr = parse("test", "50% + 5pt").expect("parses");
        assert_eq!(
            evaluate(&expr, &context).expect("evaluates"),
            Eval::length(105.0)
        );
    }

    #[test]
    fn percentages_survive_when_no_reference_is_available() {
        assert_eq!(eval("50%"), Eval::Percent(50.0));
    }

    #[test]
    fn property_value_functions_reach_the_context() {
        assert_eq!(
            eval("from-parent(start-indent)"),
            Eval::Value(Value::Length(Length::from_points(3.0)))
        );
        assert_eq!(
            eval("inherited-property-value()"),
            Eval::Value(Value::Length(Length::from_points(5.0)))
        );
    }

    #[test]
    fn unknown_functions_are_errors() {
        let context = Bare {
            font_size: 10.0,
            reference: None,
        };
        let expr = parse("test", "nonesuch(1)").expect("parses");
        assert!(evaluate(&expr, &context).is_err());
    }
}
