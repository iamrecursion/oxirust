//! Property refinement: turning specified values into computed ones.
//!
//! This module implements [XSL 1.1 §5](https://www.w3.org/TR/xsl11/#prrefine)
//! end to end for one formatting object tree:
//!
//! 1. shorthands are expanded in the precedence order of §5.2;
//! 2. `font-size` is computed first, as §5.9.2 requires, then `writing-mode`
//!    and `color`, on which other properties depend;
//! 3. every applicable property gets a specified value from the stylesheet, from
//!    inheritance (§5.1.4) or from its initial value (§5.1.1), and expressions
//!    are evaluated against the property context of §5.9.1;
//! 4. the corresponding-property rules of §5.3 relate the writing-mode-relative
//!    spellings to the absolute ones, including the `start-indent` and
//!    `end-indent` formulae and the height/width mapping;
//! 5. the trait mappings of §5.4 that refinement can carry out without an area
//!    tree are applied.
//!
//! # What refinement deliberately leaves undone
//!
//! A percentage whose reference value is a property of an *area* — the width of
//! a containing block, say — cannot be resolved before layout. Such percentages
//! survive into the computed value as [`Value::Percent`] (or
//! [`Measure::Percent`](crate::property::Measure::Percent)) unless the caller
//! supplies the reference through [`RefineOptions`]. Likewise `label-end()`
//! yields [`Value::Absent`] when no reference inline-progression-dimension is
//! known, and `system-font()` yields [`Value::Absent`] because font metrics
//! live in the renderer.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::Result;
use crate::fo::FoElement;
use crate::property::corresponding::{FAMILIES, WritingMode, end_indent_terms, start_indent_terms};
use crate::property::datatype::{
    Color, Conditionality, Length, LengthConditional, LengthRange, Measure, Space,
};
use crate::property::expr::EvalContext;
use crate::property::id::PropertyId;
use crate::property::key::PropertyKey;
use crate::property::parse::{
    Specified, absolute_font_size, border_width_keyword, initial_value, parse_specified,
    set_component,
};
use crate::property::shorthand::{expand, precedence};
use crate::property::spec::{Compound, PercentBase};
use crate::property::value::Value;
use crate::tree::{FoTree, NodeId, ResolvedProperties};

/// Knobs for property refinement.
///
/// The defaults describe a document whose geometry is not yet known: no
/// reference area dimensions, `px` treated as `pt`, and the initial `font-size`
/// of `medium` taken as 12pt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefineOptions {
    /// Device resolution used to resolve the `px` unit (XSL 1.1 §5.9.13.1
    /// explicitly permits a fixed conversion factor).
    pub pixels_per_inch: f64,
    /// The length the initial `font-size` of `medium` stands for.
    pub initial_font_size: Length,
    /// The inline-progression-dimension of the reference area percentages are
    /// measured against, when the caller knows it.
    pub reference_inline_progression_dimension: Option<Length>,
    /// The block-progression-dimension of the reference area percentages are
    /// measured against, when the caller knows it.
    pub reference_block_progression_dimension: Option<Length>,
    /// Reject properties that do not apply to the formatting object they are
    /// written on (XSL 1.1 §6).
    pub validate_applicability: bool,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            pixels_per_inch: 72.0,
            initial_font_size: Length::from_points(12.0),
            reference_inline_progression_dimension: None,
            reference_block_progression_dimension: None,
            validate_applicability: true,
        }
    }
}

/// The specified property text of one formatting object, after shorthand
/// expansion.
struct SpecifiedSet {
    values: BTreeMap<PropertyKey, String>,
    direct: BTreeSet<PropertyId>,
    via_shorthand: BTreeSet<PropertyId>,
}

impl SpecifiedSet {
    fn collect(specified: &[(PropertyKey, String)]) -> Result<Self> {
        let mut set = Self {
            values: BTreeMap::new(),
            direct: BTreeSet::new(),
            via_shorthand: BTreeSet::new(),
        };
        let mut ordered: Vec<&(PropertyKey, String)> = specified.iter().collect();
        ordered.sort_by_key(|(key, _)| precedence(key.id));
        for (key, raw) in ordered {
            if key.id.is_shorthand() {
                for (target, value) in expand(key.id, raw)? {
                    set.via_shorthand.insert(target.id);
                    set.values.insert(target, value);
                }
            } else {
                set.direct.insert(key.id);
                set.values.insert(*key, raw.clone());
            }
        }
        Ok(set)
    }

    fn raw(&self, key: PropertyKey) -> Option<&str> {
        self.values.get(&key).map(String::as_str)
    }

    fn components(&self, id: PropertyId) -> impl Iterator<Item = (&'static str, &str)> {
        self.values.iter().filter_map(move |(key, value)| {
            if key.id == id {
                key.component.map(|component| (component, value.as_str()))
            } else {
                None
            }
        })
    }
}

/// The property context an expression is evaluated in.
struct Context<'a> {
    tree: &'a FoTree,
    id: NodeId,
    element: FoElement,
    options: &'a RefineOptions,
    computed: &'a ResolvedProperties,
    font_size: f64,
    parent_font_size: f64,
    property: PropertyId,
}

impl Context<'_> {
    fn parent(&self) -> Option<NodeId> {
        self.tree.parent(self.id)
    }

    fn named(&self, property: Option<&str>) -> PropertyId {
        property
            .and_then(PropertyId::parse)
            .unwrap_or(self.property)
    }

    fn parent_value(&self, property: PropertyId) -> Value {
        match self.parent() {
            Some(parent) => self
                .tree
                .properties(parent)
                .get(property)
                .cloned()
                .unwrap_or_else(|| initial_value(property, self)),
            None => initial_value(property, self),
        }
    }
}

impl EvalContext for Context<'_> {
    fn property_name(&self) -> &str {
        self.property.name()
    }

    fn font_size(&self) -> f64 {
        self.font_size
    }

    fn pixels_per_inch(&self) -> f64 {
        self.options.pixels_per_inch
    }

    fn percent_reference(&self) -> Option<Length> {
        if self.property == PropertyId::FontSize {
            return Some(Length::from_points(self.parent_font_size));
        }
        match self.property.spec().percent_base {
            PercentBase::FontSize => Some(Length::from_points(self.font_size)),
            PercentBase::LineHeight => self
                .computed
                .get(PropertyId::LineHeight)
                .and_then(Value::as_length)
                .or(Some(Length::from_points(self.font_size * 1.2))),
            PercentBase::InheritedValue => self.parent_value(self.property).as_length(),
            PercentBase::ContainingBlockWidth
            | PercentBase::InlineProgressionDimension
            | PercentBase::PaddingRectangle => self.options.reference_inline_progression_dimension,
            PercentBase::ContainingBlockHeight | PercentBase::BlockProgressionDimension => {
                self.options.reference_block_progression_dimension
            }
            _ => None,
        }
    }

    fn from_parent(&self, property: Option<&str>) -> Result<Value> {
        Ok(self.parent_value(self.named(property)))
    }

    fn inherited_property_value(&self, property: Option<&str>) -> Result<Value> {
        let id = self.named(property);
        if !id.is_inherited() {
            return Err(crate::error::FoError::expression(
                self.property_name(),
                format!("`{}` is not an inherited property", id.name()),
            ));
        }
        Ok(self.parent_value(id))
    }

    fn from_nearest_specified_value(&self, property: Option<&str>) -> Result<Value> {
        let id = self.named(property);
        for ancestor in self.tree.ancestors(self.id) {
            if self.tree.node(ancestor).specifies(id)
                && let Some(value) = self.tree.properties(ancestor).get(id)
            {
                return Ok(value.clone());
            }
        }
        Ok(initial_value(id, self))
    }

    fn from_page_master_region(&self, property: Option<&str>) -> Result<Value> {
        let id = self.named(property);
        match self.region_in_force() {
            Some(region) => Ok(self
                .tree
                .properties(region)
                .get(id)
                .cloned()
                .unwrap_or_else(|| initial_value(id, self))),
            None => Ok(initial_value(id, self)),
        }
    }

    fn from_table_column(&self, property: Option<&str>) -> Result<Value> {
        let id = self.named(property);
        match self.table_column_in_force() {
            Some(column) => Ok(self
                .tree
                .properties(column)
                .get(id)
                .cloned()
                .unwrap_or_else(|| initial_value(id, self))),
            None => Ok(initial_value(id, self)),
        }
    }

    fn merge_property_values(&self, property: Option<&str>) -> Result<Value> {
        let id = self.named(property);
        let Some(parent) = self.parent() else {
            return Ok(initial_value(id, self));
        };
        if self.tree.element(parent) != Some(FoElement::MultiProperties) {
            return Ok(initial_value(id, self));
        }
        let mut merged = None;
        for sibling in self.tree.children(parent) {
            if self.tree.element(*sibling) == Some(FoElement::MultiPropertySet)
                && self.tree.node(*sibling).specifies(id)
                && let Some(value) = self.tree.properties(*sibling).get(id)
            {
                merged = Some(value.clone());
            }
        }
        Ok(merged.unwrap_or_else(|| initial_value(id, self)))
    }

    fn label_end(&self) -> Result<Value> {
        // §7.30.11:
        //   label-end() = width of the content-rectangle of the reference-area
        //                 - (provisional-distance-between-starts + start-indent
        //                    + start-intrusion-adjustment
        //                    - provisional-label-separation)
        //                 of the closest ancestor fo:list-block.
        let Some(reference) = self.options.reference_inline_progression_dimension else {
            // The width of the reference area is a layout result; refinement
            // cannot know it, so the value stays open for the formatter.
            return Ok(Value::Absent);
        };
        let distance = self.list_length(PropertyId::ProvisionalDistanceBetweenStarts);
        let separation = self.list_length(PropertyId::ProvisionalLabelSeparation);
        let start_indent = self.list_length(PropertyId::StartIndent);
        Ok(Value::Length(
            reference - (distance + start_indent - separation),
        ))
    }

    fn body_start(&self) -> Result<Value> {
        // §7.30.12: body-start() = start-indent + start-intrusion-adjustment
        //                          + provisional-distance-between-starts
        //                          of the closest ancestor fo:list-block.
        let distance = self.list_length(PropertyId::ProvisionalDistanceBetweenStarts);
        let start_indent = self.list_length(PropertyId::StartIndent);
        Ok(Value::Length(start_indent + distance))
    }

    fn system_font(&self, _font: &str, _property: Option<&str>) -> Result<Value> {
        // Font metrics belong to the renderer; see the module documentation.
        Ok(Value::Absent)
    }

    fn system_color(&self, name: &str) -> Result<Color> {
        Ok(Color::System(name.to_owned()))
    }
}

impl Context<'_> {
    /// A length taken from the closest ancestor `fo:list-block`, as the
    /// `label-end()` and `body-start()` formulae of §7.30.11–12 require.
    ///
    /// When there is no ancestral `fo:list-block`, those sections fall back to
    /// `inherited-property-value()` of the object the function is evaluated on.
    fn list_length(&self, property: PropertyId) -> Length {
        self.tree
            .nearest_ancestor(self.id, &[FoElement::ListBlock])
            .and_then(|list| self.tree.properties(list).get(property).cloned())
            .unwrap_or_else(|| self.parent_value(property))
            .as_length()
            .unwrap_or(Length::ZERO)
    }

    /// The region definition that generated the region reference area this
    /// object's areas end up in, for `from-page-master-region()`.
    fn region_in_force(&self) -> Option<NodeId> {
        let flow = self
            .tree
            .nearest_ancestor(self.id, &[FoElement::Flow, FoElement::StaticContent])
            .or(
                if matches!(self.element, FoElement::Flow | FoElement::StaticContent) {
                    Some(self.id)
                } else {
                    None
                },
            )?;
        let flow_name = self
            .tree
            .properties(flow)
            .get(PropertyId::FlowName)
            .and_then(|value| value.as_str())
            .map(str::to_owned)?;
        let sequence = self
            .tree
            .nearest_ancestor(self.id, &[FoElement::PageSequence])?;
        let master_reference = self
            .tree
            .properties(sequence)
            .get(PropertyId::MasterReference)
            .and_then(|value| value.as_str())
            .map(str::to_owned)?;
        let masters = self.layout_master_set()?;
        let page_master = self.resolve_page_master(masters, &master_reference)?;
        self.tree.children(page_master).iter().copied().find(|id| {
            self.tree
                .properties(*id)
                .get(PropertyId::RegionName)
                .and_then(|value| value.as_str())
                == Some(flow_name.as_str())
        })
    }

    fn layout_master_set(&self) -> Option<NodeId> {
        let root = self.tree.root();
        self.tree
            .children(root)
            .iter()
            .copied()
            .find(|id| self.tree.element(*id) == Some(FoElement::LayoutMasterSet))
    }

    fn master_named(&self, masters: NodeId, name: &str) -> Option<NodeId> {
        self.tree.children(masters).iter().copied().find(|id| {
            self.tree
                .properties(*id)
                .get(PropertyId::MasterName)
                .and_then(|value| value.as_str())
                == Some(name)
        })
    }

    /// Follows a `master-reference` to a `fo:simple-page-master`, stepping
    /// through an `fo:page-sequence-master` when necessary.
    fn resolve_page_master(&self, masters: NodeId, name: &str) -> Option<NodeId> {
        let master = self.master_named(masters, name)?;
        match self.tree.element(master) {
            Some(FoElement::SimplePageMaster) => Some(master),
            Some(FoElement::PageSequenceMaster) => {
                let reference = self.tree.children(master).iter().copied().find_map(|id| {
                    self.tree
                        .properties(id)
                        .get(PropertyId::MasterReference)
                        .and_then(|value| value.as_str())
                        .map(str::to_owned)
                        .or_else(|| {
                            self.tree.children(id).iter().copied().find_map(|nested| {
                                self.tree
                                    .properties(nested)
                                    .get(PropertyId::MasterReference)
                                    .and_then(|value| value.as_str())
                                    .map(str::to_owned)
                            })
                        })
                })?;
                let target = self.master_named(masters, &reference)?;
                if self.tree.element(target) == Some(FoElement::SimplePageMaster) {
                    Some(target)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// The `fo:table-column` that governs this cell, for `from-table-column()`.
    fn table_column_in_force(&self) -> Option<NodeId> {
        let cell = if self.element == FoElement::TableCell {
            self.id
        } else {
            self.tree
                .nearest_ancestor(self.id, &[FoElement::TableCell])?
        };
        let table = self.tree.nearest_ancestor(cell, &[FoElement::Table])?;
        let column_number = self
            .tree
            .properties(cell)
            .get(PropertyId::ColumnNumber)
            .and_then(Value::as_integer)?;
        let span = self
            .tree
            .properties(cell)
            .get(PropertyId::NumberColumnsSpanned)
            .and_then(Value::as_integer)
            .unwrap_or(1);
        let columns: Vec<NodeId> = self
            .tree
            .children(table)
            .iter()
            .copied()
            .filter(|id| self.tree.element(*id) == Some(FoElement::TableColumn))
            .collect();
        let matching = |wanted_span: i64| {
            columns.iter().copied().find(|id| {
                let properties = self.tree.properties(*id);
                properties
                    .get(PropertyId::ColumnNumber)
                    .and_then(Value::as_integer)
                    == Some(column_number)
                    && properties
                        .get(PropertyId::NumberColumnsSpanned)
                        .and_then(Value::as_integer)
                        .unwrap_or(1)
                        == wanted_span
            })
        };
        matching(span).or_else(|| matching(1))
    }
}

/// Refines every formatting object of `tree` in document order.
pub(crate) fn refine(tree: &mut FoTree, options: &RefineOptions) -> Result<()> {
    for index in 0..tree.element_order.len() {
        let id = tree.element_order[index];
        let properties = refine_node(tree, id, options)?;
        tree.nodes[id.index()].properties = properties;
    }
    Ok(())
}

fn font_size_of(tree: &FoTree, id: Option<NodeId>, options: &RefineOptions) -> f64 {
    match id {
        Some(parent) => tree
            .properties(parent)
            .get(PropertyId::FontSize)
            .and_then(Value::as_length)
            .unwrap_or(options.initial_font_size)
            .points(),
        None => options.initial_font_size.points(),
    }
}

/// Everything about a formatting object that stays fixed while its properties
/// are computed one after another.
struct NodeContext<'a> {
    tree: &'a FoTree,
    id: NodeId,
    element: FoElement,
    options: &'a RefineOptions,
    specified: SpecifiedSet,
    parent_font_size: f64,
}

fn refine_node(tree: &FoTree, id: NodeId, options: &RefineOptions) -> Result<ResolvedProperties> {
    let node = tree.node(id);
    let Some(element) = node.element() else {
        return Ok(ResolvedProperties::default());
    };
    let parent_font_size = font_size_of(tree, tree.parent(id), options);
    let node_context = NodeContext {
        tree,
        id,
        element,
        options,
        specified: SpecifiedSet::collect(&node.specified)?,
        parent_font_size,
    };

    let mut computed = ResolvedProperties::default();
    let mut font_size = parent_font_size;

    // §5.9.2: `font-size` is evaluated before every other property.
    let font_size_value = compute_one(&node_context, &computed, font_size, PropertyId::FontSize)?;
    let font_size_value = normalize_font_size(font_size_value, parent_font_size);
    if let Some(length) = font_size_value.as_length() {
        font_size = length.points();
    }
    computed.set(PropertyId::FontSize, font_size_value);

    // `color` and `writing-mode` come next: the border colours take their
    // initial value from `color`, and the corresponding-property mapping needs
    // the writing mode.
    for early in [PropertyId::Color, PropertyId::WritingMode] {
        let value = compute_one(&node_context, &computed, font_size, early)?;
        computed.set(early, value);
    }

    let writing_mode = computed
        .get(PropertyId::WritingMode)
        .and_then(Value::as_keyword)
        .and_then(WritingMode::parse)
        .unwrap_or_default();

    for property in element.applicable_properties() {
        if property.is_shorthand()
            || matches!(
                *property,
                PropertyId::FontSize | PropertyId::Color | PropertyId::WritingMode
            )
        {
            continue;
        }
        let value = compute_one(&node_context, &computed, font_size, *property)?;
        computed.set(*property, value);
    }

    let specified = node_context.specified;

    // The border widths must be lengths before the corresponding pass runs:
    // the `start-indent` formula of §5.3.2 adds them.
    normalize_border_widths(&mut computed);
    apply_corresponding(tree, id, element, writing_mode, &specified, &mut computed);
    normalize_border_widths(&mut computed);
    apply_trait_mappings(tree, id, element, writing_mode, &specified, &mut computed);
    Ok(computed)
}

/// `larger`, `smaller` and the `<absolute-size>` keywords of §7.9.4.
fn normalize_font_size(value: Value, parent_font_size: f64) -> Value {
    match value.as_keyword() {
        Some("larger") => Value::Length(Length::from_points(parent_font_size * 1.2)),
        Some("smaller") => Value::Length(Length::from_points(parent_font_size / 1.2)),
        Some(keyword) => match absolute_font_size(keyword) {
            Some(length) => Value::Length(length),
            None => value,
        },
        None => value,
    }
}

/// Computes one property against the partially refined `computed` set.
fn compute_one(
    node: &NodeContext<'_>,
    computed: &ResolvedProperties,
    font_size: f64,
    property: PropertyId,
) -> Result<Value> {
    let NodeContext {
        tree,
        id,
        element,
        options,
        specified,
        parent_font_size,
    } = node;
    let (tree, id, element, options, parent_font_size) =
        (*tree, *id, *element, *options, *parent_font_size);
    let context = Context {
        tree,
        id,
        element,
        options,
        computed,
        font_size,
        parent_font_size,
        property,
    };
    let key = PropertyKey::whole(property);
    let mut value = match specified.raw(key) {
        Some(raw) => match parse_specified(key, raw, &context)? {
            Specified::Inherit => context.parent_value(property),
            Specified::Value(value) => value,
        },
        None => {
            if property.is_inherited() && tree.parent(id).is_some() {
                context.parent_value(property)
            } else {
                initial_for(property, element, &context)
            }
        }
    };

    // Complete-form component specifications override the short form (§5.11).
    let compound = property.compound();
    if compound != Compound::None {
        let mut has_components = false;
        for (component, raw) in specified.components(property) {
            if !has_components {
                value = ensure_compound(value, compound);
                has_components = true;
            }
            let component_key =
                PropertyKey::component(property, component).unwrap_or(PropertyKey::whole(property));
            if let Specified::Value(component_value) =
                parse_specified(component_key, raw, &context)?
            {
                set_component(&mut value, compound, component, &component_value);
            }
        }
    }
    Ok(value)
}

/// Resolves the initial-value markers of the property table.
fn initial_for(property: PropertyId, element: FoElement, context: &Context<'_>) -> Value {
    match property.spec().initial {
        "#current-color" => context
            .computed
            .get(PropertyId::Color)
            .cloned()
            .unwrap_or(Value::Color(Color::BLACK)),
        "#fo-precedence" => Value::Integer(match element {
            FoElement::Table => 6,
            FoElement::TableCell => 5,
            FoElement::TableColumn => 4,
            FoElement::TableRow => 3,
            FoElement::TableBody => 2,
            FoElement::TableHeader => 1,
            _ => 0,
        }),
        // §5.4.2: the initial `column-number` is derived from the position of
        // the object, which `apply_trait_mappings` fills in.
        "#column-number" => Value::Absent,
        _ => initial_value(property, context),
    }
}

/// Turns a scalar value into the compound datatype so that a `.component`
/// specification has something to write into.
fn ensure_compound(value: Value, compound: Compound) -> Value {
    match (compound, &value) {
        (Compound::Space, Value::Space(_))
        | (Compound::Keep, Value::Keep(_))
        | (Compound::LengthRange, Value::LengthRange(_))
        | (Compound::LengthConditional, Value::LengthConditional(_))
        | (Compound::LengthBpIp, Value::LengthBpIp(_)) => value,
        (Compound::Space, _) => {
            Value::Space(value.as_measure().map_or_else(Space::zero, Space::uniform))
        }
        (Compound::Keep, _) => Value::Keep(crate::property::datatype::Keep::default()),
        (Compound::LengthRange, _) => Value::LengthRange(
            value
                .as_measure()
                .map_or_else(LengthRange::auto, LengthRange::uniform),
        ),
        (Compound::LengthConditional, _) => Value::LengthConditional(LengthConditional::new(
            value.as_measure().unwrap_or(Measure::ZERO),
        )),
        (Compound::LengthBpIp, _) => {
            Value::LengthBpIp(crate::property::datatype::LengthBpIp::uniform(
                value.as_measure().unwrap_or(Measure::ZERO),
            ))
        }
        (Compound::None, _) => value,
    }
}

// -------------------------------------------------------- corresponding pass

fn measure_of(value: &Value) -> Option<Measure> {
    match value {
        Value::LengthConditional(conditional) => Some(conditional.length),
        Value::Space(space) => Some(space.optimum),
        Value::LengthRange(range) => Some(range.optimum),
        other => other.as_measure(),
    }
}

/// Rewrites `value` into the datatype `target` expects.
fn convert_for(target: PropertyId, value: &Value, from_margin: bool) -> Value {
    match target.compound() {
        Compound::LengthConditional => {
            let conditionality = match value {
                Value::LengthConditional(conditional) => conditional.conditionality,
                _ => Conditionality::Discard,
            };
            Value::LengthConditional(LengthConditional {
                length: measure_of(value).unwrap_or(Measure::ZERO),
                conditionality,
            })
        }
        Compound::Space => {
            let measure = measure_of(value).unwrap_or(Measure::ZERO);
            // §5.3.2: a margin of `auto` contributes 0pt and the conditionality
            // of a space derived from a margin is `retain`.
            let measure = if matches!(measure, Measure::Auto) {
                Measure::ZERO
            } else {
                measure
            };
            match value {
                Value::Space(space) if !from_margin => Value::Space(*space),
                _ => Value::Space(Space {
                    conditionality: if from_margin {
                        Conditionality::Retain
                    } else {
                        Conditionality::Discard
                    },
                    ..Space::uniform(measure)
                }),
            }
        }
        Compound::LengthRange => Value::LengthRange(LengthRange::uniform(
            measure_of(value).unwrap_or(Measure::Auto),
        )),
        _ => match value {
            Value::LengthConditional(conditional) => match conditional.length {
                Measure::Length(length) => Value::Length(length),
                Measure::Percent(percent) => Value::Percent(percent),
                Measure::Auto => Value::keyword("auto"),
            },
            Value::Space(space) => match space.optimum {
                Measure::Length(length) => Value::Length(length),
                Measure::Percent(percent) => Value::Percent(percent),
                Measure::Auto => Value::keyword("auto"),
            },
            other => other.clone(),
        },
    }
}

fn apply_corresponding(
    tree: &FoTree,
    id: NodeId,
    element: FoElement,
    mode: WritingMode,
    specified: &SpecifiedSet,
    computed: &mut ResolvedProperties,
) {
    // §5.3.1 and §5.3.2: the two-variant families.
    for family in &FAMILIES {
        let from_margin = family.absolute[0] == PropertyId::MarginTop;
        for (relative, absolute) in family.pairs(mode) {
            // Only relate the two spellings when both apply to this object;
            // `fo:block`, for instance, has `space-before` but no `space-start`.
            if computed.get(relative).is_none() || computed.get(absolute).is_none() {
                continue;
            }
            let absolute_direct = specified.direct.contains(&absolute);
            let relative_direct = specified.direct.contains(&relative);
            let absolute_shorthand = specified.via_shorthand.contains(&absolute);
            if absolute_direct {
                if let Some(value) = computed.get(absolute).cloned() {
                    computed.set(relative, convert_for(relative, &value, from_margin));
                }
            } else if relative_direct {
                if let Some(value) = computed.get(relative).cloned() {
                    computed.set(absolute, convert_for(absolute, &value, false));
                }
            } else if absolute_shorthand {
                if let Some(value) = computed.get(absolute).cloned() {
                    computed.set(relative, convert_for(relative, &value, from_margin));
                }
            }
        }
    }

    apply_indents(tree, id, element, mode, specified, computed);
    apply_dimensions(mode, specified, computed);
}

fn length_of(computed: &ResolvedProperties, property: PropertyId) -> Option<Length> {
    match computed.get(property)? {
        Value::Length(length) => Some(*length),
        Value::LengthConditional(conditional) => conditional.length.length(),
        Value::Space(space) => space.optimum.length(),
        Value::Keyword(keyword) if &**keyword == "auto" => Some(Length::ZERO),
        _ => None,
    }
}

fn apply_indents(
    tree: &FoTree,
    id: NodeId,
    element: FoElement,
    mode: WritingMode,
    specified: &SpecifiedSet,
    computed: &mut ResolvedProperties,
) {
    let generates_reference_area = element.generates_reference_area();
    for (indent, terms) in [
        (PropertyId::StartIndent, start_indent_terms(mode)),
        (PropertyId::EndIndent, end_indent_terms(mode)),
    ] {
        if computed.get(indent).is_none() {
            continue;
        }
        let inherited = tree
            .parent(id)
            .and_then(|parent| {
                tree.properties(parent)
                    .get(indent)
                    .and_then(Value::as_length)
            })
            .unwrap_or(Length::ZERO);
        let padding = length_of(computed, terms.padding).unwrap_or(Length::ZERO);
        let border = length_of(computed, terms.border_width).unwrap_or(Length::ZERO);

        // §5.3.2: the margin drives the indent when the stylesheet gave one —
        // whether directly or through the `margin` shorthand — unless the
        // indent itself was named explicitly, which is more specific.
        let margin_specified = specified.direct.contains(&terms.margin)
            || specified.via_shorthand.contains(&terms.margin);
        if margin_specified && !specified.direct.contains(&indent) {
            let Some(margin) = length_of(computed, terms.margin) else {
                continue;
            };
            let value = if generates_reference_area {
                margin + padding + border
            } else {
                inherited + margin + padding + border
            };
            computed.set(indent, Value::Length(value));
        } else if computed.get(terms.margin).is_some()
            && let Some(current) = computed.get(indent).and_then(Value::as_length)
        {
            // The margin is derived from the indent instead — but only when the
            // margin property applies to this formatting object at all.
            let margin = current - inherited - padding - border;
            computed.set(terms.margin, Value::Length(margin));
        }
    }
}

/// §5.3.3: `height`/`width` map onto the block- and inline-progression
/// dimensions according to the writing mode.
fn apply_dimensions(
    mode: WritingMode,
    specified: &SpecifiedSet,
    computed: &mut ResolvedProperties,
) {
    let vertical = mode.block_progression_is_vertical();
    let groups: [(PropertyId, PropertyId, PropertyId, PropertyId); 2] = [
        (
            PropertyId::Height,
            PropertyId::MinHeight,
            PropertyId::MaxHeight,
            if vertical {
                PropertyId::BlockProgressionDimension
            } else {
                PropertyId::InlineProgressionDimension
            },
        ),
        (
            PropertyId::Width,
            PropertyId::MinWidth,
            PropertyId::MaxWidth,
            if vertical {
                PropertyId::InlineProgressionDimension
            } else {
                PropertyId::BlockProgressionDimension
            },
        ),
    ];
    for (size, minimum, maximum, target) in groups {
        let any_specified = [size, minimum, maximum]
            .iter()
            .any(|property| specified.direct.contains(property));
        if !any_specified || computed.get(target).is_none() {
            continue;
        }
        let mut range = if specified.direct.contains(&size) {
            let measure = computed
                .get(size)
                .and_then(Value::as_measure)
                .unwrap_or(Measure::Auto);
            LengthRange::uniform(measure)
        } else {
            LengthRange::auto()
        };
        if specified.direct.contains(&minimum)
            && let Some(measure) = computed.get(minimum).and_then(Value::as_measure)
        {
            range.minimum = measure;
        }
        if specified.direct.contains(&maximum) {
            range.maximum = match computed.get(maximum) {
                Some(Value::Keyword(keyword)) if &**keyword == "none" => Measure::Auto,
                Some(value) => value.as_measure().unwrap_or(Measure::Auto),
                None => Measure::Auto,
            };
        }
        computed.set(target, Value::LengthRange(range));
    }
}

// ------------------------------------------------------------ trait mappings

const BORDER_WIDTH_AND_STYLE: [(PropertyId, PropertyId); 8] = [
    (PropertyId::BorderBeforeWidth, PropertyId::BorderBeforeStyle),
    (PropertyId::BorderAfterWidth, PropertyId::BorderAfterStyle),
    (PropertyId::BorderStartWidth, PropertyId::BorderStartStyle),
    (PropertyId::BorderEndWidth, PropertyId::BorderEndStyle),
    (PropertyId::BorderTopWidth, PropertyId::BorderTopStyle),
    (PropertyId::BorderBottomWidth, PropertyId::BorderBottomStyle),
    (PropertyId::BorderLeftWidth, PropertyId::BorderLeftStyle),
    (PropertyId::BorderRightWidth, PropertyId::BorderRightStyle),
];

/// §7.8.9 and §5.1.2: the border width keywords become lengths, and a width
/// whose style is `none` is forced to zero.
fn normalize_border_widths(computed: &mut ResolvedProperties) {
    for (width, _) in BORDER_WIDTH_AND_STYLE {
        if let Some(keyword) = computed.get(width).and_then(Value::as_keyword)
            && let Some(length) = border_width_keyword(keyword)
        {
            let value = if width.compound() == Compound::LengthConditional {
                Value::LengthConditional(LengthConditional::new(Measure::Length(length)))
            } else {
                Value::Length(length)
            };
            computed.set(width, value);
        }
    }

    for (width, style) in BORDER_WIDTH_AND_STYLE {
        let none = matches!(
            computed.get(style).and_then(Value::as_keyword),
            Some("none" | "hidden")
        );
        if none && computed.get(width).is_some() {
            let zero = if width.compound() == Compound::LengthConditional {
                Value::LengthConditional(LengthConditional::new(Measure::ZERO))
            } else {
                Value::Length(Length::ZERO)
            };
            computed.set(width, zero);
        }
    }
}

fn apply_trait_mappings(
    tree: &FoTree,
    id: NodeId,
    element: FoElement,
    mode: WritingMode,
    specified: &SpecifiedSet,
    computed: &mut ResolvedProperties,
) {
    // §5.4.3 and §5.4.4: `left` and `right` become writing-mode relative.
    for property in [PropertyId::TextAlign, PropertyId::TextAlignLast] {
        let Some(keyword) = computed.get(property).and_then(Value::as_keyword) else {
            continue;
        };
        let relative = match keyword {
            "left" => Some(mode.relative_edge(crate::property::corresponding::AbsoluteEdge::Left)),
            "right" => {
                Some(mode.relative_edge(crate::property::corresponding::AbsoluteEdge::Right))
            }
            _ => None,
        };
        if let Some(edge) = relative {
            use crate::property::corresponding::RelativeEdge;
            let name = match edge {
                RelativeEdge::Start => "start",
                RelativeEdge::End => "end",
                RelativeEdge::Before | RelativeEdge::After => continue,
            };
            computed.set(property, Value::keyword(name));
        }
    }

    // §5.4.1: the background position keywords become lengths.
    for (property, table) in [
        (
            PropertyId::BackgroundPositionHorizontal,
            [("left", 0.0), ("center", 50.0), ("right", 100.0)],
        ),
        (
            PropertyId::BackgroundPositionVertical,
            [("top", 0.0), ("center", 50.0), ("bottom", 100.0)],
        ),
    ] {
        if let Some(keyword) = computed.get(property).and_then(Value::as_keyword)
            && let Some((_, percent)) = table.iter().find(|(name, _)| *name == keyword)
        {
            computed.set(property, Value::Percent(*percent));
        }
    }

    // §5.4.5: `z-index` accumulates down the tree.
    if let Some(own) = computed.get(PropertyId::ZIndex).and_then(Value::as_integer) {
        let parent = tree
            .parent(id)
            .and_then(|parent| {
                tree.properties(parent)
                    .get(PropertyId::ZIndex)
                    .and_then(Value::as_integer)
            })
            .unwrap_or(0);
        computed.set(PropertyId::ZIndex, Value::Integer(own + parent));
    }

    // §5.4.2: the initial `column-number` follows from the object's position.
    if matches!(element, FoElement::TableColumn | FoElement::TableCell)
        && !specified.direct.contains(&PropertyId::ColumnNumber)
    {
        computed.set(
            PropertyId::ColumnNumber,
            Value::Integer(implied_column_number(tree, id, element)),
        );
    }
}

fn implied_column_number(tree: &FoTree, id: NodeId, element: FoElement) -> i64 {
    let Some(parent) = tree.parent(id) else {
        return 1;
    };
    let mut number = 1i64;
    for sibling in tree.children(parent) {
        if *sibling == id {
            break;
        }
        if tree.element(*sibling) != Some(element) {
            continue;
        }
        let properties = tree.properties(*sibling);
        let span = if element == FoElement::TableColumn {
            properties
                .get(PropertyId::NumberColumnsRepeated)
                .and_then(Value::as_integer)
                .unwrap_or(1)
        } else {
            properties
                .get(PropertyId::NumberColumnsSpanned)
                .and_then(Value::as_integer)
                .unwrap_or(1)
        };
        number += span.max(1);
    }
    number
}
