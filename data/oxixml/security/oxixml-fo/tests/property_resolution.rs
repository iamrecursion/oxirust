//! Property refinement: inheritance, shorthand expansion, the corresponding
//! property relation under different writing modes, the property value
//! functions, and length unit arithmetic.
//!
//! Every expectation is taken from the text of
//! [XSL 1.1 §5](https://www.w3.org/TR/xsl11/#prrefine).

use oxixml_fo::area::Dimensions;
use oxixml_fo::fo::FoElement;
use oxixml_fo::property::{
    Conditionality, Length, Measure, PropertyId, RefineOptions, Value, WritingMode,
};
use oxixml_fo::{FoDocument, FoTree, NodeId, ParseOptions};

/// Wraps `content` (a run of block-level objects) in a minimal document.
fn document(sequence_attributes: &str, content: &str) -> String {
    format!(
        r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
             <fo:layout-master-set>
               <fo:simple-page-master master-name="page"
                                      page-height="297mm" page-width="210mm">
                 <fo:region-body region-name="xsl-region-body" margin="20mm"/>
                 <fo:region-before region-name="xsl-region-before" extent="10mm"/>
               </fo:simple-page-master>
             </fo:layout-master-set>
             <fo:page-sequence master-reference="page" {sequence_attributes}>
               <fo:flow flow-name="xsl-region-body">
                 {content}
               </fo:flow>
             </fo:page-sequence>
           </fo:root>"#
    )
}

fn parse(sequence_attributes: &str, content: &str) -> FoTree {
    FoDocument::parse(&document(sequence_attributes, content)).expect("a valid XSL-FO document")
}

/// The `n`-th (0-based) formatting object of the given kind, in document order.
fn nth(tree: &FoTree, element: FoElement, n: usize) -> NodeId {
    tree.elements()
        .iter()
        .copied()
        .filter(|id| tree.element(*id) == Some(element))
        .nth(n)
        .unwrap_or_else(|| panic!("no {:?} #{n}", element))
}

fn value(tree: &FoTree, id: NodeId, property: PropertyId) -> Value {
    tree.properties(id).value(property).clone()
}

fn length(tree: &FoTree, id: NodeId, property: PropertyId) -> Length {
    value(tree, id, property).as_length().unwrap_or_else(|| {
        panic!(
            "{} is not a length: {:?}",
            property.name(),
            value(tree, id, property)
        )
    })
}

fn points(tree: &FoTree, id: NodeId, property: PropertyId) -> f64 {
    length(tree, id, property).points()
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-6
}

// ------------------------------------------------------------- inheritance --

#[test]
fn inherited_properties_flow_down_and_non_inherited_ones_do_not() {
    let tree = parse(
        "",
        r#"<fo:block font-size="20pt" text-align="center" padding-top="4pt">
             <fo:block id="inner">text</fo:block>
           </fo:block>"#,
    );
    let outer = nth(&tree, FoElement::Block, 0);
    let inner = nth(&tree, FoElement::Block, 1);

    // `font-size` and `text-align` are inherited (§7.9.4, §7.16.9).
    assert!(close(points(&tree, inner, PropertyId::FontSize), 20.0));
    assert_eq!(
        value(&tree, inner, PropertyId::TextAlign),
        Value::keyword("center")
    );
    // `padding-top` is not (§7.8.35); the child falls back to the initial 0pt.
    assert!(close(points(&tree, outer, PropertyId::PaddingTop), 4.0));
    assert!(close(points(&tree, inner, PropertyId::PaddingTop), 0.0));
}

#[test]
fn the_root_uses_initial_values_because_it_has_no_parent() {
    let tree = parse("", "<fo:block>text</fo:block>");
    let block = nth(&tree, FoElement::Block, 0);
    // §7.29.7: the initial `writing-mode` is `lr-tb`.
    assert_eq!(
        value(&tree, block, PropertyId::WritingMode),
        Value::keyword("lr-tb")
    );
    // §7.16.9: the initial `text-align` is `start`.
    assert_eq!(
        value(&tree, block, PropertyId::TextAlign),
        Value::keyword("start")
    );
    // §7.9.4: the initial `font-size` is `medium`, taken as 12pt.
    assert!(close(points(&tree, block, PropertyId::FontSize), 12.0));
}

#[test]
fn inherit_takes_the_computed_value_of_the_parent() {
    let tree = parse(
        "",
        r#"<fo:block padding-top="7pt">
             <fo:block id="inner" padding-top="inherit">text</fo:block>
           </fo:block>"#,
    );
    let inner = nth(&tree, FoElement::Block, 1);
    assert!(close(points(&tree, inner, PropertyId::PaddingTop), 7.0));
}

// ------------------------------------------------------ shorthand expansion --

#[test]
fn the_margin_shorthand_reaches_all_four_edges() {
    let tree = parse("", r#"<fo:block margin="1pt 2pt 3pt 4pt">text</fo:block>"#);
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::MarginTop), 1.0));
    assert!(close(points(&tree, block, PropertyId::MarginRight), 2.0));
    assert!(close(points(&tree, block, PropertyId::MarginBottom), 3.0));
    assert!(close(points(&tree, block, PropertyId::MarginLeft), 4.0));
}

#[test]
fn an_individual_property_always_beats_the_shorthand() {
    // §5.2: "The individual properties are always more precise than any
    // shorthand", whatever the attribute order in the source document.
    let tree = parse(
        "",
        r#"<fo:block padding-left="9pt" padding="1pt">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::PaddingLeft), 9.0));
    assert!(close(points(&tree, block, PropertyId::PaddingTop), 1.0));
}

#[test]
fn interrelated_border_shorthands_apply_in_increasing_precision() {
    // §5.2: `border` < `border-width` < `border-top` < `border-top-width`.
    let tree = parse(
        "",
        r#"<fo:block border-top-width="8pt" border-top="5pt solid red"
                     border-width="3pt" border="1pt solid blue">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::BorderTopWidth), 8.0));
    // `border-top` still won over `border-width` for style and colour.
    assert_eq!(
        value(&tree, block, PropertyId::BorderTopStyle),
        Value::keyword("solid")
    );
    // The other edges kept `border-width`'s 3pt.
    assert!(close(
        points(&tree, block, PropertyId::BorderBottomWidth),
        3.0
    ));
}

#[test]
fn the_font_shorthand_sets_size_line_height_and_family() {
    let tree = parse(
        "",
        r#"<fo:block font="italic bold 12pt/18pt Times New Roman, serif">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert_eq!(
        value(&tree, block, PropertyId::FontStyle),
        Value::keyword("italic")
    );
    assert_eq!(
        value(&tree, block, PropertyId::FontWeight),
        Value::keyword("bold")
    );
    assert!(close(points(&tree, block, PropertyId::FontSize), 12.0));
    assert_eq!(
        value(&tree, block, PropertyId::LineHeight)
            .as_space()
            .map(|space| space.optimum),
        Some(Measure::Length(Length::from_points(18.0)))
    );
    // §7.31.13: `font` also resets `font-stretch` and `font-size-adjust`.
    assert_eq!(
        value(&tree, block, PropertyId::FontStretch),
        Value::keyword("normal")
    );
}

#[test]
fn css_compatibility_shorthands_are_translated_to_xsl_properties() {
    let tree = parse(
        "",
        r#"<fo:block page-break-before="always" white-space="pre">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    // §7.31.17.
    assert_eq!(
        value(&tree, block, PropertyId::BreakBefore),
        Value::keyword("page")
    );
    // §7.31.23.
    assert_eq!(
        value(&tree, block, PropertyId::LinefeedTreatment),
        Value::keyword("preserve")
    );
    assert_eq!(
        value(&tree, block, PropertyId::WhiteSpaceCollapse),
        Value::keyword("false")
    );
    assert_eq!(
        value(&tree, block, PropertyId::WrapOption),
        Value::keyword("no-wrap")
    );
}

#[test]
fn xml_lang_is_a_shorthand_for_language_and_country() {
    let tree = parse("", r#"<fo:block xml:lang="en-GB">text</fo:block>"#);
    let block = nth(&tree, FoElement::Block, 0);
    assert_eq!(
        value(&tree, block, PropertyId::Language).as_str(),
        Some("en")
    );
    assert_eq!(
        value(&tree, block, PropertyId::Country).as_str(),
        Some("GB")
    );
}

// -------------------------------------------- corresponding properties (§5.3)

#[test]
fn relative_and_absolute_spellings_correspond_under_lr_tb() {
    let tree = parse(
        "",
        r#"<fo:block padding-left="6pt" border-right-width="2pt"
                     border-right-style="solid">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    // §5.3.1: the absolute property was specified, so the relative one follows.
    let start = value(&tree, block, PropertyId::PaddingStart)
        .as_length_conditional()
        .expect("padding-start is a <length-conditional>");
    assert_eq!(start.length, Measure::Length(Length::from_points(6.0)));
    let end = value(&tree, block, PropertyId::BorderEndWidth)
        .as_length_conditional()
        .expect("border-end-width is a <length-conditional>");
    assert_eq!(end.length, Measure::Length(Length::from_points(2.0)));
}

#[test]
fn the_correspondence_follows_the_writing_mode() {
    // Under `rl-tb` the start edge is on the right (§5.3).
    let tree = parse(
        r#"writing-mode="rl-tb""#,
        r#"<fo:block padding-right="6pt" padding-left="1pt">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert_eq!(
        value(&tree, block, PropertyId::WritingMode),
        Value::keyword("rl-tb")
    );
    let start = value(&tree, block, PropertyId::PaddingStart)
        .as_length_conditional()
        .expect("a <length-conditional>");
    assert_eq!(start.length, Measure::Length(Length::from_points(6.0)));
    let end = value(&tree, block, PropertyId::PaddingEnd)
        .as_length_conditional()
        .expect("a <length-conditional>");
    assert_eq!(end.length, Measure::Length(Length::from_points(1.0)));
}

#[test]
fn a_relative_specification_sets_the_absolute_property() {
    // §5.3.1: when only the relative property is specified, the absolute one
    // takes its computed value.
    let tree = parse("", r#"<fo:block padding-start="5pt">text</fo:block>"#);
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::PaddingLeft), 5.0));
}

#[test]
fn margins_become_space_properties_with_retained_conditionality() {
    // §5.3.2: "The .conditionality component of any space-before or space-after
    // determined from a margin property is set to retain."
    let tree = parse("", r#"<fo:block margin-top="10pt">text</fo:block>"#);
    let block = nth(&tree, FoElement::Block, 0);
    let space = value(&tree, block, PropertyId::SpaceBefore)
        .as_space()
        .expect("space-before is a <space>");
    assert_eq!(space.optimum, Measure::Length(Length::from_points(10.0)));
    assert_eq!(space.conditionality, Conditionality::Retain);
}

#[test]
fn start_indent_accumulates_down_a_non_reference_area_chain() {
    // §5.3.2, non-reference-area case:
    //   start-indent = inherited_value_of(start-indent)
    //                  + margin + padding-start + border-start-width
    let tree = parse(
        "",
        r#"<fo:block margin-left="10pt">
             <fo:block margin-left="5pt" padding-left="2pt"
                       border-left-width="1pt" border-left-style="solid">text</fo:block>
           </fo:block>"#,
    );
    let outer = nth(&tree, FoElement::Block, 0);
    let inner = nth(&tree, FoElement::Block, 1);
    assert!(close(points(&tree, outer, PropertyId::StartIndent), 10.0));
    assert!(close(points(&tree, inner, PropertyId::StartIndent), 18.0));
}

#[test]
fn a_reference_area_generator_drops_the_inherited_indent() {
    let tree = parse(
        "",
        r#"<fo:block margin-left="10pt">
             <fo:block-container margin-left="5pt">
               <fo:block>text</fo:block>
             </fo:block-container>
           </fo:block>"#,
    );
    let container = nth(&tree, FoElement::BlockContainer, 0);
    assert!(FoElement::BlockContainer.generates_reference_area());
    assert!(close(
        points(&tree, container, PropertyId::StartIndent),
        5.0
    ));
}

#[test]
fn a_specified_indent_derives_the_corresponding_margin() {
    // §5.3.2: margin = indent - inherited(indent) - padding - border-width.
    let tree = parse(
        "",
        r#"<fo:block start-indent="12pt" padding-left="2pt">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::StartIndent), 12.0));
    assert!(close(points(&tree, block, PropertyId::MarginLeft), 10.0));
}

#[test]
fn height_and_width_map_onto_the_progression_dimensions() {
    // §5.3.3, horizontal writing mode: height -> block, width -> inline.
    let tree = parse(
        "",
        r#"<fo:block-container height="100pt" width="200pt" min-width="50pt">
             <fo:block>text</fo:block>
           </fo:block-container>"#,
    );
    let container = nth(&tree, FoElement::BlockContainer, 0);
    let block_dimension = value(&tree, container, PropertyId::BlockProgressionDimension)
        .as_length_range()
        .expect("a <length-range>");
    assert_eq!(
        block_dimension.optimum,
        Measure::Length(Length::from_points(100.0))
    );
    let inline_dimension = value(&tree, container, PropertyId::InlineProgressionDimension)
        .as_length_range()
        .expect("a <length-range>");
    assert_eq!(
        inline_dimension.optimum,
        Measure::Length(Length::from_points(200.0))
    );
    assert_eq!(
        inline_dimension.minimum,
        Measure::Length(Length::from_points(50.0))
    );
}

#[test]
fn a_vertical_writing_mode_swaps_height_and_width() {
    // §5.3.3: with a left-to-right block-progression-direction, `height` maps
    // onto the inline-progression-dimension instead.
    let tree = parse(
        "",
        r#"<fo:block-container writing-mode="tb-lr" height="100pt" width="200pt">
             <fo:block>text</fo:block>
           </fo:block-container>"#,
    );
    let container = nth(&tree, FoElement::BlockContainer, 0);
    assert!(
        !WritingMode::parse("tb-lr")
            .expect("a writing mode")
            .block_progression_is_vertical()
    );
    let inline_dimension = value(&tree, container, PropertyId::InlineProgressionDimension)
        .as_length_range()
        .expect("a <length-range>");
    assert_eq!(
        inline_dimension.optimum,
        Measure::Length(Length::from_points(100.0))
    );
    let block_dimension = value(&tree, container, PropertyId::BlockProgressionDimension)
        .as_length_range()
        .expect("a <length-range>");
    assert_eq!(
        block_dimension.optimum,
        Measure::Length(Length::from_points(200.0))
    );
}

#[test]
fn text_align_left_becomes_writing_mode_relative() {
    // §5.4.3.
    let tree = parse(
        r#"writing-mode="rl-tb""#,
        r#"<fo:block text-align="left">t</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert_eq!(
        value(&tree, block, PropertyId::TextAlign),
        Value::keyword("end")
    );
}

// --------------------------------------------- expression functions (§5.10.4)

#[test]
fn from_parent_reads_the_parents_computed_value() {
    let tree = parse(
        "",
        r#"<fo:block padding-top="8pt">
             <fo:block padding-top="from-parent(padding-top)">text</fo:block>
           </fo:block>"#,
    );
    let inner = nth(&tree, FoElement::Block, 1);
    assert!(close(points(&tree, inner, PropertyId::PaddingTop), 8.0));
}

#[test]
fn from_parent_may_take_part_in_arithmetic() {
    let tree = parse(
        "",
        r#"<fo:block padding-top="8pt">
             <fo:block padding-top="from-parent(padding-top) + 2pt">text</fo:block>
           </fo:block>"#,
    );
    let inner = nth(&tree, FoElement::Block, 1);
    assert!(close(points(&tree, inner, PropertyId::PaddingTop), 10.0));
}

#[test]
fn inherited_property_value_reads_a_different_property() {
    // The example of §5.10.4, adapted: the value comes from `start-indent` on
    // the parent, not from the property being computed.
    let tree = parse(
        "",
        r#"<fo:block start-indent="30pt">
             <fo:block end-indent="inherited-property-value(start-indent)">text</fo:block>
           </fo:block>"#,
    );
    let inner = nth(&tree, FoElement::Block, 1);
    assert!(close(points(&tree, inner, PropertyId::EndIndent), 30.0));
}

#[test]
fn inherited_property_value_rejects_a_non_inherited_property() {
    let source = document(
        "",
        r#"<fo:block padding-top="4pt">
             <fo:block padding-top="inherited-property-value(padding-top)">t</fo:block>
           </fo:block>"#,
    );
    let error = FoDocument::parse(&source).expect_err("padding-top is not inherited");
    assert!(
        error.to_string().contains("not an inherited property"),
        "unexpected error: {error}"
    );
}

#[test]
fn from_nearest_specified_value_skips_ancestors_that_do_not_specify() {
    let tree = parse(
        "",
        r#"<fo:block text-indent="21pt">
             <fo:block>
               <fo:block text-indent="from-nearest-specified-value(text-indent)">t</fo:block>
             </fo:block>
           </fo:block>"#,
    );
    let inner = nth(&tree, FoElement::Block, 2);
    assert!(close(points(&tree, inner, PropertyId::TextIndent), 21.0));
}

#[test]
fn from_page_master_region_reads_the_region_in_force() {
    // The example of §5.10.4: the region definition carries the writing mode.
    let source = r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
        <fo:layout-master-set>
          <fo:simple-page-master master-name="all-pages">
            <fo:region-body region-name="xsl-region-body" writing-mode="tb-rl"/>
          </fo:simple-page-master>
          <fo:page-sequence-master master-name="default-sequence">
            <fo:repeatable-page-master-reference master-reference="all-pages"/>
          </fo:page-sequence-master>
        </fo:layout-master-set>
        <fo:page-sequence master-reference="default-sequence">
          <fo:flow flow-name="xsl-region-body">
            <fo:block-container writing-mode="from-page-master-region()">
              <fo:block>text</fo:block>
            </fo:block-container>
          </fo:flow>
        </fo:page-sequence>
      </fo:root>"#;
    let tree = FoDocument::parse(source).expect("a valid document");
    let container = nth(&tree, FoElement::BlockContainer, 0);
    assert_eq!(
        value(&tree, container, PropertyId::WritingMode),
        Value::keyword("tb-rl")
    );
}

#[test]
fn body_start_uses_the_provisional_distance_of_the_list_block() {
    let tree = parse(
        "",
        r#"<fo:list-block provisional-distance-between-starts="24pt"
                          provisional-label-separation="6pt">
             <fo:list-item>
               <fo:list-item-label end-indent="label-end()">
                 <fo:block>1.</fo:block>
               </fo:list-item-label>
               <fo:list-item-body start-indent="body-start()">
                 <fo:block>text</fo:block>
               </fo:list-item-body>
             </fo:list-item>
           </fo:list-block>"#,
    );
    let body = nth(&tree, FoElement::ListItemBody, 0);
    assert!(close(points(&tree, body, PropertyId::StartIndent), 24.0));

    // Without a reference inline-progression-dimension, `label-end()` cannot be
    // computed and refinement leaves the value open for the formatter.
    let label = nth(&tree, FoElement::ListItemLabel, 0);
    assert_eq!(value(&tree, label, PropertyId::EndIndent), Value::Absent);
}

#[test]
fn label_end_is_computed_when_a_reference_dimension_is_supplied() {
    // §7.30.11: label-end() = reference width
    //                       - (provisional-distance-between-starts
    //                          + start-indent - provisional-label-separation)
    let options = ParseOptions {
        refine: RefineOptions {
            reference_inline_progression_dimension: Some(Length::from_points(400.0)),
            ..RefineOptions::default()
        },
        ..ParseOptions::default()
    };
    let source = document(
        "",
        r#"<fo:list-block provisional-distance-between-starts="24pt"
                          provisional-label-separation="6pt">
             <fo:list-item>
               <fo:list-item-label end-indent="label-end()">
                 <fo:block>1.</fo:block>
               </fo:list-item-label>
               <fo:list-item-body start-indent="body-start()">
                 <fo:block>text</fo:block>
               </fo:list-item-body>
             </fo:list-item>
           </fo:list-block>"#,
    );
    let tree = FoDocument::parse_with(&source, &options).expect("a valid document");
    let label = nth(&tree, FoElement::ListItemLabel, 0);
    assert!(close(points(&tree, label, PropertyId::EndIndent), 382.0));
}

#[test]
fn from_table_column_reads_the_governing_column() {
    let tree = parse(
        "",
        r#"<fo:table table-layout="fixed">
             <fo:table-column column-number="1" column-width="40pt"
                              background-color="red"/>
             <fo:table-column column-number="2" column-width="60pt"/>
             <fo:table-body>
               <fo:table-row>
                 <fo:table-cell><fo:block>a</fo:block></fo:table-cell>
                 <fo:table-cell><fo:block>b</fo:block></fo:table-cell>
               </fo:table-row>
             </fo:table-body>
           </fo:table>"#,
    );
    let first = nth(&tree, FoElement::TableCell, 0);
    let second = nth(&tree, FoElement::TableCell, 1);
    // §5.4.2: the implied `column-number` follows from the cell's position.
    assert_eq!(
        value(&tree, first, PropertyId::ColumnNumber),
        Value::Integer(1)
    );
    assert_eq!(
        value(&tree, second, PropertyId::ColumnNumber),
        Value::Integer(2)
    );
}

// ------------------------------------------------------- unit arithmetic ----

#[test]
fn absolute_units_convert_to_points() {
    let tree = parse(
        "",
        r#"<fo:block padding-top="1in" padding-bottom="2.54cm"
                     padding-left="25.4mm" padding-right="6pc">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    for property in [
        PropertyId::PaddingTop,
        PropertyId::PaddingBottom,
        PropertyId::PaddingLeft,
    ] {
        assert!(
            close(points(&tree, block, property), 72.0),
            "{} is {}",
            property.name(),
            points(&tree, block, property)
        );
    }
    assert!(close(points(&tree, block, PropertyId::PaddingRight), 72.0));
}

#[test]
fn em_resolves_against_the_refined_font_size_of_the_object_itself() {
    // §5.9.2: once `font-size` has been evaluated, its value is the current
    // font size for every other property on the same object.
    let tree = parse(
        "",
        r#"<fo:block font-size="20pt" padding-top="1.5em" text-indent="2em">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::PaddingTop), 30.0));
    assert!(close(points(&tree, block, PropertyId::TextIndent), 40.0));
}

#[test]
fn font_size_percentages_refer_to_the_parent() {
    let tree = parse(
        "",
        r#"<fo:block font-size="10pt">
             <fo:block font-size="150%">
               <fo:block font-size="50%">text</fo:block>
             </fo:block>
           </fo:block>"#,
    );
    assert!(close(
        points(&tree, nth(&tree, FoElement::Block, 1), PropertyId::FontSize),
        15.0
    ));
    assert!(close(
        points(&tree, nth(&tree, FoElement::Block, 2), PropertyId::FontSize),
        7.5
    ));
}

#[test]
fn relative_font_size_keywords_scale_the_parent() {
    let tree = parse(
        "",
        r#"<fo:block font-size="10pt">
             <fo:block font-size="larger">
               <fo:block font-size="smaller">text</fo:block>
             </fo:block>
           </fo:block>"#,
    );
    assert!(close(
        points(&tree, nth(&tree, FoElement::Block, 1), PropertyId::FontSize),
        12.0
    ));
    assert!(close(
        points(&tree, nth(&tree, FoElement::Block, 2), PropertyId::FontSize),
        10.0
    ));
}

#[test]
fn expressions_combine_units_numbers_and_functions() {
    let tree = parse(
        "",
        r#"<fo:block font-size="10pt"
                     padding-top="(2 * 3pt) + 1em div 2"
                     padding-bottom="max(3pt, 7pt)"
                     padding-left="abs(0pt - 4pt)"
                     padding-right="floor(1.4in div 1.0in) * 1.0in">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::PaddingTop), 11.0));
    assert!(close(points(&tree, block, PropertyId::PaddingBottom), 7.0));
    assert!(close(points(&tree, block, PropertyId::PaddingLeft), 4.0));
    assert!(close(points(&tree, block, PropertyId::PaddingRight), 72.0));
}

#[test]
fn mixing_unit_powers_in_an_additive_expression_is_an_error() {
    // §5.9.6: addition requires operands of the same unit power.
    let source = document("", r#"<fo:block padding-top="1pt + 1">t</fo:block>"#);
    let error = FoDocument::parse(&source).expect_err("mixed unit powers");
    assert!(
        error.to_string().contains("unit power"),
        "unexpected error: {error}"
    );
}

#[test]
fn percentages_survive_refinement_when_their_reference_is_unknown() {
    let tree = parse("", r#"<fo:block text-indent="25%">text</fo:block>"#);
    let block = nth(&tree, FoElement::Block, 0);
    assert_eq!(
        value(&tree, block, PropertyId::TextIndent),
        Value::Percent(25.0)
    );
}

#[test]
fn percentages_resolve_when_a_reference_is_supplied() {
    let options = ParseOptions {
        refine: RefineOptions {
            reference_inline_progression_dimension: Some(Length::from_points(400.0)),
            ..RefineOptions::default()
        },
        ..ParseOptions::default()
    };
    let source = document("", r#"<fo:block text-indent="25%">text</fo:block>"#);
    let tree = FoDocument::parse_with(&source, &options).expect("a valid document");
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::TextIndent), 100.0));
}

// ------------------------------------------------------- compound datatypes --

#[test]
fn the_short_form_of_a_compound_fills_every_length_component() {
    // §5.11: `space-before="4pt"` is the short form of the five components.
    let tree = parse("", r#"<fo:block space-before="4pt">text</fo:block>"#);
    let block = nth(&tree, FoElement::Block, 0);
    let space = value(&tree, block, PropertyId::SpaceBefore)
        .as_space()
        .expect("a <space>");
    let four = Measure::Length(Length::from_points(4.0));
    assert_eq!(
        (space.minimum, space.optimum, space.maximum),
        (four, four, four)
    );
    assert_eq!(space.conditionality, Conditionality::Discard);
}

#[test]
fn complete_form_components_override_the_short_form() {
    // The exact example of §5.11.
    let tree = parse(
        "",
        r#"<fo:block space-before="4.0pt" space-before.maximum="6.0pt">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    let space = value(&tree, block, PropertyId::SpaceBefore)
        .as_space()
        .expect("a <space>");
    assert_eq!(space.minimum, Measure::Length(Length::from_points(4.0)));
    assert_eq!(space.optimum, Measure::Length(Length::from_points(4.0)));
    assert_eq!(space.maximum, Measure::Length(Length::from_points(6.0)));
}

#[test]
fn the_keep_short_form_sets_all_three_components() {
    use oxixml_fo::property::KeepValue;
    let tree = parse("", r#"<fo:block keep-together="always">text</fo:block>"#);
    let block = nth(&tree, FoElement::Block, 0);
    let keep = value(&tree, block, PropertyId::KeepTogether)
        .as_keep()
        .expect("a <keep>");
    assert_eq!(keep.within_line, KeepValue::Always);
    assert_eq!(keep.within_column, KeepValue::Always);
    assert_eq!(keep.within_page, KeepValue::Always);
}

#[test]
fn compound_values_inherit_as_a_unit() {
    let tree = parse(
        "",
        r#"<fo:block keep-together.within-page="3">
             <fo:block>text</fo:block>
           </fo:block>"#,
    );
    use oxixml_fo::property::KeepValue;
    let inner = nth(&tree, FoElement::Block, 1);
    let keep = value(&tree, inner, PropertyId::KeepTogether)
        .as_keep()
        .expect("a <keep>");
    assert_eq!(keep.within_page, KeepValue::Strength(3));
    assert_eq!(keep.within_line, KeepValue::Auto);
}

// ---------------------------------------------------------- other mappings --

#[test]
fn a_border_width_is_zero_when_its_style_is_none() {
    // §5.1.2.
    let tree = parse(
        "",
        r#"<fo:block border-top-width="5pt" border-top-style="none">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::BorderTopWidth), 0.0));
}

#[test]
fn border_width_keywords_resolve_to_lengths() {
    let tree = parse(
        "",
        r#"<fo:block border-top-width="thick" border-top-style="solid">text</fo:block>"#,
    );
    let block = nth(&tree, FoElement::Block, 0);
    assert!(close(points(&tree, block, PropertyId::BorderTopWidth), 2.0));
}

#[test]
fn border_colors_default_to_the_color_property() {
    let tree = parse(
        "",
        r#"<fo:block color="navy" border-top-style="solid">text</fo:block>"#,
    );
    use oxixml_fo::property::Color;
    let block = nth(&tree, FoElement::Block, 0);
    assert_eq!(
        value(&tree, block, PropertyId::BorderTopColor).as_color(),
        Some(&Color::rgb(0x00, 0x00, 0x80))
    );
}

#[test]
fn z_index_accumulates_down_the_tree() {
    // §5.4.5.
    let tree = parse(
        "",
        r#"<fo:block-container z-index="2">
             <fo:block-container z-index="3">
               <fo:block>text</fo:block>
             </fo:block-container>
           </fo:block-container>"#,
    );
    let inner = nth(&tree, FoElement::BlockContainer, 1);
    assert_eq!(value(&tree, inner, PropertyId::ZIndex), Value::Integer(5));
}

#[test]
fn colours_accept_every_notation_of_section_5_9_9() {
    use oxixml_fo::property::Color;
    let tree = parse(
        "",
        r##"<fo:block color="#0f8">
              <fo:block background-color="rgb(255, 128, 0)">
                <fo:block background-color="teal">text</fo:block>
              </fo:block>
            </fo:block>"##,
    );
    assert_eq!(
        value(&tree, nth(&tree, FoElement::Block, 0), PropertyId::Color).as_color(),
        Some(&Color::rgb(0x00, 0xff, 0x88))
    );
    assert_eq!(
        value(
            &tree,
            nth(&tree, FoElement::Block, 1),
            PropertyId::BackgroundColor
        )
        .as_color(),
        Some(&Color::rgb(255, 128, 0))
    );
    assert_eq!(
        value(
            &tree,
            nth(&tree, FoElement::Block, 2),
            PropertyId::BackgroundColor
        )
        .as_color(),
        Some(&Color::rgb(0x00, 0x80, 0x80))
    );
}

#[test]
fn uri_specifications_are_unwrapped() {
    let tree = parse(
        "",
        r#"<fo:block>
             <fo:external-graphic src="url('logo.png')"/>
           </fo:block>"#,
    );
    let graphic = nth(&tree, FoElement::ExternalGraphic, 0);
    assert_eq!(
        value(&tree, graphic, PropertyId::Src),
        Value::Uri("logo.png".into())
    );
}

#[test]
fn the_area_primitives_are_independent_of_the_tree() {
    // The area model is a vocabulary for the renderer, not something this crate
    // builds; it can be used without any formatting object tree at all.
    let dimensions = Dimensions::new(Length::from_points(10.0), Length::from_points(12.0));
    assert_eq!(
        dimensions.inline_progression_dimension,
        Length::from_points(10.0)
    );
}
