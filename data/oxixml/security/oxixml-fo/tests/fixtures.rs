//! Two complete XSL-FO fixtures, taken from the shape of the examples in
//! [XSL 1.1 §6](https://www.w3.org/TR/xsl11/#fo-section), whose structure and
//! refined properties are asserted end to end.
//!
//! The point of these tests is the *hand-off* to a renderer: after
//! [`FoDocument::parse`] the tree carries everything a formatter needs, and
//! nothing that requires a formatter to produce.

use oxixml_fo::area::{
    AreaTraits, BlockArea, BlockContent, Dimensions, Edges, Item, LineArea, LineBreaker,
    LineBreakingStrategy,
};
use oxixml_fo::fo::FoElement;
use oxixml_fo::property::{Length, Measure, PropertyId, RefineOptions, Value};
use oxixml_fo::{FoDocument, FoTree, NodeId, ParseOptions};

/// A paginated article: page masters, a header, a numbered flow and a table.
const ARTICLE: &str = r##"<?xml version="1.0" encoding="UTF-8"?>
<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
  <fo:layout-master-set>
    <fo:simple-page-master master-name="A4-portrait"
                           page-height="297mm" page-width="210mm"
                           margin-top="15mm" margin-bottom="15mm"
                           margin-left="20mm" margin-right="20mm">
      <fo:region-body region-name="body" margin-top="15mm" margin-bottom="15mm"/>
      <fo:region-before region-name="header" extent="12mm"/>
      <fo:region-after region-name="footer" extent="12mm"/>
    </fo:simple-page-master>
    <fo:page-sequence-master master-name="article">
      <fo:repeatable-page-master-reference master-reference="A4-portrait"/>
    </fo:page-sequence-master>
  </fo:layout-master-set>

  <fo:page-sequence master-reference="article" initial-page-number="1"
                    format="1" language="en" country="GB">
    <fo:static-content flow-name="header">
      <fo:block text-align="end" font-size="8pt" border-after-style="solid"
                border-after-width="0.5pt">
        <fo:page-number/>
      </fo:block>
    </fo:static-content>

    <fo:flow flow-name="body">
      <fo:block font="bold 18pt/22pt Helvetica, sans-serif"
                space-after="12pt" keep-with-next="always">
        <fo:marker marker-class-name="section">Introduction</fo:marker>
        Introduction
      </fo:block>

      <fo:block font-size="10pt" text-align="justify" text-indent="1em"
                space-after="6pt" hyphenate="true">
        The quick brown fox jumps over the lazy dog, and then
        <fo:inline font-style="italic">emphasises</fo:inline> the point with a
        <fo:basic-link internal-destination="table">cross reference</fo:basic-link>
        on page <fo:page-number-citation ref-id="table"/>.
      </fo:block>

      <fo:block-container writing-mode="lr-tb" start-indent="12pt" end-indent="12pt"
                          border="1pt solid black" padding="4pt"
                          background-color="#f0f0f0">
        <fo:block font-size="9pt">A boxed aside.</fo:block>
      </fo:block-container>

      <fo:table id="table" table-layout="fixed" width="100%"
                border-collapse="separate" border-separation="2pt"
                space-before="12pt">
        <fo:table-column column-number="1" column-width="proportional-column-width(2)"/>
        <fo:table-column column-number="2" column-width="proportional-column-width(1)"/>
        <fo:table-header>
          <fo:table-row keep-together="always">
            <fo:table-cell border="0.5pt solid black" padding="2pt">
              <fo:block font-weight="bold">Name</fo:block>
            </fo:table-cell>
            <fo:table-cell border="0.5pt solid black" padding="2pt">
              <fo:block font-weight="bold">Value</fo:block>
            </fo:table-cell>
          </fo:table-row>
        </fo:table-header>
        <fo:table-body>
          <fo:table-row>
            <fo:table-cell padding="2pt"><fo:block>alpha</fo:block></fo:table-cell>
            <fo:table-cell padding="2pt"><fo:block>1</fo:block></fo:table-cell>
          </fo:table-row>
          <fo:table-row>
            <fo:table-cell number-columns-spanned="2" padding="2pt">
              <fo:block text-align="center">spanned</fo:block>
            </fo:table-cell>
          </fo:table-row>
        </fo:table-body>
      </fo:table>
    </fo:flow>
  </fo:page-sequence>
</fo:root>
"##;

/// The enumerated list of XSL 1.1 §6.8.1.1.1, verbatim in its "Result Instance"
/// form, wrapped in the pagination the example leaves implicit.
const ENUMERATED_LIST: &str = r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
  <fo:layout-master-set>
    <fo:simple-page-master master-name="page" page-height="11in" page-width="8.5in">
      <fo:region-body region-name="xsl-region-body"/>
    </fo:simple-page-master>
  </fo:layout-master-set>
  <fo:page-sequence master-reference="page">
    <fo:flow flow-name="xsl-region-body">
      <fo:list-block provisional-distance-between-starts="15mm"
                     provisional-label-separation="5mm">
        <fo:list-item>
          <fo:list-item-label start-indent="5mm" end-indent="label-end()">
            <fo:block>a. </fo:block>
          </fo:list-item-label>
          <fo:list-item-body start-indent="body-start()">
            <fo:block>List item 1. </fo:block>
          </fo:list-item-body>
        </fo:list-item>
        <fo:list-item>
          <fo:list-item-label start-indent="5mm" end-indent="label-end()">
            <fo:block>b. </fo:block>
          </fo:list-item-label>
          <fo:list-item-body start-indent="body-start()">
            <fo:block>List item 2. </fo:block>
          </fo:list-item-body>
        </fo:list-item>
      </fo:list-block>
    </fo:flow>
  </fo:page-sequence>
</fo:root>
"#;

fn all(tree: &FoTree, element: FoElement) -> Vec<NodeId> {
    tree.elements()
        .iter()
        .copied()
        .filter(|id| tree.element(*id) == Some(element))
        .collect()
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-6
}

fn points(tree: &FoTree, id: NodeId, property: PropertyId) -> f64 {
    tree.properties(id)
        .value(property)
        .as_length()
        .unwrap_or_else(|| {
            panic!(
                "{} is not a length: {:?}",
                property.name(),
                tree.properties(id).value(property)
            )
        })
        .points()
}

#[test]
fn the_article_fixture_round_trips_its_structure() {
    let tree = FoDocument::parse(ARTICLE).expect("the article fixture is valid XSL-FO");

    // The structural skeleton, as an outline.
    let outline = tree.outline();
    for expected in [
        "fo:root\n",
        "  fo:layout-master-set\n",
        "    fo:simple-page-master\n",
        "      fo:region-body\n",
        "  fo:page-sequence\n",
        "    fo:static-content\n",
        "    fo:flow\n",
    ] {
        assert!(
            outline.contains(expected),
            "missing {expected:?}:\n{outline}"
        );
    }

    // Every formatting object the fixture uses made it into the tree.
    for (element, expected) in [
        (FoElement::SimplePageMaster, 1),
        (FoElement::PageSequenceMaster, 1),
        (FoElement::RegionBody, 1),
        (FoElement::RegionBefore, 1),
        (FoElement::RegionAfter, 1),
        (FoElement::StaticContent, 1),
        (FoElement::Flow, 1),
        (FoElement::Marker, 1),
        (FoElement::Inline, 1),
        (FoElement::BasicLink, 1),
        (FoElement::PageNumber, 1),
        (FoElement::PageNumberCitation, 1),
        (FoElement::BlockContainer, 1),
        (FoElement::Table, 1),
        (FoElement::TableColumn, 2),
        (FoElement::TableHeader, 1),
        (FoElement::TableBody, 1),
        (FoElement::TableRow, 3),
        (FoElement::TableCell, 5),
    ] {
        assert_eq!(
            all(&tree, element).len(),
            expected,
            "wrong number of {element:?}"
        );
    }

    // Document order is preserved: the static content precedes the flow.
    let sequence = all(&tree, FoElement::PageSequence)[0];
    let children: Vec<FoElement> = tree
        .children(sequence)
        .iter()
        .filter_map(|id| tree.element(*id))
        .collect();
    assert_eq!(children, vec![FoElement::StaticContent, FoElement::Flow]);
}

#[test]
fn the_article_fixture_refines_its_properties() {
    let tree = FoDocument::parse(ARTICLE).expect("the article fixture is valid XSL-FO");

    // Page geometry: 210mm x 297mm in points.
    let master = all(&tree, FoElement::SimplePageMaster)[0];
    assert!(close(
        points(&tree, master, PropertyId::PageWidth),
        210.0 * 72.0 / 25.4
    ));
    assert!(close(
        points(&tree, master, PropertyId::PageHeight),
        297.0 * 72.0 / 25.4
    ));

    // The `font` shorthand on the first flow block.
    let heading = all(&tree, FoElement::Block)
        .into_iter()
        .find(|id| tree.text_content(*id).contains("Introduction"))
        .expect("the heading block");
    assert!(close(points(&tree, heading, PropertyId::FontSize), 18.0));
    assert_eq!(
        tree.properties(heading).value(PropertyId::FontWeight),
        &Value::keyword("bold")
    );
    assert_eq!(
        tree.properties(heading)
            .value(PropertyId::LineHeight)
            .as_space()
            .map(|space| space.optimum),
        Some(Measure::Length(Length::from_points(22.0)))
    );
    // `keep-with-next` is a compound whose short form fills every component.
    assert_eq!(
        tree.properties(heading)
            .value(PropertyId::KeepWithNext)
            .as_keep()
            .map(|keep| keep.within_page),
        Some(oxixml_fo::property::KeepValue::Always)
    );

    // `text-indent="1em"` uses the refined font size of the paragraph itself.
    let paragraph = all(&tree, FoElement::Block)
        .into_iter()
        .find(|id| tree.text_content(*id).contains("quick brown fox"))
        .expect("the paragraph block");
    assert!(close(points(&tree, paragraph, PropertyId::FontSize), 10.0));
    assert!(close(
        points(&tree, paragraph, PropertyId::TextIndent),
        10.0
    ));
    assert_eq!(
        tree.properties(paragraph).value(PropertyId::TextAlign),
        &Value::keyword("justify")
    );

    // The block container: `border` and `padding` shorthands, and the
    // start-indent formula for a reference-area generator.
    let container = all(&tree, FoElement::BlockContainer)[0];
    // `border-start-width` is a <length-conditional>, so its `.length`
    // component carries the width.
    assert_eq!(
        tree.properties(container)
            .value(PropertyId::BorderStartWidth)
            .as_length_conditional()
            .map(|conditional| conditional.length),
        Some(Measure::Length(Length::from_points(1.0)))
    );
    assert_eq!(
        tree.properties(container).value(PropertyId::BorderTopStyle),
        &Value::keyword("solid")
    );
    assert_eq!(
        tree.properties(container)
            .value(PropertyId::PaddingStart)
            .as_length_conditional()
            .map(|conditional| conditional.length),
        Some(Measure::Length(Length::from_points(4.0)))
    );
    assert!(close(
        points(&tree, container, PropertyId::StartIndent),
        12.0
    ));

    // The table: `border-separation` from the short form, and per-object
    // border precedences.
    let table = all(&tree, FoElement::Table)[0];
    let separation = tree
        .properties(table)
        .value(PropertyId::BorderSeparation)
        .as_length_bp_ip()
        .expect("border-separation is a <length-bp-ip-direction>");
    assert_eq!(
        separation.inline_progression_direction,
        Measure::Length(Length::from_points(2.0))
    );
    assert_eq!(
        tree.properties(table)
            .get(PropertyId::BorderAfterPrecedence),
        Some(&Value::Integer(6))
    );

    // Column numbers: explicit on the columns, implied on the cells, and the
    // spanning cell restarts at 1 in its own row.
    let cells = all(&tree, FoElement::TableCell);
    let numbers: Vec<i64> = cells
        .iter()
        .map(|id| {
            tree.properties(*id)
                .value(PropertyId::ColumnNumber)
                .as_integer()
                .unwrap_or_default()
        })
        .collect();
    assert_eq!(numbers, vec![1, 2, 1, 2, 1]);

    // Inheritance reaches the deepest blocks: `language` was set on the page
    // sequence and is inherited by everything below it.
    let aside = all(&tree, FoElement::Block)
        .into_iter()
        .find(|id| tree.text_content(*id).contains("boxed aside"))
        .expect("the aside block");
    assert_eq!(
        tree.properties(aside).value(PropertyId::Language).as_str(),
        Some("en")
    );
    assert!(close(points(&tree, aside, PropertyId::FontSize), 9.0));
}

#[test]
fn the_enumerated_list_fixture_matches_the_specifications_example() {
    // With no reference dimension the label's `end-indent` stays open, but
    // `body-start()` is computable from the provisional distance alone.
    let tree = FoDocument::parse(ENUMERATED_LIST).expect("the list fixture is valid XSL-FO");
    assert_eq!(all(&tree, FoElement::ListItem).len(), 2);

    let millimetre = 72.0 / 25.4;
    for body in all(&tree, FoElement::ListItemBody) {
        // body-start() = start-indent + provisional-distance-between-starts,
        // where start-indent is inherited from the list block (0pt here).
        assert!(close(
            points(&tree, body, PropertyId::StartIndent),
            15.0 * millimetre
        ));
    }
    for label in all(&tree, FoElement::ListItemLabel) {
        assert!(close(
            points(&tree, label, PropertyId::StartIndent),
            5.0 * millimetre
        ));
        assert_eq!(
            tree.properties(label).value(PropertyId::EndIndent),
            &Value::Absent,
            "label-end() needs a reference area width"
        );
    }

    // Supplying the reference width closes the last gap.
    let options = ParseOptions {
        refine: RefineOptions {
            reference_inline_progression_dimension: Some(Length::from_points(468.0)),
            ..RefineOptions::default()
        },
        ..ParseOptions::default()
    };
    let tree = FoDocument::parse_with(ENUMERATED_LIST, &options).expect("still valid");
    for label in all(&tree, FoElement::ListItemLabel) {
        // §7.30.11 measures every term on the closest ancestor fo:list-block,
        // whose start-indent is the initial 0pt here:
        //   label-end() = 468pt - (15mm + 0pt - 5mm)
        let expected = 468.0 - (15.0 * millimetre - 5.0 * millimetre);
        assert!(
            close(points(&tree, label, PropertyId::EndIndent), expected),
            "got {}",
            points(&tree, label, PropertyId::EndIndent)
        );
    }
}

#[test]
fn a_renderer_can_drive_the_area_primitives_from_a_refined_tree() {
    // This is the whole hand-off: the crate refines properties, and the
    // renderer uses the area vocabulary and the line breaker itself. Nothing
    // below is done by `oxixml-fo` on the caller's behalf.
    let tree = FoDocument::parse(ARTICLE).expect("the article fixture is valid XSL-FO");
    let paragraph = all(&tree, FoElement::Block)
        .into_iter()
        .find(|id| tree.text_content(*id).contains("quick brown fox"))
        .expect("the paragraph block");
    let properties = tree.properties(paragraph);

    // A renderer reads the refined geometry it needs...
    let measure = Length::from_points(300.0);
    let start_indent = properties
        .value(PropertyId::StartIndent)
        .as_length()
        .unwrap_or(Length::ZERO);
    let end_indent = properties
        .value(PropertyId::EndIndent)
        .as_length()
        .unwrap_or(Length::ZERO);
    let font_size = properties
        .value(PropertyId::FontSize)
        .as_length()
        .expect("a refined font size");

    // ... measures its own text (font metrics live in the renderer) ...
    let text = tree.text_content(paragraph);
    let words: Vec<&str> = text.split_whitespace().collect();
    assert!(words.len() > 5);
    let mut items = Vec::new();
    for (index, word) in words.iter().enumerate() {
        if index > 0 {
            items.push(Item::glue(
                font_size * 0.33,
                font_size * 0.17,
                font_size * 0.11,
            ));
        }
        items.push(Item::boxed(font_size * 0.5 * word.chars().count() as f64));
    }
    items.push(Item::forced_break());

    // ... and drives the line breaker.
    let available = measure - start_indent - end_indent;
    let breaker = LineBreaker::new(available).with_strategy(LineBreakingStrategy::TotalFit);
    let breaks = breaker.break_lines(&items);
    assert!(breaks.len() > 1, "the paragraph wraps");
    assert!(breaks.iter().all(|line| line.start < line.end));
    assert_eq!(breaks[0].start, 0);
    assert_eq!(breaks.last().map(|line| line.end), Some(items.len() - 1));

    // The resulting area tree is built from this crate's data types.
    let lines: Vec<LineArea> = breaks
        .iter()
        .enumerate()
        .map(|(index, line)| LineArea {
            traits: AreaTraits {
                dimensions: Dimensions::new(line.natural_width, font_size * 1.2),
                ..AreaTraits::default()
            },
            children: Vec::new(),
            baseline_offset: font_size,
            is_last: index + 1 == breaks.len(),
        })
        .collect();
    let block_extent = lines
        .iter()
        .map(|line| line.traits.dimensions.block_progression_dimension)
        .sum::<Length>();
    let area = BlockArea {
        traits: AreaTraits {
            dimensions: Dimensions::new(measure, block_extent),
            padding: Edges::uniform(Length::ZERO),
            ..AreaTraits::default()
        },
        start_indent,
        end_indent,
        content: BlockContent::Lines(lines),
        generated_by: Some(paragraph),
        ..BlockArea::default()
    };
    assert_eq!(area.available_inline_extent(), available);
    assert!(matches!(area.content, BlockContent::Lines(ref lines) if !lines.is_empty()));
    assert_eq!(area.generated_by, Some(paragraph));
}

#[test]
fn refinement_is_deterministic() {
    let first = FoDocument::parse(ARTICLE).expect("valid");
    let second = FoDocument::parse(ARTICLE).expect("valid");
    assert_eq!(first.outline(), second.outline());
    for (left, right) in first.elements().iter().zip(second.elements()) {
        assert_eq!(first.properties(*left), second.properties(*right));
    }
}

#[test]
fn every_formatting_object_gets_exactly_its_applicable_properties() {
    let tree = FoDocument::parse(ARTICLE).expect("valid");
    for id in tree.elements() {
        let element = tree.element(*id).expect("a formatting object");
        let refined: Vec<PropertyId> = tree.properties(*id).iter().map(|(id, _)| id).collect();
        for property in &refined {
            assert!(
                element.accepts_property(*property),
                "{} refined {} which does not apply to it",
                element.qualified_name(),
                property.name()
            );
        }
        // Shorthands never survive refinement; they are expanded away.
        assert!(
            refined.iter().all(|property| !property.is_shorthand()),
            "{} kept a shorthand",
            element.qualified_name()
        );
    }
}
