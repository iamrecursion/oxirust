//! FO tree construction for representative documents, and the detection of
//! invalid children, invalid character data and inapplicable properties.

use oxixml_fo::fo::{Category, FoElement};
use oxixml_fo::property::{PropertyId, RefineOptions, Value};
use oxixml_fo::{FoDocument, FoError, FoTree, NodeId, NodeKind, ParseOptions};

const PAGE_MASTER: &str = r#"
  <fo:layout-master-set>
    <fo:simple-page-master master-name="page"
                           page-height="11in" page-width="8.5in"
                           margin-top="0.5in" margin-bottom="0.5in"
                           margin-left="0.75in" margin-right="0.75in">
      <fo:region-body region-name="xsl-region-body" margin-top="1in" margin-bottom="1in"
                      column-count="2" column-gap="12pt"/>
      <fo:region-before region-name="xsl-region-before" extent="0.75in"/>
      <fo:region-after region-name="xsl-region-after" extent="0.75in"/>
      <fo:region-start region-name="xsl-region-start" extent="0.5in"/>
      <fo:region-end region-name="xsl-region-end" extent="0.5in"/>
    </fo:simple-page-master>
    <fo:page-sequence-master master-name="default">
      <fo:repeatable-page-master-reference master-reference="page"/>
    </fo:page-sequence-master>
  </fo:layout-master-set>"#;

fn wrap(body: &str) -> String {
    format!(
        r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">{PAGE_MASTER}
             <fo:page-sequence master-reference="default">
               <fo:static-content flow-name="xsl-region-before">
                 <fo:block>Header</fo:block>
               </fo:static-content>
               <fo:flow flow-name="xsl-region-body">{body}</fo:flow>
             </fo:page-sequence>
           </fo:root>"#
    )
}

fn parse(body: &str) -> FoTree {
    FoDocument::parse(&wrap(body)).expect("a valid XSL-FO document")
}

fn first(tree: &FoTree, element: FoElement) -> NodeId {
    tree.elements()
        .iter()
        .copied()
        .find(|id| tree.element(*id) == Some(element))
        .unwrap_or_else(|| panic!("no {element:?} in the tree"))
}

/// The last formatting object of the given kind, in document order. The flow
/// content of the test documents follows the static content, so this is how a
/// test reaches the objects it wrote itself.
fn last(tree: &FoTree, element: FoElement) -> NodeId {
    tree.elements()
        .iter()
        .copied()
        .rfind(|id| tree.element(*id) == Some(element))
        .unwrap_or_else(|| panic!("no {element:?} in the tree"))
}

fn count(tree: &FoTree, element: FoElement) -> usize {
    tree.elements()
        .iter()
        .filter(|id| tree.element(**id) == Some(element))
        .count()
}

// ------------------------------------------------------- tree construction --

#[test]
fn a_page_master_and_flow_become_a_typed_tree() {
    let tree = parse("<fo:block>Body</fo:block>");
    assert_eq!(tree.element(tree.root()), Some(FoElement::Root));

    let master = first(&tree, FoElement::SimplePageMaster);
    assert_eq!(
        tree.properties(master)
            .value(PropertyId::MasterName)
            .as_str(),
        Some("page")
    );
    assert_eq!(
        tree.properties(master).get(PropertyId::PageWidth),
        Some(&Value::Length(oxixml_fo::property::Length::from_points(
            612.0
        )))
    );

    // All five regions of the simple page master are present, in order.
    let regions: Vec<FoElement> = tree
        .children(master)
        .iter()
        .filter_map(|id| tree.element(*id))
        .collect();
    assert_eq!(
        regions,
        vec![
            FoElement::RegionBody,
            FoElement::RegionBefore,
            FoElement::RegionAfter,
            FoElement::RegionStart,
            FoElement::RegionEnd
        ]
    );

    let body = first(&tree, FoElement::RegionBody);
    assert_eq!(
        tree.properties(body).get(PropertyId::ColumnCount),
        Some(&Value::Number(2.0))
    );

    // The flow and the static content both hang off the page sequence.
    let sequence = first(&tree, FoElement::PageSequence);
    let children: Vec<FoElement> = tree
        .children(sequence)
        .iter()
        .filter_map(|id| tree.element(*id))
        .collect();
    assert_eq!(
        children,
        vec![FoElement::StaticContent, FoElement::Flow],
        "static content precedes the flow"
    );
}

#[test]
fn a_table_keeps_its_columns_header_and_bodies() {
    let tree = parse(
        r#"<fo:table table-layout="fixed" border-collapse="separate">
             <fo:table-column column-width="proportional-column-width(1)"/>
             <fo:table-column column-width="2in"/>
             <fo:table-header>
               <fo:table-row>
                 <fo:table-cell><fo:block>H1</fo:block></fo:table-cell>
                 <fo:table-cell><fo:block>H2</fo:block></fo:table-cell>
               </fo:table-row>
             </fo:table-header>
             <fo:table-body>
               <fo:table-row>
                 <fo:table-cell number-columns-spanned="2">
                   <fo:block>Spanned</fo:block>
                 </fo:table-cell>
               </fo:table-row>
             </fo:table-body>
           </fo:table>"#,
    );
    assert_eq!(count(&tree, FoElement::TableColumn), 2);
    assert_eq!(count(&tree, FoElement::TableRow), 2);
    assert_eq!(count(&tree, FoElement::TableCell), 3);

    let table = first(&tree, FoElement::Table);
    let structure: Vec<FoElement> = tree
        .children(table)
        .iter()
        .filter_map(|id| tree.element(*id))
        .collect();
    assert_eq!(
        structure,
        vec![
            FoElement::TableColumn,
            FoElement::TableColumn,
            FoElement::TableHeader,
            FoElement::TableBody
        ]
    );

    // `proportional-column-width()` survives refinement as a typed value.
    let column = first(&tree, FoElement::TableColumn);
    assert_eq!(
        tree.properties(column).get(PropertyId::ColumnWidth),
        Some(&Value::ProportionalColumnWidth(1.0))
    );

    // Border precedence takes its per-object initial value (§7.28.1).
    assert_eq!(
        tree.properties(table)
            .get(PropertyId::BorderBeforePrecedence),
        Some(&Value::Integer(6))
    );
    let cell = first(&tree, FoElement::TableCell);
    assert_eq!(
        tree.properties(cell)
            .get(PropertyId::BorderBeforePrecedence),
        Some(&Value::Integer(5))
    );
}

#[test]
fn a_list_block_keeps_label_and_body_pairs() {
    let tree = parse(
        r#"<fo:list-block provisional-distance-between-starts="18pt">
             <fo:list-item>
               <fo:list-item-label><fo:block>1.</fo:block></fo:list-item-label>
               <fo:list-item-body><fo:block>First</fo:block></fo:list-item-body>
             </fo:list-item>
             <fo:list-item>
               <fo:list-item-label><fo:block>2.</fo:block></fo:list-item-label>
               <fo:list-item-body><fo:block>Second</fo:block></fo:list-item-body>
             </fo:list-item>
           </fo:list-block>"#,
    );
    assert_eq!(count(&tree, FoElement::ListItem), 2);
    let item = first(&tree, FoElement::ListItem);
    let children: Vec<FoElement> = tree
        .children(item)
        .iter()
        .filter_map(|id| tree.element(*id))
        .collect();
    assert_eq!(
        children,
        vec![FoElement::ListItemLabel, FoElement::ListItemBody]
    );
}

#[test]
fn inline_formatting_objects_nest_inside_a_block() {
    let tree = parse(
        r#"<fo:block>
             before
             <fo:inline font-style="italic">emphasis</fo:inline>
             <fo:basic-link internal-destination="target">link</fo:basic-link>
             <fo:leader leader-pattern="dots" leader-length.optimum="2in"/>
             <fo:page-number/>
             <fo:page-number-citation ref-id="target"/>
             <fo:character character="x"/>
             <fo:wrapper><fo:inline>wrapped</fo:inline></fo:wrapper>
             <fo:footnote>
               <fo:inline>1</fo:inline>
               <fo:footnote-body><fo:block>A note.</fo:block></fo:footnote-body>
             </fo:footnote>
             after
           </fo:block>"#,
    );
    for element in [
        FoElement::Inline,
        FoElement::BasicLink,
        FoElement::Leader,
        FoElement::PageNumber,
        FoElement::PageNumberCitation,
        FoElement::Character,
        FoElement::Wrapper,
        FoElement::Footnote,
        FoElement::FootnoteBody,
    ] {
        assert!(count(&tree, element) >= 1, "missing {element:?}");
    }
    let block = tree
        .elements()
        .iter()
        .copied()
        .find(|id| {
            tree.element(*id) == Some(FoElement::Block) && tree.text_content(*id).contains("before")
        })
        .expect("the block that holds the inline material");
    assert!(
        tree.text_content(block).contains("after"),
        "character data is preserved around the inline children"
    );

    // `fo:leader`'s `leader-length` is a compound; only `.optimum` was given.
    let leader = first(&tree, FoElement::Leader);
    let range = tree
        .properties(leader)
        .value(PropertyId::LeaderLength)
        .as_length_range()
        .expect("leader-length is a <length-range>");
    assert_eq!(
        range.optimum,
        oxixml_fo::property::Measure::Length(oxixml_fo::property::Length::from_points(144.0))
    );
}

#[test]
fn markers_and_retrieve_markers_are_accepted_where_the_prose_allows_them() {
    let tree = parse(
        r#"<fo:block>
             <fo:marker marker-class-name="section"><fo:inline>Chapter</fo:inline></fo:marker>
             <fo:initial-property-set font-variant="small-caps"/>
             Body text.
           </fo:block>"#,
    );
    assert_eq!(count(&tree, FoElement::Marker), 1);
    assert_eq!(count(&tree, FoElement::InitialPropertySet), 1);
    assert_eq!(FoElement::Marker.category(), Category::Marker);
}

#[test]
fn instream_foreign_objects_keep_their_foreign_subtree() {
    let tree = parse(
        r#"<fo:block>
             <fo:instream-foreign-object content-height="2cm">
               <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                 <circle cx="50" cy="50" r="40"/>
               </svg>
             </fo:instream-foreign-object>
           </fo:block>"#,
    );
    let object = first(&tree, FoElement::InstreamForeignObject);
    let svg = tree
        .children(object)
        .iter()
        .copied()
        .find(|id| tree.node(*id).foreign().is_some())
        .expect("the SVG element is kept");
    let foreign = tree.node(svg).foreign().expect("a foreign element");
    assert_eq!(foreign.local_name, "svg");
    assert_eq!(
        foreign.namespace.as_deref(),
        Some("http://www.w3.org/2000/svg")
    );
    assert!(
        foreign
            .attributes
            .iter()
            .any(|(_, name, value)| name == "width" && value == "100")
    );
    assert!(
        tree.descendants(svg)
            .any(|id| matches!(tree.node(id).kind(), NodeKind::Foreign(child) if child.local_name == "circle"))
    );
}

#[test]
fn the_outline_reflects_the_document_structure() {
    let tree = parse("<fo:block><fo:inline>x</fo:inline></fo:block>");
    let outline = tree.outline();
    assert!(outline.starts_with("fo:root\n"));
    assert!(outline.contains("  fo:layout-master-set\n"));
    assert!(outline.contains("        fo:inline\n"), "{outline}");
}

#[test]
fn document_order_puts_every_ancestor_before_its_descendants() {
    let tree = parse("<fo:block><fo:block><fo:inline>x</fo:inline></fo:block></fo:block>");
    let order = tree.elements();
    for (index, id) in order.iter().enumerate() {
        for ancestor in tree.ancestors(*id) {
            let position = order
                .iter()
                .position(|candidate| *candidate == ancestor)
                .expect("ancestors are formatting objects");
            assert!(position < index, "an ancestor came after its descendant");
        }
    }
}

// --------------------------------------------------------------- validation --

#[test]
fn the_document_element_must_be_fo_root() {
    let error = FoDocument::parse(r#"<fo:block xmlns:fo="http://www.w3.org/1999/XSL/Format"/>"#)
        .expect_err("fo:block is not a document element");
    assert!(matches!(error, FoError::NotAnFoDocument { .. }), "{error}");
}

#[test]
fn an_unknown_element_in_the_fo_namespace_is_rejected() {
    let error = FoDocument::parse(&wrap("<fo:paragraph>x</fo:paragraph>"))
        .expect_err("fo:paragraph does not exist");
    assert!(
        matches!(error, FoError::UnknownFormattingObject { ref name, .. } if name == "paragraph"),
        "{error}"
    );
}

#[test]
fn a_block_is_not_allowed_inside_a_leader() {
    // §6.6.9: an fo:leader may contain only #PCDATA and %inline;.
    let error = FoDocument::parse(&wrap(
        "<fo:block><fo:leader><fo:block>x</fo:block></fo:leader></fo:block>",
    ))
    .expect_err("a leader may not contain a block");
    assert!(
        matches!(
            error,
            FoError::InvalidChild {
                parent: FoElement::Leader,
                child: FoElement::Block,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_region_may_not_appear_outside_a_page_master() {
    let error = FoDocument::parse(&wrap("<fo:block><fo:region-body/></fo:block>"))
        .expect_err("a region is not flow content");
    assert!(
        matches!(
            error,
            FoError::InvalidChild {
                parent: FoElement::Block,
                child: FoElement::RegionBody,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn character_data_is_rejected_where_pcdata_is_not_allowed() {
    let error = FoDocument::parse(&wrap(
        "<fo:table><fo:table-body>oops<fo:table-row>\
         <fo:table-cell><fo:block>x</fo:block></fo:table-cell>\
         </fo:table-row></fo:table-body></fo:table>",
    ))
    .expect_err("fo:table-body has no #PCDATA");
    assert!(
        matches!(
            error,
            FoError::InvalidText {
                parent: FoElement::TableBody,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn whitespace_between_structural_elements_is_ignored_by_default() {
    // The very same document without the offending text parses.
    let tree = parse(
        "<fo:table>\n  <fo:table-body>\n    <fo:table-row>\n\
         <fo:table-cell><fo:block>x</fo:block></fo:table-cell>\n\
         </fo:table-row>\n  </fo:table-body>\n</fo:table>",
    );
    assert_eq!(count(&tree, FoElement::TableCell), 1);
}

#[test]
fn structural_whitespace_can_be_made_an_error() {
    let options = ParseOptions {
        ignore_structural_whitespace: false,
        ..ParseOptions::default()
    };
    let error = FoDocument::parse_with(
        &wrap(
            "<fo:table>\n<fo:table-body><fo:table-row>\
               <fo:table-cell><fo:block>x</fo:block></fo:table-cell>\
               </fo:table-row></fo:table-body></fo:table>",
        ),
        &options,
    )
    .expect_err("whitespace is text");
    assert!(matches!(error, FoError::InvalidText { .. }), "{error}");
}

#[test]
fn a_missing_required_child_violates_the_content_model() {
    // §6.7.3: fo:table requires at least one fo:table-body.
    let error = FoDocument::parse(&wrap(
        r#"<fo:table><fo:table-column column-width="1in"/></fo:table>"#,
    ))
    .expect_err("a table needs a body");
    assert!(
        matches!(
            error,
            FoError::ContentModel {
                parent: FoElement::Table,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn children_must_appear_in_the_order_the_content_model_gives() {
    // §6.7.3: columns, then header, then footer, then bodies.
    let error = FoDocument::parse(&wrap(
        r#"<fo:table>
             <fo:table-body><fo:table-row>
               <fo:table-cell><fo:block>x</fo:block></fo:table-cell>
             </fo:table-row></fo:table-body>
             <fo:table-column column-width="1in"/>
           </fo:table>"#,
    ))
    .expect_err("the column comes after the body");
    assert!(
        matches!(
            error,
            FoError::ContentModel {
                parent: FoElement::Table,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_table_body_may_not_mix_rows_and_cells() {
    // §6.7.8: `(table-row+ | table-cell+)`.
    let error = FoDocument::parse(&wrap(
        r#"<fo:table><fo:table-body>
             <fo:table-row><fo:table-cell><fo:block>a</fo:block></fo:table-cell></fo:table-row>
             <fo:table-cell><fo:block>b</fo:block></fo:table-cell>
           </fo:table-body></fo:table>"#,
    ))
    .expect_err("rows and cells cannot be mixed");
    assert!(
        matches!(
            error,
            FoError::ContentModel {
                parent: FoElement::TableBody,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn a_list_item_needs_both_a_label_and_a_body() {
    let error = FoDocument::parse(&wrap(
        "<fo:list-block><fo:list-item>\
         <fo:list-item-label><fo:block>1.</fo:block></fo:list-item-label>\
         </fo:list-item></fo:list-block>",
    ))
    .expect_err("the body is missing");
    assert!(
        matches!(
            error,
            FoError::ContentModel {
                parent: FoElement::ListItem,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn an_inapplicable_property_is_rejected() {
    // §6.4.19: `column-width` is a table property and is not inherited, so it
    // may not appear on fo:flow. (`font-size` could, because §5.1.4 allows any
    // inherited property on any formatting object.)
    let error = FoDocument::parse(
        r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
             <fo:layout-master-set>
               <fo:simple-page-master master-name="page"><fo:region-body/></fo:simple-page-master>
             </fo:layout-master-set>
             <fo:page-sequence master-reference="page">
               <fo:flow flow-name="xsl-region-body" column-width="2in">
                 <fo:block>x</fo:block>
               </fo:flow>
             </fo:page-sequence>
           </fo:root>"#,
    )
    .expect_err("column-width does not apply to fo:flow");
    assert!(
        matches!(
            error,
            FoError::InapplicableProperty {
                fo: FoElement::Flow,
                ..
            }
        ),
        "{error}"
    );
}

#[test]
fn applicability_checking_can_be_switched_off() {
    let options = ParseOptions {
        refine: RefineOptions {
            validate_applicability: false,
            ..RefineOptions::default()
        },
        ..ParseOptions::default()
    };
    let tree = FoDocument::parse_with(
        r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
             <fo:layout-master-set>
               <fo:simple-page-master master-name="page"><fo:region-body/></fo:simple-page-master>
             </fo:layout-master-set>
             <fo:page-sequence master-reference="page">
               <fo:flow flow-name="xsl-region-body" column-width="2in">
                 <fo:block>x</fo:block>
               </fo:flow>
             </fo:page-sequence>
           </fo:root>"#,
        &options,
    )
    .expect("a lenient parse");
    // The value is recorded as specified even though refinement does not
    // compute an inapplicable property.
    let flow = first(&tree, FoElement::Flow);
    assert!(tree.node(flow).specifies(PropertyId::ColumnWidth));
}

#[test]
fn an_unknown_attribute_is_rejected_or_kept_as_foreign() {
    let error = FoDocument::parse(&wrap(r#"<fo:block nonesuch="1">x</fo:block>"#))
        .expect_err("`nonesuch` is not a property");
    assert!(matches!(error, FoError::UnknownProperty { .. }), "{error}");

    let options = ParseOptions {
        reject_unknown_properties: false,
        ..ParseOptions::default()
    };
    let tree = FoDocument::parse_with(&wrap(r#"<fo:block nonesuch="1">x</fo:block>"#), &options)
        .expect("a lenient parse");
    let block = last(&tree, FoElement::Block);
    assert_eq!(
        tree.node(block).foreign_attributes(),
        &[(None, "nonesuch".to_owned(), "1".to_owned())]
    );
}

#[test]
fn attributes_in_a_foreign_namespace_are_kept() {
    let tree = parse(r#"<fo:block xmlns:x="urn:x" x:hint="fast">text</fo:block>"#);
    let block = last(&tree, FoElement::Block);
    assert_eq!(
        tree.node(block).foreign_attributes(),
        &[(
            Some("urn:x".to_owned()),
            "hint".to_owned(),
            "fast".to_owned()
        )]
    );
}

#[test]
fn an_unknown_component_of_a_compound_property_is_rejected() {
    let error = FoDocument::parse(&wrap(r#"<fo:block space-before.middle="1pt">x</fo:block>"#))
        .expect_err("`middle` is not a component of <space>");
    assert!(matches!(error, FoError::UnknownProperty { .. }), "{error}");
}

#[test]
fn an_invalid_keyword_is_rejected() {
    let error = FoDocument::parse(&wrap(r#"<fo:block break-before="sideways">x</fo:block>"#))
        .expect_err("`sideways` is not a break-before keyword");
    assert!(
        matches!(error, FoError::InvalidPropertyValue { .. }),
        "{error}"
    );

    // `text-align`, by contrast, does accept a <string> (§7.16.9), so an
    // unrecognized token is kept rather than rejected.
    let tree = parse(r#"<fo:block text-align="sideways">x</fo:block>"#);
    assert_eq!(
        tree.properties(last(&tree, FoElement::Block))
            .value(PropertyId::TextAlign),
        &Value::Text("sideways".into())
    );
}

#[test]
fn malformed_xml_is_reported_with_a_position() {
    let error = FoDocument::parse(
        r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
             <fo:layout-master-set></fo:root>"#,
    )
    .expect_err("the document is not well-formed");
    assert!(matches!(error, FoError::Xml { .. }), "{error}");
}

#[test]
fn every_error_renders_a_useful_message() {
    let error = FoDocument::parse(&wrap(
        "<fo:block><fo:leader><fo:block>x</fo:block></fo:leader></fo:block>",
    ))
    .expect_err("invalid child");
    let message = error.to_string();
    assert!(message.contains("fo:block"), "{message}");
    assert!(message.contains("fo:leader"), "{message}");
    assert!(message.contains("byte"), "{message}");
}
