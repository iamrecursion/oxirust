//! Shared plumbing for the subcommands that load an XML document into a DOM tree before doing
//! something with it (`fmt`, `lint`'s well-formedness-only cousins, `validate`, `canon`).

use std::io::Read;

use oxixml_catalog::Catalog;
use oxixml_dom::{BuilderOptions, Document, build};

use crate::catalog_support;
use crate::global;
use crate::io_util;

/// Loads the OASIS XML Catalog `globals.catalog` names, if one was given.
///
/// # Errors
///
/// A human-readable message when the catalog cannot be read or parsed.
pub fn load_catalog(
    globals: &global::Options,
    stdin: &mut dyn Read,
) -> Result<Option<Catalog>, String> {
    match &globals.catalog {
        Some(path) => catalog_support::load(path, stdin).map(Some),
        None => Ok(None),
    }
}

/// Reads `input_path` and parses it into a [`Document`], applying XInclude processing when
/// `globals.xinclude` is set.
///
/// # Errors
///
/// A human-readable message when the input cannot be read, is not well-formed, or XInclude
/// processing fails.
pub fn load_document(
    input_path: &str,
    globals: &global::Options,
    catalog: Option<&Catalog>,
    check_well_formedness: bool,
    stdin: &mut dyn Read,
) -> Result<Document, String> {
    let text = io_util::read_string(input_path, stdin).map_err(|error| {
        format!(
            "cannot read `{}`: {error}",
            io_util::display_path(input_path)
        )
    })?;
    let options = BuilderOptions::new().with_well_formedness_checks(check_well_formedness);
    let mut document = build::parse_str_with(&text, &options)
        .map_err(|error| format!("{}: {error}", io_util::display_path(input_path)))?;
    if globals.xinclude {
        let loader = catalog_support::XIncludeLoader::new(catalog.cloned());
        let base = globals
            .base
            .clone()
            .or_else(|| io_util::file_base_uri(input_path))
            .unwrap_or_else(|| "file:///".to_owned());
        oxixml_xinclude::process_document(&mut document, &base, &loader).map_err(|error| {
            format!(
                "XInclude processing of `{}` failed: {error}",
                io_util::display_path(input_path)
            )
        })?;
    }
    Ok(document)
}
