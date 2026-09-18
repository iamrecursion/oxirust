//! Build-time reflection of `src/constants.rs` into a flat `LIST_CONSTANTS`
//! table.
//!
//! Eliminates drift between the Rust constants and any tool that needs to
//! enumerate them (e.g. `oximedia-cv2 --list-constants`). The generated
//! file is `$OUT_DIR/constants_list.rs` and is included from
//! `src/lib.rs` via `pub mod constants_list { include!(...); }`.
//!
//! The emitted table is a `&[(&str, &str, &str, &str)]` of tuples
//! `(category, name, type, value)` — splitting category from the bare
//! constant name lets the binary group the listing by sub-module while
//! still enabling alphabetical sort. Top-level (non-`pub mod`) constants
//! get the empty category string `""`.

use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

use quote::ToTokens;
use syn::{parse_file, Expr, Item, Lit, UnOp};

fn main() -> io::Result<()> {
    println!("cargo:rerun-if-changed=src/constants.rs");
    println!("cargo:rerun-if-changed=build.rs");

    let src = fs::read_to_string("src/constants.rs")?;
    let ast = parse_file(&src).map_err(|e| io::Error::other(format!("syn parse error: {e}")))?;

    let mut entries: Vec<(String, String, String, String)> = Vec::new();
    collect_consts(&ast.items, "", &mut entries);

    // Sort first by category (top-level "" sorts before module names),
    // then by name within each category for deterministic build output.
    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut out = String::from(
        "/// Auto-generated list of `pub const` items reflected from `src/constants.rs`\n\
         /// at build time. Each entry is `(category, name, type, value-as-string)`.\n\
         ///\n\
         /// `category` is the immediate enclosing `pub mod` name, or the empty\n\
         /// string for items declared directly in `constants.rs`.\n\
         pub static LIST_CONSTANTS: &[(&str, &str, &str, &str)] = &[\n",
    );
    for (category, name, ty, val) in &entries {
        out.push_str(&format!("    ({category:?}, {name:?}, {ty:?}, {val:?}),\n"));
    }
    out.push_str("];\n");

    let out_dir = env::var("OUT_DIR").map_err(|_| io::Error::other("OUT_DIR not set by Cargo"))?;
    let out_path = PathBuf::from(out_dir).join("constants_list.rs");
    fs::write(out_path, out)?;
    Ok(())
}

/// Recursively walk a list of `Item`s, harvesting `pub const` declarations.
///
/// `prefix` is the immediate enclosing `pub mod` name (last segment only),
/// or empty for the file root. The `constants` module only nests one level
/// deep, so a single segment is the useful "category" for grouped display.
fn collect_consts(
    items: &[Item],
    prefix: &str,
    entries: &mut Vec<(String, String, String, String)>,
) {
    for item in items {
        match item {
            Item::Const(c) if matches!(c.vis, syn::Visibility::Public(_)) => {
                let name = c.ident.to_string();
                let ty = c.ty.to_token_stream().to_string();
                let val = format_value_expr(&c.expr);
                entries.push((prefix.to_string(), name, ty, val));
            }
            Item::Mod(m) if matches!(m.vis, syn::Visibility::Public(_)) => {
                if let Some((_, ref nested)) = m.content {
                    let new_prefix = m.ident.to_string();
                    collect_consts(nested, &new_prefix, entries);
                }
            }
            _ => {}
        }
    }
}

/// Render a constant's `expr` as a clean string.
///
/// `quote!()` emits a token stream with whitespace between every token,
/// so a unary `-1` literal comes back as `"- 1"`. For integer constants
/// — the only kind in this file — we recognise plain literals and
/// `-literal` patterns and fold them to their canonical form. Anything
/// else falls back to the raw `quote!()` rendering.
fn format_value_expr(expr: &Expr) -> String {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Int(int) => int.base10_digits().to_string(),
            other => other.to_token_stream().to_string(),
        },
        Expr::Unary(unary) => {
            if matches!(unary.op, UnOp::Neg(_)) {
                if let Expr::Lit(lit) = unary.expr.as_ref() {
                    if let Lit::Int(int) = &lit.lit {
                        return format!("-{}", int.base10_digits());
                    }
                }
            }
            expr.to_token_stream().to_string()
        }
        _ => expr.to_token_stream().to_string(),
    }
}
