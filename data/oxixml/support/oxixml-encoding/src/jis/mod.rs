//! The three legacy multi-byte Japanese encodings: Shift_JIS, EUC-JP, and ISO-2022-JP.
//!
//! Each submodule implements one encoding's decoder and encoder as a direct transliteration of
//! the corresponding section of the
//! [WHATWG Encoding Standard](https://encoding.spec.whatwg.org/) (§12), sharing the generated
//! JIS X 0208 / JIS X 0212 tables in [`crate::tables`]. [`crate::decoder`] and [`crate::encoder`]
//! are the only callers of anything in this module — see [`crate::Encoding::ShiftJis`],
//! [`crate::Encoding::EucJp`], and [`crate::Encoding::Iso2022Jp`] for the public entry points.

pub(crate) mod euc_jp;
pub(crate) mod iso2022jp;
pub(crate) mod shift_jis;
