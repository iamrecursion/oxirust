//! The legacy multi-byte Chinese and Korean encodings: GBK, gb18030, Big5, and EUC-KR.
//!
//! Each submodule implements one encoding family's decoder and encoder as a direct
//! transliteration of the corresponding section of the
//! [WHATWG Encoding Standard](https://encoding.spec.whatwg.org/) (§10-§13), sharing the
//! generated index tables in [`crate::tables`]. [`crate::decoder`] and [`crate::encoder`] are the
//! only callers of anything in this module — see [`crate::Encoding::Gbk`],
//! [`crate::Encoding::Gb18030`], [`crate::Encoding::Big5`], and [`crate::Encoding::EucKr`] for
//! the public entry points.

pub(crate) mod big5;
pub(crate) mod euc_kr;
pub(crate) mod gb18030;
