//! License management module

pub mod agreement;
mod license_type;
pub mod manage;
pub mod terms;

pub use agreement::LicenseAgreement;
pub use license_type::LicenseType;
pub use manage::LicenseManager;
pub use terms::LicenseTerms;
