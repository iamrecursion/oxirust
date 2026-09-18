//! Loss functions for training (standard and logical constraint-based).

mod bcewithlogitsloss_traits;
mod constraintviolationloss_traits;
mod contrastiveloss_traits;
mod crossentropyloss_traits;
mod diceloss_traits;
mod focalloss_traits;
mod functions;
mod hingeloss_traits;
mod huberloss_traits;
mod kldivergenceloss_traits;
mod lossconfig_traits;
mod mseloss_traits;
mod polyloss_traits;
mod rulesatisfactionloss_traits;
mod tripletloss_traits;
mod tverskyloss_traits;
mod types;

pub use functions::Loss;
pub use types::{
    BCEWithLogitsLoss, ConstraintViolationLoss, ContrastiveLoss, CrossEntropyLoss, DiceLoss,
    FocalLoss, HingeLoss, HuberLoss, KLDivergenceLoss, LogicalLoss, LossConfig, MseLoss, PolyLoss,
    RuleSatisfactionLoss, TripletLoss, TverskyLoss,
};
