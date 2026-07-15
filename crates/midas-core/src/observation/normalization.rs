use crate::{
    Result,
    config::{ObservationNormalizationConfig, ObservationNormalizationKind},
};

use crate::MidasError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationRule {
    None,
    /// Historical Prompt 4 behaviour: clamp an already ratio-like value.
    RatioClipped {
        minimum: f64,
        maximum: f64,
    },
    MinMax {
        minimum: f64,
        maximum: f64,
    },
    ZScore {
        mean: f64,
        standard_deviation: f64,
    },
    Log,
    SignedLog,
    Percentage {
        scale: f64,
    },
    /// An explicit finite affine transform for research-only schema entries.
    Custom {
        scale: f64,
        offset: f64,
    },
}

impl NormalizationRule {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::None | Self::Log | Self::SignedLog => Ok(()),
            Self::RatioClipped { minimum, maximum } | Self::MinMax { minimum, maximum } => {
                if minimum.is_finite() && maximum.is_finite() && minimum < maximum {
                    Ok(())
                } else {
                    Err(MidasError::Config(
                        "observation normalization bounds are invalid".into(),
                    ))
                }
            }
            Self::ZScore {
                mean,
                standard_deviation,
            } => {
                if mean.is_finite() && standard_deviation.is_finite() && *standard_deviation > 0.0 {
                    Ok(())
                } else {
                    Err(MidasError::Config(
                        "observation z-score normalization is invalid".into(),
                    ))
                }
            }
            Self::Percentage { scale } => {
                if scale.is_finite() && *scale != 0.0 {
                    Ok(())
                } else {
                    Err(MidasError::Config(
                        "observation percentage normalization scale is invalid".into(),
                    ))
                }
            }
            Self::Custom { scale, offset } => {
                if scale.is_finite() && offset.is_finite() {
                    Ok(())
                } else {
                    Err(MidasError::Config(
                        "observation custom normalization is invalid".into(),
                    ))
                }
            }
        }
    }
}

pub(crate) fn apply_override(
    default: &NormalizationRule,
    name: &str,
    overrides: &std::collections::BTreeMap<String, ObservationNormalizationConfig>,
) -> Result<NormalizationRule> {
    match overrides.get(name) {
        Some(config) => normalization_from_config(config),
        None => Ok(default.clone()),
    }
}

pub(crate) fn normalization_from_config(
    config: &ObservationNormalizationConfig,
) -> Result<NormalizationRule> {
    config.validate()?;
    let missing =
        |field: &str| MidasError::Config(format!("observation normalization is missing {field}"));
    Ok(match config.kind {
        ObservationNormalizationKind::None => NormalizationRule::None,
        ObservationNormalizationKind::RatioClipped => NormalizationRule::RatioClipped {
            minimum: config.minimum.ok_or_else(|| missing("minimum"))?,
            maximum: config.maximum.ok_or_else(|| missing("maximum"))?,
        },
        ObservationNormalizationKind::MinMax => NormalizationRule::MinMax {
            minimum: config.minimum.ok_or_else(|| missing("minimum"))?,
            maximum: config.maximum.ok_or_else(|| missing("maximum"))?,
        },
        ObservationNormalizationKind::ZScore => NormalizationRule::ZScore {
            mean: config.mean.ok_or_else(|| missing("mean"))?,
            standard_deviation: config
                .standard_deviation
                .ok_or_else(|| missing("standard_deviation"))?,
        },
        ObservationNormalizationKind::Log => NormalizationRule::Log,
        ObservationNormalizationKind::SignedLog => NormalizationRule::SignedLog,
        ObservationNormalizationKind::Percentage => NormalizationRule::Percentage {
            scale: config.scale.ok_or_else(|| missing("scale"))?,
        },
        ObservationNormalizationKind::Custom => NormalizationRule::Custom {
            scale: config.scale.ok_or_else(|| missing("scale"))?,
            offset: config.offset.ok_or_else(|| missing("offset"))?,
        },
    })
}

pub(crate) fn normalize(value: f64, rule: &NormalizationRule) -> Result<f32> {
    if !value.is_finite() {
        return Err(MidasError::Policy(
            "observation feature is non-finite".into(),
        ));
    }
    let normalized = match rule {
        NormalizationRule::None => value,
        NormalizationRule::RatioClipped { minimum, maximum } => value.clamp(*minimum, *maximum),
        NormalizationRule::MinMax { minimum, maximum } => {
            ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
        }
        NormalizationRule::ZScore {
            mean,
            standard_deviation,
        } => (value - mean) / standard_deviation,
        NormalizationRule::Log => {
            if value <= 0.0 {
                return Err(MidasError::Policy(
                    "log-normalized observation must be positive".into(),
                ));
            }
            value.ln()
        }
        NormalizationRule::SignedLog => value.signum() * value.abs().ln_1p(),
        NormalizationRule::Percentage { scale } => value * scale,
        NormalizationRule::Custom { scale, offset } => value * scale + offset,
    };
    let normalized = normalized as f32;
    if normalized.is_finite() {
        Ok(normalized)
    } else {
        Err(MidasError::Policy(
            "observation feature conversion is non-finite".into(),
        ))
    }
}
