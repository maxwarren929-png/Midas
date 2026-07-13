use crate::{
    Result,
    config::{Config, FeatureKind, ObservationFeatureKind},
    execution::Portfolio,
    market::FeatureRow,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NormalizationRule {
    RatioClipped { minimum: f64, maximum: f64 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PortfolioFeatureSpec {
    CashAllocation,
    AssetAllocation,
    UnrealizedReturn,
    NetWorthToStarting,
    CurrentDrawdown,
    PortfolioReturn1,
    TurnoverToNetWorth,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AgentStateFeatureSpec {
    NormalizedAge,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationFeatureSpec {
    Market {
        feature: FeatureKind,
        normalization: NormalizationRule,
    },
    Portfolio {
        feature: PortfolioFeatureSpec,
        normalization: NormalizationRule,
    },
    AgentState {
        feature: AgentStateFeatureSpec,
        normalization: NormalizationRule,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationSchema {
    pub version: u32,
    pub market_schema_version: u32,
    pub features: Vec<ObservationFeatureSpec>,
}

impl ObservationSchema {
    pub fn from_config(config: &Config) -> Self {
        let mut features = Vec::new();
        for configured in &config.observation.features {
            match configured {
                ObservationFeatureKind::Market => {
                    for feature in &config.features.features {
                        features.push(ObservationFeatureSpec::Market {
                            feature: *feature,
                            normalization: NormalizationRule::RatioClipped {
                                minimum: -config.features.clip,
                                maximum: config.features.clip,
                            },
                        });
                    }
                }
                ObservationFeatureKind::CashAllocation => {
                    features.push(portfolio(PortfolioFeatureSpec::CashAllocation, 0.0, 1.0))
                }
                ObservationFeatureKind::AssetAllocation => {
                    features.push(portfolio(PortfolioFeatureSpec::AssetAllocation, 0.0, 1.0))
                }
                ObservationFeatureKind::UnrealizedReturn => features.push(portfolio(
                    PortfolioFeatureSpec::UnrealizedReturn,
                    -10.0,
                    10.0,
                )),
                ObservationFeatureKind::NetWorthToStarting => features.push(portfolio(
                    PortfolioFeatureSpec::NetWorthToStarting,
                    -10.0,
                    10.0,
                )),
                ObservationFeatureKind::CurrentDrawdown => {
                    features.push(portfolio(PortfolioFeatureSpec::CurrentDrawdown, 0.0, 1.0))
                }
                ObservationFeatureKind::PortfolioReturn1 => features.push(portfolio(
                    PortfolioFeatureSpec::PortfolioReturn1,
                    -10.0,
                    10.0,
                )),
                ObservationFeatureKind::TurnoverToNetWorth => features.push(portfolio(
                    PortfolioFeatureSpec::TurnoverToNetWorth,
                    0.0,
                    10.0,
                )),
                ObservationFeatureKind::NormalizedAge => {
                    features.push(ObservationFeatureSpec::AgentState {
                        feature: AgentStateFeatureSpec::NormalizedAge,
                        normalization: NormalizationRule::RatioClipped {
                            minimum: 0.0,
                            maximum: 1.0,
                        },
                    })
                }
            }
        }
        Self {
            version: config.observation.version,
            market_schema_version: config.features.schema_version,
            features,
        }
    }
    pub fn dimension(&self, asset_count: usize) -> usize {
        self.features
            .iter()
            .map(|spec| match spec {
                ObservationFeatureSpec::Market { .. } => asset_count,
                ObservationFeatureSpec::Portfolio {
                    feature: PortfolioFeatureSpec::AssetAllocation
                    | PortfolioFeatureSpec::UnrealizedReturn,
                    ..
                } => asset_count,
                _ => 1,
            })
            .sum()
    }
    pub fn validate(&self, asset_count: usize) -> Result<()> {
        if self.version == 0 || self.market_schema_version == 0 || asset_count == 0 {
            return Err(crate::MidasError::Config(
                "observation schema dimensions and versions must be positive".into(),
            ));
        }
        for spec in &self.features {
            let normalization = match spec {
                ObservationFeatureSpec::Market { normalization, .. }
                | ObservationFeatureSpec::Portfolio { normalization, .. }
                | ObservationFeatureSpec::AgentState { normalization, .. } => normalization,
            };
            let NormalizationRule::RatioClipped { minimum, maximum } = normalization;
            if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
                return Err(crate::MidasError::Config(
                    "observation normalization bounds are invalid".into(),
                ));
            }
        }
        Ok(())
    }
    pub fn identity(&self, asset_count: usize) -> Result<String> {
        self.validate(asset_count)?;
        Ok(blake3::hash(&serde_json::to_vec(&(self, asset_count))?)
            .to_hex()
            .to_string())
    }
}
fn portfolio(feature: PortfolioFeatureSpec, minimum: f64, maximum: f64) -> ObservationFeatureSpec {
    ObservationFeatureSpec::Portfolio {
        feature,
        normalization: NormalizationRule::RatioClipped { minimum, maximum },
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Observation {
    pub schema_hash: String,
    pub values: Vec<f32>,
}
pub struct ObservationContext<'a> {
    pub market: &'a FeatureRow,
    pub portfolio: &'a Portfolio,
    pub prices: &'a [Option<f64>],
    pub starting_net_worth: f64,
    pub maximum_net_worth: f64,
    pub age: u64,
    pub episode_steps: usize,
}

pub trait ObservationBuilder: Send + Sync {
    fn schema(&self) -> &ObservationSchema;
    fn build(&self, context: &ObservationContext<'_>) -> Result<Observation>;
}
pub struct SchemaObservationBuilder {
    schema: ObservationSchema,
    market_features: Vec<FeatureKind>,
    asset_count: usize,
    schema_hash: String,
}
impl SchemaObservationBuilder {
    pub fn new(
        schema: ObservationSchema,
        market_features: Vec<FeatureKind>,
        asset_count: usize,
    ) -> Result<Self> {
        schema.validate(asset_count)?;
        if market_features.is_empty() {
            return Err(crate::MidasError::Config(
                "observation market feature list must not be empty".into(),
            ));
        }
        let schema_hash = schema.identity(asset_count)?;
        Ok(Self {
            schema,
            market_features,
            asset_count,
            schema_hash,
        })
    }
}
impl ObservationBuilder for SchemaObservationBuilder {
    fn schema(&self) -> &ObservationSchema {
        &self.schema
    }
    fn build(&self, c: &ObservationContext<'_>) -> Result<Observation> {
        if c.market.values.len() != self.asset_count * self.market_features.len()
            || c.market.tradable.len() != self.asset_count
            || c.portfolio.positions.len() != self.asset_count
            || c.prices.len() != self.asset_count
            || !c.starting_net_worth.is_finite()
            || c.starting_net_worth <= 0.0
            || !c.maximum_net_worth.is_finite()
            || c.maximum_net_worth <= 0.0
        {
            return Err(crate::MidasError::Policy(
                "invalid observation context dimensions or bounds".into(),
            ));
        }
        let allocations = c.portfolio.allocations(c.prices)?;
        let mut values = Vec::with_capacity(self.schema.dimension(self.asset_count));
        for spec in &self.schema.features {
            match spec {
                ObservationFeatureSpec::Market {
                    feature,
                    normalization,
                } => {
                    let feature_index = self
                        .market_features
                        .iter()
                        .position(|f| f == feature)
                        .ok_or_else(|| {
                            crate::MidasError::Policy(format!(
                                "observation market feature {feature:?} is unavailable"
                            ))
                        })?;
                    for asset in 0..self.asset_count {
                        values.push(normalize(
                            c.market.values[asset * self.market_features.len() + feature_index]
                                as f64,
                            normalization,
                        )?);
                    }
                }
                ObservationFeatureSpec::Portfolio {
                    feature,
                    normalization,
                } => match feature {
                    PortfolioFeatureSpec::CashAllocation => {
                        values.push(normalize(allocations[0], normalization)?);
                    }
                    PortfolioFeatureSpec::AssetAllocation => {
                        for value in allocations.iter().skip(1) {
                            values.push(normalize(*value, normalization)?);
                        }
                    }
                    PortfolioFeatureSpec::UnrealizedReturn => {
                        for (position, price) in c.portfolio.positions.iter().zip(c.prices) {
                            let value =
                                if position.quantity > 0.0 && position.average_entry_price > 0.0 {
                                    price.unwrap_or(position.average_entry_price)
                                        / position.average_entry_price
                                        - 1.0
                                } else {
                                    0.0
                                };
                            values.push(normalize(value, normalization)?);
                        }
                    }
                    PortfolioFeatureSpec::NetWorthToStarting => values.push(normalize(
                        c.portfolio.net_worth / c.starting_net_worth,
                        normalization,
                    )?),
                    PortfolioFeatureSpec::CurrentDrawdown => values.push(normalize(
                        1.0 - c.portfolio.net_worth / c.maximum_net_worth,
                        normalization,
                    )?),
                    PortfolioFeatureSpec::PortfolioReturn1 => values.push(normalize(
                        c.portfolio.net_worth / c.portfolio.previous_net_worth.max(1e-9) - 1.0,
                        normalization,
                    )?),
                    PortfolioFeatureSpec::TurnoverToNetWorth => values.push(normalize(
                        c.portfolio.cumulative_turnover / c.portfolio.net_worth.max(1e-9),
                        normalization,
                    )?),
                },
                ObservationFeatureSpec::AgentState {
                    feature: AgentStateFeatureSpec::NormalizedAge,
                    normalization,
                } => values.push(normalize(
                    c.age as f64 / c.episode_steps.max(1) as f64,
                    normalization,
                )?),
            }
        }
        if values.len() != self.schema.dimension(self.asset_count)
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(crate::MidasError::Policy(
                "observation builder produced an invalid vector".into(),
            ));
        }
        Ok(Observation {
            schema_hash: self.schema_hash.clone(),
            values,
        })
    }
}

fn normalize(value: f64, rule: &NormalizationRule) -> Result<f32> {
    let NormalizationRule::RatioClipped { minimum, maximum } = rule;
    if !value.is_finite() {
        return Err(crate::MidasError::Policy(
            "observation feature is non-finite".into(),
        ));
    }
    let value = value.clamp(*minimum, *maximum) as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(crate::MidasError::Policy(
            "observation feature conversion is non-finite".into(),
        ))
    }
}
