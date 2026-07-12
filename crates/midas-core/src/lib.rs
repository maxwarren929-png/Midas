pub mod brain;
pub mod config;
pub mod execution;
pub mod market;
pub mod metrics;
pub mod simulation;

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            Eq,
            Hash,
            Ord,
            PartialEq,
            PartialOrd,
            Serialize,
            Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u64);
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}
id_type!(AssetId);
id_type!(AgentId);
id_type!(ExperimentId);

pub type Result<T> = std::result::Result<T, MidasError>;

#[derive(Debug, thiserror::Error)]
pub enum MidasError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("market data error: {0}")]
    Data(String),
    #[error("accounting invariant failed: {0}")]
    Accounting(String),
    #[error("policy error: {0}")]
    Policy(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),
    #[error(transparent)]
    Toml(#[from] toml::de::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Bincode(#[from] Box<bincode::ErrorKind>),
}
