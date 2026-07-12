use crate::{
    AssetId, MidasError, Result,
    config::{
        Config, CorporateActionConfig, CorporateActionMode, DataFormat, DateRange, FeatureKind,
    },
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MarketBar {
    pub date: NaiveDate,
    pub asset_id: AssetId,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub adjusted_close: f64,
    pub volume: f64,
    pub tradable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CorporateAction {
    Split {
        asset_id: AssetId,
        date: NaiveDate,
        ratio: f64,
    },
    CashDividend {
        asset_id: AssetId,
        date: NaiveDate,
        amount_per_share: f64,
    },
}
impl CorporateAction {
    pub fn asset_id(&self) -> AssetId {
        match self {
            Self::Split { asset_id, .. } | Self::CashDividend { asset_id, .. } => *asset_id,
        }
    }
    pub fn date(&self) -> NaiveDate {
        match self {
            Self::Split { date, .. } | Self::CashDividend { date, .. } => *date,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketDataset {
    pub assets: Vec<String>,
    pub bars: Vec<MarketBar>,
    pub dates: Vec<NaiveDate>,
    pub identity: String,
    pub corporate_actions: Vec<CorporateAction>,
    #[serde(skip)]
    index: BTreeMap<(NaiveDate, AssetId), usize>,
}

impl MarketDataset {
    pub fn load(config: &Config) -> Result<Self> {
        let frame = match config.data.format {
            DataFormat::Csv => CsvReadOptions::default()
                .with_has_header(true)
                .try_into_reader_with_file_path(Some(config.data.path.clone().into()))?
                .finish()?,
            DataFormat::Parquet => ParquetReader::new(File::open(&config.data.path)?).finish()?,
        };
        let actions = load_corporate_actions(config)?;
        Self::from_frame(
            frame,
            &config.market.assets,
            actions,
            &config.corporate_actions,
        )
    }

    pub fn from_frame(
        frame: DataFrame,
        required_assets: &[String],
        actions: Vec<CorporateAction>,
        corporate_config: &CorporateActionConfig,
    ) -> Result<Self> {
        let required = [
            "date",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ];
        for name in required {
            if frame.column(name).is_err() {
                return Err(MidasError::Data(format!(
                    "missing required column '{name}'"
                )));
            }
        }
        let dates = frame.column("date")?.str()?;
        let symbols = frame.column("asset")?.str()?;
        let opens_column = frame.column("open")?.cast(&DataType::Float64)?;
        let opens = opens_column.f64()?;
        let highs_column = frame.column("high")?.cast(&DataType::Float64)?;
        let highs = highs_column.f64()?;
        let lows_column = frame.column("low")?.cast(&DataType::Float64)?;
        let lows = lows_column.f64()?;
        let closes_column = frame.column("close")?.cast(&DataType::Float64)?;
        let closes = closes_column.f64()?;
        let adjusted_column = frame.column("adjusted_close")?.cast(&DataType::Float64)?;
        let adjusted = adjusted_column.f64()?;
        let volumes_column = frame.column("volume")?.cast(&DataType::Float64)?;
        let volumes = volumes_column.f64()?;
        let asset_lookup: HashMap<&str, AssetId> = required_assets
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), AssetId(i as u64)))
            .collect();
        let mut bars = Vec::with_capacity(frame.height());
        for row in 0..frame.height() {
            let symbol = symbols
                .get(row)
                .ok_or_else(|| MidasError::Data(format!("null asset at row {row}")))?;
            let Some(&asset_id) = asset_lookup.get(symbol) else {
                continue;
            };
            let date = NaiveDate::parse_from_str(
                dates
                    .get(row)
                    .ok_or_else(|| MidasError::Data(format!("null date at row {row}")))?,
                "%Y-%m-%d",
            )
            .map_err(|e| MidasError::Data(format!("invalid date at row {row}: {e}")))?;
            let get = |column: &Float64Chunked, name: &str| {
                column
                    .get(row)
                    .ok_or_else(|| MidasError::Data(format!("null {name} at row {row}")))
            };
            bars.push(MarketBar {
                date,
                asset_id,
                open: get(opens, "open")?,
                high: get(highs, "high")?,
                low: get(lows, "low")?,
                close: get(closes, "close")?,
                adjusted_close: get(adjusted, "adjusted_close")?,
                volume: get(volumes, "volume")?,
                tradable: true,
            });
        }
        Self::new_with_actions(required_assets.to_vec(), bars, actions, corporate_config)
    }

    pub fn new(assets: Vec<String>, bars: Vec<MarketBar>) -> Result<Self> {
        Self::new_with_actions(assets, bars, Vec::new(), &CorporateActionConfig::default())
    }

    pub fn new_with_actions(
        assets: Vec<String>,
        mut bars: Vec<MarketBar>,
        mut corporate_actions: Vec<CorporateAction>,
        corporate_config: &CorporateActionConfig,
    ) -> Result<Self> {
        for bar in &bars {
            let prices = [bar.open, bar.high, bar.low, bar.close, bar.adjusted_close];
            if prices.iter().any(|v| !v.is_finite() || *v <= 0.0)
                || !bar.volume.is_finite()
                || bar.volume < 0.0
            {
                return Err(MidasError::Data(format!(
                    "invalid numeric value for asset {} on {}",
                    bar.asset_id, bar.date
                )));
            }
            if bar.low > bar.open
                || bar.open > bar.high
                || bar.low > bar.close
                || bar.close > bar.high
            {
                return Err(MidasError::Data(format!(
                    "OHLC invariant failed for asset {} on {}",
                    bar.asset_id, bar.date
                )));
            }
            if bar.asset_id.0 as usize >= assets.len() {
                return Err(MidasError::Data("bar references unknown asset".into()));
            }
        }
        bars.sort_by_key(|b| (b.date, b.asset_id));
        if bars
            .windows(2)
            .any(|w| w[0].date == w[1].date && w[0].asset_id == w[1].asset_id)
        {
            return Err(MidasError::Data("duplicate asset/date row".into()));
        }
        let present: BTreeSet<_> = bars.iter().map(|b| b.asset_id).collect();
        if present.len() != assets.len() {
            return Err(MidasError::Data(
                "one or more required assets have no rows".into(),
            ));
        }
        let dates = bars
            .iter()
            .map(|b| b.date)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let index = bars
            .iter()
            .enumerate()
            .map(|(i, b)| ((b.date, b.asset_id), i))
            .collect();
        corporate_actions.sort_by_key(|action| (action.date(), action.asset_id()));
        validate_corporate_actions(&assets, &bars, &corporate_actions, corporate_config)?;
        let identity = blake3::hash(&bincode::serialize(&(
            assets.as_slice(),
            bars.as_slice(),
            corporate_actions.as_slice(),
        ))?)
        .to_hex()
        .to_string();
        Ok(Self {
            assets,
            bars,
            dates,
            identity,
            corporate_actions,
            index,
        })
    }

    pub fn rebuild_index(&mut self) {
        self.index = self
            .bars
            .iter()
            .enumerate()
            .map(|(i, b)| ((b.date, b.asset_id), i))
            .collect();
    }
    pub fn bar(&self, date: NaiveDate, asset: AssetId) -> Option<&MarketBar> {
        self.index.get(&(date, asset)).map(|i| &self.bars[*i])
    }
    pub fn split_dates(&self, range: &DateRange) -> Vec<NaiveDate> {
        self.dates
            .iter()
            .copied()
            .filter(|d| *d >= range.start && *d <= range.end)
            .collect()
    }
    pub fn last_close(&self, date: NaiveDate, asset: AssetId) -> Option<f64> {
        self.bars
            .iter()
            .rev()
            .find(|b| b.asset_id == asset && b.date <= date)
            .map(|b| b.close)
    }
    pub fn corporate_actions_on(&self, date: NaiveDate) -> impl Iterator<Item = &CorporateAction> {
        self.corporate_actions
            .iter()
            .filter(move |action| action.date() == date)
    }
    pub fn inspect(&self) -> DataInspection {
        DataInspection {
            assets: self.assets.clone(),
            rows: self.bars.len(),
            dates: self.dates.len(),
            first_date: self.dates.first().copied(),
            last_date: self.dates.last().copied(),
            identity: self.identity.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DataInspection {
    pub assets: Vec<String>,
    pub rows: usize,
    pub dates: usize,
    pub first_date: Option<NaiveDate>,
    pub last_date: Option<NaiveDate>,
    pub identity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FeatureSchema {
    pub version: u32,
    pub asset_count: usize,
    pub features: Vec<FeatureKind>,
}
impl FeatureSchema {
    pub fn market_size(&self) -> usize {
        self.asset_count * self.features.len()
    }
    pub fn identity(&self) -> Result<String> {
        Ok(blake3::hash(&serde_json::to_vec(self)?)
            .to_hex()
            .to_string())
    }
}

#[derive(Clone, Debug)]
pub struct FeatureRow {
    pub date: NaiveDate,
    pub values: Vec<f32>,
    pub tradable: Vec<bool>,
}

pub trait FeaturePipeline: Send + Sync {
    fn schema(&self) -> &FeatureSchema;
    fn generate(&self, data: &MarketDataset, dates: &[NaiveDate]) -> Result<Vec<FeatureRow>>;
}

pub struct BackwardFeaturePipeline {
    schema: FeatureSchema,
    clip: f64,
}
impl BackwardFeaturePipeline {
    pub fn from_config(config: &Config) -> Self {
        Self {
            schema: FeatureSchema {
                version: config.features.schema_version,
                asset_count: config.market.assets.len(),
                features: config.features.features.clone(),
            },
            clip: config.features.clip,
        }
    }
}

impl FeaturePipeline for BackwardFeaturePipeline {
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }
    fn generate(&self, data: &MarketDataset, dates: &[NaiveDate]) -> Result<Vec<FeatureRow>> {
        let mut history: Vec<Vec<FeaturePoint<'_>>> =
            (0..self.schema.asset_count).map(|_| Vec::new()).collect();
        let mut split_factor = vec![1.0; self.schema.asset_count];
        let wanted: BTreeSet<_> = dates.iter().copied().collect();
        let mut rows = Vec::new();
        for date in &data.dates {
            for action in data.corporate_actions_on(*date) {
                if let CorporateAction::Split {
                    asset_id, ratio, ..
                } = action
                {
                    split_factor[asset_id.0 as usize] *= ratio;
                }
            }
            for (asset, asset_history) in history.iter_mut().enumerate() {
                if let Some(bar) = data.bar(*date, AssetId(asset as u64)) {
                    asset_history.push(FeaturePoint {
                        bar,
                        price: bar.close * split_factor[asset],
                    });
                }
            }
            if !wanted.contains(date) {
                continue;
            }
            let mut values = Vec::with_capacity(self.schema.market_size());
            let mut tradable = Vec::with_capacity(self.schema.asset_count);
            for hist in &history {
                let live = hist
                    .last()
                    .is_some_and(|point| point.bar.date == *date && point.bar.tradable);
                tradable.push(live);
                for kind in &self.schema.features {
                    values.push(feature(kind, hist, self.clip) as f32);
                }
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(MidasError::Data(format!("non-finite feature on {date}")));
            }
            rows.push(FeatureRow {
                date: *date,
                values,
                tradable,
            });
        }
        Ok(rows)
    }
}

#[derive(Clone, Copy)]
struct FeaturePoint<'a> {
    bar: &'a MarketBar,
    price: f64,
}

fn feature(kind: &FeatureKind, h: &[FeaturePoint<'_>], clip: f64) -> f64 {
    let Some(now) = h.last() else { return 0.0 };
    let ret = |n: usize| {
        if h.len() > n {
            now.price / h[h.len() - 1 - n].price - 1.0
        } else {
            0.0
        }
    };
    let mean = |n: usize, volume: bool| {
        let s = &h[h.len().saturating_sub(n)..];
        if volume {
            s.iter().map(|point| point.bar.volume).sum::<f64>() / s.len() as f64
        } else {
            s.iter().map(|point| point.price).sum::<f64>() / s.len() as f64
        }
    };
    let raw = match kind {
        FeatureKind::Return1 => ret(1),
        FeatureKind::Return5 => ret(5),
        FeatureKind::Return20 => ret(20),
        FeatureKind::PriceToMa20 => now.price / mean(20, false) - 1.0,
        FeatureKind::PriceToMa50 => now.price / mean(50, false) - 1.0,
        FeatureKind::VolumeChange => {
            if h.len() > 1 && h[h.len() - 2].bar.volume > 0.0 {
                now.bar.volume / h[h.len() - 2].bar.volume - 1.0
            } else {
                0.0
            }
        }
        FeatureKind::VolumeToMa20 => {
            let m = mean(20, true);
            if m > 0.0 {
                now.bar.volume / m - 1.0
            } else {
                0.0
            }
        }
        FeatureKind::DistanceHigh20 => {
            let high = h[h.len().saturating_sub(20)..]
                .iter()
                .map(|point| point.price)
                .fold(0.0, f64::max);
            now.price / high - 1.0
        }
        FeatureKind::Tradable => {
            if now.bar.tradable {
                1.0
            } else {
                0.0
            }
        }
        FeatureKind::Volatility20 => {
            let s = &h[h.len().saturating_sub(21)..];
            if s.len() < 2 {
                0.0
            } else {
                let rs = s
                    .windows(2)
                    .map(|w| w[1].price / w[0].price - 1.0)
                    .collect::<Vec<_>>();
                let m = rs.iter().sum::<f64>() / rs.len() as f64;
                (rs.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / rs.len() as f64).sqrt()
            }
        }
    };
    raw.clamp(-clip, clip)
}

pub trait MarketDataSource {
    fn load(&self, config: &Config) -> Result<MarketDataset>;
}

#[derive(Deserialize)]
struct CorporateActionRecord {
    date: String,
    asset: String,
    kind: String,
    value: f64,
}

fn load_corporate_actions(config: &Config) -> Result<Vec<CorporateAction>> {
    let Some(path) = &config.data.corporate_actions_path else {
        return Ok(Vec::new());
    };
    let mut reader = csv::Reader::from_path(path)?;
    let asset_lookup: HashMap<&str, AssetId> = config
        .market
        .assets
        .iter()
        .enumerate()
        .map(|(i, asset)| (asset.as_str(), AssetId(i as u64)))
        .collect();
    let mut actions = Vec::new();
    for record in reader.deserialize::<CorporateActionRecord>() {
        let record = record?;
        let asset_id = *asset_lookup.get(record.asset.as_str()).ok_or_else(|| {
            MidasError::Data(format!(
                "corporate action references unknown asset {}",
                record.asset
            ))
        })?;
        let date = NaiveDate::parse_from_str(&record.date, "%Y-%m-%d")
            .map_err(|e| MidasError::Data(format!("invalid corporate action date: {e}")))?;
        let action = match record.kind.as_str() {
            "split" => CorporateAction::Split {
                asset_id,
                date,
                ratio: record.value,
            },
            "cash_dividend" => CorporateAction::CashDividend {
                asset_id,
                date,
                amount_per_share: record.value,
            },
            other => {
                return Err(MidasError::Data(format!(
                    "unsupported corporate action kind {other}"
                )));
            }
        };
        actions.push(action);
    }
    Ok(actions)
}

fn validate_corporate_actions(
    assets: &[String],
    bars: &[MarketBar],
    actions: &[CorporateAction],
    config: &CorporateActionConfig,
) -> Result<()> {
    if config.mode == CorporateActionMode::Reject && !actions.is_empty() {
        return Err(MidasError::Data(
            "corporate-action records are forbidden in reject mode".into(),
        ));
    }
    for action in actions {
        if action.asset_id().0 as usize >= assets.len() {
            return Err(MidasError::Data(
                "corporate action references unknown asset".into(),
            ));
        }
        match action {
            CorporateAction::Split { ratio, .. } if !ratio.is_finite() || *ratio <= 0.0 => {
                return Err(MidasError::Data(
                    "split ratio must be finite and positive".into(),
                ));
            }
            CorporateAction::CashDividend {
                amount_per_share, ..
            } if !amount_per_share.is_finite() || *amount_per_share < 0.0 => {
                return Err(MidasError::Data(
                    "cash dividend must be finite and non-negative".into(),
                ));
            }
            _ => {}
        }
    }
    for bar in bars {
        if ((bar.adjusted_close - bar.close) / bar.close).abs() > config.adjusted_price_tolerance {
            return Err(MidasError::Data(format!(
                "provider-adjusted close is not point-in-time safe for asset {} on {}; supply raw close values",
                bar.asset_id, bar.date
            )));
        }
    }
    let mut previous: Vec<Option<&MarketBar>> = vec![None; assets.len()];
    for bar in bars {
        if let Some(prior) = previous[bar.asset_id.0 as usize] {
            let change = (bar.open / prior.close - 1.0).abs();
            let declared_split = actions.iter().any(|action| matches!(action, CorporateAction::Split { asset_id, date, .. } if *asset_id == bar.asset_id && *date == bar.date));
            if change > config.discontinuity_threshold && !declared_split {
                return Err(MidasError::Data(format!(
                    "likely unsupported corporate action for asset {} on {}: {:.2}% discontinuity",
                    bar.asset_id,
                    bar.date,
                    change * 100.0
                )));
            }
        }
        previous[bar.asset_id.0 as usize] = Some(bar);
    }
    Ok(())
}
pub struct PolarsDataSource;
impl MarketDataSource for PolarsDataSource {
    fn load(&self, config: &Config) -> Result<MarketDataset> {
        MarketDataset::load(config)
    }
}

pub fn write_parquet(dataset: &MarketDataset, path: &Path) -> Result<()> {
    let mut frame = df!("date"=>dataset.bars.iter().map(|b|b.date.to_string()).collect::<Vec<_>>(),"asset"=>dataset.bars.iter().map(|b|dataset.assets[b.asset_id.0 as usize].clone()).collect::<Vec<_>>(),"open"=>dataset.bars.iter().map(|b|b.open).collect::<Vec<_>>(),"high"=>dataset.bars.iter().map(|b|b.high).collect::<Vec<_>>(),"low"=>dataset.bars.iter().map(|b|b.low).collect::<Vec<_>>(),"close"=>dataset.bars.iter().map(|b|b.close).collect::<Vec<_>>(),"adjusted_close"=>dataset.bars.iter().map(|b|b.adjusted_close).collect::<Vec<_>>(),"volume"=>dataset.bars.iter().map(|b|b.volume).collect::<Vec<_>>())?;
    ParquetWriter::new(File::create(path)?).finish(&mut frame)?;
    Ok(())
}
