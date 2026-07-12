use crate::{AssetId, MidasError, Result, config::ExecutionConfig, market::MarketBar};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub asset_id: AssetId,
    pub quantity: f64,
    pub average_entry_price: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Portfolio {
    pub cash: f64,
    pub positions: Vec<Position>,
    pub net_worth: f64,
    pub previous_net_worth: f64,
    pub liabilities: f64,
    pub cumulative_commission: f64,
    pub cumulative_spread: f64,
    pub cumulative_slippage: f64,
    pub cumulative_turnover: f64,
    pub cumulative_survival_cost: f64,
}
impl Portfolio {
    pub fn new(cash: f64, assets: usize) -> Self {
        Self {
            cash,
            positions: (0..assets)
                .map(|i| Position {
                    asset_id: AssetId(i as u64),
                    quantity: 0.0,
                    average_entry_price: 0.0,
                })
                .collect(),
            net_worth: cash,
            previous_net_worth: cash,
            liabilities: 0.0,
            cumulative_commission: 0.0,
            cumulative_spread: 0.0,
            cumulative_slippage: 0.0,
            cumulative_turnover: 0.0,
            cumulative_survival_cost: 0.0,
        }
    }
    pub fn value(&self, closes: &[Option<f64>]) -> Result<f64> {
        let mut total = self.cash - self.liabilities;
        for (p, price) in self.positions.iter().zip(closes) {
            if p.quantity < 0.0 || !p.quantity.is_finite() {
                return Err(MidasError::Accounting(
                    "negative or non-finite holding".into(),
                ));
            }
            if p.quantity > 0.0 {
                total += p.quantity
                    * price.ok_or_else(|| {
                        MidasError::Accounting(format!(
                            "no historical valuation for asset {}",
                            p.asset_id
                        ))
                    })?
            }
        }
        if !total.is_finite() {
            return Err(MidasError::Accounting("non-finite portfolio value".into()));
        }
        Ok(total)
    }
    pub fn allocations(&self, prices: &[Option<f64>]) -> Result<Vec<f64>> {
        let gross = self.cash
            + self
                .positions
                .iter()
                .zip(prices)
                .map(|(p, v)| p.quantity * v.unwrap_or(0.0))
                .sum::<f64>();
        if gross <= 0.0 {
            return Ok(std::iter::once(1.0)
                .chain(std::iter::repeat_n(0.0, self.positions.len()))
                .collect());
        }
        let mut a = vec![self.cash / gross];
        a.extend(
            self.positions
                .iter()
                .zip(prices)
                .map(|(p, v)| p.quantity * v.unwrap_or(0.0) / gross),
        );
        Ok(a)
    }
    pub fn apply_survival_cost(&mut self, cost: f64) {
        let paid = self.cash.min(cost);
        self.cash -= paid;
        self.liabilities += cost - paid;
        self.cumulative_survival_cost += cost;
        self.net_worth -= cost;
    }
    pub fn check(&self) -> Result<()> {
        if !self.cash.is_finite() || self.cash < -1e-8 {
            return Err(MidasError::Accounting(format!(
                "cash invariant failed: {}",
                self.cash
            )));
        }
        if self
            .positions
            .iter()
            .any(|p| p.quantity < -1e-10 || !p.quantity.is_finite())
        {
            return Err(MidasError::Accounting("holding invariant failed".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Trade {
    pub asset_id: AssetId,
    pub side: TradeSide,
    pub quantity: f64,
    pub base_price: f64,
    pub execution_price: f64,
    pub commission: f64,
    pub spread_cost: f64,
    pub slippage_cost: f64,
    pub notional: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExecutionReport {
    pub trades: Vec<Trade>,
    pub rejected_assets: Vec<AssetId>,
    pub invalid_action: bool,
}

pub trait ExecutionModel: Send + Sync {
    fn rebalance(
        &self,
        portfolio: &mut Portfolio,
        target: &[f64],
        open_bars: &[Option<&MarketBar>],
        valuation_prices: &[Option<f64>],
        minimum_trade: f64,
    ) -> Result<ExecutionReport>;
}

#[derive(Clone)]
pub struct CostExecutionModel {
    pub costs: ExecutionConfig,
}
impl CostExecutionModel {
    fn execute_sell(
        &self,
        p: &mut Portfolio,
        i: usize,
        value: f64,
        bar: &MarketBar,
        report: &mut ExecutionReport,
    ) {
        let qty = (value / bar.open).min(p.positions[i].quantity);
        if qty <= 0.0 {
            return;
        }
        let half = self.costs.spread_rate / 2.0;
        let price = bar.open * (1.0 - half - self.costs.slippage_rate);
        let proceeds = qty * price;
        let commission = (self.costs.commission_fixed
            + self.costs.commission_rate * qty * bar.open)
            .min(proceeds);
        p.cash += proceeds - commission;
        p.positions[i].quantity -= qty;
        if p.positions[i].quantity < 1e-12 {
            p.positions[i].quantity = 0.0;
            p.positions[i].average_entry_price = 0.0
        }
        let spread = qty * bar.open * half;
        let slippage = qty * bar.open * self.costs.slippage_rate;
        record_costs(p, commission, spread, slippage, qty * bar.open);
        report.trades.push(Trade {
            asset_id: bar.asset_id,
            side: TradeSide::Sell,
            quantity: qty,
            base_price: bar.open,
            execution_price: price,
            commission,
            spread_cost: spread,
            slippage_cost: slippage,
            notional: qty * bar.open,
        });
    }
    fn execute_buy(
        &self,
        p: &mut Portfolio,
        i: usize,
        value: f64,
        bar: &MarketBar,
        report: &mut ExecutionReport,
    ) {
        let half = self.costs.spread_rate / 2.0;
        let price = bar.open * (1.0 + half + self.costs.slippage_rate);
        let max_notional = ((p.cash - self.costs.commission_fixed).max(0.0)
            / (1.0 + self.costs.commission_rate + half + self.costs.slippage_rate))
            .max(0.0);
        let notional = value.min(max_notional);
        if notional <= 0.0 {
            return;
        }
        let qty = notional / bar.open;
        let commission = self.costs.commission_fixed + self.costs.commission_rate * notional;
        let spend = qty * price + commission;
        if spend > p.cash + 1e-8 {
            return;
        }
        let old = p.positions[i].quantity;
        let new = old + qty;
        p.positions[i].average_entry_price = if new > 0.0 {
            (old * p.positions[i].average_entry_price + qty * price) / new
        } else {
            0.0
        };
        p.positions[i].quantity = new;
        p.cash = (p.cash - spend).max(0.0);
        let spread = notional * half;
        let slippage = notional * self.costs.slippage_rate;
        record_costs(p, commission, spread, slippage, notional);
        report.trades.push(Trade {
            asset_id: bar.asset_id,
            side: TradeSide::Buy,
            quantity: qty,
            base_price: bar.open,
            execution_price: price,
            commission,
            spread_cost: spread,
            slippage_cost: slippage,
            notional,
        });
    }
}
fn record_costs(p: &mut Portfolio, commission: f64, spread: f64, slippage: f64, turnover: f64) {
    p.cumulative_commission += commission;
    p.cumulative_spread += spread;
    p.cumulative_slippage += slippage;
    p.cumulative_turnover += turnover;
}

impl ExecutionModel for CostExecutionModel {
    fn rebalance(
        &self,
        p: &mut Portfolio,
        target: &[f64],
        bars: &[Option<&MarketBar>],
        valuation_prices: &[Option<f64>],
        minimum: f64,
    ) -> Result<ExecutionReport> {
        let mut report = ExecutionReport::default();
        let n = p.positions.len();
        let valid = target.len() == n + 1
            && target.iter().all(|v| v.is_finite() && *v >= 0.0)
            && ((target.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        let fallback = std::iter::once(1.0)
            .chain(std::iter::repeat_n(0.0, n))
            .collect::<Vec<_>>();
        let target = if valid {
            target
        } else {
            report.invalid_action = true;
            &fallback
        };
        let opens = bars
            .iter()
            .zip(valuation_prices)
            .map(|(b, fallback)| b.map(|x| x.open).or(*fallback))
            .collect::<Vec<_>>();
        let worth = p.value(&opens)?;
        let current = p
            .positions
            .iter()
            .zip(&opens)
            .map(|(pos, price)| pos.quantity * price.unwrap_or(0.0))
            .collect::<Vec<_>>();
        for i in 0..n {
            let desired = worth * target[i + 1];
            if desired + minimum < current[i] {
                if let Some(bar) = bars[i].filter(|b| b.tradable) {
                    self.execute_sell(p, i, current[i] - desired, bar, &mut report)
                } else {
                    report.rejected_assets.push(AssetId(i as u64))
                }
            }
        }
        let requests = (0..n)
            .map(|i| {
                let desired = worth * target[i + 1];
                if desired > current[i] + minimum {
                    desired - current[i]
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let total = requests
            .iter()
            .enumerate()
            .filter(|(i, _)| bars[*i].is_some_and(|b| b.tradable))
            .map(|(_, v)| *v)
            .sum::<f64>();
        let scale = if total > 0.0 {
            (p.cash / total).min(1.0)
        } else {
            0.0
        };
        for (i, request) in requests.into_iter().enumerate() {
            if request <= 0.0 {
                continue;
            }
            if let Some(bar) = bars[i].filter(|b| b.tradable) {
                self.execute_buy(p, i, request * scale, bar, &mut report)
            } else {
                report.rejected_assets.push(AssetId(i as u64))
            }
        }
        p.check()?;
        Ok(report)
    }
}
