use crate::{
    AssetId, MidasError, Result,
    action::{ValidatedAction, ValidatedAllocation, ValidatedTradeImpulses},
    config::ExecutionConfig,
    market::{CorporateAction, MarketBar},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
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
    pub fn value(&self, prices: &[Option<f64>]) -> Result<f64> {
        if prices.len() != self.positions.len() {
            return Err(MidasError::Accounting(
                "valuation price length mismatch".into(),
            ));
        }
        let mut total = self.cash - self.liabilities;
        for (p, price) in self.positions.iter().zip(prices) {
            if p.quantity < 0.0 || !p.quantity.is_finite() {
                return Err(MidasError::Accounting("invalid holding".into()));
            }
            if let Some(price) = price
                && (!price.is_finite() || *price <= 0.0)
            {
                return Err(MidasError::Accounting("invalid valuation price".into()));
            }
            if p.quantity > 0.0 {
                total += p.quantity
                    * price.ok_or_else(|| {
                        MidasError::Accounting(format!("no valuation for asset {}", p.asset_id))
                    })?;
            }
        }
        if !total.is_finite() {
            return Err(MidasError::Accounting("non-finite portfolio value".into()));
        }
        Ok(total)
    }
    pub fn allocations(&self, prices: &[Option<f64>]) -> Result<Vec<f64>> {
        if prices.len() != self.positions.len() {
            return Err(MidasError::Accounting(
                "allocation price length mismatch".into(),
            ));
        }
        let mut gross = self.cash;
        for (position, price) in self.positions.iter().zip(prices) {
            if position.quantity < 0.0 || !position.quantity.is_finite() {
                return Err(MidasError::Accounting("invalid holding".into()));
            }
            if let Some(price) = price {
                if !price.is_finite() || *price <= 0.0 {
                    return Err(MidasError::Accounting("invalid allocation price".into()));
                }
                gross += position.quantity * price;
            }
        }
        if !gross.is_finite() {
            return Err(MidasError::Accounting("non-finite allocation value".into()));
        }
        if gross <= 0.0 {
            let mut w = vec![0.0; self.positions.len() + 1];
            w[0] = 1.0;
            return Ok(w);
        }
        let mut w = vec![self.cash / gross];
        w.extend(
            self.positions
                .iter()
                .zip(prices)
                .map(|(p, v)| p.quantity * v.unwrap_or(0.0) / gross),
        );
        Ok(w)
    }
    pub fn apply_survival_cost(&mut self, cost: f64) {
        let paid = self.cash.min(cost);
        self.cash -= paid;
        self.liabilities += cost - paid;
        self.cumulative_survival_cost += cost;
        self.net_worth -= cost;
    }
    pub fn apply_corporate_action(&mut self, action: &CorporateAction) -> Result<()> {
        match action {
            CorporateAction::Split {
                asset_id, ratio, ..
            } => {
                if !ratio.is_finite() || *ratio <= 0.0 {
                    return Err(MidasError::Accounting("invalid split ratio".into()));
                }
                let p = self
                    .positions
                    .iter_mut()
                    .find(|position| position.asset_id == *asset_id)
                    .ok_or_else(|| {
                        MidasError::Accounting(format!("unknown split asset {asset_id}"))
                    })?;
                p.quantity *= ratio;
                if p.average_entry_price > 0.0 {
                    p.average_entry_price /= ratio;
                }
            }
            CorporateAction::CashDividend {
                asset_id,
                amount_per_share,
                ..
            } => {
                if !amount_per_share.is_finite() || *amount_per_share < 0.0 {
                    return Err(MidasError::Accounting("invalid dividend".into()));
                }
                let quantity = self
                    .positions
                    .iter()
                    .find(|position| position.asset_id == *asset_id)
                    .ok_or_else(|| {
                        MidasError::Accounting(format!("unknown dividend asset {asset_id}"))
                    })?
                    .quantity;
                self.cash += quantity * amount_per_share;
            }
        }
        self.check()
    }
    pub fn check(&self) -> Result<()> {
        let mut asset_ids = BTreeSet::new();
        if !self.cash.is_finite()
            || self.cash < -1e-8
            || !self.liabilities.is_finite()
            || self.liabilities < 0.0
            || self.positions.iter().any(|p| {
                !asset_ids.insert(p.asset_id)
                    || p.quantity < -1e-10
                    || !p.quantity.is_finite()
                    || !p.average_entry_price.is_finite()
                    || p.average_entry_price < 0.0
            })
        {
            return Err(MidasError::Accounting("portfolio invariant failed".into()));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct ExecutionSnapshot<'a> {
    pub bars: &'a [Option<&'a MarketBar>],
    pub valuation_prices: &'a [Option<f64>],
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlannedTrade {
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
pub struct ExecutionPlan {
    pub trades: Vec<PlannedTrade>,
    pub rejected_assets: Vec<AssetId>,
    pub affordability_scale: f64,
    #[serde(default)]
    pub requested_buy_notional: f64,
    #[serde(default)]
    pub requested_sell_notional: f64,
    #[serde(default)]
    pub rejected_notional: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExecutionReport {
    pub trades: Vec<PlannedTrade>,
    pub rejected_assets: Vec<AssetId>,
}
pub trait ExecutionPlanner: Send + Sync {
    fn plan(
        &self,
        portfolio: &Portfolio,
        action: &ValidatedAllocation,
        market: &ExecutionSnapshot<'_>,
        minimum_trade: f64,
    ) -> Result<ExecutionPlan>;

    fn plan_action(
        &self,
        portfolio: &Portfolio,
        action: &ValidatedAction,
        market: &ExecutionSnapshot<'_>,
        minimum_trade: f64,
    ) -> Result<ExecutionPlan> {
        match action {
            ValidatedAction::Allocation(allocation) => {
                self.plan(portfolio, allocation, market, minimum_trade)
            }
            ValidatedAction::TradeImpulses(impulses) => {
                self.plan_trade_impulses(portfolio, impulses, market, minimum_trade)
            }
        }
    }

    fn plan_trade_impulses(
        &self,
        _portfolio: &Portfolio,
        _action: &ValidatedTradeImpulses,
        _market: &ExecutionSnapshot<'_>,
        _minimum_trade: f64,
    ) -> Result<ExecutionPlan> {
        Err(MidasError::Policy(
            "execution planner does not support incremental trade impulses".into(),
        ))
    }
}
pub trait ExecutionApplier: Send + Sync {
    fn apply(&self, portfolio: &mut Portfolio, plan: &ExecutionPlan) -> Result<ExecutionReport>;
}
#[derive(Clone)]
pub struct CostExecutionEngine {
    pub costs: ExecutionConfig,
}
impl CostExecutionEngine {
    fn validate_costs(&self) -> Result<()> {
        let c = &self.costs;
        if !c.commission_fixed.is_finite()
            || c.commission_fixed < 0.0
            || !c.commission_rate.is_finite()
            || c.commission_rate < 0.0
            || !c.spread_rate.is_finite()
            || c.spread_rate < 0.0
            || c.spread_rate >= 2.0
            || !c.slippage_rate.is_finite()
            || c.slippage_rate < 0.0
            || c.slippage_rate >= 1.0
            || c.spread_rate / 2.0 + c.slippage_rate >= 1.0
        {
            return Err(MidasError::Config(
                "execution costs cannot produce positive finite prices".into(),
            ));
        }
        Ok(())
    }
    fn trade(
        &self,
        asset_id: AssetId,
        side: TradeSide,
        notional: f64,
        base: f64,
    ) -> Result<PlannedTrade> {
        let half = self.costs.spread_rate / 2.0;
        let execution_price = match side {
            TradeSide::Buy => base * (1.0 + half + self.costs.slippage_rate),
            TradeSide::Sell => base * (1.0 - half - self.costs.slippage_rate),
        };
        let quantity = notional / base;
        let commission = self.costs.commission_fixed + self.costs.commission_rate * notional;
        let spread_cost = notional * half;
        let slippage_cost = notional * self.costs.slippage_rate;
        if [
            notional,
            base,
            execution_price,
            quantity,
            commission,
            spread_cost,
            slippage_cost,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || notional <= 0.0
            || base <= 0.0
            || execution_price <= 0.0
            || quantity <= 0.0
            || commission < 0.0
            || spread_cost < 0.0
            || slippage_cost < 0.0
        {
            return Err(MidasError::Accounting(
                "invalid planned trade economics".into(),
            ));
        }
        if matches!(side, TradeSide::Sell) && quantity * execution_price - commission < 0.0 {
            return Err(MidasError::Accounting("sell costs exceed proceeds".into()));
        }
        Ok(PlannedTrade {
            asset_id,
            side,
            quantity,
            base_price: base,
            execution_price,
            commission,
            spread_cost,
            slippage_cost,
            notional,
        })
    }
}
impl ExecutionPlanner for CostExecutionEngine {
    fn plan(
        &self,
        p: &Portfolio,
        a: &ValidatedAllocation,
        m: &ExecutionSnapshot<'_>,
        minimum: f64,
    ) -> Result<ExecutionPlan> {
        self.validate_costs()?;
        if !minimum.is_finite() || minimum < 0.0 {
            return Err(MidasError::Config(
                "minimum trade value must be finite and non-negative".into(),
            ));
        }
        if m.bars.len() != p.positions.len() || m.valuation_prices.len() != p.positions.len() {
            return Err(MidasError::Accounting(
                "execution snapshot dimension mismatch".into(),
            ));
        }
        if m.bars.iter().enumerate().any(|(index, bar)| {
            bar.is_some_and(|value| value.asset_id != p.positions[index].asset_id)
        }) {
            return Err(MidasError::Accounting(
                "execution snapshot asset order mismatch".into(),
            ));
        }
        if !a.rebalance() {
            return Ok(ExecutionPlan {
                affordability_scale: 1.0,
                ..Default::default()
            });
        }
        let prices = m
            .bars
            .iter()
            .zip(m.valuation_prices)
            .map(|(b, f)| b.map(|x| x.open).or(*f))
            .collect::<Vec<_>>();
        let worth = p.value(&prices)?;
        let current = p
            .positions
            .iter()
            .zip(&prices)
            .map(|(pos, price)| pos.quantity * price.unwrap_or(0.0))
            .collect::<Vec<_>>();
        let mut plan = ExecutionPlan::default();
        let mut available = p.cash;
        for (i, current_value) in current.iter().enumerate() {
            let desired = worth * a.weights()[i + 1];
            if *current_value > desired + minimum {
                if let Some(bar) = m.bars[i].filter(|b| b.tradable) {
                    let trade = self.trade(
                        p.positions[i].asset_id,
                        TradeSide::Sell,
                        *current_value - desired,
                        bar.open,
                    )?;
                    available += trade.quantity * trade.execution_price - trade.commission;
                    plan.trades.push(trade)
                } else {
                    plan.rejected_assets.push(p.positions[i].asset_id);
                }
            }
        }
        let mut eligible = (0..p.positions.len())
            .filter_map(|i| {
                let desired = worth * a.weights()[i + 1];
                if desired > current[i] + minimum {
                    if m.bars[i].is_some_and(|b| b.tradable) {
                        Some((i, desired - current[i]))
                    } else {
                        plan.rejected_assets.push(p.positions[i].asset_id);
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let variable = 1.0
            + self.costs.commission_rate
            + self.costs.spread_rate / 2.0
            + self.costs.slippage_rate;
        let mut scale;
        loop {
            if eligible.is_empty() {
                scale = 0.0;
                break;
            }
            let requested = eligible.iter().map(|(_, v)| *v).sum::<f64>();
            let fixed = self.costs.commission_fixed * eligible.len() as f64;
            scale = ((available - fixed) / requested / variable).clamp(0.0, 1.0);
            let before = eligible.len();
            eligible.retain(|(_, request)| {
                let notional = request * scale;
                notional > 0.0 && notional >= minimum
            });
            if eligible.len() == before {
                break;
            }
        }
        plan.affordability_scale = scale;
        eligible.sort_by_key(|(i, _)| p.positions[*i].asset_id);
        for (i, request) in eligible {
            let bar = m.bars[i].ok_or_else(|| {
                MidasError::Accounting("eligible buy is missing an execution bar".into())
            })?;
            plan.trades.push(self.trade(
                p.positions[i].asset_id,
                TradeSide::Buy,
                request * scale,
                bar.open,
            )?);
        }
        Ok(plan)
    }

    fn plan_trade_impulses(
        &self,
        p: &Portfolio,
        a: &ValidatedTradeImpulses,
        m: &ExecutionSnapshot<'_>,
        minimum: f64,
    ) -> Result<ExecutionPlan> {
        self.validate_costs()?;
        if !minimum.is_finite() || minimum < 0.0 {
            return Err(MidasError::Config(
                "minimum trade value must be finite and non-negative".into(),
            ));
        }
        if m.bars.len() != p.positions.len() || m.valuation_prices.len() != p.positions.len() {
            return Err(MidasError::Accounting(
                "execution snapshot dimension mismatch".into(),
            ));
        }
        if a.impulses().len() != p.positions.len()
            || m.bars.iter().enumerate().any(|(index, bar)| {
                bar.is_some_and(|value| value.asset_id != p.positions[index].asset_id)
            })
        {
            return Err(MidasError::Accounting(
                "incremental trade action or snapshot dimension mismatch".into(),
            ));
        }
        // Trade impulses are sized from the observation close, not from the
        // next opening print. This keeps the policy's configured fraction a
        // stable environment-state quantity while execution still uses opens.
        let worth = p.value(m.valuation_prices)?;
        if !worth.is_finite() || worth <= 0.0 {
            return Err(MidasError::Accounting(
                "incremental trade net worth must be positive and finite".into(),
            ));
        }
        let fraction = a.maximum_trade_fraction();
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            return Err(MidasError::Config(
                "invalid maximum incremental trade fraction".into(),
            ));
        }
        let mut plan = ExecutionPlan {
            affordability_scale: 1.0,
            ..Default::default()
        };
        let mut available = p.cash;
        let mut buys = Vec::<(usize, f64)>::new();
        for (i, impulse) in a.impulses().iter().copied().enumerate() {
            if !impulse.is_finite() || !(-1.0..=1.0).contains(&impulse) {
                return Err(MidasError::Policy(
                    "validated incremental trade impulse is invalid".into(),
                ));
            }
            let requested = impulse.abs() * fraction * worth;
            if !requested.is_finite() || requested <= 0.0 {
                continue;
            }
            if impulse < 0.0 {
                plan.requested_sell_notional += requested;
                let Some(bar) = m.bars[i].filter(|bar| bar.tradable) else {
                    plan.rejected_assets.push(p.positions[i].asset_id);
                    plan.rejected_notional += requested;
                    continue;
                };
                let available_holding = p.positions[i].quantity * bar.open;
                let notional = requested.min(available_holding.max(0.0));
                if notional < minimum || notional <= 0.0 {
                    plan.rejected_notional += requested;
                    continue;
                }
                let trade =
                    self.trade(p.positions[i].asset_id, TradeSide::Sell, notional, bar.open)?;
                available += trade.quantity * trade.execution_price - trade.commission;
                if requested > notional {
                    plan.rejected_notional += requested - notional;
                }
                plan.trades.push(trade);
            } else {
                plan.requested_buy_notional += requested;
                if let Some(bar) = m.bars[i].filter(|bar| bar.tradable) {
                    if requested >= minimum {
                        buys.push((i, requested));
                    } else {
                        plan.rejected_notional += requested;
                    }
                    if !bar.open.is_finite() || bar.open <= 0.0 {
                        return Err(MidasError::Accounting(
                            "invalid incremental trade execution price".into(),
                        ));
                    }
                } else {
                    plan.rejected_assets.push(p.positions[i].asset_id);
                    plan.rejected_notional += requested;
                }
            }
        }
        let variable = 1.0
            + self.costs.commission_rate
            + self.costs.spread_rate / 2.0
            + self.costs.slippage_rate;
        let mut scale = 0.0;
        loop {
            if buys.is_empty() {
                break;
            }
            let requested = buys.iter().map(|(_, value)| *value).sum::<f64>();
            let fixed = self.costs.commission_fixed * buys.len() as f64;
            scale = ((available - fixed) / requested / variable).clamp(0.0, 1.0);
            let before = buys.len();
            buys.retain(|(_, value)| *value * scale >= minimum && *value * scale > 0.0);
            if buys.len() == before {
                break;
            }
        }
        plan.affordability_scale = if buys.is_empty() { 0.0 } else { scale };
        buys.sort_by_key(|(index, _)| p.positions[*index].asset_id);
        for (i, requested) in buys {
            let bar = m.bars[i].ok_or_else(|| {
                MidasError::Accounting("eligible incremental buy is missing a bar".into())
            })?;
            let notional = requested * scale;
            if notional <= 0.0 {
                continue;
            }
            let trade = self.trade(p.positions[i].asset_id, TradeSide::Buy, notional, bar.open)?;
            plan.trades.push(trade);
            plan.rejected_notional += (requested - notional).max(0.0);
        }
        Ok(plan)
    }
}
impl ExecutionApplier for CostExecutionEngine {
    fn apply(&self, p: &mut Portfolio, plan: &ExecutionPlan) -> Result<ExecutionReport> {
        self.validate_costs()?;
        if !plan.affordability_scale.is_finite() || !(0.0..=1.0).contains(&plan.affordability_scale)
        {
            return Err(MidasError::Accounting(
                "invalid execution-plan affordability scale".into(),
            ));
        }
        let mut candidate = p.clone();
        let mut applied_trades = Vec::with_capacity(plan.trades.len());
        for t in &plan.trades {
            let expected = self.trade(t.asset_id, t.side.clone(), t.notional, t.base_price)?;
            if !trades_match(t, &expected) {
                return Err(MidasError::Accounting(
                    "execution plan trade economics mismatch".into(),
                ));
            }
            let position_index = candidate
                .positions
                .iter()
                .position(|position| position.asset_id == t.asset_id)
                .ok_or_else(|| {
                    MidasError::Accounting(format!("unknown execution asset {}", t.asset_id))
                })?;
            let pos = &mut candidate.positions[position_index];
            match expected.side {
                TradeSide::Sell => {
                    if expected.quantity > pos.quantity + 1e-9 {
                        return Err(MidasError::Accounting(
                            "planned sell exceeds holding".into(),
                        ));
                    }
                    pos.quantity = (pos.quantity - expected.quantity).max(0.0);
                    candidate.cash +=
                        expected.quantity * expected.execution_price - expected.commission;
                    if pos.quantity < 1e-12 {
                        pos.average_entry_price = 0.0;
                    }
                }
                TradeSide::Buy => {
                    let spend = expected.quantity * expected.execution_price + expected.commission;
                    if spend > candidate.cash + 1e-8 {
                        return Err(MidasError::Accounting("planned buys exceed cash".into()));
                    }
                    let old = pos.quantity;
                    let new = old + expected.quantity;
                    pos.average_entry_price = (old * pos.average_entry_price
                        + expected.quantity * expected.execution_price)
                        / new;
                    pos.quantity = new;
                    candidate.cash = (candidate.cash - spend).max(0.0);
                }
            }
            candidate.cumulative_commission += expected.commission;
            candidate.cumulative_spread += expected.spread_cost;
            candidate.cumulative_slippage += expected.slippage_cost;
            candidate.cumulative_turnover += expected.notional;
            applied_trades.push(expected);
        }
        candidate.check()?;
        *p = candidate;
        Ok(ExecutionReport {
            trades: applied_trades,
            rejected_assets: plan.rejected_assets.clone(),
        })
    }
}

fn trades_match(actual: &PlannedTrade, expected: &PlannedTrade) -> bool {
    actual.asset_id == expected.asset_id
        && actual.side == expected.side
        && [
            actual.quantity,
            actual.base_price,
            actual.execution_price,
            actual.commission,
            actual.spread_cost,
            actual.slippage_cost,
            actual.notional,
        ]
        .iter()
        .zip([
            expected.quantity,
            expected.base_price,
            expected.execution_price,
            expected.commission,
            expected.spread_cost,
            expected.slippage_cost,
            expected.notional,
        ])
        .all(|(actual, expected)| {
            let tolerance = 1e-10 * actual.abs().max(expected.abs()).max(1.0);
            (*actual - expected).abs() <= tolerance
        })
}
