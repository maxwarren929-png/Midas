use super::*;

#[test]
fn execution_rate_validation_rejects_invalid_economics() {
    for mutate in [0, 1, 2] {
        let mut c = config();
        match mutate {
            0 => c.execution.spread_rate = 2.0,
            1 => c.execution.slippage_rate = 1.0,
            _ => {
                c.execution.spread_rate = 1.8;
                c.execution.slippage_rate = 0.11
            }
        }
        assert!(c.validate().is_err());
    }
    let mut c = config();
    c.execution.spread_rate = 2.0;
    let engine = CostExecutionEngine { costs: c.execution };
    let p = Portfolio::new(100.0, 1);
    let b = bar(0, "2020-01-01", 10.0);
    let action = validator(vec![0.0, 1.0], &[true], 1.0);
    assert!(
        engine
            .plan(
                &p,
                &action,
                &ExecutionSnapshot {
                    bars: &[Some(&b)],
                    valuation_prices: &[Some(10.0)]
                },
                1.0
            )
            .is_err()
    );
}

#[test]
fn planned_trade_values_are_positive_and_commission_non_negative() {
    let c = config();
    let engine = CostExecutionEngine { costs: c.execution };
    let p = Portfolio::new(100.0, 1);
    let b = bar(0, "2020-01-01", 10.0);
    let action = validator(vec![0.0, 1.0], &[true], 1.0);
    let plan = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&b)],
                valuation_prices: &[Some(10.0)],
            },
            1.0,
        )
        .unwrap();
    assert!(plan.trades.iter().all(|t| t.execution_price > 0.0
        && t.quantity > 0.0
        && t.commission >= 0.0
        && t.spread_cost >= 0.0
        && t.slippage_cost >= 0.0));
}

#[test]
fn action_validator_masks_caps_and_falls_back() {
    let valid = validator(vec![0.0, 0.9, 0.1], &[true, false], 0.4);
    assert!(valid.weights()[1] <= 0.4 + 1e-9);
    assert_eq!(valid.weights()[2], 0.0);
    assert!((valid.weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    let fallback = validator(vec![f64::NAN, 1.0, 0.0], &[true, true], 0.4);
    assert!(fallback.used_fallback());
    assert_eq!(fallback.weights(), &[1.0, 0.0, 0.0]);
}

#[test]
fn action_validator_handles_edge_cases_without_drift_or_looping() {
    let proposals = [
        vec![0.0, 1e300, 1e-300, 5.0, 0.0, 9.0],
        vec![0.0; 6],
        vec![f64::NAN; 6],
        vec![f64::INFINITY, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    ];
    for maximum in [1.0, 0.01, 0.0001] {
        for (tradable, proposal) in [
            (vec![true, true, true, true, true], proposals[0].clone()),
            (
                vec![false, false, false, false, false],
                proposals[1].clone(),
            ),
            (vec![true, false, true, false, true], proposals[2].clone()),
            (vec![false, true, false, true, false], proposals[3].clone()),
            (vec![true, true, false, false, true], proposals[4].clone()),
        ] {
            let value = validator(proposal, &tradable, maximum);
            assert!(value.weights().iter().all(|weight| weight.is_finite()));
            assert!((value.weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(
                value
                    .weights()
                    .iter()
                    .skip(1)
                    .zip(&tradable)
                    .all(|(weight, available)| {
                        (!*available && *weight == 0.0) || *weight <= maximum + 1e-9
                    })
            );
        }
    }
    let one = validator(vec![0.0, 10.0, 1.0], &[false, true], 0.0001);
    assert_eq!(one.weights()[1], 0.0);
    assert!(one.weights()[2] <= 0.0001 + 1e-9);
    let many = vec![true; 127];
    let mut proposal = vec![0.0; many.len() + 1];
    proposal[1..].fill(1.0);
    let value = validator(proposal, &many, 0.003);
    assert!((value.weights().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(
        value
            .weights()
            .iter()
            .skip(1)
            .all(|weight| *weight <= 0.003 + 1e-9)
    );
}

#[test]
fn random_and_malicious_proposals_cannot_bypass_concentration() {
    let mut c = config();
    c.brain.traits.maximum_allocation.initial = 0.25;
    for seed in 0..100 {
        let mut policy = midas_core::brain::RandomPolicy::new(seed);
        let observation = Observation {
            schema_hash: "x".into(),
            values: vec![],
        };
        let proposal = midas_core::brain::Policy::propose_action(
            &mut policy,
            &observation,
            &midas_core::brain::PolicyContext {
                tradable: &[true, true, true],
                current_allocations: &[1.0, 0.0, 0.0, 0.0],
            },
        )
        .unwrap();
        let valid = LongOnlyActionValidator
            .validate_action(
                proposal,
                &ActionValidationContext {
                    asset_count: 3,
                    tradable: &[true, true, true],
                    policy_maximum_asset_allocation: Some(0.25),
                    policy_maximum_trade_fraction: None,
                    market_maximum_asset_allocation: None,
                },
            )
            .unwrap();
        let midas_core::action::ValidatedAction::Allocation(valid) = valid else {
            panic!("random benchmark must emit an allocation action")
        };
        assert!(valid.weights().iter().skip(1).all(|v| *v <= 0.25 + 1e-9));
    }
    let valid = validator(vec![0.0, 100.0, 0.0, 0.0], &[true, true, true], 0.25);
    assert!(valid.weights().iter().skip(1).all(|v| *v <= 0.25 + 1e-9));
}

#[test]
fn proportional_planning_includes_costs_and_is_order_invariant() {
    let mut c = config();
    c.execution.commission_fixed = 2.0;
    c.execution.commission_rate = 0.01;
    c.execution.spread_rate = 0.02;
    c.execution.slippage_rate = 0.01;
    let engine = CostExecutionEngine { costs: c.execution };
    let p = Portfolio::new(100.0, 2);
    let a = bar(0, "2020-01-01", 10.0);
    let b = bar(1, "2020-01-01", 20.0);
    let action = validator(vec![0.0, 0.5, 0.5], &[true, true], 1.0);
    let plan = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&a), Some(&b)],
                valuation_prices: &[Some(10.0), Some(20.0)],
            },
            1.0,
        )
        .unwrap();
    let buys = plan
        .trades
        .iter()
        .filter(|t| t.side == TradeSide::Buy)
        .collect::<Vec<_>>();
    assert_eq!(buys.len(), 2);
    assert!((buys[0].notional / 50.0 - plan.affordability_scale).abs() < 1e-9);
    assert!((buys[1].notional / 50.0 - plan.affordability_scale).abs() < 1e-9);
    let mut reversed_a = b.clone();
    reversed_a.asset_id = AssetId(0);
    let mut reversed_b = a.clone();
    reversed_b.asset_id = AssetId(1);
    let reversed = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&reversed_a), Some(&reversed_b)],
                valuation_prices: &[Some(20.0), Some(10.0)],
            },
            1.0,
        )
        .unwrap();
    assert!((plan.affordability_scale - reversed.affordability_scale).abs() < 1e-12);
    assert_eq!(
        plan.trades.iter().map(|t| t.notional).sum::<f64>(),
        reversed.trades.iter().map(|t| t.notional).sum::<f64>()
    );
}

#[test]
fn incremental_impulses_preserve_portfolio_and_execute_sells_before_buys() {
    let c = config();
    let engine = CostExecutionEngine { costs: c.execution };
    let validator = LongOnlyActionValidator;
    let mut portfolio = Portfolio::new(50.0, 2);
    portfolio.positions[0].quantity = 5.0;
    portfolio.positions[0].average_entry_price = 10.0;
    let bar0 = bar(0, "2020-01-01", 10.0);
    let bar1 = bar(1, "2020-01-01", 20.0);
    let bars = [Some(&bar0), Some(&bar1)];
    let prices = [Some(10.0), Some(20.0)];
    let context = ActionValidationContext {
        asset_count: 2,
        tradable: &[true, true],
        policy_maximum_asset_allocation: None,
        policy_maximum_trade_fraction: None,
        market_maximum_asset_allocation: None,
    };
    let hold = validator
        .validate_action(
            ProposedAction::TradeImpulses(ProposedTradeImpulses::new(vec![0.0, 0.0], 0.05)),
            &context,
        )
        .unwrap();
    let hold_plan = engine
        .plan_action(
            &portfolio,
            &hold,
            &ExecutionSnapshot {
                bars: &bars,
                valuation_prices: &prices,
            },
            0.0,
        )
        .unwrap();
    assert!(hold_plan.trades.is_empty());

    let action = validator
        .validate_action(
            ProposedAction::TradeImpulses(ProposedTradeImpulses::new(vec![-1.0, 1.0], 0.5)),
            &context,
        )
        .unwrap();
    let plan = engine
        .plan_action(
            &portfolio,
            &action,
            &ExecutionSnapshot {
                bars: &bars,
                valuation_prices: &prices,
            },
            0.0,
        )
        .unwrap();
    assert_eq!(
        plan.trades.first().map(|trade| &trade.side),
        Some(&TradeSide::Sell)
    );
    assert!(plan.trades.iter().any(|trade| trade.side == TradeSide::Buy));
    engine.apply(&mut portfolio, &plan).unwrap();
    assert!(portfolio.cash >= -1e-9);
    assert!(
        portfolio
            .positions
            .iter()
            .all(|position| position.quantity >= -1e-9)
    );
    assert!(portfolio.positions[0].quantity < 5.0);
    assert!(portfolio.positions[1].quantity > 0.0);
}

#[test]
fn incremental_impulses_cannot_bypass_configured_trade_fraction() {
    let validator = LongOnlyActionValidator;
    let error = validator
        .validate_action(
            ProposedAction::TradeImpulses(ProposedTradeImpulses::new(vec![0.5], 0.2)),
            &ActionValidationContext {
                asset_count: 1,
                tradable: &[true],
                policy_maximum_asset_allocation: None,
                policy_maximum_trade_fraction: Some(0.05),
                market_maximum_asset_allocation: None,
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("incremental trade impulse"));
}

#[test]
fn incremental_minimum_trade_filter_reports_removed_buy_notional() {
    let c = config();
    let engine = CostExecutionEngine { costs: c.execution };
    let validator = LongOnlyActionValidator;
    let portfolio = Portfolio::new(100.0, 2);
    let bar0 = bar(0, "2020-01-01", 100.0);
    let bar1 = bar(1, "2020-01-01", 100.0);
    let bars = [Some(&bar0), Some(&bar1)];
    let prices = [Some(100.0), Some(100.0)];
    let action = validator
        .validate_action(
            ProposedAction::TradeImpulses(ProposedTradeImpulses::new(vec![0.01, 0.0], 0.05)),
            &ActionValidationContext {
                asset_count: 2,
                tradable: &[true, true],
                policy_maximum_asset_allocation: None,
                policy_maximum_trade_fraction: None,
                market_maximum_asset_allocation: None,
            },
        )
        .unwrap();
    let plan = engine
        .plan_action(
            &portfolio,
            &action,
            &ExecutionSnapshot {
                bars: &bars,
                valuation_prices: &prices,
            },
            1.0,
        )
        .unwrap();
    assert!(plan.trades.is_empty());
    assert!(plan.rejected_notional > 0.0);
}

#[test]
fn execution_uses_stable_asset_ids_and_rejects_tampered_plans() {
    let mut c = config();
    c.execution.commission_fixed = 1.0;
    c.execution.commission_rate = 0.01;
    c.execution.spread_rate = 0.02;
    c.execution.slippage_rate = 0.01;
    let engine = CostExecutionEngine { costs: c.execution };
    let mut first = Portfolio::new(100.0, 2);
    first.positions[0].asset_id = AssetId(10);
    first.positions[1].asset_id = AssetId(20);
    let first_a = MarketBar {
        asset_id: AssetId(10),
        ..bar(10, "2020-01-01", 10.0)
    };
    let first_b = MarketBar {
        asset_id: AssetId(20),
        ..bar(20, "2020-01-01", 20.0)
    };
    let first_bars = [Some(&first_a), Some(&first_b)];
    let first_prices = [Some(10.0), Some(20.0)];
    let action = validator(vec![0.0, 0.5, 0.5], &[true, true], 1.0);
    let first_plan = engine
        .plan(
            &first,
            &action,
            &ExecutionSnapshot {
                bars: &first_bars,
                valuation_prices: &first_prices,
            },
            0.0,
        )
        .unwrap();
    let mut second = Portfolio::new(100.0, 2);
    second.positions.swap(0, 1);
    second.positions[0].asset_id = AssetId(20);
    second.positions[1].asset_id = AssetId(10);
    let second_bars = [Some(&first_b), Some(&first_a)];
    let second_prices = [Some(20.0), Some(10.0)];
    let second_action = validator(vec![0.0, 0.5, 0.5], &[true, true], 1.0);
    let second_plan = engine
        .plan(
            &second,
            &second_action,
            &ExecutionSnapshot {
                bars: &second_bars,
                valuation_prices: &second_prices,
            },
            0.0,
        )
        .unwrap();
    let by_id = |plan: &midas_core::execution::ExecutionPlan| {
        plan.trades
            .iter()
            .map(|trade| (trade.asset_id, (trade.quantity, trade.notional)))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    assert_eq!(by_id(&first_plan), by_id(&second_plan));
    let first_report = engine.apply(&mut first, &first_plan).unwrap();
    assert!(first.cash >= 0.0);
    assert!(
        first_report
            .trades
            .iter()
            .all(|trade| trade.asset_id.0 >= 10)
    );
    let mut tampered = first_plan.clone();
    tampered.trades[0].commission += 1.0;
    assert!(
        engine
            .apply(&mut Portfolio::new(100.0, 2), &tampered)
            .is_err()
    );
    let unavailable_bars = [Some(&first_a), None];
    let mut unavailable_portfolio = Portfolio::new(100.0, 2);
    unavailable_portfolio.positions[0].asset_id = AssetId(10);
    unavailable_portfolio.positions[1].asset_id = AssetId(20);
    let unavailable = engine
        .plan(
            &unavailable_portfolio,
            &action,
            &ExecutionSnapshot {
                bars: &unavailable_bars,
                valuation_prices: &first_prices,
            },
            0.0,
        )
        .unwrap();
    assert!(unavailable.rejected_assets.contains(&AssetId(20)));
    assert!(
        unavailable
            .trades
            .iter()
            .all(|trade| trade.asset_id != AssetId(20))
    );
}

#[test]
fn minimum_trade_filter_stabilizes_and_apply_is_atomic() {
    let mut c = config();
    c.execution.commission_fixed = 10.0;
    let engine = CostExecutionEngine { costs: c.execution };
    let mut p = Portfolio::new(30.0, 2);
    let a = bar(0, "2020-01-01", 10.0);
    let b = bar(1, "2020-01-01", 10.0);
    let action = validator(vec![0.0, 0.9, 0.1], &[true, true], 1.0);
    let plan = engine
        .plan(
            &p,
            &action,
            &ExecutionSnapshot {
                bars: &[Some(&a), Some(&b)],
                valuation_prices: &[Some(10.0), Some(10.0)],
            },
            5.0,
        )
        .unwrap();
    assert_eq!(
        plan.trades
            .iter()
            .filter(|t| t.side == TradeSide::Buy)
            .count(),
        1
    );
    engine.apply(&mut p, &plan).unwrap();
    assert!(p.cash >= 0.0);

    let mut expensive = config();
    expensive.execution.commission_fixed = 100.0;
    let expensive_engine = CostExecutionEngine {
        costs: expensive.execution,
    };
    let no_trade = expensive_engine
        .plan(
            &Portfolio::new(30.0, 1),
            &validator(vec![0.0, 1.0], &[true], 1.0),
            &ExecutionSnapshot {
                bars: &[Some(&a)],
                valuation_prices: &[Some(10.0)],
            },
            0.0,
        )
        .unwrap();
    assert!(no_trade.trades.is_empty());
    assert_eq!(no_trade.affordability_scale, 0.0);
}
