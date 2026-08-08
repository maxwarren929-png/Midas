use super::*;

#[test]
fn no_hardcoded_observation_formula_remains() {
    let source = include_str!("../../src/simulation.rs");
    assert!(!source.contains("assets * 2 + 5"));
    assert!(!source.contains("market + assets"));
}

#[test]
fn earlier_observations_do_not_change_with_extreme_future_price() {
    let c = config();
    let pipeline = BackwardFeaturePipeline::from_config(&c);
    let a = (0..5)
        .map(|i| bar(0, &format!("2020-01-0{}", i + 1), 10.0 + i as f64))
        .collect::<Vec<_>>();
    let mut b = a.clone();
    b[4] = bar(0, "2020-01-05", 10_000.0);
    let ca = CorporateActionConfig {
        mode: CorporateActionMode::ExplicitEvents,
        discontinuity_threshold: 0.99,
        ..Default::default()
    };
    let da = MarketDataset::new_with_actions(vec!["A".into()], a, vec![], &ca).unwrap();
    let db = MarketDataset::new_with_actions(
        vec!["A".into()],
        b,
        vec![CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-05"),
            ratio: 0.0014,
        }],
        &ca,
    )
    .unwrap();
    let dates = da.dates[..4].to_vec();
    assert_eq!(
        pipeline
            .generate(&da, &dates)
            .unwrap()
            .iter()
            .map(|r| r.values.clone())
            .collect::<Vec<_>>(),
        pipeline
            .generate(&db, &dates)
            .unwrap()
            .iter()
            .map(|r| r.values.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn malformed_ohlc_and_duplicates_are_rejected() {
    assert!(
        MarketDataset::new(
            vec!["A".into()],
            vec![MarketBar {
                high: 5.0,
                ..bar(0, "2020-01-01", 10.0)
            }]
        )
        .is_err()
    );
    let value = bar(0, "2020-01-01", 10.0);
    assert!(MarketDataset::new(vec!["A".into()], vec![value.clone(), value]).is_err());
}

proptest! {
    #[test]
    fn planned_and_applied_allocations_never_make_balances_negative(weight in 0.0f64..1.0) {
        let c = config(); let engine = CostExecutionEngine { costs: c.execution }; let mut portfolio = Portfolio::new(1000.0, 1); let market_bar = bar(0, "2020-01-01", 10.0);
        let action = validator(vec![1.0 - weight, weight], &[true], 1.0);
        let plan = engine.plan(&portfolio, &action, &ExecutionSnapshot { bars: &[Some(&market_bar)], valuation_prices: &[Some(10.0)] }, 0.0).unwrap();
        engine.apply(&mut portfolio, &plan).unwrap();
        prop_assert!(portfolio.cash >= 0.0); prop_assert!(portfolio.positions[0].quantity >= 0.0);
    }
}

#[test]
fn observation_hash_includes_market_version_and_normalization() {
    let c = config();
    let original = ObservationSchema::from_config(&c)
        .identity(c.market.assets.len())
        .unwrap();
    let mut changed = c.clone();
    changed.features.clip *= 2.0;
    assert_ne!(
        original,
        ObservationSchema::from_config(&changed)
            .identity(changed.market.assets.len())
            .unwrap()
    );
    changed = c.clone();
    changed.features.schema_version += 1;
    assert_ne!(
        original,
        ObservationSchema::from_config(&changed)
            .identity(changed.market.assets.len())
            .unwrap()
    );
}
