use super::*;

#[test]
fn csv_and_parquet_round_trip() {
    let mut c = config();
    let data = loaded(&c);
    assert_eq!(data.bars.len(), 72);
    let file = tempfile::NamedTempFile::new().unwrap();
    write_parquet(&data, file.path()).unwrap();
    c.data.path = file.path().display().to_string();
    c.data.format = midas_core::config::DataFormat::Parquet;
    let again = loaded(&c);
    assert_eq!(data.bars, again.bars);
}

#[test]
fn split_preserves_value_and_entry_price() {
    let mut p = Portfolio::new(0.0, 1);
    p.positions[0].quantity = 10.0;
    p.positions[0].average_entry_price = 100.0;
    let before = p.value(&[Some(100.0)]).unwrap();
    p.apply_corporate_action(&CorporateAction::Split {
        asset_id: AssetId(0),
        date: date("2020-01-02"),
        ratio: 2.0,
    })
    .unwrap();
    assert_eq!(p.positions[0].quantity, 20.0);
    assert_eq!(p.positions[0].average_entry_price, 50.0);
    assert_eq!(before, p.value(&[Some(50.0)]).unwrap());
}

#[test]
fn reverse_and_multiple_splits_compose_without_feature_discontinuity() {
    let config = CorporateActionConfig {
        mode: CorporateActionMode::ExplicitEvents,
        ..Default::default()
    };
    let bars = vec![
        bar(0, "2020-01-01", 100.0),
        bar(0, "2020-01-02", 50.0),
        bar(0, "2020-01-03", 25.0),
    ];
    let actions = vec![
        CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            ratio: 2.0,
        },
        CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-03"),
            ratio: 2.0,
        },
    ];
    let dataset =
        MarketDataset::new_with_actions(vec!["A".into()], bars, actions, &config).unwrap();
    let series = PointInTimeFeaturePriceSeries::new(&dataset);
    assert_eq!(
        series.price(AssetId(0), date("2020-01-01")).unwrap(),
        Some(100.0)
    );
    assert_eq!(
        series.price(AssetId(0), date("2020-01-02")).unwrap(),
        Some(100.0)
    );
    assert_eq!(
        series.price(AssetId(0), date("2020-01-03")).unwrap(),
        Some(100.0)
    );
    let mut portfolio = Portfolio::new(0.0, 1);
    portfolio.positions[0].quantity = 100.0;
    portfolio.positions[0].average_entry_price = 10.0;
    portfolio
        .apply_corporate_action(&CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            ratio: 0.2,
        })
        .unwrap();
    assert_eq!(portfolio.positions[0].quantity, 20.0);
    assert_eq!(portfolio.positions[0].average_entry_price, 50.0);
    assert_eq!(portfolio.value(&[Some(50.0)]).unwrap(), 1000.0);
}

#[test]
fn dividend_is_credited_once_on_declared_event() {
    let action = CorporateAction::CashDividend {
        asset_id: AssetId(0),
        date: date("2020-01-02"),
        amount_per_share: 1.25,
    };
    let mut portfolio = Portfolio::new(10.0, 1);
    portfolio.positions[0].quantity = 4.0;
    portfolio.apply_corporate_action(&action).unwrap();
    assert_eq!(portfolio.cash, 15.0);
    assert_eq!(portfolio.cash, 15.0);
    assert_eq!(portfolio.positions[0].quantity, 4.0);
}

#[test]
fn same_day_split_and_dividend_follow_documented_order() {
    let corporate = CorporateActionConfig {
        mode: CorporateActionMode::ExplicitEvents,
        ..Default::default()
    };
    let actions = vec![
        CorporateAction::CashDividend {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            amount_per_share: 1.0,
        },
        CorporateAction::Split {
            asset_id: AssetId(0),
            date: date("2020-01-02"),
            ratio: 2.0,
        },
    ];
    let dataset = MarketDataset::new_with_actions(
        vec!["A".into()],
        vec![bar(0, "2020-01-01", 100.0), bar(0, "2020-01-02", 50.0)],
        actions,
        &corporate,
    )
    .unwrap();
    let mut portfolio = Portfolio::new(0.0, 1);
    portfolio.positions[0].quantity = 10.0;
    portfolio.positions[0].average_entry_price = 100.0;
    for action in &dataset.corporate_actions {
        portfolio.apply_corporate_action(action).unwrap();
    }
    assert_eq!(portfolio.positions[0].quantity, 20.0);
    assert_eq!(portfolio.cash, 20.0);
}

#[test]
fn reject_mode_refuses_actions_adjustments_and_discontinuities() {
    let reject = CorporateActionConfig::default();
    let bars = vec![bar(0, "2020-01-01", 100.0), bar(0, "2020-01-02", 50.0)];
    let action = CorporateAction::Split {
        asset_id: AssetId(0),
        date: date("2020-01-02"),
        ratio: 2.0,
    };
    assert!(
        MarketDataset::new_with_actions(vec!["A".into()], bars.clone(), vec![action], &reject)
            .is_err()
    );
    assert!(MarketDataset::new_with_actions(vec!["A".into()], bars, vec![], &reject).is_err());
    let mut adjusted = bar(0, "2020-01-01", 100.0);
    adjusted.adjusted_close = 50.0;
    assert!(MarketDataset::new(vec!["A".into()], vec![adjusted]).is_err());
}

#[test]
fn split_features_are_point_in_time_and_future_safe() {
    let mut c = config();
    c.market.assets = vec!["A".into()];
    c.market.benchmark_asset = "A".into();
    c.corporate_actions.mode = CorporateActionMode::ExplicitEvents;
    let bars = vec![
        bar(0, "2020-01-01", 100.0),
        bar(0, "2020-01-02", 101.0),
        bar(0, "2020-01-03", 50.5),
        bar(0, "2020-01-04", 51.0),
    ];
    let action = CorporateAction::Split {
        asset_id: AssetId(0),
        date: date("2020-01-03"),
        ratio: 2.0,
    };
    let with = MarketDataset::new_with_actions(
        vec!["A".into()],
        bars.clone(),
        vec![action.clone()],
        &c.corporate_actions,
    )
    .unwrap();
    let without = MarketDataset::new_with_actions(
        vec!["A".into()],
        bars[..2].to_vec(),
        vec![],
        &c.corporate_actions,
    )
    .unwrap();
    let pipeline = BackwardFeaturePipeline::from_config(&c);
    let early = without.dates.clone();
    let a = pipeline.generate(&with, &early).unwrap();
    let b = pipeline.generate(&without, &early).unwrap();
    assert_eq!(
        a.iter().map(|r| &r.values).collect::<Vec<_>>(),
        b.iter().map(|r| &r.values).collect::<Vec<_>>()
    );
    let rows = pipeline.generate(&with, &with.dates).unwrap();
    let return_index = c
        .features
        .features
        .iter()
        .position(|f| *f == midas_core::config::FeatureKind::Return1)
        .unwrap();
    assert!((rows[2].values[return_index]).abs() < 1e-6);
}
