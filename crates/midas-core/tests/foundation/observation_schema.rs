use super::*;

#[test]
fn observation_schema_drives_dimension_order_and_unrealized_return() {
    let mut c = config();
    c.market.assets = vec!["A".into()];
    c.observation.features = vec![
        ObservationFeatureKind::UnrealizedReturn,
        ObservationFeatureKind::CashAllocation,
    ];
    let schema = ObservationSchema::from_config(&c);
    assert_eq!(schema.dimension(1), 2);
    let builder = SchemaObservationBuilder::new(schema, c.features.features.clone(), 1).unwrap();
    let mut p = Portfolio::new(50.0, 1);
    p.positions[0].quantity = 2.0;
    p.positions[0].average_entry_price = 10.0;
    p.net_worth = 80.0;
    let market = midas_core::market::FeatureRow {
        date: date("2020-01-01"),
        values: vec![0.0; c.features.features.len()],
        tradable: vec![true],
    };
    let context = ObservationContext {
        market: &market,
        portfolio: &p,
        prices: &[Some(15.0)],
        starting_net_worth: 80.0,
        maximum_net_worth: 80.0,
        age: 0,
        episode_steps: 1,
        recent_reward: 0.0,
        rolling_mean_reward: 0.0,
        rolling_reward_volatility: 0.0,
        reward_trend: 0.0,
        current_temperature: 1.0,
        initial_temperature: 1.0,
        daily_survival_cost: 0.0,
    };
    let observation = builder.build(&context).unwrap();
    assert!((observation.values[0] - 0.5).abs() < 1e-6);
    assert!((observation.values[1] - 0.625).abs() < 1e-6);
    c.observation.features.reverse();
    let reversed = SchemaObservationBuilder::new(
        ObservationSchema::from_config(&c),
        c.features.features.clone(),
        1,
    )
    .unwrap()
    .build(&context)
    .unwrap();
    assert_eq!(reversed.values, vec![0.625, 0.5]);
}
