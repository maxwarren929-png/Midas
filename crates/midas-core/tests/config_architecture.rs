const OBSERVATION_CORE_SOURCES: [(&str, &str); 11] = [
    ("builder.rs", include_str!("../src/observation/builder.rs")),
    (
        "contracts.rs",
        include_str!("../src/observation/contracts.rs"),
    ),
    (
        "normalization.rs",
        include_str!("../src/observation/normalization.rs"),
    ),
    ("options.rs", include_str!("../src/observation/options.rs")),
    (
        "registry.rs",
        include_str!("../src/observation/registry.rs"),
    ),
    ("schema.rs", include_str!("../src/observation/schema.rs")),
    (
        "providers/mod.rs",
        include_str!("../src/observation/providers/mod.rs"),
    ),
    (
        "providers/agent_state.rs",
        include_str!("../src/observation/providers/agent_state.rs"),
    ),
    (
        "providers/market.rs",
        include_str!("../src/observation/providers/market.rs"),
    ),
    (
        "providers/portfolio.rs",
        include_str!("../src/observation/providers/portfolio.rs"),
    ),
    (
        "providers/rolling_market_stats.rs",
        include_str!("../src/observation/providers/rolling_market_stats.rs"),
    ),
];

#[test]
fn observation_core_keeps_root_config_in_the_compatibility_adapter() {
    for (name, source) in OBSERVATION_CORE_SOURCES {
        assert!(
            !source.contains("config::Config")
                && !source.contains("config::{Config")
                && !source.contains("&Config"),
            "observation core source {name} depends on root Config"
        );
    }

    let compatibility = include_str!("../src/observation/config_compat.rs");
    assert!(compatibility.contains("config::Config"));
    assert!(compatibility.contains("pub fn from_config"));
    assert!(compatibility.contains("pub fn try_from_config"));
}
