# Prompt 5 Default Observation Equivalence

Prompt 5 is an observation-architecture refactor. It must not change Prompt 4 economics when the default schema is selected.

## Compared run

- Prompt 4 archived result: `reports/prompt4/control-seed-41.txt`, produced by commit `c5dee48f33b23d89a0c46ff532ecf1d4c7cfaaa5`.
- Prompt 5 rerun: the same unchanged `configs/prompt3b/runs/control-seed-41.toml`, training split, release binary.
- Market horizon: 1,259 steps, seed 41, eight static neural agents, no learning or evolution.

The normalized JSON reports were compared after removing only `observation_schema_identity`, which intentionally changes because format 12 hashes explicit registry metadata and index entries. The remaining canonical JSON SHA-256 was identical in both runs:

```text
3fcacff5362a9ed29121d37c7a56001191d9b1e0615b4cb4156d4e348f465577
```

This includes every portfolio result, return, drawdown, turnover, trade count, fee, death, birth, neural action, lineage, and lifecycle field. The historical positional schema identity was `2b3e53ca1817d526fa717eabd762e83446393049d84c47a3a42d953ebb3f313b`; the modular registry schema identity is `95af73d3a1e48b89c35af1b1da1eb1b9786fcb414121b003fe2043a4506b091b`.

The core regression `observation::tests::modular_default_builder_matches_prompt4_positional_vector` also compares the complete default vector from the retired Prompt 4 positional contract with the compiled provider vector exactly. Together these checks demonstrate that the default registry changes schema identity and introspection, not simulated trading behaviour.
