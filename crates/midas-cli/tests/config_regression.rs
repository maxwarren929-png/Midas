use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn config_paths(directory: &Path, root: &Path, output: &mut Vec<String>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            // Prompt 9 run configs are generated experiment inputs and are
            // intentionally gitignored. They are not shipped compatibility
            // fixtures and must not make this test depend on local artifacts.
            if path == root.join("configs/prompt9/runs") {
                continue;
            }
            config_paths(&path, root, output);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "toml")
        {
            output.push(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

#[test]
fn all_shipped_configs_match_the_cli_hash_manifest() {
    let root = root();
    let manifest = include_str!("fixtures/config-hashes.tsv")
        .lines()
        .map(|line| line.split_once('\t').unwrap())
        .map(|(path, hash)| (path.to_owned(), hash.to_owned()))
        .collect::<BTreeMap<_, _>>();

    let mut shipped = Vec::new();
    config_paths(&root.join("configs"), &root, &mut shipped);
    shipped.sort();
    assert_eq!(shipped, manifest.keys().cloned().collect::<Vec<_>>());

    let configured_binary = std::env::var_os("MIDAS_CONFIG_REGRESSION_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_midas-cli")));
    let binary = if configured_binary.is_absolute() {
        configured_binary
    } else {
        root.join(configured_binary)
    };

    for (path, hash) in manifest {
        let output = Command::new(&binary)
            .current_dir(&root)
            .args(["validate-config", "--config", &path])
            .output()
            .unwrap_or_else(|error| panic!("failed to run {}: {error}", binary.display()));
        assert!(output.status.success(), "{path} failed with {output:?}");
        assert_eq!(output.stderr, b"", "unexpected stderr for {path}");
        assert_eq!(
            output.stdout,
            format!("valid configuration {hash}\n").as_bytes(),
            "unexpected stdout for {path}"
        );
    }
}
