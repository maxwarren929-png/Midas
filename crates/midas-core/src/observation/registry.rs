use crate::Result;
use std::collections::{BTreeMap, BTreeSet};

use super::{
    contracts::{ObservationMetadata, ObservationProvider},
    schema::ObservationSchema,
};

pub struct ObservationRegistry {
    providers: BTreeMap<String, Box<dyn ObservationProvider>>,
    metadata: BTreeMap<String, ObservationMetadata>,
}

impl ObservationRegistry {
    pub fn new() -> Self {
        Self {
            providers: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, provider: Box<dyn ObservationProvider>) -> Result<()> {
        let provider_name = provider.provider_name().to_owned();
        if provider_name.trim().is_empty() || self.providers.contains_key(&provider_name) {
            return Err(crate::MidasError::Config(format!(
                "duplicate or empty observation provider {provider_name}"
            )));
        }
        let metadata = provider.metadata();
        if metadata.is_empty() {
            return Err(crate::MidasError::Config(format!(
                "observation provider {provider_name} does not publish observations"
            )));
        }
        for item in &metadata {
            item.validate()?;
            if item.provider != provider_name || self.metadata.contains_key(&item.name) {
                return Err(crate::MidasError::Config(format!(
                    "duplicate observation name {}",
                    item.name
                )));
            }
        }
        for item in metadata {
            self.metadata.insert(item.name.clone(), item);
        }
        self.providers.insert(provider_name, provider);
        Ok(())
    }

    pub fn metadata(&self) -> Vec<&ObservationMetadata> {
        self.metadata.values().collect()
    }

    pub fn metadata_by_name(&self, name: &str) -> Option<&ObservationMetadata> {
        self.metadata.get(name)
    }

    pub fn validate_schema(&self, schema: &ObservationSchema, asset_count: usize) -> Result<()> {
        schema.validate(asset_count)?;
        let expected_providers: BTreeSet<_> = schema
            .entries
            .iter()
            .map(|entry| entry.provider.as_str())
            .collect();
        let registered_providers: BTreeSet<_> = self.providers.keys().map(String::as_str).collect();
        if expected_providers != registered_providers {
            return Err(crate::MidasError::Config(
                "configured observation providers do not match registry".into(),
            ));
        }
        let schema_names: BTreeSet<_> = schema
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        let metadata_names: BTreeSet<_> = self.metadata.keys().map(String::as_str).collect();
        if schema_names != metadata_names {
            let missing = schema_names.difference(&metadata_names).next().copied();
            let unused = metadata_names.difference(&schema_names).next().copied();
            return Err(crate::MidasError::Config(format!(
                "observation schema and registry differ; missing={missing:?}, unused={unused:?}"
            )));
        }
        for entry in &schema.entries {
            let metadata = self
                .metadata
                .get(&entry.name)
                .expect("set equality checked");
            if metadata.provider != entry.provider || metadata.shape.dimension() <= entry.component
            {
                return Err(crate::MidasError::Config(format!(
                    "observation metadata does not match schema entry {}",
                    entry.name
                )));
            }
            if metadata
                .ordering_hint
                .is_some_and(|index| index != entry.index)
            {
                return Err(crate::MidasError::Config(format!(
                    "observation metadata ordering hint disagrees with schema entry {}",
                    entry.name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn providers(&self) -> impl Iterator<Item = &Box<dyn ObservationProvider>> {
        self.providers.values()
    }
}

impl Default for ObservationRegistry {
    fn default() -> Self {
        Self::new()
    }
}
