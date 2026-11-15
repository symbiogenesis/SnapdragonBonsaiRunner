//! Resolve original import routes only against explicitly retained image bytes.

use super::*;
use crate::pe_closure::{ImportSymbol, symbol_exports};

const ROUTE_LIMIT: usize = 65_536;
const HOP_LIMIT: usize = 262_144;
const TEXT_LIMIT: usize = 16 * 1024 * 1024;

struct Provider {
    identity: Identity,
    exports: symbol_exports::Table,
    external: bool,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Symbol {
    Name(String),
    Ordinal(u16),
}

impl Symbol {
    fn value(&self) -> Value {
        match self {
            Self::Name(name) => json!({"name":name}),
            Self::Ordinal(ordinal) => json!({"ordinal":ordinal}),
        }
    }
}

fn parse_forwarder(value: &str) -> Result<(String, Symbol)> {
    let (module, symbol) = value
        .split_once('.')
        .ok_or("PE forwarder requires an explicit module and symbol")?;
    let module = format!("{}.dll", module.to_ascii_lowercase());
    if !safe_name(&module) || module == ".dll" || symbol.is_empty() {
        return Err("PE forwarder has an unsupported module or symbol".into());
    }
    let symbol = if let Some(ordinal) = symbol.strip_prefix('#') {
        if ordinal.is_empty() || !ordinal.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("PE forwarder has an invalid ordinal".into());
        }
        Symbol::Ordinal(
            ordinal
                .parse()
                .map_err(|_| "PE forwarder ordinal exceeds its supported range")?,
        )
    } else {
        Symbol::Name(symbol.into())
    };
    Ok((module, symbol))
}

pub(super) struct Boundary {
    providers: BTreeMap<String, Provider>,
    used: BTreeSet<String>,
    routes: Vec<Value>,
    hops: usize,
    text_bytes: usize,
    recursive: bool,
}

impl Boundary {
    pub(super) fn import_module(&mut self, module: &str) -> Result<()> {
        if !safe_name(module) || !extension(module, "dll") {
            return Err("Native link dependency import has an unsupported module name.".into());
        }
        if module.starts_with("api-") || module.starts_with("ext-") {
            return Err(
                "Native link runtime route requires separately qualified API-set routing.".into(),
            );
        }
        if !self.providers.contains_key(module) {
            return Err(format!(
                "Native link dependency import lacks a retained provider image: {module}"
            ));
        }
        self.used.insert(module.into());
        Ok(())
    }

    pub(super) fn prepare(
        manifest: &Capture,
        package: &BTreeMap<String, Identity>,
        intake: &mut Intake,
    ) -> Result<Self> {
        if manifest.dependency_images.len() > 128 {
            return Err("Native link dependency images exceed their bound.".into());
        }
        let mut providers = BTreeMap::new();
        let mut paths = BTreeSet::new();
        for (module, identity, external) in package
            .iter()
            .filter(|(name, _)| image_name(name))
            .map(|(name, identity)| (name.as_str(), identity, false))
            .chain(
                manifest
                    .dependency_images
                    .iter()
                    .map(|dependency| (dependency.module.as_str(), &dependency.identity, true)),
            )
        {
            if !safe_name(module)
                || (external && !extension(module, "dll"))
                || providers.contains_key(module)
                || !paths.insert(path_key(&identity.path)?)
                || (external
                    && !identity
                        .path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.eq_ignore_ascii_case(module)))
            {
                return Err(
                    "Native link dependency images repeat, alias or have invalid module names."
                        .into(),
                );
            }
            let image = crate::pe_closure::parse_image(intake.read(identity, IMAGE_LIMIT)?)?;
            let exports = symbol_exports::inspect(&image)?;
            providers.insert(
                module.to_owned(),
                Provider {
                    identity: identity.clone(),
                    exports,
                    external,
                },
            );
        }
        Ok(Self {
            providers,
            used: BTreeSet::new(),
            routes: Vec::new(),
            hops: 0,
            text_bytes: 0,
            recursive: manifest.schema_version >= 5,
        })
    }

    pub(super) fn observe(
        &mut self,
        importer: &str,
        module: &str,
        delayed: bool,
        symbol: &ImportSymbol<'_>,
    ) -> Result<()> {
        if self.routes.len() >= ROUTE_LIMIT {
            return Err("Native link runtime routes exceed their aggregate bound.".into());
        }
        let original = match symbol {
            ImportSymbol::Name(name) => Symbol::Name((*name).into()),
            ImportSymbol::Ordinal(ordinal) => Symbol::Ordinal(*ordinal),
        };
        let mut current = (module.to_owned(), original.clone());
        let mut seen = BTreeSet::new();
        let mut stages = Vec::new();
        let mut known = BTreeSet::new();
        loop {
            if stages.len() >= 32 || !seen.insert(current.clone()) {
                return Err("Native link runtime route cycles or exceeds its depth bound.".into());
            }
            self.hops += 1;
            if self.hops > HOP_LIMIT {
                return Err("Native link runtime routes exceed their aggregate hop bound.".into());
            }
            let (module, symbol) = &current;
            if module.starts_with("api-") || module.starts_with("ext-") {
                return Err(
                    "Native link runtime route requires separately qualified API-set routing."
                        .into(),
                );
            }
            let provider = self.providers.get(module).ok_or_else(|| {
                format!("Native link runtime route lacks a retained provider image: {module}")
            })?;
            let ordinal = match symbol {
                Symbol::Name(name) => {
                    if coff::known(name.as_bytes()) {
                        known.insert(name.clone());
                    }
                    *provider.exports.names.get(name).ok_or_else(|| {
                        format!("Native link runtime route names an absent export: {module}!{name}")
                    })?
                }
                Symbol::Ordinal(ordinal) => u32::from(*ordinal),
            };
            let export = provider.exports.ordinals.get(&ordinal).ok_or_else(|| {
                format!("Native link runtime route names an absent ordinal: {module}!#{ordinal}")
            })?;
            let requested_bytes = match symbol {
                Symbol::Name(name) => name.len(),
                Symbol::Ordinal(_) => 0,
            };
            self.text_bytes = export
                .names
                .iter()
                .map(String::len)
                .chain([
                    module.len(),
                    importer.len(),
                    requested_bytes,
                    export.forwarder.as_ref().map_or(0, String::len),
                ])
                .try_fold(self.text_bytes, usize::checked_add)
                .filter(|total| *total <= TEXT_LIMIT)
                .ok_or("Native link runtime route names exceed their aggregate byte bound.")?;
            for name in &export.names {
                if coff::known(name.as_bytes()) {
                    known.insert(name.clone());
                }
            }
            self.used.insert(module.clone());
            stages.push(json!({
                "module":module,"sha256":provider.identity.sha256,
                "requestedSymbol":symbol.value(),"exportOrdinal":ordinal,
                "exportNames":export.names,"targetRva":export.rva,
                "forwarder":export.forwarder
            }));
            if let Some(forwarder) = &export.forwarder {
                current = parse_forwarder(forwarder)?;
            } else {
                break;
            }
        }
        let mut route = json!({
            "packageName":importer,"module":module,"delayed":delayed,
            "importSymbol":original.value(),"stages":stages,"knownCppEhNames":known
        });
        if self.recursive {
            route["importerLocation"] = if self.providers.get(importer).is_some_and(|p| p.external)
            {
                // An external dependency is not a packaged image.
                let object = route.as_object_mut().ok_or("Missing native import route")?;
                object.remove("packageName");
                object.insert("dependencyModule".into(), importer.into());
                "retained-dependency".into()
            } else {
                "package".into()
            };
        }
        self.routes.push(route);
        Ok(())
    }

    fn dependency_imports(&mut self, intake: &mut Intake) -> Result<Vec<Value>> {
        let mut checked = Vec::new();
        let mut visited = BTreeSet::new();
        if !self.recursive {
            return Ok(checked);
        }
        // New forwarder/import targets extend `used`; each external image is
        // visited once, so legitimate mutual DLL imports cannot restart traversal.
        loop {
            let next = self
                .used
                .iter()
                .find(|module| {
                    !visited.contains(*module)
                        && self.providers.get(*module).is_some_and(|p| p.external)
                })
                .cloned();
            let Some(module) = next else { break };
            visited.insert(module.clone());
            let identity = self
                .providers
                .get(&module)
                .ok_or("Missing import provider")?
                .identity
                .clone();
            // Reverify under the same intake/guards, keeping only one external
            // image in memory while traversing its original import records.
            let image = crate::pe_closure::parse_image(intake.read(&identity, IMAGE_LIMIT)?)?;
            let mut modules = BTreeSet::new();
            for delayed in [false, true] {
                for target in crate::pe_closure::imports(&image, delayed)? {
                    self.import_module(&target)?;
                    modules.insert(target);
                }
                crate::pe_closure::visit_import_symbols(&image, delayed, |target, symbol| {
                    self.observe(&module, target, delayed, &symbol)
                })?;
            }
            checked.push(json!({"module":module,"bytes":identity.bytes,
                "sha256":identity.sha256,"importedModules":modules}));
        }
        Ok(checked)
    }

    pub(super) fn finish(mut self, intake: &mut Intake) -> Result<Value> {
        let dependency_imports = self.dependency_imports(intake)?;
        if self
            .providers
            .iter()
            .any(|(module, provider)| provider.external && !self.used.contains(module))
        {
            return Err(
                "Native link dependency image selection contains an unused provider.".into(),
            );
        }
        let images: Vec<_> = self
            .providers
            .into_iter()
            .filter(|(module, _)| self.used.contains(module))
            .map(|(module, provider)| {
                json!({"module":module,"path":provider.identity.path,
                    "bytes":provider.identity.bytes,"sha256":provider.identity.sha256,
                    "location":if provider.external {"retained-dependency"} else {"package"}})
            })
            .collect();
        let mut result = json!({"images":images,"routes":self.routes,
            "apiSetRoutingQualified":false,"recursiveDependencyImportsChecked":self.recursive,
            "providerImplementationLanguageQualified":false});
        if self.recursive {
            result["dependencyImportImages"] = dependency_imports.into();
        }
        Ok(result)
    }
}
