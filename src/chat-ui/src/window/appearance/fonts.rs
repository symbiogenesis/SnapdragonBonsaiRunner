//! Reading faces are loaded once before the native event loop starts.
use super::egui;
use std::{fs::File, path::PathBuf, sync::OnceLock};

const STRONG_FAMILY: &str = "Bonsai Strong";
const MAX_FONT_BYTES: u64 = 4 * 1024 * 1024;
static FONTS: OnceLock<egui::FontDefinitions> = OnceLock::new();

fn fallback() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    let proportional = fonts
        .families
        .get(&egui::FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    fonts
        .families
        .insert(egui::FontFamily::Name(STRONG_FAMILY.into()), proportional);
    let monospace = fonts
        .families
        .get(&egui::FontFamily::Monospace)
        .cloned()
        .unwrap_or_default();
    fonts.families.insert(
        egui::FontFamily::Name("Bonsai Code Block".into()),
        monospace,
    );
    fonts
}

fn load() -> egui::FontDefinitions {
    let mut fonts = fallback();
    let Some(directory) = super::super::platform::fonts_directory().or_else(|| {
        std::env::var_os("SystemRoot")
            .map(PathBuf::from)
            .filter(|directory| directory.is_absolute())
            .map(|directory| directory.join("Fonts"))
    }) else {
        return fonts;
    };
    let face = |name| {
        crate::bounded_file::read(File::open(directory.join(name)).ok()?, MAX_FONT_BYTES)
            .ok()?
            .filter(|bytes| !bytes.is_empty())
            .map(egui::FontData::from_owned)
    };
    // Install a matched pair, preserving the bundled fallback if either native
    // face is unavailable. System-font discovery still supplies other scripts.
    let (Some(regular), Some(strong)) = (face("segoeui.ttf"), face("segoeuib.ttf")) else {
        return fonts;
    };
    fonts
        .font_data
        .insert("Bonsai Reading".into(), regular.into());
    fonts
        .font_data
        .insert("Bonsai Strong".into(), strong.into());
    for (family, name) in [
        (egui::FontFamily::Proportional, "Bonsai Reading"),
        (
            egui::FontFamily::Name(STRONG_FAMILY.into()),
            "Bonsai Strong",
        ),
    ] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, name.into());
    }
    fonts
}

/// Call before creating the native window so production configure never hits disk.
pub(super) fn prepare() {
    let _ = FONTS.get_or_init(load);
}

pub(super) fn configure(ctx: &egui::Context) {
    ctx.set_fonts(FONTS.get_or_init(load).clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_native_reading_faces_keep_distinct_weight_and_bundled_fallbacks() {
        prepare();
        let fonts = FONTS.get().cloned().unwrap_or_else(fallback);
        let regular = &fonts.families[&egui::FontFamily::Proportional];
        let strong = &fonts.families[&egui::FontFamily::Name(STRONG_FAMILY.into())];
        assert_eq!(regular.first().map(String::as_str), Some("Bonsai Reading"));
        assert_eq!(strong.first().map(String::as_str), Some("Bonsai Strong"));
        assert_ne!(
            fonts.font_data["Bonsai Reading"],
            fonts.font_data["Bonsai Strong"]
        );
        for family in [regular, strong] {
            assert!(family.iter().any(|name| name == "Ubuntu-Light"));
        }
        let fallback = fallback();
        assert_eq!(
            fallback.families[&egui::FontFamily::Name(STRONG_FAMILY.into())],
            fallback.families[&egui::FontFamily::Proportional]
        );
        assert_eq!(
            fallback.families[&egui::FontFamily::Name("Bonsai Code Block".into())],
            fallback.families[&egui::FontFamily::Monospace]
        );
    }
}
