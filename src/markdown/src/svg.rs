//! Bounded offline SVG validation and CPU rasterization via `resvg`.

mod policy;

#[cfg(test)]
pub(crate) use policy::{ALLOWED_ATTRIBUTES, ALLOWED_ELEMENTS};

use std::sync::{Arc, OnceLock};

/// Maximum UTF-8 byte length accepted for a single inline SVG document.
pub const MAX_SVG_BYTES: usize = 256 * 1024;
/// Maximum number of inline SVG images extracted per assistant message.
pub const MAX_SVG_IMAGES: usize = 8;
/// Maximum rasterized pixel width or height supported per request.
pub const MAX_RASTER_DIMENSION: u32 = 2560;
/// Maximum total pixels (16 MiB premultiplied RGBA8) per rasterization.
pub const MAX_RASTER_PIXELS: usize = 4_194_304;
/// Response magic identifying a bounded raster packet (`[magic][ok:u8][w:u32][h:u32][rgba...]`).
pub(crate) const RASTER_MAGIC: &[u8; 4] = b"BMG1";
/// Byte length of the raster header (`BMG1` + `ok:u8` + `width:u32` + `height:u32`).
pub(crate) const RASTER_HEADER_BYTES: usize = 13;

struct EmbeddedFonts {
    database: Arc<resvg::usvg::fontdb::Database>,
    default_family: String,
}

fn load_embedded_fonts() -> &'static EmbeddedFonts {
    static FONTS: OnceLock<EmbeddedFonts> = OnceLock::new();
    FONTS.get_or_init(|| {
        let mut database = resvg::usvg::fontdb::Database::new();
        database.load_font_data(epaint_default_fonts::UBUNTU_LIGHT.to_vec());
        let sans_family = database
            .faces()
            .next()
            .and_then(|face| face.families.first())
            .map_or_else(|| "sans-serif".to_owned(), |(name, _)| name.clone());
        let existing_count = database.faces().count();
        database.load_font_data(epaint_default_fonts::HACK_REGULAR.to_vec());
        let mono_family = database
            .faces()
            .nth(existing_count)
            .and_then(|face| face.families.first())
            .map_or_else(|| sans_family.clone(), |(name, _)| name.clone());

        database.set_sans_serif_family(&sans_family);
        database.set_serif_family(&sans_family);
        database.set_cursive_family(&sans_family);
        database.set_fantasy_family(&sans_family);
        database.set_monospace_family(&mono_family);

        EmbeddedFonts {
            database: Arc::new(database),
            default_family: sans_family,
        }
    })
}

fn locked_options() -> Option<resvg::usvg::Options<'static>> {
    let fonts = load_embedded_fonts();
    let default_size = resvg::usvg::Size::from_wh(100.0, 100.0)?;
    Some(resvg::usvg::Options {
        resources_dir: None,
        dpi: 96.0,
        font_family: fonts.default_family.clone(),
        font_size: 12.0,
        languages: vec!["en".to_owned()],
        shape_rendering: resvg::usvg::ShapeRendering::GeometricPrecision,
        text_rendering: resvg::usvg::TextRendering::OptimizeLegibility,
        image_rendering: resvg::usvg::ImageRendering::OptimizeQuality,
        default_size,
        // Disallow all external or data-URI `<image>` resolution as defense in depth
        // behind `policy::validate_untrusted_xml` (which already rejects `<image>`).
        image_href_resolver: resvg::usvg::ImageHrefResolver {
            resolve_data: Box::new(|_, _, _| None),
            resolve_string: Box::new(|_, _| None),
        },
        font_resolver: resvg::usvg::FontResolver {
            select_font: resvg::usvg::FontResolver::default_font_selector(),
            select_fallback: resvg::usvg::FontResolver::default_fallback_selector(),
        },
        fontdb: Arc::clone(&fonts.database),
        style_sheet: None,
        max_depth: policy::MAX_SVG_DEPTH,
    })
}

fn parse_tree(svg: &str) -> Option<resvg::usvg::Tree> {
    let trimmed = svg.trim();
    if trimmed.is_empty()
        || svg.len() > MAX_SVG_BYTES
        || trimmed.as_bytes().starts_with(&[0x1f, 0x8b])
    {
        return None;
    }
    let xml = resvg::usvg::roxmltree::Document::parse_with_options(
        trimmed,
        resvg::usvg::roxmltree::ParsingOptions {
            allow_dtd: false,
            nodes_limit: policy::MAX_XML_NODES,
            entity_resolver: None,
        },
    )
    .ok()?;
    if !policy::validate_untrusted_xml(&xml) {
        return None;
    }
    let options = locked_options()?;
    let tree = resvg::usvg::Tree::from_xmltree(&xml, &options).ok()?;
    policy::validate_resolved_tree(&tree).then_some(tree)
}

/// Validates SVG syntax/structure offline and returns finite positive `(width, height)`.
pub(crate) fn inspect(svg: &str) -> Option<(f32, f32)> {
    let tree = parse_tree(svg)?;
    let size = tree.size();
    let width = size.width();
    let height = size.height();
    (width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0)
        .then_some((width, height))
}

fn clamp_round_u16(value: f64, limit: u16) -> u16 {
    let mut low = 1_u16;
    let mut high = limit.max(1);
    while low < high {
        let mid = low + (high - low) / 2;
        if f64::from(mid) + 0.5 <= value {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

fn target_dimensions(
    intrinsic_w: f32,
    intrinsic_h: f32,
    max_width: u32,
    max_height: u32,
) -> Option<(u32, u32)> {
    if !intrinsic_w.is_finite()
        || !intrinsic_h.is_finite()
        || intrinsic_w <= 0.0
        || intrinsic_h <= 0.0
        || max_width == 0
        || max_height == 0
    {
        return None;
    }
    let max_w = u16::try_from(max_width.min(MAX_RASTER_DIMENSION)).ok()?;
    let max_h = u16::try_from(max_height.min(MAX_RASTER_DIMENSION)).ok()?;
    let iw = f64::from(intrinsic_w);
    let ih = f64::from(intrinsic_h);
    let scale = (f64::from(max_w) / iw).min(f64::from(max_h) / ih);
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let mut w = clamp_round_u16(iw * scale, max_w);
    let mut h = clamp_round_u16(ih * scale, max_h);
    let max_pixels = u32::try_from(MAX_RASTER_PIXELS).ok()?;
    let pixels = usize::from(w).saturating_mul(usize::from(h));
    if pixels > MAX_RASTER_PIXELS {
        let shrink = (f64::from(max_pixels) / f64::from(u32::from(w) * u32::from(h))).sqrt();
        w = clamp_round_u16(f64::from(w) * shrink, max_w);
        h = clamp_round_u16(f64::from(h) * shrink, max_h);
        while usize::from(w).saturating_mul(usize::from(h)) > MAX_RASTER_PIXELS {
            if w >= h && w > 1 {
                w -= 1;
            } else if h > 1 {
                h -= 1;
            } else {
                break;
            }
        }
    }
    Some((u32::from(w), u32::from(h)))
}

fn failed_packet() -> Vec<u8> {
    let mut packet = Vec::with_capacity(RASTER_HEADER_BYTES);
    packet.extend_from_slice(RASTER_MAGIC);
    packet.extend_from_slice(&[0; RASTER_HEADER_BYTES - RASTER_MAGIC.len()]);
    packet
}

/// Renders bounded SVG source into an inert `BMG1` premultiplied RGBA8 packet.
///
/// Unrenderable or out-of-range inputs return a well-framed `ok = 0` packet so
/// the caller can fall back gracefully without tearing down the worker pipe.
#[must_use]
pub fn rasterize(svg: &str, max_width: u32, max_height: u32) -> Vec<u8> {
    let Some(tree) = parse_tree(svg) else {
        return failed_packet();
    };
    let size = tree.size();
    let Some((width, height)) =
        target_dimensions(size.width(), size.height(), max_width, max_height)
    else {
        return failed_packet();
    };
    let Some(w16) = u16::try_from(width).ok() else {
        return failed_packet();
    };
    let Some(h16) = u16::try_from(height).ok() else {
        return failed_packet();
    };
    let sx = f32::from(w16) / size.width();
    let sy = f32::from(h16) / size.height();
    if !sx.is_finite() || !sy.is_finite() || sx <= 0.0 || sy <= 0.0 {
        return failed_packet();
    }
    let Some(mut pixmap) = resvg::tiny_skia::Pixmap::new(width, height) else {
        return failed_packet();
    };
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(sx, sy),
        &mut pixmap.as_mut(),
    );
    let data = pixmap.data();
    let Some(packet_len) = RASTER_HEADER_BYTES.checked_add(data.len()) else {
        return failed_packet();
    };
    let mut packet = Vec::with_capacity(packet_len);
    packet.extend_from_slice(RASTER_MAGIC);
    packet.push(1);
    packet.extend_from_slice(&width.to_le_bytes());
    packet.extend_from_slice(&height.to_le_bytes());
    packet.extend_from_slice(data);
    packet
}
