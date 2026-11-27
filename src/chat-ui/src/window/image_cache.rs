//! Bounded LRU GPU texture cache for offline-rasterized SVG artwork.
#![forbid(unsafe_code)]

use super::egui;
use crate::markdown_service::{self, MAX_RASTER_DIMENSION, Service, Status, Ticket};
use base64ct::{Base64, Encoding as _};
use std::{io::Cursor, sync::Arc, time::Duration};

const MAX_TEXTURES: usize = 24;
const MAX_PNG_TEXTURES: usize = 32;
const MAX_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_PNG_DIMENSION: u32 = 2_048;
const MAX_PNG_BYTES: usize = 16 * 1024 * 1024;
const MAX_IN_FLIGHT: usize = 8;
const QUANTUM_PX: u16 = 64;
const REPAINT_INTERVAL: Duration = Duration::from_millis(16);

/// Immutable SVG source string paired with its precomputed FNV-1a hash.
#[derive(Clone, Debug)]
pub(in crate::window) struct PrehashedSvg {
    svg: Arc<str>,
    hash: u64,
}

impl PrehashedSvg {
    /// Stores `svg` alongside a once-computed FNV-1a content hash.
    pub(in crate::window) fn from_str(svg: &str) -> Self {
        Self {
            hash: fnv1a64(svg.as_bytes()),
            svg: Arc::from(svg),
        }
    }

    /// Returns the precomputed FNV-1a hash of the UTF-8 SVG payload.
    pub(in crate::window) const fn hash(&self) -> u64 {
        self.hash
    }

    /// Consumes this handle and yields the underlying shared SVG string.
    pub(in crate::window) fn into_arc(self) -> Arc<str> {
        self.svg
    }

    /// Compares two SVG handles, checking precomputed hashes and pointer identity first.
    pub(in crate::window) fn same_content(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.svg.len() == other.svg.len()
            && (Arc::ptr_eq(&self.svg, &other.svg) || self.svg.as_ref() == other.svg.as_ref())
    }
}

impl PartialEq for PrehashedSvg {
    fn eq(&self, other: &Self) -> bool {
        self.same_content(other)
    }
}

impl Eq for PrehashedSvg {}

impl std::ops::Deref for PrehashedSvg {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.svg
    }
}

/// Presentation state for a requested SVG texture at a target physical pixel size.
pub(super) enum RasterSlot {
    /// Texture at the quantized target resolution is ready on the GPU.
    Ready(egui::TextureHandle),
    /// Rasterization is in progress; carries an optional existing resolution for progressive painting.
    Loading {
        fallback: Option<egui::TextureHandle>,
    },
    /// Offline rasterization could not render this SVG (or the worker failed); show source fallback.
    FallbackSource,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    hash: u64,
    len: usize,
    max_width: u32,
    max_height: u32,
}

enum SlotState {
    Pending(Ticket),
    Ready(egui::TextureHandle),
    Unrenderable,
    Failed,
}

struct Entry {
    key: Key,
    svg: PrehashedSvg,
    last_used: u64,
    bytes: usize,
    state: SlotState,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PngDomain {
    RawBytes,
    DataUrl,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct PngKey {
    domain: PngDomain,
    hash: u64,
    len: usize,
}

struct PngEntry {
    key: PngKey,
    last_used: u64,
    bytes: usize,
    texture: Option<egui::TextureHandle>,
}

fn texture_bytes(texture: &egui::TextureHandle) -> usize {
    let [width, height] = texture.size();
    width.saturating_mul(height).saturating_mul(4)
}

/// Coordinates bounded offline SVG raster requests and retains bounded GPU textures.
#[derive(Default)]
pub(super) struct ImageCache {
    service: Option<Arc<Service>>,
    entries: Vec<Entry>,
    png_entries: Vec<PngEntry>,
    clock: u64,
}

impl ImageCache {
    /// Drives active offline rasterization tickets to GPU texture upload.
    pub(super) fn poll(&mut self, ctx: &egui::Context) {
        let Some(service) = Service::get_or_init(&mut self.service, ctx) else {
            return;
        };
        let mut any_pending = false;
        let mut any_changed = false;
        let mut clock = self.clock;
        for entry in &mut self.entries {
            let SlotState::Pending(ticket) = entry.state else {
                continue;
            };
            match service.poll(ticket) {
                Status::Pending => {
                    any_pending = true;
                }
                Status::Ready(bytes) => {
                    service.release(ticket);
                    clock = clock.wrapping_add(1);
                    entry.last_used = clock;
                    entry.state = upload_from_packet(ctx, service, &bytes);
                    entry.bytes = match &entry.state {
                        SlotState::Ready(texture) => texture_bytes(texture),
                        _ => 0,
                    };
                    any_changed = true;
                }
                Status::Failed => {
                    service.release(ticket);
                    entry.state = SlotState::Failed;
                    entry.bytes = 0;
                    any_changed = true;
                }
            }
        }
        self.clock = clock;
        if any_changed {
            self.evict_finished();
            ctx.request_repaint();
        } else if any_pending {
            ctx.request_repaint_after(REPAINT_INTERVAL);
        }
    }

    /// Returns whether any offline SVG rasterization ticket is currently in flight.
    pub(super) fn has_pending(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| matches!(entry.state, SlotState::Pending(_)))
    }

    /// Returns whether `svg` already failed offline validation or rasterization.
    pub(super) fn is_rejected(&self, svg: &PrehashedSvg) -> bool {
        self.entries.iter().any(|entry| {
            entry.svg.same_content(svg)
                && matches!(entry.state, SlotState::Unrenderable | SlotState::Failed)
        })
    }

    /// Returns a GPU texture for `svg` sized for `target_points`, scheduling rasterization if needed.
    pub(super) fn get_or_request(
        &mut self,
        ctx: &egui::Context,
        svg: &PrehashedSvg,
        target_points: egui::Vec2,
    ) -> RasterSlot {
        let pixels_per_point = ctx.pixels_per_point();
        let max_width = quantize_px(target_points.x, pixels_per_point);
        let max_height = quantize_px(target_points.y, pixels_per_point);
        let key = Key {
            hash: svg.hash(),
            len: svg.len(),
            max_width,
            max_height,
        };

        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;

        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.key == key && entry.svg.same_content(svg))
        {
            self.entries[index].last_used = stamp;
            return match &self.entries[index].state {
                SlotState::Ready(texture) => RasterSlot::Ready(texture.clone()),
                SlotState::Pending(_) => RasterSlot::Loading {
                    fallback: self.best_fallback(svg),
                },
                SlotState::Unrenderable | SlotState::Failed => RasterSlot::FallbackSource,
            };
        }

        // If any exact SVG entry already proved unrenderable or failed at a valid size,
        // avoid re-submitting identical failing payloads across resizes.
        if self.is_rejected(svg) {
            return RasterSlot::FallbackSource;
        }

        let fallback = self.best_fallback(svg);
        let Some(service) = Service::get_or_init(&mut self.service, ctx) else {
            #[cfg(test)]
            {
                let packet = bonsai_markdown::rasterize(svg, max_width, max_height);
                let state = match markdown_service::decode_raster(&packet) {
                    Ok(Some(decoded)) => {
                        create_texture(ctx, &decoded).map_or(SlotState::Failed, SlotState::Ready)
                    }
                    Ok(None) => SlotState::Unrenderable,
                    Err(_) => SlotState::Failed,
                };
                let (bytes, result) = match &state {
                    SlotState::Ready(texture) => {
                        (texture_bytes(texture), RasterSlot::Ready(texture.clone()))
                    }
                    _ => (0, RasterSlot::FallbackSource),
                };
                self.entries.push(Entry {
                    key,
                    svg: svg.clone(),
                    last_used: stamp,
                    bytes,
                    state,
                });
                self.evict_finished();
                return result;
            }
            #[cfg(not(test))]
            return RasterSlot::Loading { fallback };
        };

        let in_flight = self
            .entries
            .iter()
            .filter(|entry| matches!(entry.state, SlotState::Pending(_)))
            .count();
        if in_flight >= MAX_IN_FLIGHT {
            ctx.request_repaint_after(REPAINT_INTERVAL);
            return RasterSlot::Loading { fallback };
        }

        let ticket = service.request_raster(Arc::clone(&svg.svg), max_width, max_height);
        if ticket.admitted() {
            self.entries.push(Entry {
                key,
                svg: svg.clone(),
                last_used: stamp,
                bytes: 0,
                state: SlotState::Pending(ticket),
            });
            ctx.request_repaint_after(REPAINT_INTERVAL);
            RasterSlot::Loading { fallback }
        } else if service.unavailable() {
            self.entries.push(Entry {
                key,
                svg: svg.clone(),
                last_used: stamp,
                bytes: 0,
                state: SlotState::Failed,
            });
            self.evict_finished();
            RasterSlot::FallbackSource
        } else {
            // Worker queue temporarily full; retry on an upcoming frame.
            ctx.request_repaint_after(Duration::from_millis(25));
            RasterSlot::Loading { fallback }
        }
    }

    /// Returns a GPU texture decoded from pre-hashed bounded PNG bytes (caching decode failures cleanly).
    pub(super) fn png_texture(
        &mut self,
        ctx: &egui::Context,
        png: &crate::attachment_import::PrehashedPng,
    ) -> Option<egui::TextureHandle> {
        self.png_bytes_texture_with_hash(ctx, png.hash(), png)
    }

    /// Returns a GPU texture decoded from bounded PNG bytes (caching decode failures cleanly).
    #[cfg(test)]
    pub(super) fn png_bytes_texture(
        &mut self,
        ctx: &egui::Context,
        bytes: &[u8],
    ) -> Option<egui::TextureHandle> {
        self.png_bytes_texture_with_hash(ctx, fnv1a64(bytes), bytes)
    }

    fn png_bytes_texture_with_hash(
        &mut self,
        ctx: &egui::Context,
        hash: u64,
        bytes: &[u8],
    ) -> Option<egui::TextureHandle> {
        let key = PngKey {
            domain: PngDomain::RawBytes,
            hash,
            len: bytes.len(),
        };
        self.cached_png(key, || decode_png_texture(ctx, bytes))
    }

    /// Returns a GPU texture decoded from a `data:image/png;base64,...` payload.
    pub(super) fn data_url_texture(
        &mut self,
        ctx: &egui::Context,
        data_url: &str,
    ) -> Option<egui::TextureHandle> {
        let key = PngKey {
            domain: PngDomain::DataUrl,
            hash: fnv1a64(data_url.as_bytes()),
            len: data_url.len(),
        };
        self.cached_png(key, || {
            let encoded = data_url.strip_prefix("data:image/png;base64,")?;
            let decoded = Base64::decode_vec(encoded).ok()?;
            decode_png_texture(ctx, &decoded)
        })
    }

    fn cached_png(
        &mut self,
        key: PngKey,
        decode: impl FnOnce() -> Option<egui::TextureHandle>,
    ) -> Option<egui::TextureHandle> {
        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;
        if let Some(entry) = self.png_entries.iter_mut().find(|entry| entry.key == key) {
            entry.last_used = stamp;
            return entry.texture.clone();
        }
        let texture = decode();
        let bytes = texture.as_ref().map_or(0, texture_bytes);
        self.png_entries.push(PngEntry {
            key,
            last_used: stamp,
            bytes,
            texture: texture.clone(),
        });
        self.evict_excess();
        texture
    }

    fn best_fallback(&self, svg: &PrehashedSvg) -> Option<egui::TextureHandle> {
        let mut best: Option<(u64, &egui::TextureHandle)> = None;
        for entry in &self.entries {
            if entry.svg.same_content(svg)
                && let SlotState::Ready(texture) = &entry.state
            {
                let area = u64::from(entry.key.max_width) * u64::from(entry.key.max_height);
                if best.is_none_or(|(best_area, _)| area > best_area) {
                    best = Some((area, texture));
                }
            }
        }
        best.map(|(_, texture)| texture.clone())
    }

    fn total_texture_bytes(&self) -> usize {
        let svg_bytes: usize = self
            .entries
            .iter()
            .map(|entry| entry.bytes)
            .fold(0_usize, usize::saturating_add);
        let png_bytes: usize = self
            .png_entries
            .iter()
            .map(|entry| entry.bytes)
            .fold(0_usize, usize::saturating_add);
        svg_bytes.saturating_add(png_bytes)
    }

    fn evict_finished(&mut self) {
        self.evict_excess();
    }

    fn evict_excess(&mut self) {
        loop {
            let finished_count = self
                .entries
                .iter()
                .filter(|entry| !matches!(entry.state, SlotState::Pending(_)))
                .count();
            if finished_count <= MAX_TEXTURES {
                break;
            }
            let Some((victim, _)) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| !matches!(entry.state, SlotState::Pending(_)))
                .min_by_key(|(_, entry)| entry.last_used)
            else {
                break;
            };
            self.entries.swap_remove(victim);
        }
        while self.png_entries.len() > MAX_PNG_TEXTURES {
            if let Some((victim, _)) = self
                .png_entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
            {
                self.png_entries.swap_remove(victim);
            } else {
                break;
            }
        }
        self.evict_over_byte_budget(MAX_CACHE_BYTES);
    }

    fn evict_over_byte_budget(&mut self, max_bytes: usize) {
        while self.total_texture_bytes() > max_bytes {
            let best_svg = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    entry.bytes > 0 && !matches!(entry.state, SlotState::Pending(_))
                })
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(idx, entry)| (idx, entry.last_used));
            let best_png = self
                .png_entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.bytes > 0)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(idx, entry)| (idx, entry.last_used));
            match (best_svg, best_png) {
                (Some((svg_idx, svg_ts)), Some((png_idx, png_ts))) => {
                    if svg_ts <= png_ts {
                        self.entries.swap_remove(svg_idx);
                    } else {
                        self.png_entries.swap_remove(png_idx);
                    }
                }
                (Some((svg_idx, _)), None) => {
                    self.entries.swap_remove(svg_idx);
                }
                (None, Some((png_idx, _))) => {
                    self.png_entries.swap_remove(png_idx);
                }
                (None, None) => break,
            }
        }
    }
}

impl Drop for ImageCache {
    fn drop(&mut self) {
        if let Some(service) = &self.service {
            for entry in &mut self.entries {
                if let SlotState::Pending(ticket) = entry.state {
                    service.release(ticket);
                }
            }
        }
    }
}

fn upload_from_packet(ctx: &egui::Context, service: &Service, bytes: &[u8]) -> SlotState {
    match markdown_service::decode_raster(bytes) {
        Ok(Some(decoded)) => {
            create_texture(ctx, &decoded).map_or(SlotState::Failed, SlotState::Ready)
        }
        Ok(None) => SlotState::Unrenderable,
        Err(_) => {
            service.reject();
            SlotState::Failed
        }
    }
}

fn create_texture(
    ctx: &egui::Context,
    decoded: &markdown_service::DecodedRaster<'_>,
) -> Option<egui::TextureHandle> {
    let width = usize::try_from(decoded.width).ok()?;
    let height = usize::try_from(decoded.height).ok()?;
    let image =
        egui::ColorImage::from_rgba_premultiplied([width, height], decoded.premultiplied_rgba);
    Some(ctx.load_texture("bonsai-svg", image, egui::TextureOptions::LINEAR))
}

fn quantize_px(points: f32, pixels_per_point: f32) -> u32 {
    let max_dim_u16 = u16::try_from(MAX_RASTER_DIMENSION).unwrap_or(2_560);
    let target = if points.is_finite()
        && pixels_per_point.is_finite()
        && points > 0.0
        && pixels_per_point > 0.0
    {
        (points * pixels_per_point).min(f32::from(max_dim_u16))
    } else {
        f32::from(QUANTUM_PX)
    };
    let mut bucket = QUANTUM_PX;
    while bucket < max_dim_u16 && f32::from(bucket) + 0.5 < target {
        bucket = bucket.saturating_add(QUANTUM_PX).min(max_dim_u16);
    }
    u32::from(bucket)
}

fn decode_png_texture(ctx: &egui::Context, bytes: &[u8]) -> Option<egui::TextureHandle> {
    if bytes.is_empty() || bytes.len() > MAX_PNG_BYTES {
        return None;
    }
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let info = reader.info();
    if info.width == 0
        || info.height == 0
        || info.width > MAX_PNG_DIMENSION
        || info.height > MAX_PNG_DIMENSION
    {
        return None;
    }
    let buffer_len = reader.output_buffer_size()?;
    if buffer_len == 0 || buffer_len > MAX_PNG_BYTES {
        return None;
    }
    let mut pixels = vec![0_u8; buffer_len];
    let output = reader.next_frame(&mut pixels).ok()?;
    let frame = &pixels[..output.buffer_size()];
    let width = usize::try_from(output.width).ok()?;
    let height = usize::try_from(output.height).ok()?;
    let pixel_count = width.checked_mul(height)?;
    let image = match output.color_type {
        png::ColorType::Rgba => {
            if frame.len() != pixel_count.checked_mul(4)? {
                return None;
            }
            egui::ColorImage::from_rgba_unmultiplied([width, height], frame)
        }
        png::ColorType::Rgb => {
            if frame.len() != pixel_count.checked_mul(3)? {
                return None;
            }
            egui::ColorImage::from_rgb([width, height], frame)
        }
        png::ColorType::Grayscale => {
            if frame.len() != pixel_count {
                return None;
            }
            let mut rgba = Vec::with_capacity(pixel_count.checked_mul(4)?);
            for &gray in frame {
                rgba.extend_from_slice(&[gray, gray, gray, 255]);
            }
            egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba)
        }
        png::ColorType::GrayscaleAlpha => {
            if frame.len() != pixel_count.checked_mul(2)? {
                return None;
            }
            let mut rgba = Vec::with_capacity(pixel_count.checked_mul(4)?);
            for pair in frame.chunks_exact(2) {
                rgba.extend_from_slice(&[pair[0], pair[0], pair[0], pair[1]]);
            }
            egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba)
        }
        png::ColorType::Indexed => return None,
    };
    Some(ctx.load_texture("bonsai-attachment-png", image, egui::TextureOptions::LINEAR))
}

pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::Harness;

    #[test]
    fn quantize_px_buckets_safely_within_worker_bounds() {
        assert_eq!(quantize_px(0.0, 1.0), 64);
        assert_eq!(quantize_px(-10.0, 2.0), 64);
        assert_eq!(quantize_px(f32::NAN, 1.0), 64);
        assert_eq!(quantize_px(64.0, 1.0), 64);
        assert_eq!(quantize_px(65.0, 1.0), 128);
        assert_eq!(quantize_px(320.0, 2.0), 640);
        assert_eq!(quantize_px(5_000.0, 2.0), 2_560);
    }

    #[test]
    fn image_cache_renders_via_worker_fixture_and_falls_back_on_unrenderable_svg()
    -> std::io::Result<()> {
        let mut harness = Harness::builder()
            .with_size([320.0, 240.0])
            .with_context_setup(super::super::configure)
            .build_ui_state(
                |ui, (cache, armed): &mut (ImageCache, bool)| {
                    if !*armed {
                        return;
                    }
                    cache.poll(ui.ctx());
                    let valid = PrehashedSvg::from_str(
                        r#"<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80"><rect width="120" height="80" fill="teal"/></svg>"#,
                    );
                    let invalid = PrehashedSvg::from_str("<not-svg>");
                    match cache.get_or_request(ui.ctx(), &valid, egui::vec2(200.0, 120.0)) {
                        RasterSlot::Ready(tex) => {
                            let [w, h] = tex.size();
                            ui.label(format!("ready:{w}x{h}"));
                        }
                        RasterSlot::Loading { .. } => {
                            ui.label("loading");
                        }
                        RasterSlot::FallbackSource => {
                            ui.label("fallback");
                        }
                    }
                    match cache.get_or_request(ui.ctx(), &invalid, egui::vec2(128.0, 128.0)) {
                        RasterSlot::FallbackSource => {
                            ui.label("invalid-fallback");
                        }
                        RasterSlot::Loading { .. } => {
                            ui.label("invalid-loading");
                        }
                        RasterSlot::Ready(_) => {
                            ui.label("invalid-ready");
                        }
                    }
                },
                (ImageCache::default(), false),
            );
        let respond = Service::fixture(&harness.ctx);
        harness.state_mut().1 = true;
        harness.step();
        assert!(respond()?.starts_with("<svg"));
        assert_eq!(&*respond()?, "<not-svg>");
        harness.step();
        let entry = harness
            .state()
            .0
            .entries
            .iter()
            .find(|e| e.svg.starts_with("<svg"))
            .ok_or_else(|| std::io::Error::other("missing SVG entry"))?;
        assert!(matches!(entry.state, SlotState::Ready(_)));
        let invalid_entry = harness
            .state()
            .0
            .entries
            .iter()
            .find(|e| &*e.svg == "<not-svg>")
            .ok_or_else(|| std::io::Error::other("missing invalid entry"))?;
        assert!(matches!(invalid_entry.state, SlotState::Unrenderable));
        Ok(())
    }

    #[test]
    fn image_cache_evicts_lru_ready_entries_above_cap() {
        let mut harness = Harness::builder()
            .with_size([160.0, 120.0])
            .with_context_setup(super::super::configure)
            .build_ui_state(
                |ui, cache: &mut ImageCache| {
                    for i in 0..30 {
                        let svg = PrehashedSvg::from_str(&format!(
                            r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><circle cx="{i}" cy="16" r="8"/></svg>"#
                        ));
                        let _ = cache.get_or_request(ui.ctx(), &svg, egui::vec2(64.0, 64.0));
                    }
                },
                ImageCache::default(),
            );
        harness.step();
        assert_eq!(harness.state().entries.len(), MAX_TEXTURES);
    }

    #[test]
    fn malformed_bmg1_response_rejects_worker_session_and_marks_svg_rejected() -> std::io::Result<()>
    {
        let svg = PrehashedSvg::from_str(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="teal"/></svg>"#,
        );
        let test_svg = svg.clone();
        let mut harness = Harness::builder()
            .with_size([240.0, 180.0])
            .with_context_setup(super::super::configure)
            .build_ui_state(
                move |ui, (cache, armed): &mut (ImageCache, bool)| {
                    if !*armed {
                        return;
                    }
                    cache.poll(ui.ctx());
                    let _ = cache.get_or_request(ui.ctx(), &test_svg, egui::vec2(128.0, 128.0));
                },
                (ImageCache::default(), false),
            );
        let (service, mut step) =
            Service::custom_responder(&harness.ctx, |_, _| b"FORGED-BMG1-BYTES".to_vec());
        harness.state_mut().1 = true;
        harness.step();
        assert!(!service.unavailable());
        assert_eq!(&*step()?, &*svg);
        harness.step();
        assert!(service.unavailable());
        assert!(harness.state().0.is_rejected(&svg));
        Ok(())
    }

    #[test]
    fn png_bytes_and_data_url_textures_decode_and_cache_failures() -> std::io::Result<()> {
        let mut png_bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png_bytes, 2, 2);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
            writer
                .write_image_data(&[
                    255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
                ])
                .map_err(std::io::Error::other)?;
        }
        let data_url = format!(
            "data:image/png;base64,{}",
            Base64::encode_string(&png_bytes)
        );
        let mut harness = Harness::builder()
            .with_size([160.0, 120.0])
            .with_context_setup(super::super::configure)
            .build_ui_state(
                move |ui, cache: &mut ImageCache| {
                    assert_eq!(
                        cache
                            .png_bytes_texture(ui.ctx(), &png_bytes)
                            .map(|tex| tex.size()),
                        Some([2, 2])
                    );
                    assert_eq!(
                        cache
                            .data_url_texture(ui.ctx(), &data_url)
                            .map(|tex| tex.size()),
                        Some([2, 2])
                    );
                    assert!(cache.png_bytes_texture(ui.ctx(), b"not-png").is_none());
                    assert!(
                        cache
                            .data_url_texture(ui.ctx(), "data:image/jpeg;base64,AAAA")
                            .is_none()
                    );
                },
                ImageCache::default(),
            );
        harness.step();
        assert_eq!(harness.state().png_entries.len(), 4);
        Ok(())
    }

    fn gray_2048_png(pixel_seed: u8) -> std::io::Result<Vec<u8>> {
        let mut png_bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut png_bytes, 2_048, 2_048);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fastest);
        let mut writer = encoder.write_header().map_err(std::io::Error::other)?;
        let mut pixels = vec![pixel_seed; 2_048 * 2_048];
        pixels[0] = pixel_seed.wrapping_add(1);
        writer
            .write_image_data(&pixels)
            .map_err(std::io::Error::other)?;
        writer.finish().map_err(std::io::Error::other)?;
        Ok(png_bytes)
    }

    #[test]
    fn image_cache_enforces_64_mib_byte_cap_via_lru_eviction() -> std::io::Result<()> {
        let pngs = std::thread::scope(|scope| -> std::io::Result<[Vec<u8>; 5]> {
            let h1 = scope.spawn(|| gray_2048_png(10));
            let h2 = scope.spawn(|| gray_2048_png(20));
            let h3 = scope.spawn(|| gray_2048_png(30));
            let h4 = scope.spawn(|| gray_2048_png(40));
            let p5 = gray_2048_png(50)?;
            let join_png = |h: std::thread::ScopedJoinHandle<'_, std::io::Result<Vec<u8>>>| {
                h.join()
                    .map_err(|_| std::io::Error::other("png fixture worker panicked"))?
            };
            Ok([
                join_png(h1)?,
                join_png(h2)?,
                join_png(h3)?,
                join_png(h4)?,
                p5,
            ])
        })?;
        let hashes: Vec<u64> = pngs.iter().map(|bytes| fnv1a64(bytes)).collect();
        let svg = PrehashedSvg::from_str(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="navy"/></svg>"#,
        );
        let svg_for_ui = svg.clone();
        let pngs_for_ui = pngs.clone();
        let mut harness = Harness::builder()
            .with_size([160.0, 120.0])
            .with_context_setup(super::super::configure)
            .build_ui_state(
                move |ui, cache: &mut ImageCache| {
                    let _ = cache.get_or_request(ui.ctx(), &svg_for_ui, egui::vec2(64.0, 64.0));
                    for png in &pngs_for_ui[0..4] {
                        assert!(cache.png_bytes_texture(ui.ctx(), png).is_some());
                    }
                    // Refresh entry 0 so entry 1 becomes the LRU before pushing the 5th 16 MiB PNG.
                    assert!(cache.png_bytes_texture(ui.ctx(), &pngs_for_ui[0]).is_some());
                    assert!(cache.png_bytes_texture(ui.ctx(), &pngs_for_ui[4]).is_some());
                    // Negative-cache zero-byte entries remain without displacing GPU textures.
                    assert!(cache.png_bytes_texture(ui.ctx(), b"not-png").is_none());
                },
                ImageCache::default(),
            );
        harness.step();
        let state = harness.state();
        assert_eq!(state.total_texture_bytes(), MAX_CACHE_BYTES);
        assert!(state.entries.is_empty());
        assert!(state.png_entries.iter().any(|e| e.key.hash == hashes[0]));
        assert!(!state.png_entries.iter().any(|e| e.key.hash == hashes[1]));
        assert!(state.png_entries.iter().any(|e| e.key.hash == hashes[2]));
        assert!(state.png_entries.iter().any(|e| e.key.hash == hashes[3]));
        assert!(state.png_entries.iter().any(|e| e.key.hash == hashes[4]));
        assert!(state.png_entries.iter().any(|e| e.bytes == 0));
        Ok(())
    }
}
