//! Bounded LRU GPU texture cache for offline-rasterized SVG artwork.
#![forbid(unsafe_code)]

use super::egui;
use crate::markdown_service::{self, MAX_RASTER_DIMENSION, Service, Status, Ticket};
use std::{sync::Arc, time::Duration};

const MAX_TEXTURES: usize = 24;
const MAX_IN_FLIGHT: usize = 8;
const QUANTUM_PX: u16 = 64;
const REPAINT_INTERVAL: Duration = Duration::from_millis(16);

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
    svg: Arc<str>,
    last_used: u64,
    state: SlotState,
}

/// Coordinates bounded offline SVG raster requests and retains up to 24 GPU textures.
#[derive(Default)]
pub(super) struct ImageCache {
    service: Option<Arc<Service>>,
    entries: Vec<Entry>,
    clock: u64,
}

impl ImageCache {
    /// Drives active offline rasterization tickets to GPU texture upload.
    pub(super) fn poll(&mut self, ctx: &egui::Context) {
        if self.service.is_none() {
            self.service = Service::get(ctx);
        }
        let Some(service) = self.service.as_ref().map(Arc::clone) else {
            return;
        };
        let mut any_pending = false;
        let mut any_changed = false;
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
                    entry.state = upload_from_packet(ctx, &service, &bytes);
                    any_changed = true;
                }
                Status::Failed => {
                    service.release(ticket);
                    entry.state = SlotState::Failed;
                    any_changed = true;
                }
            }
        }
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
    pub(super) fn is_rejected(&self, svg: &Arc<str>) -> bool {
        let hash = fnv1a64(svg.as_bytes());
        self.entries.iter().any(|entry| {
            entry.key.hash == hash
                && entry.key.len == svg.len()
                && same_svg(&entry.svg, svg)
                && matches!(entry.state, SlotState::Unrenderable | SlotState::Failed)
        })
    }

    /// Returns a GPU texture for `svg` sized for `target_points`, scheduling rasterization if needed.
    pub(super) fn get_or_request(
        &mut self,
        ctx: &egui::Context,
        svg: &Arc<str>,
        target_points: egui::Vec2,
    ) -> RasterSlot {
        let pixels_per_point = ctx.pixels_per_point();
        let max_width = quantize_px(target_points.x, pixels_per_point);
        let max_height = quantize_px(target_points.y, pixels_per_point);
        let hash = fnv1a64(svg.as_bytes());
        let key = Key {
            hash,
            len: svg.len(),
            max_width,
            max_height,
        };

        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;

        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.key == key && same_svg(&entry.svg, svg))
        {
            self.entries[index].last_used = stamp;
            return match &self.entries[index].state {
                SlotState::Ready(texture) => RasterSlot::Ready(texture.clone()),
                SlotState::Pending(_) => RasterSlot::Loading {
                    fallback: self.best_fallback(hash, svg),
                },
                SlotState::Unrenderable | SlotState::Failed => RasterSlot::FallbackSource,
            };
        }

        // If any exact SVG entry already proved unrenderable or failed at a valid size,
        // avoid re-submitting identical failing payloads across resizes.
        if self.entries.iter().any(|entry| {
            entry.key.hash == hash
                && entry.key.len == svg.len()
                && same_svg(&entry.svg, svg)
                && matches!(entry.state, SlotState::Unrenderable | SlotState::Failed)
        }) {
            return RasterSlot::FallbackSource;
        }

        let fallback = self.best_fallback(hash, svg);
        if self.service.is_none() {
            self.service = Service::get(ctx);
        }

        #[cfg(test)]
        if self.service.is_none() {
            let packet = bonsai_markdown::rasterize(svg, max_width, max_height);
            let state = match markdown_service::decode_raster(&packet) {
                Ok(Some(decoded)) => {
                    create_texture(ctx, &decoded).map_or(SlotState::Failed, SlotState::Ready)
                }
                Ok(None) => SlotState::Unrenderable,
                Err(_) => SlotState::Failed,
            };
            let result = match &state {
                SlotState::Ready(texture) => RasterSlot::Ready(texture.clone()),
                _ => RasterSlot::FallbackSource,
            };
            self.entries.push(Entry {
                key,
                svg: Arc::clone(svg),
                last_used: stamp,
                state,
            });
            self.evict_finished();
            return result;
        }

        let Some(service) = self.service.as_ref().map(Arc::clone) else {
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

        let ticket = service.request_raster(Arc::clone(svg), max_width, max_height);
        if ticket.admitted() {
            self.entries.push(Entry {
                key,
                svg: Arc::clone(svg),
                last_used: stamp,
                state: SlotState::Pending(ticket),
            });
            ctx.request_repaint_after(REPAINT_INTERVAL);
            RasterSlot::Loading { fallback }
        } else if service.unavailable() {
            self.entries.push(Entry {
                key,
                svg: Arc::clone(svg),
                last_used: stamp,
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

    fn best_fallback(&self, hash: u64, svg: &Arc<str>) -> Option<egui::TextureHandle> {
        let mut best: Option<(u64, &egui::TextureHandle)> = None;
        for entry in &self.entries {
            if entry.key.hash == hash
                && entry.key.len == svg.len()
                && same_svg(&entry.svg, svg)
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

    fn evict_finished(&mut self) {
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

fn same_svg(left: &Arc<str>, right: &Arc<str>) -> bool {
    Arc::ptr_eq(left, right) || left.as_ref() == right.as_ref()
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

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
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
                    let valid: Arc<str> = Arc::from(
                        r#"<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80"><rect width="120" height="80" fill="teal"/></svg>"#,
                    );
                    let invalid: Arc<str> = Arc::from("<not-svg>");
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
                        let svg: Arc<str> = Arc::from(format!(
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
        let svg: Arc<str> = Arc::from(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="teal"/></svg>"#,
        );
        let test_svg = Arc::clone(&svg);
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
}
