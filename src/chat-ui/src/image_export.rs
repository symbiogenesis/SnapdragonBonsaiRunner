//! Asynchronous native image export (PNG via sandboxed rasterization or raw SVG source).
#![deny(unsafe_code)]

use std::{
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use eframe::egui;

use crate::{
    markdown_service::{self, Service, Status, Ticket},
    window::atomic_file,
};

mod picker;

use picker::{Format, Selection};

const EXPORT_RASTER_DIM: u32 = 2_048;
const POLL: Duration = Duration::from_millis(10);
const RASTER_UNAVAILABLE: &str = "Image rasterization is unavailable. Choose SVG format to save.";
const RASTER_FAILED: &str = "The image could not be rasterized to PNG.";
const UNRENDERABLE_IMAGE: &str = "This SVG could not be rendered as a PNG. Save as SVG instead.";
const SAVE_WRITE_FAILED: &str = "Could not write the selected image file.";

#[derive(Default)]
struct Shared {
    cancelled: AtomicBool,
    picker: picker::Cancel,
}

/// Terminal or in-progress state of an image export job.
pub(crate) enum Outcome {
    /// Dialog selection, rasterization, or atomic file write is still in progress.
    Pending,
    /// User closed or cancelled the save dialog before writing.
    Cancelled,
    /// Complete image file was published atomically to disk.
    Saved(PathBuf),
    /// Export failed cleanly without leaving partial target files.
    Failed(&'static str),
}

enum PendingRaster {
    Ticket(Arc<Service>, Ticket),
    Retry {
        service: Arc<Service>,
        svg: Arc<str>,
    },
}

/// Coordinates the background native Save dialog alongside bounded offline PNG rasterization.
pub(crate) struct Export {
    shared: Arc<Shared>,
    outcome: Receiver<Outcome>,
    raster_tx: Option<SyncSender<Result<Arc<[u8]>, &'static str>>>,
    raster: Option<PendingRaster>,
    worker: Option<JoinHandle<()>>,
    finished: bool,
}

impl Export {
    /// Launches a native Save dialog on an STA worker and prepares a 2048x2048 PNG raster if needed.
    pub(crate) fn start(
        ctx: &egui::Context,
        owner: isize,
        svg: Arc<str>,
        suggested_stem: String,
    ) -> io::Result<Self> {
        let (raster_tx, raster_rx) = mpsc::sync_channel::<Result<Arc<[u8]>, &'static str>>(1);
        let raster = prepare_raster(ctx, &svg, &raster_tx);
        let shared = Arc::new(Shared::default());
        let worker_shared = Arc::clone(&shared);
        let (outcome_tx, outcome_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("bonsai-image-save".to_owned())
            .spawn(move || {
                let outcome = run_worker(
                    &worker_shared,
                    || {
                        picker::pick_save_target(
                            owner,
                            &worker_shared.picker,
                            &worker_shared.cancelled,
                            &suggested_stem,
                        )
                    },
                    &svg,
                    &raster_rx,
                );
                let _sent = outcome_tx.send(outcome);
            })?;
        Ok(Self {
            shared,
            outcome: outcome_rx,
            raster_tx: Some(raster_tx),
            raster,
            worker: Some(worker),
            finished: false,
        })
    }

    #[cfg(test)]
    fn start_with_selection(
        ctx: &egui::Context,
        selection: Selection,
        svg: Arc<str>,
    ) -> io::Result<Self> {
        let (raster_tx, raster_rx) = mpsc::sync_channel::<Result<Arc<[u8]>, &'static str>>(1);
        let raster = prepare_raster(ctx, &svg, &raster_tx);
        let shared = Arc::new(Shared::default());
        let worker_shared = Arc::clone(&shared);
        let (outcome_tx, outcome_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("bonsai-image-save-test-worker".to_owned())
            .spawn(move || {
                let outcome = run_worker(&worker_shared, || Ok(Some(selection)), &svg, &raster_rx);
                let _sent = outcome_tx.send(outcome);
            })?;
        Ok(Self {
            shared,
            outcome: outcome_rx,
            raster_tx: Some(raster_tx),
            raster,
            worker: Some(worker),
            finished: false,
        })
    }

    /// Drives pending rasterization and returns the latest export status without blocking.
    pub(crate) fn poll(&mut self) -> Outcome {
        if self.finished {
            return Outcome::Cancelled;
        }
        self.shared.picker.reap_finished();
        self.drive_raster();
        match self.outcome.try_recv() {
            Ok(outcome) => {
                self.finish_resources();
                outcome
            }
            Err(TryRecvError::Empty) => Outcome::Pending,
            Err(TryRecvError::Disconnected) => {
                self.finish_resources();
                Outcome::Cancelled
            }
        }
    }

    fn drive_raster(&mut self) {
        let Some(raster) = self.raster.take() else {
            return;
        };
        let Some(tx) = &self.raster_tx else {
            return;
        };
        match raster {
            PendingRaster::Retry { service, svg } => {
                let ticket =
                    service.request_raster(Arc::clone(&svg), EXPORT_RASTER_DIM, EXPORT_RASTER_DIM);
                if ticket.admitted() {
                    self.raster = Some(PendingRaster::Ticket(service, ticket));
                } else if service.unavailable() {
                    let _sent = tx.try_send(Err(RASTER_FAILED));
                    drop(self.raster_tx.take());
                } else {
                    self.raster = Some(PendingRaster::Retry { service, svg });
                }
            }
            PendingRaster::Ticket(service, ticket) => match service.poll(ticket) {
                Status::Pending => {
                    self.raster = Some(PendingRaster::Ticket(service, ticket));
                }
                Status::Ready(bytes) => {
                    let result = if markdown_service::decode_raster(&bytes).is_ok() {
                        Ok(bytes)
                    } else {
                        service.reject();
                        Err(RASTER_FAILED)
                    };
                    service.release(ticket);
                    let _sent = tx.try_send(result);
                    drop(self.raster_tx.take());
                }
                Status::Failed => {
                    service.release(ticket);
                    let _sent = tx.try_send(Err(RASTER_FAILED));
                    drop(self.raster_tx.take());
                }
            },
        }
    }

    fn finish_resources(&mut self) {
        self.finished = true;
        drop(self.raster_tx.take());
        if let Some(PendingRaster::Ticket(service, ticket)) = self.raster.take() {
            service.release(ticket);
        }
        if let Some(worker) = self.worker.take() {
            let _joined = worker.join();
        }
        self.shared.picker.join();
    }
}

impl Drop for Export {
    fn drop(&mut self) {
        self.shared.cancelled.store(true, Ordering::Release);
        self.shared.picker.cancel(&self.shared.cancelled);
        self.finish_resources();
    }
}

fn prepare_raster(
    ctx: &egui::Context,
    svg: &Arc<str>,
    raster_tx: &SyncSender<Result<Arc<[u8]>, &'static str>>,
) -> Option<PendingRaster> {
    if let Some(service) = Service::get(ctx) {
        let ticket = service.request_raster(Arc::clone(svg), EXPORT_RASTER_DIM, EXPORT_RASTER_DIM);
        if ticket.admitted() {
            Some(PendingRaster::Ticket(service, ticket))
        } else if service.unavailable() {
            let _sent = raster_tx.try_send(Err(RASTER_FAILED));
            None
        } else {
            Some(PendingRaster::Retry {
                service,
                svg: Arc::clone(svg),
            })
        }
    } else {
        #[cfg(test)]
        {
            let packet = Arc::from(bonsai_markdown::rasterize(
                svg,
                EXPORT_RASTER_DIM,
                EXPORT_RASTER_DIM,
            ));
            let _sent = raster_tx.try_send(Ok(packet));
            None
        }
        #[cfg(not(test))]
        {
            let _sent = raster_tx.try_send(Err(RASTER_UNAVAILABLE));
            None
        }
    }
}

fn run_worker(
    shared: &Shared,
    select: impl FnOnce() -> Result<Option<Selection>, &'static str>,
    svg: &str,
    raster_rx: &Receiver<Result<Arc<[u8]>, &'static str>>,
) -> Outcome {
    match select() {
        Ok(Some(Selection {
            path,
            format: Format::Svg,
        })) => write_export(&path, svg.as_bytes()),
        Ok(Some(Selection {
            path,
            format: Format::Png,
        })) => {
            match await_raster(shared, raster_rx).and_then(|packet| encode_png_from_bmg1(&packet)) {
                Ok(png_bytes) => write_export(&path, &png_bytes),
                Err(message) if message == picker::CANCELLED => Outcome::Cancelled,
                Err(message) => Outcome::Failed(message),
            }
        }
        Ok(None) => Outcome::Cancelled,
        Err(message) if message == picker::CANCELLED => Outcome::Cancelled,
        Err(message) => Outcome::Failed(message),
    }
}

fn await_raster(
    shared: &Shared,
    raster_rx: &Receiver<Result<Arc<[u8]>, &'static str>>,
) -> Result<Arc<[u8]>, &'static str> {
    loop {
        if shared.cancelled.load(Ordering::Acquire) {
            return Err(picker::CANCELLED);
        }
        match raster_rx.recv_timeout(POLL) {
            Ok(result) => return result,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Err(RASTER_UNAVAILABLE),
        }
    }
}

fn write_export(path: &Path, bytes: &[u8]) -> Outcome {
    match atomic_file::replace(path, bytes) {
        Ok(()) => Outcome::Saved(path.to_path_buf()),
        Err(_) => Outcome::Failed(SAVE_WRITE_FAILED),
    }
}

fn encode_png_from_bmg1(packet: &[u8]) -> Result<Vec<u8>, &'static str> {
    let decoded = markdown_service::decode_raster(packet)
        .map_err(|_| RASTER_FAILED)?
        .ok_or(UNRENDERABLE_IMAGE)?;
    let straight = unpremultiply_rgba8(decoded.premultiplied_rgba)?;
    let mut output = Vec::new();
    let mut encoder = png::Encoder::new(&mut output, decoded.width, decoded.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|_| SAVE_WRITE_FAILED)?;
    writer
        .write_image_data(&straight)
        .map_err(|_| SAVE_WRITE_FAILED)?;
    writer.finish().map_err(|_| SAVE_WRITE_FAILED)?;
    Ok(output)
}

fn unpremultiply_rgba8(premultiplied: &[u8]) -> Result<Vec<u8>, &'static str> {
    if !premultiplied.len().is_multiple_of(4) {
        return Err(RASTER_FAILED);
    }
    let mut straight = Vec::with_capacity(premultiplied.len());
    for pixel in premultiplied.chunks_exact(4) {
        let [r, g, b, a] = [pixel[0], pixel[1], pixel[2], pixel[3]];
        match a {
            0 => straight.extend_from_slice(&[0, 0, 0, 0]),
            255 => straight.extend_from_slice(&[r, g, b, 255]),
            alpha => {
                straight.push(unpremultiply_channel(r, alpha));
                straight.push(unpremultiply_channel(g, alpha));
                straight.push(unpremultiply_channel(b, alpha));
                straight.push(alpha);
            }
        }
    }
    Ok(straight)
}

fn unpremultiply_channel(channel: u8, alpha: u8) -> u8 {
    let a = u16::from(alpha);
    let scaled = (u16::from(channel) * 255 + a / 2) / a;
    u8::try_from(scaled.min(255)).unwrap_or(255)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_test_dir(tag: &str) -> io::Result<PathBuf> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "BonsaiImageExport-{tag}-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&dir)?;
        Ok(dir)
    }

    fn wait_outcome(export: &mut Export) -> Outcome {
        for _ in 0..200 {
            match export.poll() {
                Outcome::Pending => thread::sleep(Duration::from_millis(5)),
                terminal => return terminal,
            }
        }
        Outcome::Failed("timed out waiting for export")
    }

    #[test]
    fn svg_export_writes_exact_source_atomically() -> io::Result<()> {
        let dir = unique_test_dir("svg")?;
        let target = dir.join("diagram.svg");
        let svg: Arc<str> = Arc::from(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80"><rect width="120" height="80" fill="#2a9d8f"/></svg>"##,
        );
        let ctx = egui::Context::default();
        let mut export = Export::start_with_selection(
            &ctx,
            Selection {
                path: target.clone(),
                format: Format::Svg,
            },
            Arc::clone(&svg),
        )?;
        match wait_outcome(&mut export) {
            Outcome::Saved(saved) => assert_eq!(saved, target),
            _ => return Err(io::Error::other("expected Saved outcome for SVG")),
        }
        assert_eq!(fs::read_to_string(&target)?, &*svg);
        fs::remove_dir_all(dir)
    }

    #[test]
    fn png_export_encodes_unpremultiplied_rgba_and_decodes_cleanly() -> io::Result<()> {
        let dir = unique_test_dir("png")?;
        let target = dir.join("diagram.png");
        // Half-alpha red premultiplied with opaque green rectangle.
        let svg: Arc<str> = Arc::from(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect x="0" y="0" width="32" height="64" fill="rgb(255,0,0)" fill-opacity="0.5"/><rect x="32" y="0" width="32" height="64" fill="rgb(0,200,0)"/></svg>"#,
        );
        let ctx = egui::Context::default();
        let mut export = Export::start_with_selection(
            &ctx,
            Selection {
                path: target.clone(),
                format: Format::Png,
            },
            Arc::clone(&svg),
        )?;
        match wait_outcome(&mut export) {
            Outcome::Saved(saved) => assert_eq!(saved, target),
            _ => return Err(io::Error::other("expected Saved outcome for PNG")),
        }
        let png_bytes = fs::read(&target)?;
        assert!(png_bytes.starts_with(b"\x89PNG\r\n\x1a\n"));

        let decoder = png::Decoder::new(io::Cursor::new(png_bytes));
        let mut reader = decoder.read_info().map_err(io::Error::other)?;
        let mut pixels = vec![0_u8; reader.output_buffer_size().unwrap_or(0)];
        let info = reader.next_frame(&mut pixels).map_err(io::Error::other)?;
        assert_eq!(info.color_type, png::ColorType::Rgba);
        assert_eq!(info.bit_depth, png::BitDepth::Eight);
        let width = usize::try_from(info.width).map_err(io::Error::other)?;
        // Left half: translucent straight red (R >> G,B and alpha around 128).
        let left_offset = (usize::try_from(info.height).map_err(io::Error::other)? / 2) * width * 4
            + (width / 4) * 4;
        assert!(pixels[left_offset] >= 245);
        assert_eq!(pixels[left_offset + 1], 0);
        assert_eq!(pixels[left_offset + 2], 0);
        assert!((120..=136).contains(&pixels[left_offset + 3]));
        // Right half: opaque green.
        let right_offset =
            (usize::try_from(info.height).map_err(io::Error::other)? / 2) * width * 4
                + (3 * width / 4) * 4;
        assert_eq!(
            &pixels[right_offset..right_offset + 4],
            &[0_u8, 200_u8, 0_u8, 255_u8]
        );
        fs::remove_dir_all(dir)
    }

    #[test]
    fn unrenderable_svg_fails_png_cleanly_while_svg_source_save_succeeds() -> io::Result<()> {
        let dir = unique_test_dir("unrenderable")?;
        let png_target = dir.join("broken.png");
        let svg_target = dir.join("broken.svg");
        let broken: Arc<str> = Arc::from("<not-valid-svg>");
        let ctx = egui::Context::default();

        let mut png_export = Export::start_with_selection(
            &ctx,
            Selection {
                path: png_target.clone(),
                format: Format::Png,
            },
            Arc::clone(&broken),
        )?;
        match wait_outcome(&mut png_export) {
            Outcome::Failed(message) => assert_eq!(message, UNRENDERABLE_IMAGE),
            _ => return Err(io::Error::other("expected PNG failure on invalid SVG")),
        }
        assert!(!png_target.exists());
        assert_eq!(fs::read_dir(&dir)?.count(), 0);

        let mut svg_export = Export::start_with_selection(
            &ctx,
            Selection {
                path: svg_target.clone(),
                format: Format::Svg,
            },
            Arc::clone(&broken),
        )?;
        match wait_outcome(&mut svg_export) {
            Outcome::Saved(saved) => assert_eq!(saved, svg_target),
            _ => return Err(io::Error::other("expected raw SVG save to succeed")),
        }
        assert_eq!(fs::read_to_string(&svg_target)?, &*broken);
        fs::remove_dir_all(dir)
    }
}
