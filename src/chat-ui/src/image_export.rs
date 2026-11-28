//! Asynchronous native image export gated on offline MXC SVG sandbox validation.
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
    markdown_service::{self, Admission, Service, Status, Ticket},
    window::atomic_file,
};

mod picker;

use picker::{Format, Selection};

const EXPORT_RASTER_DIM: u32 = 2_048;
const POLL: Duration = Duration::from_millis(10);
const RASTER_UNAVAILABLE: &str = "Offline SVG sandbox is unavailable; image export was blocked.";
const RASTER_FAILED: &str = "The offline SVG sandbox failed to validate or rasterize this image.";
const UNRENDERABLE_IMAGE: &str = "This SVG failed static safety validation and cannot be exported.";
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
        Self::start_with(ctx, svg, "bonsai-image-save", move |shared| {
            picker::pick_save_target(owner, &shared.picker, &shared.cancelled, &suggested_stem)
        })
    }

    #[cfg(test)]
    fn start_with_selection(
        ctx: &egui::Context,
        selection: Selection,
        svg: Arc<str>,
    ) -> io::Result<Self> {
        Self::start_with(ctx, svg, "bonsai-image-save-test-worker", move |_| {
            Ok(Some(selection))
        })
    }

    fn start_with(
        ctx: &egui::Context,
        svg: Arc<str>,
        thread_name: &'static str,
        select: impl FnOnce(&Shared) -> Result<Option<Selection>, &'static str> + Send + 'static,
    ) -> io::Result<Self> {
        let (raster_tx, raster_rx) = mpsc::sync_channel::<Result<Arc<[u8]>, &'static str>>(1);
        let raster = prepare_raster(ctx, &svg, &raster_tx);
        let shared = Arc::new(Shared::default());
        let worker_shared = Arc::clone(&shared);
        let (outcome_tx, outcome_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name(thread_name.to_owned())
            .spawn(move || {
                let outcome =
                    run_worker(&worker_shared, || select(&worker_shared), &svg, &raster_rx);
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

    /// Requests non-blocking cancellation of an active Save dialog or pending raster export.
    pub(crate) fn cancel(&self) {
        self.shared.cancelled.store(true, Ordering::Release);
        self.shared.picker.cancel(&self.shared.cancelled);
    }

    #[cfg(test)]
    pub(crate) fn fixture_pending() -> io::Result<Self> {
        let (raster_tx, _raster_rx) = mpsc::sync_channel::<Result<Arc<[u8]>, &'static str>>(1);
        let shared = Arc::new(Shared::default());
        let worker_shared = Arc::clone(&shared);
        let (outcome_tx, outcome_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("bonsai-image-export-fixture".to_owned())
            .spawn(move || {
                while !worker_shared.cancelled.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(2));
                }
                let _ = outcome_tx.send(Outcome::Cancelled);
            })?;
        Ok(Self {
            shared,
            outcome: outcome_rx,
            raster_tx: Some(raster_tx),
            raster: None,
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
                match service.request_raster(Arc::clone(&svg), EXPORT_RASTER_DIM, EXPORT_RASTER_DIM)
                {
                    Admission::Admitted(ticket) => {
                        self.raster = Some(PendingRaster::Ticket(service, ticket));
                    }
                    Admission::InvalidInput => {
                        let _sent = tx.try_send(Err(UNRENDERABLE_IMAGE));
                        drop(self.raster_tx.take());
                    }
                    Admission::Unavailable => {
                        let _sent = tx.try_send(Err(RASTER_FAILED));
                        drop(self.raster_tx.take());
                    }
                    Admission::Busy => {
                        self.raster = Some(PendingRaster::Retry { service, svg });
                    }
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
            let _joined = crate::com_dialog::join_thread_bounded(
                worker,
                crate::com_dialog::DROP_JOIN_TIMEOUT,
            );
        }
        let _joined = self
            .shared
            .picker
            .join_bounded(crate::com_dialog::DROP_JOIN_TIMEOUT);
    }
}

impl Drop for Export {
    fn drop(&mut self) {
        self.cancel();
        self.finish_resources();
    }
}

fn prepare_raster(
    ctx: &egui::Context,
    svg: &Arc<str>,
    raster_tx: &SyncSender<Result<Arc<[u8]>, &'static str>>,
) -> Option<PendingRaster> {
    if let Some(service) = Service::get(ctx) {
        match service.request_raster(Arc::clone(svg), EXPORT_RASTER_DIM, EXPORT_RASTER_DIM) {
            Admission::Admitted(ticket) => Some(PendingRaster::Ticket(service, ticket)),
            Admission::InvalidInput => {
                let _sent = raster_tx.try_send(Err(UNRENDERABLE_IMAGE));
                None
            }
            Admission::Unavailable => {
                let _sent = raster_tx.try_send(Err(RASTER_FAILED));
                None
            }
            Admission::Busy => Some(PendingRaster::Retry {
                service,
                svg: Arc::clone(svg),
            }),
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
        })) => {
            // Never write untrusted SVG markup to disk unless the offline MXC worker
            // accepted and rasterized the exact payload under the static-SVG policy.
            match await_raster(shared, raster_rx).and_then(|packet| verify_validated_bmg1(&packet))
            {
                Ok(()) => write_export(&path, svg.as_bytes()),
                Err(message) if message == picker::CANCELLED => Outcome::Cancelled,
                Err(message) => Outcome::Failed(message),
            }
        }
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

fn verify_validated_bmg1(packet: &[u8]) -> Result<(), &'static str> {
    let _decoded = markdown_service::decode_raster(packet)
        .map_err(|_| RASTER_FAILED)?
        .ok_or(UNRENDERABLE_IMAGE)?;
    Ok(())
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
    match atomic_file::replace_user_export_file(path, bytes) {
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
    fn unrenderable_or_scripted_svg_blocks_both_png_and_svg_exports() -> io::Result<()> {
        let dir = unique_test_dir("unrenderable")?;
        let png_target = dir.join("broken.png");
        let svg_target = dir.join("scripted.svg");
        let broken: Arc<str> = Arc::from("<not-valid-svg>");
        let scripted: Arc<str> = Arc::from(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><script>alert(1)</script><rect width="64" height="64"/></svg>"#,
        );
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

        let mut svg_export = Export::start_with_selection(
            &ctx,
            Selection {
                path: svg_target.clone(),
                format: Format::Svg,
            },
            Arc::clone(&scripted),
        )?;
        match wait_outcome(&mut svg_export) {
            Outcome::Failed(message) => assert_eq!(message, UNRENDERABLE_IMAGE),
            _ => {
                return Err(io::Error::other(
                    "expected SVG save failure on scripted SVG",
                ));
            }
        }
        assert!(!svg_target.exists());
        assert_eq!(fs::read_dir(&dir)?.count(), 0);
        fs::remove_dir_all(dir)
    }

    #[test]
    fn malformed_worker_packet_rejects_session_and_blocks_svg_export() -> io::Result<()> {
        let dir = unique_test_dir("forged")?;
        let svg_target = dir.join("forged.svg");
        let svg: Arc<str> = Arc::from(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="teal"/></svg>"#,
        );
        let ctx = egui::Context::default();
        // Forged non-premultiplied RGBA8 (`r = 250 > a = 10`) from a compromised worker.
        let (service, mut step) = Service::custom_responder(&ctx, |_, _| {
            let mut bad = b"BMG1".to_vec();
            bad.push(1);
            bad.extend_from_slice(&1_u32.to_le_bytes());
            bad.extend_from_slice(&1_u32.to_le_bytes());
            bad.extend_from_slice(&[250, 0, 0, 10]);
            bad
        });

        let mut export = Export::start_with_selection(
            &ctx,
            Selection {
                path: svg_target.clone(),
                format: Format::Svg,
            },
            Arc::clone(&svg),
        )?;
        assert_eq!(&*step()?, &*svg);
        match wait_outcome(&mut export) {
            Outcome::Failed(message) => assert_eq!(message, RASTER_FAILED),
            _ => return Err(io::Error::other("expected forged worker packet to fail")),
        }
        assert!(service.unavailable());
        assert!(!svg_target.exists());
        assert_eq!(fs::read_dir(&dir)?.count(), 0);
        fs::remove_dir_all(dir)
    }

    #[test]
    fn svg_export_supports_git_directories_percent_names_and_multi_dot_extensions() -> io::Result<()>
    {
        let dir = unique_test_dir("win11-path")?;
        let git_dir = dir.join(".git");
        fs::create_dir(&git_dir)?;
        let target = git_dir.join("budget 100%.v2.svg");
        let svg: Arc<str> = Arc::from(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="#2a9d8f"/></svg>"##,
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
            _ => {
                return Err(io::Error::other(
                    "expected Saved outcome for .git/%/multi-dot export",
                ));
            }
        }
        assert_eq!(fs::read_to_string(&target)?, &*svg);
        assert_eq!(fs::read_dir(&git_dir)?.count(), 1);
        fs::remove_dir_all(dir)
    }

    #[test]
    fn cancel_request_is_non_blocking_and_poll_reaps_cancelled_worker() -> io::Result<()> {
        let ctx = egui::Context::default();
        let svg: Arc<str> = Arc::from(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32"/></svg>"#,
        );
        let mut export = Export::start_with(&ctx, svg, "bonsai-image-cancel-test", |shared| {
            while !shared.cancelled.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(5));
            }
            Err(picker::CANCELLED)
        })?;
        assert!(matches!(export.poll(), Outcome::Pending));
        export.cancel();
        assert!(matches!(wait_outcome(&mut export), Outcome::Cancelled));
        Ok(())
    }

    #[test]
    fn export_handles_invalid_busy_retry_and_unavailable_admissions() -> io::Result<()> {
        let dir = unique_test_dir("admission")?;
        let svg: Arc<str> = Arc::from(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="navy"/></svg>"#,
        );

        // Admission::InvalidInput -> UNRENDERABLE_IMAGE without opening picker save.
        let ctx_invalid = egui::Context::default();
        let _respond_invalid = Service::fixture(&ctx_invalid);
        let mut invalid_export = Export::start_with_selection(
            &ctx_invalid,
            Selection {
                path: dir.join("empty.png"),
                format: Format::Png,
            },
            Arc::from(""),
        )?;
        match wait_outcome(&mut invalid_export) {
            Outcome::Failed(message) => assert_eq!(message, UNRENDERABLE_IMAGE),
            _ => return Err(io::Error::other("expected UNRENDERABLE_IMAGE on empty SVG")),
        }

        // Admission::Busy -> PendingRaster::Retry, then Ticket once queue clears -> Saved.
        let ctx_busy = egui::Context::default();
        let respond_busy = Service::fixture(&ctx_busy);
        let service =
            Service::get(&ctx_busy).ok_or_else(|| io::Error::other("missing fixture service"))?;
        let mut filler = Vec::new();
        while let Admission::Admitted(ticket) = service.request(Arc::from("filler")) {
            filler.push(ticket);
        }
        let target = dir.join("retried.png");
        let mut busy_export = Export::start_with_selection(
            &ctx_busy,
            Selection {
                path: target.clone(),
                format: Format::Png,
            },
            Arc::clone(&svg),
        )?;
        assert!(matches!(busy_export.poll(), Outcome::Pending));
        if let Some(first) = filler.pop() {
            let _ = respond_busy();
            service.release(first);
        }
        assert!(matches!(busy_export.poll(), Outcome::Pending));
        // Drain remaining filler requests so the retry ticket reaches the responder.
        for _ in 0..filler.len() {
            let _ = respond_busy();
        }
        assert_eq!(&*respond_busy()?, &*svg);
        match wait_outcome(&mut busy_export) {
            Outcome::Saved(saved) => assert_eq!(saved, target),
            _ => return Err(io::Error::other("expected Saved after Busy retry")),
        }

        // Admission::Unavailable -> RASTER_FAILED.
        service.reject();
        let mut unavail_export = Export::start_with_selection(
            &ctx_busy,
            Selection {
                path: dir.join("unavail.png"),
                format: Format::Png,
            },
            Arc::clone(&svg),
        )?;
        match wait_outcome(&mut unavail_export) {
            Outcome::Failed(message) => assert_eq!(message, RASTER_FAILED),
            _ => return Err(io::Error::other("expected RASTER_FAILED on Unavailable")),
        }

        fs::remove_dir_all(dir)
    }
}
