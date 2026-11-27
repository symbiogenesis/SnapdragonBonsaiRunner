//! Bounded PCM playback with sequential model requests and audio prefetch.

use super::{Arc, AtomicU64, JoinHandle, Ordering, Receiver, VecDeque, io, mpsc, thread, tts};
use std::{
    sync::{
        Condvar, Mutex,
        atomic::AtomicBool,
        mpsc::{SyncSender, TrySendError},
    },
    time::{Duration, Instant},
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0},
        Media::Audio::{
            AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, ActivateAudioInterfaceAsync,
            IActivateAudioInterfaceCompletionHandler, IAudioClient, IAudioRenderClient,
            WAVEFORMATEX,
        },
        System::Threading::{CreateEventW, WaitForSingleObject},
    },
    core::{HRESULT, Interface, PCWSTR, w},
};

const AUDIO_FAILURE: tts::Failure =
    tts::Failure::Message("Audio playback failed. Check the Windows audio output device.");
const POLL: Duration = Duration::from_millis(5);
/// Default 24 kHz mono output sample rate.
pub(super) const SAMPLE_RATE: u32 = 24_000;
/// Upgraded 48 kHz mono output sample rate (`takuma104/Qwen3-TTS-Tokenizer-12Hz-48kHz`).
pub(super) const SAMPLE_RATE_48K: u32 = 48_000;
// Normalize callbacks into 100 ms of mono PCM16 (4,800 bytes at 24 kHz; 9,600 bytes
// at 48 kHz), so queue bounds describe audio duration even when chunks are tiny.
const PACKET_BYTES: usize = 4_800;
pub(super) const PACKET_BYTES_48K: usize = 9_600;
// Five milliseconds at the largest supported rate. Hold only this tail so exact
// packet-aligned segment endings can also reach silence without losing samples.
const MAX_EDGE_BYTES: usize = PACKET_BYTES_48K / 20;
const PREFETCH_PACKETS: usize = 160; // 16 seconds / 768,000 bytes at 24 kHz, outside WASAPI.
const PLAYBACK_PACKETS: usize = 8;
// Only packets fully written into WASAPI enter this pool. Full queues discard
// excess buffers, keeping retained spare PCM capacity bounded to one playback queue.
const RECYCLED_PACKETS: usize = PLAYBACK_PACKETS;
// The Rust codec emits one 80 ms frame per callback. Observe three packet
// intervals before starting, retaining four packets to absorb callback jitter.
const PREROLL_PACKETS: usize = 4;
const CODEC_BURST: Duration = Duration::from_millis(80);
const MAX_BUFFER_BYTES: usize = 48_000 * 120;

pub(super) const fn packet_bytes_for_rate(sample_rate: u32) -> usize {
    if sample_rate == SAMPLE_RATE_48K {
        PACKET_BYTES_48K
    } else {
        PACKET_BYTES
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Clamped i16 PCM range [-32768, 32767] fits in i16 after rounding."
)]
fn float_to_i16(value: f32) -> i16 {
    value
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX))
        .round() as i16
}

// Explicit endpoints also cover a one-sample clip and avoid rounding a residual
// step into silence. The half cosine has zero slope at both ends of the ramp.
fn edge_gain(index: usize, count: usize) -> f32 {
    if index == 0 || count <= 1 {
        0.0
    } else if index + 1 >= count {
        1.0
    } else {
        let step = f32::from(u16::try_from(index).unwrap_or(0));
        let denominator = f32::from(u16::try_from(count - 1).unwrap_or(1));
        0.5 * (1.0 - (core::f32::consts::PI * step / denominator).cos())
    }
}

enum Audio {
    Frame(Vec<u8>),
    SegmentEnd,
}

pub(super) struct Pipeline {
    segments: Option<SyncSender<String>>,
    finished: Receiver<Result<(), tts::Failure>>,
    cancelled: Arc<AtomicBool>,
    playing: Arc<AtomicBool>,
    notified_playing: bool,
    worker: Option<JoinHandle<()>>,
}

impl Pipeline {
    pub(super) fn start(
        client: Arc<tts::Client>,
        voice: String,
        generation: u64,
        current: Arc<AtomicU64>,
    ) -> io::Result<Self> {
        let (complete, finished) = mpsc::channel();
        let (segments, incoming) = mpsc::sync_channel(2);
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&cancelled);
        let playing = Arc::new(AtomicBool::new(false));
        let output_playing = Arc::clone(&playing);
        let worker = thread::Builder::new()
            .name("bonsai-tts-prefetch".to_owned())
            .spawn(move || {
                let result = produce(
                    &client,
                    &incoming,
                    &voice,
                    generation,
                    &current,
                    &stop,
                    &output_playing,
                );
                let _sent = complete.send(result);
            })?;
        Ok(Self {
            segments: Some(segments),
            finished,
            cancelled,
            playing,
            notified_playing: false,
            worker: Some(worker),
        })
    }

    pub(super) fn feed(&mut self, queued: &mut VecDeque<String>, finished: bool) {
        if let Some(sender) = &self.segments {
            while let Some(mut text) = queued.pop_front() {
                // Batch before the two-entry channel: punctuation need not
                // restart synthesis when more of the answer is already ready.
                let mut characters = text.chars().count();
                while let Some(next) = queued.front() {
                    if !append_segment(&mut text, &mut characters, next) {
                        break;
                    }
                    let _merged = queued.pop_front();
                }
                match sender.try_send(text) {
                    Ok(()) => {}
                    Err(TrySendError::Full(text)) => {
                        queued.push_front(text);
                        break;
                    }
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
        }
        if finished && queued.is_empty() {
            self.segments = None;
        }
    }

    pub(super) fn playback_changed(&mut self) -> Option<bool> {
        let playing = self.playing.load(Ordering::Acquire);
        if playing == self.notified_playing {
            return None;
        }
        self.notified_playing = playing;
        Some(playing)
    }

    pub(super) fn result(&self) -> Option<Result<(), tts::Failure>> {
        match self.finished.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(AUDIO_FAILURE)),
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _joined = worker.join();
        }
    }
}

fn produce(
    client: &tts::Client,
    segments: &Receiver<String>,
    voice: &str,
    generation: u64,
    current: &Arc<AtomicU64>,
    stop: &Arc<AtomicBool>,
    playing: &Arc<AtomicBool>,
) -> Result<(), tts::Failure> {
    let sample_rate = client.sample_rate();
    let packet_bytes = packet_bytes_for_rate(sample_rate);
    let (sender, incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
    let (recycle, recycled) = mpsc::sync_channel(RECYCLED_PACKETS);
    let output_stop = Arc::clone(stop);
    let output_current = Arc::clone(current);
    let output_playing = Arc::clone(playing);
    let playback = thread::Builder::new()
        .name("bonsai-tts-pcm".to_owned())
        .spawn(move || {
            let result = consume_at_rate(
                &incoming,
                generation,
                &output_current,
                &output_stop,
                recycle,
                sample_rate,
                &output_playing,
            );
            if result.is_err() {
                output_stop.store(true, Ordering::Release);
            }
            result
        })
        .map_err(|_| AUDIO_FAILURE)?;
    let synthesis = (|| {
        let mut pending = None;
        let mut frames = SegmentFrames::reusing_with_packet_bytes(recycled, packet_bytes);
        while let Some(text) = next_segment(segments, &mut pending, generation, current, stop)? {
            client.synthesize_pcm(&text, voice, generation, current, stop, |pcm| {
                frames.push(pcm, |packet| {
                    send(&sender, Audio::Frame(packet), generation, current, stop)
                })
            })?;
            frames.finish_segment(|packet| {
                send(&sender, Audio::Frame(packet), generation, current, stop)
            })?;
            send(&sender, Audio::SegmentEnd, generation, current, stop)?;
        }
        Ok(())
    })();
    if synthesis.is_err() {
        stop.store(true, Ordering::Release);
    }
    drop(sender);
    let played = playback.join().unwrap_or(Err(AUDIO_FAILURE));
    if let Err(failure) = played
        && failure != tts::Failure::Cancelled
    {
        return Err(failure);
    }
    synthesis?;
    played
}

fn next_segment(
    segments: &Receiver<String>,
    pending: &mut Option<String>,
    generation: u64,
    current: &AtomicU64,
    stop: &AtomicBool,
) -> Result<Option<String>, tts::Failure> {
    let mut text = loop {
        check(generation, current, stop)?;
        if let Some(text) = pending.take() {
            break text;
        }
        match segments.recv_timeout(POLL) {
            Ok(text) => break text,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
        }
    };
    let mut characters = text.chars().count();
    while let Ok(next) = segments.try_recv() {
        check(generation, current, stop)?;
        if !append_segment(&mut text, &mut characters, &next) {
            *pending = Some(next);
            break;
        }
    }
    Ok(Some(text))
}

fn append_segment(text: &mut String, characters: &mut usize, next: &str) -> bool {
    // Preserve paragraph cues supplied by segmentation and charge only the
    // separator actually needed. An extra space can otherwise force a new
    // model request even when the original two pieces fit exactly.
    let needs_space =
        !text.ends_with(char::is_whitespace) && !next.starts_with(char::is_whitespace);
    let added = usize::from(needs_space) + next.chars().count();
    if *characters + added > super::MAX_SEGMENT {
        return false;
    }
    if needs_space {
        text.push(' ');
    }
    text.push_str(next);
    *characters += added;
    true
}

struct Frames {
    pending: Vec<u8>,
    recycled: Option<Receiver<Vec<u8>>>,
    packet_bytes: usize,
}

impl Default for Frames {
    fn default() -> Self {
        Self {
            pending: Vec::with_capacity(PACKET_BYTES),
            recycled: None,
            packet_bytes: PACKET_BYTES,
        }
    }
}

impl Frames {
    #[cfg(test)]
    fn reusing(recycled: Receiver<Vec<u8>>) -> Self {
        Self::reusing_with_packet_bytes(recycled, PACKET_BYTES)
    }

    fn reusing_with_packet_bytes(recycled: Receiver<Vec<u8>>, packet_bytes: usize) -> Self {
        Self {
            pending: Vec::with_capacity(packet_bytes),
            recycled: Some(recycled),
            packet_bytes,
        }
    }

    fn replacement(&self) -> Vec<u8> {
        self.recycled
            .as_ref()
            .and_then(|recycled| recycled.try_recv().ok())
            .unwrap_or_else(|| Vec::with_capacity(self.packet_bytes))
    }

    fn push(
        &mut self,
        mut pcm: &[u8],
        mut sink: impl FnMut(Vec<u8>) -> Result<(), tts::Failure>,
    ) -> Result<(), tts::Failure> {
        while !pcm.is_empty() {
            let start_offset = self.pending.len();
            let count = pcm.len().min(self.packet_bytes - start_offset);
            self.pending.extend_from_slice(&pcm[..count]);
            pcm = &pcm[count..];
            if self.pending.len() == self.packet_bytes {
                let replacement = self.replacement();
                sink(std::mem::replace(&mut self.pending, replacement))?;
            }
        }
        Ok(())
    }

    fn finish(
        &mut self,
        sink: impl FnOnce(Vec<u8>) -> Result<(), tts::Failure>,
    ) -> Result<(), tts::Failure> {
        if self.pending.is_empty() {
            Ok(())
        } else if self.pending.len().is_multiple_of(2) {
            let replacement = self.replacement();
            sink(std::mem::replace(&mut self.pending, replacement))
        } else {
            Err(AUDIO_FAILURE)
        }
    }
}

// Smoothing belongs to synthesized segments, rather than codec/HTTP callbacks.
// PCM framing remains a separate byte-preserving operation with recycled packets.
struct SegmentFrames {
    frames: Frames,
    tail: [u8; MAX_EDGE_BYTES],
    tail_len: usize,
    split_byte: Option<u8>,
    onset_samples: usize,
}

impl Default for SegmentFrames {
    fn default() -> Self {
        Self::with_frames(Frames::default())
    }
}

impl SegmentFrames {
    const fn with_frames(frames: Frames) -> Self {
        Self {
            frames,
            tail: [0; MAX_EDGE_BYTES],
            tail_len: 0,
            split_byte: None,
            onset_samples: 0,
        }
    }

    fn reusing_with_packet_bytes(recycled: Receiver<Vec<u8>>, packet_bytes: usize) -> Self {
        Self::with_frames(Frames::reusing_with_packet_bytes(recycled, packet_bytes))
    }

    const fn edge_bytes(&self) -> usize {
        self.frames.packet_bytes / 20
    }

    fn retain_tail(
        &mut self,
        pcm: &[u8],
        sink: &mut impl FnMut(Vec<u8>) -> Result<(), tts::Failure>,
    ) -> Result<(), tts::Failure> {
        let release = (self.tail_len + pcm.len()).saturating_sub(self.edge_bytes());
        let old = release.min(self.tail_len);
        self.frames.push(&self.tail[..old], &mut *sink)?;
        self.tail.copy_within(old..self.tail_len, 0);
        self.tail_len -= old;
        let incoming = release - old;
        self.frames.push(&pcm[..incoming], sink)?;
        let remaining = &pcm[incoming..];
        self.tail[self.tail_len..self.tail_len + remaining.len()].copy_from_slice(remaining);
        self.tail_len += remaining.len();
        Ok(())
    }

    fn push_samples(
        &mut self,
        pcm: &[u8],
        sink: &mut impl FnMut(Vec<u8>) -> Result<(), tts::Failure>,
    ) -> Result<(), tts::Failure> {
        let edge_samples = self.edge_bytes() / 2;
        let onset_bytes = pcm.len().min((edge_samples - self.onset_samples) * 2);
        if onset_bytes > 0 {
            let mut onset = [0; MAX_EDGE_BYTES];
            for (input, output) in pcm[..onset_bytes]
                .as_chunks::<2>()
                .0
                .iter()
                .zip(onset[..onset_bytes].as_chunks_mut::<2>().0.iter_mut())
            {
                let sample = f32::from(i16::from_le_bytes([input[0], input[1]]));
                output.copy_from_slice(
                    &float_to_i16(sample * edge_gain(self.onset_samples, edge_samples))
                        .to_le_bytes(),
                );
                self.onset_samples += 1;
            }
            self.retain_tail(&onset[..onset_bytes], sink)?;
        }
        self.retain_tail(&pcm[onset_bytes..], sink)
    }

    fn push(
        &mut self,
        mut pcm: &[u8],
        mut sink: impl FnMut(Vec<u8>) -> Result<(), tts::Failure>,
    ) -> Result<(), tts::Failure> {
        if pcm.is_empty() {
            return Ok(());
        }
        if let Some(first) = self.split_byte.take() {
            self.push_samples(&[first, pcm[0]], &mut sink)?;
            pcm = &pcm[1..];
        }
        let aligned = pcm.len() - pcm.len() % 2;
        self.push_samples(&pcm[..aligned], &mut sink)?;
        self.split_byte = pcm.get(aligned).copied();
        Ok(())
    }

    fn finish_segment(
        &mut self,
        mut sink: impl FnMut(Vec<u8>) -> Result<(), tts::Failure>,
    ) -> Result<(), tts::Failure> {
        if self.split_byte.is_some() {
            return Err(AUDIO_FAILURE);
        }
        let count = self.tail_len / 2;
        for (index, sample) in self.tail[..self.tail_len]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .enumerate()
        {
            let value = f32::from(i16::from_le_bytes([sample[0], sample[1]]));
            sample.copy_from_slice(
                &float_to_i16(value * edge_gain(count - index - 1, count)).to_le_bytes(),
            );
        }
        self.frames.push(&self.tail[..self.tail_len], &mut sink)?;
        self.frames.finish(sink)?;
        self.tail_len = 0;
        self.onset_samples = 0;
        Ok(())
    }
}

fn recycle(sender: &SyncSender<Vec<u8>>, mut pcm: Vec<u8>) {
    if pcm.capacity() == PACKET_BYTES || pcm.capacity() == PACKET_BYTES_48K {
        pcm.clear();
        let _returned = sender.try_send(pcm);
    }
}

fn check(generation: u64, current: &AtomicU64, stop: &AtomicBool) -> Result<(), tts::Failure> {
    if stop.load(Ordering::Acquire) || current.load(Ordering::Acquire) != generation {
        Err(tts::Failure::Cancelled)
    } else {
        Ok(())
    }
}

fn send(
    sender: &SyncSender<Audio>,
    mut pcm: Audio,
    generation: u64,
    current: &AtomicU64,
    stop: &AtomicBool,
) -> Result<(), tts::Failure> {
    loop {
        check(generation, current, stop)?;
        match sender.try_send(pcm) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(packet)) => {
                pcm = packet;
                thread::sleep(POLL);
            }
            Err(TrySendError::Disconnected(_)) => return Err(AUDIO_FAILURE),
        }
    }
}

fn consume_at_rate(
    incoming: &Receiver<Audio>,
    generation: u64,
    current: &AtomicU64,
    stop: &AtomicBool,
    recycle: SyncSender<Vec<u8>>,
    sample_rate: u32,
    playing: &AtomicBool,
) -> Result<(), tts::Failure> {
    consume_with_progress(
        incoming,
        generation,
        current,
        stop,
        &mut WasapiOutput::open_with_rate(recycle, sample_rate)?,
        playing,
    )
}

trait Output {
    fn queued(&self) -> usize;
    fn push(&mut self, pcm: Vec<u8>) -> Result<(), tts::Failure>;
    fn reap(&mut self) -> Result<(), tts::Failure>;
    fn pause(&mut self) -> Result<(), tts::Failure>;
    fn start(&mut self) -> Result<(), tts::Failure>;
    fn wait(&mut self, _timeout: Duration) {}
}

#[cfg(test)]
fn consume_with(
    incoming: &Receiver<Audio>,
    generation: u64,
    current: &AtomicU64,
    stop: &AtomicBool,
    output: &mut impl Output,
) -> Result<(), tts::Failure> {
    consume_with_progress(
        incoming,
        generation,
        current,
        stop,
        output,
        &AtomicBool::new(false),
    )
}

fn consume_with_progress(
    incoming: &Receiver<Audio>,
    generation: u64,
    current: &AtomicU64,
    stop: &AtomicBool,
    output: &mut impl Output,
    playing: &AtomicBool,
) -> Result<(), tts::Failure> {
    output.pause()?;
    playing.store(false, Ordering::Release);
    let mut started = false;
    let mut buffered = Buffered::default();
    let mut pending = None;
    let mut finished = false;
    loop {
        check(generation, current, stop)?;
        output.reap()?;
        if started
            && output.queued() == 0
            && buffered.ready.is_empty()
            && pending.is_none()
            && !finished
        {
            // A late consumer wake can find the WASAPI buffer empty while the producer
            // has already queued more audio. Refill before latching slow synthesis;
            // otherwise scheduler jitter buffers the rest of a healthy segment.
            (pending, finished) =
                refill_queued(incoming, &mut buffered, generation, current, stop)?;
        }
        if started && output.queued() == 0 && buffered.ready.is_empty() {
            output.pause()?;
            if buffered.released {
                buffered.underrun();
            }
            started = false;
            playing.store(false, Ordering::Release);
        }
        while output.queued() < PLAYBACK_PACKETS {
            if let Some(pcm) = buffered.pop() {
                output.push(pcm)?;
                let flushed = buffered.ready.is_empty() && buffered.flush;
                if flushed {
                    buffered.flush = false;
                }
                if !started && (output.queued() >= PREROLL_PACKETS || flushed) {
                    output.start()?;
                    started = true;
                    playing.store(true, Ordering::Release);
                }
            } else {
                break;
            }
        }
        if buffered.ready.is_empty() && buffered.flush {
            buffered.flush = false;
            if !started && output.queued() > 0 {
                output.start()?;
                started = true;
                playing.store(true, Ordering::Release);
            }
        }
        if finished && buffered.ready.is_empty() && output.queued() == 0 {
            return Ok(());
        }
        let audio = if pending.is_some() {
            pending.take()
        } else if finished {
            None
        } else {
            match incoming.recv_timeout(POLL) {
                Ok(audio) => Some(audio),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if buffered.segment_bytes != 0 {
                        return Err(AUDIO_FAILURE);
                    }
                    finished = true;
                    None
                }
            }
        };
        check(generation, current, stop)?;
        match audio {
            Some(Audio::Frame(pcm)) => {
                if let Some(pcm) = buffered.frame(pcm, Instant::now())? {
                    pending = Some(Audio::Frame(pcm));
                }
            }
            Some(Audio::SegmentEnd) => buffered.end()?,
            None => {}
        }
        if pending.is_some() || finished {
            output.wait(POLL);
        }
    }
}

fn refill_queued(
    incoming: &Receiver<Audio>,
    buffered: &mut Buffered,
    generation: u64,
    current: &AtomicU64,
    stop: &AtomicBool,
) -> Result<(Option<Audio>, bool), tts::Failure> {
    for _ in 0..PREFETCH_PACKETS {
        check(generation, current, stop)?;
        match incoming.try_recv() {
            Ok(Audio::Frame(pcm)) => {
                if let Some(pcm) = buffered.frame(pcm, Instant::now())? {
                    return Ok((Some(Audio::Frame(pcm)), false));
                }
            }
            Ok(Audio::SegmentEnd) => buffered.end()?,
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                if buffered.segment_bytes != 0 {
                    return Err(AUDIO_FAILURE);
                }
                return Ok((None, true));
            }
        }
        if !buffered.ready.is_empty() {
            break;
        }
    }
    Ok((None, false))
}

// At most one 120-second clip of application staging, plus the 16-second
// producer channel and 800 ms WASAPI queue. Slow synthesis waits for a validated
// segment instead of repeatedly playing short bursts separated by silence.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Playback staging tracks four independent boolean lifecycle latches."
)]
struct Buffered {
    ready: VecDeque<Vec<u8>>,
    segment: VecDeque<Vec<u8>>,
    bytes: usize,
    segment_bytes: usize,
    packet_bytes: usize,
    first: Option<(Instant, usize)>,
    released: bool,
    slow: bool,
    underrun: bool,
    flush: bool,
}

impl Buffered {
    fn frame(&mut self, pcm: Vec<u8>, now: Instant) -> Result<Option<Vec<u8>>, tts::Failure> {
        let max_packet = if pcm.len() > PACKET_BYTES || self.packet_bytes == PACKET_BYTES_48K {
            PACKET_BYTES_48K
        } else {
            PACKET_BYTES
        };
        self.packet_bytes = max_packet;
        let max_buffer = if max_packet == PACKET_BYTES_48K {
            MAX_BUFFER_BYTES * 2
        } else {
            MAX_BUFFER_BYTES
        };
        if pcm.is_empty()
            || pcm.len() > max_packet
            || !pcm.len().is_multiple_of(2)
            || self.segment_bytes + pcm.len() > max_buffer
        {
            return Err(AUDIO_FAILURE);
        }
        if self.bytes + pcm.len() > max_buffer {
            return Ok(Some(pcm));
        }
        let (first, initial) = *self.first.get_or_insert((now, pcm.len()));
        self.segment_bytes += pcm.len();
        self.bytes += pcm.len();
        if self.released {
            self.ready.push_back(pcm);
        } else {
            let bytes_per_ms = if max_packet == PACKET_BYTES_48K {
                96
            } else {
                48
            };
            self.segment.push_back(pcm);
            if self.segment.len() >= PREROLL_PACKETS
                && self.segment_bytes >= PREROLL_PACKETS * max_packet
            {
                // Exclude cold first-audio latency. This compares later supply
                // time to the duration it produced, not to model load time.
                let duration = Duration::from_millis(
                    u64::try_from(self.segment_bytes - initial).map_err(|_| AUDIO_FAILURE)?
                        / bytes_per_ms,
                );
                // Allow half a codec frame of observed jitter while rejecting sustained
                // slower-than-playback supply before releasing the first audio.
                self.slow |= now.saturating_duration_since(first) > duration + CODEC_BURST / 2;
                if !self.slow {
                    self.ready.append(&mut self.segment);
                    self.released = true;
                }
            }
        }
        Ok(None)
    }

    fn end(&mut self) -> Result<(), tts::Failure> {
        if self.segment_bytes == 0 {
            return Err(AUDIO_FAILURE);
        }
        if self.segment.is_empty() {
            self.underrun = false;
        }
        self.ready.append(&mut self.segment);
        self.segment_bytes = 0;
        self.first = None;
        self.released = false;
        self.slow = self.underrun;
        self.flush = true;
        Ok(())
    }

    fn pop(&mut self) -> Option<Vec<u8>> {
        let pcm = self.ready.pop_front()?;
        self.bytes -= pcm.len();
        Some(pcm)
    }

    const fn underrun(&mut self) {
        self.underrun = true;
        self.slow = true;
        self.released = false;
    }
}

const DEVINTERFACE_AUDIO_RENDER_PATH: PCWSTR = w!("{e6327cad-dcec-4949-ae8a-991e976a79d2}");
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WASAPI_BUFFER_HNS: i64 = 8_000_000;
const WASAPI_STREAM_FLAGS: u32 = AUDCLNT_STREAMFLAGS_EVENTCALLBACK
    | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
    | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
const ACTIVATE_TIMEOUT: Duration = Duration::from_secs(5);

const fn normalize_sample_rate(sample_rate: u32) -> u32 {
    if sample_rate == SAMPLE_RATE_48K {
        SAMPLE_RATE_48K
    } else {
        SAMPLE_RATE
    }
}

const fn float_wave_format(sample_rate: u32) -> WAVEFORMATEX {
    let rate = normalize_sample_rate(sample_rate);
    WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_IEEE_FLOAT,
        nChannels: 1,
        nSamplesPerSec: rate,
        nAvgBytesPerSec: rate * 4,
        nBlockAlign: 4,
        wBitsPerSample: 32,
        cbSize: 0,
    }
}

fn pcm16_to_f32(sample: [u8; 2]) -> f32 {
    f32::from(i16::from_le_bytes(sample)) * (1.0 / 32_768.0)
}

#[allow(
    trivial_casts,
    clippy::ref_as_ptr,
    clippy::inline_always,
    reason = "windows::core::implement macro expands COM vtable casts and inline methods."
)]
mod activation {
    use std::sync::{Arc, Condvar, Mutex};
    use windows::{
        Win32::Media::Audio::{
            IActivateAudioInterfaceAsyncOperation, IActivateAudioInterfaceCompletionHandler,
            IActivateAudioInterfaceCompletionHandler_Impl,
        },
        core::{Ref, Result as WinResult, implement},
    };

    #[implement(IActivateAudioInterfaceCompletionHandler)]
    pub(super) struct Handler(Arc<(Mutex<bool>, Condvar)>);

    impl Handler {
        pub(super) fn new(state: Arc<(Mutex<bool>, Condvar)>) -> Self {
            Self(state)
        }
    }

    impl IActivateAudioInterfaceCompletionHandler_Impl for Handler_Impl {
        fn ActivateCompleted(
            &self,
            _activateoperation: Ref<'_, IActivateAudioInterfaceAsyncOperation>,
        ) -> WinResult<()> {
            let (lock, cvar) = &*self.0;
            if let Ok(mut done) = lock.lock() {
                *done = true;
            }
            cvar.notify_all();
            Ok(())
        }
    }
}

struct EventHandle(HANDLE);

impl EventHandle {
    fn new_auto_reset() -> Result<Self, tts::Failure> {
        // SAFETY: Null security attributes and null name create a local auto-reset event.
        unsafe { CreateEventW(None, false, false, PCWSTR::null()) }
            .map(Self)
            .map_err(|_| AUDIO_FAILURE)
    }

    fn wait(&self, timeout_ms: u32) -> bool {
        // SAFETY: `self.0` stays a valid kernel event handle until `EventHandle::drop`.
        unsafe { WaitForSingleObject(self.0, timeout_ms) == WAIT_OBJECT_0 }
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` is an exclusively owned handle created by `CreateEventW`.
        let _closed = unsafe { CloseHandle(self.0) };
    }
}

enum StreamError {
    Invalidated,
    Failed,
}

fn classify_wasapi_error(error: &windows::core::Error) -> StreamError {
    if error.code() == AUDCLNT_E_DEVICE_INVALIDATED {
        StreamError::Invalidated
    } else {
        StreamError::Failed
    }
}

struct RenderStaging {
    pending: VecDeque<Vec<u8>>,
    front_offset_bytes: usize,
    recycle: SyncSender<Vec<u8>>,
    max_packet_bytes: usize,
    packet_frames: u32,
}

impl RenderStaging {
    fn new(recycle: SyncSender<Vec<u8>>, sample_rate: u32) -> Result<Self, tts::Failure> {
        let rate = normalize_sample_rate(sample_rate);
        let max_packet_bytes = packet_bytes_for_rate(rate);
        let packet_frames = u32::try_from(max_packet_bytes / 2).map_err(|_| AUDIO_FAILURE)?;
        Ok(Self {
            pending: VecDeque::new(),
            front_offset_bytes: 0,
            recycle,
            max_packet_bytes,
            packet_frames,
        })
    }

    fn enqueue(&mut self, pcm: Vec<u8>) -> Result<(), tts::Failure> {
        if self.pending.len() >= PLAYBACK_PACKETS
            || pcm.is_empty()
            || pcm.len() > self.max_packet_bytes
            || !pcm.len().is_multiple_of(2)
        {
            return Err(AUDIO_FAILURE);
        }
        self.pending.push_back(pcm);
        Ok(())
    }

    fn pending_frames(&self) -> u32 {
        let mut total_bytes = 0usize;
        for (index, packet) in self.pending.iter().enumerate() {
            let offset = if index == 0 {
                self.front_offset_bytes.min(packet.len())
            } else {
                0
            };
            total_bytes = total_bytes.saturating_add(packet.len().saturating_sub(offset));
        }
        u32::try_from(total_bytes / 2).unwrap_or(u32::MAX)
    }

    fn queued_packets(&self, padding_frames: u32) -> usize {
        let total_frames = padding_frames.saturating_add(self.pending_frames());
        let packet_frames = self.packet_frames.max(1);
        usize::try_from(total_frames.div_ceil(packet_frames))
            .unwrap_or(usize::MAX)
            .max(self.pending.len())
    }

    fn write_into(&self, dst: &mut [u8]) -> Result<u32, tts::Failure> {
        let (slots, rem) = dst.as_chunks_mut::<4>();
        if !rem.is_empty() {
            return Err(AUDIO_FAILURE);
        }
        let mut written = 0u32;
        let mut slot_iter = slots.iter_mut();
        for (index, packet) in self.pending.iter().enumerate() {
            let offset = if index == 0 {
                self.front_offset_bytes.min(packet.len())
            } else {
                0
            };
            let slice = packet.get(offset..).ok_or(AUDIO_FAILURE)?;
            let (samples, trailing) = slice.as_chunks::<2>();
            if !trailing.is_empty() {
                return Err(AUDIO_FAILURE);
            }
            for sample in samples {
                let Some(slot) = slot_iter.next() else {
                    return Ok(written);
                };
                *slot = pcm16_to_f32(*sample).to_le_bytes();
                written = written.saturating_add(1);
            }
        }
        Ok(written)
    }

    fn consume_frames(&mut self, frames: u32) {
        let mut remaining_bytes = usize::try_from(frames)
            .unwrap_or(usize::MAX)
            .saturating_mul(2);
        while remaining_bytes > 0 {
            let Some(front) = self.pending.front() else {
                self.front_offset_bytes = 0;
                break;
            };
            let avail = front.len().saturating_sub(self.front_offset_bytes);
            if remaining_bytes < avail {
                self.front_offset_bytes = self.front_offset_bytes.saturating_add(remaining_bytes);
                break;
            }
            remaining_bytes = remaining_bytes.saturating_sub(avail);
            self.front_offset_bytes = 0;
            if let Some(packet) = self.pending.pop_front() {
                recycle(&self.recycle, packet);
            }
        }
    }
}

fn activate_default_render_client() -> Result<IAudioClient, tts::Failure> {
    let state = Arc::new((Mutex::new(false), Condvar::new()));
    let handler: IActivateAudioInterfaceCompletionHandler =
        activation::Handler::new(Arc::clone(&state)).into();
    // SAFETY: `DEVINTERFACE_AUDIO_RENDER_PATH` and `IAudioClient::IID` are static valid constants,
    // and `handler` is an agile COM callback object kept alive through the call.
    let operation = unsafe {
        ActivateAudioInterfaceAsync(
            DEVINTERFACE_AUDIO_RENDER_PATH,
            &IAudioClient::IID,
            None,
            &handler,
        )
    }
    .map_err(|_| AUDIO_FAILURE)?;
    let (lock, cvar) = &*state;
    let guard = lock.lock().map_err(|_| AUDIO_FAILURE)?;
    let (guard, timeout) = cvar
        .wait_timeout_while(guard, ACTIVATE_TIMEOUT, |done| !*done)
        .map_err(|_| AUDIO_FAILURE)?;
    if timeout.timed_out() && !*guard {
        return Err(AUDIO_FAILURE);
    }
    let mut activate_result = HRESULT(0);
    let mut unknown = None;
    // SAFETY: Activation completed; both out-parameters are initialized writable slots.
    unsafe { operation.GetActivateResult(&raw mut activate_result, &raw mut unknown) }
        .map_err(|_| AUDIO_FAILURE)?;
    activate_result.ok().map_err(|_| AUDIO_FAILURE)?;
    unknown
        .ok_or(AUDIO_FAILURE)?
        .cast()
        .map_err(|_| AUDIO_FAILURE)
}

// Drop `render` and `client` before closing `event` so WASAPI never observes a closed handle.
struct WasapiStream {
    render: IAudioRenderClient,
    client: IAudioClient,
    event: EventHandle,
    buffer_frames: u32,
}

impl WasapiStream {
    fn open(sample_rate: u32) -> Result<Self, tts::Failure> {
        // Declare `event` before `client` so reverse-order local drop on error
        // releases `IAudioClient` before closing `EventHandle`.
        let event = EventHandle::new_auto_reset()?;
        let client = activate_default_render_client()?;
        let format = float_wave_format(sample_rate);
        // SAFETY: `format` is a valid complete `WAVEFORMATEX` descriptor for shared-mode init.
        unsafe {
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                WASAPI_STREAM_FLAGS,
                WASAPI_BUFFER_HNS,
                0,
                &raw const format,
                None,
            )
        }
        .map_err(|_| AUDIO_FAILURE)?;
        // SAFETY: `client` is initialized.
        let buffer_frames = unsafe { client.GetBufferSize() }.map_err(|_| AUDIO_FAILURE)?;
        // SAFETY: `event.0` remains owned until after `client` and `render` drop.
        unsafe { client.SetEventHandle(event.0) }.map_err(|_| AUDIO_FAILURE)?;
        // SAFETY: `client` is initialized in render mode.
        let render: IAudioRenderClient =
            unsafe { client.GetService() }.map_err(|_| AUDIO_FAILURE)?;
        Ok(Self {
            render,
            client,
            event,
            buffer_frames,
        })
    }

    fn padding_frames(&self) -> Result<u32, StreamError> {
        // SAFETY: `self.client` is an initialized shared-mode `IAudioClient`.
        unsafe { self.client.GetCurrentPadding() }.map_err(|err| classify_wasapi_error(&err))
    }

    fn start(&self) -> Result<(), StreamError> {
        // SAFETY: `self.client` is an initialized shared-mode `IAudioClient`.
        unsafe { self.client.Start() }.map_err(|err| classify_wasapi_error(&err))
    }

    fn stop(&self) -> Result<(), StreamError> {
        // SAFETY: `self.client` is an initialized shared-mode `IAudioClient`.
        unsafe { self.client.Stop() }.map_err(|err| classify_wasapi_error(&err))
    }

    fn release_buffer(&self, frames: u32) -> Result<(), StreamError> {
        // SAFETY: `frames` matches samples written between `GetBuffer` and `ReleaseBuffer`.
        unsafe { self.render.ReleaseBuffer(frames, 0) }.map_err(|err| classify_wasapi_error(&err))
    }

    fn flush(&self, staging: &mut RenderStaging) -> Result<u32, StreamError> {
        let padding = self.padding_frames()?;
        let available = self.buffer_frames.saturating_sub(padding);
        let to_write = available.min(staging.pending_frames());
        if to_write == 0 {
            return Ok(padding);
        }
        let byte_len = usize::try_from(to_write)
            .map_err(|_| StreamError::Failed)?
            .checked_mul(size_of::<f32>())
            .ok_or(StreamError::Failed)?;
        // SAFETY: `to_write` is nonzero and within available WASAPI buffer frames.
        let ptr = unsafe { self.render.GetBuffer(to_write) }
            .map_err(|err| classify_wasapi_error(&err))?;
        if ptr.is_null() {
            let _released = self.release_buffer(0);
            return Err(StreamError::Failed);
        }
        // SAFETY: `GetBuffer(to_write)` returns `byte_len` writable bytes until `ReleaseBuffer`.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr, byte_len) };
        let Ok(written) = staging.write_into(dst) else {
            let _released = self.release_buffer(0);
            return Err(StreamError::Failed);
        };
        self.release_buffer(written)?;
        staging.consume_frames(written);
        Ok(padding.saturating_add(written))
    }
}

impl Drop for WasapiStream {
    fn drop(&mut self) {
        // SAFETY: Best-effort stop before releasing COM interfaces and closing the event handle.
        let _stopped = unsafe { self.client.Stop() };
        // SAFETY: Best-effort reset clears any queued unplayed frames on teardown.
        let _reset = unsafe { self.client.Reset() };
    }
}

// Keep `_apartment` last so `WasapiStream` drops all COM interfaces before `CoUninitialize`.
struct WasapiOutput {
    stream: Option<WasapiStream>,
    staging: RenderStaging,
    sample_rate: u32,
    started: bool,
    last_padding_frames: u32,
    _apartment: crate::com_dialog::Apartment,
}

impl WasapiOutput {
    fn open_with_rate(
        recycle: SyncSender<Vec<u8>>,
        sample_rate: u32,
    ) -> Result<Self, tts::Failure> {
        let rate = normalize_sample_rate(sample_rate);
        let apartment =
            crate::com_dialog::Apartment::com_mta_or_existing().map_err(|_| AUDIO_FAILURE)?;
        let stream = WasapiStream::open(rate)?;
        Ok(Self {
            stream: Some(stream),
            staging: RenderStaging::new(recycle, rate)?,
            sample_rate: rate,
            started: false,
            last_padding_frames: 0,
            _apartment: apartment,
        })
    }

    fn reopen_stream(&mut self) -> Result<(), tts::Failure> {
        self.stream = None;
        self.last_padding_frames = 0;
        let stream = WasapiStream::open(self.sample_rate)?;
        let padding = stream.flush(&mut self.staging).map_err(|_| AUDIO_FAILURE)?;
        self.last_padding_frames = padding;
        if self.started && stream.start().is_err() {
            return Err(AUDIO_FAILURE);
        }
        self.stream = Some(stream);
        Ok(())
    }

    fn run_with_reopen(
        &mut self,
        mut op: impl FnMut(&WasapiStream, &mut RenderStaging, &mut u32) -> Result<(), StreamError>,
    ) -> Result<(), tts::Failure> {
        let first = if let Some(stream) = &self.stream {
            op(stream, &mut self.staging, &mut self.last_padding_frames)
        } else {
            Err(StreamError::Invalidated)
        };
        match first {
            Ok(()) => Ok(()),
            Err(StreamError::Failed) => Err(AUDIO_FAILURE),
            Err(StreamError::Invalidated) => {
                self.reopen_stream()?;
                let stream = self.stream.as_ref().ok_or(AUDIO_FAILURE)?;
                match op(stream, &mut self.staging, &mut self.last_padding_frames) {
                    Ok(()) => Ok(()),
                    Err(StreamError::Invalidated | StreamError::Failed) => Err(AUDIO_FAILURE),
                }
            }
        }
    }
}

impl Output for WasapiOutput {
    fn queued(&self) -> usize {
        let padding = self
            .stream
            .as_ref()
            .and_then(|stream| stream.padding_frames().ok())
            .unwrap_or(self.last_padding_frames);
        self.staging.queued_packets(padding)
    }

    fn pause(&mut self) -> Result<(), tts::Failure> {
        self.started = false;
        self.run_with_reopen(|stream, _, last_padding| {
            stream.stop()?;
            *last_padding = stream.padding_frames()?;
            Ok(())
        })
    }

    fn start(&mut self) -> Result<(), tts::Failure> {
        self.started = true;
        self.run_with_reopen(|stream, staging, last_padding| {
            *last_padding = stream.flush(staging)?;
            stream.start()?;
            Ok(())
        })
    }

    fn push(&mut self, pcm: Vec<u8>) -> Result<(), tts::Failure> {
        self.staging.enqueue(pcm)?;
        self.run_with_reopen(|stream, staging, last_padding| {
            *last_padding = stream.flush(staging)?;
            Ok(())
        })
    }

    fn reap(&mut self) -> Result<(), tts::Failure> {
        self.run_with_reopen(|stream, staging, last_padding| {
            *last_padding = stream.flush(staging)?;
            Ok(())
        })
    }

    fn wait(&mut self, timeout: Duration) {
        if self.started
            && let Some(stream) = &self.stream
        {
            let timeout_ms = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
            let _signaled = stream.event.wait(timeout_ms);
        } else {
            thread::sleep(timeout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Arc, AtomicBool, AtomicU64, Audio, Buffered, Duration, Frames, Instant, MAX_BUFFER_BYTES,
        Ordering, Output, PACKET_BYTES, PACKET_BYTES_48K, PLAYBACK_PACKETS, PREFETCH_PACKETS,
        PREROLL_PACKETS, SAMPLE_RATE, SAMPLE_RATE_48K, SegmentFrames, VecDeque, check,
        consume_with, consume_with_progress, mpsc, packet_bytes_for_rate, send,
    };
    use crate::tts::Failure;

    fn render_smoothed_segment(pcm: &[u8], packet_bytes: usize) -> Result<Vec<u8>, Failure> {
        let mut frames = SegmentFrames::with_frames(Frames {
            packet_bytes,
            ..Frames::default()
        });
        let mut output = Vec::new();
        frames.push(pcm, |packet| {
            output.extend(packet);
            Ok(())
        })?;
        frames.finish_segment(|packet| {
            output.extend(packet);
            Ok(())
        })?;
        Ok(output)
    }

    #[test]
    fn prosody_fades_are_smooth_monotone_and_never_amplify_signed_pcm() -> Result<(), Failure> {
        for amplitude in [i16::MIN, -24_000, -1, 0, 1, 24_000, i16::MAX] {
            let pcm = amplitude.to_le_bytes().repeat(2_400);
            let output = render_smoothed_segment(&pcm, PACKET_BYTES)?;
            let samples: Vec<_> = output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|sample| i32::from(i16::from_le_bytes(*sample)))
                .collect();
            let magnitude = i32::from(amplitude).abs();
            assert!(samples.iter().all(|sample| sample.abs() <= magnitude));
            assert!(samples.iter().all(|sample| {
                *sample == 0 || sample.signum() == i32::from(amplitude).signum()
            }));
            let onset = &samples[..120];
            let offset = &samples[samples.len() - 120..];
            assert!(
                onset[..119]
                    .iter()
                    .zip(&onset[1..])
                    .all(|(before, after)| before.abs() <= after.abs())
            );
            assert!(
                offset[..119]
                    .iter()
                    .zip(&offset[1..])
                    .all(|(before, after)| before.abs() >= after.abs())
            );
            if magnitude >= 24_000 {
                // Independent broad Hann anchors reject linear ramps, hard
                // mutes and steps without duplicating the implementation formula.
                assert!((12 * magnitude..=17 * magnitude).contains(&(onset[29].abs() * 100)));
                assert!((48 * magnitude..=52 * magnitude).contains(&(onset[59].abs() * 100)));
                assert!((82 * magnitude..=87 * magnitude).contains(&(onset[89].abs() * 100)));
                let largest_step = onset[..119]
                    .iter()
                    .zip(&onset[1..])
                    .map(|(before, after)| (after - before).abs())
                    .max()
                    .unwrap_or_default();
                assert!(largest_step <= magnitude / 60 + 1);
            }
        }
        Ok(())
    }

    #[test]
    fn prosody_fade_window_tracks_five_milliseconds_at_each_sample_rate() -> Result<(), Failure> {
        // These sample counts come from time * sample rate, independently of
        // packet geometry or the production edge-window calculation.
        for (packet_bytes, fade_samples, samples) in
            [(PACKET_BYTES, 120, 2_400), (PACKET_BYTES_48K, 240, 4_800)]
        {
            let pcm = 24_000_i16.to_le_bytes().repeat(samples);
            let output = render_smoothed_segment(&pcm, packet_bytes)?;
            let decoded: Vec<_> = output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|sample| i16::from_le_bytes(*sample))
                .collect();
            assert_eq!(decoded.len(), samples);
            assert!(
                decoded[..fade_samples - 1]
                    .iter()
                    .all(|sample| *sample < 24_000)
            );
            assert!(
                decoded[samples - fade_samples + 1..]
                    .iter()
                    .all(|sample| *sample < 24_000)
            );
            assert!(
                decoded[fade_samples - 1..=samples - fade_samples]
                    .iter()
                    .all(|sample| *sample == 24_000)
            );
            assert_eq!(decoded[0], 0);
            assert_eq!(decoded[samples - 1], 0);
        }
        Ok(())
    }

    #[test]
    fn prosody_recycled_segments_reset_across_varied_lengths_and_empty_callbacks()
    -> Result<(), Failure> {
        for packet_bytes in [PACKET_BYTES, PACKET_BYTES_48K] {
            let (returned, recycled) = mpsc::sync_channel(super::RECYCLED_PACKETS);
            let mut frames = SegmentFrames::reusing_with_packet_bytes(recycled, packet_bytes);
            for (samples, amplitude) in [
                (1, i16::MIN),
                (packet_bytes / 2, i16::MAX),
                (17, 0),
                (packet_bytes + 13, -24_000),
                (121, 24_000),
            ] {
                let pcm = amplitude.to_le_bytes().repeat(samples);
                let expected = render_smoothed_segment(&pcm, packet_bytes)?;
                let mut actual = Vec::new();
                // Empty callbacks between bytes must leave split samples and
                // each segment's onset intact, including with recycled storage.
                for fragment in pcm.chunks(17) {
                    frames.push(&[], |_| Err(super::AUDIO_FAILURE))?;
                    frames.push(fragment, |packet| {
                        actual.extend_from_slice(&packet);
                        super::recycle(&returned, packet);
                        Ok(())
                    })?;
                }
                frames.finish_segment(|packet| {
                    actual.extend_from_slice(&packet);
                    super::recycle(&returned, packet);
                    Ok(())
                })?;
                assert_eq!(actual, expected);
                // A second flush must neither replay the tail nor influence
                // the next independent voice segment.
                frames.finish_segment(|_| Err(super::AUDIO_FAILURE))?;
            }
        }
        Ok(())
    }

    #[test]
    fn prosody_segment_output_refusal_propagates_during_streaming_and_final_flush() {
        let failure = Failure::Message("Synthetic output refusal");
        for packet_bytes in [PACKET_BYTES, PACKET_BYTES_48K] {
            let mut frames = SegmentFrames::with_frames(Frames {
                packet_bytes,
                ..Frames::default()
            });
            let mut calls = 0;
            assert_eq!(
                frames.push(&vec![0; packet_bytes * 2], |_| {
                    calls += 1;
                    Err(failure)
                }),
                Err(failure)
            );
            assert_eq!(calls, 1);

            let mut frames = SegmentFrames::with_frames(Frames {
                packet_bytes,
                ..Frames::default()
            });
            assert_eq!(frames.push(&[0; 24], |_| Err(failure)), Ok(()));
            calls = 0;
            assert_eq!(
                frames.finish_segment(|_| {
                    calls += 1;
                    Err(failure)
                }),
                Err(failure)
            );
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn segment_edges_reach_silence_at_both_rates_without_changing_duration() -> Result<(), Failure>
    {
        assert_eq!(packet_bytes_for_rate(SAMPLE_RATE), PACKET_BYTES);
        assert_eq!(packet_bytes_for_rate(SAMPLE_RATE_48K), PACKET_BYTES_48K);
        for packet_bytes in [PACKET_BYTES, PACKET_BYTES_48K] {
            let edge_bytes = packet_bytes / 20;
            for bytes in [
                2,
                4,
                edge_bytes - 2,
                edge_bytes,
                edge_bytes + 2,
                packet_bytes - 2,
                packet_bytes,
                packet_bytes + 2,
                packet_bytes * 2 + 42,
            ] {
                let mut frames = SegmentFrames {
                    frames: Frames {
                        pending: Vec::with_capacity(packet_bytes),
                        packet_bytes,
                        ..Frames::default()
                    },
                    ..SegmentFrames::default()
                };
                // A subsequent opposite-polarity segment must start from its own
                // silence edge, including when the previous ending was packet aligned.
                for amplitude in [12_000_i16, -12_000] {
                    let source = amplitude.to_le_bytes().repeat(bytes / 2);
                    let mut packets = Vec::new();
                    frames.push(&source, |packet| {
                        packets.push(packet);
                        Ok(())
                    })?;
                    assert!(frames.tail_len <= edge_bytes);
                    frames.finish_segment(|packet| {
                        packets.push(packet);
                        Ok(())
                    })?;
                    assert!(packets.iter().all(|packet| packet.len() <= packet_bytes));
                    assert!(packets.iter().all(|packet| packet.len().is_multiple_of(2)));
                    let output = packets.concat();
                    assert_eq!(output.len(), source.len());
                    assert_eq!(&output[..2], &[0, 0]);
                    assert_eq!(&output[output.len() - 2..], &[0, 0]);
                    if bytes >= edge_bytes * 2 {
                        assert_eq!(
                            &output[edge_bytes..bytes - edge_bytes],
                            &source[edge_bytes..bytes - edge_bytes]
                        );
                        assert_eq!(
                            &output[edge_bytes - 2..edge_bytes],
                            &amplitude.to_le_bytes()
                        );
                        assert_eq!(
                            &output[bytes - edge_bytes..bytes - edge_bytes + 2],
                            &amplitude.to_le_bytes()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn segment_onset_smoothing_is_identical_across_fragmented_pcm_callbacks() -> Result<(), Failure>
    {
        fn render(
            packet_bytes: usize,
            samples: usize,
            fragment: usize,
        ) -> Result<Vec<u8>, Failure> {
            let mut frames = SegmentFrames {
                frames: Frames {
                    packet_bytes,
                    ..Frames::default()
                },
                ..SegmentFrames::default()
            };
            let mut output = Vec::new();
            // The first utterance edge and both following transitions share the
            // same policy. Odd fragments exercise samples split across callbacks.
            for polarity in [1_i16, -1, 1] {
                let pcm: Vec<u8> = (0_i16..180)
                    .cycle()
                    .take(samples)
                    .flat_map(|index| (polarity * (12_000 - index * 100)).to_le_bytes())
                    .collect();
                for bytes in pcm.chunks(fragment) {
                    frames.push(bytes, |packet| {
                        output.extend(packet);
                        Ok(())
                    })?;
                }
                frames.finish_segment(|packet| {
                    output.extend(packet);
                    Ok(())
                })?;
            }
            Ok(output)
        }

        for packet_bytes in [PACKET_BYTES, PACKET_BYTES_48K] {
            for samples in [180, packet_bytes / 2, packet_bytes / 2 + 1] {
                let contiguous = render(packet_bytes, samples, usize::MAX)?;
                for fragment in [1, 2, 3, 17, 128, packet_bytes - 1, packet_bytes] {
                    assert_eq!(render(packet_bytes, samples, fragment)?, contiguous);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn segment_edges_hold_only_five_milliseconds_and_reject_a_split_final_sample()
    -> Result<(), Failure> {
        let mut frames = SegmentFrames::default();
        let mut output = Vec::new();
        frames.push(&vec![0; PACKET_BYTES + PACKET_BYTES / 20], |packet| {
            output.extend(packet);
            Ok(())
        })?;
        assert_eq!(output.len(), PACKET_BYTES);
        assert_eq!(frames.frames.pending, Vec::<u8>::new());
        assert_eq!(frames.tail_len, PACKET_BYTES / 20);
        frames.push(&[7], |_| Err(super::AUDIO_FAILURE))?;
        assert_eq!(frames.finish_segment(|_| Ok(())), Err(super::AUDIO_FAILURE));
        assert_eq!(output.len(), PACKET_BYTES);
        Ok(())
    }

    #[test]
    fn segment_edges_allow_realtime_codec_supply_to_prime_at_both_rates() -> Result<(), Failure> {
        let now = Instant::now();
        for packet_bytes in [PACKET_BYTES, PACKET_BYTES_48K] {
            let mut frames = SegmentFrames::with_frames(Frames {
                packet_bytes,
                ..Frames::default()
            });
            let mut buffered = Buffered::default();
            // Tail holdback can defer a full packet until the next 80ms codec
            // callback. It must still satisfy the existing supply jitter bound.
            let codec_frame = vec![0; packet_bytes * 4 / 5];
            for tick in 1..=10 {
                frames.push(&codec_frame, |packet| {
                    assert!(
                        buffered
                            .frame(packet, now + Duration::from_millis(tick * 80))?
                            .is_none()
                    );
                    Ok(())
                })?;
                assert!(!buffered.slow);
                if tick == 6 {
                    assert_eq!(buffered.ready.len(), PREROLL_PACKETS);
                }
            }
            frames.finish_segment(|packet| {
                assert!(
                    buffered
                        .frame(packet, now + Duration::from_millis(800))?
                        .is_none()
                );
                Ok(())
            })?;
            buffered.end()?;
            assert_eq!(buffered.ready.len(), 8);
        }
        Ok(())
    }

    #[test]
    fn streaming_segment_input_waits_through_gaps_then_finishes_or_cancels() {
        let (sender, receiver) = mpsc::sync_channel(2);
        let current = AtomicU64::new(4);
        let stop = AtomicBool::new(false);
        assert!(sender.send("First".to_owned()).is_ok());
        assert_eq!(
            super::next_segment(&receiver, &mut None, 4, &current, &stop),
            Ok(Some("First".to_owned()))
        );
        current.store(5, Ordering::Release);
        assert_eq!(
            super::next_segment(&receiver, &mut None, 4, &current, &stop),
            Err(Failure::Cancelled)
        );
        drop(sender);
        assert_eq!(
            super::next_segment(&receiver, &mut None, 5, &current, &stop),
            Ok(None)
        );
        let (sender, receiver) = mpsc::sync_channel(2);
        let waiting = std::thread::spawn(move || {
            super::next_segment(
                &receiver,
                &mut None,
                4,
                &AtomicU64::new(4),
                &AtomicBool::new(false),
            )
        });
        assert!(sender.send("Second".to_owned()).is_ok());
        assert_eq!(waiting.join().ok(), Some(Ok(Some("Second".to_owned()))));
    }

    #[test]
    fn bounded_prefetch_cancels_while_the_audio_queue_is_full() {
        let (sender, _incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
        assert_eq!(PREFETCH_PACKETS * PACKET_BYTES, 48_000 * 16);
        for _ in 0..PREFETCH_PACKETS {
            assert!(sender.try_send(Audio::Frame(vec![0; PACKET_BYTES])).is_ok());
        }
        let current = Arc::new(AtomicU64::new(4));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_current = Arc::clone(&current);
        let worker_stop = Arc::clone(&stop);
        let worker = std::thread::spawn(move || {
            send(
                &sender,
                Audio::Frame(vec![0; 2]),
                4,
                &worker_current,
                &worker_stop,
            )
        });
        current.store(5, Ordering::Release);
        assert_eq!(worker.join().ok(), Some(Err(Failure::Cancelled)));
        stop.store(true, Ordering::Release);
        assert_eq!(check(5, &current, &stop), Err(Failure::Cancelled));
    }

    #[test]
    fn queued_short_text_coalesces_without_reordering_or_exceeding_the_request_bound() {
        let (sender, receiver) = mpsc::sync_channel(5);
        assert!(sender.send("Hello".to_owned()).is_ok());
        assert!(sender.send("another clause.".to_owned()).is_ok());
        assert!(sender.send("Separate sentence.".to_owned()).is_ok());
        assert!(sender.send("x".repeat(super::super::MAX_SEGMENT)).is_ok());
        assert!(sender.send("Last.".to_owned()).is_ok());
        drop(sender);
        let mut pending = None;
        let current = AtomicU64::new(4);
        let stop = AtomicBool::new(false);
        for expected in [
            "Hello another clause. Separate sentence.".to_owned(),
            "x".repeat(super::super::MAX_SEGMENT),
            "Last.".to_owned(),
        ] {
            assert_eq!(
                super::next_segment(&receiver, &mut pending, 4, &current, &stop),
                Ok(Some(expected))
            );
        }
        assert_eq!(
            super::next_segment(&receiver, &mut pending, 4, &current, &stop),
            Ok(None)
        );
    }

    #[test]
    fn paragraph_cues_survive_both_coalescing_stages_at_the_character_limit()
    -> Result<(), &'static str> {
        let now = Instant::now();
        let input = format!("{}.\n\nSecond. tail", "a".repeat(246));
        let expected = format!("{}.\n\nSecond.", "a".repeat(246));
        assert_eq!(expected.chars().count(), super::super::MAX_SEGMENT);
        let mut text = super::super::stream::Text::default();
        text.append(&input, now)?;
        let mut queued = text.ready(now);
        assert_eq!(queued.len(), 2);

        // Request intake must keep the same paragraph and fit the original cap.
        let (sender, receiver) = mpsc::sync_channel(2);
        for piece in &queued {
            sender.send(piece.clone()).map_err(|_| "Closed queue")?;
        }
        assert_eq!(
            super::next_segment(
                &receiver,
                &mut None,
                4,
                &AtomicU64::new(4),
                &AtomicBool::new(false),
            ),
            Ok(Some(expected.clone()))
        );

        // The producer's earlier coalescing stage follows the same rule.
        let (_complete, finished) = mpsc::channel();
        let mut pipeline = super::Pipeline {
            segments: Some(sender),
            finished,
            cancelled: Arc::new(AtomicBool::new(false)),
            playing: Arc::new(AtomicBool::new(false)),
            notified_playing: false,
            worker: None,
        };
        pipeline.feed(&mut queued, true);
        assert_eq!(receiver.try_recv(), Ok(expected));
        assert!(queued.is_empty());
        Ok(())
    }

    #[test]
    fn prosody_coalescing_preserves_unicode_paragraph_prefix_after_overcap_deferral() {
        let first = format!("{}?!", "🙂".repeat(super::super::MAX_SEGMENT - 3));
        let (sender, receiver) = mpsc::sync_channel(3);
        assert!(sender.send(first.clone()).is_ok());
        assert!(sender.send("\n中".to_owned()).is_ok());
        assert!(sender.send("続き？".to_owned()).is_ok());
        drop(sender);
        let current = AtomicU64::new(4);
        let stop = AtomicBool::new(false);
        let mut pending = None;
        // The two-character continuation exceeds the scalar-character cap by
        // exactly one, even though the first piece occupies many more bytes.
        assert_eq!(
            super::next_segment(&receiver, &mut pending, 4, &current, &stop),
            Ok(Some(first))
        );
        assert_eq!(pending.as_deref(), Some("\n中"));
        assert_eq!(
            super::next_segment(&receiver, &mut pending, 4, &current, &stop),
            Ok(Some("\n中 続き？".to_owned()))
        );
        assert_eq!(pending, None);
        assert_eq!(
            super::next_segment(&receiver, &mut pending, 4, &current, &stop),
            Ok(None)
        );
    }

    #[test]
    fn prosody_paragraphs_survive_a_full_channel_and_later_refeed() {
        let (sender, receiver) = mpsc::sync_channel(2);
        assert!(sender.send("Earlier one.".to_owned()).is_ok());
        assert!(sender.send("Earlier two.".to_owned()).is_ok());
        let (_complete, finished) = mpsc::channel();
        let mut pipeline = super::Pipeline {
            segments: Some(sender),
            finished,
            cancelled: Arc::new(AtomicBool::new(false)),
            playing: Arc::new(AtomicBool::new(false)),
            notified_playing: false,
            worker: None,
        };
        let mut queued = VecDeque::from([
            "\r\n\r\n「First?!」".to_owned(),
            "\n\nSecond; still the same sentence.".to_owned(),
            "Third…".to_owned(),
        ]);
        let expected = "\r\n\r\n「First?!」\n\nSecond; still the same sentence. Third…";
        pipeline.feed(&mut queued, true);
        assert_eq!(queued, [expected.to_owned()]);
        assert!(pipeline.segments.is_some());
        assert_eq!(receiver.try_recv(), Ok("Earlier one.".to_owned()));
        assert_eq!(receiver.try_recv(), Ok("Earlier two.".to_owned()));
        pipeline.feed(&mut queued, true);
        assert_eq!(receiver.try_recv(), Ok(expected.to_owned()));
        assert_eq!(queued, VecDeque::<String>::new());
        assert!(pipeline.segments.is_none());
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn punctuation_batches_before_the_bounded_channel_and_preserves_backpressure() {
        let (sender, incoming) = mpsc::sync_channel(2);
        let (_complete, finished) = mpsc::channel();
        let mut pipeline = super::Pipeline {
            segments: Some(sender),
            finished,
            cancelled: Arc::new(AtomicBool::new(false)),
            playing: Arc::new(AtomicBool::new(false)),
            notified_playing: false,
            worker: None,
        };
        let sentence = "One short sentence.";
        let per_request = (super::super::MAX_SEGMENT + 1) / (sentence.len() + 1);
        let expected = vec![sentence; per_request].join(" ");
        let mut queued = VecDeque::from(vec![sentence.to_owned(); per_request * 4]);
        pipeline.feed(&mut queued, true);
        // Two full requests enter the channel; all remaining text is retained.
        assert_eq!(incoming.try_recv(), Ok(expected.clone()));
        assert_eq!(incoming.try_recv(), Ok(expected.clone()));
        assert!(incoming.try_recv().is_err());
        assert!(pipeline.segments.is_some());
        pipeline.feed(&mut queued, true);
        assert_eq!(incoming.try_recv(), Ok(expected.clone()));
        assert_eq!(incoming.try_recv(), Ok(expected));
        assert!(queued.is_empty());
        assert!(pipeline.segments.is_none());
        assert!(incoming.try_recv().is_err());
    }

    #[test]
    fn tiny_pcm_callbacks_form_fixed_frames_and_flush_the_final_tail_exactly() -> Result<(), Failure>
    {
        let source: Vec<_> = (0_u8..=255).cycle().take(PACKET_BYTES * 3 + 42).collect();
        let mut frames = Frames::default();
        let mut packets = Vec::new();
        for tiny in source.chunks(2) {
            frames.push(tiny, |packet| {
                packets.push(packet);
                Ok(())
            })?;
            assert!(frames.pending.len() < PACKET_BYTES);
        }
        frames.finish(|packet| {
            packets.push(packet);
            Ok(())
        })?;
        assert_eq!(
            packets.iter().map(Vec::len).collect::<Vec<_>>(),
            [PACKET_BYTES, PACKET_BYTES, PACKET_BYTES, 42]
        );
        assert_eq!(packets.concat(), source);
        Ok(())
    }

    #[test]
    fn pcm_packet_allocations_reuse_only_returned_playback_storage() -> Result<(), Failure> {
        let (recycle, recycled) = mpsc::sync_channel(super::RECYCLED_PACKETS);
        let mut frames = Frames::reusing(recycled);
        let first_pointer = frames.pending.as_ptr();
        let mut packets = VecDeque::new();
        frames.push(&vec![1; PACKET_BYTES], |packet| {
            packets.push_back(packet);
            Ok(())
        })?;
        assert_ne!(frames.pending.as_ptr(), first_pointer);
        let first = packets.pop_front().ok_or(super::AUDIO_FAILURE)?;
        assert_eq!(first.as_ptr(), first_pointer);
        assert_eq!(first, vec![1; PACKET_BYTES]);
        super::recycle(&recycle, first);

        frames.push(&vec![2; PACKET_BYTES], |packet| {
            packets.push_back(packet);
            Ok(())
        })?;
        assert_eq!(frames.pending.as_ptr(), first_pointer);
        assert_eq!(frames.pending, Vec::<u8>::new());
        assert_eq!(frames.pending.capacity(), PACKET_BYTES);
        assert_eq!(packets.front(), Some(&vec![2; PACKET_BYTES]));

        frames.push(&[3; 42], |_| Err(super::AUDIO_FAILURE))?;
        frames.finish(|packet| {
            assert_eq!(packet.as_ptr(), first_pointer);
            assert_eq!(packet, [3; 42]);
            Ok(())
        })?;
        assert_eq!(frames.pending, Vec::<u8>::new());
        frames.finish(|_| Err(super::AUDIO_FAILURE))?;
        frames.push(&[4; 64], |_| Err(super::AUDIO_FAILURE))?;
        frames.finish(|packet| {
            assert_eq!(packet, [4; 64]);
            Ok(())
        })?;
        frames.push(&[5], |_| Err(super::AUDIO_FAILURE))?;
        assert_eq!(frames.finish(|_| Ok(())), Err(super::AUDIO_FAILURE));
        Ok(())
    }

    #[test]
    fn pcm_recycling_has_a_fixed_capacity_and_never_waits_for_the_producer() {
        let (sender, receiver) = mpsc::sync_channel(super::RECYCLED_PACKETS);
        for _ in 0..super::RECYCLED_PACKETS * 3 {
            super::recycle(&sender, vec![7; PACKET_BYTES]);
        }
        let returned: Vec<_> = receiver.try_iter().collect();
        assert_eq!(returned.len(), super::RECYCLED_PACKETS);
        assert!(returned.iter().all(Vec::is_empty));
        assert_eq!(
            returned.iter().map(Vec::capacity).sum::<usize>(),
            PLAYBACK_PACKETS * PACKET_BYTES
        );
        // A larger external allocation cannot expand retained spare capacity.
        super::recycle(&sender, vec![0; PACKET_BYTES * 3]);
        assert!(receiver.try_recv().is_err());
        drop(receiver);
        super::recycle(&sender, vec![0; PACKET_BYTES]);
    }

    #[test]
    fn wasapi_float_format_and_render_route_match_windows_audio_contract() -> Result<(), Failure> {
        use windows::{Win32::Media::Audio::DEVINTERFACE_AUDIO_RENDER, core::GUID};

        assert_eq!(
            DEVINTERFACE_AUDIO_RENDER,
            GUID::from_u128(0xe632_7cad_dcec_4949_ae8a_991e_976a_79d2)
        );
        let expected_path: Vec<u16> = "{e6327cad-dcec-4949-ae8a-991e976a79d2}"
            .encode_utf16()
            .collect();
        // SAFETY: `DEVINTERFACE_AUDIO_RENDER_PATH` is a static null-terminated UTF-16 literal.
        let actual_path = unsafe { super::DEVINTERFACE_AUDIO_RENDER_PATH.as_wide() };
        assert_eq!(actual_path, expected_path);

        for (requested_rate, expected_rate) in [
            (SAMPLE_RATE, SAMPLE_RATE),
            (SAMPLE_RATE_48K, SAMPLE_RATE_48K),
            (16_000, SAMPLE_RATE),
        ] {
            let format = super::float_wave_format(requested_rate);
            assert_eq!({ format.wFormatTag }, super::WAVE_FORMAT_IEEE_FLOAT);
            assert_eq!({ format.nChannels }, 1);
            assert_eq!({ format.nSamplesPerSec }, expected_rate);
            assert_eq!({ format.nAvgBytesPerSec }, expected_rate * 4);
            assert_eq!({ format.nBlockAlign }, 4);
            assert_eq!({ format.wBitsPerSample }, 32);
            assert_eq!({ format.cbSize }, 0);
        }

        let (sender, _returned) = mpsc::sync_channel(PLAYBACK_PACKETS);
        let mut staging = super::RenderStaging::new(sender, SAMPLE_RATE)?;
        let mut pcm = Vec::new();
        for sample in [i16::MIN, 0, i16::MAX] {
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
        staging.enqueue(pcm)?;
        let mut dst = vec![0u8; 12];
        assert_eq!(staging.write_into(&mut dst)?, 3);
        let decoded: Vec<u32> = dst
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes).to_bits())
            .collect();
        assert_eq!(
            decoded,
            vec![
                (-1.0_f32).to_bits(),
                0.0_f32.to_bits(),
                (f32::from(i16::MAX) * (1.0 / 32_768.0)).to_bits(),
            ]
        );
        assert!(staging.write_into(&mut [0u8; 2]).is_err());
        Ok(())
    }

    #[test]
    fn wasapi_staging_recycles_rendered_packets_immediately_and_bounds_overflow()
    -> Result<(), Failure> {
        let (sender, returned) = mpsc::sync_channel(PLAYBACK_PACKETS);
        let mut staging = super::RenderStaging::new(sender, SAMPLE_RATE)?;
        for invalid in [Vec::new(), vec![0], vec![0; PACKET_BYTES + 2]] {
            assert!(staging.enqueue(invalid).is_err());
            assert!(staging.pending.is_empty());
        }

        let mut addresses = Vec::new();
        for index in 0..PLAYBACK_PACKETS {
            let mut packet = Vec::with_capacity(PACKET_BYTES);
            let value = i16::try_from(index + 1).map_err(|_| super::AUDIO_FAILURE)?;
            for _ in 0..(PACKET_BYTES / 2) {
                packet.extend_from_slice(&value.to_le_bytes());
            }
            addresses.push(packet.as_ptr());
            staging.enqueue(packet)?;
        }
        // Rejection cannot bypass the queue cap or drop already-staged PCM.
        assert!(staging.enqueue(vec![0; PACKET_BYTES]).is_err());
        assert_eq!(staging.pending.len(), PLAYBACK_PACKETS);
        assert_eq!(staging.queued_packets(0), PLAYBACK_PACKETS);

        // Partial write advances within the front packet without prematurely recycling it.
        let mut partial = vec![0u8; 2_000 * 4];
        assert_eq!(staging.write_into(&mut partial)?, 2_000);
        staging.consume_frames(2_000);
        assert_eq!(staging.pending.len(), PLAYBACK_PACKETS);
        assert_eq!(staging.front_offset_bytes, 4_000);
        assert!(returned.try_recv().is_err());

        // Finishing the front packet plus two full packets recycles all three allocations.
        let remaining_frames = 400 + 2 * 2_400;
        let mut batch = vec![0u8; remaining_frames * 4];
        assert_eq!(
            staging.write_into(&mut batch)?,
            u32::try_from(remaining_frames).map_err(|_| super::AUDIO_FAILURE)?
        );
        staging.consume_frames(u32::try_from(remaining_frames).map_err(|_| super::AUDIO_FAILURE)?);
        assert_eq!(staging.pending.len(), PLAYBACK_PACKETS - 3);
        assert_eq!(staging.front_offset_bytes, 0);
        let recycled_addrs: Vec<_> = returned
            .try_iter()
            .map(|pkt| (pkt.as_ptr(), pkt.capacity(), pkt.len()))
            .collect();
        assert_eq!(recycled_addrs.len(), 3);
        for (index, (addr, cap, len)) in recycled_addrs.into_iter().enumerate() {
            assert_eq!(addr, addresses[index]);
            assert_eq!(cap, PACKET_BYTES);
            assert_eq!(len, 0);
        }
        Ok(())
    }

    #[test]
    fn wasapi_ceiling_padding_retains_partial_tails_and_recovers_unwritten_frames_on_invalidation()
    -> Result<(), Failure> {
        let (sender, returned) = mpsc::sync_channel(PLAYBACK_PACKETS);
        let mut staging = super::RenderStaging::new(sender, SAMPLE_RATE)?;
        let packet_frames = staging.packet_frames;

        // Even a single unplayed tail frame keeps queued_packets >= 1 until padding hits 0.
        assert_eq!(staging.queued_packets(0), 0);
        assert_eq!(staging.queued_packets(1), 1);
        assert_eq!(staging.queued_packets(packet_frames), 1);
        assert_eq!(staging.queued_packets(packet_frames + 1), 2);

        // Enqueue 3 packets with distinct samples, write 1.5 packets cleanly, and simulate an
        // invalidated device failing ReleaseBuffer before consume_frames commits the next chunk.
        let mut expected_f32 = Vec::new();
        for packet_idx in 0..3 {
            let mut pkt = Vec::with_capacity(PACKET_BYTES);
            let sample = i16::try_from((packet_idx + 1) * 100).map_err(|_| super::AUDIO_FAILURE)?;
            for _ in 0..(PACKET_BYTES / 2) {
                pkt.extend_from_slice(&sample.to_le_bytes());
            }
            staging.enqueue(pkt)?;
        }

        let first_frames = 3_600usize; // 1.5 packets
        let mut first_buf = vec![0u8; first_frames * 4];
        assert_eq!(staging.write_into(&mut first_buf)?, 3_600);
        staging.consume_frames(3_600);
        expected_f32.extend(
            first_buf
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
        assert_eq!(returned.try_iter().count(), 1);

        // Simulate write_into filling a buffer followed by ReleaseBuffer returning
        // AUDCLNT_E_DEVICE_INVALIDATED before consume_frames is called: staging retains all 3,600 frames.
        let mut aborted_buf = vec![0u8; 3_600 * 4];
        assert_eq!(staging.write_into(&mut aborted_buf)?, 3_600);
        // Do NOT call consume_frames on failure; verify recovery onto a fresh stream gets every frame.
        assert_eq!(staging.pending_frames(), 3_600);
        assert_eq!(staging.queued_packets(0), 2);

        let mut recovered_buf = vec![0u8; 3_600 * 4];
        assert_eq!(staging.write_into(&mut recovered_buf)?, 3_600);
        staging.consume_frames(3_600);
        expected_f32.extend(
            recovered_buf
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
        assert_eq!(staging.pending_frames(), 0);
        assert_eq!(staging.queued_packets(0), 0);
        assert_eq!(returned.try_iter().count(), 2);
        assert_eq!(expected_f32.len(), 7_200);
        assert_eq!(
            expected_f32[0].to_bits(),
            (f32::from(100i16) * (1.0 / 32_768.0)).to_bits()
        );
        assert_eq!(
            expected_f32[2_400].to_bits(),
            (f32::from(200i16) * (1.0 / 32_768.0)).to_bits()
        );
        assert_eq!(
            expected_f32[4_800].to_bits(),
            (f32::from(300i16) * (1.0 / 32_768.0)).to_bits()
        );
        Ok(())
    }

    #[test]
    fn fast_supply_primes_after_cold_start_but_slow_supply_waits_for_validated_segment()
    -> Result<(), Failure> {
        let cold = Instant::now() + Duration::from_secs(90);
        let mut fast = Buffered::default();
        let mut slow = Buffered::default();
        for index in 0..PREROLL_PACKETS {
            let tick = u64::try_from(index).map_err(|_| super::AUDIO_FAILURE)?;
            assert!(
                fast.frame(
                    vec![1; PACKET_BYTES],
                    cold + Duration::from_millis(tick * 50)
                )?
                .is_none()
            );
            assert!(
                slow.frame(
                    vec![2; PACKET_BYTES],
                    cold + Duration::from_millis(tick * 800)
                )?
                .is_none()
            );
            if index + 1 < PREROLL_PACKETS {
                assert!(fast.ready.is_empty());
            }
            assert!(slow.ready.is_empty());
        }
        assert_eq!(fast.ready.len(), PREROLL_PACKETS);
        assert!(!fast.slow);
        assert!(slow.slow);
        slow.end()?;
        assert_eq!(slow.ready.len(), PREROLL_PACKETS);
        assert!(slow.flush);
        Ok(())
    }

    #[test]
    fn realtime_codec_frames_prime_with_jitter_and_reject_slow_supply() -> Result<(), Failure> {
        let now = Instant::now() + Duration::from_secs(90);
        let mut buffered = Buffered::default();
        let mut frames = Frames::default();
        for millis in [80, 160, 240, 360, 400, 480, 560, 640, 720, 800] {
            frames.push(&[0; 3_840], |frame| {
                assert!(
                    buffered
                        .frame(frame, now + Duration::from_millis(millis))?
                        .is_none()
                );
                Ok(())
            })?;
            if millis == 400 {
                assert_eq!(buffered.ready.len(), PREROLL_PACKETS);
            }
        }
        assert!(!buffered.slow);
        assert_eq!(buffered.ready.len(), 8);
        for interval in [120, 160] {
            let mut slow = Buffered::default();
            let mut frames = Frames::default();
            for index in 1..=5 {
                frames.push(&[0; 3_840], |frame| {
                    assert!(
                        slow.frame(frame, now + Duration::from_millis(index * interval))?
                            .is_none()
                    );
                    Ok(())
                })?;
            }
            assert!(slow.slow);
            assert!(slow.ready.is_empty());
            slow.end()?;
            assert_eq!(slow.ready.len(), PREROLL_PACKETS);
        }
        Ok(())
    }

    #[test]
    fn underrun_requires_complete_segments_and_staging_rejects_overlong_audio()
    -> Result<(), Failure> {
        let now = Instant::now();
        let mut buffered = Buffered::default();
        buffered.underrun();
        for _ in 0..MAX_BUFFER_BYTES / PACKET_BYTES {
            assert!(buffered.frame(vec![0; PACKET_BYTES], now)?.is_none());
        }
        assert!(buffered.ready.is_empty());
        assert_eq!(buffered.bytes, MAX_BUFFER_BYTES);
        assert!(buffered.frame(vec![0; 2], now).is_err());
        buffered.end()?;
        assert_eq!(buffered.frame(vec![0; 2], now)?, Some(vec![0; 2]));
        assert!(buffered.pop().is_some());
        assert!(buffered.frame(vec![0; 2], now)?.is_none());
        assert!(buffered.slow);
        Ok(())
    }

    #[derive(Default)]
    struct RecordingOutput {
        queued: VecDeque<Vec<u8>>,
        played: Vec<u8>,
        starts: Vec<usize>,
        paused: bool,
        maximum: usize,
        progress: Option<Arc<AtomicBool>>,
        playing_observations: usize,
    }

    impl Output for RecordingOutput {
        fn queued(&self) -> usize {
            self.queued.len()
        }

        fn push(&mut self, pcm: Vec<u8>) -> Result<(), Failure> {
            self.queued.push_back(pcm);
            self.maximum = self.maximum.max(self.queued.len());
            Ok(())
        }

        fn reap(&mut self) -> Result<(), Failure> {
            if !self.paused
                && let Some(pcm) = self.queued.pop_front()
            {
                if let Some(progress) = &self.progress {
                    assert!(progress.load(Ordering::Acquire));
                    self.playing_observations += 1;
                }
                self.played.extend(pcm);
            }
            Ok(())
        }

        fn pause(&mut self) -> Result<(), Failure> {
            self.paused = true;
            Ok(())
        }

        fn start(&mut self) -> Result<(), Failure> {
            if let Some(progress) = &self.progress {
                assert!(!progress.load(Ordering::Acquire));
            }
            self.starts.push(self.queued.len());
            self.paused = false;
            Ok(())
        }
    }

    #[test]
    fn queued_producer_audio_refills_before_a_late_wake_can_latch_an_underrun()
    -> Result<(), Failure> {
        struct DelayedOutput {
            output: RecordingOutput,
            producer: Option<mpsc::SyncSender<Audio>>,
        }

        impl Output for DelayedOutput {
            fn queued(&self) -> usize {
                self.output.queued()
            }

            fn push(&mut self, pcm: Vec<u8>) -> Result<(), Failure> {
                self.output.push(pcm)
            }

            fn reap(&mut self) -> Result<(), Failure> {
                if !self.output.paused
                    && let Some(producer) = self.producer.take()
                {
                    // Simulate a late consumer wake after the native queue has
                    // drained, with subsequent audio already in the channel.
                    while let Some(pcm) = self.output.queued.pop_front() {
                        self.output.played.extend(pcm);
                    }
                    for _ in 0..PREROLL_PACKETS {
                        producer
                            .send(Audio::Frame(vec![9; PACKET_BYTES]))
                            .map_err(|_| super::AUDIO_FAILURE)?;
                    }
                    producer
                        .send(Audio::SegmentEnd)
                        .map_err(|_| super::AUDIO_FAILURE)?;
                    Ok(())
                } else {
                    self.output.reap()
                }
            }

            fn pause(&mut self) -> Result<(), Failure> {
                self.output.pause()
            }

            fn start(&mut self) -> Result<(), Failure> {
                self.output.start()
            }
        }

        let (sender, incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
        for _ in 0..PREROLL_PACKETS {
            sender
                .send(Audio::Frame(vec![7; PACKET_BYTES]))
                .map_err(|_| super::AUDIO_FAILURE)?;
        }
        let mut output = DelayedOutput {
            output: RecordingOutput::default(),
            producer: Some(sender),
        };
        consume_with(
            &incoming,
            4,
            &AtomicU64::new(4),
            &AtomicBool::new(false),
            &mut output,
        )?;
        assert_eq!(output.output.starts, vec![PREROLL_PACKETS]);
        let mut expected = vec![7; PACKET_BYTES * PREROLL_PACKETS];
        expected.extend(vec![9; PACKET_BYTES * PREROLL_PACKETS]);
        assert_eq!(output.output.played, expected);
        assert!(output.output.maximum <= PLAYBACK_PACKETS);
        Ok(())
    }

    #[test]
    fn playback_progress_tracks_output_start_and_buffering_without_an_audio_device()
    -> Result<(), Failure> {
        let (sender, incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
        assert!(sender.send(Audio::Frame(vec![7; PACKET_BYTES])).is_ok());
        assert!(sender.send(Audio::SegmentEnd).is_ok());
        drop(sender);
        let progress = Arc::new(AtomicBool::new(true));
        let mut output = RecordingOutput {
            progress: Some(Arc::clone(&progress)),
            ..RecordingOutput::default()
        };
        consume_with_progress(
            &incoming,
            4,
            &AtomicU64::new(4),
            &AtomicBool::new(false),
            &mut output,
            &progress,
        )?;
        assert_eq!(output.playing_observations, 1);
        assert!(!progress.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn synthetic_output_primes_and_drains_short_final_segments_without_an_audio_device()
    -> Result<(), Failure> {
        for frames in [1, PREROLL_PACKETS + 5] {
            let (sender, incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
            for _ in 0..frames {
                assert!(sender.send(Audio::Frame(vec![7; PACKET_BYTES])).is_ok());
            }
            assert!(sender.send(Audio::SegmentEnd).is_ok());
            drop(sender);
            let mut output = RecordingOutput::default();
            consume_with(
                &incoming,
                4,
                &AtomicU64::new(4),
                &AtomicBool::new(false),
                &mut output,
            )?;
            assert_eq!(output.played, vec![7; PACKET_BYTES * frames]);
            assert_eq!(output.starts.first(), Some(&frames.min(PREROLL_PACKETS)));
            assert!(output.maximum <= PLAYBACK_PACKETS);
        }
        let (sender, incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
        for _ in 0..PREROLL_PACKETS {
            assert!(sender.send(Audio::Frame(vec![1; PACKET_BYTES])).is_ok());
        }
        assert!(sender.send(Audio::SegmentEnd).is_ok());
        for _ in 0..PREROLL_PACKETS {
            assert!(sender.send(Audio::Frame(vec![2; PACKET_BYTES])).is_ok());
        }
        assert!(sender.send(Audio::Frame(vec![2; PACKET_BYTES])).is_ok());
        assert!(sender.send(Audio::SegmentEnd).is_ok());
        drop(sender);
        let mut multi_segment_output = RecordingOutput::default();
        consume_with(
            &incoming,
            4,
            &AtomicU64::new(4),
            &AtomicBool::new(false),
            &mut multi_segment_output,
        )?;
        assert_eq!(multi_segment_output.starts, vec![PREROLL_PACKETS]);
        assert_eq!(
            multi_segment_output.played.len(),
            PACKET_BYTES * (PREROLL_PACKETS * 2 + 1)
        );
        let (sender, incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
        assert!(sender.send(Audio::Frame(vec![0; 2])).is_ok());
        drop(sender);
        assert!(
            consume_with(
                &incoming,
                4,
                &AtomicU64::new(4),
                &AtomicBool::new(false),
                &mut RecordingOutput::default()
            )
            .is_err()
        );
        assert_eq!(
            consume_with(
                &incoming,
                4,
                &AtomicU64::new(5),
                &AtomicBool::new(false),
                &mut RecordingOutput::default()
            ),
            Err(Failure::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn punctuation_boundaries_drain_48k_partial_tails_and_reset_false_underruns()
    -> Result<(), Failure> {
        // 1. A 48 kHz sentence's partial tail must drain without hanging paused;
        // the next queued sentence continues through its punctuation boundary.
        let (sender, incoming) = mpsc::sync_channel(PREFETCH_PACKETS);
        let mut frames_48k = SegmentFrames {
            frames: Frames {
                packet_bytes: PACKET_BYTES_48K,
                ..Frames::default()
            },
            ..SegmentFrames::default()
        };
        let sentence_one = vec![100_u8; PACKET_BYTES_48K * 4 + 1_920];
        frames_48k.push(&sentence_one, |packet| {
            sender
                .send(Audio::Frame(packet))
                .map_err(|_| super::AUDIO_FAILURE)
        })?;
        frames_48k.finish_segment(|packet| {
            sender
                .send(Audio::Frame(packet))
                .map_err(|_| super::AUDIO_FAILURE)
        })?;
        assert!(sender.send(Audio::SegmentEnd).is_ok());
        let sentence_two = vec![50_u8; PACKET_BYTES_48K * PREROLL_PACKETS];
        frames_48k.push(&sentence_two, |packet| {
            sender
                .send(Audio::Frame(packet))
                .map_err(|_| super::AUDIO_FAILURE)
        })?;
        frames_48k.finish_segment(|packet| {
            sender
                .send(Audio::Frame(packet))
                .map_err(|_| super::AUDIO_FAILURE)
        })?;
        assert!(sender.send(Audio::SegmentEnd).is_ok());
        drop(sender);

        let mut output_48k = RecordingOutput::default();
        consume_with(
            &incoming,
            1,
            &AtomicU64::new(1),
            &AtomicBool::new(false),
            &mut output_48k,
        )?;
        assert_eq!(
            output_48k.played.len(),
            sentence_one.len() + sentence_two.len()
        );
        assert_eq!(output_48k.starts, vec![PREROLL_PACKETS]);

        // 2. A boundary underrun right after all released packets of a sentence have
        // drained (before SegmentEnd is popped) must reset when SegmentEnd arrives so
        // the next sentence after punctuation still streams after PREROLL_PACKETS.
        let now = Instant::now();
        let mut buffered = Buffered::default();
        for _ in 0..PREROLL_PACKETS {
            assert!(buffered.frame(vec![1; PACKET_BYTES], now)?.is_none());
        }
        assert!(buffered.released);
        while buffered.pop().is_some() {}
        buffered.underrun();
        assert!(buffered.slow);
        buffered.end()?;
        assert!(!buffered.slow);
        for _ in 0..PREROLL_PACKETS {
            assert!(buffered.frame(vec![2; PACKET_BYTES], now)?.is_none());
        }
        assert!(buffered.released);
        assert_eq!(buffered.ready.len(), PREROLL_PACKETS);
        Ok(())
    }
}
