# Native Rust UI and local API contract

The first-party UI must remain **entirely native Rust**, including state, events,
rendering and network clients. Native OS APIs and audited Rust bindings are
allowed. The installed app owns first-run downloads, verification and repair in
its native window. PowerShell remains contributor build and launch tooling.

No JavaScript/TypeScript frontend, HTML/CSS UI, browser/WebView shell, Electron,
web-rendered Tauri frontend or inference-server browser UI is allowed, including
in a replacement toolkit. Interoperability belongs at the HTTP API boundary.

## Native application

`src/chat-ui/` provides the Windows `bonsai-chat` application using `egui`,
`eframe`, a native window and `wgpu`. State, widgets and HTTP clients are Rust.
The production manifest selects DX12, Vulkan and OpenGL/GLES through eframe's
`wgpu_no_default_features`, retaining upstream adapter selection and
`WGPU_BACKEND`. Source launches can explicitly override that variable for the chat
child with `--graphics-backend dx12|vulkan|gl`; omission preserves inheritance and
upstream selection. Browser/Metal backends are not requested; upstream test features
and installed drivers can affect backend availability. Toolkit portability does
not make Runner cross-platform: process ownership, credentials and speech remain
Windows-specific.

The native UI offers System, Light and Dark appearance and runs separately from
the broker/inference worker. Network, disk,
DPAPI and speech stay off the event loop, with bounded queues, cancellation and
joined shutdown. Windows high contrast, per-monitor scaling, labelled
AccessKit controls and keyboard navigation are supported. Full bidirectional
editing and native-control accessibility parity still require qualification.

Presentation primitives live in `window/controls.rs`: theme-aware buttons,
dialog surfaces and lazy clipboard controls. They accept borrowed presentation
values and return egui responses; they do not depend on application state,
services or persistence. Feature controls own their labels, availability and
stable widget identities. Conversation rows and dialogs return actions or
decisions for workspace controllers to apply after rendering. Settings value
editors borrow only the preferences they edit; the window owns persistence and
generation locks. The composer keeps text editing, cursor/focus restoration,
IME-safe submission, responsive actions and feedback in separate helpers.
Clipboard payloads are constructed only on activation, with caller-owned source
identities for transient confirmation.

### Launch and ownership

The native `bonsai-tools chat` command owns build reuse,
immutable tool snapshots and the per-user launch guard through teardown (replacing
a strictly older running instance when a newer version starts, while keeping any
same-version or newer instance).
The [development guide](development-guide.md#source-launch-caches-and-runtime-selection)
owns source-launch receipts/cache rules; the [package contract](chat-packages.md#launch-without-a-checkout)
owns verified installed launches.

The native Rust single-file launcher embeds five application executables, their
manifest and notices. It verifies and extracts the inventory into the per-user cache
and uses verified package launch. Broker, UI and speech retain separate ownership,
containment and teardown. The preparation worker provisions the compiled Swift
weights, selected adapter and runtime before broker handoff. Portable update
behavior is described [below](#application-updates).

[Runtime selection](development-guide.md#runtime-selection) owns engine,
representation, adapter, worker, cache and placement options. Retry preserves
explicit choices. The broker verifies exact runtime readiness and binds
[memory-selected context](swift-memory.md#memory-selected-serving-context)
to immutable cache/scratch charges, process limits and a 4 GiB host reserve.
Retry observes memory again; the UI uses broker-reported capacity for guidance.
These development defaults do not establish model, device or promotion acceptance.

The launcher checks/builds presentation and the headless broker image before
opening the window. The UI warms the Markdown parser before starting Swift or
speech preparation, without waiting for a response to request formatting.
Drafting and browsing remain available during its asynchronous handshake.
The first formatted viewport or terminal parser failure releases model setup;
offscreen responses do not delay it. Failure leaves source visible without a
parser retry. An owned preparation worker then
verifies/provisions the selected pinned Rust runtime and hands off text chat.
Speech artifact preparation, companion builds and fresh speech credentials continue separately
while Swift loads and serves text chat. A speech preparation failure leaves text
chat and its drafts/history intact; closing cancels both owned stages.
Each preparation process receives one ten-second cleanup allowance. A failed
explicit cleanup cannot restart that wait when its process owner is dropped.
Speech artifact preparation uses at most two concurrent workers, preserves exact
verification and partial-download ownership, and cancels peers on failure.
The complete speech preparation Job, including compiler descendants, runs at
below-normal CPU priority so it yields to model loading and text chat. This does
not reserve disk bandwidth or change inference priority. Full-file artifact
verification uses bounded 1 MiB sequential reads and retains complete SHA-256,
length, cancellation and immutable-file checks before use.
Default Swift preparation uses compiled immutable locks without a checkout. It
checks free disk space, reuses verified files and resumes partial downloads. Damaged
model files stay in place until a complete verified replacement is published;
runtime repair stages verified files and refuses unknown or redirected entries.
Explicit PTQ1 remains verify-only and never falls back to PQ2. Fixed bounded status
reports and partial lengths are read by the preparation worker, outside the event
loop. The compact setup card distinguishes a selected model found at attempt start
from first-time setup using current file availability. Explicit activity reports distinguish downloads/resume, saved-file and
download verification, and local runtime preparation. Byte progress appears only
during a reported download; legacy reports retain conservative preparation text.
Pause and Retry setup preserve drafts and launch options, use fresh attempt
directories and require confirmed owned-process cleanup; unexpected worker exits
disable retry. Speech has its own retry path and cannot block text chat. Drafts and
recovered conversations survive preparation;
model preparation failures leave the window available. Closing cancels preparation
and waits for owned descendants. Artifact verification and memory admission precede model use.
Owned inference and speech require [native containment](sandboxing.md), with no
uncontained fallback. Microphone consent is a separate UI capability check.

The client atomically places its broker in a separate kill-on-close Job before
the broker starts. Each attempt also has fresh ACL-protected private shutdown
storage. Stop and startup cancellation ask the broker to stop its independent
model Job and publish a fixed empty cleanup acknowledgment. The UI requires that
acknowledgment, broker exit and confirmed outer-Job cleanup before offering Retry;
a brokered PSEC model can lie outside the outer Job. All UI cleanup phases share
one fifteen-second absolute allowance, including the broker's ten-second model
stop and three seconds reserved for outer cleanup. Synchronous OS calls remain
nonpreemptible. Failed or timed-out teardown, or unknown model creation cleanup,
leaves the model unavailable and requires closing the app. This private file
control adds no HTTP endpoint and changes no model sandbox or resource admission.

The UI cannot load weights, choose arbitrary runtime arguments or bypass broker
artifact, model, memory, context or sampler policy. Explicit
[runtime/format and adapter choices](development-guide.md#runtime-selection) do not
establish promotion or full-model/device qualification.

### Long drafts and context guidance

Draft/edit transactions and serialized requests retain independent
[storage bounds](streaming-resources.md#native-chat-and-transcript): 1 MiB
text-only and 16 MiB when multimodal PNG `image_url` data URLs are attached
(up to 24 `image_url` parts across history and the current turn). Reject an
oversized edit as a whole, preserving text, selection and undo. Restore rejected
or interrupted prompts only to an empty composer in the originating conversation
(and restore staged draft attachments with rejected prompts when the originating
composer remains unedited).

Context guidance is advisory; exact [broker admission](broker-policy.md#text-chat-admission)
governs Send. It never trims history, changes reply limits or disables Send.
Capacity stays pending until authenticated startup reports it; Retry clears stale
capacity. Extra effort guidance covers its first pass only. A smaller reply limit
releases prompt capacity; only a fresh launch/Retry can change selected memory capacity.

### Application updates

Current-source portable builds check GitHub after startup and every 15 minutes and automatically
download the highest newer numeric version, including published prereleases.
The native Settings view shows the version, update status and one action for the
current state: check, download/retry or restart to update. It opens no additional
settings area. Discovery exhausts at most three pages of 100 releases; a full
final page fails explicitly because catalog completeness is unknown. Drafts are
excluded, and invalid newest assets never cause fallback. GitHub SHA-256
identities, the release manifest, checksum inventory
and tag/source revision must agree before download or use.

Metadata requests use the shared native Rust HTTPS client with Windows Schannel,
credential-free HTTPS URLs at every hop, at most three redirects, the GitHub JSON
media type, application User-Agent and an explicit API version. Read-only discovery
uses a fixed
six-slot disk cache binding each body to its exact URL and SHA-256. It prefers
`If-None-Match`, with `If-Modified-Since` as a fallback; a valid 304 reuses the bounded
body without a cache rewrite. This user-writable cache cannot authenticate release
publication: download, queue, launcher and replacement-helper gates request fresh
catalog, manifest, checksum and source metadata unconditionally. During discovery,
manifest and checksum bodies can be reused only when their length and SHA-256 match
the revalidated catalog. Requests advertise gzip and identity; native decoding
independently bounds the compressed and expanded bodies at 2 MiB. The HTTP client
does not decompress automatically, and no curl executable is required.
No authentication, conversation or model data accompanies
these requests.
Network and HTTP failures persist an exponential cooldown across helper processes,
honoring `Retry-After` and exhausted GitHub rate-limit reset instructions. Failed
requests never fall back to cached success or immediately retry. Offline or
no-internet update failures do not surface in the shared chat-area alert panel
and appear only as a small inline notice in the Settings update area.

After the download is verified, a nonmodal nudge offers **Restart to update** or
**Later** without interrupting typing or generation. A pending-update badge stays
on the bottom-left Settings button. Choosing restart from the nudge or Settings
opens one shared native confirmation popup; **Restart and update** authorizes
immediate installation, while **Cancel** leaves the app running without authorizing
installation. Confirmation never opens automatically after a background check or
download. Later dismisses the nudge for that verified release during the session
and defers installation until normal close. Downloads do not depend on acceptance.
Immediate installation closes and reopens Swift; closing with a deferred verified
portable update replaces the executable and leaves the app closed. Closing during
an unfinished download exits normally.

Replacing the executable requires the current verified release and successful
owned model, speech and UI teardown. Explicit immediate installation or a normal
close with the deferred update authorizes only that release. The launcher consumes
its launch-specific handoff after fresh publication/staging verification and starts
only the verified replacement
helper. This bounded helper runs from the staged new bundle, independently
revalidates publication and staged bytes, copies and verifies a sibling file, and
atomically replaces the original after its launcher exits. Changed originals,
network/verification failures or exhausted sharing retries retain the old file.
Immediate replacement reopens with the same launch options and user data directory;
deferred-close replacement skips restart. The handoff and helper explicitly carry
that choice. Restart
failure leaves the updated executable available for manual launch.
Models, runtime downloads, conversations and preferences stay separate.
Source and preview launches start no update discovery or installer work.

MSI operations are an optional distribution path. Settings offers download before
installation and uses the shared confirmation popup with **Close and install**.
MSI installation requires explicit confirmation, fresh publication/staged-byte
verification, successful teardown and the dedicated update exit. A leftover queue
or ordinary close cannot install.
Installed upgrade/rollback acceptance is independent of portable updates.

### Conversation and failure semantics

| Component | Contract |
| --- | --- |
| Workspace | Each conversation owns its draft, response settings, visible transcript and exact committed history. Creation/branching has no count cap; encoded recovery and per-conversation message bounds still apply. Transcript eviction never changes model history. |
| Presentation | Assistant answers render supported Markdown and standalone `` ```svg `` blocks with native Rust widgets; user messages stay literal. Generated links and external image labels remain inert, HTML stays literal and scripts or event attributes never execute. Bounded parsing, offline SVG rasterization, bounded progressive GPU texture caching, full-window Lightbox inspection, native PNG/SVG file export, and Find-hit source fallback keep Copy, speech fence stripping, recovery and model history tied to exact source. |
| Message iteration | Edit/Regenerate branch the verified history prefix, preserving the original draft. Branch creates without sending; Continue sends a followup turn preserving the draft. Identified interrupted answers can enter history; intermediate effort drafts cannot. |
| Response retry | When a reply reaches the selected output cap and the per-conversation **Auto-continue long responses** setting is enabled (default on), the client worker seamlessly continues up to 32 segments within context capacity (`768 KiB` stitched cap, minimum `64` continuation tokens), folding all streamed output into one continuous assistant message and one committed history pair without raising a model alert or disrupting the composer, cursor, or active read-aloud stream. In **Extra effort**, Draft, Review, Draft A, and Synthesis stages likewise auto-continue across token limits while preserving stage labels and initial Draft A confidence (Plan and speculative Draft B stay single-pass). Continuations are prefix-reuse-aware: when `PromptReuse` is enabled, they preserve an append-only prefix (keeping `enable_thinking`/`reasoning_effort` intact with `reasoning_budget = 1` when cognitive controls are active, folding any triggered memory appendix onto the original user turn, and compacting cleanly near the 127-message or serialized-request byte limit); when `PromptReuse` is disabled (or without cognitive controls), they coalesce prior segments into one stitched assistant turn with `enable_thinking = false`. The worker also rescues thinking-only truncation, preserves closing fences while stripping preambles/redundant opening fences/ellipses/overlaps, inserts safe SVG/Markdown block-boundary separators, guards control tokens, settles overlap-only natural endings as complete, clamps/steps down budgets with a sticky 413 ceiling, retries up to three 429 limits per segment, and stops cleanly on intra-segment degeneration, repetition-loop, byte, or context limits. If continuation hits a limit or auto-continue is disabled, visible **Continue response** and **Retry response** controls belong to the identified failed prompt, interrupted answer, or reply-length-capped answer. Continuing the latest incomplete answer sends an explicit followup turn in the current conversation while preserving the composer and cursor. Retrying an unanswered prompt reuses its existing transcript card and preserves the composer and cursor; retrying a partial or reply-capped answer creates a branch from its original prompt, preserving the source conversation and draft. After confirmed teardown, Reload model only restores readiness; a separate Continue response or Retry response action sends the prompt. |
| Recovery | User-scoped DPAPI, atomic bounded saves with a rolling encrypted backup (`chat-recovery.backup.dpapi`), unreadable-file quarantine (`chat-recovery.preserved.dpapi`), tolerant salvage/backup fallback, session-scoped Undo for the latest deleted conversation, one writer/data folder, and persistent opt-out that removes all encrypted copies. Restore never replays requests or starts capture/playback. Preserve unreadable bytes until an explicit save. See [recovery operation](development-guide.md#optional-recovery-and-preview). |
| Preferences | Window geometry, appearance/sidebar and encrypted voice profile persist independently of conversation saving. Writes coalesce off the event loop and join on close. Preview opens none of these stores; response choices remain per conversation. |
| Appearance | System/Light/Dark, with Windows high contrast taking precedence; 14–24-point reading text and 640–1,040-point reading width. Default/minimum window: 1,080×780 / 480×480 points. Sidebar: 216–340 points; below 900 points it becomes a modal drawer. Manual width survives layout changes. |
| Settings | Native modal with immediate changes and explicit conversation/device scope. Response opens first; narrow/short layouts wrap navigation. Done/Close/Escape restores composer focus. Open selectors consume Escape first; background Send/Find/Stop stay blocked while generation may continue. |
| Response controls | Per-conversation settings lock during generation. The composer popup exposes Thinking/Extra effort; advanced reply/adapter controls live in Settings. [Application controls](development-guide.md#conversations-and-response-settings) own defaults and actions. |
| Stop/retry | No generation elapsed-time cutoff. Response Stop freezes queued updates, wakes network/backpressure waits, and restores readiness without raising a model alert (resetting the warm native session in place via authenticated `/reset`, or automatically restarting a fresh warm instance after confirmed Job teardown on publisher rollback). Explicit model teardown (`StopModel`, such as before installing an update) parks cleanly until `Reload model` or shutdown. Retry/reload requires acknowledged teardown, fresh credentials/diagnostics and repeated artifact/identity/memory admission. Preserve completed turns/newer drafts; never replay or reduce the profile automatically. Completion winning the Stop race must leave transcript and history consistent. |
| Rejections | Only the exact [broker recovery errors](broker-policy.md#native-client-completion-and-recovery) retain readiness and restore input. Other disconnects require explicit Retry and identify no token/memory cause. |
| Alerts | One shared native panel selects among setup/model/saving/speech/audio/update failures. Each has a two-line summary, an `×` dismiss control and source-specific actions. Details provides a bounded scrollable modal; Copy error copies the complete message. Successful progress stays separate. Offline or no-internet update failures are excluded from this shared panel and appear only as a small inline notice inside the Settings update area. |

Markdown parsing and offline SVG rasterization use the same persistent [offline native sandbox worker](sandboxing.md#markdown-parser-worker),
with bounded background pipe I/O and trusted-parent validation (`BMD2` segmented
parse responses and `BSMG0001`/`BMG1` premultiplied RGBA8 raster packets). The
pure Rust `bonsai-markdown` and `resvg` libraries run in a separate private
`--markdown-worker` invocation of the configured headless broker image, before
model, credential or network setup. The GUI image cannot initialize under the
worker's required Win32k denial. The prepared broker path starts one parser
handshake before model setup, including for direct configured launches. No
synthetic source request is needed. Older source launchers without that handoff
retain deferred formatting until their configured broker is prepared.
The UI creates native egui text widgets and inline image cards from admitted
results; incomplete streamed `` ```svg `` fences show a bounded placeholder until
closed, while unrenderable SVGs fall back cleanly to source without scripts,
event attributes, external URLs or UI-thread XML parsing. Unavailable or failed
parsing leaves complete plain source visible and ends the session; parsing and
rasterization never fall back into the trusted production UI process. A bounded
progressive 24-texture GPU cache (`ImageCache`) sharpens previews on demand,
double-clicking or `Expand` opens the near-full-window Lightbox (~94%×92%
viewport, dismissed by `Escape`, `Close`, or scrim click), and `Save` writes
`.png` or `.svg` files atomically via the native `IFileSaveDialog`.

Markdown uses the current appearance theme and reading size: six heading sizes,
native bold faces for headings/emphasis/table headers, comfortable line leading
and compact paragraph gaps. Matched Windows Segoe UI regular/bold faces load once
before the window starts, with bundled fonts and system-script fallback. Nested
lists indent and quotes repeat their marker on authored lines. Inline code has
a compact background; code blocks use rounded full-width native surfaces behind
selectable text and Find highlights. Surface bounds share the galley's cache and
reflow with width, DPI and reading size. A fixed code-block style bit and closed
heading-size allowlist remain subject to trusted-parent validation; the worker
supplies no arbitrary fonts, coordinates or theme colors. Windows high contrast
continues to take precedence. These styles add no separately persisted theme.

The model states are unavailable, preparing, ready, generating, awaiting teardown,
stopping, stopping model, stopped, retryable and restarting. Send requires
acknowledged readiness; Retry/reload requires teardown notification and explicit
action (except automatic fresh warm restart after a response stop on the publisher
rollback runtime). Replace the ordered event channel on worker attachment. Late
events cannot reopen admission in another phase.

Drafting, browsing, conversation creation and message queuing remain available during
preparation, failure and generation. Only one request runs at a time: incoming
content, reasoning, stages and history belong to its submitting conversation,
which cannot be deleted until it settles. While generating, `Queue  ↑` or `Enter`
queues up to `8` messages (text and/or staged attachments) FIFO across
conversations without disturbing active drafts, cursors or unrelated read-aloud
playback. Stop and Escape-to-stop control generation only while the generating
owner is selected; other conversations show `Queue  ↑` (without Stop) and a
single background-response notice below the composer, with no duplicate workspace
banner or return button. Queued messages for the selected conversation appear in a
compact strip above the composer with `Edit` (when the composer is empty), `×`
remove, `Clear`, and (when ready) `Send next`, while inactive sidebar rows with
queued messages show a `N queued` badge. Normal completion (`Finished`) dispatches
the next queued message automatically; Stop, reply-cap completion (`TokenLimit`),
rejection, failure or update teardown pauses the queue without discarding items or
replaying them after recovery.
Its sidebar row shows an animated vertical activity indicator while generating,
including thinking and responses received while another conversation is selected;
selecting that row returns to the response.
Completion, Stop, rejection or failure removes the indicator.
Coalesce history changes and restore the selected history before the next Send,
without replay. Teardown/unacknowledged restart blocks navigation until turn events
settle. Late rejections restore only their still-existing originating conversation.

Switching conversations cancels speech/capture and invalidates late input;
hidden responses cannot restart playback or finish another conversation's readout.
Speech readiness is independent of text readiness and cannot queue playback while
preparing. Explicit readout retains on-demand loading/authenticated readiness.

Sidebar search and Find share Unicode scalar lowercase matching. Selecting a
sidebar result opens Find at the trimmed query, including title-only results with
no body match. Query changes update an open Find without taking focus, and clearing
the sidebar query closes Find across conversations; unchanged queries preserve local
Find edits, dismissal and navigation. Find edits/closure leave the sidebar filter
intact. Ctrl+F precedes text editors, while dialogs/popups retain keyboard priority.
[Streaming resources](streaming-resources.md#native-chat-and-transcript)
owns matching, layout invalidation and work bounds.

The Context ring shows approximate reserved occupancy, clamped full; pending
capacity uses a hollow ring/dash without stale values or animation. Its popup and
Thinking consume Escape/Enter before Find, Stop or background Send. The active
owner stays updating until its turn settles; browsing estimates the selected draft.
Guidance performs no network/disk work.

Message headers contain speaker, Copy, read-aloud and iteration actions. Cards use
16-point horizontal/12-point vertical insets with a 12-point gap; text wrapping and
Find geometry use those same insets. Thinking expands to measured text height.
Read-aloud/Stop stays keyboard accessible; unavailable speech reserves no footer.

Full model errors remain in session memory across conversation switches; recovery
saves only bounded status. Recent bounded diagnostics have a separate copy action.
Fatal launcher/graphics errors use a native scrollable/copyable dialog, falling back
to a short OS error message if dialog creation fails. Preview uses marked sample
content with no broker, credentials, microphone or speech service.

## Interoperability belongs at the API boundary

Use local LLM HTTP/JSON/SSE conventions: numeric loopback, bearer authorization,
`POST /v1/chat/completions`, text messages (or multimodal user content arrays
combining `text` and bounded PNG `image_url` `data:image/png;base64,...` parts
when the backend reports `Input modalities: text, image`) and `text/event-stream`. Preserve supplied
assistant reasoning separately as `reasoning_content`. [Broker policy](broker-policy.md#public-request-contract)
owns the supported subset, bounds, errors and native completion semantics.
Route names alone establish neither arbitrary-server nor OpenAI SDK compatibility;
clients emitting extra fields need an explicit adapter.

Protocol changes must preserve plain-text/streaming semantics and document
unsupported features. Keep backend policy outside presentation. Compatibility
cannot bypass ownership, admission, credentials or context limits.

## Speech input and output

Rust owns companion HTTP/authentication, schema validation, WAV processing and
verified model files. TTS weight loading, Talker/CodePredictor, 24 kHz codec and
speaker inference run in Rust in the owned speech executable. The ASR executable
retains its separate reviewed native C++ core. Application executables carry the
static CRT, so installed use requires no compiler or Visual C++ runtime installer.
The dependency-free `bonsai-audio` codec borrows
validated mono PCM16 WAV payloads and opens no devices; callers retain their own
sample-rate, duration and error policies.

Qwen3-ASR 1.7B and the Rust TTS engine use CPU. TTS does not load GGML backend
DLLs or dispatch the experimental Hexagon helper. Swift's resident `native-q2`
acceleration remains separate. The [TTS contract](../src/speech-tts/README.md)
records the current engine and pending parity, audio-quality and resource gates;
containment and resource admission still apply.

Microphone access is requested on the UI thread only after explicit action.
Denied, cancelled or unavailable access fails closed; stale permission, capture
or transcription results cannot reactivate it. Audio stays on authenticated
numeric loopback. The composer microphone records one bounded utterance; finishing
capture transcribes with Qwen3-ASR 1.7B and leaves the text in the draft for
explicit Send, with no automatic submission. Cancel, conversation changes and
teardown invalidate late results. Speech output uses only visible assistant
answers, excluding reasoning/private drafts.

ASR preparation, generation ownership and the successful-input idle lease follow
the [speech scheduling contract](streaming-resources.md#speech-scheduling-and-playback).
No audio upload or inference precedes complete finalized-WAV validation.

The composer's `+` button (`Add attachment`) opens the native `Attachment options`
popup with `Audio files`, `Text files` and `Images and Videos`. `Audio files` is
disabled with a spinner and `Loading ASR…` until Qwen3-ASR is ready (or
`ASR unavailable` when disabled or unconfigured). While Swift is loading or
responding, `Text files` and `Images and Videos` remain enabled and attach files
strictly to the active draft without auto-sending on import completion or model
readiness.
Up to `8` attachments fit on one message: UTF-8 text files (`256 KiB` per file,
`512 KiB` per message) format as fenced blocks on send; source media files
(`<= 64 MiB`) decode off the UI thread inside the private headless
`--attachment-import-worker` entry point in an offline AppContainer (WIC
normalization to `<= 1,280` px PNGs for images, and sequential Media Foundation
extraction of up to `6` evenly spaced `<= 768` px PNG keyframes for videos).
Staged attachments render as removable composer chips and committed transcript
cards, opening the `Attachment preview` lightbox (with `Previous frame` /
`Next frame` stepping for videos). `Send` remains disabled with guidance if
media attachments are present without vision (`mmproj`) capability.

When ASR is ready, choosing `Audio files` selects one audio file, and `Ctrl+V` in
the focused composer accepts audio files copied from Explorer. Selection stays in the trusted UI;
retained file/ancestor handles bind that authorization through decoder cleanup.
One offline MXC process normalizes MP3/WAV and Windows-supported M4A/AAC/FLAC/WMA
to in-memory mono 16 kHz
PCM16 WAV through one persistent asynchronous Media Foundation source reader.
The existing chat executable has a private headless decoder entry point, admitted
only inside an actual AppContainer. It adds no package executable or public API.
The parent grants only the selected input, executable/runtime and private scratch,
with no device, presentation or network capability. Binary pipe frames are bounded
and validated before ASR use; no complete recording is staged or base64 encoded.
Native initialization/admission failures stop import without decoding in the UI.
The reader advances sequentially, without reopening, seeking or recreating a
transcoder for each ASR request. File size and recording duration have no fixed
admission cap: available physical RAM and commit govern decoder buffers and
transcript growth. Each segment fits the recognizer's 30-second request contract;
an allocation-free PCM energy scan prefers a quiet 100 ms window in its final
five seconds. A pause needs RMS at most 256 PCM units and at most one quarter of
the loudest searched window's RMS. Without a qualifying pause, or when the final
window is equally quiet, the segment stays whole. The cut retains the exact PCM
suffix, and the next decode fills only the remaining request capacity. One bounded
decoder frame of lookahead identifies actual end of stream, including an exact
request boundary; duration metadata never truncates decoding. Final input flushes
every remaining sample. Progress ranges follow emitted samples. The heuristic neither
changes samples nor establishes transcription accuracy. Memory admission includes
the retained PCM allocation, bounded decoder frame and one additional WAV segment.
Recognition releases one decoder advance before each nonfinal request so decoding
can overlap inference. Backpressure retains at most the segment being recognized
and one following segment; recognition requests remain sequential. Callback waits
poll cancellation and share one deadline
per segment. Unsupported output-format changes, timeline gaps and decoder failures
stop import explicitly. Native source creation and release remain synchronous OS
calls; asynchronous sample waits do not make those calls preemptible.
Reader teardown in the helper waits at most two seconds for Flush and callback
quiescence. The parent requires successful helper exit and an empty Job before
accepting final audio, and stops the helper on cancellation or a segment deadline.
Its explicit cleanup wait is two seconds; the shared native owner's best-effort
Drop may wait a further ten seconds. Unconfirmed cleanup retains one bounded
quarantine of source/executable guards, scratch and pipe-thread ownership until
application exit, reports failure and closes file-import admission until restart.
Native Media Foundation/Win32k compatibility and measured boundary overhead remain
separate [acceptance gates](validation.md#boundary-overhead-acceptance).
File input needs no microphone permission. A native composer card shows Cancel,
persistent actionable failures and the current segment range against an estimated
total duration. Its completed-audio bar appears only when the duration estimate
supports it; this is no wall-time estimate. Imports use authenticated Qwen3-ASR
and insert at the draft selection/cursor, leaving the text in the draft for
review, editing and explicit Send with no automatic submission. Model readiness
or closing the transcript viewer never sends it later.
The centered native transcript modal has read-only selectable scrolling text,
copy feedback and background shortcut blocking. Escape closes the viewer.
Full/partial transcripts stay viewable and copyable until dismissed or a new
import's first segment is accepted; cancelling the picker or failing a new import
preserves the previous transcript. Oversized transcripts remain complete in that
view, preserve the draft and send no message.
File decoding adds no public route or model capability.

[Speech operation](development-guide.md#speech-setup-and-operation) defines
provisioning, capture lifecycle, fixed voice conditioning, mutually exclusive
talkers, budgets, service lifetime and external-service requirements (including
`consistent_voice: true` and complete-WAV fallback). The [ASR](../tools/canary-asr/README.md)
and [Qwen](../tools/qwen-tts/README.md) READMEs define companion protocols.
Service failure exposes setup guidance without substituting a recognizer or
Windows voice. Image and video keyframe context requires a vision-capable Swift
launch (`Input modalities: text, image` with verified `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`
on pinned native Rust `0.1.34` and `--publisher-engine`; `--no-mmproj` or
`--spec-type suffix` disables
the projector), while raster image and video
generation remain unavailable in the pinned Swift profile.

## Candidate native improvements

The [native application and writing references](research-register.md#native-application-and-writing-references)
identify candidate mechanisms, not implemented features or selected dependencies.
[PLAN.md](../PLAN.md) owns their implementation and qualification order. Source
reuse requires a reviewed immutable revision, license and compatibility with the
current egui version; native Rust presentation and broker policy remain binding.

For long transcripts, hello_egui's variable-height list is a reference for lazy row
measurement, bounded layout work and preserving the first visible item through
width, display-scale and font changes. Runner's focused Rust implementation works
with the pinned egui without adding a list dependency. It retains measured row
heights as estimates, prioritizes the exact queued Find target and visible rows,
and admits at most eight changed-message layout jobs per egui frame, shared across
additional passes. A 256 KiB text target admits one oversized existing bounded
message alone; whole selectable galleys mean this is not a hard per-frame glyph
or elapsed-time limit. Spare work measures offscreen rows without keeping an idle
window awake solely to complete them. Identity-based reading and visible active
Find anchors compensate estimated-height changes while user scrolling chooses a
new anchor. Exact Find navigation waits for current geometry. Native selection,
streaming, conversation switches and width/DPI/font changes retain independent
device acceptance under [validation](validation.md).

`draft_buffer.rs` implements a Ferrite-style byte-bounded inverse-operation
undo/redo log (`<= 1_048_576` retained bytes, `<= 64` groups, independent
transaction scratch capped at `4 * INPUT_LIMIT + 4_096` bytes, `max_undos: 0` on
`egui::TextEditState`, `TextFingerprint` invalidation, and single-group IME
preedit-to-commit batching) in place of full-draft snapshots while preserving
whole-edit overflow rejection, cursor and undo restoration, and atomic source
insertion. Text-engine update note: `cosmic-text` supplies
multilingual and bidirectional editing cases rather than a selected replacement
text engine. In `bonsai-chat` source (`window/branch_diff.rs`, `window/iteration.rs`,
and `window.rs`), bounded Myers comparison using `similar` (`diff_unicode_spans`,
off-thread cancellable `BranchDiffJob`, immutable `BranchDiffSource`, and
explicit `accept_comparison_replacement`) is implemented while retaining source
text and branch ownership without silently merging drafts; desktop interactive
UI qualification remains open.

Spindle's scene packets and SillyTavern's context cards motivate an optional native
writing-context design: user-approved character/world facts, scene relevance,
explicit insertion order and a visible token budget with inclusion/exclusion
reasons. Generated canon changes remain proposals until accepted. Local encrypted
storage and saving opt-out must cover any retained writing context. Exact broker
counting remains authoritative over the assembled request; budget overflow cannot
silently trim committed history. These references select neither an MCP tool
surface nor a web UI, and authorize no external manuscript transmission.

## Review and validation

Review native Rust presentation and protocol preservation through native builds,
focused UI/protocol checks and actual-model acceptance as appropriate. Full suites
and benchmarks stay CI-only under [validation](validation.md). Window launch and
synthetic work counts establish neither model quality, speech behavior, sustained
performance nor daily-service isolation. [PLAN.md](../PLAN.md) owns pending qualification.
