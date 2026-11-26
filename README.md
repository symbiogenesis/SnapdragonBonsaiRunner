# Snapdragon Bonsai Runner

*(Formerly **Swift Bonsai 2 Runner**; some internal paths, package names and
legacy documentation still use the earlier name.)*

A local creative-writing, brainstorming and chat app for Windows ARM64 laptops
with Snapdragon X Elite (X1) or Snapdragon X2 Elite Extreme (X2) processors. It
uses a native Rust UI and
[Prism Inference](https://github.com/symbiogenesis/prism-inference) on CPU, with
experimental Hexagon NPU acceleration on supported X2 installations.

Runner defaults to
[PrismML's Ternary Bonsai 2.5 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF)
(`PQ2_0`, 1.58-bit ternary weights, Apache-2.0), based on
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), replacing the retired
native Swift Bonsai 2 and Orca abliteration LoRA defaults. It also supports:

- Opt-in `PTQ1_0` Bonsai 2.5 weights for a smaller memory footprint, and the
  pinned `mmproj-Q8_0` multimodal projector for image and video attachments.
- An optional, stackable
  [DaisyRose-27B](https://huggingface.co/RadixArk/DaisyRose-27B) NPU-friendly
  multi-rank LoRA adapter that preserves Hexagon NPU FFN fusion when locally
  verified (clean installs run standalone Bonsai 2.5 until remote hosting is
  enabled or a local copy is provided).

Prompts and conversations stay on your device.

**Prerelease.** [Download a Windows ARM64 build](https://github.com/symbiogenesis/SnapdragonBonsaiRunner/releases)
or build a portable executable with the
[development guide](docs/development-guide.md#build-a-single-executable).
Clean-machine setup, actual-model behavior, quality, performance and isolation
still need [acceptance testing](PLAN.md).

## Features

- Separate conversations and drafts, local search, and encrypted workspace
  storage that can be disabled in Settings.
- Optional Thinking, Extra effort planning/review and multi-draft synthesis,
  cognitive/sampling controls, and adjustable DaisyRose multi-rank LoRA strength
  when configured.
- UTF-8 text/code file attachments and isolated image/video decoding for
  multimodal prompts.
- Local dictation, audio-file transcription, and read-aloud with a voice
  described in plain words. Microphone capture requires explicit permission.
- In-app model/runtime setup, progress, pause and retry. Text chat can work while
  speech finishes preparing.
- Verified portable updates with immediate or deferred installation.

Default `Ternary-Bonsai-2.5-27B-PQ2_0` weights are a 6.76 GiB download
(`PTQ1_0` is 5.58 GiB, and the multimodal projector `mmproj-Q8_0` is 600 MiB).
Startup selects context and its memory budget from available RAM, reserving
4 GiB for Windows and other apps. One reply runs at a time. Setup and startup
failures report errors without silently changing engines or model files.

## Documentation

- [Model selection and Bonsai 2.5 pins](docs/model-selection.md)
- [Application controls and contributor builds](docs/development-guide.md)
- [Portable distribution, setup and updates](docs/releases.md)
- [Technical contracts and tooling](docs/README.md)
- [Validation policy](docs/validation.md) and [outstanding work](PLAN.md)

## License

Original code, documentation and configuration use [Apache 2.0](LICENSE).
Third-party code retains its own terms; see [NOTICE](NOTICE).
Model weights and adapters are separate downloads with publisher licenses.
The [licensing guide](docs/licensing.md) covers redistribution and notices.
