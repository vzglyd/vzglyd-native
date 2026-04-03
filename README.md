# VZGLYD Native Host

Native (Linux/Raspberry Pi) host implementation for the VZGLYD display engine.

## Overview

This crate integrates the platform-agnostic `vzglyd-kernel` with:
- **winit** for windowing and event handling
- **wgpu** for GPU rendering
- **wasmtime** for WASM slide instantiation
- **std::fs** for asset loading

## Building

```bash
cargo build --release
```

## Running

```bash
# Run without slides (shows colored background)
cargo run

# Run with a shared slides repo
cargo run -- --slides-dir slides/

# Run with verbose output
cargo run -- --slides-dir slides/ -v
```

## Shared Slides Repo

The native host now expects `--slides-dir` to point at a shared slide repository root.
That repo must contain a required `playlist.json`, and each playlist entry path must be
repo-root-relative and point to a `.vzglyd` bundle.

### Repo Layout

```
slides/
├── playlist.json
├── clock.vzglyd
├── weather.vzglyd
└── daily/
    └── headlines.vzglyd
```

Missing or invalid `playlist.json` is treated as a startup error. Use `--scene <PATH>`
when you want to run one bundle directly without the shared repo contract.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Native Host                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ winit       │  │ wasmtime     │  │ std::fs          │   │
│  │ event loop  │  │ WASM loader  │  │ asset loading    │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ wgpu        │  │ NativeHost   │  │ RenderCommand    │   │
│  │ device/queue│  │ : Host       │  │ → wgpu execution │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │ implements Host trait
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  VZGLYD Kernel                              │
│  - Engine state machine                                     │
│  - Slide scheduling                                         │
│  - Transition logic                                         │
│  - RenderCommand generation                                 │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT OR Apache-2.0
