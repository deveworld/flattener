# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Delegated to the recommended TypeScript + Vite stack. The application must be deployable as static files.

## Product Purpose

Flattener turns document photographs into flat scans. The web application lets a person choose a photograph, inspect the correction, and save an image.

## Positioning

Open-source SOTA document scanning is the project's chosen positioning. Recorded benchmark results and their protocol are in README.md. Those results describe the Python evaluation pipeline; the browser implementation needs separate validation.

## Capabilities and Constraints

- The user explicitly selected in-browser ONNX/WebGPU inference. Photos must not be uploaded to a processing server.
- The current Python implementation includes dewarping, page detection, orientation, book splitting, and guarded rendering. Browser support must describe what is actually implemented.
- Models are available locally in a release bundle. Public distribution and hosting remain undecided.
- The web application must show progress, useful failures, and a downloadable result.
- GPU support varies by browser. A local WASM fallback is planned.

## Brand Commitments

The product name is Flattener. Existing repository documentation is in English; web interface copy will follow it. There is no established public web visual identity.

## Evidence on Hand

README.md and benchmarks/ contain model and evaluation evidence. The scanner source lives in src/flattener/scan/. No independent browser benchmark is available yet.

## Open Decisions

The primary audience beyond people scanning document photos, deployment destination, and final visual direction are not yet specified.
