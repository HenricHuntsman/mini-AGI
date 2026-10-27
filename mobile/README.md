# DragonForge Mobile

DragonForge is expanding beyond the Linux training host.

## iPhone — v0.1 in development

The native SwiftUI companion lives in `iphone/DragonForge/`.

Initial scope:
- connect to a local DragonForge/Greenlight Linux host
- training status, step, loss, expert count and VRAM
- Training Doctor health
- read-only companion API first
- secure pairing before any training-control endpoints

## Android — v0.1 companion scaffold

The native Android client is in [`android/`](../android/README.md) and uses the same `/api/v1` local protocol as iPhone.

Current v0.1 supports a configurable host address, read-only status, telemetry and Training Doctor diagnostics. Secure pairing, notifications and remote training controls are future work.

## Source and validation boundaries

The Android and iPhone folders are independent native client projects.
Updating their READMEs does not rebuild APKs, sign apps, or install anything
on a phone. The Linux host remains the source of truth for training and
checkpoint health. Review local network access and pairing before enabling
any remote control. See [the repository file inventory](../docs/FILE-INVENTORY.md)
for a complete tracked-file map.

## Linux — core platform

Linux remains the primary training, diagnostics, checkpoint, experiment and
local-model platform. Mobile work must not weaken Linux CI or change training
behavior merely to support a client.
