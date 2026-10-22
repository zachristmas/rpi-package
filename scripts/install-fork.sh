#!/usr/bin/env bash
# Build and install the rpi-providers and rpi-herdr extensions from this checkout.
# Symbols stay unstripped: macOS 27 refuses the stripped dylib at load time
# ("mis-aligned LINKEDIT string pool"). Run again after every `git pull`.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_PROFILE_RELEASE_STRIP=none
rpi install --path packages/rpi-providers --force
rpi install --path packages/rpi-herdr --force
rpi --list-models zai
