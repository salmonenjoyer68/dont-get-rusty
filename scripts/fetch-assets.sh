#!/usr/bin/env bash
# Fetch and verify the pinned CPython Guest Runtime Image into assets/.
#
# The Runtime Image is the directory mounted read-only at `/` in the Guest:
#   assets/python.wasm
#   assets/lib/python3.15/...
#   assets/checksums.txt
#
# It is gitignored; this script is the only supported way to populate it.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$ROOT/assets"
CACHE="${DGR_ASSET_CACHE:-$ROOT/.agent/cache}"

ZIP_NAME="python-3.15.2-wasi_sdk-33.zip"
ZIP_URL="https://github.com/brettcannon/cpython-wasi-build/releases/download/v3.15.2/${ZIP_NAME}"
ZIP_SHA256="b8a63559c56760bbdc310dedb74fd69d10bcbafae26b705bea0ad353ad256888"
PYTHON_SHA256="dd076791fb6d1286a5dfe105852dfb3367e10562de68ca0a618e615625cdf636"
FIX="scripts/fetch-assets.sh"

fail() {
    printf 'fetch-assets: %s\n' "$*" >&2
    printf 'fetch-assets: fix it by running `%s` from the repository root\n' "$FIX" >&2
    exit 1
}

if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    fail "need sha256sum or shasum to verify the asset"
fi

write_checksums() {
    local target="$1"
    {
        printf '# Pinned by %s; do not edit.\n' "$FIX"
        printf '%s  %s\n' "$ZIP_SHA256" "$ZIP_NAME"
        printf '%s  python.wasm\n' "$PYTHON_SHA256"
    } >"$target"
}

if [[ -f "$ASSETS/python.wasm" && -d "$ASSETS/lib/python3.15" ]]; then
    if [[ "$(sha256 "$ASSETS/python.wasm")" == "$PYTHON_SHA256" ]]; then
        write_checksums "$ASSETS/checksums.txt"
        printf 'fetch-assets: runtime image already present and verified (%s)\n' "$ASSETS"
        exit 0
    fi
    printf 'fetch-assets: existing runtime image is corrupt; refetching\n' >&2
fi

mkdir -p "$CACHE"

ZIP="$CACHE/$ZIP_NAME"
if [[ ! -f "$ZIP" ]] || [[ "$(sha256 "$ZIP")" != "$ZIP_SHA256" ]]; then
    printf 'fetch-assets: downloading %s\n' "$ZIP_URL" >&2
    partial="$ZIP.part"
    rm -f "$partial"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --connect-timeout 30 -o "$partial" "$ZIP_URL" ||
            fail "download failed: $ZIP_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$partial" "$ZIP_URL" || fail "download failed: $ZIP_URL"
    else
        fail "need curl or wget to download the asset"
    fi
    mv "$partial" "$ZIP"
fi

actual_zip="$(sha256 "$ZIP")"
[[ "$actual_zip" == "$ZIP_SHA256" ]] ||
    fail "SHA-256 mismatch for $ZIP: got $actual_zip, expected $ZIP_SHA256"

stage="$(mktemp -d "${TMPDIR:-/tmp}/dgr-assets.XXXXXX")"
cleanup() { rm -rf "$stage"; }
trap cleanup EXIT

if command -v unzip >/dev/null 2>&1; then
    unzip -q "$ZIP" -d "$stage" || fail "failed to extract $ZIP"
elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$ZIP" "$stage" || fail "failed to extract $ZIP"
else
    fail "need unzip or python3 to extract $ZIP"
fi

[[ -f "$stage/python.wasm" ]] || fail "archive is missing python.wasm"
[[ -d "$stage/lib/python3.15" ]] || fail "archive is missing lib/python3.15/"

actual_python="$(sha256 "$stage/python.wasm")"
[[ "$actual_python" == "$PYTHON_SHA256" ]] ||
    fail "SHA-256 mismatch for extracted python.wasm: got $actual_python, expected $PYTHON_SHA256"

write_checksums "$stage/checksums.txt"

rm -rf "$ASSETS"
mv "$stage" "$ASSETS"
trap - EXIT

printf 'fetch-assets: runtime image ready at %s\n' "$ASSETS"
