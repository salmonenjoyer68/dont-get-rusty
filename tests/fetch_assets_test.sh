#!/usr/bin/env bash
# Offline regression test for scripts/fetch-assets.sh manifest verification.
#
# Startup (src/assets.rs) rejects a checksums.txt that lists no image files, but
# the fetch script used to accept an empty or comments-only manifest and report
# the Runtime Image as verified, so the broken image was never refetched. This
# test drives the script's verification against temporary fake images; it never
# downloads the real Guest and does not need `assets/` to exist.
#
# Usage: bash tests/fetch_assets_test.sh
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FETCH_SCRIPT="$SCRIPT_DIR/../scripts/fetch-assets.sh"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/dgr-fetch-assets-test.XXXXXX")"
finished=0
failures=0

cleanup() {
    local exit_status=$?
    rm -rf "$TMP_ROOT"
    if [[ "$finished" -eq 0 ]]; then
        printf 'not ok - fetch-assets.sh did not return control when sourced (did it run the fetch?)\n' >&2
        exit_status=1
    fi
    exit "$exit_status"
}
trap cleanup EXIT

# The script must expose its verification functions when sourced, without
# running the fetch. The `main` guard in fetch-assets.sh is what makes this safe.
source "$FETCH_SCRIPT"
set +e

report_failure() {
    printf 'not ok - %s\n' "$*" >&2
    failures=$((failures + 1))
}

report_success() {
    printf 'ok - %s\n' "$*"
}

# A throwaway Runtime Image: the Guest plus one stdlib file.
new_image() {
    local root
    root="$(mktemp -d "$TMP_ROOT/image.XXXXXX")" || return 1
    mkdir -p "$root/lib/python3.15" || return 1
    printf 'not the real guest\n' >"$root/python.wasm"
    printf 'print(1)\n' >"$root/lib/python3.15/os.py"
    printf '%s\n' "$root"
}

# Prints a `<sha256>  <relative path>` entry for one file in the image.
write_entry() {
    local root="$1" relative="$2"
    printf '%s  %s\n' "$(sha256 "$root/$relative")" "$relative"
}

expect_reject() {
    local description="$1" root="$2"
    if verify_checksums "$root"; then
        report_failure "$description: manifest was accepted"
    else
        report_success "$description"
    fi
}

expect_accept() {
    local description="$1" root="$2"
    if verify_checksums "$root"; then
        report_success "$description"
    else
        report_failure "$description: manifest was rejected"
    fi
}

fake_curl() {
    local bin="$1"
    mkdir -p "$bin"
    cat >"$bin/curl" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$FAKE_CURL_LOG"
exit 22
EOF
    chmod +x "$bin/curl"
}

# Runs the fetch script's main flow with a fake image and a fake `curl`, so a
# refetch attempt is observable but never touches the network. The embedded
# python.wasm pin is overridden to the fake Guest's hash: the case under test is
# the manifest, not the pin.
run_main() {
    local image="$1" cache="$2" bin="$3" out="$4" log="$5"
    local rc=0
    (
        ASSETS="$image"
        CACHE="$cache"
        PYTHON_SHA256="$(sha256 "$image/python.wasm")"
        export FAKE_CURL_LOG="$log"
        PATH="$bin:$PATH"
        main
    ) >"$out" 2>&1 || rc=$?
    printf '%s\n' "$rc"
}

printf '# verify_checksums\n'

# The regression: an unusable manifest must be rejected.
root="$(new_image)"
: >"$root/checksums.txt"
expect_reject "empty checksums.txt is rejected" "$root"

root="$(new_image)"
printf '# Pinned by scripts/fetch-assets.sh; do not edit.\n# no entries yet\n' >"$root/checksums.txt"
expect_reject "comments-only checksums.txt is rejected" "$root"

root="$(new_image)"
write_entry "$root" lib/python3.15/os.py >"$root/checksums.txt"
expect_reject "manifest without a python.wasm entry is rejected" "$root"

root="$(new_image)"
rm -f "$root/checksums.txt"
expect_reject "missing checksums.txt is rejected" "$root"

# Startup also refuses absolute and `..` paths, so an image carrying them is
# unusable even though the hashes match.
root="$(new_image)"
{
    write_entry "$root" python.wasm
    printf '%s  %s\n' "$(sha256 "$root/lib/python3.15/os.py")" "/lib/python3.15/os.py"
} >"$root/checksums.txt"
expect_reject "manifest with an absolute path is rejected" "$root"

root="$(new_image)"
{
    write_entry "$root" python.wasm
    printf '%s  %s\n' \
        "$(sha256 "$root/lib/python3.15/os.py")" \
        "../$(basename "$root")/lib/python3.15/os.py"
} >"$root/checksums.txt"
expect_reject "manifest with a .. path is rejected" "$root"

# A CurDir component is not a Normal component: startup's parse_checksums
# rejects `.` everywhere, including a leading `./`.
root="$(new_image)"
{
    write_entry "$root" python.wasm
    printf '%s  %s\n' "$(sha256 "$root/lib/python3.15/os.py")" "./lib/python3.15/os.py"
} >"$root/checksums.txt"
expect_reject "manifest with a leading . path component is rejected" "$root"

root="$(new_image)"
{
    write_entry "$root" python.wasm
    printf '%s  %s\n' "$(sha256 "$root/lib/python3.15/os.py")" "lib/./python3.15/os.py"
} >"$root/checksums.txt"
expect_reject "manifest with an inner . path component is rejected" "$root"

# The behaviour that must keep working.
root="$(new_image)"
{
    write_entry "$root" python.wasm
    write_entry "$root" lib/python3.15/os.py
} >"$root/checksums.txt"
expect_accept "complete manifest is accepted" "$root"

root="$(new_image)"
{
    write_entry "$root" python.wasm
    write_entry "$root" lib/python3.15/os.py
} >"$root/checksums.txt"
printf 'tampered\n' >"$root/lib/python3.15/os.py"
expect_reject "manifest entry that no longer matches is rejected" "$root"

printf '# fetch script\n'

# An unusable manifest on an otherwise matching image must send the script down
# the refetch path instead of reporting the image as verified.
expect_refetch() {
    local description="$1" manifest="$2"
    local work image rc
    work="$(mktemp -d "$TMP_ROOT/refetch.XXXXXX")"
    image="$work/image"
    mkdir -p "$image/lib/python3.15" "$work/cache" "$work/bin"
    printf 'not the real guest\n' >"$image/python.wasm"
    printf '%s' "$manifest" >"$image/checksums.txt"
    fake_curl "$work/bin"
    rc="$(run_main "$image" "$work/cache" "$work/bin" "$work/out" "$work/curl.log")"
    if [[ "$rc" -eq 0 ]]; then
        report_failure "$description: script reported the image as verified"
    elif ! grep -q 'corrupt; refetching' "$work/out"; then
        report_failure "$description: script did not report the image as corrupt"
    elif [[ ! -s "$work/curl.log" ]]; then
        report_failure "$description: script did not attempt a refetch"
    else
        report_success "$description"
    fi
}

expect_refetch "empty manifest triggers a refetch" ''
expect_refetch "comments-only manifest triggers a refetch" \
    $'# Pinned by scripts/fetch-assets.sh; do not edit.\n'

# A valid manifest must still short-circuit without a download.
work="$(mktemp -d "$TMP_ROOT/verified.XXXXXX")"
image="$work/image"
mkdir -p "$image/lib/python3.15" "$work/cache" "$work/bin"
printf 'not the real guest\n' >"$image/python.wasm"
printf 'print(1)\n' >"$image/lib/python3.15/os.py"
{
    write_entry "$image" python.wasm
    write_entry "$image" lib/python3.15/os.py
} >"$image/checksums.txt"
fake_curl "$work/bin"
rc="$(run_main "$image" "$work/cache" "$work/bin" "$work/out" "$work/curl.log")"
if [[ "$rc" -ne 0 ]]; then
    report_failure "valid manifest: script exited $rc instead of accepting the image"
elif ! grep -q 'already present and verified' "$work/out"; then
    report_failure "valid manifest: script did not report the image as verified"
elif [[ -s "$work/curl.log" ]]; then
    report_failure "valid manifest: script attempted a refetch"
else
    report_success "valid manifest short-circuits without a refetch"
fi

finished=1
if [[ "$failures" -ne 0 ]]; then
    printf '\n%d check(s) failed\n' "$failures" >&2
    exit 1
fi
printf '\nall checks passed\n'
exit 0
