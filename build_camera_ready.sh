#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for command in pdflatex pdfinfo; do
  command -v "$command" >/dev/null || { echo "Required command not found: $command" >&2; exit 1; }
done
export SOURCE_DATE_EPOCH=1789257600
export FORCE_SOURCE_DATE=1
export TZ=UTC
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
job=SafeDecontam_ICCC2026_camera_ready_5p
for pass in 1 2 3; do
  pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -file-line-error \
    -output-directory="$work" -jobname="$job" SafeDecontam_ICCC2026_revised.tex >"$work/pass-$pass.txt" 2>&1 \
    || { cat "$work/pass-$pass.txt" >&2; exit 1; }
done
if test -n "${BUILD_LOG_PATH:-}"; then cp "$work/$job.log" "$BUILD_LOG_PATH"; fi
pages=$(pdfinfo "$work/$job.pdf" | awk '/^Pages:/ {print $2}')
test "$pages" = 5 || { echo "Build rejected: expected exactly 5 pages, got $pages" >&2; exit 1; }
if grep -Eq 'Overfull|Missing character|undefined|LaTeX Error' "$work/$job.log"; then
  cat "$work/$job.log" >&2
  exit 1
fi
cp "$work/$job.pdf" "$job.pdf"
if test -f SafeDecontam_ICCC2026_camera_ready.pdf; then
  cp "$job.pdf" SafeDecontam_ICCC2026_camera_ready.pdf
fi
printf 'Verified: %s contains exactly 5 pages.\n' "$job.pdf"
