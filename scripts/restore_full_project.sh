#!/usr/bin/env bash
# Restore a GitHub Release backup into the current directory.
set -euo pipefail

archive_prefix="qwen_handwriting_alignment_full_20261008.tar"
manifest="SHA256SUMS"

if [[ ! -f "$manifest" ]]; then
  printf 'Missing %s in the current directory.\n' "$manifest" >&2
  exit 2
fi

shopt -s nullglob
parts=("${archive_prefix}".part-*)
if (( ${#parts[@]} == 0 )); then
  printf 'No archive parts matching %s.part-* found.\n' "$archive_prefix" >&2
  exit 2
fi

sha256sum -c "$manifest"
cat "${parts[@]}" | tar -xf -
printf 'Restored ./%s\n' "qwen_handwriting_alignment"
