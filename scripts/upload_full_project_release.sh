#!/usr/bin/env bash
# Create/resume the full-project GitHub Release backup without storing secrets
# in the project. Run on the training machine after pushing the source commit.
set -euo pipefail

repo_owner="SHIBEI1"
repo_name="qwen_handwriting_alignment"
project_dir="/mnt/Disk1/kl/qwen_handwriting_alignment"
stage_dir="/mnt/Disk1/kl/qwen_handwriting_alignment_release_staging"
token_file="$HOME/.config/qwen_upload/github_token"
tag_name="full-project-2026-10-08"
release_name="Full project backup — 2026-10-08"
archive_prefix="qwen_handwriting_alignment_full_20261008.tar"
part_size="1800m"
api="https://api.github.com/repos/${repo_owner}/${repo_name}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -d "$project_dir" ]] || fail "Project directory not found: $project_dir"
[[ -s "$token_file" ]] || fail "GitHub token file is missing: $token_file"
mkdir -p "$stage_dir"
token="$(<"$token_file")"

api_get() {
  curl --fail --silent --show-error --retry 5 --retry-all-errors \
    --connect-timeout 20 -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.github+json' "$1"
}

api_json() {
  curl --fail --silent --show-error --retry 5 --retry-all-errors \
    --connect-timeout 20 -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.github+json' -H 'Content-Type: application/json' \
    -X "$1" "$2" --data "$3"
}

manifest="$stage_dir/SHA256SUMS"
mapfile -t existing_parts < <(find "$stage_dir" -maxdepth 1 -type f -name "${archive_prefix}.part-*" -printf '%f\n' | sort)
if [[ ! -f "$manifest" ]]; then
  if (( ${#existing_parts[@]} > 0 )); then
    fail "Found incomplete archive parts in $stage_dir. Inspect or remove that staging directory before restarting."
  fi
  printf 'Creating full archive in %s ...\n' "$stage_dir"
  tar --exclude='qwen_handwriting_alignment/.git' -C /mnt/Disk1/kl -cf - qwen_handwriting_alignment \
    | split --numeric-suffixes=0 --suffix-length=5 --bytes="$part_size" - "$stage_dir/${archive_prefix}.part-"
  (cd "$stage_dir" && sha256sum "${archive_prefix}".part-* > SHA256SUMS)
fi

mapfile -t parts < <(find "$stage_dir" -maxdepth 1 -type f -name "${archive_prefix}.part-*" -printf '%f\n' | sort)
(( ${#parts[@]} > 0 )) || fail "No archive parts were created."
(cd "$stage_dir" && sha256sum -c SHA256SUMS)

release_json="$stage_dir/release.json"
if ! api_get "$api/releases/tags/$tag_name" > "$release_json" 2>"$stage_dir/release_lookup.err"; then
  status="$(grep -o '[0-9][0-9][0-9]' "$stage_dir/release_lookup.err" | tail -1 || true)"
  if [[ "$status" == "404" ]]; then
    payload="$(python3 -c 'import json; print(json.dumps({"tag_name": "full-project-2026-10-08", "name": "Full project backup — 2026-10-08", "body": "Complete split archive of the training-machine project directory. Verify SHA256SUMS before restoration. See docs/BACKUP_RELEASE.md.", "draft": True, "prerelease": False}))')"
    api_json POST "$api/releases" "$payload" > "$release_json"
  else
    cat "$stage_dir/release_lookup.err" >&2
    fail "Unable to query or create release $tag_name"
  fi
fi

release_id="$(python3 -c 'import json; print(json.load(open("'$release_json'"))["id"])')"
upload_url="$(python3 -c 'import json; print(json.load(open("'$release_json'"))["upload_url"].split("{")[0])')"
api_get "$api/releases/$release_id/assets?per_page=100" > "$stage_dir/assets.json"

is_uploaded() {
  python3 - "$stage_dir/assets.json" "$1" "$2" <<'PY'
import json, sys
assets = json.load(open(sys.argv[1]))
name, size = sys.argv[2], int(sys.argv[3])
raise SystemExit(0 if any(a.get('name') == name and a.get('size') == size for a in assets) else 1)
PY
}

upload_asset() {
  local file="$1"
  local name size
  name="$(basename "$file")"
  size="$(stat -c '%s' "$file")"
  if is_uploaded "$name" "$size"; then
    printf 'Already verified on release: %s\n' "$name"
    return
  fi
  printf 'Uploading %s (%s bytes) ...\n' "$name" "$size"
  curl --fail --silent --show-error --retry 5 --retry-all-errors \
    --connect-timeout 20 -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.github+json' -H 'Content-Type: application/octet-stream' \
    --data-binary "@$file" "${upload_url}?name=${name}" > "$stage_dir/upload_${name}.json"
  api_get "$api/releases/$release_id/assets?per_page=100" > "$stage_dir/assets.json"
  is_uploaded "$name" "$size" || fail "Release asset verification failed: $name"
}

for part in "${parts[@]}"; do
  upload_asset "$stage_dir/$part"
done
upload_asset "$manifest"

payload='{"draft":false}'
api_json PATCH "$api/releases/$release_id" "$payload" > "$stage_dir/release_final.json"
printf 'Release published: %s\n' "$(python3 -c 'import json; print(json.load(open("'$stage_dir/release_final.json'"))["html_url"])')"
