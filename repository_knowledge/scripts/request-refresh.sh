#!/usr/bin/env bash
# %% optional deterministic refresh queue
set -euo pipefail

repository=$(git rev-parse --show-toplevel)
revision=$(git rev-parse --verify HEAD)
cache="${repository}/repository_knowledge/.cache"
mkdir -p -- "$cache"
pending=$(mktemp "${cache}/refresh-request.XXXXXX")
trap 'rm -f -- "$pending"' EXIT
printf '%s\n' "$revision" > "$pending"
mv -- "$pending" "${cache}/refresh-request"
