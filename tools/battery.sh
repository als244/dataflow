#!/usr/bin/env bash
# shellcheck disable=SC2029  # client-side expansion into ssh commands is
# the design: $rpath_q is printf-%q-escaped precisely so the expanded
# word survives the remote shell intact.
#
# battery.sh — rsync-driven remote battery runner (maintainer rig).
#
# Usage:
#   tools/battery.sh <remote-host> <remote-path>
#
#   remote-host   ssh destination (e.g. tubingen)
#   remote-path   directory on the remote that holds the battery mirror
#                 (created if absent; contents are OWNED by this script —
#                 delete-mode rsync prunes anything not in the local tree)
#
# Environment:
#   BATTERY_PYTHON   python used on the remote (default: python3).
#                    Point it at the env that has pytest + pytest-timeout,
#                    e.g. BATTERY_PYTHON=~/miniconda3/envs/dataflow/bin/python
#
# What it does, in order:
#   1. rsync src tests tools examples reference_models conftest.py
#      pyproject.toml to <remote-host>:<remote-path> with --delete and
#      --delete-excluded, filtered through .gitignore (so the mirror is
#      exactly the git-visible tree — no stale __pycache__/*.pyc survive).
#   2. Computes a tree hash over every tracked+untracked .py file in the
#      synced set — locally via `git ls-files -c -o --exclude-standard`,
#      remotely via `find` over the mirror — and REFUSES to run if the
#      two differ. The stale-copy failure class (battery green against
#      yesterday's code because a sync silently failed or hit the wrong
#      path) has bitten three times; the hash gate makes it impossible.
#   3. Runs `pytest tests/ -q --timeout=1200` on the remote mirror.
#
# The hash covers .py files only: pyproject.toml is synced (pytest reads
# its config) but a config-only drift still reruns the same code, which
# is the failure class this gate exists for.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <remote-host> <remote-path>" >&2
    exit 2
fi

host="$1"
rpath="$2"
repo="$(cd "$(dirname "$0")/.." && pwd)"
remote_python="${BATTERY_PYTHON:-python3}"

sync_set=(src tests tools examples reference_models conftest.py pyproject.toml)

rpath_q="$(printf '%q' "$rpath")"

echo "[battery] rsync -> ${host}:${rpath}"
ssh "$host" "mkdir -p $rpath_q"
rsync -az --delete --delete-excluded \
    --filter=':- .gitignore' \
    --exclude='__pycache__/' --exclude='*.pyc' --exclude='.git' \
    "${sync_set[@]/#/$repo/}" "${host}:${rpath}/"

# Tree hash: per-file sha256 of every .py in the synced set, then a sha
# over the sorted list. LC_ALL=C on both sides so ordering is identical.
hash_local="$(cd "$repo" && \
    git ls-files --cached --others --exclude-standard -- "${sync_set[@]}" \
    | grep '\.py$' | LC_ALL=C sort \
    | xargs -d '\n' sha256sum | sha256sum | awk '{print $1}')"

hash_remote="$(ssh "$host" "cd $rpath_q && \
    find src tests tools examples reference_models conftest.py \
         -type f -name '*.py' -not -path '*/__pycache__/*' \
    | LC_ALL=C sort \
    | xargs -d '\n' sha256sum | sha256sum | awk '{print \$1}'")"

echo "[battery] tree hash local:  ${hash_local}"
echo "[battery] tree hash remote: ${hash_remote}"
if [ "$hash_local" != "$hash_remote" ]; then
    echo "[battery] REFUSING TO RUN: local and remote tree hashes differ." >&2
    echo "[battery] The remote mirror is not the code you think it is" >&2
    echo "[battery] (the stale-copy failure class). Check the rsync" >&2
    echo "[battery] output, the remote path, and locally-deleted but" >&2
    echo "[battery] still-tracked files, then rerun." >&2
    exit 1
fi

echo "[battery] hashes agree — running pytest on ${host}"
ssh "$host" "cd $rpath_q && PYTHONPATH=src $remote_python -m pytest tests/ -q --timeout=1200"
