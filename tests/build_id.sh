#!/usr/bin/env bash
# The build ID script (cmake/azgo_build_info.cmake) in a scratch git
# repository: a clean checkout gets the bare commit even with untracked files
# outside the source directories; a changed tracked file, or an untracked
# file where the build globs its sources, gives "+hash", and editing that
# untracked file changes the hash. Skipped (77) without git.
#
# usage: tests/build_id.sh SOURCE_DIR
#
# Mark Castelluccio, with assistance from Claude (Anthropic).
set -euo pipefail

SRC="$(cd "${1:?usage: build_id.sh SOURCE_DIR}" && pwd)"
command -v git > /dev/null || { echo "build_id: SKIPPED (no git)"; exit 77; }
W="$(mktemp -d "${TMPDIR:-/tmp}/azgo_build_id.XXXXXX")"
trap 'rm -rf "$W"' EXIT

fail() { echo "build_id: FAILED: $*" >&2; exit 1; }
# id: the build ID the script computes for the scratch repository.
id() {
    cmake -DSRC="$W/repo" -DOUT="$W/out.inc" -DTEMPLATE="$SRC/cmake/build_info.hpp.in" -DOVERRIDE= \
          -P "$SRC/cmake/azgo_build_info.cmake"
    sed -n 's/^#define AZGO_BUILD_ID "\(.*\)"$/\1/p' "$W/out.inc"
}

R="$W/repo"
mkdir -p "$R/src" "$R/include"
echo 'int f() { return 1; }' > "$R/src/f.cpp"
echo 'cmake_minimum_required(VERSION 3.16)' > "$R/CMakeLists.txt"
git -C "$R" init -q
git -C "$R" -c user.name=t -c user.email=t@t add -A
git -C "$R" -c user.name=t -c user.email=t@t commit -q -m one
head="$(git -C "$R" rev-parse HEAD)"

[ "$(id)" = "$head" ] || fail "a clean checkout is not its commit: $(id)"

# Untracked files outside the source directories: still clean.
mkdir -p "$R/.superpowers" "$R/runs"
echo notes > "$R/.superpowers/notes.md"
echo run > "$R/runs/log.csv"
[ "$(id)" = "$head" ] || fail "untracked notes made the ID dirty: $(id)"

# An untracked source: dirty, and its contents count.
echo 'int g() { return 2; }' > "$R/src/g.cpp"
one="$(id)"
[[ "$one" =~ ^$head\+[0-9a-f]{8}$ ]] || fail "an untracked source left the ID clean: $one"
echo 'int g() { return 3; }' > "$R/src/g.cpp"
two="$(id)"
[[ "$two" =~ ^$head\+[0-9a-f]{8}$ ]] || fail "unexpected ID $two"
[ "$one" != "$two" ] || fail "editing an untracked source did not change the ID"
rm "$R/src/g.cpp"
[ "$(id)" = "$head" ] || fail "removing the untracked source did not restore the commit"

# A changed tracked file: dirty, and the change counts.
echo 'int f() { return 4; }' > "$R/src/f.cpp"
three="$(id)"
[[ "$three" =~ ^$head\+[0-9a-f]{8}$ ]] || fail "a changed tracked file left the ID clean: $three"
git -C "$R" checkout -q -- src/f.cpp
[ "$(id)" = "$head" ] || fail "restoring the tracked file did not restore the commit"

echo "build_id: all checks passed"
