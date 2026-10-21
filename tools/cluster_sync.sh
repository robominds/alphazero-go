#!/usr/bin/env bash
# cluster_sync.sh: brings every remote machine of the hosts file to this
# repository's commit, so distributed self-play's build-ID check passes.
#
# For each machine with ssh= in the hosts file, the repository on it is
# exe's grandparent directory and the build directory exe's parent
# (exe=<repo>/build-metal/selfplay). The commit is sent with
# `git archive HEAD | ssh ... tar -x` into <repo>.new, the old tree is
# swapped out while its build directory moves along (incremental
# rebuilds), and selfplay is built with -DAZGO_BUILD_ID=<commit> (the copy
# has no .git) and the options the build directory's name asks for:
# -DAZ_USE_METAL=ON when one of the name's words (split at - _ .) is "metal"
# (a Mac's GPU), -DAZ_USE_CUDA=ON when one is "cuda" (an NVIDIA GPU on
# Linux; it implies BLAS, so the binary serves machine/cuda and
# machine/blas), -DAZ_USE_BLAS=ON when one is "blas" (Accelerate, or
# OpenBLAS on Linux), none of them otherwise
# (set explicitly, so a kept CMake cache cannot hold an old choice). The
# name must start with "build", so it can never be one of the tree's own
# directories. Its --version must then equal this repository's commit. The
# remote may run macOS or Linux.
#
# A dirty tree is refused: a remote cannot rebuild uncommitted changes, and
# their build ID would differ anyway. <repo> is replaced wholesale, so it
# must be a dedicated copy (for example ~/azgo-cluster), never a checkout:
# the script refuses a directory it did not make itself (it marks its copies
# with a .azgo-sync file) and one holding .git or runs/.
#
# Mark Castelluccio, with assistance from Claude (Anthropic).
set -euo pipefail

usage() {
    cat <<'EOF'
usage: tools/cluster_sync.sh [--hosts FILE] [--build DIR] [--dry-run]

Syncs every machine with ssh= in the hosts file (default: $AZGO_HOSTS, else
~/.config/azgo/hosts) to this repository's HEAD and builds selfplay there in
<repo>/<build>, <repo> being exe's grandparent and <build> exe's parent
(--build names it for every machine instead); its name starts with
"build". One with the word "metal" in its name (words split at - _ .) is
configured with -DAZ_USE_METAL=ON, one with "cuda" with -DAZ_USE_CUDA=ON
(Linux; nvcc from the remote's PATH, else /usr/local/cuda/bin/nvcc), one
with "blas" with -DAZ_USE_BLAS=ON (OpenBLAS on Linux), others with none:
build-nometal gets none. Checks
that each machine's selfplay --version reports HEAD. --dry-run prints the
commands instead of running them. Refuses a tree with uncommitted changes.
<repo> is replaced wholesale: it must be a dedicated copy (e.g.
~/azgo-cluster), never a checkout. A directory without the .azgo-sync marker
this script leaves in its copies, or with .git or runs/, is refused.
EOF
}

HOSTS="" BUILD_DIR="" DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --hosts) HOSTS="${2:?--hosts needs a file}"; shift 2 ;;
        --build) BUILD_DIR="${2:?--build needs a directory name}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "$HOSTS" ] || HOSTS="${AZGO_HOSTS:-$HOME/.config/azgo/hosts}"
[ -f "$HOSTS" ] || { echo "error: no hosts file at $HOSTS" >&2; exit 1; }
SAFE='^[A-Za-z0-9_./~@:+-]+$'
BUILD_NAME='^build[A-Za-z0-9_.-]*$'
[ -z "$BUILD_DIR" ] || [[ "$BUILD_DIR" =~ $BUILD_NAME ]] || { echo "error: bad --build '$BUILD_DIR'" >&2; exit 2; }

# alias ssh exe, one machine per line, for the machines with ssh=.
machines=()
while read -r alias rest; do
    case "$alias" in ''|\#*) continue ;; esac
    ssh_target="" exe=""
    for pair in $rest; do
        case "$pair" in
            \#*) break ;;
            ssh=*) ssh_target="${pair#ssh=}" ;;
            exe=*) exe="${pair#exe=}" ;;
        esac
    done
    [ -n "$ssh_target" ] || continue
    for value in "$ssh_target" "$exe"; do
        [[ "$value" =~ $SAFE ]] && [[ "$value" != -* ]] || { echo "error: $alias: unsafe value '$value'" >&2; exit 1; }
    done
    machines+=("$alias $ssh_target $exe")
done < <(sed 's/#.*//' "$HOSTS")
if [ ${#machines[@]} -eq 0 ]; then
    echo "no remote machines (no ssh= line) in $HOSTS"
    exit 0
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [ -n "$(git status --porcelain --untracked-files=no)" ] ||
   [ -n "$(git ls-files --others --exclude-standard -- CMakeLists.txt cmake src include apps tests)" ]; then
    echo "error: the tree has uncommitted changes; commit them first (a remote builds HEAD)" >&2
    exit 1
fi
COMMIT="$(git rev-parse HEAD)"

run() {   # run CMD...: print it with --dry-run, else run it
    if [ "$DRY" = 1 ]; then printf '%q ' "$@"; echo; else "$@"; fi
}

# sync_machine ALIAS TARGET EXE: one machine; on a failure, says which
# step failed and returns 1 (the other machines are still synced).
sync_machine() {
    local alias=$1 target=$2 exe=$3
    local repo build options
    repo="$(dirname "$(dirname "$exe")")"
    case "$repo" in ''|.|/|'~'|'~/')
        echo "error: $alias: exe=$exe gives no usable repository directory" >&2; return 1 ;;
    esac
    build="${BUILD_DIR:-$(basename "$(dirname "$exe")")}"
    [[ "$build" =~ $BUILD_NAME ]] ||
        { echo "error: $alias: exe=$exe is not in a build directory (build, build-blas, build-metal, ...)" >&2; return 1; }
    # The name's words, not substrings: build-metal and build_metal_2 are
    # metal; build-nometal and build-metal2 are not. metal, then cuda, win
    # over blas (a Metal or CUDA build has BLAS too).
    local words word flavour=""
    IFS='-_.' read -r -a words <<<"$build"
    for word in "${words[@]}"; do
        case "$word" in
            metal) flavour=metal ;;
            cuda) [ "$flavour" = metal ] || flavour=cuda ;;
            blas) [ -n "$flavour" ] || flavour=blas ;;
        esac
    done
    # Every option is set explicitly, so a kept CMake cache cannot hold an
    # old choice. A non-interactive ssh shell often lacks /usr/local/cuda/bin
    # on its PATH: nvcc is looked for there too (CMake keeps it after the
    # first configure).
    case "$flavour" in
        metal) options="-DAZ_USE_METAL=ON -DAZ_USE_CUDA=OFF" ;;
        cuda) options="-DAZ_USE_METAL=OFF -DAZ_USE_CUDA=ON -DCMAKE_CUDA_COMPILER=\$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)" ;;
        blas) options="-DAZ_USE_METAL=OFF -DAZ_USE_CUDA=OFF -DAZ_USE_BLAS=ON" ;;
        *) options="-DAZ_USE_METAL=OFF -DAZ_USE_CUDA=OFF -DAZ_USE_BLAS=OFF" ;;
    esac
    echo "== $alias: $COMMIT -> $repo ($build)"
    local ssh_=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$target")
    # The repository is replaced wholesale, so it must be a copy this script
    # made (it carries .azgo-sync), never a checkout or a directory with runs.
    run "${ssh_[@]}" "if [ -e $repo ] && [ ! -f $repo/.azgo-sync ]; then
            echo '$repo exists but was not made by cluster_sync.sh (no .azgo-sync); refusing to replace it: point exe at a dedicated copy such as ~/azgo-cluster/$build/selfplay' >&2; exit 3; fi
        if [ -e $repo/.git ] || [ -e $repo/runs ]; then
            echo '$repo holds .git or runs/; refusing to replace it' >&2; exit 3; fi" ||
        { echo "error: $alias: $repo is not a copy cluster_sync.sh may replace" >&2; return 1; }
    run "${ssh_[@]}" "rm -rf $repo.new && mkdir -p $repo.new && touch $repo.new/.azgo-sync" ||
        { echo "error: $alias: cannot create $repo.new" >&2; return 1; }
    if [ "$DRY" = 1 ]; then
        echo "git archive $COMMIT | ${ssh_[*]} 'tar -x -m -C $repo.new'"
    else
        # -m: files get the time of extraction, not of their commit, so make
        # never takes an object built earlier for newer than a changed source
        # (every sync then rebuilds what the moved build directory holds).
        git archive "$COMMIT" | "${ssh_[@]}" "tar -x -m -C $repo.new" ||
            { echo "error: $alias: copying the tree failed" >&2; return 1; }
    fi
    # Swap the trees; the build directory (CMake cache and objects) moves along.
    run "${ssh_[@]}" "if [ -d $repo/$build ]; then mv $repo/$build $repo.new/; fi &&
        rm -rf $repo.old && if [ -d $repo ]; then mv $repo $repo.old; fi && mv $repo.new $repo && rm -rf $repo.old" ||
        { echo "error: $alias: swapping in the new tree failed" >&2; return 1; }
    # getconf: the core count on macOS and Linux alike.
    run "${ssh_[@]}" "cmake -S $repo -B $repo/$build -DCMAKE_BUILD_TYPE=Release $options \
        -DAZGO_BUILD_ID=$COMMIT > /dev/null && cmake --build $repo/$build -j \$(getconf _NPROCESSORS_ONLN) --target selfplay" ||
        { echo "error: $alias: building selfplay failed" >&2; return 1; }
    if [ "$DRY" = 1 ]; then
        run "${ssh_[@]}" "$exe --version"
        return 0
    fi
    local remote_id
    remote_id="$("${ssh_[@]}" "$exe --version" | sed -n 's/^build //p')" || true
    if [ "$remote_id" != "$COMMIT" ]; then
        echo "error: $alias: selfplay reports build '$remote_id', expected $COMMIT" >&2
        return 1
    fi
    echo "$alias: selfplay $remote_id"
}

status=0
for machine in "${machines[@]}"; do
    read -r alias target exe <<<"$machine"
    sync_machine "$alias" "$target" "$exe" || status=1
done
exit $status
