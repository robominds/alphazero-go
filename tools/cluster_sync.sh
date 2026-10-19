#!/usr/bin/env bash
# cluster_sync.sh: brings every remote machine of the hosts file to this
# repository's commit, so distributed self-play's build-ID check passes.
#
# For each machine with ssh= in the hosts file, the repository on it is
# exe's grandparent directory (exe=<repo>/build-metal/selfplay). The commit
# is sent with `git archive HEAD | ssh ... tar -x` into <repo>.new, the old
# tree is swapped out while its build directory moves along (incremental
# rebuilds), and selfplay is built with -DAZ_USE_METAL=ON and
# -DAZGO_BUILD_ID=<commit> (the copy has no .git). Its --version must then
# equal this repository's commit.
#
# A dirty tree is refused: a remote cannot rebuild uncommitted changes, and
# their build ID would differ anyway.
#
# Mark Castelluccio, with assistance from Claude (Anthropic).
set -euo pipefail

usage() {
    cat <<'EOF'
usage: tools/cluster_sync.sh [--hosts FILE] [--build build-metal] [--dry-run]

Syncs every machine with ssh= in the hosts file (default: $AZGO_HOSTS, else
~/.config/azgo/hosts) to this repository's HEAD and builds selfplay there in
<repo>/<build> (default build-metal), <repo> being exe's grandparent. Checks
that each machine's selfplay --version reports HEAD. --dry-run prints the
commands instead of running them. Refuses a tree with uncommitted changes.
EOF
}

HOSTS="" BUILD_DIR="build-metal" DRY=0
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
[[ "$BUILD_DIR" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "error: bad --build '$BUILD_DIR'" >&2; exit 2; }

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

status=0
for machine in "${machines[@]}"; do
    read -r alias target exe <<<"$machine"
    repo="$(dirname "$(dirname "$exe")")"
    case "$repo" in ''|.|/|'~'|'~/') echo "error: $alias: exe=$exe gives no usable repository directory" >&2; exit 1 ;; esac
    echo "== $alias: $COMMIT -> $repo ($BUILD_DIR)"
    ssh_=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$target")
    run "${ssh_[@]}" "rm -rf $repo.new && mkdir -p $repo.new"
    if [ "$DRY" = 1 ]; then
        echo "git archive $COMMIT | ${ssh_[*]} 'tar -x -m -C $repo.new'"
    else
        # -m: files get the time of extraction, not of their commit, so make
        # never takes an object built earlier for newer than a changed source
        # (every sync then rebuilds what the moved build directory holds).
        git archive "$COMMIT" | "${ssh_[@]}" "tar -x -m -C $repo.new"
    fi
    # Swap the trees; the build directory (CMake cache and objects) moves along.
    run "${ssh_[@]}" "if [ -d $repo/$BUILD_DIR ]; then mv $repo/$BUILD_DIR $repo.new/; fi;
        rm -rf $repo.old; if [ -d $repo ]; then mv $repo $repo.old; fi; mv $repo.new $repo; rm -rf $repo.old"
    run "${ssh_[@]}" "cmake -S $repo -B $repo/$BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DAZ_USE_METAL=ON \
        -DAZGO_BUILD_ID=$COMMIT > /dev/null && cmake --build $repo/$BUILD_DIR -j \$(sysctl -n hw.ncpu) --target selfplay"
    if [ "$DRY" = 1 ]; then
        run "${ssh_[@]}" "$exe --version"
        continue
    fi
    remote_id="$("${ssh_[@]}" "$exe --version" | sed -n 's/^build //p')"
    if [ "$remote_id" = "$COMMIT" ]; then
        echo "$alias: selfplay $remote_id"
    else
        echo "error: $alias: selfplay reports build '$remote_id', expected $COMMIT" >&2
        status=1
    fi
done
exit $status
