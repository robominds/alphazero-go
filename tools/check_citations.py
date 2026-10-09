#!/usr/bin/env python3
"""Verify the file:line citations in docs/algorithm-explained.{md,html}.

The alphazero-go explainer quotes the real C++ -- library sources in src/
(the Metal kernels in src/metal/*.metal and Objective-C++ in *.mm too),
headers in include/azgo/, the apps in apps/, a few tests in tests/ and the
Metal spike in tools/ --
and captions each excerpt with the file and line it came from. Those line
numbers rot: every commit that touches a source file above a quoted line
silently invalidates a citation, and nothing about the document looks
wrong afterwards. This catches that. (Ported from alphazero-tictactoe.)

Three checks:

  markdown captions   `*src/foo.cpp:NN -- `Sym`*` (or include/azgo/foo.hpp,
                      apps/foo.cpp, tests/test_foo.cpp) must name the line
                      where `Sym` is defined in that file.
  html captions       `<span class="src">src/foo.cpp:NN -- <code>Sym</code>`
                      must agree with the markdown caption for the same
                      symbol.
  html concept map    `<span class="loc">-- src/foo.cpp:NN, MM</span>` line
                      numbers must be the definition lines of the
                      `<code>Sym</code>` entries that precede them.
  markdown concept    table rows `| ... | `Sym`, `Sym2` -- `src/foo.cpp:NN, MM` |`
  map                 (symbols separated by commas or "and"), checked the
                      same way as the html concept map.

A definition is an unindented line where `Sym(` appears (a function or
member function defined at file scope), or failing that an unindented
`struct Sym` / `class Sym` (a type declared in a header).

The HTML checks run only once docs/algorithm-explained.html exists.
Run with --fix to rewrite the numbers in place. Standard library only; not
part of the build, run it by hand or from a pre-commit hook:

    python3 tools/check_citations.py          # report, exit 1 on problems
    python3 tools/check_citations.py --fix    # rewrite stale line numbers
"""
import argparse
import html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "algorithm-explained.md"
HTML = ROOT / "docs" / "algorithm-explained.html"

MD_CAPTION = re.compile(r"^\*(?P<path>[\w./]+\.(?:cpp|hpp|mm|metal)):(?P<line>\d+)\s+—\s+`(?P<sym>[^`]+)`(?P<tail>[^*]*)\*\s*$")
HTML_CAPTION = re.compile(r'<span class="src">(?P<path>[\w./]+\.(?:cpp|hpp|mm|metal)):(?P<line>\d+)\s+—\s+<code>(?P<sym>[^<]+)</code>')
MD_LOC = re.compile(r'(?P<codes>(?:`[^`]+`(?:,\s*|\s+and\s+)?)+)\s*—\s*`(?P<path>[\w./]+\.(?:cpp|hpp|mm|metal)):(?P<lines>[\d,\s]+)`')
HTML_LOC = re.compile(r'(?P<codes>(?:<code>[^<]+</code>(?:,\s*)?)+)\s*<span class="loc">—\s*(?P<path>[\w./]+\.(?:cpp|hpp|mm|metal)):(?P<lines>[\d,\s]+)</span>')

_source_cache = {}


def source_lines(path):
    if path not in _source_cache:
        _source_cache[path] = (ROOT / path).read_text().splitlines()
    return _source_cache[path]


def excerpt_above(md_lines, caption_index):
    """The lines inside the fenced block immediately above a caption."""
    i = caption_index - 1
    while i >= 0 and not md_lines[i].strip():
        i -= 1
    if i < 0 or not md_lines[i].startswith("```"):
        return None
    start = i - 1
    while start >= 0 and not md_lines[start].startswith("```"):
        start -= 1
    return md_lines[start + 1:i]


def normalize(text):
    """Strip a trailing line comment and collapse whitespace.

    The explainer abridges what it quotes -- dedenting, dropping casts,
    appending an explanatory `//` note -- so an exact match is too strict.
    """
    text = re.sub(r"\s*//.*$", "", text)
    return " ".join(text.split())


def find_line(path, text):
    """Line numbers (1-based) in `path` matching `text` after normalizing."""
    want = normalize(text)
    if not want:
        return []
    return [n for n, line in enumerate(source_lines(path), 1) if normalize(line) == want]


def anchor_line(path, excerpt_lines, symbol):
    """The line a caption should name: the captioned symbol's definition.

    Matching the quoted text instead would be stricter but does not work
    here -- the explainer abridges what it quotes, dedenting, dropping
    casts and appending explanatory notes, so most excerpts appear nowhere
    verbatim. Citing the definition is the convention the document already
    followed in the citations that were correct when it was written, and
    unlike a mid-function line it survives edits inside the function body.
    """
    del excerpt_lines  # kept for callers; see the docstring
    return definition_line(path, symbol)


def definition_line(path, symbol):
    """Line where `symbol` is defined, preferring a definition over a call."""
    bare = symbol.split("::")[-1].rstrip("()")
    lines = source_lines(path)
    for n, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "*")):
            continue
        if re.search(rf"\b{re.escape(symbol)}\s*\(", line) and not line.startswith((" ", "\t")):
            return n
    for n, line in enumerate(lines, 1):
        if re.search(rf"\b{re.escape(bare)}\s*\(", line) and not line.startswith((" ", "\t")):
            return n
    # Types are cited by their definition: `struct MctsEdge {`, `class Evaluator {`,
    # never a forward declaration such as `class Game;`.
    for n, line in enumerate(lines, 1):
        if re.match(rf"(?:struct|class)\s+{re.escape(bare)}\b(?!\s*;)", line):
            return n
    return None


def check_markdown(fix):
    lines = MD.read_text().splitlines()
    problems, repairs = [], 0
    truth = {}
    for index, line in enumerate(lines):
        m = MD_CAPTION.match(line)
        if not m:
            continue
        path, cited, sym = m["path"], int(m["line"]), m["sym"]
        excerpt = excerpt_above(lines, index)
        if excerpt is None:
            problems.append(f"{MD.name}:{index + 1}: caption for `{sym}` has no fenced block above it")
            continue
        correct = anchor_line(path, excerpt, sym)
        if correct is None:
            problems.append(f"{MD.name}:{index + 1}: cannot locate `{sym}` in {path} — fix by hand")
            continue
        truth[(path, sym)] = correct
        if correct == cited:
            continue
        if fix:
            lines[index] = line.replace(f"{path}:{cited}", f"{path}:{correct}", 1)
            repairs += 1
        else:
            problems.append(f"{MD.name}:{index + 1}: `{sym}` cites {path}:{cited}, belongs at {correct}")
    if fix and repairs:
        MD.write_text("\n".join(lines) + "\n")
    return problems, repairs, truth


def check_locations(match, symbols, fix, label, counter):
    """Concept-map entry: each symbol's cited line must be its definition.

    Returns the (possibly corrected) matched text; problems and repairs are
    appended to / counted in `counter` (a dict with "problems" and "repairs").
    """
    cited = [int(n) for n in re.findall(r"\d+", match["lines"])]
    if len(symbols) != len(cited):
        counter["problems"].append(f"{label}: concept-map entry lists {len(symbols)} symbols but {len(cited)} lines")
        return match.group(0)
    corrected = []
    for sym, line in zip(symbols, cited):
        actual = definition_line(match["path"], sym)
        if actual is None:
            counter["problems"].append(f"{label}: cannot locate `{sym}` in {match['path']}")
            corrected.append(line)
        elif actual != line:
            if fix:
                counter["repairs"] += 1
                corrected.append(actual)
            else:
                counter["problems"].append(
                    f"{label}: concept map cites {match['path']}:{line} for `{sym}`, defined at {actual}")
                corrected.append(line)
        else:
            corrected.append(line)
    if not fix or corrected == cited:
        return match.group(0)
    return match.group(0).replace(match["lines"].strip(), ", ".join(str(n) for n in corrected), 1)


def check_markdown_map(fix):
    """The markdown concept-map table, checked like the html one."""
    lines = MD.read_text().splitlines()
    counter = {"problems": [], "repairs": 0}
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        label = f"{MD.name}:{index + 1}"

        def fix_loc(m):
            symbols = re.findall(r"`([^`]+)`", m["codes"])
            return check_locations(m, symbols, fix, label, counter)

        lines[index] = MD_LOC.sub(fix_loc, line)
    if fix and counter["repairs"]:
        MD.write_text("\n".join(lines) + "\n")
    return counter["problems"], counter["repairs"]


def check_html(truth, fix):
    text = HTML.read_text()
    problems, repairs = [], 0

    def fix_caption(m):
        nonlocal repairs
        path, cited, sym = m["path"], int(m["line"]), html.unescape(m["sym"])
        correct = truth.get((path, sym))
        if correct is None or correct == cited:
            if correct is None:
                problems.append(f"{HTML.name}: caption for `{sym}` ({path}:{cited}) has no markdown counterpart")
            return m.group(0)
        if not fix:
            problems.append(f"{HTML.name}: `{sym}` cites {path}:{cited}, markdown says {correct}")
            return m.group(0)
        repairs += 1
        return m.group(0).replace(f"{path}:{cited}", f"{path}:{correct}", 1)

    text = HTML_CAPTION.sub(fix_caption, text)

    counter = {"problems": problems, "repairs": 0}

    def fix_loc(m):
        symbols = [html.unescape(s) for s in re.findall(r"<code>([^<]+)</code>", m["codes"])]
        return check_locations(m, symbols, fix, HTML.name, counter)

    text = HTML_LOC.sub(fix_loc, text)
    repairs += counter["repairs"]
    if fix and repairs:
        HTML.write_text(text)
    return problems, repairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="rewrite stale line numbers in place")
    args = parser.parse_args()

    md_problems, md_repairs, truth = check_markdown(args.fix)
    map_problems, map_repairs = check_markdown_map(args.fix)
    md_problems += map_problems
    md_repairs += map_repairs
    # The HTML rendering is written after the markdown; until it exists there
    # is nothing to compare against.
    html_present = HTML.exists()
    if html_present:
        html_problems, html_repairs = check_html(truth, args.fix)
    else:
        html_problems, html_repairs = [], 0
        print("html: not present, skipped")
    problems = md_problems + html_problems

    if args.fix:
        print(f"repaired {md_repairs} markdown and {html_repairs} html citations")
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} citation problem(s)")
        return 1
    if html_present:
        print(f"all {len(truth)} markdown citations resolve, and the html agrees")
    else:
        print(f"all {len(truth)} markdown citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
