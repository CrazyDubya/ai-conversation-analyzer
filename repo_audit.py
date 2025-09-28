#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import stat
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ----------------------------
# Config
# ----------------------------
IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "dist", "build", ".next", ".nuxt",
    ".venv", "venv", "env", ".tox", ".ruff_cache",
    "coverage", "htmlcov", ".DS_Store",
}
CODE_EXTS_PY = {".py", ".pyi"}
CODE_EXTS_JS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
CODE_EXTS_SHELL = {".sh", ".bash"}
NOTEBOOK_EXTS = {".ipynb"}
TEXT_LIKE_EXTS = CODE_EXTS_PY | CODE_EXTS_JS | CODE_EXTS_SHELL | NOTEBOOK_EXTS | {
    ".json", ".yml", ".yaml", ".toml", ".md", ".txt", ".ini", ".cfg", ".conf"
}
CONFIG_FILES = {
    "pyproject.toml", "requirements.txt", "Pipfile", "poetry.lock", "setup.py",
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
}
ARCHIVE_DIR_HINTS = {"archive", "archives", "old", "legacy", "deprecated", "tmp", "backup", "backups"}
LEGACY_NAME_HINTS = re.compile(r"(?:^|[\W_])(v\d+|old|legacy|deprecated|backup|copy|new)(?:$|[\W_])", re.I)

ENTRYPOINT_PY_FILENAMES = {"main.py", "__main__.py", "app.py", "manage.py", "wsgi.py", "asgi.py"}
ENTRYPOINT_JS_CANDIDATES = {"index.js", "index.ts", "src/index.js", "src/index.ts", "server.js", "app.js"}

# ----------------------------
# Utilities
# ----------------------------

def is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
            # Heuristic: non-text ratio
            # ASCII control characters: 7=BEL, 8=BS, 9=TAB, 10=LF, 12=FF, 13=CR, 27=ESC
            textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)))
            nontext = chunk.translate(None, textchars)
            return len(nontext) / max(1, len(chunk)) > 0.30
    except Exception:
        return False

def git_last_commit_date(path: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False
        )
        s = res.stdout.strip()
        return s or None
    except Exception:
        return None

def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def load_json_safe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def load_toml(path: Path):
    try:
        import tomllib  # py311+
        try:
            return tomllib.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return None
    except ImportError:
        try:
            import tomli  # type: ignore
            try:
                return tomli.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                return None
        except ImportError:
            return None

def norm_relpath(p: Path, root: Path) -> str:
    return str(p.relative_to(root).as_posix())

def file_mtime_iso(p: Path) -> str:
    try:
        t = p.stat().st_mtime
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))
    except Exception:
        return ""

def ext_of(p: Path) -> str:
    return p.suffix.lower()

def is_executable(p: Path) -> bool:
    try:
        mode = p.stat().st_mode
        return bool(mode & stat.S_IXUSR)
    except Exception:
        return False

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class FileInfo:
    path: str
    size: int
    ext: str
    is_binary: bool
    mtime: str
    last_commit: Optional[str]
    executable: bool

@dataclass
class ProjectInfo:
    manifests: Dict[str, bool]
    package_json: Optional[dict]
    pyproject: Optional[dict]

@dataclass
class Graph:
    # adjacency list of local file dependencies
    edges: Dict[str, Set[str]]
    nodes: Set[str]

# ----------------------------
# Scanning
# ----------------------------
def scan_files(root: Path) -> Dict[str, FileInfo]:
    out: Dict[str, FileInfo] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignore dirs
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".git")]
        for fn in filenames:
            full = Path(dirpath) / fn
            try:
                rel = norm_relpath(full, root)
            except Exception:
                continue
            if rel.startswith(".git/"):
                continue
            try:
                st = full.stat()
                size = st.st_size
            except Exception:
                size = 0
            info = FileInfo(
                path=rel,
                size=size,
                ext=ext_of(full),
                is_binary=is_binary_file(full),
                mtime=file_mtime_iso(full),
                last_commit=git_last_commit_date(full),
                executable=is_executable(full),
            )
            out[rel] = info
    return out

def detect_project(root: Path, files: Dict[str, FileInfo]) -> ProjectInfo:
    manifests = {k: (k in files) for k in CONFIG_FILES}
    package_json = load_json_safe(root/"package.json") if manifests.get("package.json") else None
    pyproject = load_toml(root/"pyproject.toml") if manifests.get("pyproject.toml") else None
    return ProjectInfo(manifests=manifests, package_json=package_json, pyproject=pyproject)

# ----------------------------
# Dependency parsing
# ----------------------------
IMPORT_PY = re.compile(r'^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))', re.M)
IMPORT_JS = re.compile(r"""
    import\s+(?:.+?\s+from\s+)?['"]([^'"]+)['"]|
    require\(\s*['"]([^'"]+)['"]\s*\)|
    import\(\s*['"]([^'"]+)['"]\s*\)
""", re.X)

def resolve_js_like(path: Path, target: str, root: Path) -> Optional[str]:
    # Only resolve relative imports
    if not target.startswith("."):
        return None
    base = (path.parent / target).resolve()
    # Try direct file with known extensions
    exts = [".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".json"]
    candidates: List[Path] = []
    if base.suffix:
        candidates.append(base)
    else:
        for e in exts:
            candidates.append(base.with_suffix(e))
        # index.* inside directory
        for e in exts:
            candidates.append(base / f"index{e}")
    for c in candidates:
        try:
            rel = norm_relpath(c, root)
        except Exception:
            continue
        if rel in files_glob:
            return rel
    return None

def resolve_py_like(path: Path, module: str, root: Path) -> Optional[str]:
    # Handles relative: .foo, ..bar
    if module.startswith("."):
        # count dots
        dots = len(module) - len(module.lstrip("."))
        leftover = module[dots:]
        p = path.parent
        for _ in range(dots):
            p = p.parent
        if leftover:
            p = p / Path(*leftover.split("."))
        # try file or package __init__.py
        for cand in [p.with_suffix(".py"), p / "__init__.py"]:
            try:
                rel = norm_relpath(cand, root)
            except Exception:
                continue
            if rel in files_glob:
                return rel
        return None
    else:
        # absolute module in repo
        p = root / Path(*module.split("."))
        for cand in [p.with_suffix(".py"), p / "__init__.py"]:
            try:
                rel = norm_relpath(cand, root)
            except Exception:
                continue
            if rel in files_glob:
                return rel
        return None

def parse_imports_for_file(rel: str, root: Path) -> Set[str]:
    p = root / rel
    ext = ext_of(p)
    text = ""
    if ext in NOTEBOOK_EXTS:
        # Extract code from notebook cells
        nb = load_json_safe(p)
        if nb and "cells" in nb:
            srcs = []
            for c in nb.get("cells", []):
                if c.get("cell_type") == "code":
                    src = "".join(c.get("source", []))
                    srcs.append(src)
            text = "\n".join(srcs)
        else:
            return set()
    else:
        text = read_text_safe(p)
    deps: Set[str] = set()
    if ext in CODE_EXTS_PY or (ext in NOTEBOOK_EXTS):
        for m in IMPORT_PY.finditer(text):
            target = m.group(1) or m.group(2)
            if not target:
                continue
            rel_dep = resolve_py_like(p, target.strip(), root)
            if rel_dep:
                deps.add(rel_dep)
    elif ext in CODE_EXTS_JS:
        for m in IMPORT_JS.finditer(text):
            target = m.group(1) or m.group(2) or m.group(3)
            if not target:
                continue
            rel_dep = resolve_js_like(p, target.strip(), root)
            if rel_dep:
                deps.add(rel_dep)
    elif ext in CODE_EXTS_SHELL:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("source ") or line.startswith(". "):
                t = line.split(maxsplit=1)[1].strip()
                if t.startswith("./") or t.startswith("../"):
                    candidate = (p.parent / t).resolve()
                    try:
                        rel_dep = norm_relpath(candidate, root)
                    except Exception:
                        rel_dep = None
                    if rel_dep and rel_dep in files_glob:
                        deps.add(rel_dep)
    return deps

def build_graph(root: Path, files: Dict[str, FileInfo]) -> Graph:
    nodes = set(files.keys())
    edges: Dict[str, Set[str]] = {k: set() for k in nodes}
    for rel in nodes:
        ext = ext_of(root / rel)
        if ext in (CODE_EXTS_PY | CODE_EXTS_JS | CODE_EXTS_SHELL | NOTEBOOK_EXTS):
            deps = parse_imports_for_file(rel, root)
            if deps:
                edges[rel].update(d for d in deps if d in nodes)
    return Graph(edges=edges, nodes=nodes)

def detect_entrypoints(root: Path, files: Dict[str, FileInfo], proj: ProjectInfo) -> Set[str]:
    eps: Set[str] = set()
    # Python
    for rel, info in files.items():
        p = root / rel
        if info.ext in CODE_EXTS_PY or info.ext in NOTEBOOK_EXTS:
            text = read_text_safe(p)
            if "__name__" in text and "__main__" in text and "if __name__" in text:
                eps.add(rel)
            if Path(rel).name in ENTRYPOINT_PY_FILENAMES:
                eps.add(rel)
    # console_scripts in pyproject
    if proj.pyproject:
        try:
            proj_scripts = proj.pyproject.get("project", {}).get("scripts", {})
            for _, target in proj_scripts.items():
                # module:function or path:func
                mod = str(target)
                mod = mod.split(":")[0]
                resolved = resolve_py_like(root / "X.py", mod, root)  # fake path for absolute
                if resolved:
                    eps.add(resolved)
        except Exception:
            pass
        # legacy 'tool.poetry.scripts'
        try:
            poe_scripts = proj.pyproject.get("tool", {}).get("poetry", {}).get("scripts", {})
            for _, target in poe_scripts.items():
                mod = str(target).split(":")[0]
                resolved = resolve_py_like(root / "X.py", mod, root)
                if resolved:
                    eps.add(resolved)
        except Exception:
            pass
    # JS entrypoints
    if proj.package_json:
        main = proj.package_json.get("main")
        if isinstance(main, str):
            p = (root / main).resolve()
            # try variants
            candidates = []
            if p.suffix:
                candidates.append(p)
            else:
                for e in [".js", ".ts", ".mjs", ".cjs", ".jsx", ".tsx"]:
                    candidates.append(p.with_suffix(e))
                candidates.extend([(p / f"index{e}") for e in [".js", ".ts", ".mjs", ".cjs", ".jsx", ".tsx"]])
            for c in candidates:
                try:
                    rel = norm_relpath(c, root)
                except Exception:
                    continue
                if rel in files:
                    eps.add(rel)
                    break
        # scripts referencing node entry files
        scripts = proj.package_json.get("scripts", {}) if isinstance(proj.package_json.get("scripts", {}), dict) else {}
        for cmd in scripts.values():
            if not isinstance(cmd, str):
                continue
            m = re.search(r"\bnode\s+([^\s;&|]+)", cmd)
            if m:
                t = m.group(1)
                if t.startswith("."):
                    c = (root / t).resolve()
                    try:
                        rel = norm_relpath(c, root)
                    except Exception:
                        rel = None
                    if rel and rel in files:
                        eps.add(rel)
    # common JS candidates
    for cand in ENTRYPOINT_JS_CANDIDATES:
        if cand in files:
            eps.add(cand)
    return eps

def reachable_from(graph: Graph, starts: Set[str]) -> Set[str]:
    seen: Set[str] = set()
    stack: List[str] = list(starts)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for m in graph.edges.get(n, ()):
            if m not in seen:
                stack.append(m)
    return seen

def guess_redundant(files: Dict[str, FileInfo], reachable: Set[str]) -> List[str]:
    redundant = []
    for rel, info in files.items():
        p = Path(rel)
        # Skip tests and docs
        name = p.name.lower()
        parts = set([seg.lower() for seg in p.parts])
        if rel in reachable:
            continue
        if name.startswith("test_") or name.endswith("_test.py") or name.startswith("spec.") or name.endswith(".spec.js"):
            continue
        if "test" in parts or "tests" in parts or "docs" in parts or "doc" in parts or "examples" in parts:
            continue
        # Skip configs & root manifests
        if p.name in CONFIG_FILES:
            continue
        # Consider code and notebooks only; mark other text as maybe
        if info.ext in (CODE_EXTS_PY | CODE_EXTS_JS | NOTEBOOK_EXTS | CODE_EXTS_SHELL):
            redundant.append(rel)
    return sorted(redundant)

def find_successor_groups(files: Dict[str, FileInfo]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    def basekey(p: Path) -> str:
        # strip version-ish suffixes from filename
        stem = p.stem
        stem_clean = LEGACY_NAME_HINTS.sub("", stem)
        return f"{p.parent.as_posix()}/{stem_clean}".lower()
    for rel in files.keys():
        p = Path(rel)
        key = basekey(p)
        groups.setdefault(key, []).append(rel)
    # Only keep groups with >1 member AND at least one name hit the legacy/new pattern
    final = {}
    for k, members in groups.items():
        if len(members) < 2:
            continue
        if any(LEGACY_NAME_HINTS.search(Path(m).stem) for m in members):
            final[k] = sorted(members)
    return final

def find_archive_dirs(files: Dict[str, FileInfo]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for rel in files.keys():
        parts = [seg.lower() for seg in Path(rel).parts[:-1]]
        for part in parts:
            if part in ARCHIVE_DIR_HINTS:
                out.setdefault(part, []).append(rel)
    return out

def summarize_exts(files: Dict[str, FileInfo]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for info in files.values():
        d[info.ext or "(noext)"] = d.get(info.ext or "(noext)", 0) + 1
    return dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

# ----------------------------
# Reporting
# ----------------------------
def write_report(root: Path,
                 files: Dict[str, FileInfo],
                 proj: ProjectInfo,
                 graph: Graph,
                 entrypoints: Set[str],
                 reachable: Set[str],
                 redundant: List[str],
                 succ_groups: Dict[str, List[str]],
                 archives: Dict[str, List[str]]) -> None:
    report_md = Path("repo_report.md")
    audit_dir = Path("audit")
    audit_dir.mkdir(exist_ok=True)
    report_json = audit_dir / "report.json"

    total_size = sum(fi.size for fi in files.values())
    exts = summarize_exts(files)

    md_lines: List[str] = []
    md_lines.append(f"# Repository Audit Report")
    md_lines.append("")
    md_lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    md_lines.append(f"- Root: {root.resolve().as_posix()}")
    md_lines.append(f"- Files tracked: {len(files)}")
    md_lines.append(f"- Total size (bytes): {total_size}")
    md_lines.append("")
    md_lines.append("## Detected Manifests")
    for k, v in sorted(proj.manifests.items()):
        md_lines.append(f"- {k}: {'present' if v else 'absent'}")
    if proj.package_json:
        md_lines.append("- package.json: detected")
    if proj.pyproject:
        md_lines.append("- pyproject.toml: detected")
    md_lines.append("")
    md_lines.append("## File Type Summary (by extension)")
    for ext, cnt in exts.items():
        md_lines.append(f"- {ext}: {cnt}")
    md_lines.append("")
    md_lines.append("## Entry Points (starting points for reachability)")
    if entrypoints:
        for ep in sorted(entrypoints):
            md_lines.append(f"- {ep}")
    else:
        md_lines.append("- None detected (no obvious Python or JS/TS entry files found)")
    md_lines.append("")
    md_lines.append("## Reachability Summary")
    md_lines.append(f"- Reachable from entry points: {len(reachable)}")
    md_lines.append(f"- Unreachable (candidates for review): {len(files) - len(reachable)}")
    md_lines.append("")
    md_lines.append("## Likely Redundant (review carefully)")
    if redundant:
        for r in redundant:
            md_lines.append(f"- {r}")
    else:
        md_lines.append("- None found by heuristics")
    md_lines.append("")
    md_lines.append("## Successor/Legacy Groups (based on names and timestamps)")
    if succ_groups:
        for base, members in succ_groups.items():
            md_lines.append(f"- Group: {base}")
            # Sort by last commit date desc
            members_sorted = sorted(
                members, key=lambda m: files[m].last_commit or files[m].mtime, reverse=True
            )
            for m in members_sorted:
                fi = files[m]
                md_lines.append(f"  - {m} (last_commit={fi.last_commit}, mtime={fi.mtime})")
    else:
        md_lines.append("- No apparent successor/legacy groups detected by naming")
    md_lines.append("")
    md_lines.append("## Files Under Archive/Legacy-like Directories")
    if archives:
        for k, vals in archives.items():
            md_lines.append(f"- {k} ({len(vals)} files)")
    else:
        md_lines.append("- None detected")
    md_lines.append("")
    md_lines.append("## Largest Files (Top 20)")
    largest = sorted(files.values(), key=lambda x: x.size, reverse=True)[:20]
    for fi in largest:
        md_lines.append(f"- {fi.path} — {fi.size} bytes")
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- Redundant and successor detections are heuristic; confirm before deleting or moving files.")
    md_lines.append("- Reachability is derived from static import analysis (Python, JS/TS, shell sources). Dynamic imports and runtime plugin loading may not be detected.")
    md_lines.append("- Tests, docs, and example folders are excluded from redundancy by default.")
    md_lines.append("")

    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    json_out = {
        "generated_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "root": root.resolve().as_posix(),
        "files": {k: asdict(v) for k, v in files.items()},
        "project": {
            "manifests": proj.manifests,
            "package_json": proj.package_json,
            "pyproject": proj.pyproject,
        },
        "graph": {
            "edges": {k: sorted(list(v)) for k, v in graph.edges.items()},
            "nodes": sorted(list(graph.nodes)),
        },
        "entrypoints": sorted(list(entrypoints)),
        "reachable": sorted(list(reachable)),
        "redundant_candidates": redundant,
        "successor_groups": succ_groups,
        "archive_dirs": archives,
    }
    report_json.write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"Wrote {report_md} and {report_json}")

# ----------------------------
# Main
# ----------------------------
def main():
    root = Path(".").resolve()
    files = scan_files(root)
    proj = detect_project(root, files)
    graph = build_graph(root, files)
    entrypoints = detect_entrypoints(root, files, proj)
    reachable = reachable_from(graph, entrypoints) if entrypoints else set()
    redundant = guess_redundant(files, reachable)
    succ_groups = find_successor_groups(files)
    archives = find_archive_dirs(files)
    write_report(root, files, proj, graph, entrypoints, reachable, redundant, succ_groups, archives)

if __name__ == "__main__":
    main()