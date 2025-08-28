# Repository Audit: Usage

## Quick Start
1. Copy `repo_audit.py` to the repository root.
2. Run: `python3 repo_audit.py`
3. Open `repo_report.md` for a human-readable summary. Raw details are in `audit/report.json`.

Requirements:
- Python 3.10+ (3.11+ preferred). `git` must be installed and accessible.
- No external dependencies needed; will use `tomllib` if on Python 3.11+, otherwise attempts `tomli` if present (the GitHub Action installs it for you automatically for older Pythons).

## What You Get
- Inventory of every tracked file (size, type, last commit date).
- Detected manifests (pyproject.toml, requirements.txt, package.json, Dockerfile, etc.).
- Entry points (Python `if __name__ == "__main__"` files, common app launchers, `package.json` main and node scripts).
- Static dependency graph across Python and JS/TS (plus `.sh` sources).
- Reachability from entry points (what's actually used at runtime).
- Candidate redundant files (not reachable; tests/docs excluded).
- Successor/legacy candidates (e.g., `v1` vs `v2`, `old`, `legacy`, `deprecated`, `backup`, `new`).
- Archive/legacy-like directories containing files (archive, old, legacy, tmp).

## Interpreting Results
- "Likely Redundant" lists files not reachable from any entry point and not obviously test or docs. Review before removal—dynamic imports/plugins may be missed.
- "Successor/Legacy Groups" show clusters of similarly named files; newer commit timestamps usually indicate successors.
- "Archive/Legacy-like Directories" help focus clean-up on historically parked files.

## Common Follow-ups
- Remove or archive redundant files (create a `archive/` folder or a dedicated branch).
- Consolidate "v1/v2/old/new" variants; keep the newest, migrate references if needed.
- Update imports and scripts after consolidation; re-run the audit to verify reachability.
- Add CI checks to fail on orphaned files (optional enhancement).

## Limitations
- Static analysis only; dynamic import patterns, reflection, and runtime file loading may not be captured.
- Python module resolution is heuristic (namespace packages and complex sys.path tweaks aren't fully modeled).
- JS resolution follows relative imports and common extension/index patterns, not full Node resolution.