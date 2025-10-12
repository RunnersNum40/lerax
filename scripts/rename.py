import re
import sys
from pathlib import Path

OLD = "lerax"
NEW = sys.argv[1]


def pascal(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", " ", s).title().replace(" ", "")


NEWP = pascal(NEW)
OLDP = pascal(OLD)

root = Path(".").resolve()
ignore_dirs = {
    ".git",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "__pycache__",
    "dist",
    "build",
}

paths = []
for pat in ("**/*.py", "**/*.md", "**/*.toml", "**/*.yaml", "**/*.yml"):
    paths += list(root.glob(pat))


def skip(p: Path):
    return any(part in ignore_dirs for part in p.parts)


for p in paths:
    if skip(p) or not p.is_file():
        continue
    text = p.read_text(encoding="utf-8")
    orig = text
    text = re.sub(rf"\b{OLD}\b", NEW, text)
    text = re.sub(rf"\b{OLDP}\b", NEWP, text)
    text = text.replace(f"--cov={OLD}", f"--cov={NEW}")
    if text != orig:
        p.write_text(text, encoding="utf-8")

old_pkg = root / OLD
new_pkg = root / NEW
if old_pkg.exists():
    if new_pkg.exists():
        print(f"Target dir {new_pkg} already exists; abort.", file=sys.stderr)
        sys.exit(2)
    old_pkg.rename(new_pkg)

print(
    f"Renamed package '{OLD}' -> '{NEW}' and references ' {OLD}/{OLDP} ' -> ' {NEW}/{NEWP} '."
)
