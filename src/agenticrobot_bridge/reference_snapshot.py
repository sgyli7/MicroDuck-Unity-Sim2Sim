"""Independent, content-verified reference copies; never link to the mutable main project."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

MANIFEST = "reference-manifest.json"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_snapshot(source: Path, target: Path, roots: list[str], revision: str) -> dict:
    source, target = source.resolve(), target.resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Reference must be separate from the source tree")
    entries = []
    for root in roots:
        origin = (source / root).resolve()
        if not origin.is_relative_to(source) or ".." in Path(root).parts:
            raise ValueError(f"Source path escapes project: {root}")
        if not origin.exists():
            raise FileNotFoundError(origin)
        for entry in sorted(origin.rglob("*")) if origin.is_dir() else [origin]:
            if entry.is_symlink() or entry.is_junction():
                raise ValueError(f"Reference cannot retain mutable links: {entry}")
            if entry.is_file():
                entries.append(entry)
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for entry in entries:
        relative = entry.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry, destination)
        digest = _sha256(entry)
        if _sha256(destination) != digest:
            raise OSError(f"Reference copy changed while copying: {relative}")
        files[relative.as_posix()] = digest
    manifest = {
        "schemaVersion": 1,
        "engine": "MuJoCo",
        "purpose": "frozen-current-reference-not-PhysX-acceptance",
        "sourceRevision": revision,
        "roots": roots,
        "files": files,
    }
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_snapshot(target: Path) -> dict:
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    missing, changed = [], []
    for relative, expected in manifest["files"].items():
        path = target / relative
        if not path.is_file():
            missing.append(relative)
        elif _sha256(path) != expected:
            changed.append(relative)
    actual = set()
    for root in manifest["roots"]:
        path = target / root
        entries = path.rglob("*") if path.is_dir() else [path]
        actual.update(p.relative_to(target).as_posix() for p in entries if p.is_file())
    added = sorted(actual - manifest["files"].keys())
    return {
        "passed": not missing and not changed and not added,
        "missing": missing, "changed": changed, "added": added,
    }
