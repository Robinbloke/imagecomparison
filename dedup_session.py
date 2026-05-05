"""
Session persistence and duplicate-group logic for the image duplicate finder.

JSON schema (schema_version 3):
  As v2, plus last_checkpoint (scan|hash|compare|complete),
  comparison_complete, go_resume_choice (auto|scan|hash|compare).
  Each user_state entry may include checked_duplicates (bool): user marked
  the group as reviewed in the duplicate-groups list.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

SCHEMA_VERSION = 3

# Common raster image extensions only (lowercase keys for lookup).
IMAGE_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        ".jfif",
    }
)

# Avoid ENAMETOOLONG (e.g. Linux PATH_MAX ~4096, NAME_MAX 255 bytes).
_PATH_MAX_BYTES = 4095
_NAME_MAX_BYTES = 255


def path_within_os_limits(path: Path) -> bool:
    """True if encoded path length and each filename fit typical OS limits."""
    try:
        full = os.fsencode(str(path))
    except UnicodeEncodeError:
        return False
    if len(full) >= _PATH_MAX_BYTES:
        return False
    for part in path.parts:
        try:
            nb = os.fsencode(part)
        except UnicodeEncodeError:
            return False
        if len(nb) > _NAME_MAX_BYTES:
            return False
    return True


def hamming_to_similarity_pct(hamming: float) -> float:
    return round(100.0 * (1.0 - float(hamming) / 64.0), 2)


def min_similarity_to_max_hamming(min_similarity_pct: float) -> int:
    p = max(0.0, min(100.0, float(min_similarity_pct)))
    return min(64, max(0, int(round(64.0 * (100.0 - p) / 100.0))))


def collect_image_paths(
    folders: List[str],
    recursive: bool,
    on_progress: Optional[Callable[[int, str, int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_every: int = 400,
) -> List[str]:
    """
    Walk folders and collect image paths. Optional on_progress(listed_count, folder,
    folder_index_1based, folder_total) is called when starting each folder and every
    progress_every glob entries so UIs can show listing progress.
    """
    out: List[Path] = []
    folder_total = max(1, len(folders))
    folder_idx = 0
    glob_steps = 0
    for raw in folders:
        if cancel_check is not None and cancel_check():
            break
        folder_idx += 1
        try:
            root = Path(raw).expanduser().resolve()
        except OSError:
            continue
        if not root.is_dir():
            continue
        if on_progress is not None:
            on_progress(len(out), str(root), folder_idx, folder_total)
        pattern = "**/*" if recursive else "*"
        try:
            candidates = root.glob(pattern)
        except OSError:
            continue
        for p in candidates:
            if cancel_check is not None and cancel_check():
                out.sort(key=lambda x: str(x))
                return [str(x) for x in out]
            glob_steps += 1
            if (
                on_progress is not None
                and progress_every > 0
                and glob_steps % progress_every == 0
            ):
                on_progress(len(out), str(root), folder_idx, folder_total)
            try:
                if p.is_dir() or p.name.startswith("."):
                    continue
                suf = p.suffix.lower()
                if suf not in IMAGE_SUFFIXES:
                    continue
                resolved = p.resolve()
                if not path_within_os_limits(resolved):
                    continue
                if not resolved.is_file():
                    continue
                out.append(resolved)
            except OSError:
                continue
    out.sort(key=lambda x: str(x))
    return [str(p) for p in out]


def build_adjacency_from_duplicates(
    duplicates: Dict[str, List],
) -> Dict[str, Set[str]]:
    """Edges for pairs returned by find_duplicates(scores=True)."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    for src, targets in duplicates.items():
        for item in targets:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                other, _score = item[0], item[1]
            else:
                other = item
            adj[src].add(other)
            adj[other].add(src)
    return {k: v for k, v in adj.items()}


def connected_components(adj: Dict[str, Set[str]]) -> List[Set[str]]:
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    for node in adj:
        if node in seen:
            continue
        stack = [node]
        comp: Set[str] = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.add(n)
            for nb in adj.get(n, ()):
                if nb not in seen:
                    stack.append(nb)
        if len(comp) > 1:
            comps.append(comp)
    return comps


def stable_group_id(paths: List[str]) -> str:
    joined = "\n".join(sorted(paths))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def build_groups(
    duplicates: Dict[str, List],
    encodings: Dict[str, str],
    hamming_distance_fn: Callable[[str, str], float],
) -> List[Dict[str, Any]]:
    adj = build_adjacency_from_duplicates(duplicates)
    comps = connected_components(adj)
    groups: List[Dict[str, Any]] = []
    for comp in comps:
        plist = sorted(comp)
        gid = stable_group_id(plist)
        ref = plist[0]
        ref_hash = encodings.get(ref, "")
        similarities: Dict[str, float] = {}
        for p in plist:
            h2 = encodings.get(p, "")
            if ref_hash and h2:
                try:
                    d = float(hamming_distance_fn(ref_hash, h2))
                    similarities[p] = hamming_to_similarity_pct(d)
                except (ValueError, TypeError):
                    similarities[p] = 0.0
            else:
                similarities[p] = 0.0
        min_sim = min(similarities.values()) if similarities else 0.0
        groups.append(
            {
                "group_id": gid,
                "paths": plist,
                "similarity_to_ref": similarities,
                "min_similarity_pct_in_group": round(min_sim, 2),
            }
        )
    groups.sort(key=lambda g: (-len(g["paths"]), g["paths"][0]))
    return groups


def default_user_state_for_group(group_id: str, paths: List[str]) -> Dict[str, Any]:
    keeper = sorted(paths)[0] if paths else ""
    return {
        "group_id": group_id,
        "keeper_path": keeper,
        "marked_duplicate_paths": [],
        "checked_duplicates": False,
    }


def merge_user_state(
    saved: Dict[str, Any],
    groups: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return group_id -> user_state, filling defaults and pruning stale paths."""
    by_id: Dict[str, Dict[str, Any]] = {}
    raw = saved.get("user_state") or {}
    if isinstance(raw, list):
        raw = {item["group_id"]: item for item in raw if isinstance(item, dict)}
    for g in groups:
        gid = g["group_id"]
        paths_set = set(g["paths"])
        prev = raw.get(gid) if isinstance(raw, dict) else None
        if not isinstance(prev, dict):
            by_id[gid] = default_user_state_for_group(gid, g["paths"])
            continue
        keeper = prev.get("keeper_path") or ""
        if keeper not in paths_set:
            keeper = sorted(paths_set)[0] if paths_set else ""
        marked = prev.get("marked_duplicate_paths") or []
        if not isinstance(marked, list):
            marked = []
        marked = [p for p in marked if p in paths_set and p != keeper]
        by_id[gid] = {
            "group_id": gid,
            "keeper_path": keeper,
            "marked_duplicate_paths": marked,
            "checked_duplicates": bool(prev.get("checked_duplicates")),
        }
    return by_id


def session_to_dict(
    folders: List[str],
    recursive: bool,
    hash_method: str,
    min_similarity_pct: float,
    phase: str,
    image_paths: List[str],
    encodings: Dict[str, str],
    groups: List[Dict[str, Any]],
    user_state: Dict[str, Dict[str, Any]],
    duplicates_raw: Optional[Dict[str, List]] = None,
    listing_total_images: int = 0,
    encoding_last_path_index: int = -1,
    images_hashed_count: int = 0,
    last_checkpoint: str = "scan",
    comparison_complete: bool = False,
    go_resume_choice: str = "auto",
) -> Dict[str, Any]:
    us_list = list(user_state.values())
    d: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "folders": folders,
        "recursive": recursive,
        "hash_method": hash_method,
        "min_similarity_pct": min_similarity_pct,
        "phase": phase,
        "image_paths": image_paths,
        "encodings": encodings,
        "groups": groups,
        "user_state": us_list,
        "listing_total_images": int(listing_total_images),
        "encoding_last_path_index": int(encoding_last_path_index),
        "images_hashed_count": int(images_hashed_count),
        "last_checkpoint": str(last_checkpoint),
        "comparison_complete": bool(comparison_complete),
        "go_resume_choice": str(go_resume_choice),
    }
    if duplicates_raw is not None:
        d["duplicates_raw"] = duplicates_raw
    return d


def session_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    ver = int(data.get("schema_version", 0))
    if ver > SCHEMA_VERSION:
        raise ValueError("Unsupported session file version")
    paths = list(data.get("image_paths") or [])
    enc = dict(data.get("encodings") or {})
    n_hashed = sum(1 for p in paths if enc.get(p))
    listed = int(data.get("listing_total_images", len(paths)))
    eli = data.get("encoding_last_path_index")
    if eli is None:
        eli = -1
    ich = data.get("images_hashed_count")
    if ich is None:
        ich = n_hashed
    groups = list(data.get("groups") or [])
    lcp = data.get("last_checkpoint")
    if lcp is None:
        if groups and n_hashed >= len(paths) and paths:
            lcp = "complete"
        elif n_hashed > 0 and paths:
            lcp = "hash"
        else:
            lcp = "scan"
    cc = data.get("comparison_complete")
    if cc is None:
        cc = bool(groups)
    grc = str(data.get("go_resume_choice") or "auto")
    if grc not in ("auto", "scan", "hash", "compare"):
        grc = "auto"
    lcp = str(lcp)
    if lcp not in ("scan", "hash", "compare", "complete"):
        lcp = "scan"
    return {
        "folders": list(data.get("folders") or []),
        "recursive": bool(data.get("recursive", True)),
        "hash_method": str(data.get("hash_method") or "phash"),
        "min_similarity_pct": float(data.get("min_similarity_pct") or 85.0),
        "phase": str(data.get("phase") or "idle"),
        "image_paths": paths,
        "encodings": enc,
        "groups": groups,
        "user_state": data.get("user_state") or {},
        "duplicates_raw": data.get("duplicates_raw"),
        "listing_total_images": listed,
        "encoding_last_path_index": int(eli),
        "images_hashed_count": int(ich),
        "last_checkpoint": str(lcp),
        "comparison_complete": bool(cc),
        "go_resume_choice": grc,
    }


def save_session(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_session(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return session_from_dict(data)
