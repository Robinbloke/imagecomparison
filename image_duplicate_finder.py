#!/usr/bin/env python3
"""
Image duplicate finder — cross-platform (Windows/Linux) tkinter GUI.

Uses imagededup perceptual hashing (PHash default) to find near-duplicates.

On Linux distributions with PEP 668, use a virtual environment:

  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/python image_duplicate_finder.py

imagededup pulls in PyTorch; see requirements.txt for notes on CPU-only installs.
"""

from __future__ import annotations

import gc
import json
import os
import queue
import subprocess
import sys
import threading
import warnings
import webbrowser
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageTk

_HASH_EXC: tuple = (OSError, ValueError, TypeError)
try:
    from PIL.Image import DecompressionBombError

    _HASH_EXC = (*_HASH_EXC, DecompressionBombError)
except ImportError:
    pass


def _max_pixels_for_hash() -> int:
    """Match PIL bomb limit so we skip before decode (no warning)."""
    lim = getattr(Image, "MAX_IMAGE_PIXELS", None)
    if isinstance(lim, int) and lim > 0:
        return lim
    return 89478485


_PREVIEW_MAX_EDGE = 220
_PREVIEW_MAX_PIXELS = 25_000_000


# Before imagededup (matplotlib); avoids X11 pixmap use from a GUI Tk process.
os.environ.setdefault("MPLBACKEND", "Agg")

from imagededup.methods import AHash, DHash, PHash, WHash

import dedup_session as ds

HASH_CLASSES: Dict[str, Callable[..., Any]] = {
    "phash": PHash,
    "dhash": DHash,
    "ahash": AHash,
    "whash": WHash,
}


def _make_hasher(method: str):
    cls = HASH_CLASSES.get(method.lower(), PHash)
    return cls(verbose=False)


def _hash_one_path(hasher: Any, path: str) -> Optional[str]:
    """Open image, compute hash, release file and pixel buffers; returns hex str only."""
    try:
        if not ds.path_within_os_limits(Path(path)):
            return None
    except (OSError, ValueError, RuntimeError):
        return None
    max_px = _max_pixels_for_hash()
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return None
            if w > max_px or h > max_px:
                return None
            if w * h > max_px:
                return None
            rgb = img.convert("RGB")
            arr = np.array(rgb, dtype=np.uint8, copy=True)
        h = hasher.encode_image(image_array=arr)
        del arr
        return h
    except _HASH_EXC:
        return None


class ImageDuplicateFinderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Image duplicate finder")
        self.geometry("920x700")
        self.minsize(640, 480)

        self._folders: List[str] = []
        self._work_q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self.session_path: Optional[str] = None
        self.image_paths: List[str] = []
        self.encodings: Dict[str, str] = {}
        self.groups: List[Dict[str, Any]] = []
        self.user_by_group: Dict[str, Dict[str, Any]] = {}
        self.duplicates_raw: Optional[Dict[str, List]] = None
        self.phase: str = "idle"
        self.listing_total_images: int = 0
        self.encoding_last_path_index: int = -1
        self.comparison_complete: bool = False
        self.last_checkpoint: str = "scan"
        self.resume_menu_var = tk.StringVar(value="auto")
        self._pulse_prog_active: bool = False
        self._preview_photo_ref: Optional[tk.PhotoImage] = None
        self._preview_current_path: Optional[str] = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Control-s>", lambda e: self._save_progress())
        self.bind("<Control-S>", lambda e: self._save_progress())

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Folders to scan").pack(anchor=tk.W)
        fl = ttk.Frame(top)
        fl.pack(fill=tk.X, pady=4)
        self.folder_list = tk.Listbox(fl, height=5, exportselection=False)
        self.folder_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(fl, command=self.folder_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.folder_list.config(yscrollcommand=sb.set)

        fb = ttk.Frame(top)
        fb.pack(fill=tk.X)
        ttk.Button(fb, text="Add folder…", command=self._add_folder).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(fb, text="Remove selected", command=self._remove_folder).pack(
            side=tk.LEFT
        )

        opts = ttk.Frame(top)
        opts.pack(fill=tk.X, pady=8)
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts, text="Include subfolders", variable=self.recursive_var
        ).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(opts, text="Hash:").pack(side=tk.LEFT)
        self.hash_var = tk.StringVar(value="phash")
        hash_cb = ttk.Combobox(
            opts,
            textvariable=self.hash_var,
            values=("phash", "dhash", "ahash", "whash"),
            width=8,
            state="readonly",
        )
        hash_cb.pack(side=tk.LEFT, padx=4)

        ttk.Label(opts, text="Min similarity %:").pack(side=tk.LEFT, padx=(12, 0))
        self.sim_var = tk.DoubleVar(value=85.0)
        sim = ttk.Scale(
            opts,
            from_=50.0,
            to=100.0,
            variable=self.sim_var,
            orient=tk.HORIZONTAL,
            length=180,
        )
        sim.pack(side=tk.LEFT, padx=4)
        self.sim_label = ttk.Label(opts, text="85")
        self.sim_label.pack(side=tk.LEFT)

        def upd_sim(_e=None) -> None:
            self.sim_label.config(text=f"{self.sim_var.get():.0f}")

        sim.bind("<ButtonRelease-1>", upd_sim)
        sim.bind("<B1-Motion>", upd_sim)
        upd_sim()

        resume_fr = ttk.Frame(top)
        resume_fr.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(resume_fr, text="Go resumes:").pack(side=tk.LEFT)
        resume_cb = ttk.Combobox(
            resume_fr,
            textvariable=self.resume_menu_var,
            values=("auto", "scan", "hash", "compare"),
            width=14,
            state="readonly",
        )
        resume_cb.pack(side=tk.LEFT, padx=6)
        ttk.Label(
            resume_fr,
            text="(auto uses saved checkpoint: scan / hash / compare)",
            foreground="gray",
        ).pack(side=tk.LEFT)

        btns = ttk.Frame(top)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Go", command=self._on_go).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btns, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btns, text="Save progress…", command=self._save_progress_as).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btns, text="Load session…", command=self._load_session_dialog).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btns, text="Recheck folders", command=self._on_recheck).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btns, text="Delete marked duplicates", command=self._delete_marked).pack(
            side=tk.LEFT
        )

        prog = ttk.Frame(top)
        prog.pack(fill=tk.X, pady=4)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(prog, textvariable=self.status_var).pack(anchor=tk.W)
        self.prog = ttk.Progressbar(prog, mode="determinate", length=400)
        self.prog.pack(fill=tk.X, pady=2)

        pan = ttk.Panedwindow(self, orient=tk.VERTICAL)
        pan.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        grp_wrap = ttk.Frame(pan)
        pan.add(grp_wrap, weight=3)
        gsplit = ttk.Panedwindow(grp_wrap, orient=tk.HORIZONTAL)
        gsplit.pack(fill=tk.BOTH, expand=True)

        left_g = ttk.Frame(gsplit, padding=(0, 0, 6, 0))
        gsplit.add(left_g, weight=1)
        ttk.Label(left_g, text="Duplicate groups (pick one)").pack(anchor=tk.W)
        self.group_count_label = ttk.Label(left_g, text="0 duplicate groups")
        self.group_count_label.pack(anchor=tk.W)
        chk_g = ttk.Frame(left_g)
        chk_g.pack(fill=tk.X, pady=(2, 4))
        self.group_checked_var = tk.BooleanVar(value=False)
        self.group_checked_btn = tk.Checkbutton(
            chk_g,
            text="Checked duplicates",
            variable=self.group_checked_var,
            command=self._on_group_checked_toggle,
            anchor=tk.W,
            justify=tk.LEFT,
        )
        self.group_checked_btn.pack(anchor=tk.W)
        self.group_checked_btn.config(state=tk.DISABLED)

        lb_row = ttk.Frame(left_g)
        lb_row.pack(fill=tk.BOTH, expand=True)
        lb_scroll = ttk.Scrollbar(lb_row)
        self.group_listbox = tk.Listbox(
            lb_row,
            height=14,
            exportselection=False,
            yscrollcommand=lb_scroll.set,
        )
        lb_scroll.config(command=self.group_listbox.yview)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.group_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.group_listbox.bind("<<ListboxSelect>>", self._on_group_list_select)

        right_g = ttk.Frame(gsplit, padding=(6, 0, 0, 0))
        gsplit.add(right_g, weight=3)
        rh = ttk.Panedwindow(right_g, orient=tk.HORIZONTAL)
        rh.pack(fill=tk.BOTH, expand=True)

        det_col = ttk.Frame(rh)
        rh.add(det_col, weight=3)
        ttk.Label(
            det_col,
            text="Files in selected group — click a path to preview",
        ).pack(anchor=tk.W)
        det_row = ttk.Frame(det_col)
        det_row.pack(fill=tk.BOTH, expand=True)
        det_scroll = ttk.Scrollbar(det_row)
        self.groups_detail_canvas = tk.Canvas(det_row, highlightthickness=0)
        self.groups_detail_inner = ttk.Frame(self.groups_detail_canvas)
        self.groups_detail_inner.bind(
            "<Configure>",
            lambda e: self.groups_detail_canvas.configure(
                scrollregion=self.groups_detail_canvas.bbox("all")
            ),
        )
        self.groups_detail_win = self.groups_detail_canvas.create_window(
            (0, 0),
            window=self.groups_detail_inner,
            anchor=tk.NW,
        )

        def _detail_cfg(_e=None) -> None:
            self.groups_detail_canvas.itemconfig(
                self.groups_detail_win,
                width=self.groups_detail_canvas.winfo_width(),
            )

        self.groups_detail_canvas.bind("<Configure>", _detail_cfg)
        det_scroll.config(command=self.groups_detail_canvas.yview)
        self.groups_detail_canvas.configure(yscrollcommand=det_scroll.set)
        self.groups_detail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        det_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _lb_wheel(event) -> str:
            if getattr(event, "delta", 0):
                self.group_listbox.yview_scroll(
                    int(-event.delta / 120), "units"
                )
            return "break"

        def _lb_wheel_lin(event) -> str:
            d = -1 if event.num == 4 else 1
            self.group_listbox.yview_scroll(d, "units")
            return "break"

        def _det_wheel(event) -> str:
            if getattr(event, "delta", 0):
                self.groups_detail_canvas.yview_scroll(
                    int(-event.delta / 120), "units"
                )
            return "break"

        def _det_wheel_lin(event) -> str:
            d = -1 if event.num == 4 else 1
            self.groups_detail_canvas.yview_scroll(d, "units")
            return "break"

        self.group_listbox.bind("<MouseWheel>", _lb_wheel)
        self.group_listbox.bind("<Button-4>", _lb_wheel_lin)
        self.group_listbox.bind("<Button-5>", _lb_wheel_lin)
        self.groups_detail_canvas.bind("<MouseWheel>", _det_wheel)
        self.groups_detail_inner.bind("<MouseWheel>", _det_wheel)
        self.groups_detail_canvas.bind("<Button-4>", _det_wheel_lin)
        self.groups_detail_canvas.bind("<Button-5>", _det_wheel_lin)
        self.groups_detail_inner.bind("<Button-4>", _det_wheel_lin)
        self.groups_detail_inner.bind("<Button-5>", _det_wheel_lin)

        prev_col = ttk.Frame(rh, padding=(10, 0, 0, 0))
        rh.add(prev_col, weight=1)
        self.preview_header_label = ttk.Label(prev_col, text="Preview")
        self.preview_header_label.pack(anchor=tk.W)
        self.preview_path_var = tk.StringVar(value="")
        prev_box = ttk.Frame(prev_col, relief=tk.SUNKEN, borderwidth=1)
        prev_box.pack(fill=tk.X, pady=(4, 6))
        self.preview_label = tk.Label(
            prev_box,
            text="(no image)",
            bg="#303030",
            fg="#b0b0b0",
            anchor=tk.CENTER,
            justify=tk.CENTER,
        )
        self.preview_label.pack(padx=4, pady=4)
        ttk.Label(
            prev_col,
            textvariable=self.preview_path_var,
            wraplength=210,
            font=("TkDefaultFont", 8),
        ).pack(anchor=tk.W, pady=(0, 6))
        prev_btns = ttk.Frame(prev_col)
        prev_btns.pack(fill=tk.X)
        ttk.Button(
            prev_btns,
            text="Open in browser",
            command=self._open_preview_in_browser,
        ).pack(fill=tk.X, pady=(0, 3))
        ttk.Button(
            prev_btns,
            text="Open in default viewer",
            command=self._open_preview_default,
        ).pack(fill=tk.X)

        help_fr = ttk.LabelFrame(pan, text="Shortcuts", padding=6)
        pan.add(help_fr, weight=0)
        ttk.Label(
            help_fr,
            text="Ctrl+S: save progress (uses last file if set). Session stores listing, "
            "hash, and compare checkpoints. Duplicate view: click a blue path for preview; "
            "browser opens file:// URL; default viewer uses the OS image app.",
        ).pack(anchor=tk.W)

    def _add_folder(self) -> None:
        d = filedialog.askdirectory(mustexist=True)
        if not d:
            return
        resolved = str(Path(d).expanduser().resolve())
        if resolved not in self._folders:
            self._folders.append(resolved)
            self.folder_list.insert(tk.END, resolved)

    def _remove_folder(self) -> None:
        sel = list(self.folder_list.curselection())
        for i in reversed(sel):
            self.folder_list.delete(i)
            del self._folders[i]

    def _folder_list_values(self) -> List[str]:
        return list(self._folders)

    def _prog_pulse_start(self) -> None:
        if self._pulse_prog_active:
            return
        self.prog.configure(mode="indeterminate")
        self.prog.start(10)
        self._pulse_prog_active = True

    def _prog_pulse_stop(self) -> None:
        if not self._pulse_prog_active:
            return
        try:
            self.prog.stop()
        except tk.TclError:
            pass
        self.prog.configure(mode="determinate")
        self._pulse_prog_active = False

    def _encoding_pass_done(self) -> bool:
        n = len(self.image_paths)
        if n == 0:
            return False
        return self.encoding_last_path_index >= n - 1

    def _checkpoint_for_save(self) -> str:
        if not self.image_paths:
            return "scan"
        if self.comparison_complete and self._encoding_pass_done():
            return "complete"
        if self._encoding_pass_done():
            return "compare"
        return "hash"

    def _resolve_resume_action(self) -> str:
        choice = self.resume_menu_var.get()
        if choice == "scan":
            return "scan"
        if choice == "hash":
            return "hash"
        if choice == "compare":
            return "compare"
        cp = self.last_checkpoint
        if cp == "complete":
            return "compare"
        if cp == "compare":
            return "compare"
        if cp == "hash":
            return "hash"
        return "scan"

    def _on_go(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return
        folders = self._folder_list_values()
        if not folders:
            messagebox.showwarning("Folders", "Add at least one folder.")
            return
        resume_action = self._resolve_resume_action()
        if resume_action == "hash" and not self.image_paths:
            messagebox.showwarning(
                "No file list",
                "Load a session or run with Folder scan first to build the image list.",
            )
            return
        if resume_action == "compare":
            missing = [p for p in self.image_paths if not self.encodings.get(p)]
            if missing:
                messagebox.showerror(
                    "Compare only",
                    f"{len(missing)} image(s) have no hash yet. "
                    f"Run Hashing only or Folder scan first.",
                )
                return
        self._cancel.clear()
        self._work_q = queue.Queue()
        self.phase = "scanning"
        self.status_var.set("Starting…")
        self._prog_pulse_stop()
        self.prog["value"] = 0
        if resume_action == "scan":
            self.encoding_last_path_index = -1
            self.comparison_complete = False
        elif resume_action == "hash":
            self.comparison_complete = False
        snap = {
            "folders": list(self._folder_list_values()),
            "recursive": bool(self.recursive_var.get()),
            "hash_method": str(self.hash_var.get()),
            "min_sim": float(self.sim_var.get()),
            "encodings_start": dict(self.encodings),
            "resume_action": resume_action,
            "saved_paths": list(self.image_paths),
        }
        self.encodings = dict(snap["encodings_start"])
        self._worker = threading.Thread(
            target=self._worker_scan_encode_compare,
            args=(snap,),
            daemon=True,
        )
        self._worker.start()
        self._schedule_poll()

    def _on_stop(self) -> None:
        self._cancel.set()
        self.status_var.set("Stop requested…")

    def _worker_run_find_duplicates(
        self,
        paths: List[str],
        enc: Dict[str, str],
        hasher: Any,
        max_h: int,
    ) -> None:
        n_img = len(enc)
        self._work_q.put(
            {
                "type": "compare_start",
                "n_images": n_img,
            }
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Parameter num_enc_workers has no effect.*",
                category=RuntimeWarning,
            )
            dups = hasher.find_duplicates(
                encoding_map=enc,
                max_distance_threshold=max_h,
                scores=True,
                num_dist_workers=1,
            )
        groups = ds.build_groups(dups, enc, hasher.hamming_distance)
        self._work_q.put(
            {
                "type": "compare_done",
                "duplicates": dups,
                "groups": groups,
                "encodings": enc,
                "image_paths": paths,
                "last_path_index": len(paths) - 1 if paths else -1,
            }
        )

    def _worker_encode_compare_from_paths(
        self, snap: Dict[str, Any], paths: List[str]
    ) -> None:
        method = snap["hash_method"]
        min_sim = snap["min_sim"]
        max_h = ds.min_similarity_to_max_hamming(min_sim)
        hasher = _make_hasher(method)
        start_enc = snap["encodings_start"]
        enc: Dict[str, str] = {}
        for p in paths:
            h = start_enc.get(p)
            if h:
                enc[p] = h
        total = len(paths)
        progress_every = 5
        gc_every = 40
        tick_every = 10
        last_done_index = -1
        for i, path in enumerate(paths):
            if self._cancel.is_set():
                self._work_q.put(
                    {
                        "type": "stopped",
                        "phase": "encoding",
                        "encodings": enc,
                        "image_paths": paths,
                        "last_path_index": last_done_index,
                    }
                )
                return
            enc_delta: Dict[str, str] = {}
            if path in enc and enc[path]:
                if i % 20 == 0 or i == total - 1:
                    self._work_q.put(
                        {
                            "type": "progress",
                            "phase": "encoding",
                            "i": i + 1,
                            "total": total,
                            "path_index": i,
                            "msg": f"Hashing {i + 1}/{total}",
                        }
                    )
            else:
                h = _hash_one_path(hasher, path)
                if h:
                    enc[path] = h
                    enc_delta[path] = h
                if i % progress_every == 0 or i == total - 1:
                    msg: Dict[str, Any] = {
                        "type": "progress",
                        "phase": "encoding",
                        "i": i + 1,
                        "total": total,
                        "path_index": i,
                        "msg": f"Hashing {i + 1}/{total}",
                    }
                    if enc_delta:
                        msg["enc_delta"] = enc_delta
                    self._work_q.put(msg)
            last_done_index = i
            if i % tick_every == 0 or i == total - 1:
                self._work_q.put(
                    {
                        "type": "encode_tick",
                        "path_index": i,
                        "total": total,
                    }
                )
            if i % gc_every == 0 and i > 0:
                gc.collect()

        if self._cancel.is_set():
            self._work_q.put(
                {
                    "type": "stopped",
                    "phase": "encoding",
                    "encodings": enc,
                    "image_paths": paths,
                    "last_path_index": last_done_index,
                }
            )
            return

        self._work_q.put(
            {
                "type": "encoding_done",
                "encodings": enc,
                "image_paths": paths,
                "last_path_index": last_done_index,
            }
        )

        self._worker_run_find_duplicates(paths, enc, hasher, max_h)

    def _worker_compare_only(self, snap: Dict[str, Any]) -> None:
        method = snap["hash_method"]
        min_sim = snap["min_sim"]
        max_h = ds.min_similarity_to_max_hamming(min_sim)
        paths = list(snap["saved_paths"])
        start_enc = snap["encodings_start"]
        enc = {p: start_enc[p] for p in paths if start_enc.get(p)}
        missing = [p for p in paths if not enc.get(p)]
        if missing:
            self._work_q.put(
                {
                    "type": "error",
                    "msg": f"Compare only: {len(missing)} path(s) lack hashes.",
                }
            )
            return
        hasher = _make_hasher(method)
        self._worker_run_find_duplicates(paths, enc, hasher, max_h)

    def _worker_hash_then_compare(self, snap: Dict[str, Any]) -> None:
        paths = list(snap["saved_paths"])
        if not paths:
            self._work_q.put(
                {"type": "error", "msg": "No image paths for hashing."}
            )
            return
        self._work_q.put(
            {"type": "scan_done", "paths": paths, "count": len(paths)}
        )
        if self._cancel.is_set():
            self._work_q.put(
                {
                    "type": "stopped",
                    "phase": "encoding",
                    "encodings": dict(snap["encodings_start"]),
                    "image_paths": paths,
                    "last_path_index": -1,
                }
            )
            return
        self._worker_encode_compare_from_paths(snap, paths)

    def _worker_full_scan(self, snap: Dict[str, Any]) -> None:
        folders = snap["folders"]
        recursive = snap["recursive"]

        self._work_q.put({"type": "status", "msg": "Listing images…"})

        def listing_progress(
            listed: int, folder: str, fi: int, ft: int
        ) -> None:
            self._work_q.put(
                {
                    "type": "listing_progress",
                    "listed": listed,
                    "folder": folder,
                    "folder_i": fi,
                    "folder_n": ft,
                }
            )

        paths = ds.collect_image_paths(
            folders,
            recursive,
            on_progress=listing_progress,
            cancel_check=self._cancel.is_set,
            progress_every=400,
        )
        if self._cancel.is_set():
            self._work_q.put(
                {
                    "type": "stopped",
                    "phase": "listing",
                    "encodings": dict(snap["encodings_start"]),
                    "image_paths": paths,
                    "last_path_index": -1,
                }
            )
            return
        self._work_q.put(
            {
                "type": "scan_done",
                "paths": paths,
                "count": len(paths),
            }
        )
        if not paths:
            self._work_q.put(
                {
                    "type": "compare_done",
                    "duplicates": {},
                    "groups": [],
                    "encodings": {},
                    "image_paths": [],
                    "last_path_index": -1,
                }
            )
            return
        if self._cancel.is_set():
            self._work_q.put(
                {
                    "type": "stopped",
                    "phase": "encoding",
                    "encodings": dict(snap["encodings_start"]),
                    "image_paths": paths,
                    "last_path_index": -1,
                }
            )
            return
        self._worker_encode_compare_from_paths(snap, paths)

    def _worker_scan_encode_compare(self, snap: Dict[str, Any]) -> None:
        try:
            action = snap.get("resume_action", "scan")
            if action == "compare":
                self._worker_compare_only(snap)
            elif action == "hash":
                self._worker_hash_then_compare(snap)
            else:
                self._worker_full_scan(snap)
        except Exception as e:
            self._work_q.put({"type": "error", "msg": str(e)})

    def _worker_recheck(self, snap: Dict[str, Any]) -> None:
        try:
            folders = snap["folders"]
            recursive = snap["recursive"]
            old_paths = set(snap["old_paths"])
            old_enc: Dict[str, str] = dict(snap["old_encodings"])

            self._work_q.put({"type": "status", "msg": "Rechecking folders…"})

            def listing_progress(
                listed: int, folder: str, fi: int, ft: int
            ) -> None:
                self._work_q.put(
                    {
                        "type": "listing_progress",
                        "listed": listed,
                        "folder": folder,
                        "folder_i": fi,
                        "folder_n": ft,
                    }
                )

            fresh = ds.collect_image_paths(
                folders,
                recursive,
                on_progress=listing_progress,
                cancel_check=self._cancel.is_set,
                progress_every=400,
            )
            if self._cancel.is_set():
                self._work_q.put({"type": "recheck_aborted"})
                return
            fresh_set = set(fresh)
            added = len(fresh_set - old_paths)
            removed = len(old_paths - fresh_set)
            new_enc = {p: h for p, h in old_enc.items() if p in fresh_set}
            changed = added > 0 or removed > 0
            self._work_q.put(
                {
                    "type": "recheck_done",
                    "paths": fresh,
                    "encodings": new_enc,
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                }
            )
        except Exception as e:
            self._work_q.put({"type": "error", "msg": str(e)})

    def _on_recheck(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Wait for the current job to finish.")
            return
        folders = self._folder_list_values()
        if not folders:
            messagebox.showwarning("Folders", "Add at least one folder.")
            return
        self._cancel.clear()
        self._work_q = queue.Queue()
        self.phase = "scanning"
        self._prog_pulse_stop()
        self.prog["value"] = 0
        snap = {
            "folders": list(folders),
            "recursive": bool(self.recursive_var.get()),
            "old_paths": list(self.image_paths),
            "old_encodings": dict(self.encodings),
        }
        self._worker = threading.Thread(
            target=self._worker_recheck,
            args=(snap,),
            daemon=True,
        )
        self._worker.start()
        self._schedule_poll()

    def _schedule_poll(self) -> None:
        self._poll_queue()
        if self._worker and self._worker.is_alive():
            self.after(80, self._schedule_poll)
        else:
            self._worker = None

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._work_q.get_nowait()
                self._handle_work_msg(msg)
        except queue.Empty:
            pass

    def _handle_work_msg(self, msg: Dict[str, Any]) -> None:
        t = msg.get("type")
        if t == "status":
            self.status_var.set(msg.get("msg", ""))
        elif t == "compare_start":
            self._prog_pulse_start()
            self.phase = "comparing"
            n = int(msg.get("n_images", 0))
            self.status_var.set(
                f"Comparing {n} images (Hamming search), please wait…"
            )
            self.last_checkpoint = "compare"
        elif t == "listing_progress":
            self._prog_pulse_start()
            listed = int(msg.get("listed", 0))
            fi = int(msg.get("folder_i", 1))
            ft = int(msg.get("folder_n", 1))
            fold = str(msg.get("folder") or "")
            disp = fold if len(fold) <= 70 else f"...{fold[-66:]}"
            self.status_var.set(
                f"Listing images… {listed} found (folder {fi}/{ft}: {disp})"
            )
        elif t == "scan_done":
            self._prog_pulse_stop()
            n = msg.get("count", 0)
            self.image_paths = list(msg.get("paths") or [])
            self.listing_total_images = n
            self.last_checkpoint = "hash"
            self.status_var.set(f"Found {n} images. Hashing…")
            self.prog["maximum"] = max(1, n)
        elif t == "encode_tick":
            pi = int(msg.get("path_index", -1))
            tot = int(msg.get("total", 0))
            if pi >= 0:
                self.encoding_last_path_index = max(
                    self.encoding_last_path_index, pi
                )
            if tot > 0:
                self.prog["maximum"] = max(1, tot)
                self.prog["value"] = pi + 1
        elif t == "progress":
            delta = msg.get("enc_delta")
            if isinstance(delta, dict) and delta:
                self.encodings.update(delta)
            i, tot = msg.get("i", 0), msg.get("total", 1)
            pi = msg.get("path_index")
            if isinstance(pi, int) and pi >= 0:
                self.encoding_last_path_index = max(
                    self.encoding_last_path_index, pi
                )
            self.status_var.set(msg.get("msg", ""))
            self.prog["maximum"] = max(1, tot)
            self.prog["value"] = i
            self.phase = str(msg.get("phase") or "encoding")
        elif t == "stopped":
            self._prog_pulse_stop()
            self.encodings = dict(msg.get("encodings") or {})
            self.image_paths = list(msg.get("image_paths") or [])
            self.phase = str(msg.get("phase") or "encoding")
            lpi = msg.get("last_path_index")
            if isinstance(lpi, int) and lpi >= -1:
                self.encoding_last_path_index = lpi
            ph = self.phase
            self.listing_total_images = len(self.image_paths)
            if ph == "listing":
                self.last_checkpoint = "scan"
                self.status_var.set(
                    "Stopped while listing. Save progress to resume."
                )
            else:
                self.last_checkpoint = "hash"
                tot = len(self.image_paths)
                at = self.encoding_last_path_index + 1
                self.status_var.set(
                    f"Stopped at image {at}/{tot} (index "
                    f"{self.encoding_last_path_index}). Save to resume."
                )
            self.prog["value"] = 0
        elif t == "encoding_done":
            self.encodings = dict(msg.get("encodings") or {})
            self.image_paths = list(msg.get("image_paths") or [])
            self.phase = "encoding_done"
            lpi = msg.get("last_path_index")
            if isinstance(lpi, int) and lpi >= -1:
                self.encoding_last_path_index = lpi
            self.last_checkpoint = "compare"
        elif t == "compare_done":
            self._prog_pulse_stop()
            self.encodings = dict(msg.get("encodings") or {})
            self.image_paths = list(msg.get("image_paths") or [])
            self.duplicates_raw = msg.get("duplicates")
            self.groups = list(msg.get("groups") or [])
            self.user_by_group = ds.merge_user_state(
                {"user_state": list(self.user_by_group.values())},
                self.groups,
            )
            self.phase = "reviewing"
            ngrp = len(self.groups)
            self.listing_total_images = len(self.image_paths)
            lpi = msg.get("last_path_index")
            if isinstance(lpi, int) and lpi >= -1:
                self.encoding_last_path_index = lpi
            if not self.image_paths:
                self.listing_total_images = 0
                self.encoding_last_path_index = -1
                self.last_checkpoint = "scan"
                self.comparison_complete = False
                self.status_var.set("No images found in the selected folders.")
            else:
                nh = sum(1 for p in self.image_paths if self.encodings.get(p))
                self.last_checkpoint = "complete"
                self.comparison_complete = True
                self.status_var.set(
                    f"Done. {ngrp} duplicate group(s). "
                    f"{nh}/{self.listing_total_images} images hashed."
                )
            self.prog["value"] = self.prog["maximum"]
            self.after(1, self._rebuild_groups_ui)
        elif t == "recheck_done":
            self._prog_pulse_stop()
            self.image_paths = list(msg.get("paths") or [])
            self.encodings = dict(msg.get("encodings") or {})
            self.listing_total_images = len(self.image_paths)
            added = int(msg.get("added", 0))
            removed = int(msg.get("removed", 0))
            if msg.get("changed"):
                self.groups = []
                self.user_by_group = {}
                self.duplicates_raw = None
                self.phase = "idle"
                self.encoding_last_path_index = -1
                self.last_checkpoint = "hash"
                self.comparison_complete = False
                self._rebuild_groups_ui()
            self.status_var.set(
                f"Recheck: +{added} new, -{removed} removed; "
                f"{len(self.image_paths)} images listed. "
                f"Press Go to hash new files and compare."
            )
        elif t == "recheck_aborted":
            self._prog_pulse_stop()
            self.status_var.set("Recheck cancelled.")
        elif t == "error":
            self._prog_pulse_stop()
            self.phase = "idle"
            self.status_var.set("Error.")
            messagebox.showerror("Error", msg.get("msg", "Unknown error"))

    def _group_listbox_line(self, idx: int, g: Dict[str, Any]) -> str:
        gid = g["group_id"]
        usb = self.user_by_group.get(gid)
        if not usb:
            usb = ds.default_user_state_for_group(gid, g["paths"])
            self.user_by_group[gid] = usb
        mark = "✓ " if usb.get("checked_duplicates") else "  "
        n = len(g["paths"])
        ms = g.get("min_similarity_pct_in_group", 0)
        line = f"{mark}#{idx + 1}   {n} files   min {ms}%"
        if len(line) > 70:
            line = f"{line[:67]}…"
        return line

    def _sync_group_checked_from_selection(self) -> None:
        if not self.groups:
            self.group_checked_var.set(False)
            self.group_checked_btn.config(state=tk.DISABLED)
            return
        self.group_checked_btn.config(state=tk.NORMAL)
        sel = self.group_listbox.curselection()
        if not sel:
            self.group_checked_var.set(False)
            return
        idx = int(sel[0])
        g = self.groups[idx]
        gid = g["group_id"]
        usb = self.user_by_group.get(gid) or ds.default_user_state_for_group(
            gid, g["paths"]
        )
        self.user_by_group[gid] = usb
        self.group_checked_var.set(bool(usb.get("checked_duplicates")))

    def _on_group_checked_toggle(self) -> None:
        sel = self.group_listbox.curselection()
        if not sel or not self.groups:
            return
        idx = int(sel[0])
        g = self.groups[idx]
        gid = g["group_id"]
        u = self.user_by_group.get(gid)
        if not u:
            u = ds.default_user_state_for_group(gid, g["paths"])
            self.user_by_group[gid] = u
        u["checked_duplicates"] = bool(self.group_checked_var.get())
        self.group_listbox.delete(idx)
        self.group_listbox.insert(idx, self._group_listbox_line(idx, g))
        self.group_listbox.selection_set(idx)
        self.group_listbox.activate(idx)

    def _on_group_list_select(self, _event: Any = None) -> None:
        sel = self.group_listbox.curselection()
        if not sel:
            self._sync_group_checked_from_selection()
            return
        self._show_group_detail(int(sel[0]))
        self._sync_group_checked_from_selection()

    def _clear_preview(self) -> None:
        self._preview_photo_ref = None
        self._preview_current_path = None
        self.preview_path_var.set("")
        self.preview_label.config(image="", text="(no image)")
        self.preview_header_label.configure(text="Preview")

    def _show_preview_for_path(
        self, path: str, image_num: Optional[int] = None
    ) -> None:
        self._preview_current_path = path
        if image_num is not None:
            self.preview_header_label.configure(
                text=f"Preview (Image {image_num})"
            )
        else:
            self.preview_header_label.configure(text="Preview")
        self.preview_path_var.set(path)
        self._preview_photo_ref = None
        pth = Path(path)
        if not pth.is_file():
            self.preview_label.config(image="", text="File missing")
            return
        try:
            try:
                rsz = Image.Resampling.LANCZOS
            except AttributeError:
                rsz = Image.LANCZOS
            with Image.open(path) as im:
                w, h = im.size
                if w <= 0 or h <= 0 or w * h > _PREVIEW_MAX_PIXELS:
                    self.preview_label.config(
                        image="", text="Too large to preview"
                    )
                    return
                rgb = im.convert("RGB")
                rgb.thumbnail((_PREVIEW_MAX_EDGE, _PREVIEW_MAX_EDGE), rsz)
                photo = ImageTk.PhotoImage(rgb)
            self._preview_photo_ref = photo
            self.preview_label.config(image=photo, text="")
        except (OSError, ValueError, tk.TclError):
            self.preview_label.config(image="", text="Could not load")

    def _open_preview_in_browser(self) -> None:
        p = self._preview_current_path
        if not p or not Path(p).is_file():
            messagebox.showinfo("Preview", "Select a file in the group first.")
            return
        uri = Path(p).expanduser().resolve().as_uri()
        webbrowser.open(uri)

    def _open_preview_default(self) -> None:
        p = self._preview_current_path
        if not p or not Path(p).is_file():
            messagebox.showinfo("Preview", "Select a file in the group first.")
            return
        path = str(Path(p).expanduser().resolve())
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except OSError as e:
            messagebox.showerror("Open failed", str(e))

    def _clear_group_detail(self) -> None:
        self._clear_preview()
        for w in self.groups_detail_inner.winfo_children():
            w.destroy()

    def _show_group_detail(
        self, idx: int, preview_path: Optional[str] = None
    ) -> None:
        self._clear_group_detail()
        if idx < 0 or idx >= len(self.groups):
            return
        g = self.groups[idx]
        gid = g["group_id"]
        usb = self.user_by_group.get(gid) or ds.default_user_state_for_group(
            gid, g["paths"]
        )
        self.user_by_group[gid] = usb
        keeper = usb.get("keeper_path") or sorted(g["paths"])[0]
        marked = set(usb.get("marked_duplicate_paths") or [])

        title = (
            f"Group {idx + 1} — min similarity "
            f"{g.get('min_similarity_pct_in_group', 0)}% "
            f"({len(g['paths'])} files)"
        )
        lf = ttk.LabelFrame(self.groups_detail_inner, text=title, padding=6)
        lf.pack(fill=tk.BOTH, expand=True, padx=2, pady=4)

        keeper_var = tk.StringVar(value=keeper)

        def trace_keeper(*_a: Any, _gid: str = gid, var: tk.StringVar = keeper_var) -> None:
            self._sync_keeper(_gid, var.get())

        keeper_var.trace_add("write", trace_keeper)

        sims: Dict[str, float] = g.get("similarity_to_ref") or {}
        for img_num, p in enumerate(g["paths"], start=1):
            row = ttk.Frame(lf)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{img_num}.", width=4, anchor=tk.E).pack(
                side=tk.LEFT
            )
            spct = sims.get(p, 0.0)
            ttk.Label(row, text=f"{spct:.1f}%", width=8).pack(side=tk.LEFT)
            short = p if len(p) < 50 else f"…{p[-46:]}"
            path_lbl = tk.Label(
                row,
                text=short,
                anchor=tk.W,
                cursor="hand2",
                fg="#1565c0",
            )
            path_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def on_path_click(
                _e: Any, _p: str = p, _n: int = img_num
            ) -> None:
                self._show_preview_for_path(_p, _n)

            path_lbl.bind("<Button-1>", on_path_click)
            ttk.Radiobutton(
                row,
                text="Keep",
                variable=keeper_var,
                value=p,
            ).pack(side=tk.LEFT, padx=4)
            mv = tk.BooleanVar(value=p in marked)

            def on_mark(
                *_x: Any,
                _gid: str = gid,
                _path: str = p,
                _mv: tk.BooleanVar = mv,
            ) -> None:
                uu = self.user_by_group.get(_gid)
                if uu and _path == uu.get("keeper_path"):
                    _mv.set(False)
                    return
                self._sync_mark(_gid, _path, _mv.get())

            cb = ttk.Checkbutton(
                row, text="Mark duplicate", variable=mv, command=on_mark
            )
            cb.pack(side=tk.LEFT)

        plist = g["paths"]
        if preview_path and preview_path in plist:
            self._show_preview_for_path(
                preview_path, plist.index(preview_path) + 1
            )
        elif plist:
            self._show_preview_for_path(plist[0], 1)

        self.groups_detail_canvas.update_idletasks()
        self.groups_detail_canvas.configure(
            scrollregion=self.groups_detail_canvas.bbox("all")
        )

    def _rebuild_groups_ui(self) -> None:
        ngrp = len(self.groups)
        self.group_count_label.configure(
            text=(
                "1 duplicate group"
                if ngrp == 1
                else f"{ngrp} duplicate groups"
            )
        )
        self.group_listbox.delete(0, tk.END)
        self._clear_group_detail()
        if not self.groups:
            self._clear_preview()
            ttk.Label(
                self.groups_detail_inner,
                text="No duplicate groups at this similarity threshold.",
            ).pack(anchor=tk.W, pady=8)
            self.groups_detail_canvas.update_idletasks()
            self.groups_detail_canvas.configure(
                scrollregion=self.groups_detail_canvas.bbox("all")
            )
            self._sync_group_checked_from_selection()
            return

        for idx, g in enumerate(self.groups):
            self.group_listbox.insert(tk.END, self._group_listbox_line(idx, g))

        self.group_listbox.selection_set(0)
        self.group_listbox.activate(0)
        self._show_group_detail(0)
        self._sync_group_checked_from_selection()

    def _sync_keeper(self, gid: str, keeper_path: str) -> None:
        u = self.user_by_group.get(gid)
        if not u:
            return
        old = u.get("keeper_path")
        u["keeper_path"] = keeper_path
        md = [p for p in u.get("marked_duplicate_paths", []) if p != keeper_path]
        u["marked_duplicate_paths"] = md
        if old != keeper_path:
            sel = self.group_listbox.curselection()
            if sel:
                self._show_group_detail(
                    int(sel[0]), preview_path=keeper_path
                )

    def _sync_mark(self, gid: str, path: str, marked: bool) -> None:
        u = self.user_by_group.get(gid)
        if not u:
            return
        keeper = u.get("keeper_path", "")
        md = list(u.get("marked_duplicate_paths", []))
        if marked and path != keeper:
            if path not in md:
                md.append(path)
        else:
            md = [p for p in md if p != path]
        u["marked_duplicate_paths"] = md

    def _collect_payload(self) -> Dict[str, Any]:
        folders = self._folder_list_values()
        paths = list(self.image_paths)
        enc = dict(self.encodings)
        nh = sum(1 for p in paths if enc.get(p))
        listed = self.listing_total_images or len(paths)
        return ds.session_to_dict(
            folders=folders,
            recursive=self.recursive_var.get(),
            hash_method=self.hash_var.get(),
            min_similarity_pct=float(self.sim_var.get()),
            phase=self.phase,
            image_paths=paths,
            encodings=enc,
            groups=list(self.groups),
            user_state=self.user_by_group,
            listing_total_images=listed,
            encoding_last_path_index=self.encoding_last_path_index,
            images_hashed_count=nh,
            last_checkpoint=self._checkpoint_for_save(),
            comparison_complete=self.comparison_complete,
            go_resume_choice=str(self.resume_menu_var.get()),
        )

    def _save_progress(self) -> None:
        if not self.session_path:
            self._save_progress_as()
            return
        try:
            pl = self._collect_payload()
            ds.save_session(self.session_path, pl)
            self.status_var.set(
                f"Saved {self.session_path} "
                f"({pl['listing_total_images']} listed, "
                f"{pl['images_hashed_count']} hashed, "
                f"index {pl['encoding_last_path_index']}, "
                f"checkpoint {pl['last_checkpoint']})"
            )
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    def _save_progress_as(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON session", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.session_path = path
        try:
            pl = self._collect_payload()
            ds.save_session(path, pl)
            self.status_var.set(
                f"Saved {path} ({pl['listing_total_images']} listed, "
                f"{pl['images_hashed_count']} hashed, "
                f"index {pl['encoding_last_path_index']}, "
                f"checkpoint {pl['last_checkpoint']})"
            )
        except OSError as e:
            messagebox.showerror("Save failed", str(e))

    def _load_session_dialog(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON session", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            data = ds.load_session(path)
        except json.JSONDecodeError as e:
            messagebox.showerror("Load failed", str(e))
            return
        except (OSError, ValueError) as e:
            messagebox.showerror("Load failed", str(e))
            return

        self.session_path = path
        self._folders = list(data["folders"])
        self.folder_list.delete(0, tk.END)
        for f in self._folders:
            self.folder_list.insert(tk.END, f)
        self.recursive_var.set(bool(data["recursive"]))
        self.hash_var.set(str(data["hash_method"]))
        self.sim_var.set(float(data["min_similarity_pct"]))
        self.sim_label.config(text=f"{self.sim_var.get():.0f}")
        self.phase = str(data["phase"])
        self.image_paths = list(data["image_paths"])
        self.encodings = dict(data["encodings"])
        self.groups = list(data["groups"])
        raw_us = data.get("user_state")
        if isinstance(raw_us, list):
            self.user_by_group = {u["group_id"]: u for u in raw_us if u.get("group_id")}
        elif isinstance(raw_us, dict):
            self.user_by_group = dict(raw_us)
        else:
            self.user_by_group = {}
        self.user_by_group = ds.merge_user_state(
            {"user_state": list(self.user_by_group.values())},
            self.groups,
        )
        self.duplicates_raw = data.get("duplicates_raw")
        self.listing_total_images = int(data["listing_total_images"])
        self.encoding_last_path_index = int(data["encoding_last_path_index"])
        self.last_checkpoint = str(data.get("last_checkpoint") or "scan")
        self.comparison_complete = bool(data.get("comparison_complete"))
        self.resume_menu_var.set(str(data.get("go_resume_choice") or "auto"))
        self._rebuild_groups_ui()
        nh = sum(1 for p in self.image_paths if self.encodings.get(p))
        tot = max(self.listing_total_images, len(self.image_paths))
        self.status_var.set(
            f"Loaded: {tot} listed, {nh} hashed, "
            f"encoding index {self.encoding_last_path_index}, "
            f"checkpoint {self.last_checkpoint}, "
            f"compare done {self.comparison_complete}, "
            f"{len(self.groups)} group(s), phase {self.phase}"
        )

    def _delete_marked(self) -> None:
        to_delete: List[str] = []
        for gid, u in self.user_by_group.items():
            keeper = u.get("keeper_path", "")
            for p in u.get("marked_duplicate_paths", []):
                if p and p != keeper:
                    to_delete.append(p)
        to_delete = list(dict.fromkeys(to_delete))
        if not to_delete:
            messagebox.showinfo("Delete", "No paths marked as duplicates.")
            return
        if not messagebox.askyesno(
            "Confirm delete",
            f"Permanently delete {len(to_delete)} file(s)? This cannot be undone.",
        ):
            return
        ok, bad = [], []
        for p in to_delete:
            try:
                Path(p).unlink(missing_ok=True)
                ok.append(p)
            except OSError:
                bad.append(p)
        for p in ok:
            self.encodings.pop(p, None)
            if p in self.image_paths:
                self.image_paths.remove(p)
        new_groups: List[Dict[str, Any]] = []
        for g in self.groups:
            paths = [x for x in g["paths"] if x not in ok]
            if len(paths) > 1:
                ng = dict(g)
                ng["paths"] = paths
                sims = {k: v for k, v in (g.get("similarity_to_ref") or {}).items() if k in paths}
                ng["similarity_to_ref"] = sims
                ng["min_similarity_pct_in_group"] = round(
                    min(sims.values()) if sims else 0.0, 2
                )
                ng["group_id"] = ds.stable_group_id(paths)
                new_groups.append(ng)
        self.groups = new_groups
        self.user_by_group = ds.merge_user_state(
            {"user_state": list(self.user_by_group.values())},
            self.groups,
        )
        self.listing_total_images = len(self.image_paths)
        self._rebuild_groups_ui()
        msg = f"Deleted {len(ok)} file(s)."
        if bad:
            msg += f" Failed: {len(bad)}."
        self.status_var.set(msg)
        if bad:
            messagebox.showwarning("Some deletes failed", "\n".join(bad[:20]))

    def _on_close(self) -> None:
        self._cancel.set()
        self.destroy()

    def destroy(self) -> None:
        super().destroy()


def main() -> None:
    app = ImageDuplicateFinderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
