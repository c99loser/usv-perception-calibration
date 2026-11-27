"""Tk desktop interface for continuous LiDAR point-cloud annotation."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import json
import math
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

from .core import (
    CLASSES,
    ViewTransform,
    box_bounds,
    box_handles,
    discover_episodes,
    load_annotation_document,
    move_box,
    point_in_box,
    propagate_boxes,
    read_pcd,
    resize_box,
)


COLORS = {
    "background": "#8b9aaa",
    "ambiguous": "#d38cff",
    "pillar": "#ffb703",
    "buoy": "#fb7185",
    "block": "#a3e635",
    "boat": "#60a5fa",
}


class AnnotationApp:
    def __init__(self, root_dir: Path, out_dir: Path, *, allow_holdout: bool = False) -> None:
        self.root_dir = Path(root_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = discover_episodes(self.root_dir, allow_holdout=allow_holdout)
        if not self.episodes:
            raise ValueError(
                f"在 {self.root_dir} 下没有找到可标注的训练 Episode。"
                "请选择包含 frame_pointcloud_map.json 的 Episode 上级目录。"
            )
        self._write_manifest(any("holdout" in path.name.lower() for path in self.episodes))

        self.episode = self.episodes[0]
        self.frame = 0
        self.frame_map: dict[str, str] = {}
        self.cloud = np.empty((0, 3), dtype=np.float32)
        self.cloud_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self.annotations: dict = {}
        self.selected_id: str | None = None
        self.mode = "select"
        self.drag: dict | None = None
        self.preview_rect: int | None = None
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.dirty = False
        self.autosave_job: str | None = None
        self.playing = False
        self.image_photo = None
        self.point_photo = None
        self.view = ViewTransform()

        self.win = tk.Tk()
        self.win.title("点云连续帧盲标工作台 v2")
        self.win.geometry("1680x980")
        self.win.minsize(1180, 720)
        self._build_ui()
        self._bind_shortcuts()
        self.load_episode(reset_frame=True)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

    def _write_manifest(self, holdout_opened: bool) -> None:
        path = self.out_dir / "session_manifest.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "purpose": "blind raw finite-XYZ LiDAR episode annotation",
            "annotation_format": "local_annotation_sidecar_v2",
            "rules": [
                "Annotations use one LiDAR, one frame, and sensor-local XYZ.",
                "Related images are context only; never infer a 3D location from image pixels.",
                "Propagated boxes are suggestions and must be reviewed.",
                "Do not inspect the holdout set before rules, models, and thresholds are frozen.",
            ],
        }
        data["source_training_project"] = str(self.root_dir)
        data["holdout_opened"] = bool(data.get("holdout_opened", False) or holdout_opened)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_ui(self) -> None:
        style = ttk.Style(self.win)
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        top = ttk.Frame(self.win, padding=(8, 6))
        top.pack(fill="x")
        self.episode_var = tk.StringVar(value=self.episode.name)
        ttk.Label(top, text="训练 Episode").pack(side="left")
        episode_menu = ttk.Combobox(
            top, textvariable=self.episode_var, values=[p.name for p in self.episodes],
            state="readonly", width=38,
        )
        episode_menu.pack(side="left", padx=(5, 12))
        episode_menu.bind("<<ComboboxSelected>>", self.change_episode)
        ttk.Button(top, text="◀", width=4, command=lambda: self.move_frame(-1)).pack(side="left")
        self.play_button = ttk.Button(top, text="▶ 播放", width=8, command=self.toggle_play)
        self.play_button.pack(side="left", padx=3)
        ttk.Button(top, text="▶", width=4, command=lambda: self.move_frame(1)).pack(side="left")
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(top, from_=0, to=1, variable=self.frame_var, command=self._slider_changed)
        self.frame_scale.pack(side="left", fill="x", expand=True, padx=10)
        self.frame_label = ttk.Label(top, text="0/0", width=11)
        self.frame_label.pack(side="left")
        self.save_button = ttk.Button(top, text="保存 Ctrl+S", command=self.save, style="Accent.TButton")
        self.save_button.pack(side="left", padx=(8, 0))

        toolbar = ttk.Frame(self.win, padding=(8, 2))
        toolbar.pack(fill="x")
        self.mode_var = tk.StringVar(value="select")
        for text, value in (("选择/调整 V", "select"), ("新建框 N", "draw"), ("平移视图 H", "pan")):
            ttk.Radiobutton(toolbar, text=text, value=value, variable=self.mode_var, command=self._mode_changed).pack(side="left", padx=3)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="放大 +", command=lambda: self.zoom_center(1.25)).pack(side="left")
        ttk.Button(toolbar, text="缩小 −", command=lambda: self.zoom_center(0.8)).pack(side="left", padx=3)
        ttk.Button(toolbar, text="适配全图 1", command=self.fit_view).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="复制上一帧", command=lambda: self.propagate_from_previous(False)).pack(side="left")
        ttk.Button(toolbar, text="跟随到下一帧", command=self.follow_to_next, style="Accent.TButton").pack(side="left", padx=3)
        self.auto_follow_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="前进时自动跟随", variable=self.auto_follow_var).pack(side="left", padx=5)
        ttk.Button(toolbar, text="撤销 Ctrl+Z", command=self.undo).pack(side="right")
        ttk.Button(toolbar, text="重做 Ctrl+Y", command=self.redo).pack(side="right", padx=3)

        form = ttk.Frame(self.win, padding=(8, 4))
        form.pack(fill="x")
        self.class_var = tk.StringVar(value="pillar")
        self.track_var = tk.StringVar(value="track_001")
        self.static_var = tk.StringVar(value="yes")
        self.conf_var = tk.StringVar(value="medium")
        self.z_var = tk.StringVar(value="0.0")
        self.height_var = tk.StringVar(value="2.0")
        fields = (
            ("类别", self.class_var, CLASSES, 12),
            ("track_id", self.track_var, None, 17),
            ("静态性", self.static_var, ("yes", "no", "uncertain"), 11),
            ("置信度", self.conf_var, ("high", "medium", "low"), 10),
            ("中心 Z(m)", self.z_var, None, 8),
            ("高度 Z(m)", self.height_var, None, 8),
        )
        for label, var, values, width in fields:
            ttk.Label(form, text=label).pack(side="left", padx=(6, 2))
            if values:
                ttk.Combobox(form, textvariable=var, values=values, state="readonly", width=width).pack(side="left")
            else:
                ttk.Entry(form, textvariable=var, width=width).pack(side="left")
        ttk.Button(form, text="应用到选中框", command=self.apply_fields).pack(side="left", padx=10)
        self.review_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="已复核传播框", variable=self.review_var, command=self.mark_reviewed).pack(side="left")

        pane = ttk.Panedwindow(self.win, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        left = ttk.Frame(pane)
        right = ttk.Frame(pane, width=490)
        pane.add(left, weight=4)
        pane.add(right, weight=2)
        self.canvas = tk.Canvas(left, bg="#06101b", highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_press)
        self.canvas.bind("<B2-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-2>", lambda _e: setattr(self, "drag", None))
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        self.image_label = ttk.Label(right, anchor="center")
        self.image_label.pack(fill="x")
        ttk.Label(
            right,
            text="同步鱼眼图仅供场景理解，不得按像素位置推断 3D 对应。",
            foreground="#a14d00", wraplength=470,
        ).pack(fill="x", pady=(2, 5))
        self.objects = tk.Listbox(right, width=62, height=24, exportselection=False)
        self.objects.pack(fill="both", expand=True)
        self.objects.bind("<<ListboxSelect>>", self.select_from_list)
        object_buttons = ttk.Frame(right)
        object_buttons.pack(fill="x", pady=4)
        ttk.Button(object_buttons, text="删除 Del", command=self.delete_selected).pack(side="left", fill="x", expand=True)
        ttk.Button(object_buttons, text="复制 Ctrl+C", command=self.copy_selected).pack(side="left", fill="x", expand=True, padx=3)
        ttk.Button(object_buttons, text="粘贴 Ctrl+V", command=self.paste_box).pack(side="left", fill="x", expand=True)
        self.help_label = ttk.Label(
            right,
            text="滚轮：以鼠标为中心缩放｜中键：平移｜选择模式：拖框移动、拖角点缩放｜新建模式：空白处拖拽画框",
            wraplength=470, foreground="#44566c",
        )
        self.help_label.pack(fill="x", pady=3)
        self.status = ttk.Label(self.win, text="", anchor="w", relief="sunken", padding=(7, 3))
        self.status.pack(fill="x", side="bottom")

    def _bind_shortcuts(self) -> None:
        self.win.bind("<Control-s>", lambda _e: self.save())
        self.win.bind("<Control-z>", lambda _e: self.undo())
        self.win.bind("<Control-y>", lambda _e: self.redo())
        self.win.bind("<Control-c>", lambda _e: self.copy_selected())
        self.win.bind("<Control-v>", lambda _e: self.paste_box())
        self.win.bind("<Delete>", lambda _e: self.delete_selected())
        self.win.bind("<Left>", lambda _e: self._shortcut_move(-1))
        self.win.bind("<Right>", lambda _e: self._shortcut_move(1))
        self.win.bind("<KeyPress-v>", lambda _e: self.set_mode("select"))
        self.win.bind("<KeyPress-n>", lambda _e: self.set_mode("draw"))
        self.win.bind("<KeyPress-h>", lambda _e: self.set_mode("pan"))
        self.win.bind("<KeyPress-1>", lambda _e: self.fit_view())
        self.win.bind("<space>", lambda _e: self.toggle_play())

    def _shortcut_move(self, delta: int) -> None:
        if self.win.focus_get() and self.win.focus_get().winfo_class() in {"Entry", "TEntry", "TCombobox"}:
            return
        self.move_frame(delta)

    def annotation_path(self) -> Path:
        return self.out_dir / "annotations" / f"{self.episode.name}.json"

    def _load_map(self) -> None:
        raw = json.loads((self.episode / "frame_pointcloud_map.json").read_text(encoding="utf-8"))
        if isinstance(raw, list):
            self.frame_map = {str(i): value for i, value in enumerate(raw)}
        else:
            self.frame_map = {str(k): v for k, v in raw.items()}
        expected = {str(i) for i in range(len(self.frame_map))}
        if set(self.frame_map) != expected:
            raise ValueError(f"帧索引必须从 0 连续编号: {self.episode}")

    def load_episode(self, *, reset_frame: bool) -> None:
        if self.annotations:
            self.save()
        self._load_map()
        self.annotations = load_annotation_document(self.annotation_path(), self.episode.name)
        if reset_frame:
            self.frame = 0
        self.frame = max(0, min(self.frame, len(self.frame_map) - 1))
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.selected_id = None
        self.frame_scale.configure(to=max(0, len(self.frame_map) - 1))
        self.fit_view(redraw=False)
        self.load_frame()

    def change_episode(self, _event=None) -> None:
        selected = next(p for p in self.episodes if p.name == self.episode_var.get())
        if selected == self.episode:
            return
        self.episode = selected
        self.load_episode(reset_frame=True)

    def _cloud_for(self, frame: int) -> np.ndarray:
        key = (self.episode.name, frame)
        if key in self.cloud_cache:
            self.cloud_cache.move_to_end(key)
            return self.cloud_cache[key]
        path = self.episode / "pointcloud" / self.frame_map[str(frame)]
        cloud = read_pcd(path)
        self.cloud_cache[key] = cloud
        while len(self.cloud_cache) > 5:
            self.cloud_cache.popitem(last=False)
        return cloud

    def load_frame(self) -> None:
        self.cloud = self._cloud_for(self.frame)
        self.frame_var.set(self.frame)
        self.frame_label.configure(text=f"{self.frame + 1}/{len(self.frame_map)}")
        self.selected_id = None
        self.load_related_image()
        self.refresh_list()
        self.redraw_canvas()
        self._update_status()

    def move_frame(self, delta: int, *, use_auto: bool = True) -> None:
        target = max(0, min(len(self.frame_map) - 1, self.frame + delta))
        if target == self.frame:
            return
        if delta == 1 and use_auto and self.auto_follow_var.get():
            source_frame = self.frame
            source_boxes = deepcopy(self.frame_boxes())
            source_cloud = self.cloud
            self.frame = target
            current = self.annotations["frames"].get(str(target), [])
            propagated = propagate_boxes(source_cloud, self._cloud_for(target), source_boxes, current, follow=True)
            if propagated != current:
                self._push_undo()
                self.annotations["frames"][str(target)] = propagated
                self._mark_dirty()
            self.load_frame()
            return
        self.frame = target
        self.load_frame()

    def _slider_changed(self, value: str) -> None:
        target = int(round(float(value)))
        if target != self.frame and 0 <= target < len(self.frame_map):
            self.frame = target
            self.load_frame()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="⏸ 暂停" if self.playing else "▶ 播放")
        if self.playing:
            self._play_tick()

    def _play_tick(self) -> None:
        if not self.playing:
            return
        if self.frame >= len(self.frame_map) - 1:
            self.toggle_play()
            return
        self.move_frame(1, use_auto=False)
        self.win.after(450, self._play_tick)

    def frame_boxes(self) -> list[dict]:
        return self.annotations["frames"].setdefault(str(self.frame), [])

    def selected_box(self) -> dict | None:
        return next((b for b in self.frame_boxes() if b.get("id") == self.selected_id), None)

    def _replace_selected(self, updated: dict) -> None:
        for index, box in enumerate(self.frame_boxes()):
            if box.get("id") == self.selected_id:
                self.frame_boxes()[index] = updated
                return

    def set_mode(self, mode: str) -> None:
        self.mode_var.set(mode)
        self._mode_changed()

    def _mode_changed(self) -> None:
        self.mode = self.mode_var.get()
        self.canvas.configure(cursor={"select": "arrow", "draw": "crosshair", "pan": "fleur"}[self.mode])

    def fit_view(self, redraw: bool = True) -> None:
        self.view.fit_range(max(self.canvas.winfo_width(), 900), max(self.canvas.winfo_height(), 700), 60.0)
        if redraw:
            self.redraw_canvas()

    def zoom_center(self, factor: float) -> None:
        w, h = self._canvas_size()
        self.view.zoom_at(factor, w / 2, h / 2, w, h)
        self.redraw_canvas()

    def on_mousewheel(self, event) -> None:
        factor = 1.18 if event.delta > 0 else 1 / 1.18
        w, h = self._canvas_size()
        self.view.zoom_at(factor, event.x, event.y, w, h)
        self.redraw_canvas()

    def _canvas_size(self) -> tuple[int, int]:
        return max(self.canvas.winfo_width(), 20), max(self.canvas.winfo_height(), 20)

    def _world(self, px: float, py: float) -> tuple[float, float]:
        w, h = self._canvas_size()
        return self.view.canvas_to_world(px, py, w, h)

    def on_pan_press(self, event) -> None:
        self.drag = {"kind": "pan", "last": (event.x, event.y)}

    def on_pan_motion(self, event) -> None:
        if not self.drag or self.drag.get("kind") != "pan":
            return
        lx, ly = self.drag["last"]
        self.view.pan_pixels(event.x - lx, event.y - ly)
        self.drag["last"] = (event.x, event.y)
        self.redraw_canvas()

    def _hit_test(self, px: float, py: float) -> tuple[str | None, str | None]:
        wx, wy = self._world(px, py)
        selected = self.selected_box()
        if selected:
            threshold = 8.0 / self.view.pixels_per_meter
            for name, (hx, hy) in box_handles(selected).items():
                if math.hypot(wx - hx, wy - hy) <= threshold:
                    return selected.get("id"), name
        for box in reversed(self.frame_boxes()):
            if point_in_box(wx, wy, box):
                return box.get("id"), None
        return None, None

    def on_left_press(self, event) -> None:
        self.canvas.focus_set()
        if self.mode == "pan":
            self.on_pan_press(event)
            return
        if self.mode == "draw":
            self.drag = {"kind": "draw", "start_px": (event.x, event.y)}
            return
        box_id, handle = self._hit_test(event.x, event.y)
        self.selected_id = box_id
        self.refresh_list()
        if not box_id:
            self.redraw_canvas()
            return
        box = deepcopy(self.selected_box())
        self._load_fields(box)
        self.drag = {
            "kind": "resize" if handle else "move",
            "handle": handle,
            "start_world": self._world(event.x, event.y),
            "original": box,
        }
        self.redraw_canvas()

    def on_left_motion(self, event) -> None:
        if not self.drag:
            return
        kind = self.drag["kind"]
        if kind == "pan":
            self.on_pan_motion(event)
            return
        if kind == "draw":
            if self.preview_rect:
                self.canvas.delete(self.preview_rect)
            x0, y0 = self.drag["start_px"]
            self.preview_rect = self.canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#e8f3ff", width=2, dash=(5, 3)
            )
            return
        wx, wy = self._world(event.x, event.y)
        original = self.drag["original"]
        if kind == "move":
            sx, sy = self.drag["start_world"]
            updated = move_box(original, wx - sx, wy - sy)
        else:
            updated = resize_box(original, self.drag["handle"], wx, wy)
        self._replace_selected(updated)
        self.redraw_canvas()

    def on_left_release(self, event) -> None:
        if not self.drag:
            return
        kind = self.drag["kind"]
        if kind == "pan":
            self.drag = None
            return
        if kind == "draw":
            x0, y0 = self.drag["start_px"]
            self.drag = None
            if self.preview_rect:
                self.canvas.delete(self.preview_rect)
                self.preview_rect = None
            if abs(event.x - x0) >= 5 and abs(event.y - y0) >= 5:
                self._create_box(x0, y0, event.x, event.y)
            return
        original = self.drag["original"]
        changed = self.selected_box() != original
        self.drag = None
        if changed:
            # During drag the visual document was updated already.  Store an
            # undo snapshot with the original box restored.
            current = deepcopy(self.annotations)
            self._replace_selected(original)
            self.undo_stack.append(deepcopy(self.annotations))
            self.annotations = current
            self.redo_stack.clear()
            self._mark_dirty()
            self.refresh_list()
            self.redraw_canvas()

    def _create_box(self, px0: float, py0: float, px1: float, py1: float) -> None:
        try:
            z = float(self.z_var.get())
            height = float(self.height_var.get())
            if height <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Z 参数错误", "中心 Z 必须是数字，高度必须是正数。")
            return
        x0, y0 = self._world(px0, py0)
        x1, y1 = self._world(px1, py1)
        self._push_undo()
        box = {
            "id": str(uuid.uuid4()),
            "class_name": self.class_var.get(),
            "track_id": self.track_var.get().strip(),
            "static_for_calibration": self.static_var.get(),
            "reviewer_confidence": self.conf_var.get(),
            "center_xyz": [round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3), z],
            "dimensions_xyz": [round(abs(x1 - x0), 3), round(abs(y1 - y0), 3), height],
            "yaw_rad": 0.0,
        }
        self.frame_boxes().append(box)
        self.selected_id = box["id"]
        self._mark_dirty()
        self.refresh_list()
        self.redraw_canvas()

    def _push_undo(self) -> None:
        self.undo_stack.append(deepcopy(self.annotations))
        self.undo_stack = self.undo_stack[-60:]
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(deepcopy(self.annotations))
        self.annotations = self.undo_stack.pop()
        self.selected_id = None
        self._mark_dirty()
        self.refresh_list()
        self.redraw_canvas()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(deepcopy(self.annotations))
        self.annotations = self.redo_stack.pop()
        self.selected_id = None
        self._mark_dirty()
        self.refresh_list()
        self.redraw_canvas()

    def delete_selected(self) -> None:
        if not self.selected_id:
            return
        self._push_undo()
        self.annotations["frames"][str(self.frame)] = [b for b in self.frame_boxes() if b.get("id") != self.selected_id]
        self.selected_id = None
        self._mark_dirty()
        self.refresh_list()
        self.redraw_canvas()

    def copy_selected(self) -> None:
        box = self.selected_box()
        if box:
            self.clipboard_box = deepcopy(box)

    def paste_box(self) -> None:
        if not hasattr(self, "clipboard_box"):
            return
        self._push_undo()
        box = move_box(self.clipboard_box, 0.3, -0.3)
        box["id"] = str(uuid.uuid4())
        box.pop("propagation", None)
        box.pop("source_box_id", None)
        self.frame_boxes().append(box)
        self.selected_id = box["id"]
        self._mark_dirty()
        self.refresh_list()
        self.redraw_canvas()

    def apply_fields(self) -> None:
        box = self.selected_box()
        if not box:
            return
        try:
            z, height = float(self.z_var.get()), float(self.height_var.get())
            if height <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "中心 Z 必须是数字，高度必须是正数。")
            return
        self._push_undo()
        box.update({
            "class_name": self.class_var.get(),
            "track_id": self.track_var.get().strip(),
            "static_for_calibration": self.static_var.get(),
            "reviewer_confidence": self.conf_var.get(),
        })
        box["center_xyz"][2] = z
        box["dimensions_xyz"][2] = height
        if box.get("propagation"):
            box["propagation"]["reviewed"] = bool(self.review_var.get())
        self._mark_dirty()
        self.refresh_list()
        self.redraw_canvas()

    def mark_reviewed(self) -> None:
        box = self.selected_box()
        if box and box.get("propagation"):
            self._push_undo()
            box["propagation"]["reviewed"] = bool(self.review_var.get())
            self._mark_dirty()
            self.refresh_list()
            self.redraw_canvas()

    def _load_fields(self, box: dict) -> None:
        self.class_var.set(box.get("class_name", "ambiguous"))
        self.track_var.set(box.get("track_id", ""))
        self.static_var.set(box.get("static_for_calibration", "uncertain"))
        self.conf_var.set(box.get("reviewer_confidence", "low"))
        self.z_var.set(str(box.get("center_xyz", [0, 0, 0])[2]))
        self.height_var.set(str(box.get("dimensions_xyz", [1, 1, 1])[2]))
        self.review_var.set(bool(box.get("propagation", {}).get("reviewed", False)))

    def propagate_from_previous(self, follow: bool = True) -> None:
        if self.frame == 0:
            messagebox.showinfo("没有上一帧", "当前已经是 Episode 第一帧。")
            return
        previous_boxes = self.annotations["frames"].get(str(self.frame - 1), [])
        if not previous_boxes:
            messagebox.showinfo("上一帧无框", "请先在上一帧创建标定框。")
            return
        before = deepcopy(self.frame_boxes())
        after = propagate_boxes(
            self._cloud_for(self.frame - 1), self.cloud, previous_boxes, before, follow=follow
        )
        if after == before:
            messagebox.showinfo("无需复制", "本帧已包含上一帧的全部 track_id，没有覆盖现有框。")
            return
        self._push_undo()
        self.annotations["frames"][str(self.frame)] = after
        self._mark_dirty()
        self.refresh_list()
        self.redraw_canvas()

    def follow_to_next(self) -> None:
        if self.frame >= len(self.frame_map) - 1:
            messagebox.showinfo("最后一帧", "当前已经是 Episode 最后一帧。")
            return
        if not self.frame_boxes():
            messagebox.showinfo("当前帧无框", "请先在当前帧创建或复制标定框。")
            return
        source_frame = self.frame
        source_boxes = deepcopy(self.frame_boxes())
        target = source_frame + 1
        current = self.annotations["frames"].get(str(target), [])
        after = propagate_boxes(self.cloud, self._cloud_for(target), source_boxes, current, follow=True)
        if after != current:
            self._push_undo()
            self.annotations["frames"][str(target)] = after
            self._mark_dirty()
        self.frame = target
        self.load_frame()

    def select_from_list(self, _event=None) -> None:
        selection = self.objects.curselection()
        if not selection:
            return
        boxes = self.frame_boxes()
        if selection[0] >= len(boxes):
            return
        self.selected_id = boxes[selection[0]].get("id")
        self._load_fields(boxes[selection[0]])
        self.redraw_canvas()

    def refresh_list(self) -> None:
        selected_index = None
        self.objects.delete(0, "end")
        for index, box in enumerate(self.frame_boxes()):
            prop = box.get("propagation")
            badge = "✓" if prop and prop.get("reviewed") else ("⚠" if prop else " ")
            quality = f" q={prop.get('quality', 0):.2f}" if prop else ""
            self.objects.insert(
                "end",
                f"{badge} {box.get('class_name')} | {box.get('track_id') or '(无轨迹)'} | "
                f"XY=({box['center_xyz'][0]:.2f},{box['center_xyz'][1]:.2f}){quality}",
            )
            if box.get("id") == self.selected_id:
                selected_index = index
        if selected_index is not None:
            self.objects.selection_set(selected_index)
            self.objects.see(selected_index)

    def load_related_image(self) -> None:
        key = Path(self.frame_map[str(self.frame)]).stem
        folder = self.episode / "related_images" / key
        files = []
        if folder.exists():
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if not files:
            self.image_label.configure(image="", text="没有同步参考图")
            self.image_photo = None
            return
        with Image.open(files[0]) as source:
            image = source.convert("RGB")
            image.thumbnail((480, 350), Image.Resampling.LANCZOS)
        self.image_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.image_photo, text="")

    def _grid_step(self) -> float:
        target_m = 80.0 / self.view.pixels_per_meter
        magnitude = 10 ** math.floor(math.log10(max(target_m, 1e-6)))
        for multiplier in (1, 2, 5, 10):
            if magnitude * multiplier >= target_m:
                return magnitude * multiplier
        return magnitude * 10

    def redraw_canvas(self) -> None:
        if not hasattr(self, "canvas") or not self.canvas.winfo_exists():
            return
        c = self.canvas
        c.delete("all")
        w, h = self._canvas_size()
        step = self._grid_step()
        x_min, y_max = self._world(0, 0)
        x_max, y_min = self._world(w, h)
        x = math.floor(x_min / step) * step
        while x <= x_max:
            px, _ = self.view.world_to_canvas(x, 0, w, h)
            c.create_line(px, 0, px, h, fill="#173047")
            c.create_text(px + 3, 4, anchor="nw", fill="#53718c", text=f"{x:g}")
            x += step
        y = math.floor(y_min / step) * step
        while y <= y_max:
            _, py = self.view.world_to_canvas(0, y, w, h)
            c.create_line(0, py, w, py, fill="#173047")
            c.create_text(4, py - 3, anchor="sw", fill="#53718c", text=f"{y:g}")
            y += step

        if len(self.cloud):
            xy = self.cloud[:, :2]
            valid = np.isfinite(xy).all(axis=1)
            xy = xy[valid]
            if len(xy) > 60000:
                xy = xy[::math.ceil(len(xy) / 60000)]
            px = np.rint(w / 2 + (xy[:, 0] - self.view.center_x) * self.view.pixels_per_meter).astype(int)
            py = np.rint(h / 2 - (xy[:, 1] - self.view.center_y) * self.view.pixels_per_meter).astype(int)
            visible = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            image = np.zeros((h, w, 3), dtype=np.uint8)
            image[..., :] = (6, 16, 27)
            image[py[visible], px[visible]] = (75, 221, 232)
            if self.view.pixels_per_meter >= 18:
                pxv, pyv = px[visible], py[visible]
                for ox, oy in ((1, 0), (0, 1)):
                    keep = (pxv + ox < w) & (pyv + oy < h)
                    image[pyv[keep] + oy, pxv[keep] + ox] = (75, 221, 232)
            point_image = Image.fromarray(image, "RGB")
            self.point_photo = ImageTk.PhotoImage(point_image)
            c.create_image(0, 0, image=self.point_photo, anchor="nw", tags="points")
            c.tag_lower("points")
            # Recreate grid above point bitmap.
            x = math.floor(x_min / step) * step
            while x <= x_max:
                gx, _ = self.view.world_to_canvas(x, 0, w, h)
                c.create_line(gx, 0, gx, h, fill="#173047")
                x += step
            y = math.floor(y_min / step) * step
            while y <= y_max:
                _, gy = self.view.world_to_canvas(0, y, w, h)
                c.create_line(0, gy, w, gy, fill="#173047")
                y += step

        origin = self.view.world_to_canvas(0, 0, w, h)
        c.create_oval(origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4, fill="#ffffff", outline="")
        c.create_text(10, 9, anchor="nw", fill="#e8f3ff", text=f"{self.episode.name}  |  sensor-local XY  |  {self.view.pixels_per_meter:.1f} px/m")
        self._draw_boxes()

    def _draw_boxes(self) -> None:
        c = self.canvas
        w, h = self._canvas_size()
        for box in self.frame_boxes():
            x0, y0, x1, y1 = box_bounds(box)
            left, bottom = self.view.world_to_canvas(x0, y0, w, h)
            right, top = self.view.world_to_canvas(x1, y1, w, h)
            selected = box.get("id") == self.selected_id
            color = COLORS.get(box.get("class_name"), "#ffb703")
            dash = (5, 3) if box.get("propagation") and not box["propagation"].get("reviewed") else None
            c.create_rectangle(left, top, right, bottom, outline=color, width=3 if selected else 2, dash=dash)
            c.create_text(left + 2, top - 3, anchor="sw", fill=color, text=f"{box.get('class_name')} · {box.get('track_id')}")
            if selected:
                for hx, hy in box_handles(box).values():
                    px, py = self.view.world_to_canvas(hx, hy, w, h)
                    c.create_rectangle(px - 5, py - 5, px + 5, py + 5, fill="#ffffff", outline=color, width=2)

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.save_button.configure(text="保存* Ctrl+S")
        self._update_status()
        if self.autosave_job is not None:
            self.win.after_cancel(self.autosave_job)
        self.autosave_job = self.win.after(1500, self._autosave_if_dirty)

    def _autosave_if_dirty(self) -> None:
        self.autosave_job = None
        if self.dirty:
            self.save()

    def _update_status(self) -> None:
        finite = int(np.isfinite(self.cloud).all(axis=1).sum()) if len(self.cloud) else 0
        propagated = sum(bool(b.get("propagation")) for b in self.frame_boxes())
        unreviewed = sum(bool(b.get("propagation")) and not b["propagation"].get("reviewed") for b in self.frame_boxes())
        self.status.configure(
            text=f"有限 XYZ: {finite:,}  |  本帧框: {len(self.frame_boxes())}  |  传播框: {propagated}（待复核 {unreviewed}）"
                 f"  |  {'有未保存修改' if self.dirty else '已保存'}"
        )

    def save(self) -> None:
        if self.autosave_job is not None:
            self.win.after_cancel(self.autosave_job)
            self.autosave_job = None
        path = self.annotation_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Saving upgrades the document wrapper to v2 while retaining all v1 boxes.
        self.annotations["schema_version"] = 2
        self.annotations["annotation_format"] = "local_annotation_sidecar_v2"
        self.annotations["coordinate_frame"] = "sensor_local_xyz"
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.annotations, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
        self.dirty = False
        self.save_button.configure(text="保存 Ctrl+S")
        self._update_status()

    def close(self) -> None:
        self.playing = False
        try:
            self.save()
        finally:
            self.win.destroy()

    def run(self) -> None:
        self.win.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="连续帧 LiDAR 点云本地盲标工作台")
    parser.add_argument("--data-root", type=Path, required=True, help="包含训练 Episode 的目录")
    parser.add_argument("--output", type=Path, required=True, help="独立的 sidecar 标注输出目录")
    parser.add_argument(
        "--allow-holdout", action="store_true",
        help="明确允许载入名称含 holdout 的 Episode（规则和模型冻结前不要使用）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        app = AnnotationApp(args.data_root, args.output, allow_holdout=args.allow_holdout)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("点云标注工具启动失败", str(exc))
        root.destroy()
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
