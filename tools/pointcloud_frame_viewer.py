"""Small read-only MCAP PointCloud2 frame viewer.

Uses Tk only, so it works on a clean Windows Python installation.  It reads
one selected PointCloud2 topic on demand and displays its XY top view.
"""
from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np


PCD_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("_pad0", "<f4"),
    ("intensity", "<f4"), ("_pad1", "<f4"), ("ring", "<u2"),
    ("_pad2", "V6"),
])


class PointCloudViewer:
    def __init__(self, mcap_path: Path, topic: str, max_frames: int) -> None:
        self.mcap_path = mcap_path
        self.topic = topic
        self.max_frames = max_frames
        self.frames: list[tuple[int, np.ndarray]] = []
        self.index = 0
        self.playing = False
        self.loading_done = False
        self.generator = self._frame_generator()

        self.root = tk.Tk()
        self.root.title("MCAP 点云逐帧查看器（只读）")
        self.root.geometry("1020x830")

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        ttk.Button(controls, text="◀ 上一帧", command=self.previous).pack(side="left")
        self.play_btn = ttk.Button(controls, text="▶ 播放", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=6)
        ttk.Button(controls, text="下一帧 ▶", command=self.next).pack(side="left")
        ttk.Button(controls, text="保存当前画面", command=self.save_view).pack(side="left", padx=14)
        self.status = ttk.Label(controls, text="正在读取第 1 帧…")
        self.status.pack(side="left", padx=8)

        self.canvas = tk.Canvas(self.root, width=1000, height=720, bg="#07111d", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0, 10), fill="both", expand=True)
        self.root.bind("<Left>", lambda _: self.previous())
        self.root.bind("<Right>", lambda _: self.next())
        self.root.bind("<space>", lambda _: self.toggle_play())
        self.root.after(20, self.show_first)

    def _frame_generator(self):
        package_root = Path(__file__).resolve().parents[1] / ".codex_pydeps"
        sys.path.insert(0, str(package_root))
        from mcap_ros2.reader import read_ros2_messages
        for msg in read_ros2_messages(str(self.mcap_path), topics=[self.topic]):
            raw = np.frombuffer(bytes(msg.ros_msg.data), dtype=PCD_DTYPE)
            valid = np.isfinite(raw["x"]) & np.isfinite(raw["y"]) & np.isfinite(raw["z"])
            yield msg.log_time_ns, raw[valid]

    def ensure(self, requested: int) -> bool:
        while len(self.frames) <= requested and not self.loading_done:
            try:
                self.status.config(text=f"正在读取第 {len(self.frames) + 1} 帧…")
                self.root.update_idletasks()
                self.frames.append(next(self.generator))
                if len(self.frames) >= self.max_frames:
                    self.loading_done = True
            except StopIteration:
                self.loading_done = True
            except Exception as exc:
                self.loading_done = True
                messagebox.showerror("读取 MCAP 失败", str(exc))
        return requested < len(self.frames)

    def show_first(self) -> None:
        if self.ensure(0):
            self.draw()

    def draw(self) -> None:
        stamp, points = self.frames[self.index]
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        # fixed 120m square gives every frame a comparable scale
        scale = min(width, height) / 120.0
        cx, cy = width / 2.0, height / 2.0
        self.canvas.create_line(0, cy, width, cy, fill="#28435d")
        self.canvas.create_line(cx, 0, cx, height, fill="#28435d")
        self.canvas.create_text(12, 12, anchor="nw", fill="#d6e8ff",
                                text="XY 顶视图  |  X 前方，Y 左方  |  ±60 m", font=("Segoe UI", 11, "bold"))
        shown = points[::max(1, len(points) // 8500)]
        for x, y in zip(shown["x"], shown["y"]):
            px, py = cx + float(x) * scale, cy - float(y) * scale
            if 0 <= px < width and 0 <= py < height:
                self.canvas.create_rectangle(px, py, px + 1, py + 1, fill="#3de8ee", outline="")
        self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#ffb703", outline="")
        self.status.config(text=(f"帧 {self.index + 1}/{len(self.frames)}  |  "
                                 f"{len(points):,} 个有效点  |  时间戳 {stamp}"))

    def next(self) -> None:
        if self.ensure(self.index + 1):
            self.index += 1
            self.draw()
        elif self.loading_done:
            self.playing = False
            self.play_btn.config(text="▶ 播放")

    def previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.draw()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.config(text="❚❚ 暂停" if self.playing else "▶ 播放")
        if self.playing:
            self.play()

    def play(self) -> None:
        if not self.playing:
            return
        self.next()
        if self.playing:
            self.root.after(100, self.play)

    def save_view(self) -> None:
        path = Path.cwd() / f"pointcloud_{self.mcap_path.stem}_frame_{self.index + 1:04d}.ps"
        self.canvas.postscript(file=str(path), colormode="color")
        self.status.config(text=f"已保存当前画面：{path.name}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()
    PointCloudViewer(args.mcap, args.topic, args.max_frames).run()


if __name__ == "__main__":
    main()
