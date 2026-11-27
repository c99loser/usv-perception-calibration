#!/usr/bin/env python3
"""Build a read-only visual decision UI for the V3.1 hard-case queue."""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


BUILD_PATH = Path(__file__).with_name("build_sparse_lidar_training_v3_1.py")
SPEC = importlib.util.spec_from_file_location("sparse_builder", BUILD_PATH)
assert SPEC and SPEC.loader
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def box_corners(center_x: float, center_y: float, size_x: float, size_y: float, yaw: float) -> np.ndarray:
    local = np.asarray([[-size_x / 2, -size_y / 2], [size_x / 2, -size_y / 2],
                        [size_x / 2, size_y / 2], [-size_x / 2, size_y / 2], [-size_x / 2, -size_y / 2]])
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return local @ rotation.T + np.asarray([center_x, center_y])


def render(path: Path, queue: dict[str, str], observation: dict[str, str]) -> None:
    points = BUILD.read_xyz_binary_pcd(Path(queue["pcd_path"]))
    center = [float(observation[f"original_center_{axis}"]) for axis in "xyz"]
    dimensions = [float(observation[f"original_size_{axis}"]) for axis in "xyz"]
    yaw = float(observation["original_yaw_rad"])
    indices = BUILD.oriented_xy_indices(points, center, dimensions, yaw)
    support = points[indices]
    corners = box_corners(center[0], center[1], dimensions[0], dimensions[1], yaw)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=110, facecolor="#10151c")
    for axis in axes.flat:
        axis.set_facecolor("#101820"); axis.tick_params(colors="#aab6c2")
        for spine in axis.spines.values(): spine.set_color("#465362")
    axes[0, 0].axis("off")
    try:
        axes[0, 0].imshow(mpimg.imread(queue["image_path"]))
    except Exception:
        axes[0, 0].text(0.5, 0.5, "image unavailable", color="white", ha="center")
    axes[0, 0].set_title("Synchronized image (no projection)", color="#f2c14e")
    axes[0, 1].scatter(points[:, 0], points[:, 1], s=1.5, c="#526575", alpha=0.55)
    if len(support): axes[0, 1].scatter(support[:, 0], support[:, 1], s=13, c="#4ee0d0")
    axes[0, 1].plot(corners[:, 0], corners[:, 1], c="#ff9838", lw=2)
    axes[0, 1].scatter([0], [0], marker="*", s=80, c="#e9f55a")
    axes[0, 1].set_aspect("equal"); axes[0, 1].set_title("Full XY: cyan=support, orange=manual footprint", color="white")
    margin = max(dimensions[0], dimensions[1], 1.0) * 1.5
    axes[1, 0].scatter(points[:, 0], points[:, 1], s=2, c="#526575", alpha=0.35)
    if len(support): axes[1, 0].scatter(support[:, 0], support[:, 1], s=20, c="#4ee0d0")
    axes[1, 0].plot(corners[:, 0], corners[:, 1], c="#ff9838", lw=2)
    axes[1, 0].set_xlim(center[0] - margin, center[0] + margin); axes[1, 0].set_ylim(center[1] - margin, center[1] + margin)
    axes[1, 0].set_aspect("equal"); axes[1, 0].set_title("Local XY review", color="white")
    if len(support):
        axes[1, 1].scatter(support[:, 0], support[:, 2], s=24, c="#4ee0d0")
        axes[1, 1].axvline(float(np.median(support[:, 0])), c="#f2c14e", ls="--", lw=1)
        axes[1, 1].axhline(float(np.median(support[:, 2])), c="#f2c14e", ls="--", lw=1)
    axes[1, 1].set_title(f"Support XZ ({len(support)} points); Z derived from returns", color="white")
    axes[1, 1].set_xlabel("sensor X / m", color="#aab6c2"); axes[1, 1].set_ylabel("sensor Z / m", color="#aab6c2")
    figure.suptitle(
        f"{queue['review_reason']} | {queue['episode_id']} frame {queue['frame_index']} | "
        f"{queue['class_name']} {queue['track_id']} | p={queue['probability']} center_err={queue['center_error_m']}m",
        color="white", fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95)); figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def page(items: list[dict[str, str]]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>V3.1 定向难例复核</title>
<style>body{{margin:0;background:#0f141a;color:#e8edf2;font:15px system-ui}}header{{padding:12px 18px;border-bottom:1px solid #34404c}}main{{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:14px;padding:14px}}img{{width:100%;max-height:calc(100vh - 90px);object-fit:contain}}aside{{background:#171e26;padding:14px;border:1px solid #34404c;border-radius:8px}}button{{width:100%;padding:10px;margin:5px 0;background:#263441;color:white;border:1px solid #526577;border-radius:6px}}button.active{{background:#c96b30}}pre{{white-space:pre-wrap;color:#aebbc7}}@media(max-width:900px){{main{{grid-template-columns:1fr}}}}</style></head>
<body><header><b>V3.1 定向难例复核</b> · <span id="progress"></span> · 只导出决定，不修改源标注</header><main><img id="preview"><aside><h3 id="title"></h3><pre id="meta"></pre><div id="buttons"></div><button id="prev">← 上一条</button><button id="next">下一条 →</button><button id="jump">跳到未复核</button><button id="export">导出 review_decisions.json</button></aside></main>
<script>const items={data},choices=[['confirm_label','确认现有标签'],['tighten_box','收紧/修正 XY 框'],['exclude','排除：无可靠监督'],['uncertain','不确定，二次复核']];let i=0,state=JSON.parse(localStorage.getItem('v31-hard-review')||'{{}}');const q=x=>document.getElementById(x),save=()=>localStorage.setItem('v31-hard-review',JSON.stringify(state));
function render(){{let x=items[i],s=state[x.annotation_id]||{{}};q('preview').src=x.preview_path;q('title').textContent=`${{i+1}}/${{items.length}} · ${{x.review_reason}}`;q('meta').textContent=`${{x.episode_id}} frame ${{x.frame_index}}\n${{x.class_name}} / ${{x.track_id}}\nsupport=${{x.support_count}}, probability=${{x.probability}}\ncenter_error=${{x.center_error_m}} m\n建议：${{x.recommended_action}}`;q('buttons').innerHTML='';for(const [v,t] of choices){{let b=document.createElement('button');b.textContent=t;b.className=s.decision===v?'active':'';b.onclick=()=>{{state[x.annotation_id]={{decision:v,reviewed_at:new Date().toISOString()}};save();render()}};q('buttons').appendChild(b)}}q('progress').textContent=`${{Object.keys(state).length}}/${{items.length}} 已复核`;}}
q('prev').onclick=()=>{{i=(i-1+items.length)%items.length;render()}};q('next').onclick=()=>{{i=(i+1)%items.length;render()}};q('jump').onclick=()=>{{let n=items.findIndex((x,j)=>j>i&&!state[x.annotation_id]);if(n<0)n=items.findIndex(x=>!state[x.annotation_id]);if(n>=0)i=n;render()}};q('export').onclick=()=>{{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({{schema:'real-lidar-targeted-review-decisions/v1',items:state}},null,2)],{{type:'application/json'}}));a.download='review_decisions.json';a.click()}};document.addEventListener('keydown',e=>{{if(e.key==='ArrowRight')q('next').click();if(e.key==='ArrowLeft')q('prev').click()}});render();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    review = args.review_dir.resolve(); preview_dir = review / "previews"
    if preview_dir.exists() or (review / "review.html").exists():
        raise SystemExit("viewer output already exists; refusing to overwrite")
    queue = load_csv(review / "review_queue.csv")
    observations = {row["annotation_id"]: row for row in load_csv(args.asset_root.resolve() / "observations.csv")}
    preview_dir.mkdir()
    items = []
    for index, row in enumerate(queue):
        observation = observations[row["annotation_id"]]
        name = f"{index:04d}_{row['annotation_id']}.png"; render(preview_dir / name, row, observation)
        items.append({**row, "preview_path": f"previews/{name}"})
    (review / "review.html").write_text(page(items), encoding="utf-8")
    print(json.dumps({"items": len(items), "html": str(review / 'review.html'), "previews": str(preview_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
