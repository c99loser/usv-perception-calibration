#!/usr/bin/env python3
"""Build a track-aggregated point-cloud review and annotation workspace.

The input is the immutable V4.11 cluster annotation package.  Each production
track is converted into:

* a centered, radial/tangential/z aggregate PCD;
* a representative full-frame clustered context PCD with the target colored;
* a six-panel PNG preview including an explicitly unprojected camera image;
* a local, browser-based track-classification interface.

No extrinsic, TF, odometry, world state, or semantic truth is consumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "real-lidar-track-annotation-viewer/v1"
PROHIBITED_INPUTS = ("/tf", "/tf_static", "odometry", "/world/*")


def _load_points(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        return np.empty((0, 3), dtype=float)
    return array[np.isfinite(array).all(axis=1)]


def _voxel(points: np.ndarray, size: float, limit: int) -> np.ndarray:
    if not len(points):
        return points.reshape((-1, 3))
    cells: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for point, key in zip(points, np.floor(points / size).astype(np.int32)):
        cells[tuple(int(value) for value in key)].append(point)
    reduced = np.asarray([np.mean(values, axis=0) for values in cells.values()])
    if len(reduced) > limit:
        indices = np.linspace(0, len(reduced) - 1, limit, dtype=int)
        reduced = reduced[indices]
    return reduced


def _canonical(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    centered = points - center
    horizontal = center[:2]
    norm = float(np.linalg.norm(horizontal))
    radial = horizontal / norm if norm > 1e-6 else np.asarray((1.0, 0.0))
    tangent = np.asarray((-radial[1], radial[0]))
    return np.column_stack(
        (centered[:, :2] @ radial, centered[:, :2] @ tangent, centered[:, 2])
    )


def _write_pcd(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if len(points) != len(colors):
        raise ValueError("point/color length mismatch")
    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z r g b",
        "SIZE 4 4 4 1 1 1",
        "TYPE F F F U U U",
        "COUNT 1 1 1 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]
    lines.extend(
        f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}"
        for (x, y, z), (r, g, b) in zip(points, colors)
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _equal_2d(ax, x: np.ndarray, y: np.ndarray, *, pad: float = 0.08) -> None:
    if not len(x):
        return
    x0, x1 = float(np.min(x)), float(np.max(x))
    y0, y1 = float(np.min(y)), float(np.max(y))
    span = max(x1 - x0, y1 - y0, 0.25)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = 0.5 * span * (1.0 + 2.0 * pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


def _scatter(ax, points: np.ndarray, a: int, b: int, *, color: str, size: float, alpha: float = 0.9) -> None:
    if len(points):
        ax.scatter(points[:, a], points[:, b], s=size, c=color, alpha=alpha, linewidths=0)


def _render_preview(
    path: Path,
    *,
    item: dict,
    image_path: Path,
    context: np.ndarray,
    target: np.ndarray,
    aggregate: np.ndarray,
    representative_local: np.ndarray,
) -> None:
    figure = plt.figure(figsize=(16, 9), dpi=120, facecolor="#101318")
    grid = figure.add_gridspec(2, 3, wspace=0.22, hspace=0.32)
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    for ax in axes:
        ax.set_facecolor("#151a21")
        ax.tick_params(colors="#aab4c0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#3d4652")
        ax.grid(color="#2b333d", alpha=0.5, linewidth=0.5)

    axes[0].axis("off")
    try:
        image = mpimg.imread(image_path)
        axes[0].imshow(image)
        axes[0].set_title("Synchronized image context (NO 2D-3D projection)", color="#f2c14e", fontsize=10)
    except Exception:
        axes[0].text(0.5, 0.5, "Image unavailable", ha="center", va="center", color="#f2c14e")

    center = np.asarray(item["representative_center_m"], dtype=float)
    distances = np.linalg.norm(context[:, :2] - center[:2], axis=1) if len(context) else np.asarray([])
    local_context = context[distances <= 8.0] if len(distances) else context
    _scatter(axes[1], local_context, 0, 1, color="#6e7c8c", size=2.2, alpha=0.35)
    _scatter(axes[1], target, 0, 1, color="#ff8a3d", size=14)
    axes[1].scatter([0], [0], marker="*", s=80, c="#55d6be", label="LiDAR")
    axes[1].set_title("Representative XY context: orange = target track", color="#e8edf2", fontsize=10)
    axes[1].set_xlabel("x / m", color="#aab4c0")
    axes[1].set_ylabel("y / m", color="#aab4c0")
    _equal_2d(axes[1], np.r_[local_context[:, 0], target[:, 0]], np.r_[local_context[:, 1], target[:, 1]])

    ax3 = figure.add_subplot(grid[0, 2], projection="3d")
    ax3.set_facecolor("#151a21")
    if len(aggregate):
        ax3.scatter(aggregate[:, 0], aggregate[:, 1], aggregate[:, 2], s=5, c="#ff8a3d", alpha=0.75)
    if len(representative_local):
        ax3.scatter(representative_local[:, 0], representative_local[:, 1], representative_local[:, 2], s=14, c="#55d6be", alpha=0.95)
    ax3.set_title("Track aggregate (orange) + best frame (cyan)", color="#e8edf2", fontsize=10)
    ax3.set_xlabel("radial / m", color="#aab4c0")
    ax3.set_ylabel("tangential / m", color="#aab4c0")
    ax3.set_zlabel("z / m", color="#aab4c0")
    ax3.tick_params(colors="#aab4c0", labelsize=7)

    views = [
        (1, 2, "Tangential-Z silhouette", "tangential / m", "z / m"),
        (0, 2, "Radial-Z thickness", "radial / m", "z / m"),
        (0, 1, "Radial-tangential top view", "radial / m", "tangential / m"),
    ]
    for ax, (a, b, title, xlabel, ylabel) in zip(axes[3:], views):
        _scatter(ax, aggregate, a, b, color="#ff8a3d", size=7, alpha=0.72)
        _scatter(ax, representative_local, a, b, color="#55d6be", size=18, alpha=0.95)
        ax.set_title(title, color="#e8edf2", fontsize=10)
        ax.set_xlabel(xlabel, color="#aab4c0")
        ax.set_ylabel(ylabel, color="#aab4c0")
        _equal_2d(ax, aggregate[:, a], aggregate[:, b])

    title = (
        f"{item['track_group_id']}  |  {item['camera']}  |  "
        f"{item['observations']} frames / {item['aggregate_points']} aggregate points  |  "
        f"{item['eligible_observations']} V4.11-eligible frames"
    )
    figure.suptitle(title, color="#f1f5f9", fontsize=13, y=0.985)
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _html_document(split: str, items: list[dict]) -> str:
    safe_data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    warning = (
        "最终留出：标注结果不得用于训练、特征设计或阈值选择。"
        if split == "holdout"
        else "训练区：允许用于训练和阈值选择；禁止混入 holdout。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4.11 {html.escape(split)} 轨迹点云标注</title>
<style>
body{{margin:0;background:#0f1318;color:#e8edf2;font:15px system-ui,sans-serif}}header{{padding:12px 18px;border-bottom:1px solid #303843;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}main{{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px;padding:14px}}img{{width:100%;max-height:calc(100vh - 110px);object-fit:contain;background:#111820}}aside{{background:#171d24;border:1px solid #303843;border-radius:8px;padding:14px}}button,select,input,textarea{{font:inherit}}button{{background:#26313c;color:#e8edf2;border:1px solid #4a5867;border-radius:6px;padding:9px 11px;cursor:pointer}}button:hover{{background:#32404e}}button.active{{background:#d66b2f;border-color:#ff9a5f;color:white}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}}label{{display:block;margin:12px 0 5px;color:#aeb8c3}}select,input,textarea{{box-sizing:border-box;width:100%;background:#0f141a;color:#e8edf2;border:1px solid #44505d;border-radius:5px;padding:8px}}textarea{{height:84px}}.row{{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}}.warn{{color:#f2c14e}}.meta{{font-family:ui-monospace,monospace;color:#aeb8c3;font-size:12px;white-space:pre-wrap}}@media(max-width:900px){{main{{grid-template-columns:1fr}}img{{max-height:none}}}}
</style>
</head>
<body>
<header><strong>V4.11 轨迹级点云标注 · {html.escape(split)}</strong><span id="progress"></span><span class="warn">{html.escape(warning)}</span></header>
<main><section><img id="preview" alt="轨迹聚合点云多视角预览"></section><aside>
<div id="identifier"></div><pre class="meta" id="meta"></pre>
<label>类别（快捷键 1–6）</label><div class="grid" id="labels"></div>
<label for="static">可作静态标定对象</label><select id="static"><option value="">未选择</option><option value="yes">yes</option><option value="no">no</option><option value="uncertain">uncertain</option></select>
<label for="confidence">置信度</label><select id="confidence"><option value="">未选择</option><option>high</option><option>medium</option><option>low</option></select>
<label for="note">备注</label><textarea id="note"></textarea>
<div class="row"><button id="prev">← 上一条</button><button id="next">下一条 →</button><button id="jump">跳到未标</button></div>
<div class="row"><button id="export">导出 JSON</button><button id="exportCsv">导出 CSV</button><label style="margin:0"><input id="import" type="file" accept="application/json" style="display:none"><button type="button" id="importButton">导入进度</button></label></div>
</aside></main>
<script>
const items={safe_data}; const split={json.dumps(split)}; const labels=['pillar','buoy','block','boat','background','ambiguous']; const key='lidar-v411-'+split; let state=JSON.parse(localStorage.getItem(key)||'{{}}'); let index=0;
const q=id=>document.getElementById(id); const save=()=>localStorage.setItem(key,JSON.stringify(state));
function currentState(){{return state[items[index].track_group_id]||(state[items[index].track_group_id]={{}})}}
function persist(){{const s=currentState();s.static_for_calibration=q('static').value;s.reviewer_confidence=q('confidence').value;s.reviewer_note=q('note').value;save();}}
function render(){{const item=items[index],s=currentState();q('preview').src='../'+item.preview_path;q('identifier').textContent=(index+1)+' / '+items.length+' · '+item.track_group_id;q('meta').textContent=`camera: ${{item.camera}}\nframes: ${{item.observations}}\neligible frames: ${{item.eligible_observations}}\naggregate points: ${{item.aggregate_points}}\nextent median: ${{item.median_reference_extent_m.join(', ')}} m`;q('static').value=s.static_for_calibration||'';q('confidence').value=s.reviewer_confidence||'';q('note').value=s.reviewer_note||'';document.querySelectorAll('#labels button').forEach(b=>b.classList.toggle('active',b.dataset.label===s.human_label));q('progress').textContent=Object.values(state).filter(v=>v.human_label).length+' / '+items.length+' 已标';}}
labels.forEach((label,i)=>{{const b=document.createElement('button');b.textContent=(i+1)+' '+label;b.dataset.label=label;b.onclick=()=>{{currentState().human_label=label;if(label==='boat')q('static').value='no';if(label==='ambiguous')q('static').value='uncertain';persist();render();}};q('labels').appendChild(b)}});
q('prev').onclick=()=>{{persist();index=(index-1+items.length)%items.length;render()}};q('next').onclick=()=>{{persist();index=(index+1)%items.length;render()}};q('jump').onclick=()=>{{persist();const found=items.findIndex((it,i)=>i>index&&!state[it.track_group_id]?.human_label);index=found>=0?found:(items.findIndex(it=>!state[it.track_group_id]?.human_label));if(index<0)index=0;render()}};
function download(name,text,type){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type}}));a.download=name;a.click();URL.revokeObjectURL(a.href)}}
q('export').onclick=()=>{{persist();download('track_labels_'+split+'.json',JSON.stringify({{schema:'real-lidar-track-labels/v1',split,items:state}},null,2),'application/json')}};
q('exportCsv').onclick=()=>{{persist();let rows=['track_group_id,human_label,static_for_calibration,reviewer_confidence,reviewer_note'];for(const item of items){{const s=state[item.track_group_id]||{{}};const vals=[item.track_group_id,s.human_label||'',s.static_for_calibration||'',s.reviewer_confidence||'',(s.reviewer_note||'').replaceAll('"','""')];rows.push(vals.slice(0,4).join(',')+',"'+vals[4]+'"')}}download('track_labels_'+split+'.csv',String.fromCharCode(0xFEFF)+rows.join(String.fromCharCode(10)),'text/csv')}};
q('importButton').onclick=()=>q('import').click();q('import').onchange=async e=>{{const data=JSON.parse(await e.target.files[0].text());if(data.split!==split){{alert('split 不匹配');return}}state=data.items||{{}};save();render()}};
['static','confidence','note'].forEach(id=>q(id).onchange=persist);document.addEventListener('keydown',e=>{{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;if(e.key>='1'&&e.key<='6')document.querySelector(`#labels button:nth-child(${{e.key}})`).click();if(e.key==='ArrowRight')q('next').click();if(e.key==='ArrowLeft')q('prev').click()}});render();
</script></body></html>"""


def _review_sheet(path: Path, items: list[dict], root: Path) -> None:
    by_camera: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_camera[item["camera"]].append(item)
    selected = []
    for camera in sorted(by_camera):
        ranked = sorted(
            by_camera[camera],
            key=lambda item: (-item["eligible_observations"], -item["aggregate_points"]),
        )
        selected.extend(ranked[:2])
    selected = selected[:6]
    canvas = Image.new("RGB", (1600, 900), "#101318")
    draw = ImageDraw.Draw(canvas)
    for position, item in enumerate(selected):
        preview = Image.open(root / item["preview_path"]).convert("RGB")
        preview.thumbnail((790, 420))
        x = 5 + (position % 2) * 800
        y = 5 + (position // 2) * 298
        crop_top = max(0, (preview.height - 285) // 2)
        preview = preview.crop((0, crop_top, preview.width, min(preview.height, crop_top + 285)))
        canvas.paste(preview, (x, y))
        draw.text((x + 8, y + 268), item["track_group_id"], fill="#f1f5f9")
    canvas.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-voxel-m", type=float, default=0.025)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    package = json.loads((args.package / "package_manifest.json").read_text(encoding="utf-8"))
    if package.get("schema") != "real-lidar-manual-annotation-pack/v4.11":
        raise SystemExit("unsupported source package")
    rows = list(csv.DictReader((args.package / "manifest.csv").open(encoding="utf-8-sig")))
    source_specs = {item["id"]: item for item in package["source_summaries"]}
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["track_group_id"]].append(row)

    args.output.mkdir(parents=True)
    for folder in ("previews", "aggregate_pcd", "context_pcd", "metadata", "train", "holdout"):
        (args.output / folder).mkdir()
    source_cache: dict[str, tuple[dict, dict]] = {}
    viewer_items: list[dict] = []
    for track_group_id, track_rows in sorted(grouped_rows.items()):
        source_id = track_rows[0]["source_id"]
        source = source_specs[source_id]
        if source_id not in source_cache:
            raw = json.loads(Path(source["raw_observations"]).read_text(encoding="utf-8"))
            tracked = json.loads(Path(source["tracked_observations"]).read_text(encoding="utf-8"))
            source_cache[source_id] = raw, tracked
        raw, tracked = source_cache[source_id]
        track_id = int(track_group_id.rsplit("_", 1)[1])
        observations = []
        for frame_index, (raw_frame, tracked_frame) in enumerate(zip(raw["frames"], tracked["frames"])):
            for cluster in tracked_frame.get("clusters", []):
                if int(cluster.get("temporal_track", {}).get("track_id", -1)) != track_id:
                    continue
                source_index = int(cluster["source_cluster_index"])
                raw_cluster = raw_frame["clusters"][source_index]
                points = _load_points(raw_cluster.get("points_sensor_m"))
                if not len(points):
                    continue
                center = np.asarray(raw_cluster["center_m"], dtype=float)
                observations.append(
                    {
                        "frame_index": frame_index,
                        "source_index": source_index,
                        "points": points,
                        "center": center,
                        "canonical": _canonical(points, center),
                        "eligible": cluster.get("calibration_quality", {}).get("eligible") is True,
                        "extent": np.asarray(cluster.get("reference_extent_m", cluster["extent_m"]), dtype=float),
                        "image": Path(raw_frame["image"]),
                    }
                )
        if not observations:
            continue
        aggregate = _voxel(np.vstack([item["canonical"] for item in observations]), args.aggregate_voxel_m, 6000)
        representative = max(observations, key=lambda item: len(item["points"]))
        raw_frame = raw["frames"][representative["frame_index"]]
        context_chunks = [_load_points(cluster.get("points_sensor_m")) for cluster in raw_frame.get("clusters", [])]
        context = _voxel(np.vstack([chunk for chunk in context_chunks if len(chunk)]), 0.05, 12000) if any(len(chunk) for chunk in context_chunks) else np.empty((0, 3))
        target = representative["points"]
        target_keys = {tuple(value) for value in np.floor(target / 0.03).astype(np.int32)}
        keep_background = np.asarray([tuple(value) not in target_keys for value in np.floor(context / 0.03).astype(np.int32)]) if len(context) else np.asarray([], dtype=bool)
        background = context[keep_background] if len(context) else context
        context_points = np.vstack((background, target))
        context_colors = np.vstack(
            (
                np.tile(np.asarray((120, 135, 150), dtype=np.uint8), (len(background), 1)),
                np.tile(np.asarray((255, 120, 45), dtype=np.uint8), (len(target), 1)),
            )
        )
        aggregate_colors = np.tile(np.asarray((255, 120, 45), dtype=np.uint8), (len(aggregate), 1))
        aggregate_rel = Path("aggregate_pcd") / f"{track_group_id}.pcd"
        context_rel = Path("context_pcd") / f"{track_group_id}.pcd"
        preview_rel = Path("previews") / f"{track_group_id}.png"
        _write_pcd(args.output / aggregate_rel, aggregate, aggregate_colors)
        _write_pcd(args.output / context_rel, context_points, context_colors)
        item = {
            "track_group_id": track_group_id,
            "source_id": source_id,
            "split_role": track_rows[0]["split_role"],
            "bag_id": track_rows[0]["bag_id"],
            "camera": track_rows[0]["camera"],
            "observations": len(observations),
            "eligible_observations": sum(item["eligible"] for item in observations),
            "aggregate_points": len(aggregate),
            "representative_frame_index": representative["frame_index"],
            "representative_center_m": representative["center"].round(4).tolist(),
            "median_reference_extent_m": np.median(np.asarray([item["extent"] for item in observations]), axis=0).round(4).tolist(),
            "preview_path": preview_rel.as_posix(),
            "aggregate_pcd_path": aggregate_rel.as_posix(),
            "context_pcd_path": context_rel.as_posix(),
            "human_label": "",
            "static_for_calibration": "",
            "reviewer_confidence": "",
            "reviewer_note": "",
        }
        _render_preview(
            args.output / preview_rel,
            item=item,
            image_path=representative["image"],
            context=context,
            target=target,
            aggregate=aggregate,
            representative_local=_canonical(target, representative["center"]),
        )
        (args.output / "metadata" / f"{track_group_id}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        viewer_items.append(item)

    fields = list(viewer_items[0])
    with (args.output / "track_labels_template.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(viewer_items)
    for split in ("train", "holdout"):
        split_items = [item for item in viewer_items if item["split_role"] == split]
        (args.output / split / "index.html").write_text(_html_document(split, split_items), encoding="utf-8")
        _review_sheet(args.output / f"{split}_review_sheet.png", split_items, args.output)
    summary = {
        "schema": SCHEMA,
        "source_package": str(args.package),
        "purpose": "human_track_semantic_classification_with_sensor_only_context",
        "prohibited_inputs": list(PROHIBITED_INPUTS),
        "tracks": len(viewer_items),
        "by_split": dict(Counter(item["split_role"] for item in viewer_items)),
        "by_split_camera": dict(Counter(f"{item['split_role']}:{item['camera']}" for item in viewer_items)),
        "annotation_entrypoints": {"train": "train/index.html", "holdout": "holdout/index.html"},
        "publication_allowed": False,
    }
    (args.output / "viewer_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "README_使用说明.md").write_text(
        """# V4.11 轨迹聚合点云标注界面

本工作区用于替代稀疏单簇直接上传 Supervisely 的做法。每个条目代表一个生产 `track_group_id`，将同轨迹多帧在纯 LiDAR 局部坐标中平移对齐并聚合，同时保留代表帧场景上下文。

1. 双击 `train/index.html` 开始训练区标注；`holdout/index.html` 只交给独立评估人员。
2. 橙色为当前轨迹，灰色为代表帧其他 V4 簇。相机图只提供场景上下文，明确没有投影。
3. 标注自动保存在浏览器 localStorage；每完成约 20 条点击“导出 JSON”备份。
4. 可用数字 1–6 选类别，左右方向键切换。
5. `aggregate_pcd/` 可在 CloudCompare 等 3D 查看器中旋转检查；`context_pcd/` 保留代表帧上下文和颜色。
6. 不得使用 TF、里程计、世界真值或名义外参投影来判断类别。
7. 轨迹聚合只用于人工识别和语义标签，不改变 V4.11 的几何中心、质量门控或外参约束。
8. 导出 JSON 后，使用项目内 `apply_track_labels_to_lidar_pack.py` 将轨迹标签传播回原始逐观测 CSV；不要手工复制 572 行标签。
""",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output).as_posix()}")
    (args.output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
