#!/usr/bin/env python3
"""Create a compact, auditable MCAP for offline calibration trial runs.

The output contains selected synchronized frames only.  Point clouds are
cleaned with finite/nonzero filtering and 0.1 mm duplicate removal; it never
applies SOR, ROR, range crop, voxel downsampling, or intensity filtering.
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import CompressionFormat, CompressionMode, StoragePlugin, Writer
from rosbags.rosbag2.reader import McapReader
from rosbags.typesys import Stores, get_typestore

CAMERAS = ("front", "back", "left", "left2", "right", "right2")
LIDARS = ("front", "back", "left", "left2", "right", "right2", "forward")

def sensor_topic(kind: str, name: str, suffix: str) -> str:
    unit = "camera" if kind == "cameras" else "lidar"
    return f"/wamv/sensors/{kind}/{name}_{unit}_sensor/{suffix}"

def selected_times(reader: McapReader, max_sets: int, lidar_limit_ms: float, image_limit_ms: float):
    streams = {sensor_topic("lidars", n, "points"): n for n in LIDARS}
    streams.update({sensor_topic("cameras", n, "image_raw"): n for n in CAMERAS})
    conns = [c for c in reader.connections if c.topic in streams]
    timeline = defaultdict(list)
    for c, stamp, _ in reader.messages(conns): timeline[c.topic].append(stamp)
    if any(not timeline[t] for t in streams):
        raise RuntimeError("A required LiDAR/image topic has no messages.")
    def near(values, t):
        ix = int(np.searchsorted(values, t)); candidates = values[max(0,ix-1):ix+1]
        return min(candidates, key=lambda x: abs(x-t))
    chosen, last = [], -10**30
    anchor = timeline[sensor_topic("lidars", "front", "points")]
    for t in anchor:
        if t-last < 500_000_000: continue
        lidar = {n:near(timeline[sensor_topic("lidars",n,"points")],t) for n in LIDARS}
        image = {n:near(timeline[sensor_topic("cameras",n,"image_raw")],t) for n in CAMERAS}
        if max(abs(x-t) for x in lidar.values()) > lidar_limit_ms*1e6: continue
        if max(abs(x-t) for x in image.values()) > image_limit_ms*1e6: continue
        chosen.append({"anchor_ns":t,"lidar":lidar,"image":image}); last=t
        if len(chosen) == max_sets: break
    return chosen

def cloud_xyz(msg):
    fields = {f.name:f for f in msg.fields}
    if not all(x in fields and fields[x].datatype == 7 for x in ("x","y","z")):
        raise ValueError("PointCloud2 needs float32 x/y/z fields")
    n = msg.width * msg.height
    rows = np.frombuffer(msg.data, np.uint8).reshape(n, msg.point_step)
    xyz = np.column_stack([rows[:,fields[x].offset:fields[x].offset+4].copy().view('<f4').ravel() for x in ("x","y","z")])
    return rows, xyz

def clean_cloud_message(msg):
    rows, xyz = cloud_xyz(msg); raw = len(rows)
    valid = np.isfinite(xyz).all(1) & (np.einsum("ij,ij->i",xyz,xyz) > 0)
    rows = rows[valid]; finite = len(rows)
    if finite:
        keys = np.floor(xyz[valid]/1e-4 + .5).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True); rows = rows[np.sort(keep)]
    msg.height, msg.width, msg.row_step = 1, int(len(rows)), int(len(rows)*msg.point_step)
    msg.data = rows.reshape(-1).astype(np.uint8, copy=False)
    msg.is_dense = True
    return msg, {"raw_points":raw,"finite_nonzero_points":finite,"deduplicated_points":int(len(rows)),"retention":float(len(rows)/raw) if raw else 0.0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mcap",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--frame-sets",type=int,default=10); ap.add_argument("--lidar-sync-ms",type=float,default=30); ap.add_argument("--image-sync-ms",type=float,default=50)
    a=ap.parse_args(); r=McapReader(a.mcap); r.open(); selected=selected_times(r,a.frame_sets,a.lidar_sync_ms,a.image_sync_ms)
    if not selected: raise RuntimeError("No frame set meets requested synchronization thresholds.")
    selected_lidar={sensor_topic("lidars",n,"points"):set() for n in LIDARS}; selected_image={sensor_topic("cameras",n,"image_raw"):set() for n in CAMERAS}
    for bundle in selected:
        for n,t in bundle["lidar"].items(): selected_lidar[sensor_topic("lidars",n,"points")].add(t)
        for n,t in bundle["image"].items(): selected_image[sensor_topic("cameras",n,"image_raw")].add(t)
    keep_topics=set(selected_lidar)|set(selected_image)|{sensor_topic("cameras",n,"camera_info") for n in CAMERAS}|{"/tf","/tf_static"}
    conns=[c for c in r.connections if c.topic in keep_topics]
    out=a.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing trial bag: {out}")
    if out.exists():
        out.rmdir()  # Only the empty directory this command may have created before Writer initialization.
    out.parent.mkdir(parents=True, exist_ok=True)
    stats=defaultdict(list); store=get_typestore(Stores.ROS2_JAZZY)
    writer=Writer(out,version=9,storage_plugin=StoragePlugin.MCAP); writer.set_compression(CompressionMode.STORAGE,CompressionFormat.ZSTD)
    with writer:
        cmap={c.id:writer.add_connection(c.topic,c.msgtype,typestore=store,serialization_format=c.ext.serialization_format,offered_qos_profiles=c.ext.offered_qos_profiles) for c in conns}
        info_written=set()
        for c,stamp,raw in r.messages(conns):
            if c.topic in selected_lidar:
                if stamp not in selected_lidar[c.topic]: continue
                msg,one=clean_cloud_message(store.deserialize_cdr(raw,c.msgtype)); raw=store.serialize_cdr(msg,c.msgtype); stats[c.topic].append(one)
            elif c.topic in selected_image:
                if stamp not in selected_image[c.topic]: continue
            elif c.topic.endswith("/camera_info"):
                if c.topic in info_written: continue
                info_written.add(c.topic)
            writer.write(cmap[c.id],stamp,raw)
    r.close()
    totals={}
    for topic,items in stats.items():
        totals[topic]={k:(sum(x[k] for x in items) if k != "retention" else sum(x["deduplicated_points"] for x in items)/max(1,sum(x["raw_points"] for x in items))) for k in ("raw_points","finite_nonzero_points","deduplicated_points","retention")}
    manifest={"source_mcap":str(a.mcap),"output_bag":str(out),"policy":"finite xyz + nonzero range + 0.1 mm dedup only; SOR/ROR/range/voxel/intensity filters disabled","thresholds_ms":{"lidar":a.lidar_sync_ms,"image":a.image_sync_ms},"frame_sets":[{"anchor_ns":b["anchor_ns"],"lidar_delta_ms":{n:round((t-b["anchor_ns"])/1e6,3) for n,t in b["lidar"].items()},"image_delta_ms":{n:round((t-b["anchor_ns"])/1e6,3) for n,t in b["image"].items()}} for b in selected],"point_retention":totals}
    (out/"calibration_trial_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    lines=["# Offline calibration trial input", "", f"Frame sets: {len(selected)}. Point policy: {manifest['policy']}", "", "## Per-stream point retention", "", "|topic|raw|finite/nonzero|deduplicated|retention|", "|---|---:|---:|---:|---:|"]
    lines += [f"|{t}|{x['raw_points']}|{x['finite_nonzero_points']}|{x['deduplicated_points']}|{x['retention']:.3%}|" for t,x in totals.items()]
    lines += ["", "## Synchronization frame list", "", "See `calibration_trial_manifest.json` for each exact MCAP timestamp and all 13 stream deltas."]
    (out/"calibration_trial_manifest.md").write_text("\n".join(lines)+"\n",encoding="utf8")

if __name__ == "__main__": main()
