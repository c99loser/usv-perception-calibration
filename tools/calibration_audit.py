#!/usr/bin/env python3
"""Audit corrected MCAP inputs before LiDAR/camera self-calibration.

Reads the MCAP directly (so a stale rosbag2 metadata filename cannot mask the
actual data), never applies SOR/ROR, and writes a self-contained JSON/Markdown
report plus six LiDAR-to-camera projection images.
"""
from __future__ import annotations

import argparse, itertools, json, math
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2.reader import McapReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial import cKDTree

CAMS = ("front", "back", "left", "left2", "right", "right2")
LIDARS = ("front", "back", "left", "left2", "right", "right2", "forward")

def json_safe(value):
    """Convert NumPy scalars and non-finite diagnostics to strict JSON values."""
    if isinstance(value, dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value

def topic(kind, name):
    return f"/wamv/sensors/{kind}/{name}_{'camera' if kind == 'cameras' else 'lidar'}_sensor"

def quat_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def transform(t, q):
    out = np.eye(4); out[:3,:3] = quat_matrix(np.asarray(q, float)); out[:3,3] = t; return out

def invert(t):
    out = np.eye(4); out[:3,:3] = t[:3,:3].T; out[:3,3] = -t[:3,:3].T @ t[:3,3]; return out

def apply(t, p): return p @ t[:3,:3].T + t[:3,3]

def clean_deduplicate(points, resolution=1e-4):
    """Only remove nonfinite/zero points and repeated 0.1-mm cells; no SOR/ROR."""
    finite = np.isfinite(points).all(axis=1) & (np.einsum("ij,ij->i", points, points) > 0)
    points = points[finite]
    if not len(points): return points, {"raw": len(finite), "finite": 0, "deduplicated": 0}
    keys = np.floor(points / resolution + 0.5).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    out = points[np.sort(keep)]
    return out, {"raw": len(finite), "finite": len(points), "deduplicated": len(out)}

def cloud_xyz(msg):
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")): return np.empty((0,3), np.float32)
    # ROS PointField FLOAT32 is datatype 7; simulation clouds use packed little-endian float32.
    if any(fields[k].datatype != 7 for k in ("x", "y", "z")): raise ValueError("x/y/z are not float32")
    n = msg.width * msg.height; data = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    return np.column_stack([data[:, fields[k].offset:fields[k].offset+4].copy().view('<f4').ravel() for k in ("x","y","z")])

def rigid(a, b):
    ca, cb = a.mean(0), b.mean(0); h = (a-ca).T @ (b-cb); u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0: vt[-1] *= -1; r = vt.T @ u.T
    out = np.eye(4); out[:3,:3] = r; out[:3,3] = cb-r@ca; return out

def icp(source, target, initial, iterations=20):
    # Deterministic thinning keeps all geometric regions without a filtering policy.
    source = source[::max(1, len(source)//12000)]; target = target[::max(1, len(target)//16000)]
    tree = cKDTree(target); current = initial.copy(); pairs = 0; rmse = float("inf")
    for threshold in (1.5, 1.0, .6, .35):
        for _ in range(iterations//4):
            moved = apply(current, source); d, ix = tree.query(moved, k=1, distance_upper_bound=threshold)
            good = np.isfinite(d)
            if good.sum() < 30: break
            delta = rigid(moved[good], target[ix[good]])
            current = delta @ current; pairs = int(good.sum()); rmse = float(np.sqrt(np.mean(d[good]**2)))
    moved = apply(current, source); d, _ = tree.query(moved, k=1, distance_upper_bound=.35); good = np.isfinite(d)
    pairs = int(good.sum()); fitness = pairs / max(1, len(source)); rmse = float(np.sqrt(np.mean(d[good]**2))) if pairs else float("inf")
    delta = current @ invert(initial); angle = math.degrees(math.acos(np.clip((np.trace(delta[:3,:3])-1)/2, -1, 1)))
    return current, {"overlap_points": pairs, "fitness": fitness, "rmse_m": rmse,
                     "delta_translation_m": float(np.linalg.norm(delta[:3,3])), "delta_rotation_deg": angle}

def nearest(items, stamp): return min(items, key=lambda x: abs(x[0]-stamp))

def decode_image(msg):
    channels = 3 if msg.encoding.lower() in ("bgr8", "rgb8") else 1
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width*channels]
    a = a.reshape(msg.height, msg.width, channels)
    return a[:,:,::-1].copy() if msg.encoding.lower()=="rgb8" else (cv2.cvtColor(a,cv2.COLOR_GRAY2BGR) if channels==1 else a.copy())

def project(points_lidar, t_base_lidar, t_base_camera, k):
    # Gazebo sensor link: x forward, y left, z up.  ROS optical: x right, y down, z forward.
    t_cam_lidar = invert(t_base_camera) @ t_base_lidar
    p = apply(t_cam_lidar, points_lidar)
    p = p @ np.array([[0,0,1],[-1,0,0],[0,-1,0]], float).T
    good = p[:,2] > .1; p = p[good]
    uv = np.column_stack((k[0]*p[:,0]/p[:,2]+k[2], k[4]*p[:,1]/p[:,2]+k[5]))
    return uv, p[:,2]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mcap", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=24); args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    r = McapReader(args.mcap); r.open(); store = get_typestore(Stores.ROS2_JAZZY)
    conns = {c.topic:c for c in r.connections}; frames = {n:[] for n in LIDARS}; images={n:[] for n in CAMS}; infos={}; transforms={}
    wanted=[]
    for n in LIDARS: wanted.append(conns.get(topic("lidars",n)+"/points"))
    for n in CAMS:
        wanted += [conns.get(topic("cameras",n)+"/image_raw"), conns.get(topic("cameras",n)+"/camera_info")]
    wanted += [conns.get("/tf")]; wanted=[x for x in wanted if x]
    for c, stamp, raw in r.messages(wanted):
        n = next((x for x in LIDARS if c.topic == topic("lidars",x)+"/points"), None)
        if n:
            if len(frames[n]) < args.max_frames:
                pts, stats = clean_deduplicate(cloud_xyz(store.deserialize_cdr(raw,c.msgtype))); frames[n].append((stamp,pts,stats))
            continue
        n = next((x for x in CAMS if c.topic == topic("cameras",x)+"/image_raw"), None)
        if n:
            if len(images[n]) < args.max_frames: images[n].append((stamp,decode_image(store.deserialize_cdr(raw,c.msgtype))))
            continue
        n = next((x for x in CAMS if c.topic == topic("cameras",x)+"/camera_info"), None)
        if n and n not in infos:
            m=store.deserialize_cdr(raw,c.msgtype); infos[n]={"frame_id":m.header.frame_id,"width":int(m.width),"height":int(m.height),"model":m.distortion_model,"K":[float(x) for x in m.k],"D":[float(x) for x in m.d]}; continue
        if c.topic == "/tf":
            m=store.deserialize_cdr(raw,c.msgtype)
            for x in m.transforms:
                child=x.child_frame_id
                for n in CAMS:
                    if child.endswith(f"/{n}_camera_sensor"):
                        transforms[f"camera:{n}"]=transform([x.transform.translation.x,x.transform.translation.y,x.transform.translation.z],[x.transform.rotation.x,x.transform.rotation.y,x.transform.rotation.z,x.transform.rotation.w])
                for n in LIDARS:
                    if child.endswith(f"/{n}_lidar_sensor"):
                        transforms[f"lidar:{n}"]=transform([x.transform.translation.x,x.transform.translation.y,x.transform.translation.z],[x.transform.rotation.x,x.transform.rotation.y,x.transform.rotation.z,x.transform.rotation.w])
    r.close()
    report={"mcap":str(args.mcap),"pointcloud_policy":"finite xyz + nonzero range + 0.1 mm dedup only; SOR/ROR disabled","thresholds":{"lidar_sync_ms":50,"min_overlap_points":100,"min_fitness":.03,"max_rmse_m":.5,"max_delta_translation_m":1,"max_delta_rotation_deg":10,"camera_sync_ms":50,"min_projected_points":100,"max_edge_proxy_median_px":5},"camera_info":infos,"image_direction":{"topic_selection":"direct /<camera>_camera_sensor/image_raw and matching direct camera_info only; optical topics excluded because their K principal point (160,120) is inconsistent with the 1280x720 image","sensor_to_optical":"sensor (x forward, y left, z up) -> optical (x=-y, y=-z, z=x)","verification":"six saved overlays are the visual direction check"},"lidar_pairs":[],"camera_pairs":[]}
    # Score every retained synchronized frame.  A pair is exportable only when
    # at least 80% of its samples pass; one accidental ICP convergence cannot
    # promote an external parameter.
    for a,b in itertools.combinations(LIDARS,2):
        if not frames[a] or not frames[b] or f"lidar:{a}" not in transforms or f"lidar:{b}" not in transforms: continue
        initial=invert(transforms[f"lidar:{b}"])@transforms[f"lidar:{a}"]; samples=[]
        for fa in frames[a]:
            fb=nearest(frames[b],fa[0]); _, one=icp(fa[1],fb[1],initial)
            one.update({"timestamp_delta_ms":abs(fa[0]-fb[0])/1e6,"a_points":fa[2],"b_points":fb[2]})
            one["accepted"] = one["timestamp_delta_ms"] <= 50 and one["overlap_points"] >= 100 and one["fitness"] >= .03 and one["rmse_m"] <= .5 and one["delta_translation_m"] <= 1 and one["delta_rotation_deg"] <= 10
            samples.append(one)
        def median(key): return float(np.median([x[key] for x in samples]))
        metric={"lidar_a":a,"lidar_b":b,"sample_count":len(samples),"accepted_samples":sum(x["accepted"] for x in samples),"timestamp_delta_ms":median("timestamp_delta_ms"),"overlap_points":int(round(median("overlap_points"))),"fitness":median("fitness"),"rmse_m":median("rmse_m"),"delta_translation_m":median("delta_translation_m"),"delta_rotation_deg":median("delta_rotation_deg"),"samples":samples}
        metric["accepted"] = metric["accepted_samples"] >= math.ceil(.8*len(samples))
        metric["rejection"] = None if metric["accepted"] else "fewer than 80% of samples pass timestamp/overlap/fitness/RMSE/transform thresholds"
        report["lidar_pairs"].append(metric)
    for n in CAMS:
        if not frames[n] or not images[n] or n not in infos or f"lidar:{n}" not in transforms or f"camera:{n}" not in transforms: continue
        f=frames[n][len(frames[n])//2]; imstamp,image=nearest(images[n],f[0]); uv,z=project(f[1],transforms[f"lidar:{n}"],transforms[f"camera:{n}"],infos[n]["K"])
        h,w=image.shape[:2]; good=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h); uv=uv[good]; z=z[good]
        edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),80,160); dist=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
        px=dist[np.clip(uv[:,1].astype(int),0,h-1),np.clip(uv[:,0].astype(int),0,w-1)] if len(uv) else np.array([])
        vis=image.copy()
        for (u,v), depth in zip(uv[::max(1,len(uv)//4000)], z[::max(1,len(z)//4000)]): cv2.circle(vis,(round(u),round(v)),1,(0,int(max(0,255-min(255,depth*12))),255),-1)
        path=args.output/f"projection_{n}.jpg"; cv2.imwrite(str(path),vis)
        median=float(np.median(px)) if len(px) else None
        timed=abs(imstamp-f[0])<=50_000_000
        proxy_ok=median is not None and median<=5
        reason="requires labelled 2D–3D correspondences for a true reprojection-error gate" if timed and len(uv)>=100 and proxy_ok else "sync/projected-point/edge-proxy threshold failed"
        report["camera_pairs"].append({"camera":n,"lidar":n,"timestamp_delta_ms":abs(imstamp-f[0])/1e6,"projected_points":int(len(uv)),"edge_distance_mean_px":float(px.mean()) if len(px) else None,"edge_distance_median_px":median,"visualization":path.name,"accepted_for_extrinsic":False,"rejection":reason,"note":"Edge distance is a projection-consistency proxy, not labelled 2D-3D reprojection error."})
    accepted= [x for x in report["lidar_pairs"] if x["accepted"]]
    accepted_camera = [x for x in report["camera_pairs"] if x["accepted_for_extrinsic"]]
    report["closure"]={"evaluated_cycles":0,"status":"not_evaluated","reason":"No three-edge cycle has passed pairwise gates." if not accepted else "Cycle evaluation requires three accepted directed edges."}
    report["extrinsic_admission"]={"offline_trial_input_ready":True,"export_calibrated_extrinsics":False,"accepted_lidar_pairwise_edges":[f"{x['lidar_a']}–{x['lidar_b']}" for x in accepted],"accepted_lidar_camera_pairs":[f"{x['lidar']}–{x['camera']}" for x in accepted_camera],"reason":"A global LiDAR solution requires validated closed loops, and every camera pair requires independent projection validation; those conditions are not met.","fallback":"All transforms below are nominal TF values only; they are explicitly not calibrated."}
    report["nominal_unverified_transforms"]={name:{"status":"nominal_unverified","T_base_sensor":transforms[name].round(12).tolist()} for name in transforms}
    (args.output/"calibration_audit.json").write_text(json.dumps(json_safe(report),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    lines=["# Corrected MCAP calibration-input audit", "", "## Decision", "", "Camera extrinsics are **not accepted** by this audit: use only the accepted rows as self-calibration seeds; edge-distance is not a labelled reprojection error.", "", "## Image direction", "", "Used direct 1280×720 image + matching direct CameraInfo. Optical topics were excluded: their K has principal point (160,120), inconsistent with the image dimensions. Projection convention is sensor (x forward, y left, z up) → optical (x=-y, y=-z, z=x); inspect the six overlays before changing the convention.", "", "## CameraInfo", "", "|camera|model|K fx,fy,cx,cy|D|", "|---|---|---|---|"]
    lines += [f"|{n}|{v['model']}|{v['K'][0]:.3f}, {v['K'][4]:.3f}, {v['K'][2]:.1f}, {v['K'][5]:.1f}|{v['D']}|" for n,v in infos.items()]
    lines += ["", "## LiDAR ICP", "", "Each row is the median of all retained synchronized frames; export requires at least 80% sample-level passes.", "", "|pair|samples passed|sync ms|overlap|fitness|RMSE m|delta m|delta deg|accepted|", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    lines += [f"|{x['lidar_a']}–{x['lidar_b']}|{x['accepted_samples']}/{x['sample_count']}|{x['timestamp_delta_ms']:.2f}|{x['overlap_points']}|{x['fitness']:.4f}|{x['rmse_m']:.4f}|{x['delta_translation_m']:.4f}|{x['delta_rotation_deg']:.3f}|{x['accepted']}|" for x in report['lidar_pairs']]
    lines += ["", "## LiDAR–Camera projections", "", "|pair|sync ms|projected points|mean edge-distance px|accepted|", "|---|---:|---:|---:|---|"]
    lines += [f"|{x['lidar']}–{x['camera']}|{x['timestamp_delta_ms']:.2f}|{x['projected_points']}|{x['edge_distance_mean_px']:.3f}|{x['accepted_for_extrinsic']}|" for x in report['camera_pairs']]
    (args.output/"calibration_audit.md").write_text("\n".join(lines)+"\n",encoding="utf8")

if __name__ == "__main__": main()
