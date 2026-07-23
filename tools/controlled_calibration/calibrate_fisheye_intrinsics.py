#!/usr/bin/env python3
"""Fit versioned fisheye intrinsics only from controlled checkerboard images."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CAMERAS = ("front", "back", "left", "left2", "right", "right2")
BOARD = (7, 5)  # internal corners, from the 8x6 target
SQUARE_M = 0.65

def images_from_bag(bag: Path):
    reader = rosbag2_py.SequentialReader(); reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {x.name:x.type for x in reader.get_all_topics_and_types()}; wanted = {f'/wamv/sensors/cameras/{c}_camera_sensor/image_raw':c for c in CAMERAS}
    bridge, result = CvBridge(), {}
    while reader.has_next() and len(result) < len(CAMERAS):
        topic, data, _ = reader.read_next(); camera = wanted.get(topic)
        if camera and camera not in result:
            result[camera] = bridge.imgmsg_to_cv2(deserialize_message(data, get_message(types[topic])), desired_encoding='bgr8')
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--bags', type=Path, nargs='+', required=True); p.add_argument('--output', type=Path, required=True); args=p.parse_args()
    object_grid=np.zeros((BOARD[0]*BOARD[1],1,3),np.float64); object_grid[:,0,:2]=np.mgrid[0:BOARD[0],0:BOARD[1]].T.reshape(-1,2)*SQUARE_M
    points={c:[] for c in CAMERAS}; images={c:[] for c in CAMERAS}; size=None
    for bag in args.bags:
        for camera,image in images_from_bag(bag).items():
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); ok,corners=cv2.findChessboardCornersSB(gray,BOARD,flags=cv2.CALIB_CB_EXHAUSTIVE)
            if ok:
                points[camera].append(object_grid); images[camera].append(corners.astype(np.float64)); size=(image.shape[1],image.shape[0])
    cameras={}
    for camera in CAMERAS:
        item={'source':'controlled_checkerboard_images','model':'fisheye','version':'controlled-fisheye-v1','image_orientation':'native_ros_image_no_flip_no_rotate','views':len(points[camera])}
        if len(points[camera]) < 3:
            item['accepted']=False; item['reason']='fewer_than_three_independent_checkerboard_views'; cameras[camera]=item; continue
        K=np.eye(3); D=np.zeros((4,1));
        try:
            rms,K,D,_,_=cv2.fisheye.calibrate(points[camera],images[camera],size,K,D,None,None,cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC|cv2.fisheye.CALIB_FIX_SKEW,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-8))
            item.update({'accepted':bool(np.isfinite(rms) and rms <= 3.0),'rms_px':float(rms),'K':K.tolist(),'D':D.reshape(-1).tolist()})
            if not item['accepted']: item['reason']='rms_exceeds_3px'
        except cv2.error as exc:
            item.update({'accepted':False,'reason':str(exc)})
        cameras[camera]=item
    result={'schema':'controlled-camera-intrinsics/v1','cameras':cameras,'board':{'internal_corners':BOARD,'square_m':SQUARE_M},'bags':[str(x) for x in args.bags]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({c:{'accepted':v['accepted'],'views':v['views'],'rms_px':v.get('rms_px')} for c,v in cameras.items()},indent=2))
    raise SystemExit(0 if all(v['accepted'] for v in cameras.values()) else 2)
if __name__=='__main__': main()
