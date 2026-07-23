#!/usr/bin/env python3
"""Offline-only validation of buoy colour masks against simulator truth.

This is an independent camera/TF check for the WAM-V simulation bags.  It
projects `/world/obstacles/buoys` using the contemporaneous odom->base TF and
the camera TF, then measures pixel distance to orange/black buoy masks.
It does not claim LiDAR-camera calibration; it supplies a labelled-image
validation target that the generic edge-distance proxy cannot provide.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CAMS=("front","back","left","left2","right","right2")

def qmat(q):
 x,y,z,w=np.asarray(q,float)/np.linalg.norm(q); return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def T(t,q):
 m=np.eye(4);m[:3,:3]=qmat(q);m[:3,3]=t;return m
def inv(m):
 o=np.eye(4);o[:3,:3]=m[:3,:3].T;o[:3,3]=-m[:3,:3].T@m[:3,3];return o
def apply(m,p):return p@m[:3,:3].T+m[:3,3]
def near(items,t):return min(items,key=lambda x:abs(x[0]-t))
def image(m):
 a=np.frombuffer(m.data,np.uint8).reshape(m.height,m.step)[:,:m.width*3].reshape(m.height,m.width,3)
 return a[:,:,::-1].copy() if m.encoding.lower()=="rgb8" else a.copy()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bag-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(a.bag_dir),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
 types={x.name:x.type for x in reader.get_all_topics_and_types()}; images={n:[] for n in CAMS};infos={};buoys=[];base=[];cams={}
 wanted={'/world/obstacles/buoys','/tf'}
 for n in CAMS:wanted |= {f'/wamv/sensors/cameras/{n}_camera_sensor/image_raw',f'/wamv/sensors/cameras/{n}_camera_sensor/camera_info'}
 if hasattr(reader, 'set_filter'):
  reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted)))
 while reader.has_next():
  topic,raw,stamp=reader.read_next()
  if topic not in wanted or topic not in types: continue
  m=deserialize_message(raw,get_message(types[topic]))
  if topic=='/world/obstacles/buoys':
   if len(m.poses):buoys.append((stamp,np.array([[p.position.x,p.position.y,p.position.z] for p in m.poses],float)))
  elif topic=='/tf':
   for x in m.transforms:
    if x.header.frame_id=='wamv/odom' and x.child_frame_id=='wamv/base_link':base.append((stamp,T([x.transform.translation.x,x.transform.translation.y,x.transform.translation.z],[x.transform.rotation.x,x.transform.rotation.y,x.transform.rotation.z,x.transform.rotation.w])))
    for n in CAMS:
     if x.child_frame_id.endswith(f'/{n}_camera_sensor'):cams[n]=T([x.transform.translation.x,x.transform.translation.y,x.transform.translation.z],[x.transform.rotation.x,x.transform.rotation.y,x.transform.rotation.z,x.transform.rotation.w])
  else:
   n=next((n for n in CAMS if topic==f'/wamv/sensors/cameras/{n}_camera_sensor/image_raw'),None)
   if n:images[n].append((stamp,image(m)));continue
   n=next((n for n in CAMS if topic==f'/wamv/sensors/cameras/{n}_camera_sensor/camera_info'),None)
   if n and n not in infos:infos[n]=(np.array(m.k,float),int(m.width),int(m.height))
 results=[]
 for n in CAMS:
  if not images[n] or n not in infos or n not in cams or not buoys or not base:continue
  stamp,img=images[n][len(images[n])//2]; _,world=near(buoys,stamp);_,tb=near(base,stamp); k,w,h=infos[n]
  pc=apply(inv(cams[n])@inv(tb),world) # world -> camera sensor
  pc=pc@np.array([[0,0,1],[-1,0,0],[0,-1,0]],float).T; good=pc[:,2]>.1;pc=pc[good]
  uv=np.column_stack([k[0]*pc[:,0]/pc[:,2]+k[2],k[4]*pc[:,1]/pc[:,2]+k[5]]);good=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv=uv[good]
  hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); orange=cv2.inRange(hsv,np.array([4,80,50]),np.array([28,255,255])); black=cv2.inRange(hsv,np.array([0,0,0]),np.array([180,255,75])); mask=cv2.bitwise_or(orange,black);mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); dist=cv2.distanceTransform(255-mask,cv2.DIST_L2,3)
  d=dist[uv[:,1].astype(int),uv[:,0].astype(int)] if len(uv) else np.array([]);vis=img.copy()
  for u,v in uv:cv2.drawMarker(vis,(round(u),round(v)),(255,255,0),cv2.MARKER_CROSS,12,1)
  fn=f'groundtruth_projection_{n}.jpg';cv2.imwrite(str(a.output/fn),vis)
  results.append({'camera':n,'timestamp_ns':stamp,'visible_groundtruth_buoys':int(len(uv)),'mean_target_mask_distance_px':float(d.mean()) if len(d) else None,'median_target_mask_distance_px':float(np.median(d)) if len(d) else None,'visualization':fn,'note':'Ground-truth buoy point to orange/black mask distance; the object center/extent and color ambiguity make this a validation metric, not a LiDAR-camera residual.'})
 report={'bag_dir':str(a.bag_dir),'purpose':'offline_evaluation_only','intrinsics_source':'MCAP CameraInfo','groundtruth_topic':'/world/obstacles/buoys','results':results};(a.output/'groundtruth_projection_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 # Avoid format-condition expressions inside the f-string format mini-language.
 lines=['# Ground-truth buoy projection validation','','|camera|visible buoys|mean px|median px|visualization|','|---|---:|---:|---:|---|']+[f"|{x['camera']}|{x['visible_groundtruth_buoys']}|{('%.2f' % x['mean_target_mask_distance_px']) if x['mean_target_mask_distance_px'] is not None else 'n/a'}|{('%.2f' % x['median_target_mask_distance_px']) if x['median_target_mask_distance_px'] is not None else 'n/a'}|{x['visualization']}|" for x in results]
 (a.output/'groundtruth_projection_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':main()
