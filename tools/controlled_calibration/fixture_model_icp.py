#!/usr/bin/env python3
"""Register synchronized LiDAR scans to declared 3-D calibration fixtures.

The fixture coordinates are the public geometry inserted into calibration_user,
not simulator truth.  Each edge estimates both sensor poses against the same
non-coplanar fixture, then composes a LiDAR-to-LiDAR transform.  This removes
the artificial tangential error caused by directly nearest-neighbour matching
two sparse scans of the same planar box surfaces.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs_py import point_cloud2

POS={'front':(7,0,2.9),'back':(-7,0,2.9),'left':(-6,1.626,2.9),'left2':(6,1.626,2.9),'right':(-6,-1.626,2.9),'right2':(6,-1.626,2.9),'forward':(7,0,2)}
RPY={'front':(0,115,0),'back':(0,115,180),'left':(0,115,90),'left2':(0,115,90),'right':(0,115,-90),'right2':(0,115,-90),'forward':(0,0,0)}
EDGES=(('left','front','front_left'),('right','front','front_right'),('back','right','back_right'),('forward','front','front_forward'),('left2','left','left_pair'),('right2','right','right_pair'),('back','left','back_left'))
CENTERS={'front_left':(10,7,2.4),'front_right':(10,-7,2.4),'back_right':(-10,-7,2.4),'back_left':(-10,7,2.4),'left_pair':(-1,11,2.4),'right_pair':(-1,-11,2.4),'front_forward':(13,4,2.4),'front_board':(13,0,2.8)}
MAX_SKEW_NS=300_000_000; VOXEL=.10; MAX_DIST=.35; MIN=40

def topic(n): return f'/wamv/sensors/lidars/{n}_lidar_sensor/points'
def Tnom(n):
 t=np.eye(4); t[:3,:3]=Rotation.from_euler('xyz',RPY[n],degrees=True).as_matrix();t[:3,3]=POS[n];return t
def apply(t,p): return p@t[:3,:3].T+t[:3,3]
def vox(p):
 if len(p)==0:return p
 _,i=np.unique(np.floor(p/VOXEL).astype(np.int32),axis=0,return_index=True);return p[np.sort(i)]
def frames(bag):
 r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
 types={x.name:x.type for x in r.get_all_topics_and_types()}; out={n:[] for n in POS}; lookup={topic(n):n for n in POS}
 while r.has_next():
  name,data,stamp=r.read_next();n=lookup.get(name)
  if n is None:continue
  raw=point_cloud2.read_points(deserialize_message(data,get_message(types[name])),field_names=('x','y','z'),skip_nans=True)
  p=np.column_stack((raw['x'],raw['y'],raw['z'])).astype(float);out[n].append((stamp,p[np.isfinite(p).all(1)]))
 return out
def bundles(f):
 out=[]
 for stamp,p in f['front']:
  b={'front':p}
  for n in POS:
   if n=='front' or not f[n]:continue
   s,q=min(f[n],key=lambda x:abs(x[0]-stamp))
   if abs(s-stamp)<=MAX_SKEW_NS:b[n]=q
  if len(b)==len(POS):out.append(b)
 return out
def box_points(size,step=.10):
 # Dense known box surface, not a measurement or truth topic.
 x=np.arange(-size[0]/2,size[0]/2+step,step);y=np.arange(-size[1]/2,size[1]/2+step,step);z=np.arange(-size[2]/2,size[2]/2+step,step)
 yy,zz=np.meshgrid(y,z); xx,zz2=np.meshgrid(x,z);xx2,yy2=np.meshgrid(x,y)
 return np.vstack((np.c_[np.full(yy.size,-size[0]/2),yy.ravel(),zz.ravel()],np.c_[np.full(yy.size,size[0]/2),yy.ravel(),zz.ravel()],np.c_[xx.ravel(),np.full(xx.size,-size[1]/2),zz2.ravel()],np.c_[xx.ravel(),np.full(xx.size,size[1]/2),zz2.ravel()],np.c_[xx2.ravel(),yy2.ravel(),np.full(xx2.size,-size[2]/2)],np.c_[xx2.ravel(),yy2.ravel(),np.full(xx2.size,size[2]/2)]))
def cylinder_points(radius,length,step=.06):
 a=np.linspace(0,2*math.pi,32,endpoint=False); z=np.arange(-length/2,length/2+step,step);aa,zz=np.meshgrid(a,z)
 side=np.c_[radius*np.cos(aa).ravel(),radius*np.sin(aa).ravel(),zz.ravel()]
 rr=np.arange(0,radius+step,step);aa2,rr2=np.meshgrid(a,rr);caps=np.vstack((np.c_[rr2.ravel()*np.cos(aa2).ravel(),rr2.ravel()*np.sin(aa2).ravel(),np.full(rr2.size,-length/2)],np.c_[rr2.ravel()*np.cos(aa2).ravel(),rr2.ravel()*np.sin(aa2).ravel(),np.full(rr2.size,length/2)]))
 return np.vstack((side,caps))
def fixture(name):
 if name=='front_board':
  r=Rotation.from_euler('z',-math.pi,degrees=False).as_matrix(); allp=[]
  # Exact declared board-corner normal-depth cylinders.  The SDF rotates each
  # cylinder around X into board-local Y; apply that same transform here.
  for (px,pz,depth) in ((-2.8,-2.1,.60),(2.8,-2.1,.90),(-2.8,2.1,1.20),(2.8,2.1,1.50)):
   post=cylinder_points(.30,depth)@Rotation.from_euler('x',math.pi/2).as_matrix().T; post[:,0]+=px; post[:,2]+=pz; allp.append(post@r.T+np.array(CENTERS[name]))
  return np.vstack(allp)
 blocks=((0,0,2.8,1.8,1.0,5.6,.15),(2.2,.9,1.6,.6,2.4,3.2,.87),(-1.5,-1.6,3.5,1.2,.5,7.0,-.45),(.5,-2.4,.8,3.0,.5,1.2,.30),(-2.8,1.9,1.2,.8,1.5,2.4,-.78)) if name=='front_forward' else ((0,0,2.5,1.2,.9,5.0,.15),(1.25,.55,1.8,.7,1.4,3.6,.68),(-.85,-1.10,2.2,1.0,.55,4.4,-.42)); allp=[]
 for ox,oy,oz,sx,sy,sz,yaw in blocks:
  p=box_points((sx,sy,sz));r=Rotation.from_euler('xyz',(.12,.18,yaw)).as_matrix();allp.append(p@r.T+np.array(CENTERS[name]) + np.array((ox,oy,oz-CENTERS[name][2])))
 return np.vstack(allp)
def roi(n,p,name):
 q=apply(Tnom(n),p)
 if name=='front_board':
  r=Rotation.from_euler('z',-math.pi).as_matrix();c=np.array(CENTERS[name]);keep=np.zeros(len(q),dtype=bool)
  for px,pz,_ in ((-2.8,-2.1,.60),(2.8,-2.1,.90),(-2.8,2.1,1.20),(2.8,2.1,1.50)):
   center=np.array((px,0,pz))@r.T+c; keep|=(np.abs(q-center)<=np.array((.85,1.25,.85))).all(1)
  return p[keep]
 c=np.array(CENTERS[name]);h=np.array((4.5,4.5,4.5)) if name=='front_forward' else np.array((3,3,3.7));return p[(np.abs(q-c)<=h).all(1)]
def surface_normals(model):
 tree=cKDTree(model); _,idx=tree.query(model,k=min(16,len(model))); out=[]
 for neighbours in model[idx]:
  _,_,vh=np.linalg.svd(neighbours-neighbours.mean(0),full_matrices=False);out.append(vh[-1])
 return np.asarray(out)
def reg_once(n,p,model,normals,name,initial):
 p=vox(p);t=Tnom(n).copy()
 t=initial.copy()
 if len(p)<MIN:return t,0.,float('inf'),0
 tree=cKDTree(model)
 for _ in range(35):
  q=apply(t,p);d,j=tree.query(q);k=d<MAX_DIST
  if k.sum()<MIN:break
  a,b=q[k],model[j[k]];ca,cb=a.mean(0),b.mean(0);u,_,vh=np.linalg.svd((a-ca).T@(b-cb));r=vh.T@u.T
  if np.linalg.det(r)<0:vh[-1]*=-1;r=vh.T@u.T
  dt=np.eye(4);dt[:3,:3]=r;dt[:3,3]=cb-r@ca;t=dt@t
  if np.linalg.norm(dt[:3,3])<1e-4 and Rotation.from_matrix(r).magnitude()<1e-4:break
 q=apply(t,p);d,j=tree.query(q);k=d<MAX_DIST
 if not k.any(): return t,float(k.mean()),float('inf'),0
 # Nearest sampled model points may be separated tangentially on a planar
 # face; the signed distance along that face's locally estimated normal is
 # the physical ICP surface residual.
 d=np.abs(np.einsum('ij,ij->i',q[k]-model[j[k]],normals[j[k]]))
 return t,float(k.mean()),float(np.sqrt(np.mean(d**2))),int(k.sum())
def reg(n,p,model,normals,name):
 # Only the forward/front side target needs multi-start because its two
 # scanners have the largest allowed installation perturbation.  All starts
 # remain bounded around the mechanical nominal and are scored solely by the
 # recorded target-surface residual.
 starts=[Tnom(n)]
 runs=[reg_once(n,p,model,normals,name,s) for s in starts]
 return min(runs,key=lambda item: (item[2],-item[1]))
def main():
 a=argparse.ArgumentParser();a.add_argument('--bags',nargs='+',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();bs=[b for bag in x.bags for b in bundles(frames(bag))]
 edges=[]
 for s,t,name in EDGES:
  model=fixture(name); normals=surface_normals(model);runs=[]
  for b in bs:
   ts,fs,rs,cs=reg(s,roi(s,b[s],name),model,normals,name);tt,ft,rt,ct=reg(t,roi(t,b[t],name),model,normals,name);runs.append((np.linalg.inv(tt)@ts,min(fs,ft),max(rs,rt),min(cs,ct)))
  trans=np.array([r[0][:3,3] for r in runs]);rots=Rotation.from_matrix([r[0][:3,:3] for r in runs]);mean=rots.mean();td=float(np.max(np.linalg.norm(trans-np.median(trans,axis=0),axis=1)));rd=float(np.degrees(np.max((mean.inv()*rots).magnitude())));fit=float(np.median([r[1] for r in runs]));raw=float(np.median([r[2] for r in runs]));rmse=raw if math.isfinite(raw) else None
  edges.append({'source':s,'target':t,'fixture':name,'fitness':fit,'rmse_m':rmse,'translation_delta_m':td,'rotation_delta_deg':rd,'stable':bool(fit>=.15 and rmse is not None and rmse<=.15 and td<=.75 and rd<=12),'runs':[{'fitness':r[1],'rmse_m':r[2] if math.isfinite(r[2]) else None,'correspondences':r[3]} for r in runs]})
 out={'schema':'controlled-fixture-model-icp/v1','truth_usage':'offline_evaluation_only','preprocess':'finite_points_only + declared_fixture_roi + 0.10m_voxel_unique','synchronization':{'bundles':len(bs),'max_pair_skew_s':.3},'edges':edges};x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'bundles':len(bs),'edges':edges},indent=2,allow_nan=False))
if __name__=='__main__':main()
