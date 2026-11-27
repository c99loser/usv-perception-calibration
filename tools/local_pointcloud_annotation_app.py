"""Offline blind annotation workbench for the raw full-FOV LiDAR training pack.

The tool reads only the training project, never opens the holdout project, and
writes sidecar JSON annotations without changing any source PCD or image.
"""
from __future__ import annotations

import argparse, json, struct, uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk

CLASSES = ("pillar", "buoy", "block", "boat", "background", "ambiguous")

def read_pcd(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = b""
        while b"DATA binary\n" not in header:
            line = f.readline()
            if not line: raise ValueError(f"Invalid PCD header: {path}")
            header += line
        lines = header.decode("ascii").splitlines()
        fields = next(x.split()[1:] for x in lines if x.startswith("FIELDS "))
        sizes = list(map(int, next(x.split()[1:] for x in lines if x.startswith("SIZE "))))
        types = next(x.split()[1:] for x in lines if x.startswith("TYPE "))
        count = int(next(x.split()[1] for x in lines if x.startswith("POINTS ")))
        dtype = np.dtype([(name, "<f4" if typ == "F" and size == 4 else "<f8") for name, size, typ in zip(fields, sizes, types)])
        raw = np.frombuffer(f.read(), dtype=dtype, count=count)
    return np.column_stack((raw["x"], raw["y"], raw["z"])).astype(np.float32)

class App:
    def __init__(self, root: Path, out: Path) -> None:
        self.root_dir, self.out_dir = root, out
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.out_dir / "session_manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({
                "purpose": "blind raw-full-FOV LiDAR annotation",
                "source_training_project": str(root),
                "holdout_opened": False,
                "annotation_format": "local_annotation_sidecar_v1",
                "rules": ["Do not use image pixels as a 3D projection.", "Do not inspect holdout before model freeze."],
            }, ensure_ascii=False, indent=2), encoding="utf8")
        self.episodes = sorted(p for p in root.iterdir() if p.is_dir())
        self.episode = self.episodes[0]; self.frame = 0; self.cloud = np.empty((0,3)); self.photo = None
        self.annotations: dict = {}; self.drag_start = None
        self.win = tk.Tk(); self.win.title("原始全视场 LiDAR 本地盲标工作台"); self.win.geometry("1550x940")
        bar = ttk.Frame(self.win, padding=6); bar.pack(fill="x")
        self.episode_var = tk.StringVar(value=self.episode.name)
        ttk.Label(bar, text="训练 Episode").pack(side="left")
        menu = ttk.Combobox(bar, textvariable=self.episode_var, values=[p.name for p in self.episodes], state="readonly", width=31); menu.pack(side="left", padx=4); menu.bind("<<ComboboxSelected>>", self.change_episode)
        ttk.Button(bar, text="◀ 前一帧", command=lambda:self.move(-1)).pack(side="left", padx=4)
        ttk.Button(bar, text="后一帧 ▶", command=lambda:self.move(1)).pack(side="left")
        self.info = ttk.Label(bar, text=""); self.info.pack(side="left", padx=12)
        form = ttk.Frame(self.win, padding=(6,0)); form.pack(fill="x")
        self.class_var=tk.StringVar(value="pillar"); self.track_var=tk.StringVar(value="track_001"); self.static_var=tk.StringVar(value="yes"); self.conf_var=tk.StringVar(value="medium"); self.z_var=tk.StringVar(value="0.0"); self.height_var=tk.StringVar(value="2.0")
        for label,var,vals in [("类别",self.class_var,CLASSES),("track_id",self.track_var,None),("静态性",self.static_var,("yes","no","uncertain")),("置信度",self.conf_var,("high","medium","low"))]:
            ttk.Label(form,text=label).pack(side="left",padx=(5,2))
            if vals: ttk.Combobox(form,textvariable=var,values=vals,state="readonly",width=12).pack(side="left")
            else: ttk.Entry(form,textvariable=var,width=16).pack(side="left")
        for label,var in (("中心Z(m)",self.z_var),("高度Z(m)",self.height_var)):
            ttk.Label(form,text=label).pack(side="left",padx=(5,2)); ttk.Entry(form,textvariable=var,width=7).pack(side="left")
        ttk.Label(form,text="拖拽左侧顶视图以画框；框仅限点云实际回波。",foreground="#9b4d00").pack(side="left",padx=14)
        body=ttk.Frame(self.win,padding=6); body.pack(fill="both",expand=True)
        self.canvas=tk.Canvas(body,width=900,height=810,bg="#07111d"); self.canvas.pack(side="left",fill="both",expand=True)
        right=ttk.Frame(body); right.pack(side="left",fill="both",padx=8)
        self.image_label=ttk.Label(right); self.image_label.pack()
        self.objects=tk.Listbox(right,width=56,height=29); self.objects.pack(fill="both",expand=True,pady=5)
        ttk.Button(right,text="删除选中框",command=self.delete).pack(fill="x")
        ttk.Button(right,text="保存当前 Episode",command=self.save).pack(fill="x",pady=4)
        ttk.Label(right,text="图片仅供场景理解，不能按像素位置推断点云对应。",wraplength=450,foreground="#9b4d00").pack()
        self.canvas.bind("<ButtonPress-1>",lambda e:setattr(self,"drag_start",(e.x,e.y)))
        self.canvas.bind("<ButtonRelease-1>",self.add_box)
        self.win.bind("<Left>",lambda e:self.move(-1)); self.win.bind("<Right>",lambda e:self.move(1)); self.win.protocol("WM_DELETE_WINDOW",self.close)
        self.load_episode()
    def annotation_path(self): return self.out_dir / "annotations" / f"{self.episode.name}.json"
    def load_episode(self):
        self.frame=0; p=self.annotation_path(); self.annotations=json.loads(p.read_text(encoding="utf8")) if p.exists() else {"schema_version":1,"episode_id":self.episode.name,"frames":{}}
        self.draw()
    def change_episode(self,_=None): self.save(); self.episode=next(p for p in self.episodes if p.name==self.episode_var.get()); self.load_episode()
    def move(self,d):
        maps=json.loads((self.episode/"frame_pointcloud_map.json").read_text()); self.frame=max(0,min(len(maps)-1,self.frame+d)); self.draw()
    def draw(self):
        maps=json.loads((self.episode/"frame_pointcloud_map.json").read_text()); pcd=self.episode/"pointcloud"/maps[str(self.frame)]; self.cloud=read_pcd(pcd)
        c=self.canvas; c.delete("all"); w,h=max(c.winfo_width(),900),max(c.winfo_height(),810); scale=min(w,h)/120; cx,cy=w/2,h/2
        for t in range(-60,61,10): c.create_line(cx+t*scale,0,cx+t*scale,h,fill="#173047"); c.create_line(0,cy-t*scale,w,cy-t*scale,fill="#173047")
        valid=np.isfinite(self.cloud).all(1); pts=self.cloud[valid][::max(1,valid.sum()//14000)]
        for x,y,_ in pts:
            px,py=cx+x*scale,cy-y*scale
            if 0<=px<w and 0<=py<h: c.create_rectangle(px,py,px+1,py+1,fill="#4bdde8",outline="")
        c.create_oval(cx-4,cy-4,cx+4,cy+4,fill="#ffb703",outline="")
        c.create_text(12,12,anchor="nw",fill="#e8f3ff",text=f"{self.episode.name} | frame {self.frame+1}/{len(maps)} | sensor-local XY, ±60m")
        self.draw_boxes(c,cx,cy,scale); self.load_image(maps[str(self.frame)].removesuffix(".pcd")); self.refresh_list(); self.info.config(text=f"{len(self.cloud):,} finite XYZ points")
    def draw_boxes(self,c,cx,cy,s):
        for a in self.annotations["frames"].get(str(self.frame),[]):
            x,y,_=a["center_xyz"]; dx,dy,_=a["dimensions_xyz"]; c.create_rectangle(cx+(x-dx/2)*s,cy-(y+dy/2)*s,cx+(x+dx/2)*s,cy-(y-dy/2)*s,outline="#ffb703",width=2)
    def load_image(self,key):
        folder=self.episode/"related_images"/key; files=list(folder.glob("*.jpg"));
        if files:
            im=Image.open(files[0]); im.thumbnail((560,430)); self.photo=ImageTk.PhotoImage(im); self.image_label.configure(image=self.photo)
    def add_box(self,e):
        if not self.drag_start or abs(e.x-self.drag_start[0])<5 or abs(e.y-self.drag_start[1])<5:return
        w,h=max(self.canvas.winfo_width(),900),max(self.canvas.winfo_height(),810); s=min(w,h)/120; cx,cy=w/2,h/2; x0,y0=self.drag_start; self.drag_start=None
        try: z, height = float(self.z_var.get()), float(self.height_var.get())
        except ValueError: messagebox.showerror("Z 参数错误","中心 Z 和高度必须是数字。"); return
        a,b=sorted((x0,e.x)); d,f=sorted((y0,e.y)); x,y=round(((a+b)/2-cx)/s,3),round((cy-(d+f)/2)/s,3); dx,dy=round((b-a)/s,3),round((f-d)/s,3)
        box={"id":str(uuid.uuid4()),"class_name":self.class_var.get(),"track_id":self.track_var.get().strip(),"static_for_calibration":self.static_var.get(),"reviewer_confidence":self.conf_var.get(),"center_xyz":[x,y,z],"dimensions_xyz":[dx,dy,height],"yaw_rad":0.0}
        self.annotations["frames"].setdefault(str(self.frame),[]).append(box); self.draw()
    def refresh_list(self):
        self.objects.delete(0,"end")
        for a in self.annotations["frames"].get(str(self.frame),[]): self.objects.insert("end",f'{a["class_name"]} | {a["track_id"]} | static={a["static_for_calibration"]} | {a["dimensions_xyz"]}')
    def delete(self):
        sel=self.objects.curselection();
        if sel: self.annotations["frames"].get(str(self.frame),[]).pop(sel[0]); self.draw()
    def save(self):
        p=self.annotation_path(); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(self.annotations,ensure_ascii=False,indent=2),encoding="utf8")
    def close(self): self.save(); self.win.destroy()
    def run(self): self.win.mainloop()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); App(a.data_root,a.output).run()
if __name__=="__main__": main()
