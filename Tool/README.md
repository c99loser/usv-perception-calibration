# 点云连续帧盲标工作台 v2

这是原 `local_pointcloud_annotation_app.py` 的兼容升级版，面向项目中的 Supervisely Point Cloud Episode 目录。工具始终在单个 LiDAR 的传感器局部 XYZ 坐标中保存 3D 框；缩放、平移和右侧鱼眼图不会改变标注坐标。

## 本版新增

- **连续帧跟随**：`跟随到下一帧` 会复制当前帧所有缺失的 `track_id`，并根据框附近点簇的鲁棒中心位移给出下一帧建议位置；已有同名轨迹不会被覆盖。
- **传播审计**：自动跟随框使用虚线和 `⚠` 标记，保存位移、点数支持和质量分数；人工检查后勾选“已复核传播框”才变为已复核状态。
- **可编辑框**：在“选择/调整”模式内，拖动框内部可移动，拖动四角白色手柄可改变 XY 尺寸；属性栏可修改类别、轨迹、静态性、置信度、Z 和高度。
- **缩放与平移**：鼠标滚轮以指针位置为中心缩放，中键拖动或 `H` 模式平移；`+`、`−` 和 `1` 可控制视图。缩放后照常新建和编辑框。
- **CVAT 风格操作**：选择/绘制/平移模式、帧时间轴、播放、撤销/重做、复制/粘贴、快捷键、1.5 秒防抖自动保存、未保存状态和传播复核标识。
- **安全兼容**：读取原 v1 sidecar JSON，保存时升级为 v2；原始 PCD 和同步图片永不改写。默认忽略名称含 `holdout` 的 Episode。

## 推荐启动（V3 train）

在 PowerShell 中执行：

```powershell
Set-Location 'F:\Challenger program\Tool'
.\start_v3_train.ps1
```

启动器固定读取：

```text
F:\Challenger program\output\real_lidar_raw_official_stable_annotation_v3_bundle\
  real_lidar_raw_official_stable_annotation_v3\supervisely_train_raw_project
```

标注结果独立保存到：

```text
F:\Challenger program\output\real_lidar_raw_official_stable_annotation_v3_bundle\
  local_annotation_workspace_v3\annotations
```

不会打开 `091054` holdout。

## 首次安装依赖

若电脑上的 Python 尚未安装 `numpy` 和 `Pillow`：

```powershell
Set-Location 'F:\Challenger program\Tool'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start_v3_train.ps1
```

Tkinter 通常随 Windows 官方 Python 安装。如果提示缺少 `tkinter`，请在 Python 安装器中启用 `tcl/tk and IDLE`。

## 自定义数据目录

```powershell
python .\run_annotator.py `
  --data-root 'D:\my_project\supervisely_train_raw_project' `
  --output 'D:\my_project\local_annotations'
```

只有在类别规则、模型和阈值已经冻结、确实进入留出评价时，才可显式增加 `--allow-holdout`。常规标注不要使用该参数。

## 推荐操作流程

1. 在第一帧按 `N` 进入新建模式，填写属性后拖拽画框。
2. 同一真实物体在连续帧中保持相同 `track_id`。
3. 点击“跟随到下一帧”，检查虚线建议框。
4. 按 `V` 进入选择模式，拖动整框或角点修正；确认无误后勾选“已复核传播框”。
5. 用左右方向键切帧，`Ctrl+S` 保存；关闭窗口也会保存。
6. 对水面、自船、多径和无法确认的稀疏点使用 `background` 或 `ambiguous`，不要强行赋予目标类别。

## 关键快捷键

| 操作 | 快捷键 |
|---|---|
| 选择/调整模式 | `V` |
| 新建框模式 | `N` |
| 平移模式 | `H` |
| 前/后一帧 | `←` / `→` |
| 播放/暂停 | `Space` |
| 适配 ±60 m 全图 | `1` |
| 保存 | `Ctrl+S` |
| 撤销/重做 | `Ctrl+Z` / `Ctrl+Y` |
| 复制/粘贴框 | `Ctrl+C` / `Ctrl+V` |
| 删除选中框 | `Delete` |

## 跟随算法的正确定位

跟随只是一种节省重复劳动的建议：它比较前后帧中框附近的有限 XY 点，计算鲁棒点簇中心的位移，并把位移限制在 3 m 以内。水面多径、稀疏小目标、遮挡、快速运动或相邻 V3 抽样帧间隔较大时都可能失效，因此：

- 自动框必须人工复核；
- 跟随质量分数不是标注置信度；
- 工具不会把传播建议当作已经确认的真值；
- 右侧图像不能用于按像素位置移动 3D 框；
- 不能用未经验证的多 LiDAR TF 融合画面生成正式真值。

## 文件说明

```text
Tool/
├─ pointcloud_annotator/
│  ├─ core.py          # PCD、坐标变换、框几何、简单跟随
│  └─ app.py           # Tkinter 交互界面
├─ tests/test_core.py   # 无界面自动测试
├─ run_annotator.py     # 通用入口
├─ start_v3_train.ps1   # 当前 V3 train 一键启动器
└─ requirements.txt
```
