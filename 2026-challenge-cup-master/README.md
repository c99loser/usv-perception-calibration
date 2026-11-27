## V1.0.5 
## 2026/07/13
### 修改内容

### `/world/obstacles/dock` 码头角度与模型不一致

**现象**：`/world/obstacles/dock` 话题发布的码头 yaw 为 `0°`，而世界 SDF 中 `dock_2022` 模型实际朝向为 `-1.658 rad`（约 −95°），相差约 95°。

**根因**：`obstacle_tracker.py` 发布的码头位姿来自参数 `dock_x/dock_y/dock_yaw`，但 `competition.launch.py` 启动该节点时没有传入这三个参数，`dock_yaw` 用了默认值 `0.0`。而码头模型 pose 硬编码在各场景生成器里，两处独立维护、缺乏单一数据源。

**修改**：采用**单一数据源**方案 —— 在场景生成器中提取 `DOCK_POSE` 常量，SDF 模型和 ROS 话题都从它派生：

- `random_world_generator_berthing.py` / `random_world_generator.py` / `random_world_generator_perception.py`：新增 `DOCK_POSE` 常量，SDF 引用它，标准输出 `DOCK=` 行
- `competition.launch.py`：解析 `DOCK=` 行，传 `dock_x/dock_y/dock_yaw` 给 `obstacle_tracker.py`

```diff
+ try: _dock = dock_pose
+ except NameError: _dock = [-533.0, 162.0, 0.3, 0.0, 0.0, -1.658]
  launch_processes.append(ExecuteProcess(
      cmd=['python3', '-u', os.path.join(vrx_gz_lib, 'obstacle_tracker.py'),
           '--ros-args', '-p', f'world:={world_name}',
-          '-p', f'world_sdf:={world_sdf}'],
+          '-p', f'world_sdf:={world_sdf}',
+          '-p', f'dock_x:={_dock[0]}',
+          '-p', f'dock_y:={_dock[1]}',
+          '-p', f'dock_yaw:={_dock[5]}'],
      name='obstacle_tracker', output='screen,
  ))
```

## 验证

```bash
colcon build --merge-install --packages-select vrx_gz
rm -f install/share/vrx_gz/worlds/berthing_user_random_*.sdf
ros2 launch vrx_gz competition.launch.py world:=berthing_user
```

```bash
# GPS / IMU 应有数据
ros2 topic hz /wamv/sensors/gps/gps/fix    # ~20 Hz
ros2 topic hz /wamv/sensors/imu/imu/data   # ~100 Hz

# dock yaw 应为 -1.658 rad (≈ -95°)
ros2 topic echo /world/obstacles/dock --once
```



## V1.0.4 
## 2026/07/10
### 修改内容
### 文件：`src/vrx/vrx_gz/random_world_generator_berthing.py`（第 237 行附近）

```diff
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
+ <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
+ <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
```

仅新增两行，补上缺失的 IMU 和 NavSat 系统插件。


