#!/usr/bin/env python3
"""竞赛4阶段随机场景生成器"""
import random, math, os, sys

# WAM-V基准点
WAMV_BASE = (-710, 350)

# dock_2022 模型在世界中的位姿 (x y z roll pitch yaw)
# 用于 SDF <include> 及 obstacle_tracker 发布 /world/obstacles/dock
DOCK_POSE = (-533, 162, 0.3, 0.0, 0.0, -1.658)

# 浮标模板映射
BUOY_URIS = {
    'red':    'https://fuel.gazebosim.org/1.0/openrobotics/models/mb_marker_buoy_red',
    'green':  'https://fuel.gazebosim.org/1.0/openrobotics/models/mb_marker_buoy_green',
    'white':  'https://fuel.gazebosim.org/1.0/openrobotics/models/mb_marker_buoy_white',
    'orange': 'https://fuel.gazebosim.org/1.0/openrobotics/models/mb_round_buoy_orange',
    'black':  'https://fuel.gazebosim.org/1.0/openrobotics/models/mb_round_buoy_black',
}

def gen_wamv():
    a = random.uniform(0, 2*math.pi)
    d = random.uniform(0, 20)
    x = WAMV_BASE[0] + d * math.cos(a)
    y = WAMV_BASE[1] + d * math.sin(a)
    yaw = random.uniform(0, 2*math.pi)
    return x, y, yaw

def point_in_polygon(px, py, poly):
    """射线法判断点是否在多边形内"""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def random_point_in_polygon(poly, margin=5):
    """在多边形内随机生成点 (带边距)"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    for _ in range(1000):
        x = random.uniform(min(xs)+margin, max(xs)-margin)
        y = random.uniform(min(ys)+margin, max(ys)-margin)
        if point_in_polygon(x, y, poly):
            return x, y
    return None, None

def check_spacing(px, py, existing, min_dist):
    for ex, ey in existing:
        if math.hypot(px-ex, py-ey) < min_dist:
            return False
    return True

def gen_pillars(poly, count, min_spacing, max_spacing):
    """在多边形内随机生成柱子"""
    pillars = []
    attempts = 0
    while len(pillars) < count and attempts < 5000:
        x, y = random_point_in_polygon(poly, margin=3)
        if x is None:
            break
        spacing = random.uniform(min_spacing, max_spacing)
        if check_spacing(x, y, pillars, spacing):
            pillars.append((x, y))
        attempts += 1
    return pillars

def gen_buoys_around(center, count, radius_min, radius_max, types):
    buoys = []
    for i in range(count):
        a = random.uniform(0, 2*math.pi)
        r = random.uniform(radius_min, radius_max)
        x = center[0] + r * math.cos(a)
        y = center[1] + r * math.sin(a)
        bt = types[i % len(types)]
        buoys.append((x, y, bt))
    return buoys

def gen_boats_in_poly(poly, count):
    boats = []
    for i in range(count):
        x, y = random_point_in_polygon(poly, margin=10)
        if x is None:
            continue
        yaw = random.uniform(0, 2*math.pi)
        boats.append((i+1, x, y, yaw))
    return boats

def buoy_xml(uid, x, y, bt):
    uri = BUOY_URIS.get(bt, BUOY_URIS['white'])
    is_round = bt in ('orange', 'black')
    if is_round:
        geom = '<sphere><radius>0.21</radius></sphere>'
        drag = '75'
        pose = '0 0 0 0 0 0'
    else:
        geom = '<cylinder><radius>0.325</radius><length>0.1</length></cylinder>'
        drag = '25'
        pose = '0 0 -0.3 0 0 0'
    return f'''    <include><name>buoy_{uid}</name><pose>{x:.0f} {y:.0f} 0 0 0 0</pose><uri>{uri}</uri>
      <plugin name="vrx::PolyhedraBuoyancyDrag" filename="libPolyhedraBuoyancyDrag.so">
        <fluid_density>1000</fluid_density><fluid_level>0.0</fluid_level>
        <linear_drag>{drag}</linear_drag><angular_drag>2</angular_drag>
        <buoyancy><link_name>link</link_name><pose>{pose}</pose>
          <geometry>{geom}</geometry></buoyancy>
        <wavefield><topic>/vrx/wavefield/parameters</topic></wavefield></plugin></include>'''

def pillar_xml(pid, x, y, h):
    return f'''    <model name="pillar_{pid}"><static>true</static><pose>{x:.0f} {y:.0f} 0 0 0 0</pose>
      <link name="body"><visual name="v"><geometry><cylinder><radius>0.8</radius><length>{h}</length></cylinder></geometry>
        <material><ambient>0.6 0.3 0.1 1</ambient><diffuse>0.6 0.3 0.1 1</diffuse></material></visual>
        <collision name="c"><geometry><cylinder><radius>0.8</radius><length>{h}</length></cylinder></geometry></collision>
      </link></model>'''

def boat_xml(bid, x, y, yaw):
    model = 'roboboat01' if bid % 2 == 1 else 'roboboat02'
    return f'    <include><name>boat{bid}</name><pose>{x:.0f} {y:.0f} 0.2 0 0 {yaw:.3f}</pose><uri>{model}</uri></include>'

def gen_world(wx, wy, wyaw, output_path):
    # Phase 1: 标定浮球 — WAM-V两侧各2排, 每排4个 (共16个, 全用圆浮球)
    perp_x = -math.sin(wyaw)
    perp_y = math.cos(wyaw)
    ahead_x = math.cos(wyaw)
    ahead_y = math.sin(wyaw)
    calib_buoys = []
    # === 运动控制测试: 屏蔽标定浮球 ( uncomment 恢复 ) ===
    # ball_types = ['orange','black','orange','black','orange','black','orange','black',
    #               'orange','black','orange','black','orange','black','orange','black']
    # for row_dist in [5, 10]:
    #     for i in range(4):
    #         along = -12 + i * 8
    #         bx = wx + along * ahead_x + row_dist * perp_x
    #         by = wy + along * ahead_y + row_dist * perp_y
    #         calib_buoys.append((bx, by, ball_types[len(calib_buoys)]))
    # for row_dist in [-5, -10]:
    #     for i in range(4):
    #         along = -12 + i * 8
    #         bx = wx + along * ahead_x + row_dist * perp_x
    #         by = wy + along * ahead_y + row_dist * perp_y
    #         calib_buoys.append((bx, by, ball_types[len(calib_buoys)]))

    # Phase 1: 游荡船 (运动控制测试: 停掉, 设为0艘)
    p1_boats = []

    # Phase 2: 静态障碍区 (35根柱子, 间隔6-15m)
    p2_poly = [(-580,420), (-640,280), (-560,260), (-500,380)]
    p2_pillars = gen_pillars(p2_poly, 35, 6, 15)

    # Phase 3: 混合障碍区 (20根柱子35m + 15浮标, 运动控制测试: 无船)
    p3_poly = [(-500,380), (-560,260), (-560,200), (-340,260)]
    p3_pillars = gen_pillars(p3_poly, 20, 35, 50)
    p3_buoys = []
    for i in range(15):
        x, y = random_point_in_polygon(p3_poly, 8)
        if x:
            p3_buoys.append((i, x, y, ['red','green','white','orange','black','red','green','white','orange','black','red','green','white','orange','black'][i]))
    p3_boats = []  # 运动控制测试: 停掉

    # Phase 4: 靠泊区 (12浮标, 运动控制测试: 无船)
    dock = (-523, 159)  # Y-8m
    north_dir = math.pi/2
    p4_buoys = []
    for i in range(24):
        a = north_dir + random.uniform(-math.pi/3, math.pi/3)  # 北±60°
        r = random.uniform(30, 85)
        bx = dock[0] + r * math.cos(a)
        by = dock[1] + r * math.sin(a)
        p4_buoys.append((i, bx, by, ['green','red','green','red','green','red','green','red','green','red','green','red',
                                      'green','red','green','red','green','red','green','red','green','red','green','red'][i]))
    p4_boats = []  # 运动控制测试: 停掉

    # === 生成SDF ===
    buoy_entries = ''
    uid_counter = 0
    for x, y, bt in calib_buoys:
        buoy_entries += buoy_xml(uid_counter, x, y, bt) + '\n'
        uid_counter += 1
    for uid, x, y, bt in p3_buoys:
        buoy_entries += buoy_xml(uid_counter, x, y, bt) + '\n'
        uid_counter += 1
    for uid, x, y, bt in p4_buoys:
        buoy_entries += buoy_xml(uid_counter, x, y, bt) + '\n'
        uid_counter += 1

    pillar_entries = ''
    for i, (x, y) in enumerate(p2_pillars):
        h = random.uniform(5, 10)
        pillar_entries += pillar_xml(f'p2_{i}', x, y, h) + '\n'
    for i, (x, y) in enumerate(p3_pillars):
        h = random.uniform(8, 15)
        pillar_entries += pillar_xml(f'p3_{i}', x, y, h) + '\n'

    boat_entries = ''
    # 船预置在场景中, 由 light_wander.py 控制游荡
    all_boats = p1_boats + p3_boats + p4_boats
    for bid, x, y, yaw in all_boats:
        boat_entries += boat_xml(bid, x, y, yaw) + '\n'

    sdf = f'''<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="berthing_test">
    <physics name="8ms" type="dart"><max_step_size>0.008</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui><title>USV Competition</title><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui>
        <engine>ogre2</engine><scene>scene</scene><ambient_light>0.4 0.4 0.4</ambient_light><background_color>0.5 0.7 1.0</background_color>
        <camera_pose>-620 280 80 0 0.7 -1.6</camera_pose><camera_clip><near>0.25</near><far>10000</far></camera_clip>
      </plugin>
      <plugin filename="EntityContextMenuPlugin" name="Entity context menu"/>
      <plugin filename="GzSceneManager" name="Scene Manager"/>
      <plugin filename="InteractiveViewControl" name="Interactive view control"/>
      <plugin filename="CameraTracking" name="Camera Tracking"/>
      <plugin filename="MarkerManager" name="Marker manager"/>
      <plugin filename="Spawn" name="Spawn Entities"/>
      <plugin filename="WorldControl" name="World control"/>
      <plugin filename="WorldStats" name="World stats"/>
      <plugin filename="TransformControl" name="Transform control"/>
    </gui>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <scene><grid>false</grid><ambient>0.8 0.9 1.0</ambient><background>0.5 0.7 1.0</background></scene>
    <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation><latitude_deg>-33.724223</latitude_deg><longitude_deg>150.679736</longitude_deg><elevation>0.0</elevation><heading_deg>0.0</heading_deg></spherical_coordinates>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose><diffuse>0.8 0.8 0.8 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.1 -0.9</direction></light>
    <include><pose>0 0 0.2 0 0 0</pose><uri>https://fuel.gazebosim.org/1.0/openrobotics/models/sydney_regatta</uri></include>
    <include><name>Coast Waves</name><pose>0 0 0 0 0 0</pose><uri>coast_waves</uri></include>
    <include><pose>{DOCK_POSE[0]} {DOCK_POSE[1]} {DOCK_POSE[2]} {DOCK_POSE[3]} {DOCK_POSE[4]} {DOCK_POSE[5]}</pose><uri>dock_2022</uri></include>
    <include><name>berth_sign</name><pose>-523 192 1.8 0 0 -1.5708</pose><uri>placard_2022</uri></include>
{pillar_entries}
{buoy_entries}
{boat_entries}
    <plugin filename="libUSVWind.so" name="vrx::USVWind"><wind_obj><name>wamv</name><link_name>wamv/base_link</link_name><coeff_vector>.5 .5 .33</coeff_vector></wind_obj><wind_direction>240</wind_direction><wind_mean_velocity>0.0</wind_mean_velocity><var_wind_gain_constants>0</var_wind_gain_constants><var_wind_time_constants>2</var_wind_time_constants><update_rate>10</update_rate></plugin>
    <plugin filename="libPublisherPlugin.so" name="vrx::PublisherPlugin"><message type="gz.msgs.Param" topic="/vrx/wavefield/parameters" every="2.0">params {{ key: "direction" value {{ type: DOUBLE double_value: 0.0 }} }}params {{ key: "gain" value {{ type: DOUBLE double_value: 0.15 }} }}params {{ key: "period" value {{ type: DOUBLE double_value: 5 }} }}params {{ key: "steepness" value {{ type: DOUBLE double_value: 0 }} }}</message></plugin>
    <model name="water_collision_plane"><static>true</static><link name="surface"><visual name="vw"><geometry><box><size>600 600 0.1</size></box></geometry></visual><collision name="w"><geometry><plane><normal>0 0 -1</normal><size>600 600</size></plane></geometry><surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision></link></model>
  </world>
</sdf>'''
    with open(output_path, 'w') as f:
        f.write(sdf)
    return len(calib_buoys)+len(p3_buoys)+len(p4_buoys), len(p2_pillars)+len(p3_pillars), len(all_boats)

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(0,99999)
    random.seed(seed)
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/berthing_random.sdf'
    wx, wy, wyaw = gen_wamv()
    n_buoys, n_pillars, n_boats = gen_world(wx, wy, wyaw, out)
    print(f"WAMV={wx:.1f},{wy:.1f},0,0,0,{wyaw:.3f}")
    print(f"WORLD={out}")
    print(f"SEED={seed}")
    print(f"BUOYS={n_buoys}")
    print(f"PILLARS={n_pillars}")
    print(f"BOATS={n_boats}")
    print(f"DOCK={DOCK_POSE[0]},{DOCK_POSE[1]},{DOCK_POSE[2]},{DOCK_POSE[3]},{DOCK_POSE[4]},{DOCK_POSE[5]}")
