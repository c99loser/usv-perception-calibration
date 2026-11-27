# Copyright 2021 Open Source Robotics Foundation, Inc.
# Modified: 4-phase competition scene generator

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import os, random, subprocess, math

import vrx_gz.launch
from vrx_gz.model import Model


def random_wamv():
    """在[-710,350]周围20m圆内随机生成WAM-V位置和航向"""
    a = random.uniform(0, 2*math.pi)
    d = random.uniform(0, 20)
    x = -710 + d * math.cos(a)
    y = 350 + d * math.sin(a)
    yaw = random.uniform(0, 2*math.pi)
    return [x, y, 0, 0, 0, yaw]


def launch(context, *args, **kwargs):
    config_file = LaunchConfiguration('config_file').perform(context)
    world_name = LaunchConfiguration('world').perform(context)
    sim_mode = LaunchConfiguration('sim_mode').perform(context)
    bridge_competition_topics = LaunchConfiguration('bridge_competition_topics').perform(context).lower() == 'true'
    robot = LaunchConfiguration('robot').perform(context)
    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'
    robot_urdf = LaunchConfiguration('urdf').perform(context)
    gz_paused = LaunchConfiguration('paused').perform(context).lower() == 'true'
    competition_mode = LaunchConfiguration('competition_mode').perform(context).lower() == 'true'
    extra_gz_args = LaunchConfiguration('extra_gz_args').perform(context)

    launch_processes = []

    world_name_base = os.path.splitext(os.path.basename(world_name))[0]
    gz_world_name = world_name_base  # Gazebo中实际世界名, 后续可能被覆盖

    models = []
    if config_file and config_file != '':
        with open(config_file, 'r') as stream:
            models = Model.FromConfig(stream)
    elif world_name_base in ('berthing_test', 'berthing_user', 'perception_user', 'calibration_user', 'projection_user'):
        # 调用随机场景生成器 (根据世界名选择不同生成器)
        gen_map = {
            'berthing_test': 'random_world_generator.py',
            'berthing_user': 'random_world_generator_berthing.py',
            'perception_user': 'random_world_generator_perception.py',
            'calibration_user': 'random_world_generator_calibration.py',
            'projection_user': 'random_world_generator_projection.py',
        }
        from ament_index_python.packages import get_package_share_directory
        _pkg_share = get_package_share_directory('vrx_gz')
        _pkg_prefix = os.path.dirname(os.path.dirname(_pkg_share))
        gen_script = os.path.join(_pkg_prefix, 'lib', 'vrx_gz', gen_map[world_name_base])
        if not os.path.exists(gen_script):
            # 兜底: 源码树中的脚本
            gen_script = os.path.join(
                os.path.dirname(__file__), '..', 'random_world_generator.py')
        random_world = f'/tmp/{world_name_base}_random_{os.getpid()}.sdf'
        seed = random.randint(0, 99999)
        result = subprocess.run(['python3', gen_script, str(seed), random_world],
            capture_output=True, text=True, timeout=15)
        output = result.stdout
        print(f"  SEED={seed}")
        wamv_pose = [-710, 350, 0, 0, 0, 0]
        # dock_2022 模型在世界中的位姿 (来自生成器, 默认值兜底)
        dock_pose = [-533.0, 162.0, 0.3, 0.0, 0.0, -1.658]
        for line in output.strip().split('\n'):
            if line.startswith('WAMV='):
                wamv_pose = [float(x) for x in line[5:].split(',')]
                dist = math.hypot(wamv_pose[0]+533, wamv_pose[1]-162)
                print(f"  WAM-V: ({wamv_pose[0]:.0f},{wamv_pose[1]:.0f}) yaw={math.degrees(wamv_pose[5]):.0f}° dist={dist:.0f}m")
            elif line.startswith('DOCK='):
                dock_pose = [float(x) for x in line[5:].split(',')]
                print(f"  Dock: ({dock_pose[0]:.0f},{dock_pose[1]:.0f}) yaw={math.degrees(dock_pose[5]):.0f}°")
            elif line.startswith('BUOYS='):
                print(f"  Buoys: {line[6:]}")
            elif line.startswith('PILLARS='):
                print(f"  Pillars: {line[8:]}")
            elif line.startswith('BOATS='):
                print(f"  Boats: {line[6:]}")
        # 拷到 install share 目录
        import shutil
        install_dir = os.path.join(_pkg_share, 'worlds')
        os.makedirs(install_dir, exist_ok=True)
        install_world = os.path.join(install_dir, os.path.basename(random_world))
        shutil.move(random_world, install_world)
        random_world = install_world
        m = Model('wamv', 'wam-v', wamv_pose)
        if robot_urdf and robot_urdf != '':
            m.set_urdf(robot_urdf)
        elif world_name_base in ('perception_user', 'calibration_user', 'projection_user'):
            # perception_user 使用大船URDF (14m×5m)
            wamv_desc_share = get_package_share_directory('wamv_description')
            large_xacro = os.path.join(wamv_desc_share, 'urdf', 'wamv_large.urdf.xacro')
            if os.path.exists(large_xacro):
                m.set_urdf(large_xacro)
                # SDF后处理: 注入甲板visual (URDF→SDF会丢失fixed joint子link的visual)
                m.deck_inject = True  # 标记需要在SDF中注入甲板
                print(f"  [大船] 使用 wamv_large.urdf.xacro (14m×5m)")
            else:
                print(f"  [警告] 大船URDF未找到: {large_xacro}")
        models.append(m)
        sdf_base = os.path.splitext(os.path.basename(random_world))[0]
        launch_processes.extend(vrx_gz.launch.simulation(sdf_base, headless, gz_paused, extra_gz_args))
    else:
        m = Model('wamv', 'wam-v', [-710, 350, 0, 0, 0, 0])
        models.append(m)
        launch_processes.extend(vrx_gz.launch.simulation(world_name_base, headless, gz_paused, extra_gz_args))

    launch_processes.extend(vrx_gz.launch.spawn(sim_mode, world_name_base, models, robot))

    if (sim_mode == 'bridge' or sim_mode == 'full') and bridge_competition_topics:
        launch_processes.extend(vrx_gz.launch.competition_bridges(world_name_base, competition_mode))

    # 相机图像由 VRX payload_bridges + competition_user/sensor_bridge 转发到 /usv/camera/*
    # 这里不再额外启动 camera_bridge，避免话题重复和 base_link 话题名错误

    # 轻量游荡船控制 (禁用: gz topic 子进程导致卡顿)
    # base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # launch_processes.append(ExecuteProcess(
    #     cmd=['python3', '-u', os.path.join(base, 'light_wander.py')],
    #     name='light_wander', output='screen',
    # ))

    # 障碍物位姿跟踪 (障碍船游走由 BoatWander C++插件内置, 零话题依赖)
    from ament_index_python.packages import get_package_prefix
    vrx_gz_lib = os.path.join(get_package_prefix('vrx_gz'), 'lib', 'vrx_gz')
    world_name = world_name_base
    try: world_sdf = random_world
    except NameError: world_sdf = ''
    # dock_2022 位姿: 由随机世界生成器输出, 否则用默认值 (非生成世界无 dock 时无影响)
    try: _dock = dock_pose
    except NameError: _dock = [-533.0, 162.0, 0.3, 0.0, 0.0, -1.658]
    launch_processes.append(ExecuteProcess(
        cmd=['python3', '-u', os.path.join(vrx_gz_lib, 'obstacle_tracker.py'),
             '--ros-args', '-p', f'world:={world_name}',
             '-p', f'world_sdf:={world_sdf}',
             '-p', f'dock_x:={_dock[0]}',
             '-p', f'dock_y:={_dock[1]}',
             '-p', f'dock_yaw:={_dock[5]}'],
        name='obstacle_tracker', output='screen',
    ))

    return launch_processes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='sydney_regatta'),
        DeclareLaunchArgument('sim_mode', default_value='full'),
        DeclareLaunchArgument('bridge_competition_topics', default_value='True'),
        DeclareLaunchArgument('config_file', default_value=''),
        DeclareLaunchArgument('robot', default_value=''),
        DeclareLaunchArgument('headless', default_value='False'),
        DeclareLaunchArgument('urdf', default_value=''),
        DeclareLaunchArgument('paused', default_value='False'),
        DeclareLaunchArgument('competition_mode', default_value='False'),
        DeclareLaunchArgument('extra_gz_args', default_value=''),
        OpaqueFunction(function=launch),
    ])
