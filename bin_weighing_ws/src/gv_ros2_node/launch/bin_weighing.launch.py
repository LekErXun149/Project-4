#!/usr/bin/env python3
"""
Integrated Launch File for Bin Weighing System
Starts:
  1. HX711 Scale Node
  2. GV Depth Camera Node
  3. Pi Camera Node (camera_ros)
  4. Bin Weighing System Node (object detection + barcode + weight)
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, LogInfo
from launch_ros.actions import Node


def generate_launch_description():

    # ── 1. HX711 Scale Node ──────────────────────────────────────────────────
    scale_node = ExecuteProcess(
        cmd=[
            'sudo', 'bash', '-c',
            'source /opt/ros/jazzy/setup.bash && '
            'source /home/group5/bin_weighing_ws/install/setup.bash && '
            'ros2 run hx711_scale scale_node'
        ],
        output='screen',
        name='scale_node'
    )

    # ── 2. GV Depth Camera ───────────────────────────────────────────────────
    depth_camera_node = Node(
        package='gv_ros2',
        executable='gv_ros2_node',
        name='gv_ros2_node',
        output='screen',
        parameters=[{
            'config_path': '/home/group5/bin_weighing_ws/install/gv_ros2/share/gv_ros2/config'
        }]
    )

    # ── 3. Pi Camera Node ────────────────────────────────────────────────────
    pi_camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'camera':  '/base/soc/i2c0mux/i2c@1/imx708@1a',
            'format':  'RGB888',
            'width':   640,
            'height':  480,
            'AfMode':  2,
        }],
        additional_env={
            'LIBCAMERA_IPA_PATH':  '/usr/local/lib/libcamera/ipa',
            'LIBCAMERA_DATA_PATH': '/usr/local/share/libcamera',
	    'HOME': '/root',
            'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        }
    )

    # ── 4. Bin Weighing System (delayed to allow cameras to init) ────────────
    bin_weighing_node = TimerAction(
        period=5.0,  # wait 5 seconds for cameras to initialize
        actions=[
            LogInfo(msg='Starting Bin Weighing System...'),
            ExecuteProcess(
                cmd=[
                    'python3',
                    '/home/group5/bin_weighing_ws/src/bin_weighing_system.py'
                ],
                output='screen',
                name='bin_weighing_system',
                additional_env={
                    'FASTDDS_DISABLE_SHM': '1',
                }
            )
        ]
    )

    return LaunchDescription([
        LogInfo(msg='=== Starting Bin Weighing System ==='),
        LogInfo(msg='Starting Scale Node...'),
        scale_node,
        LogInfo(msg='Starting Depth Camera...'),
        depth_camera_node,
        LogInfo(msg='Starting Pi Camera...'),
        pi_camera_node,
        bin_weighing_node,
    ])
