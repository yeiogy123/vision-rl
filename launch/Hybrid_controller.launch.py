#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('td3'), 'launch')
    rviz_file = os.path.join(get_package_share_directory('td3'), 'launch', 'pioneer3dx.rviz')

    return LaunchDescription([
        Node(
            package='td3',
            executable='Hybrid_controller.py',
            name='Hybrid_controller_node',
            output='screen',
        ),
    ])
