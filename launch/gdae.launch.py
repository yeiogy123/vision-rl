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
            executable='gdae.py',
            output='screen',
            parameters=[{
                "node_vicinity": 2.5,
                "deleted_node_vicinity": 0.75,
                "min_in": 0.62,
                "side_min_in": 0.62,
                "delete_nodes_range": 0.3,
                "acceleration_low": 0.009,
                "acceleration_high": 0.012,
                "deceleration_low": 0.016,
                "deceleration_high": 0.022,
                "angular_acceleration": 0.016,
                "nr_of_nodes": 400,
                "nr_of_closed_nodes": 500,
                "nr_of_deleted_nodes": 500,
                "update_rate": 50,
                "remove_rate": 700,
                "stddev_threshold": 0.16,
                "freeze_rate": 400,
            }]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_file],
            output='screen'
        ),
    ])
