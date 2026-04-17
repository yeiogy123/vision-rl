#!/usr/bin/env python3

from rclpy.clock import Clock
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import rclpy
from rclpy.node import Node
import threading

import math
import random
from nav_msgs.msg import Path

import point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException
from geometry_msgs.msg import PoseStamped, PoseArray
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
import transforms3d
from tf2_geometry_msgs import do_transform_pose
from rclpy.node import Node
from rclpy.logging import get_logger
from rclpy.time import Time
import tf_transformations
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math
import std_msgs.msg
from scipy import interpolate
from tf2_ros import Buffer, TransformListener, TransformException


GOAL_REACHED_DIST = 0.5
COLLISION_DIST = 0.35
TIME_DELTA = 0.2

last_odom = None
environment_dim = 20
velodyne_data = np.ones(environment_dim) * 10


def gotozero():
    navigator = BasicNavigator()
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'odom'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = last_odom.pose.pose.position.x
    initial_pose.pose.position.y = last_odom.pose.pose.position.y
    initial_pose.pose.orientation.z = last_odom.pose.pose.orientation.z
    initial_pose.pose.orientation.w = last_odom.pose.pose.orientation.w
    initial_pose.pose.orientation.x = last_odom.pose.pose.orientation.x
    initial_pose.pose.orientation.y = last_odom.pose.pose.orientation.y
    navigator.setInitialPose(initial_pose)

    navigator.waitUntilNav2Active()
    goal_poses = []
    goal_pose1 = PoseStamped()
    goal_pose1.header.frame_id = 'odom'
    goal_pose1.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose1.pose.position.x = 0.0
    goal_pose1.pose.position.y = 0.0
    goal_pose1.pose.orientation.w = 0.0
    goal_pose1.pose.orientation.z = 0.0
    i = 0
    navigator.goToPose(goal_pose1)
    while not navigator.isTaskComplete():
        i = i + 1

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# td3 network
class td3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        print(("%s/%s_actor.pth" % (directory, filename)))
        print(os.getcwd())
        # Function to load network parameters
        try:
            self.actor.load_state_dict(
                torch.load("%s/%s_actor.pth" % (directory, filename), map_location="cpu")
            )
        except Exception as e:
            print("[ERROR] Failed to load actor model:")
            print("File path:", "%s/%s_actor.pth" % (directory, filename))
            print("Exception type:", type(e).__name__)
            print("Exception message:", str(e))

class Env(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')

        self.environment_dim = 20
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 3.0
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0

        self.max_time = 100  # 最大允許時間（秒）
        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel_td3", 1)

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)
        # 訂閱 global plan
        self.global_plan_sub = self.create_subscription(
            Path,
            '/planning/global_plan',  # Nav2 global planner topic
            self.global_plan_callback,
            10
        )


        self.current_goal = None  # 第一個 intermediate point

        # TF 相關
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def global_plan_callback(self, msg: Path):
        if len(msg.poses) > 0:
            self.current_goal = msg.poses[0].pose
            self.get_logger().info(
                f"[TD3ActorNode] New local goal: ({self.current_goal.position.x:.2f}, {self.current_goal.position.y:.2f})"
            )
    


    # Perform an action and read a new state
    def step(self, action):
        global velodyne_data
        target = False
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)
        time.sleep(TIME_DELTA)
        done, collision, min_laser = self.observe_collision(velodyne_data)
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]
        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target = True
            done = True
        
        elapsed_time = Clock().now().to_msg().sec - self.start_time        
        
        if elapsed_time > self.max_time:
            done = True  # 結束 episode
            robot_state = [distance, theta, action[0], action[1]]
            state = np.append(laser_state, robot_state)
            reward = -10  # 給予一個懲罰（可根據需求調整）
            env.get_logger().info("Time Limited!! GOAL is not reached! Negative reward")
            return state, reward, done, False

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target
    
    def reset(self):
        
        angle = 0
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        x = 0.0
        y = 0.0

        self.odom_x = 0.0
        self.odom_y = 0.0

        self.publish_markers([0.0, 0.0])

        time.sleep(TIME_DELTA)

        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)

        self.start_time = Clock().now().to_msg().sec  # 重置開始時間

        return state
    
    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            env.get_logger().info("Collision is detected!")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            env.get_logger().info("reward 100")
            return 100.0
        elif collision:
            env.get_logger().info("reward -100")
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            50)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class Velodyne_subscriber(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            50)
        self.subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.55:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

if __name__ == '__main__':

    rclpy.init(args=None)

    # Set the parameters for the implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
    seed = 0  # Random seed number
    max_ep = 500  # maximum number of steps per episode
    file_name = "td3_velodyne"  # name of the file to load the policy from
    environment_dim = 20
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2

    # Create the network
    network = td3(state_dim, action_dim)
    try:
        network.load(file_name, "./pytorch_models")
    except:
        raise ValueError("Could not load the stored model parameters")

    done = False
    episode_timesteps = 0

    # Create the testing environment
    env = Env()
    odom_subscriber = Odom_subscriber()
    velodyne_subscriber = Velodyne_subscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)
    state = env.reset()
    # Begin the testing loop
    while rclpy.ok():

        # On termination of episode
        if done:
            # gotozero()
            # state = env.reset()
            env.goal_x = env.goal_x + 3
            action = network.get_action(np.array(state))
            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            a_in = [(action[0] + 1)/2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            done = 1 if episode_timesteps + 1 == max_ep else int(done)

            done = False
            episode_timesteps = 0
        else:
            action = network.get_action(np.array(state))
            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            a_in = [(action[0] + 1)/2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            done = 1 if episode_timesteps + 1 == max_ep else int(done)

            state = next_state
            episode_timesteps += 1

    rclpy.shutdown()
