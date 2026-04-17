#!/usr/bin/env python3
from rclpy.clock import Clock
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from replay_buffer import ReplayBuffer
from sensor_msgs.msg import Image  # 必須導入
from cv_bridge import CvBridge    # 用於轉換 ROS 影像
import cv2
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import threading
import yaml
import math
import random
from rclpy.qos import qos_profile_sensor_data

import point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from got_DRL import SAC
from tqdm import tqdm
from collections import deque

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.2

# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

last_odom = None
environment_dim = 20
velodyne_data = np.ones(environment_dim) * 10
latest_camera_image = None 
intervention = 0
class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')


        self.max_time = 100  # 最大允許時間（秒）
        self.start_time = Clock().now().to_msg().sec  # 記錄 episode 開始時間

        self.environment_dim = 20
        self.odom_x = 0.0
        self.odom_y = 0

        self.goal_x = 20.0
        self.goal_y = 0.0

        self.upper = 20.0
        self.lower = -5.0
        self.olddist = math.sqrt(math.pow(self.odom_x-self.goal_x,2)+math.pow(self.odom_y-self.goal_y,2))
        self.last_act = 0.0

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

        self.bridge = CvBridge()

    # Perform an action and read a new state
    def step(self, action):
        global velodyne_data
        global latest_camera_image
        target = False
        
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
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

       
        
        r_heuristic = (self.olddist - distance) * 20
        r_action = action[0]*2 - abs(action[1])
        r_smooth = -abs(action[1]-self.last_act)/4
        r_target = 0.0
        r_collision = 0.0

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target = True
            done = True
            self.olddist = math.sqrt(math.pow(self.odom_x-self.goal_x,2)+math.pow(self.odom_y-self.goal_y,2))
            r_target = 100.0
        
        if collision:
            r_collision = -100.0

        reward = r_heuristic + r_action + r_smooth + r_target + r_collision

        
        elapsed_time = Clock().now().to_msg().sec - self.start_time        
        
        if elapsed_time > self.max_time:
            done = True  # 結束 episode
            robot_state = [distance, theta, action[0], action[1]]

            camera = latest_camera_image
            image = self.bridge.imgmsg_to_cv2(camera, "mono8")
            image = image[80:400, 140:500]
            image = cv2.resize(image, (160, 128))
            image = np.expand_dims(image, axis=2)

            state = image / 255.0
            reward = -10  # 給予一個懲罰（可根據需求調整）
            env.get_logger().info("Time Limited!! GOAL is not reached! Negative reward")
            return state, reward, done, False

        robot_state = [distance, theta, action[0], action[1]]
        
        camera = latest_camera_image
        image = self.bridge.imgmsg_to_cv2(camera, "mono8")
        image = image[80:400, 140:500]
        image = cv2.resize(image, (160, 128))
        image = np.expand_dims(image, axis=2)

        state = image / 255.0
        # reward = self.get_reward(target, collision, action, min_laser)
        Dist = min(distance/15, 1.0)
        theta = theta / np.pi
        toGoal = np.array([Dist, theta, action[0], action[1]])
        self.last_act = action[1]
        return state, r_heuristic, r_action, r_collision, r_target, reward, done, toGoal, target

    def reset(self):
        global latest_camera_image
        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service("/gazebo/reset_world")
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0.0
        y = 0.0
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment


        self.change_goal()
        
        
        # randomly scatter boxes in the environment
        # self.random_box()
        self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")
        while latest_camera_image is None:
            env.get_logger().info("wait for image.")
            time.sleep(0.1)
        camera = latest_camera_image
        image = self.bridge.imgmsg_to_cv2(camera, "mono8")
        image = image[80:400, 140:500]
        image = cv2.resize(image, (160, 128))
        image = np.expand_dims(image, axis=2)

        state = image / 255.0
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        self.olddist = math.sqrt(math.pow(self.odom_x-self.goal_x,2)+math.pow(self.odom_y-self.goal_y,2))
        self.last_act = 0.0
    
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

        self.start_time = Clock().now().to_msg().sec  # 重置開始時間

        Dist = min(distance/15, 1.0)
        theta = theta / np.pi
        toGoal = np.array([Dist, theta, 0.0, 0.0])
        return state, toGoal

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

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


class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data


class Image_subscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera1/image_raw',  
            self.image_callback,
             qos_profile_sensor_data)

    def image_callback(self, msg):
        global latest_camera_image
        try:
            latest_camera_image = msg
        except Exception as e:
            self.get_logger().error(f"影像轉換出錯: {e}")

class keyboard_subscriber(Node):
    def __init__(self):
        super().__init__('keyboard_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            '/scout/telekey',
            self.keyboard_callback,
            10)
        self.subscription

    def keyboard_callback(self, msg):
        global intervention
        global key_cmd
        key_cmd = msg
        intervention = msg.twist.angular.x

class Velodyne_subscriber(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            qos_profile_sensor_data)
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
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

def check_pos(x, y):
    # Check if the random goal position is located on an obstacle and do not accept it if it is
    goal_ok = False
    
    if x > -0.55 and 1.5 > y > -1.5:
        goal_ok = True
    if 7.8 > x > 7.4 and 0.0 > y > -0.05:
        goal_ok = False
    if 11.0 > x > 10.0 and 1.5 > y > 1.0:
        goal_ok = False
    if 11 > x > 10 and -1.0 > y > -1.5:
        goal_ok = False


    return goal_ok

def evaluate(network, eval_episodes=10, epoch=0):
    env.collision = 0
    avg_reward_list = []

    for ep in range(eval_episodes):

        obs_list = deque(maxlen=frame_stack)  # 🔥 每次重建

        obs, goal = env.reset()
        done = False
        avg_reward = 0.0
        count = 0

        # 初始化 frame stack
        for _ in range(frame_stack):
            obs_list.append(obs)

        observation = np.concatenate((obs_list[-4], obs_list[-3], obs_list[-2], obs_list[-1]), axis=-1)


        while not done and count < max_steps:

            action = network.choose_action(
                np.array(observation),
                np.array(goal[:2]),   # 🔥 修正
                evaluate=True
            ).clip(-max_action, max_action)

            a_in = [
                (action[0] + 1) * linear_cmd_scale,
                action[1] * angular_cmd_scale
            ]

            obs_, _, _, _, _ , reward, done, goal, target = env.step(a_in)

            avg_reward += reward

            # 🔥 更新 state
            obs_list.append(obs_)
            observation = np.concatenate(list(obs_list), axis=-1)

            count += 1

        avg_reward_list.append(avg_reward)

        print(f"\nEpisode {ep+1}, Steps: {count}, Reward: {avg_reward}, Collision: {env.collision}")

    reward = np.mean(avg_reward_list)
    col = env.collision

    print("\n======================================")
    print(f"Eval @ Epoch {epoch} → Avg Reward: {reward}, Collisions: {col}")
    print("======================================")

    return reward


def plot_animation_figure():

    plt.figure()
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC") + ' Lr_a: ' + str(lr_a) + ' Lr_c: ' + str(lr_c) +' Target Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep_real), reward_target_list)

    plt.subplot(2, 2, 2)
    plt.title('Collision Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep_real), reward_collision_list)


    plt.subplot(2, 2, 3)
    plt.title('Pedal ' + str(ep_real))
    plt.scatter(np.arange(len(pedal_list)), pedal_list, s=6, c='coral')
    
    plt.subplot(2, 2, 4)
    plt.title('Steering')
    plt.scatter(np.arange(len(steering_list)), steering_list, s=6, c='coral')
    
    plt.tight_layout()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC") + ' Lr_a: ' + str(lr_a) + ' Lr_c: ' + str(lr_c))
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.plot(np.arange(ep_real), reward_list)
    plt.plot(np.arange(ep_real), reward_mean_list)

    plt.subplot(2, 2, 2)
    plt.title('Heuristic Reward')
    plt.xlabel('Episode')
    plt.ylabel('Heuristic Reward')
    plt.plot(np.arange(ep_real), reward_heuristic_list)

    plt.subplot(2, 2, 3)
    plt.title('Action Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep_real), reward_action_list)

    plt.tight_layout()

    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    rclpy.init(args=None)

    # Set the parameters for the implementation
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")  # cuda or cpu

    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ##### Individual parameters for each model ######
    model = 'GoT-SAC'
    mode_param = config[model]
    model_name = mode_param['name']
    policy_type = mode_param['actor_type']
    critic_type = mode_param['critic_type']
    transformer_block = mode_param['block']
    transformer_head = mode_param['head']

    ###### Default parameters for DRL ######
    max_steps = config['MAX_STEPS']
    max_episodes = config['MAX_EPISODES']
    batch_size = config['BATCH_SIZE']
    lr_a = config['LR_A']
    lr_c = config['LR_C']
    gamma = config['GAMMA']
    tau = config['TAU']
    policy_freq = config['ACTOR_FREQ']
    buffer_size = config['BUFFER_SIZE']
    frame_stack = config['FRAME_STACK']
    plot_interval = config['PLOT_INTERVAL']
    
    ##### Evaluation #####
    save_interval = config['SAVE_INTERVAL']
    save_threshold = config['SAVE_THRESHOLD']
    reward_threshold = config['REWARD_THRESHOLD']
    eval_threshold = config['EVAL_THRESHOLD'] 
    eval_ep = config['EVAL_EPOCH']
    save_models = config['SAVE']

    ##### Attention #####
    pre_train = config['PRE_TRAIN'] # whether intialize with pre-trained parameter
    attention_only = config['ATTENTION_ONLY'] # whether load the attention only from the pretrained GoT
    policy_attention_fix = config['P_ATTENTION_FIX'] # whether fix the weights and bias of policy attention
    critic_attention_fix = config['C_ATTENTION_FIX'] #whether fix the weights and bias of value attention

    ##### Human Intervention #####
    pre_buffer = config['PRE_BUFFER'] # Human expert buffer
    human_guidence = config['HUMAN_INTERVENTION'] # whether need guidance from human driver

    ##### Entropy ######
    auto_tune = config['AUTO_TUNE']
    alpha = config['ALPHA']
    lr_alpha = config['LR_ALPHA']

    ##### Environment ######
    seed = config['SEED']
    env_name = config['ENV_NAME']
    driver = config['DRIVER']
    robot = config['ROBOT']
    linear_cmd_scale = config['L_SCALE']
    angular_cmd_scale = config['A_SCALE']

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    folder_name = "./final_curves"
    if save_models and not os.path.exists(folder_name):
        os.makedirs(folder_name)
    folder_name = "./final_models"
    if save_models and not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    env = GazeboEnv()
    odom_subscriber = Odom_subscriber()
    velodyne_subscriber = Velodyne_subscriber()
    keyboard_subscriber = keyboard_subscriber()
    Image_subscriber = Image_subscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)
    executor.add_node(keyboard_subscriber)
    executor.add_node(Image_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    state, _ = env.reset()
    state_dim = state.shape
    action_dim = 2
    physical_state_dim = 2 # Polar coordinate
    max_action = 1
    
    # Initialize the agent
    ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
              critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
              buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block,
              head=transformer_head, automatic_entropy_tuning=auto_tune)

    ###### Initializing pretrained network if possible ######
    if pre_train:
        if attention_only:
            # policy_type = "DeterministicTransformer"
            name = "SAC_IL_scout_image_rrc_fisheye_GoT_normalize_Oscar_seed1_64patches_2depth_8heads_2048mlp"
            il_ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, 
                         policy_attention_fix, critic_attention_fix, human_guidence, 
                         seed, lr_c, lr_a, lr_alpha, buffer_size, tau, policy_freq,
                         gamma, alpha, block=transformer_block, head=transformer_head,
                         automatic_entropy_tuning=auto_tune)
            il_ego.load_actor(name, directory="./final_models")
    
            ###### Assign the attention only ########
            ego.policy.trans = il_ego.policy.trans
            ego.policy.fc_embed = il_ego.policy.fc_embed

        else:
            name = 'SAC_IL_scout_image_rrc_fisheye_GoT_normalize_Oscar_seed1_64patches_2depth_8heads_2048mlp'
            ego.load_actor(name, directory="./final_models")

    ###### Pre intialiaze corner replay buffer, Optional #######
    if pre_buffer:
        data_dir = '/home/oscar/ws_oscar/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts'
        files = natsorted(glob.glob(os.path.join(data_dir) + '/IL/Data/' + env_name + '/' + driver + '/*.npz'))
        obs_list = []
        act_list = []
        goal_list = []
        r_list = []
        next_obs_list = []
        next_goal_list = []
        done_list = []
        
        for idx, file in enumerate(files):
            
            obs = np.load(file)['obs']
            act = np.load(file)['act']
            goal = np.load(file)['goal']
            r = np.load(file)['reward']
            next_obs = np.load(file)['next_obs']
            next_goal = np.load(file)['next_goal']
            done = np.load(file)['done']
            
            obs_list.append(np.array(obs))
            act_list.append(np.array(act))
            goal_list.append(np.array(goal))
            r_list.append(np.array(r))
            next_obs_list.append(np.array(next_obs))
            next_goal_list.append(np.array(next_goal))
            done_list.append(np.array(done))
        
        obs_dataset = np.concatenate(obs_list, axis=0)
        act_dataset = np.concatenate(act_list, axis=0)
        goal_dataset = np.concatenate(goal_list, axis=0)
        reward_dataset = np.concatenate(r_list, axis=0)
        next_obs_dataset = np.concatenate(next_obs_list, axis=0)
        next_goal_dataset = np.concatenate(next_goal_list, axis=0)
        done_dataset = np.concatenate(done_list, axis=0)
    
        ego.initialize_expert_buffer(obs_dataset, act_dataset, goal_dataset[:,:2], 
                                     next_goal_dataset[:,:2], reward_dataset,
                                     next_obs_dataset, done_dataset)

    # Create evaluation data store
    evaluations = []
    
    ep_real = 0
    done = False
    reward_list = []
    reward_heuristic_list = []
    reward_action_list = []
    reward_freeze_list = []
    reward_target_list = []
    reward_collision_list = []
    reward_mean_list = []
    
    pedal_list = []
    steering_list = []

    # plt.ion()

    total_timestep = 0
    try:
        while rclpy.ok():
            # Begin the training loop
            
            for ep in tqdm(range(0, max_episodes), ascii=True):
                episode_reward = 0
                episode_heu_reward = 0.0
                episode_act_reward = 0.0
                episode_tar_reward = 0.0
                episode_col_reward = 0.0
                episode_fr_reward = 0.0
                s_list = deque(maxlen=frame_stack)
                s, goal = env.reset()

                for i in range(4):
                    s_list.append(s)

                state = np.concatenate((s_list[-4], s_list[-3], s_list[-2], s_list[-1]), axis=-1)

                for timestep in range(max_steps):
                    # On termination of episode
                    if timestep == 0:
                        action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                        a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                        last_goal = goal
                        s_, _, _, _, _ , reward, done, goal, target = env.step(a_in)        
                        state = np.concatenate((s_, s_, s_, s_), axis=-1)
                        
                        for i in range(4):
                            s_list.append(s_)           

                        if done:
                            print("Bad Initialization, skip this episode.")
                            break

                        continue
                    
                    if done or timestep == max_steps-1:
                        ep_real += 1
            
                        done = False

                        reward_list.append(episode_reward)
                        reward_mean_list.append(np.mean(reward_list[-20:]))
                        reward_heuristic_list.append(episode_heu_reward)
                        reward_action_list.append(episode_act_reward)
                        reward_target_list.append(episode_tar_reward)
                        reward_collision_list.append(episode_col_reward)
                        reward_freeze_list.append(episode_fr_reward)

                        # if reward_mean_list[-1] >= reward_threshold and ep_real > eval_threshold:
                        #     reward_threshold = reward_mean_list[-1]
                        #     print("Evaluating the Performance.")
                        #     avg_reward = evaluate(ego, eval_ep, ep_real)
                        #     evaluations.append(avg_reward)
                        #     if avg_reward > save_threshold:
                        #         ego.save(file_name, directory=folder_name, reward=int(np.floor(avg_reward)), seed=seed)
                        #         save_threshold = avg_reward

                        pedal_list.clear()
                        steering_list.clear()
                        total_timestep += timestep 
                        print('\n',
                            '\n',
                            'Robot: ', 'Scout',
                            'Episode:', ep_real,
                            'Step:', timestep,
                            'Tottal Steps:', total_timestep,
                            'R:', episode_reward,
                            'Overak R:', reward_mean_list[-1],
                            'Expert Batch:', np.int8(ego.batch_expert),
                            'Temperature:', ego.alpha.detach().cpu().numpy().item(),
                            'Lr_a:', lr_a,
                            'Lr_c', lr_c,
                            'seed:', seed,
                            'Env:', env_name,
                            "Filename:", model_name,
                            '\n')

                        if (ep_real % save_interval == 0):
                            np.save(os.path.join('final_curves', 'reward_seed' + str(seed) + '_' + model_name),
                                    reward_mean_list, allow_pickle=True, fix_imports=True)

                        # if ep_real % plot_interval == 0:
                        #     plot_animation_figure()
                        #     plt.ioff()
                        #     plt.show()

                        break

                    if intervention:
                        while key_cmd is None:
                            env.get_logger().info("wait for human expert to input action.")
                            time.sleep(0.1)
                        env.get_logger().info("intervention.")
                        action_exp = [key_cmd.linear.x, key_cmd.angular.z]
                        action = None
                        a_in = [(action_exp[0] + 1) * linear_cmd_scale, action_exp[1]*angular_cmd_scale]
                        pedal_list.append(round((action_exp[0] + 1)/2,2))
                        steering_list.append(round(action_exp[1],2))
                    else:
                        action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                        action_exp = None
                        a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                        pedal_list.append(round((action[0] + 1)/2,2))
                        steering_list.append(round(action[1],2))

                    last_goal = goal
                    s_, r_h, r_a, r_c, r_t, reward, done, goal, target = env.step(a_in)

                    episode_reward += reward
                    episode_heu_reward += r_h
                    episode_act_reward += r_a
                    episode_col_reward += r_c
                    episode_tar_reward += r_t

                    next_state = np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)

                    # Save the tuple in replay buffer
                    ego.store_transition(state, action, last_goal[:2], goal[:2], reward, next_state, intervention, action_exp, done)

                    # Train the SAC model
                    if human_guidence or pre_buffer:
                        ego.learn_guidence(intervention, batch_size)
                    else:
                        ego.learn(batch_size)

                    # Update the counters
                    state = next_state
                    s_list.append(s_)

            # After the training is done, evaluate the network and save it
            avg_reward = evaluate(ego, eval_ep, ep_real)
            evaluations.append(avg_reward)
            if avg_reward > save_threshold:
                ego.save(model_name, directory=folder_name, reward=int(np.floor(avg_reward)), seed=seed)

            np.save(os.path.join('final_curves', 'reward_seed' + str(seed) + '_' + model_name), reward_mean_list, allow_pickle=True, fix_imports=True)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
