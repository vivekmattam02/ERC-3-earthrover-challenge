import base64
import time
import numpy as np
from PIL import Image
import os
import sys
import io
from pathlib import Path
# import tensorflow as tf
import logging
import utm
import math
import requests
import json
import pygame 
import torch
import yaml
from PIL import Image as PILImage
from utils_logonav import to_numpy, transform_images_mbra, load_model

import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

MODEL_WEIGHTS_PATH = REPO_ROOT / "deployment" / "model_weights"
MODEL_CONFIG_PATH = REPO_ROOT / "train" / "config"

MAX_V = 0.5
MAX_W = 1.0
RATE = 0.333

# Load the model 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # uncomment this line to use GPU if available
device = torch.device("cpu")

def decode_from_base64(encoded_image):
    """Decode a base64-encoded image to a PIL Image."""
    image_bytes = base64.b64decode(encoded_image)
    return PILImage.open(io.BytesIO(image_bytes))


def request_json(url, method="get", data=None, timeout=5):
    try:
        if method == "get":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, data=data, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP request failed: {method.upper()} {url}: {exc}") from exc

    body = (resp.text or "").strip().replace("\n", " ")
    if resp.status_code >= 400:
        raise RuntimeError(
            f"HTTP {resp.status_code} from {method.upper()} {url}: {body[:200]}"
        )

    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Non-JSON response from {method.upper()} {url} "
            f"(status={resp.status_code}): {body[:200]}"
        ) from exc

def calculate_relative_position(x_a, y_a, x_b, y_b):
    delta_x = x_b - x_a
    delta_y = y_b - y_a
    return delta_x, delta_y


def calculate_distance(x_a, y_a, x_b, y_b):
    return math.hypot(x_b - x_a, y_b - y_a)

# Function to rotate the relative position to the robot's local coordinate system
def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):    
    # Apply the rotation matrix for the local frame
    relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
    relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)    
    return relative_x, relative_y


def clip_angle(angle_rad):
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

class LogoNav_run():
    def __init__(self, goal_utm, goal_compass):
        self.tick_rate = 3 #policy frequency 3 Hz
        self.start_time = time.time()

        self.clear_obs()
        self.context_queue = []
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        
        self.linear = 0.0
        self.angular = 0.0
        self.id_goal = 0
        self.navigation_complete = False

    def clear_obs(self):
        self.observations = []
        self.imgs = {"front": [],
                     "rear": [],
                     "map": []}
        self.traj_len = 0
    
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()    
        start_time_c = time.time()
          
        while True:
            cur_time = time.time()
            if cur_time - start_time > loop_time:
                self.linear, self.angular = self.policy_calc()
                start_time = time.time()
            self.control_send(self.linear, self.angular) 
    
    def policy_calc(self):
        if self.navigation_complete:
            return 0.0, 0.0

        linear_vel = None
        angular_vel = None
        distances = None
        waypoints = None

        #front image from Frodobot        
        newsize = (96, 96)    
        try:
            response = request_json("http://127.0.0.1:8000/v2/front", method="get")
            if "front_frame" not in response:
                raise RuntimeError(f"v2/front missing 'front_frame': {response}")
            img_PIL_resize = decode_from_base64(response["front_frame"]).resize(newsize).convert('RGB')
            # save the images for testing
            img_PIL_resize.save("front_frame.png")
        except Exception as exc:
            print(f"[WARN] Front frame unavailable ({exc}); sending zero command for this cycle.")
            return 0.0, 0.0
        
        #GPS data from Frodobot
        try:
            gpsdata = request_json("http://127.0.0.1:8000/data", method="get")
        except Exception as exc:
            print(f"[WARN] /data unavailable ({exc}); sending zero command for this cycle.")
            return 0.0, 0.0

        # Convert GPS coordinates to UTM. Indoor/no-fix telemetry may be invalid.
        try:
            lat = float(gpsdata.get("latitude"))
            lon = float(gpsdata.get("longitude"))
            if (not np.isfinite(lat)) or (not np.isfinite(lon)) or lat < -80.0 or lat > 84.0 or lon < -180.0 or lon > 180.0:
                raise ValueError(f"invalid lat/lon: lat={lat}, lon={lon}")
            cur_utm = utm.from_latlon(lat, lon)
            # print(f"Current GPS (lat, lon): ({lat}, {lon}), UTM: ({cur_utm[0]}, {cur_utm[1]})")
        except Exception as exc:
            print(f"[WARN] Invalid GPS from /data ({exc}); sending zero command for this cycle.")
            return 0.0, 0.0

        try:
            cur_compass = -float(gpsdata.get("orientation", 0.0)) / 180.0 * 3.141592
        except Exception:
            cur_compass = 0.0
        if context_size is not None:
            if len(self.context_queue) < context_size + 1:
                self.context_queue.append(img_PIL_resize)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(img_PIL_resize)
               
        if len(self.context_queue) > context_size:
            obs_images = transform_images_mbra(self.context_queue)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1) 
            obs_images = obs_images.to(device)    

            metric_waypoint_spacing = 0.25
            target_utm_x = self.goal_utm[self.id_goal][0]
            target_utm_y = self.goal_utm[self.id_goal][1]
            distance_to_target = calculate_distance(cur_utm[0], cur_utm[1], target_utm_x, target_utm_y)
            print(
                f"distance to target {self.id_goal}: {distance_to_target:.2f} m "
                f"(current=({cur_utm[0]:.2f}, {cur_utm[1]:.2f}), "
                f"target=({target_utm_x:.2f}, {target_utm_y:.2f}))"
            )

            is_final_goal = self.id_goal == len(self.goal_utm) - 1
            stop_distance = 1.0  # meters
            if is_final_goal and distance_to_target < stop_distance:
                self.navigation_complete = True
                print(f"Final goal reached within {stop_distance:.2f} m. Stopping robot.")
                return 0.0, 0.0

            delta_x, delta_y = calculate_relative_position(cur_utm[0], cur_utm[1], target_utm_x, target_utm_y)
            relative_x, relative_y = rotate_to_local_frame(delta_x, delta_y, cur_compass)
                
            ## For multiple goal pose navigation: START ##
            thres_dist = 30.0
            thres_update = 5.0 
            distance_goal = np.sqrt(relative_x**2 + relative_y**2)
            if distance_goal > thres_dist:
                relative_x = relative_x/distance_goal*thres_dist
                relative_y = relative_y/distance_goal*thres_dist   
            
            goal_pose = np.array([relative_y/metric_waypoint_spacing, -relative_x/metric_waypoint_spacing, np.cos(self.goal_compass[self.id_goal]-cur_compass), np.sin(self.goal_compass[self.id_goal]-cur_compass)])  
            if distance_goal < thres_update and self.id_goal != len(self.goal_compass)-1:
                self.id_goal += 1            
            
            goal_pose_torch = torch.from_numpy(goal_pose).unsqueeze(0).float().to(device)            
            print("relative pose", goal_pose[0]*metric_waypoint_spacing, goal_pose[1]*metric_waypoint_spacing, goal_pose[2], goal_pose[3], "currently at angle", cur_compass)
            
            with torch.no_grad():  
                waypoints = model(obs_images, goal_pose_torch)                 
            waypoints = to_numpy(waypoints)
                      
        else:
            linear_vel_value = 0.0
            angular_vel_value = 0.0
            
        if waypoints is not None:
            if True:  
                chosen_waypoint = waypoints[0][2].copy()
                if True: #if we apply normalization in training
                    MAX_v = 0.3
                    RATE = 3.0
                    chosen_waypoint[:2] *= (MAX_v / RATE)
                
                dx, dy, hx, hy = chosen_waypoint

                EPS = 1e-8 #default value of NoMaD inference
                DT = 1/4 #default value of NoMaD inference
                
                if np.abs(dx) < EPS and np.abs(dy) < EPS:
                    linear_vel_value = 0
                    angular_vel_value = clip_angle(np.arctan2(hy, hx))/DT
                elif np.abs(dx) < EPS:
                    linear_vel_value =  0
                    angular_vel_value = np.sign(dy) * np.pi/(2*DT)
                else:
                    linear_vel_value = dx / DT
                    angular_vel_value = np.arctan(dy/dx) / DT
                linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
                angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)         

        maxv = 0.3
        maxw = 0.3            
        
        if np.absolute(linear_vel_value) <= maxv:
            if np.absolute(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value/angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.absolute(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.absolute(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value/angular_vel_value
                if np.absolute(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.absolute(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.absolute(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)        
            
        if linear_vel_value_limit < 0.05 and np.absolute(angular_vel_value_limit) < 0.2 and np.absolute(angular_vel_value_limit) > 0.05:
            angular_vel_value_limit = np.sign(angular_vel_value)*0.2
            linear_vel_value_limit = linear_vel_value_limit*0.2/np.absolute(angular_vel_value_limit)
            
        return linear_vel_value_limit, angular_vel_value_limit
                
    def control_send(self, linear, angular):
        data = json.dumps({"command": {"linear": linear, "angular": angular}})
        request_json("http://127.0.0.1:8000/control", method="post", data=data)
            
if __name__ == "__main__":
    # load model parameters
    model_config_path = MODEL_CONFIG_PATH / "LogoNav.yaml"
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = MODEL_WEIGHTS_PATH / "logonav.pth"
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()
    
    #Goal pose, this is under world coordinates
    latlon_g = [[9.973703384399414,-84.37767028808594]] 
    yaw_ang_g = 0.0/180*3.14
    goal_compass_g = [yaw_ang_g] #clock-wise [deg]
    
    goal_utm = []
    goal_compass = []
    for i in range(len(latlon_g)):
        goal_utm.append(utm.from_latlon(latlon_g[i][0], latlon_g[i][1]))
        goal_compass.append(-float(goal_compass_g[i])/180.0*3.141592) #counter clock-wise [rad]

    LogoNav_run(goal_utm = goal_utm, goal_compass = goal_compass).run() 
