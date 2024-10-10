from matplotlib.pyplot import axis
import numpy as np
import os
import random
import torch
import pickle

from utils.torch_jit_utils import *
# from isaacgym.torch_utils import *

from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import matplotlib.pyplot as plt
from PIL import Image as Im
import cv2
from torch import nn
import torch.nn.functional as F

from typing import Dict
    
class RewardComponents:
    def __init__(self, cfg: Dict[str, float]):
        self.init_height_bonus = cfg["initHeightBonus"]
        self.height_reward = cfg["heightReward"]
        self.success_bonus = cfg["successBonus"]
        self.proximity_reward = cfg["proximityReward"]
        self.grasp_reward = cfg["graspReward"]
        self.split_reward = cfg["splitReward"]
        self.split_penalty = cfg["splitPenalty"]
        self.drop_penalty = cfg["dropPenalty"]
        self.action_penalty = cfg["actionPenalty"]
        self.vel_penalty = cfg["velocityPenalty"]
        self.acc_penalty = cfg["accelerationPenalty"]
        self.other_plate_penalty = cfg["otherPlatePenalty"]

class ActionComponents:
    def __init__(self, cfg: Dict[str, float]):
        self.phase1_perturbation_scale = cfg["phase1PerturbationScale"]
        self.phase1_realtime_tracking = cfg["phase1RealtimeTracking"]
        self.phase2_transition_time = cfg["phase2TransitionTime"]
        self.phase2_increment = cfg["phase2Increment"]
        self.phase2_target_height = cfg["phase2TargetHeight"]
        self.phase3_transition_time = cfg["phase3TransitionTime"]
        self.phase3_increment = cfg["phase3Increment"]
        self.phase3_target_height = cfg["phase3TargetHeight"]

class ObservationComponents:
    def __init__(self, cfg):
        self.stack_frames = cfg["stackFrames"]
        self.use_allegro_dof_pos = cfg["useAllegroDofPos"]
        self.use_allegro_dof_vel = cfg["useAllegroDofVel"]
        self.use_allegro_finger_tip_state = cfg["useAllegroFingerTip"]
        self.use_target_plate_corner_dist = cfg["useTargetPlateCornerDist"]
        self.use_ee_info = cfg["useEEInfo"]
        self.use_last_action = cfg["useLastAction"]
        self.use_tactile_sensor = cfg["useTactileSensor"]
        self.use_progress = cfg["useProgress"]

class curriculumLearningComponents:
    def __init__(self, cfg):
        self.success_threshold = cfg["successThreshold"]
        # self.increase_pos_noise = cfg["increasePositioNoise"]
        # self.increase_pos_noise_step = cfg["increasePositionNoiseStep"]
        self.max_pos_noise = cfg["maxPositionNoise"]
        self.increase_mass = cfg["increaseMass"]
        self.increase_mass_scale = cfg["increaseMassScale"]
        self.max_mass = cfg["maxMass"]
        self.decrease_spacing = cfg["decreaseSpacing"]
        self.decrease_spacing_scale = cfg["decreaseSpacingScale"]
        # self.decrease_action = cfg["decreaseAction"]
        # self.decrease_action_scale = cfg["decreaseActionScale"]
        # self.decrease_action_init = cfg["decreaseActionInit"]
        # self.decrease_action_min = cfg["decreaseActionMin"]

class RandomizationComponent:
    def __init__(self, cfg):
        self.reset_target_plate = cfg["resetTargetPlate"]
        self.reset_center_pos_noise = cfg["resetCenterPositionNoise"]
        self.reset_spacing_noise = cfg["resetSpacingNoise"]
        self.reset_orit_noise = cfg["resetOrientationNoise"]
        self.reset_plate_thickness = cfg["resetPlateThickness"]
        self.reset_box_width = cfg["resetBoxWidth"]
        self.reset_ee_pos_noise = cfg["resetEEPosNoise"]

class EvaluationComponent:
    def __init__(self, cfg):
        self.consecutive_time_steps = cfg["consecutiveTimeSteps"]
        self.log_trajectory = cfg["logTrajectory"]
        self.log_trajectory_file = cfg["logTrajectoryFile"]
        self.run_trajectory = cfg["runTrajectory"]
        self.run_trajectory_file = cfg["runTrajectoryFile"]
        self.log_dof_state = cfg["logDofState"]
        self.log_dof_pos_file = cfg["logDofPosFile"]
        self.log_dof_vel_file = cfg["logDofVelFile"]
        self.log_dof_effort_file = cfg["logDofEffortFile"]
        self.log_obs_space = cfg["logObsSpace"]
        self.log_obs_space_file = cfg["logObsSpaceFile"]
        self.stop_eps = cfg["stopEps"]

class DebugConfig:
    def __init__(self, cfg):
        self.target_init_pos = cfg["debugTargetInitPos"]
        self.hand_center = cfg["debugHandCenter"]
        self.allegro_ee_pos = cfg["debugAllegroEEPos"]
        self.hand_tips = cfg["debugHandTips"]
        self.contact_sensors = cfg["debugContactSensors"]
        self.is_contact = cfg["debugIsContact"]
        self.plate_center = cfg["debugPlateCenter"]
        self.plate_corners = cfg["debugPlateCorners"]
        self.plate_edges = cfg["debugPlateEdges"]
        self.box_corners = cfg["debugBoxCorners"]
        self.neighbor_pose = cfg["debugNeighborPose"]
        
        self.viz = any([
            self.target_init_pos,
            self.hand_center,
            self.allegro_ee_pos,
            self.hand_tips,
            self.contact_sensors,
            self.is_contact,
            self.plate_center,
            self.plate_corners,
            self.plate_edges,
            self.box_corners,
            self.neighbor_pose
        ])

class TableConfig:
    def __init__(self, cfg):
        self.size = cfg["tableSize"]
        self.pos = cfg["tablePosition"]

    def load_asset(self, gym, sim):
        self.gym = gym
        self.sim = sim

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(self.pos[0], self.pos[1], self.pos[2])

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = gym.create_box(sim, self.size[0], self.size[1], self.size[2], asset_options)

        return table_pose, table_asset
    
    def get_base_pose(self, device, box_thin):
        return to_torch([self.pos[0], self.pos[1], self.pos[2] + self.size[2] / 2 + box_thin], device=device)
    
class BoxConfig:
    def __init__(self, cfg):
        self.thickness = cfg["boxThickness"]
        self.height = cfg["boxHeight"]
        # self.box_plate_spacing = cfg["boxPlateSpacing"]
        self.width = cfg["boxWidth"]
        self.box_plate_side_spacing = cfg["boxPlateSideSpacing"]
        self.box_deg = cfg["boxDegree"]
        self.friction = cfg["boxFriction"]
        self.restitution = cfg["boxRestitution"]

    def load_asset(self, gym, sim, device, table_cfg, plate_cfg):
        self.gym = gym
        self.sim = sim
        self.device = device

        # create box asset
        box_assets = []
        box_start_poses = []

        plate_size = plate_cfg.size
        plate_spacing = plate_cfg.spacing
        plate_thickness = plate_cfg.thickness
        # calculated_box_width = plate_thickness * plate_cfg.num_plates + plate_spacing * (plate_cfg.num_plates - 1) \
        #     + self.box_plate_spacing * 2 + self.thickness * 2
        
        x_size = self.width
        y_size = plate_size + self.box_plate_side_spacing * 2 + self.thickness * 2
        z_size = self.height + self.thickness
        self.box_xyz = [x_size, y_size, z_size]
        box_center = [table_cfg.pos[0], table_cfg.pos[1], table_cfg.pos[2] + table_cfg.size[2] / 2]

        box_asset_options = gymapi.AssetOptions()
        box_asset_options.fix_base_link = True
        box_asset_options.flip_visual_attachments = True
        box_asset_options.collapse_fixed_joints = True
        box_asset_options.disable_gravity = True
        box_asset_options.thickness = 0.001

        box_bottom_asset = gym.create_box(sim, self.box_xyz[0], self.box_xyz[1], self.thickness, box_asset_options)
        box_left_asset = gym.create_box(sim, self.box_xyz[0], self.thickness, self.box_xyz[2], box_asset_options)
        box_right_asset = gym.create_box(sim, self.box_xyz[0], self.thickness, self.box_xyz[2], box_asset_options)
        box_former_asset = gym.create_box(sim, self.thickness, self.box_xyz[1], self.box_xyz[2], box_asset_options)
        box_after_asset = gym.create_box(sim, self.thickness, self.box_xyz[1], self.box_xyz[2], box_asset_options)

        box_bottom_start_pose = gymapi.Transform()
        rotated_pos, rotated_quat = rotate_around_z_axis_with_offset(
            to_torch([
                0.0 + box_center[0],
                0.0 + box_center[1],
                (self.thickness) / 2 + box_center[2]], device=self.device),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg,
            to_torch(box_center, device=self.device)
        )
        box_bottom_start_pose.p = gymapi.Vec3(rotated_pos[0], rotated_pos[1], rotated_pos[2])
        box_bottom_start_pose.r = gymapi.Quat(rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3])
        
        box_left_start_pose = gymapi.Transform()
        rotated_pos, rotated_quat = rotate_around_z_axis_with_offset(
            to_torch([
                0.0 + box_center[0],
                (self.box_xyz[1] - self.thickness) / 2 + box_center[1],
                (self.box_xyz[2]) / 2 + box_center[2]], device=self.device),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg,
            to_torch(box_center, device=self.device)
        )
        box_left_start_pose.p = gymapi.Vec3(rotated_pos[0], rotated_pos[1], rotated_pos[2])
        box_left_start_pose.r = gymapi.Quat(rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3])
        
        box_right_start_pose = gymapi.Transform()
        rotated_pos, rotated_quat = rotate_around_z_axis_with_offset(
            to_torch([
                0.0 + box_center[0],
                -(self.box_xyz[1] - self.thickness) / 2 + box_center[1],
                (self.box_xyz[2]) / 2 + box_center[2]], device=self.device),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg,
            to_torch(box_center, device=self.device)
        )
        box_right_start_pose.p = gymapi.Vec3(rotated_pos[0], rotated_pos[1], rotated_pos[2])
        box_right_start_pose.r = gymapi.Quat(rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3])
        
        box_former_start_pose = gymapi.Transform()
        rotated_pos, rotated_quat = rotate_around_z_axis_with_offset(
            to_torch([
                (self.box_xyz[0] - self.thickness) / 2 + box_center[0],
                0.0 + box_center[1],
                (self.box_xyz[2]) / 2 + box_center[2]], device=self.device),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg,
            to_torch(box_center, device=self.device)
        )
        box_former_start_pose.p = gymapi.Vec3(rotated_pos[0], rotated_pos[1], rotated_pos[2])
        box_former_start_pose.r = gymapi.Quat(rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3])
        
        box_after_start_pose = gymapi.Transform()
        rotated_pos, rotated_quat = rotate_around_z_axis_with_offset(
            to_torch([
                -(self.box_xyz[0] - self.thickness) / 2 + box_center[0],
                0.0 + box_center[1],
                (self.box_xyz[2]) / 2 + box_center[2]], device=self.device),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg,
            to_torch(box_center, device=self.device)
        )
        box_after_start_pose.p = gymapi.Vec3(rotated_pos[0], rotated_pos[1], rotated_pos[2])
        box_after_start_pose.r = gymapi.Quat(rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3])

        box_assets.append(box_bottom_asset)
        box_assets.append(box_left_asset)
        box_assets.append(box_right_asset)
        box_assets.append(box_former_asset)
        box_assets.append(box_after_asset)
        box_start_poses.append(box_bottom_start_pose)
        box_start_poses.append(box_left_start_pose)
        box_start_poses.append(box_right_start_pose)
        box_start_poses.append(box_former_start_pose)
        box_start_poses.append(box_after_start_pose)

        self.original_box_center = to_torch(box_center, device=self.device) # [3]

        return box_start_poses, box_assets

    def init_tensors(self, device, num_envs, indices, start_state):
        self.device = device
        self.num_envs = num_envs
        self.indices = to_torch(indices, dtype=torch.long, device=self.device
                                ).view(self.num_envs, 5)
        self.original_start_state = to_torch(start_state, dtype=torch.float, device=self.device
                                    ).view(self.num_envs, 5, 13)
        self.start_state = torch.clone(self.original_start_state)
        self.box_center = torch.clone(self.original_box_center).unsqueeze(0).repeat(self.num_envs, 1) # [num_envs, 3]

    def get_corners(self):
        half_size = to_torch(self.box_xyz, device=self.device) / 2  # [3]
        # box_deg = to_torch(self.box_deg, device=self.device)
        # rotation_matrix = torch.tensor([
        #     [torch.cos(box_deg), -torch.sin(box_deg), 0],
        #     [torch.sin(box_deg), torch.cos(box_deg), 0],
        #     [0, 0, 1]], device=self.device)
        # half_size = torch.matmul(rotation_matrix, half_size.unsqueeze(1)).squeeze(1)

        bottom_right, _ = rotate_around_z_axis_with_offset(
            to_torch([half_size[0], -half_size[1], 0], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        top_right, _ = rotate_around_z_axis_with_offset(
            to_torch([half_size[0], half_size[1], 0], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        bottom_right_top, _ = rotate_around_z_axis_with_offset(
            to_torch([half_size[0], -half_size[1], 2*half_size[2]], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        top_right_top, _ = rotate_around_z_axis_with_offset(
            to_torch([half_size[0], half_size[1], 2*half_size[2]], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        
        bottom_left, _ = rotate_around_z_axis_with_offset(
            to_torch([-half_size[0], -half_size[1], 0], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        top_left, _ = rotate_around_z_axis_with_offset(
            to_torch([-half_size[0], half_size[1], 0], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        bottom_left_top, _ = rotate_around_z_axis_with_offset(
            to_torch([-half_size[0], -half_size[1], 2*half_size[2]], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        top_left_top, _ = rotate_around_z_axis_with_offset(
            to_torch([-half_size[0], half_size[1], 2*half_size[2]], device=self.device).repeat(self.num_envs, 1),
            to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
            self.box_deg, to_torch([0.0, 0.0, 0.0], device=self.device)
        )
        
        corners_global = torch.stack([
            bottom_right, top_right, bottom_right_top, top_right_top,
            bottom_left, top_left, bottom_left_top, top_left_top
            ], dim=1) + self.box_center.unsqueeze(1) # [num_envs, 8, 3]
        return corners_global

    def randomize_pos(self, env_ids, rand_pos):
        # rand_pos: [len(env_ids), 2]
        self.start_state[env_ids, :, 0:2] = self.original_start_state[env_ids, :, 0:2] + rand_pos.unsqueeze(1) # [len(env_ids), 5, 2]
        self.box_center[env_ids, :2] = self.original_box_center[:2].unsqueeze(0) + rand_pos # [len(env_ids), 3]
    
    def reset_state(self, env_ids, root_state_tensor):
        # root_state_tensor: [num_envs * num_actors, 13]
        root_state_tensor[self.indices[env_ids]] = self.start_state[env_ids] # [len(env_ids), num_box_assets, 13]

class PlateConfig:
    def __init__(self, cfg):
        self.num_plates = cfg["numPlates"]
        self.size = cfg["plateSize"]
        self.thickness = cfg["plateThickness"]
        self.density = cfg["plateDensity"]
        self.spacing = cfg["plateSpacing"]
        self.initial_height = cfg["plateInitialHeight"]
        self.friction = cfg["plateFriction"]
        self.restitution = cfg["plateRestitution"]
        self.original_target_idx = cfg["targetPlateIndex"]

        assert 0 <= self.original_target_idx < self.num_plates

        # Precompute colors
        colors = [
            gymapi.Vec3(1.0, 0.5, 0.5),  # Light Red
            gymapi.Vec3(1.0, 0.8, 0.4),  # Peach
            gymapi.Vec3(1.0, 1.0, 0.5),  # Light Yellow
            gymapi.Vec3(0.5, 1.0, 0.5),  # Light Green
            gymapi.Vec3(0.5, 0.5, 1.0),  # Light Blue
            gymapi.Vec3(0.5, 0.2, 0.5),  # Plum
            gymapi.Vec3(0.7, 0.3, 1.0),  # Lavender
        ]
        self.precomputed_colors = []
        for i in range(self.num_plates):
            self.precomputed_colors.append(colors[i % len(colors)])
        self.target_color = gymapi.Vec3(1, 0, 0)

    def load_asset(self, gym, sim):
        self.gym = gym
        self.sim = sim

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.002
        asset_options.density = self.density

        plate_asset = gym.create_box(sim, self.thickness, self.size, self.size, asset_options)

        return plate_asset

    def init_tensors(self, device, num_envs, indices, start_state, plates_handles):
        self.device = device
        self.num_envs = num_envs
        self.dims = to_torch([self.thickness, self.size, self.size], device=self.device)
        self.indices = to_torch(indices, dtype=torch.long, device=self.device
                                ).view(self.num_envs, self.num_plates)
        self.original_start_state = to_torch(start_state, dtype=torch.float, device=self.device \
                                    ).view(self.num_envs, self.num_plates, 13)
        self.start_state = torch.clone(self.original_start_state)
        self.plates_handles = plates_handles # [num_envs, num_plates] in python array

        self.target_indices = torch.zeros((num_envs), dtype=torch.int64, device=self.device)
        self.target_indices[:] = self.original_target_idx

        self.corners = torch.zeros((self.num_envs, self.num_plates, 4, 3), device=self.device)
        self.edges = torch.zeros((self.num_envs, self.num_plates, 4, 3), device=self.device)
        self.target_nearby_vec = torch.zeros((self.num_envs, 2, 4, 3), # [num_envs, 2, 4, 3]
                                  dtype=torch.float, device=self.device)
        self.target_nearby_pose = torch.zeros((self.num_envs, 2, 7), # [num_envs, 2, 7]
                                  dtype=torch.float, device=self.device)
        
        self.is_contact = torch.zeros((self.num_envs, self.num_plates, 4), # [num_envs, num_plates. 4]
                                      dtype=torch.bool, device=self.device)
        # self.is_grasping = torch.zeros((self.num_envs, self.num_plates), # [num_envs, num_plates]
        #                           dtype=torch.bool, device=self.device)
        
        # for curriculum learning
        self.mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.spacing = torch.full((self.num_envs,), self.spacing, dtype=torch.float, device=self.device)
        
    def _to_plate_frame(self, input_pos, plate_pos, plate_quat):
        # Convert the global position to the plate's local frame
        
        # Inverse the plate's quaternion
        plate_quat_inv = quat_conjugate(plate_quat)
        
        # Translate fingertip position to plate's origin
        local_pos = input_pos - plate_pos
        
        # Convert local_pos to tensor if it's not already
        if not isinstance(local_pos, torch.Tensor):
            local_pos = to_torch(local_pos, device=self.device)
        
        # Rotate to account for plate's orientation
        local_pos = quat_apply(plate_quat_inv, local_pos)
        
        return local_pos
        
    def _get_corners(self, plate_pos, plate_quat):
        half_size = self.dims / 2

        bottom_left = quat_apply(plate_quat, to_torch([0, -half_size[1], -half_size[2]], device=self.device).repeat(self.num_envs, 1))
        bottom_right = quat_apply(plate_quat, to_torch([0, half_size[1], -half_size[2]], device=self.device).repeat(self.num_envs, 1))
        top_left = quat_apply(plate_quat, to_torch([0, -half_size[1], half_size[2]], device=self.device).repeat(self.num_envs, 1))
        top_right = quat_apply(plate_quat, to_torch([0, half_size[1], half_size[2]], device=self.device).repeat(self.num_envs, 1))
        
        # mid_bottom = quat_apply(plate_quat, to_torch([0, -half_size[1], 0], device=self.device).repeat(self.num_envs, 1))
        # mid_right = quat_apply(plate_quat, to_torch([half_size[0], 0, 0], device=self.device).repeat(self.num_envs, 1))
        # mid_top = quat_apply(plate_quat, to_torch([0, half_size[1], 0], device=self.device).repeat(self.num_envs, 1))
        # mid_left = quat_apply(plate_quat, to_torch([-half_size[0], 0, 0], device=self.device).repeat(self.num_envs, 1))

        corners_global = torch.stack([
            bottom_left, bottom_right, top_left, top_right, 
            # mid_bottom, mid_right, mid_top, mid_left
            ], dim=1) + plate_pos.unsqueeze(1)
        return corners_global
    
    def _get_edges(self, plate_pos, plate_quat):
        half_size = self.dims / 2

        mid_right = quat_apply(plate_quat, to_torch([0, -half_size[1], 0], device=self.device).repeat(self.num_envs, 1))
        mid_bottom = quat_apply(plate_quat, to_torch([0, 0, half_size[2]], device=self.device).repeat(self.num_envs, 1))
        mid_left = quat_apply(plate_quat, to_torch([0, half_size[1], 0], device=self.device).repeat(self.num_envs, 1))
        mid_top = quat_apply(plate_quat, to_torch([0, 0, -half_size[2]], device=self.device).repeat(self.num_envs, 1))

        edges_global = torch.stack([
            mid_bottom, mid_right, mid_top, mid_left
            ], dim=1) + plate_pos.unsqueeze(1)
        return edges_global

    def update_state(self, root_state_tensor, box_cfg):
        # root_state_tensor: [num_envs * num_actors, 13]
        self.state = root_state_tensor[self.indices, 0:13] # [num_envs, num_plates, 13]
        self.pose = root_state_tensor[self.indices, 0:7] # [num_envs, num_plates, 7]
        self.pos = root_state_tensor[self.indices, 0:3] # [num_envs, num_plates, 3]
        self.rot = root_state_tensor[self.indices, 3:7] # [num_envs, num_plates, 4]
        self.linvel = root_state_tensor[self.indices, 7:10] # [num_envs, num_plates, 3]
        self.angvel = root_state_tensor[self.indices, 10:13] # [num_envs, num_plates, 3]

        # get all plates corners & edges
        for i in range(self.num_plates):
            plate_corners = self._get_corners(self.pos[:, i, :], self.rot[:, i, :])  # [num_envs, 4, 3]
            plate_edges = self._get_edges(self.pos[:, i, :], self.rot[:, i, :])  # [num_envs, 4, 3]
            self.corners[:, i, :, :] = plate_corners
            self.edges[:, i, :, :] = plate_edges

        # Get corners once for all environments
        box_corners = box_cfg.get_corners()  # [num_envs, 8, 3]
        box_corner_tmp = box_corners.clone()
        subtract_idx = 1 if box_cfg.box_deg == 0 else 0
        box_corner_tmp[:, :, subtract_idx] -= 0.015 * torch.sign(box_corner_tmp[:, :, subtract_idx] - box_cfg.box_center[:, subtract_idx].unsqueeze(1))
        box_corner_tmp[:, :, 2] += (self.dims[2] - box_cfg.height)

        # Calculate vectors for target_idx == 0 and target_idx == num_plates - 1 cases separately
        zero_indices = (self.target_indices == 0) # [num_envs]
        non_zero_indices = ~zero_indices # [num_envs]
        max_indices = (self.target_indices == self.num_plates - 1) # [num_envs]
        non_max_indices = ~max_indices # [num_envs]

        # Handling target_idx == 0
        self.target_nearby_vec[zero_indices, 0, :, :] = \
            box_corner_tmp[zero_indices, 4:, :] - self.corners[zero_indices, 0, :, :]
        
        # Default case for target_idx > 0 and not the last plate
        self.target_nearby_vec[non_zero_indices, 0, :, :] = \
            self.corners[non_zero_indices, self.target_indices[non_zero_indices] - 1, :, :] - \
            self.corners[non_zero_indices, self.target_indices[non_zero_indices], :, :]

        # Handling target_idx == num_plates - 1
        self.target_nearby_vec[max_indices, 1, :, :] = \
            box_corner_tmp[max_indices, :4, :] - self.corners[max_indices, self.num_plates - 1, :, :]

        # Default case for target_idx < num_plates - 1
        self.target_nearby_vec[non_max_indices, 1, :, :] = \
            self.corners[non_max_indices, self.target_indices[non_max_indices] + 1, :, :] - \
            self.corners[non_max_indices, self.target_indices[non_max_indices], :, :]     
        
        imaginary_plate_pose = torch.zeros((self.num_envs, 7), device=self.device)
        imaginary_plate_pose[:, 3:] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

        imaginary_plate_pose[zero_indices, :3] = torch.mean(box_corner_tmp[zero_indices, 4:, :], dim=-2)
        self.target_nearby_pose[zero_indices, 0] = imaginary_plate_pose[zero_indices].clone()
        self.target_nearby_pose[non_zero_indices, 0] = self.pose[non_zero_indices, self.target_indices[non_zero_indices] - 1, :]
        
        imaginary_plate_pose[max_indices, :3] = torch.mean(box_corner_tmp[max_indices, :4, :], dim=-2)
        self.target_nearby_pose[max_indices, 1] = imaginary_plate_pose[max_indices].clone()
        self.target_nearby_pose[non_max_indices, 1] = self.pose[non_max_indices, self.target_indices[non_max_indices] + 1, :]
        
    # def get_target_plate_info(self):
    #     poses = self.plate_cfg.pose[torch.arange(self.num_envs), self.plate_cfg.target_indices, :] # [num_envs, 7]
    #     linvels = self.plate_cfg.linvel[torch.arange(self.num_envs), self.plate_cfg.target_indices, :] # [num_envs, 3]
    #     angvels = self.plate_cfg.angvel[torch.arange(self.num_envs), self.plate_cfg.target_indices, :] # [num_envs, 3]
    #     concatenated = torch.cat([poses, linvels, angvels], dim=-1)  # Shape: [num_envs, 13]
    #     return concatenated

    # def get_other_max_height(self):
    #     all_indices = torch.arange(self.num_plates, device=self.device).expand(self.num_envs, -1) # [num_envs, num_plates]
    #     mask = all_indices != self.target_indices.unsqueeze(1) # [num_envs, num_plates]
    #     other_plates_pose = self.plate_pose[mask].view(self.num_envs, self.num_plates-1, -1) # [num_envs, num_plates-1, 7]
    #     other_plates_height = other_plates_pose[:, :, 2] - 0.5 * self.plate_dims[2].unsqueeze(-1) # [num_envs, num_plates-1]
    #     other_height = other_plates_height - base_height.unsqueeze(1) # [num_envs, num_plates-1]
    #     other_height = torch.max(other_height, torch.zeros_like(other_height)) # at least zero
    #     max_height = torch.max(other_height, dim=1).values # [num_envs]
    #     return max_height

    def calc_fingertip_plate(self, allegro_hand_ff_pos, allegro_hand_mf_pos, 
                        allegro_hand_rf_pos, allegro_hand_th_pos):
        grasping_threshold = 0.018
        ignoring_edge_width = 0.01
        # Assuming self.num_envs and self.num_plates are defined

        for i in range(self.num_plates):
            all_fingers = [allegro_hand_ff_pos, allegro_hand_mf_pos, 
                                allegro_hand_rf_pos, allegro_hand_th_pos]
            local_contact_pos = torch.stack([self._to_plate_frame(pos, self.pos[:, i, :], self.rot[:, i, :]) 
                                            for pos in all_fingers], dim=1)  # [num_envs, num_contacts, 3]

            # Vectorized clamping and distance calculation
            clamped_x = torch.clamp(local_contact_pos[:, :, 0], -self.dims[0] / 2, self.dims[0] / 2)
            clamped_y = torch.clamp(local_contact_pos[:, :, 1], -self.dims[1] / 2, self.dims[1] / 2)
            clamped_z = torch.clamp(local_contact_pos[:, :, 2], -self.dims[2] / 2 - ignoring_edge_width, self.dims[2] / 2 - ignoring_edge_width)
            clamped_pos = torch.stack([clamped_x, clamped_y, clamped_z], dim=2)  # [num_envs, num_contacts, 3]
            dist_to_surface = torch.norm(local_contact_pos - clamped_pos, dim=2)  # [num_envs, num_contacts]

            # Check if the fingertip is within the grasping threshold
            is_contact = dist_to_surface < grasping_threshold # [num_envs, num_contacts]
            self.is_contact[:, i, :] = is_contact # [num_envs, num_contacts]

            # # Determine which side of the block the fingertip is on
            # side_x = torch.sign(clamped_pos[:, :, 0]) # [num_envs, num_contacts]

            # # Check for contacts on opposite sides along any dimension
            # opposite_sides_x = (side_x.unsqueeze(2) * side_x.unsqueeze(1)) < 0

            # # Ensure y and z positions are within the plate dimensions
            # within_bounds_y = (local_contact_pos[..., 1].abs() < self.dims[1] / 2 - 0.005).unsqueeze(2) & \
            #                 (local_contact_pos[..., 1].abs() < self.dims[1] / 2 - 0.005).unsqueeze(1)
            # within_bounds_z = (local_contact_pos[..., 2].abs() < self.dims[2] / 2 - 0.005).unsqueeze(2) & \
            #                 (local_contact_pos[..., 2].abs() < self.dims[2] / 2 - 0.005).unsqueeze(1)
            
            # valid_opposite_and_within_bounds = opposite_sides_x & within_bounds_y & within_bounds_z

            # # Ensure that only contacts within the threshold are considered
            # valid_contacts = is_contact.unsqueeze(2) & is_contact.unsqueeze(1)
            # opposite_and_valid = valid_opposite_and_within_bounds & valid_contacts

            # # Check if there's at least one pair of valid opposite contacts for each environment
            # self.is_grasping[:, i] = opposite_and_valid.any(dim=(1, 2)) # [num_envs]

    def reset_state(self, env_ids, root_state_tensor):
        # root_state_tensor: [num_envs * num_actors, 13]
        root_state_tensor[self.indices[env_ids]] = self.start_state[env_ids] # [len(env_ids), num_plates, 13]
    
    def randomize_pos(self, env_ids, rand_pos):
        # randomize the plate position
        self.start_state[env_ids, :, 0:2] = self.original_start_state[env_ids, :, 0:2] + rand_pos.unsqueeze(1) # [len(env_ids), num_plates, 2]

    def randomize_pitch(self, env_ids, rand_pitch):
        # randomize the plate pitch
        for j in range(self.num_plates):
            roll, pitch, yaw = get_euler_xyz(self.start_state[env_ids, j, 3:7]) # [num_envs, 3]
            pitch += rand_pitch[:, 0]  # Only perturb the pitch component
            noisy_quat = quat_from_euler_xyz(roll, pitch, yaw) # [num_envs, 4]
            self.start_state[env_ids, j, 3:7] = noisy_quat

    def randomize_mass(self, envs, env_ids, rand_mass):
        self.mass[env_ids] = rand_mass.squeeze(1) # [len(env_ids)]
        for env_id in env_ids:
            for j in range(self.num_plates):
                plate_handle = self.plates_handles[env_id][j]
                mass_props = self.gym.get_actor_rigid_body_properties(envs[env_id], plate_handle)
                for prop in mass_props:
                    prop.mass = self.mass[env_id]
                self.gym.set_actor_rigid_body_properties(envs[env_id], plate_handle, mass_props, recomputeInertia=True)

    def reset_spacing(self, env_ids, spacing, box_cfg):
        self.spacing[env_ids] = spacing # [len(env_ids)]

        for env_id in env_ids:
            plates_width = self.num_plates * self.thickness + (self.num_plates - 1) * self.spacing[env_id]
            plates_center_x = box_cfg.box_center[env_id][0]
            plates_left_x = plates_center_x - 0.5 * plates_width
            plates_right_x = plates_center_x + 0.5 * plates_width
            plates_x_counter = plates_left_x

            for j in range(self.num_plates):
                x = plates_x_counter + 0.5 * self.thickness
                plates_x_counter += self.thickness + self.spacing[env_id]
                y = box_cfg.box_center[env_id][1]
                z = box_cfg.box_center[env_id][2] + box_cfg.thickness + 0.5 * self.size
                rotated_pos, rotated_quat = rotate_around_z_axis_with_offset(
                    to_torch([x, y, z + self.initial_height], device=self.device),
                    to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
                    box_cfg.box_deg, box_cfg.box_center[env_id]
                )
                # update the plate initial position
                self.start_state[env_id, j, 0:7] = to_torch([rotated_pos[0], rotated_pos[1], rotated_pos[2], 
                                                             rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3]], 
                                                             device=self.device)
            assert plates_x_counter - self.spacing[env_id] - plates_right_x < 10e-5

    def randomize_spacing(self, env_ids, rand_spacing, box_cfg):
        self.reset_spacing(env_ids, rand_spacing.squeeze(1), box_cfg)

    def perturb_pose(self, env_ids, pos_noise_std=0.005, rot_noise_std=0.02):
        # Generate unique noise for each environment and each plate
        pos_noise = torch.randn(len(env_ids), self.num_plates, 3, device=self.device) * pos_noise_std # [num_envs, num_plates, 3]
        rot_noise = torch.randn(len(env_ids), self.num_plates, 3, device=self.device) * rot_noise_std # [num_envs, num_plates, 3]

        # Apply position noise
        self.start_state[env_ids, :, 0:3] += pos_noise # [num_envs, num_plates, 3]

        # Apply rotation noise
        for j in range(self.num_plates):
            roll, pitch, yaw = get_euler_xyz(self.start_state[env_ids, j, 3:7]) # [num_envs, 3]
            euler_angles = torch.stack([roll, pitch, yaw], dim=1) # [num_envs, 3]
            noisy_rot = euler_angles + rot_noise[:, j, :] # [num_envs, 3]
            noisy_quat = quat_from_euler_xyz(noisy_rot[:, 0], noisy_rot[:, 1], noisy_rot[:, 2]) # [num_envs, 4]
            self.start_state[env_ids, j, 3:7] = noisy_quat

    def increase_mass(self, envs, env_ids, increase_scale=2, max_mass=1.0):
        for env_id in env_ids:
            for j in range(self.num_plates):
                plate_handle = self.plates_handles[env_id][j]
                mass_props = self.gym.get_actor_rigid_body_properties(envs[env_id], plate_handle)
                for prop in mass_props:
                    prop.mass = min(prop.mass * increase_scale, max_mass)
                    self.mass[env_id] = prop.mass
                self.gym.set_actor_rigid_body_properties(envs[env_id], plate_handle, mass_props, recomputeInertia=True)

        return self.mass # [num_envs]
    
    def decrease_spacing(self, env_ids, box_cfg, decrease_scale=0.5):
        spacing = self.spacing[env_ids] * decrease_scale # update spacing
        spacing = torch.where(spacing <= 0.002, 0.0, spacing)

        self.reset_spacing(env_ids, spacing, box_cfg)
        return self.spacing
    
    def randomize_target_idx(self, envs, env_ids):
        num_envs = len(env_ids)
        prev_target_idx = self.target_indices[env_ids].clone()
        new_target_idx = torch.randint(0, self.num_plates, (num_envs,), device=self.device, dtype=torch.long)
        
        self.target_indices[env_ids] = new_target_idx
        
        for i, env_id in enumerate(env_ids):
            if prev_target_idx[i] != new_target_idx[i]:
                self.gym.set_rigid_body_color(envs[env_id], self.plates_handles[env_id][prev_target_idx[i]], 0, gymapi.MESH_VISUAL_AND_COLLISION, self.precomputed_colors[prev_target_idx[i]])
                self.gym.set_rigid_body_color(envs[env_id], self.plates_handles[env_id][new_target_idx[i]], 0, gymapi.MESH_VISUAL_AND_COLLISION, self.target_color)

class AllegroVertical(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        # for randomization
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.rand_cfg = RandomizationComponent(self.cfg["env"]["randomization"])
        self.eval_cfg = EvaluationComponent(self.cfg["env"]["evaluation"])

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        
        # other configurations
        self.act_moving_average_hand = self.cfg["env"]["actionsMovingAverageHand"]
        self.act_moving_average_arm = self.cfg["env"]["actionsMovingAverageArm"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        self.up_axis = 'z'
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        # load table configurations
        self.table_cfg = TableConfig(self.cfg["env"]["tableConfig"])

        # load box configurations
        self.box_cfg = BoxConfig(self.cfg["env"]["boxConfig"])

        # load allegro hand configurations
        self.allegro_config = self.cfg["env"]["pandaAllegroConfig"]
        self.fingertip_names = self.allegro_config["fingertipNames"]
        self.arm_movable_dof_names = self.allegro_config["armMovableDofNames"]
        self.contact_sensor_names = self.fingertip_names

        # load plate configurations
        self.plate_cfg = PlateConfig(self.cfg["env"]["plateConfig"])

        # debug configurations
        self.debug = DebugConfig(self.cfg["env"]["debugConfig"])

        # set the action space
        self.act = ActionComponents(self.cfg["env"]["actionDesign"])

        # set the observation space
        self.obs = ObservationComponents(self.cfg["env"]["observationDesign"])

        # set the state space
        self.state = ObservationComponents(self.cfg["env"]["stateDesign"])

        # set the curriculum learning components
        self.curriculum = curriculumLearningComponents(self.cfg["env"]["curriculumLearningDesign"])

        self.num_arm_dof = self.allegro_config["armDof"]
        self.num_total_dof = self.allegro_config["totalDof"]
        self.one_frame_num_obs = 0
        if self.obs.use_allegro_dof_pos:
            self.one_frame_num_obs += self.num_total_dof - self.num_arm_dof
        if self.obs.use_allegro_dof_vel:
            self.one_frame_num_obs += self.num_total_dof - self.num_arm_dof
        if self.obs.use_allegro_finger_tip_state:
            self.one_frame_num_obs += 4 * (13 - 6)
        if self.obs.use_target_plate_corner_dist:
            self.one_frame_num_obs += (6 + 12)
        else:
            self.one_frame_num_obs += 3 * 7
        if self.obs.use_ee_info:
            self.one_frame_num_obs += 9 # 12
        if self.obs.use_last_action:
            self.one_frame_num_obs += self.num_total_dof - self.num_arm_dof # + 3
        if self.obs.use_tactile_sensor:
            self.one_frame_num_obs += len(self.contact_sensor_names)
        if self.obs.use_progress:
            self.one_frame_num_obs += 1

        # extract reward scalers
        self.rewards_1 = RewardComponents(self.cfg["env"]["rewardPhase1"])
        self.rewards_2 = RewardComponents(self.cfg["env"]["rewardPhase2"])
        self.rewards_3 = RewardComponents(self.cfg["env"]["rewardPhase3"])

        # set the state space
        self.one_frame_num_states = 0
        if self.asymmetric_obs:
            if self.state.use_allegro_dof_pos:
                self.one_frame_num_states += self.num_total_dof
            if self.state.use_allegro_dof_vel:
                self.one_frame_num_states += self.num_total_dof
            if self.state.use_allegro_finger_tip_state:
                self.one_frame_num_states += 4 * 13
            if self.state.use_target_plate_corner_dist:
                self.one_frame_num_states += 13 + 12 + 24
            else:
                self.one_frame_num_states += 21 + 12 + 24
            if self.state.use_ee_info:
                self.one_frame_num_states += 12
            if self.state.use_last_action:
                self.one_frame_num_states += self.num_total_dof
            if self.state.use_tactile_sensor:
                self.one_frame_num_states += len(self.contact_sensor_names)
            if self.state.use_progress:
                self.one_frame_num_states += 1
        
        self.cfg["env"]["numActions"] =self.num_total_dof
        self.cfg["env"]["numObservations"] = self.one_frame_num_obs * self.obs.stack_frames
        self.cfg["env"]["numStates"] = self.one_frame_num_states * self.obs.stack_frames

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        
        super().__init__(cfg=self.cfg)

        # set up the viewer
        if self.viewer != None:
            direction = self.cfg["env"]["viewerConfig"]["direction"]
            if direction == "left":
                cam_pos = gymapi.Vec3(0.5, 1, 0.5)
            elif direction == "right":
                cam_pos = gymapi.Vec3(0.5, -1, 0.5)
            elif direction == "top":
                cam_pos = gymapi.Vec3(0, 0, 1)
            else:
                cam_pos = gymapi.Vec3(0.5, 1, 0.5) # default left view

            cam_target = gymapi.Vec3(0.5, 0, 0.3)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "hand")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # [num_envs * num_dofs, 2]

        """ This part is for the allegro hand dof states, mostly used when computing the action space """
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2) # [num_envs, num_dofs, 2]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0] # [num_envs, num_dofs]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1] # [num_envs, num_dofs]
        print("self.dof_state:", self.dof_state.shape)
        print("self.allegro_hand_dof_state:", self.allegro_hand_dof_state.shape)
        print("self.allegro_hand_dof_pos:", self.allegro_hand_dof_pos.shape)
        print("self.allegro_hand_dof_vel:", self.allegro_hand_dof_vel.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        assert self.num_dofs == self.allegro_hand_dof_state.shape[1] # this assertion is false if we add new objects with dof
        print("self.num_dofs:", self.num_dofs)
        
        """ This part is for the allegro hand dof targets, used when applying actions (pre_physics_step) & resetting the environment """
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        """ This part is for the rigid body states, ONLY used when computing the observation space """
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13) # [num_envs, num_bodies, 13]
        self.num_bodies = self.rigid_body_states.shape[1]
        print("self.rigid_body_states:", self.rigid_body_states.shape)
        """
        in this case, num_bodies = 1 table, 5 for box, x plates, 23 allegro rigid bodies = 29 + x
        13 contains: 
            3 floats for position
            4 floats for quaternion
            3 floats for linear velocity
            3 floats for angular velocity
        """

        """ This part is for the root state, mostly used when resetting the environment for each actor """
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        print("self.root_state_tensor:", self.root_state_tensor.shape) # [num_envs * num_actors, 13]

        """"""
        self.net_contact_state = gymtorch.wrap_tensor(net_contact_tensor).view(self.num_envs, -1, 3) # [num_envs, num_bodies, 3]
        print("self.net_contact_state:", self.net_contact_state.shape)

        """ This part is for the contact force tensor, used when computing the reward """
        self.sensor_state = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, -1, 6) # [num_envs, num_sensors, 6]
        print("self.sensor_state:", self.sensor_state.shape)
        assert len(self.contact_sensor_names) == self.sensor_state.shape[1]

        """ This part is for the jacobian tensor, used when computing the action (IK) """
        self.jacobian_state = gymtorch.wrap_tensor(jacobian_tensor) # [num_envs, num_links-1, 6, num_dofs]

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.successes_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.successes_indicator = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.total_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.total_eps = torch.ones(self.num_envs, dtype=torch.float, device=self.device) * -1
        self.successes_last_100 = torch.zeros((self.num_envs, 100), dtype=torch.float, device=self.device)
        self.current_indices = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        # for observation stacking
        self.obs_buf_stack_frames = []
        self.state_buf_stack_frames = []
        for _ in range(self.obs.stack_frames):
            self.obs_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs]))
            self.state_buf_stack_frames.append(torch.zeros_like(self.states_buf[:, 0:self.one_frame_num_states]))

        # self.extras["plate_pos_noise"] = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        # self.extras["action_scale"] = torch.full((self.num_envs,), \
        #     self.curriculum.decrease_action_init, dtype=torch.float, device=self.device)
        
        self.penalty_system = ProgressivePenaltySystem(self.device, self.num_envs, self.plate_cfg.num_plates,
                                                        height_threshold=0.01, initial_penalty=0, penalty_growth_rate=0.01)
        
        self.contact_thresh = 0.01

        # domain randomization additional setup
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        plate_rand = self.randomization_params["actor_params"]["plate"]
        self.randomization_params["actor_params"].pop("plate")
        for j in range(self.plate_cfg.num_plates):
            self.randomization_params["actor_params"][f"plate_{j}"] = plate_rand

        # PD control parameters
        # p_gain_tensor = torch.tensor(self.allegro_config["armDofProperties"]["pGain"]
        #     + self.allegro_config["allegroDofProperties"]["pGain"], device=self.device, dtype=torch.float)
        # d_gain_tensor = torch.tensor(self.allegro_config["armDofProperties"]["dGain"]
        #     + self.allegro_config["allegroDofProperties"]["dGain"], device=self.device, dtype=torch.float)

        # self.p_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs), device=self.device, dtype=torch.float) * p_gain_tensor
        # self.d_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs), device=self.device, dtype=torch.float) * d_gain_tensor

        # self.pd_previous_dof_pos = torch.zeros((self.num_envs, self.num_allegro_hand_dofs), device=self.device, dtype=torch.float)
        # self.pd_dof_pos = torch.zeros((self.num_envs, self.num_allegro_hand_dofs), device=self.device, dtype=torch.float)

        # update plate state
        self.plate_cfg.update_state(self.root_state_tensor, self.box_cfg)

        target_top_edge = self.plate_cfg.edges[torch.arange(self.num_envs), self.plate_cfg.target_indices, 0, :3]
        self.target_top_edge_offset = to_torch(self.allegro_config["targetTopEdgeOffset"], dtype=torch.float, device=self.device)
        self.target_init_top_edge = target_top_edge + self.target_top_edge_offset
        self.allegro_ee_noise = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _load_allegro_hand(self):
        allegro_hand_pose = gymapi.Transform()
        x, y, z = self.allegro_config["basePosition"]
        allegro_hand_pose.p = gymapi.Vec3(x, y, z)
        # allegro_hand_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # asset_options.override_com = True
        # asset_options.override_inertia = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 3000000
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        asset_root = self.allegro_config["assetRoot"]
        allegro_hand_asset_file = self.allegro_config["assetFileName"]
        print("Loading asset '%s' from '%s'" % (allegro_hand_asset_file, asset_root))
        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)

        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)

        assert self.num_total_dof == self.num_allegro_hand_dofs

        # default dof for allegro hand
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)

        self.hand_orient = self.allegro_config["handOrientation"]
        if self.hand_orient == "horizontal":
            self.now_euler_angle = to_torch([0.0, 3.14, 1.57], dtype=torch.float, device=self.device)
            self.allegro_ee_offset = to_torch([0.18, 0.01, -0.24], dtype=torch.float, device=self.device)
            self.allegro_hand_default_dof_pos[:self.num_arm_dof] = \
                to_torch([0, -0.70, 0, -2.68, 0, 2.00, 1.48], dtype=torch.float, device=self.device)
        elif self.hand_orient == "vertical":
            self.now_euler_angle = to_torch([0.0, 3.14, 3.14], dtype=torch.float, device=self.device)
            self.allegro_ee_offset = to_torch([0.01, 0.18, -0.24], dtype=torch.float, device=self.device)
            self.allegro_hand_default_dof_pos[:self.num_arm_dof] = \
                to_torch([0.335, -0.02, -0.66, -2.04, -0.01, 2.02, -0.32], dtype=torch.float, device=self.device)
        else:
            raise ValueError("Invalid hand orientation")
        self.allegro_hand_default_dof_pos[self.num_arm_dof:] = to_torch( # hand
            self.allegro_config["allegroDefaultDof"],
            dtype=torch.float, device=self.device)

        self.actuated_dof_indices = [i for i in range(16)]

        # set allegro_hand dof properties
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            
            # allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            # if i < self.num_arm_dof:
            #     allegro_hand_dof_props['velocity'][i] = 1.0
            # else:
            #     allegro_hand_dof_props['velocity'][i] = 3.14
            # allegro_hand_dof_props['effort'][i] = 20.0

            # allegro_hand_dof_props['stiffness'][i] = 0
            # allegro_hand_dof_props['damping'][i] = 0

            # allegro_hand_dof_props['friction'][i] = 0.1
            # allegro_hand_dof_props['armature'][i] = 0.1
            
            allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < self.num_arm_dof:
                # allegro_hand_dof_props['friction'][i] = 0.01
                # allegro_hand_dof_props['armature'][i] = 0.001
                allegro_hand_dof_props['stiffness'][i] = 400 # self.allegro_config["armDofProperties"]["stiffness"][i]
                # if "effort" in self.allegro_config["armDofProperties"]:
                allegro_hand_dof_props['effort'][i] = 200 # self.allegro_config["armDofProperties"]["effort"]
                # if "damping" in self.allegro_config["armDofProperties"]:
                allegro_hand_dof_props['damping'][i] = 80 # self.allegro_config["armDofProperties"]["damping"][i]
            else:
                allegro_hand_dof_props['velocity'][i] = 10.0 # self.allegro_config["allegroDofProperties"]["velocity"]
                allegro_hand_dof_props['effort'][i] = 0.7 # self.allegro_config["allegroDofProperties"]["effort"]
                allegro_hand_dof_props['stiffness'][i] = 20 # self.allegro_config["allegroDofProperties"]["stiffness"][i-self.num_arm_dof]
                allegro_hand_dof_props['damping'][i] = 1 # self.allegro_config["allegroDofProperties"]["damping"][i-self.num_arm_dof]

        # add sensors
        self.sensor_indices = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self.contact_sensor_names]
        sensor_pose = gymapi.Transform()
        for idx in self.sensor_indices:
            self.gym.create_asset_force_sensor(allegro_hand_asset, idx, sensor_pose)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)

        return allegro_hand_pose, allegro_hand_asset, allegro_hand_dof_props

    def _create_envs(self, num_envs, spacing, num_per_row):
        # load and create assets (table / plate / allegro hand)
        table_pose, table_asset = self.table_cfg.load_asset(self.gym, self.sim) # table_pose is initialized

        box_poses, box_assets = self.box_cfg.load_asset(self.gym, self.sim, self.device,
                                                        self.table_cfg, self.plate_cfg) # box_poses is initialized

        plate_pose = gymapi.Transform()
        plate_asset = self.plate_cfg.load_asset(self.gym, self.sim)

        allegro_hand_pose, allegro_hand_asset, allegro_hand_dof_props = self._load_allegro_hand() # allegro_hand_pose is initialized
        
        box_thin = self.box_cfg.thickness
        plate_spacing = self.plate_cfg.spacing
        plate_initial_height = self.plate_cfg.initial_height
        
        # set up the env grid
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # store some common handles for later use
        self.envs = []
        self.allegro_handles = []
        plates_handles = [[] for _ in range(num_envs)]
        
        # store some common indices for later use
        box_indices = []
        plates_indices = []
        self.hand_indices = []

        # store some common poses for later use
        box_start_state = []
        plates_start_state = []

        for i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)

            # add table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            
            plates_width = self.plate_cfg.num_plates * self.plate_cfg.thickness + (self.plate_cfg.num_plates - 1) * plate_spacing
            plates_center_x = table_pose.p.x
            plates_left_x = plates_center_x - 0.5 * plates_width
            plates_right_x = plates_center_x + 0.5 * plates_width
            plates_x_counter = plates_left_x
            
            # add box
            for j in range(len(box_assets)):
                box_actor = self.gym.create_actor(env_ptr, box_assets[j], box_poses[j], "box_" + str(j), i, 0)
                self.gym.set_rigid_body_color(env_ptr, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))
                
                box_idx = self.gym.get_actor_index(env_ptr, box_actor, gymapi.DOMAIN_SIM)
                box_indices.append(box_idx)

                box_start_state.append([box_poses[j].p.x, box_poses[j].p.y, box_poses[j].p.z,
                                             box_poses[j].r.x, box_poses[j].r.y, box_poses[j].r.z, box_poses[j].r.w,
                                                0, 0, 0, 0, 0, 0])
                
                box_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, box_actor)
                for box_shape_prop in box_shape_props:
                    box_shape_prop.friction = self.box_cfg.friction
                    box_shape_prop.restitution = self.box_cfg.restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, box_actor, box_shape_props)

            for j in range(self.plate_cfg.num_plates):
                x = plates_x_counter + 0.5 * self.plate_cfg.thickness
                plates_x_counter += self.plate_cfg.thickness + plate_spacing
                y = table_pose.p.y
                z = self.table_cfg.size[2] + box_thin + 0.5 * self.plate_cfg.size
                rotated_pos, rotated_quat = rotate_around_z_axis_with_offset(
                    to_torch([x, y, z + plate_initial_height], device=self.device),
                    to_torch([0.0, 0.0, 0.0, 1.0], device=self.device),
                    self.box_cfg.box_deg, self.box_cfg.original_box_center
                )
                plate_pose.p = gymapi.Vec3(rotated_pos[0], rotated_pos[1], rotated_pos[2])
                plate_pose.r = gymapi.Quat(rotated_quat[0], rotated_quat[1], rotated_quat[2], rotated_quat[3])

                # add plate
                plate_actor = self.gym.create_actor(env_ptr, plate_asset, plate_pose, "plate_" + str(j), i, 0)
                self.gym.set_rigid_body_color(env_ptr, plate_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, 
                            self.plate_cfg.precomputed_colors[j] if j != self.plate_cfg.original_target_idx 
                            else self.plate_cfg.target_color)

                plate_idx = self.gym.get_actor_index(env_ptr, plate_actor, gymapi.DOMAIN_SIM)
                plates_indices.append(plate_idx)

                plates_start_state.append([plate_pose.p.x, plate_pose.p.y, plate_pose.p.z,
                                           plate_pose.r.x, plate_pose.r.y, plate_pose.r.z, plate_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
                
                plate_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, plate_actor)
                for plate_shape_prop in plate_shape_props:
                    plate_shape_prop.friction = self.plate_cfg.friction
                    plate_shape_prop.restitution = self.plate_cfg.restitution
                self.gym.set_actor_rigid_shape_properties(env_ptr, plate_actor, plate_shape_props)

                plates_handles[i].append(plate_actor)
            
            assert plates_x_counter - plate_spacing - plates_right_x < 10e-5

            # add allegro hand
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, allegro_hand_pose, "hand", i, 1)
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, allegro_hand_actor)
            for hand_shape_prop in hand_shape_props:
                hand_shape_prop.friction = self.allegro_config["allegroFriction"]
                hand_shape_prop.restitution = self.allegro_config["allegroRestitution"]
            self.gym.set_actor_rigid_shape_properties(env_ptr, allegro_hand_actor, hand_shape_props)

            self.allegro_handles.append(allegro_hand_actor)

        self.finger_tip_indices = [self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_actor, name) for name in self.fingertip_names]
        self.arm_dof_indices = [self.gym.find_actor_dof_handle(env_ptr, allegro_hand_actor, name) for name in self.arm_movable_dof_names]
        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_actor, "panda_link7")

        # convert to torch tensors
        self.finger_tip_indices = to_torch(self.finger_tip_indices, dtype=torch.long, device=self.device)
        self.arm_dof_indices = to_torch(self.arm_dof_indices, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)

        self.box_cfg.init_tensors(self.device, self.num_envs, box_indices, box_start_state)
        self.plate_cfg.init_tensors(self.device, self.num_envs, plates_indices, plates_start_state, plates_handles)

    def refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.plate_cfg.update_state(self.root_state_tensor, self.box_cfg)

    def update_controller(self):
        """
        Update the PD controller for the allegro hand
        """
        self.pd_previous_dof_pos[:, :self.num_allegro_hand_dofs] = self.allegro_hand_dof_pos.clone()

        self.refresh_state()

        self.pd_dof_pos[:, :self.num_allegro_hand_dofs] = self.allegro_hand_dof_pos.clone()
        dof_vel = (self.pd_dof_pos - self.pd_previous_dof_pos) / self.dt
        self.dof_vel_finite_diff = dof_vel.clone()
        torques = self.p_gain * (self.cur_targets - self.pd_dof_pos) - self.d_gain * dof_vel
        self.torques = torques.clone()
        self.torques = torch.clip(self.torques, -2000.0, 2000.0)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        # print(dof_vel[0])
        # print(self.torques[0])
        # breakpoint()

        # self.pd_previous_dof_pos[:, :self.num_allegro_hand_dofs] = self.pd_dof_pos.clone()

    def reset_target_pose(self, env_ids, apply_reset=False):
        # Curriculum learning
        lift_eps_len = self.max_episode_length - self.act.phase3_transition_time
        success_env_ids = env_ids[self.successes[env_ids] > lift_eps_len * self.curriculum.success_threshold]
        num_succ_envs = len(success_env_ids)
        if num_succ_envs > 0:
            # if self.curriculum.increase_pos_noise:
            #     # increase the plate position noise
            #     self.extras["plate_pos_noise"][success_env_ids] += self.curriculum.increase_pos_noise_step
            #     self.extras["plate_pos_noise"][success_env_ids] = torch.clamp(
            #         self.extras["plate_pos_noise"][success_env_ids], 
            #         0.0, self.curriculum.max_pos_noise)

            #     rand_pos = generate_exact_infinity_norm_noise(num_succ_envs, self.extras["plate_pos_noise"][success_env_ids], self.device)
            #     assert rand_pos.shape == (num_succ_envs, 2)
            #     self.box_cfg.randomize_pos(success_env_ids, rand_pos)
            #     self.plate_cfg.randomize_pos(success_env_ids, rand_pos)

            if self.curriculum.increase_mass:
                updated_mass = self.plate_cfg.increase_mass(self.envs, success_env_ids, 
                                                            increase_scale=self.curriculum.increase_mass_scale, 
                                                            max_mass=self.curriculum.max_mass)
                self.extras["plate_mass"] = updated_mass
            
            if self.curriculum.decrease_spacing:
                updated_spacing = self.plate_cfg.decrease_spacing(success_env_ids, self.box_cfg, 
                                                                  decrease_scale=self.curriculum.decrease_spacing_scale)
                self.extras["plate_spacing"] = updated_spacing

            # if self.curriculum.decrease_action:
            #     self.extras["action_scale"][success_env_ids] *= self.curriculum.decrease_action_scale
            #     self.extras["action_scale"][success_env_ids] = torch.clamp(
            #         self.extras["action_scale"][success_env_ids], 
            #         self.curriculum.decrease_action_min, None)

        # Randomization
        if self.rand_cfg.reset_center_pos_noise[0]:
            rand_pos = torch_rand_float(self.rand_cfg.reset_center_pos_noise[1], # lower bound
                                        self.rand_cfg.reset_center_pos_noise[2], # upper bound
                                        (len(env_ids), 2), device=self.device)
            self.box_cfg.randomize_pos(env_ids, rand_pos)
            self.plate_cfg.randomize_pos(env_ids, rand_pos)

        if self.rand_cfg.reset_spacing_noise[0]:
            rand_spacing = torch_rand_float(self.rand_cfg.reset_spacing_noise[1], # lower bound
                                            self.rand_cfg.reset_spacing_noise[2], # upper bound
                                            (len(env_ids), 1), device=self.device)
            self.plate_cfg.randomize_spacing(env_ids, rand_spacing, self.box_cfg)

        if self.rand_cfg.reset_orit_noise[0]:
            rand_pitch = torch_rand_float(self.rand_cfg.reset_orit_noise[1], # lower bound
                                        self.rand_cfg.reset_orit_noise[2], # upper bound
                                        (len(env_ids), 1), device=self.device)
            self.plate_cfg.randomize_pitch(env_ids, rand_pitch)

        # self.plate_cfg.perturb_pose(env_ids)

        # Reset the state
        self.box_cfg.reset_state(env_ids, self.root_state_tensor)
        self.plate_cfg.reset_state(env_ids, self.root_state_tensor)
        self.plate_cfg.update_state(self.root_state_tensor, self.box_cfg)

        # randomize the target index
        if self.rand_cfg.reset_target_plate:
            self.plate_cfg.randomize_target_idx(self.envs, env_ids)

        if apply_reset:
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.allegro_hand_dof_pos[env_ids] = self.allegro_hand_default_dof_pos
        self.allegro_hand_dof_vel[env_ids] = 0

        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(hand_indices))

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        self.reset_target_pose(env_ids, apply_reset=False)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

        self.refresh_state()
        target_top_edge = self.plate_cfg.edges[torch.arange(self.num_envs), self.plate_cfg.target_indices, 0, :3]
        self.target_init_top_edge = target_top_edge + self.target_top_edge_offset
        if self.rand_cfg.reset_ee_pos_noise[0]:
            self.allegro_ee_noise[env_ids] = torch_rand_float(
                self.rand_cfg.reset_ee_pos_noise[1], 
                self.rand_cfg.reset_ee_pos_noise[2], 
                (len(env_ids), 3), device=self.device)

        # self.post_reset(env_ids, hand_indices)
        self.calculate_ik_for_reset(env_ids, hand_indices)

        self.total_successes[env_ids] += self.successes_indicator[env_ids]
        self.total_eps[env_ids] += 1

        self.current_indices[env_ids] = (self.current_indices[env_ids] + 1) % 100
        self.successes_last_100[env_ids, self.current_indices[env_ids]] = self.successes_indicator[env_ids].float()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.successes_counter[env_ids] = 0
        self.successes_indicator[env_ids] = 0

        if self.eval_cfg.stop_eps >= 0 and torch.all(self.total_eps >= self.eval_cfg.stop_eps):
            success_rate_last_100_eps = torch.where(self.total_eps == 0, 0.0, torch.sum(self.successes_last_100, dim=1) / torch.clamp(self.total_eps, max=100)) # [num_envs]
            ave_success_rate = torch.mean(success_rate_last_100_eps)
            with open("sample/eval_result.txt", "a") as f:
                f.write(f"Average success rate over last 100 episodes: {ave_success_rate.item()}\n")
            exit()

    def calculate_ik_for_reset(self, env_ids, hand_indices):
        self.render()
        self.gym.simulate(self.sim)
        
        self.refresh_state()
        
        target_top_edge = self.plate_cfg.edges[torch.arange(self.num_envs), self.plate_cfg.target_indices, 0, :3]
        self.target_init_top_edge = target_top_edge + self.target_top_edge_offset
        
        # Precompute IK targets
        self.allegro_ee_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3] + self.allegro_ee_offset + self.allegro_ee_noise
        pos_err = self.target_init_top_edge - self.allegro_ee_pos  # [num_envs, 3]
        
        target_rot = quat_from_euler_xyz(self.now_euler_angle[0], self.now_euler_angle[1], self.now_euler_angle[2]).repeat((self.num_envs, 1))
        rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone()) * 5
        
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(self.jacobian_state[:, self.hand_base_rigid_body_index-1, :, :7], self.device, dpose, self.num_envs)
        targets = self.allegro_hand_dof_pos[:, :self.num_arm_dof] + delta[:, :7]  # [num_envs, 7]
        
        # Set the computed joint positions directly
        self.allegro_hand_dof_pos[env_ids, :self.num_arm_dof] = targets[env_ids, :]
        self.allegro_hand_dof_vel[env_ids] = 0
        
        self.cur_targets[env_ids, :self.num_arm_dof] = targets[env_ids, :]
        self.cur_targets[env_ids, self.num_arm_dof:self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos[self.num_arm_dof:self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos[env_ids, self.num_arm_dof:self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos[self.num_arm_dof:self.num_allegro_hand_dofs]
        
        self.prev_targets[env_ids, :] = self.cur_targets[env_ids, :]
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        self.gym.simulate(self.sim)
        self.refresh_state()
    
    def post_reset(self, env_ids, hand_indices):
        for i in range(2):
            self.render()
            self.gym.simulate(self.sim)

        self.refresh_state()

        target_top_edge = self.plate_cfg.edges[torch.arange(self.num_envs), self.plate_cfg.target_indices, 0, :3]
        self.target_init_top_edge = target_top_edge + self.target_top_edge_offset

        self.allegro_hand_dof_pos[env_ids] = self.allegro_hand_default_dof_pos
        self.allegro_hand_dof_vel[env_ids] = 0

        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        self.gym.simulate(self.sim)

        # before grasping, approach the target plate
        for i in range(50):
            self.refresh_state()

            # move the hand to the target position
            self.allegro_ee_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3] \
                + self.allegro_ee_offset + self.allegro_ee_noise
            pos_err = self.target_init_top_edge - self.allegro_ee_pos # [num_envs, 3]

            target_rot = quat_from_euler_xyz(self.now_euler_angle[0], self.now_euler_angle[1], self.now_euler_angle[2]).repeat((self.num_envs, 1))
            rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone()) * 5

            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_state[:, self.hand_base_rigid_body_index-1, :, :7], self.device, dpose, self.num_envs)
            targets = self.allegro_hand_dof_pos[:, :self.num_arm_dof] + delta[:, :7] # [num_envs, 7]

            self.cur_targets[env_ids, :self.num_arm_dof] = targets[env_ids, :]
            self.allegro_hand_dof_pos[env_ids, :self.num_arm_dof] = targets[env_ids, :]
            self.allegro_hand_dof_vel[env_ids, :] = 0

            # fix the hand
            self.cur_targets[env_ids, self.num_arm_dof:self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos[self.num_arm_dof:self.num_allegro_hand_dofs]
            self.allegro_hand_dof_pos[env_ids, self.num_arm_dof:self.num_allegro_hand_dofs] = self.allegro_hand_default_dof_pos[self.num_arm_dof:self.num_allegro_hand_dofs]

            self.prev_targets[env_ids, :] = self.cur_targets[env_ids, :]

            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.prev_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

            self.render()
            self.gym.simulate(self.sim)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device) # [num_envs, num_dofs]

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.cur_targets[:, self.actuated_dof_indices + self.num_arm_dof] = \
            scale(self.actions[:, self.num_arm_dof:self.num_allegro_hand_dofs],
                  self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + self.num_arm_dof], 
                  self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + self.num_arm_dof])
        
        self.cur_targets[:, self.actuated_dof_indices + self.num_arm_dof] = \
            self.act_moving_average_hand * self.cur_targets[:, self.actuated_dof_indices + self.num_arm_dof] \
            + (1.0 - self.act_moving_average_hand) * self.prev_targets[:, self.actuated_dof_indices + self.num_arm_dof]
        
        ### use relative control for the arm
        # self.cur_targets[:, self.arm_dof_indices] = self.prev_targets[:, self.arm_dof_indices] \
        #     + self.actions[:, self.arm_dof_indices] * self.dt * self.extras["action_scale"].unsqueeze(-1)

        # self.cur_targets[:, [1, 3]] = self.prev_targets[:, [1, 3]] + self.actions[:, [1, 3]] * 0.1
        
        ### use absolute control for the arm
        # self.cur_targets[:, self.arm_dof_indices] = \
        #     scale(self.actions[:, self.arm_dof_indices], 
        #           self.allegro_hand_dof_lower_limits[self.arm_dof_indices], 
        #           self.allegro_hand_dof_upper_limits[self.arm_dof_indices])
        # self.cur_targets[:, self.arm_dof_indices] = \
        #     self.act_moving_average_arm * self.cur_targets[:, self.arm_dof_indices] \
        #     + (1.0 - self.act_moving_average_arm) * self.prev_targets[:, self.arm_dof_indices]
        
        # direction = self.cur_targets[:, self.arm_dof_indices] - self.prev_targets[:, self.arm_dof_indices]
        # direction = torch.clamp(direction, -0.05, 0.05)
        # direction *= 0.1
        # self.cur_targets[:, self.arm_dof_indices] = self.prev_targets[:, self.arm_dof_indices] + direction

        ### use IK for the hand
        if self.act.phase1_realtime_tracking:
            realtime_top_edge = self.plate_cfg.edges[torch.arange(self.num_envs), self.plate_cfg.target_indices, 0, :3]
            target_pos = realtime_top_edge + self.target_top_edge_offset
        else:
            target_pos = self.target_init_top_edge
        
        self.allegro_ee_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3] \
            + self.allegro_ee_offset + self.allegro_ee_noise
        pos_err = target_pos - self.allegro_ee_pos # [num_envs, 3]

        ## Option 1
        # pos_err[:] += self.actions[:, 0:3] * 0.1

        ## Option 2
        # pos_err[:, 0:2] += self.actions[:, 0:2] * 0.05
        # pos_err[:, 2] += self.actions[:, 2] * 0.2

        ## Option 3
        phase1_ids = self.progress_buf <= self.act.phase2_transition_time
        phase3_ids = self.progress_buf > self.act.phase3_transition_time
        phase2_ids = ~phase1_ids & ~phase3_ids
        phase12_ids = phase1_ids | phase2_ids
        phase23_ids = phase2_ids | phase3_ids

        # apply action perturbation in phase 1 & 2
        pos_err[phase12_ids] += self.actions[phase12_ids, :3] * self.act.phase1_perturbation_scale

        # Gradually move the end-effector during phase 2 & 3
        current_height = self.allegro_ee_pos[:, 2] # [num_envs]
        target_height = torch.zeros_like(current_height) # [num_envs]
        target_final_height = torch.zeros_like(current_height) # [num_envs]
        target_height[phase2_ids] = current_height[phase2_ids] + self.act.phase2_increment
        target_height[phase3_ids] = current_height[phase3_ids] + self.act.phase3_increment
        target_final_height[phase2_ids] = self.target_init_top_edge[phase2_ids, 2] + self.act.phase2_target_height
        target_final_height[phase3_ids] = self.target_init_top_edge[phase3_ids, 2] + self.act.phase3_target_height
        target_height = torch.min(target_height, target_final_height) # [num_envs]
        pos_err[phase23_ids, 2] = (target_height - current_height)[phase23_ids]

        target_rot = quat_from_euler_xyz(self.now_euler_angle[0], self.now_euler_angle[1], self.now_euler_angle[2]).repeat((self.num_envs, 1))
        rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone()) * 5

        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(self.jacobian_state[:, self.hand_base_rigid_body_index-1, :, :7], self.device, dpose, self.num_envs)
        self.cur_targets[:, :self.num_arm_dof] = self.allegro_hand_dof_pos[:, :self.num_arm_dof] + delta[:, :7]

        # fix the hand after grasping
        self.cur_targets[:, self.num_arm_dof:self.num_allegro_hand_dofs][phase3_ids] = self.prev_targets[:, self.num_arm_dof:self.num_allegro_hand_dofs][phase3_ids]

        # clamp the targets
        self.cur_targets[:, 0:self.num_allegro_hand_dofs] = \
            tensor_clamp(self.cur_targets[:, 0:self.num_allegro_hand_dofs],
                                            self.allegro_hand_dof_lower_limits[:],
                                            self.allegro_hand_dof_upper_limits[:])
        
        # print(self.cur_targets[0, self.arm_dof_indices] - self.prev_targets[0, self.arm_dof_indices])
        
        self.prev_targets[:, :] = self.cur_targets[:, :]

        # Print sample action / trajectory to tmp.py
        if self.eval_cfg.log_trajectory:
            with open(self.eval_cfg.log_trajectory_file, 'a') as file:
                # file.write(str(self.actions[0].tolist()) + ',\n')
                file.write(str(self.cur_targets[0].tolist()) + ',\n')

        if self.eval_cfg.run_trajectory:
            from sample.traj1 import get_traj
            self.cur_targets[0] = torch.tensor(get_traj()[self.progress_buf[0]])
            
            # from sample.sin_traj import get_sin_traj
            # self.cur_targets[0] = torch.tensor(
            #     get_sin_traj(
            #         self.allegro_hand_default_dof_pos.numpy(), 
            #         [6, 7, 11, 15, 19]
            #     )[self.progress_buf[0]]
            # )

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        # self.update_controller()

        # Apply random forces to the plates (optional)
        if self.force_scale > 0.0:
            # print("Apply force!")
            self.rb_forces *= torch.pow(to_torch(self.force_decay), self.dt / self.force_decay_interval)

            for j in range(self.plate_cfg.num_plates):
                obj_mass = to_torch(
                    [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, f"plate_{j}"))[0].mass for
                    env in self.envs], device=self.device)
                prob = 0.25 #self.random_force_prob_scalar
                force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
                plate_indices = self.plate_cfg.plates_handles[0][j]
                self.rb_forces[force_indices, plate_indices, :] = torch.randn(
                    self.rb_forces[force_indices, plate_indices, :].shape,
                    device=self.device) * obj_mass[force_indices, None] * self.force_scale
                self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def compute_observations(self):
        self.refresh_state()

        # get the current state of allegro hand + base
        self.allegro_hand_base_pos = self.root_state_tensor[self.hand_indices, 0:3] # [num_envs, 3]

        # get hand center state info
        self.allegro_hand_center_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3] # [num_envs, 3]
        self.allegro_hand_center_rot = self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7] # [num_envs, 4]
        self.allegro_hand_center_linvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 7:10] # [num_envs, 3]
        self.allegro_hand_center_angvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 10:13] # [num_envs, 3]

        # get allegro ee position
        self.allegro_ee_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3] \
            + self.allegro_ee_offset + self.allegro_ee_noise # [num_envs, 3]

        # get each finger tip pos / rot
        self.allegro_hand_ff_pos = self.rigid_body_states[:, self.finger_tip_indices[0], 0:3] # [num_envs, 3]
        self.allegro_hand_ff_rot = self.rigid_body_states[:, self.finger_tip_indices[0], 3:7] # [num_envs, 4]
        self.allegro_hand_ff_linvel = self.rigid_body_states[:, self.finger_tip_indices[0], 7:10] # [num_envs, 3]
        self.allegro_hand_ff_angvel = self.rigid_body_states[:, self.finger_tip_indices[0], 10:13] # [num_envs, 3]

        self.allegro_hand_mf_pos = self.rigid_body_states[:, self.finger_tip_indices[1], 0:3] # [num_envs, 3]
        self.allegro_hand_mf_rot = self.rigid_body_states[:, self.finger_tip_indices[1], 3:7] # [num_envs, 4]
        self.allegro_hand_mf_linvel = self.rigid_body_states[:, self.finger_tip_indices[1], 7:10] # [num_envs, 3]
        self.allegro_hand_mf_angvel = self.rigid_body_states[:, self.finger_tip_indices[1], 10:13] # [num_envs, 3]

        self.allegro_hand_rf_pos = self.rigid_body_states[:, self.finger_tip_indices[2], 0:3] # [num_envs, 3]
        self.allegro_hand_rf_rot = self.rigid_body_states[:, self.finger_tip_indices[2], 3:7] # [num_envs, 4]
        self.allegro_hand_rf_linvel = self.rigid_body_states[:, self.finger_tip_indices[2], 7:10] # [num_envs, 3]
        self.allegro_hand_rf_angvel = self.rigid_body_states[:, self.finger_tip_indices[2], 10:13] # [num_envs, 3]

        self.allegro_hand_th_pos = self.rigid_body_states[:, self.finger_tip_indices[3], 0:3] # [num_envs, 3]
        self.allegro_hand_th_rot = self.rigid_body_states[:, self.finger_tip_indices[3], 3:7] # [num_envs, 4]
        self.allegro_hand_th_linvel = self.rigid_body_states[:, self.finger_tip_indices[3], 7:10] # [num_envs, 3]
        self.allegro_hand_th_angvel = self.rigid_body_states[:, self.finger_tip_indices[3], 10:13] # [num_envs, 3]

        self.allegro_hand_ff_pos = self.allegro_hand_ff_pos + quat_apply(self.allegro_hand_ff_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.allegro_hand_mf_pos = self.allegro_hand_mf_pos + quat_apply(self.allegro_hand_mf_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.allegro_hand_rf_pos = self.allegro_hand_rf_pos + quat_apply(self.allegro_hand_rf_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.allegro_hand_th_pos = self.allegro_hand_th_pos + quat_apply(self.allegro_hand_th_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)

        # get all plates state info
        # self.plate_cfg.update_state(self.root_state_tensor, self.box_cfg)
        self.plate_cfg.calc_fingertip_plate( \
            self.allegro_hand_ff_pos, self.allegro_hand_mf_pos, \
            self.allegro_hand_rf_pos, self.allegro_hand_th_pos)  # [num_envs, num_plates]

        # contacts (4 contact points)
        # self.allegro_hand_contacts = self.net_contact_state[:, self.finger_tip_indices, :] # [num_envs, 4, 3]
        allegro_hand_contacts = self.net_contact_state[:, self.finger_tip_indices, :] # [num_envs, 4, 3]
        allegro_hand_contacts = torch.norm(allegro_hand_contacts, dim=-1) # [num_envs, 4]
        self.allegro_hand_contacts = torch.where(allegro_hand_contacts >= self.contact_thresh, 1.0, 0.0) # [num_envs, 4]

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs, 63), device=self.device)

        self.compute_sim2real_observation(rand_floats)

        if self.asymmetric_obs:
            self.compute_sim2real_asymmetric_obs(rand_floats)

        if self.eval_cfg.log_dof_state:
            with open(self.eval_cfg.log_dof_pos_file, 'a') as file:
                file.write(str(self.allegro_hand_dof_pos[0].tolist()) + ',\n')
            with open(self.eval_cfg.log_dof_vel_file, 'a') as file:
                file.write(str(self.allegro_hand_dof_vel[0].tolist()) + ',\n')
            # with open(self.eval_cfg.log_dof_torque_file, 'a') as file:
            #     file.write(str(self.torques[0].tolist()) + ',\n')
        
        if self.eval_cfg.log_obs_space:
            with open(self.eval_cfg.log_obs_space_file, 'a') as file:
                file.write(str(self.obs_buf[0].tolist()) + ',\n')

    def compute_sim2real_observation(self, rand_floats):
        obs_counter = 0

        # fill hand dof state
        if self.obs.use_allegro_dof_pos:
            total_dof_pos = unscale(self.allegro_hand_dof_pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
            allegro_dof_pos = total_dof_pos[:, self.num_arm_dof:self.num_allegro_hand_dofs]
            obs_size = self.num_allegro_hand_dofs - self.num_arm_dof

            self.obs_buf[:, obs_counter:obs_counter+obs_size] = allegro_dof_pos
            
            # self.obs_buf[:, obs_counter:obs_counter+self.num_arm_dof] = 0
            # self.obs_buf[:, obs_counter+1] = self.allegro_hand_dof_pos[:, 1]
            # self.obs_buf[:, obs_counter+3] = self.allegro_hand_dof_pos[:, 3]

            obs_counter += obs_size

        if self.obs.use_allegro_dof_vel:
            total_dof_vel = self.allegro_hand_dof_vel
            allegro_dof_vel = total_dof_vel[:, self.num_arm_dof:self.num_allegro_hand_dofs]
            obs_size = self.num_allegro_hand_dofs - self.num_arm_dof

            self.obs_buf[:, obs_counter:obs_counter+obs_size] = allegro_dof_vel * self.vel_obs_scale
            obs_counter += obs_size

        # fill hand rigid body state
        if self.obs.use_allegro_finger_tip_state:
            fingertips_state = torch.cat([
                self.allegro_hand_ff_pos - self.target_init_top_edge, self.allegro_hand_ff_rot, #self.allegro_hand_ff_linvel, self.allegro_hand_ff_angvel,
                self.allegro_hand_mf_pos - self.target_init_top_edge, self.allegro_hand_mf_rot, #self.allegro_hand_mf_linvel, self.allegro_hand_mf_angvel,
                self.allegro_hand_rf_pos - self.target_init_top_edge, self.allegro_hand_rf_rot, #self.allegro_hand_rf_linvel, self.allegro_hand_rf_angvel,
                self.allegro_hand_th_pos - self.target_init_top_edge, self.allegro_hand_th_rot, #self.allegro_hand_th_linvel, self.allegro_hand_th_angvel
            ], dim=-1)
            obs_size = 4 * (13 - 6)

            self.obs_buf[:, obs_counter:obs_counter+obs_size] = fingertips_state
            obs_counter += obs_size

        # fill plates state
        if self.obs.use_target_plate_corner_dist:
            ## use target plate corners
            # add the target plate corners
            top_corners = self.plate_cfg.corners[torch.arange(self.num_envs), self.plate_cfg.target_indices, 2:, :] \
                - self.target_init_top_edge.unsqueeze(1) # [num_envs, 2, 3]
            corners_state = top_corners.reshape(self.num_envs, -1) # [num_envs, 6]
            obs_size = 6
            
            self.obs_buf[:, obs_counter:obs_counter+obs_size] = corners_state
            obs_counter += obs_size

            # add the target plate nearby vecs
            top_target_nearby_vec = self.plate_cfg.target_nearby_vec[:, :, 2:, :] # [num_envs, 2, 2, 3]
            nearby_vec_state = top_target_nearby_vec.reshape(self.num_envs, -1)  # [num_envs, 12]
            obs_size = 12

            self.obs_buf[:, obs_counter:obs_counter+obs_size] = nearby_vec_state
            obs_counter += obs_size

        else:
            ## use target plate position & orientation
            target_pose = self.plate_cfg.pose[torch.arange(self.num_envs), self.plate_cfg.target_indices, :]  # [num_envs, 7]
            front_pose = self.plate_cfg.target_nearby_pose[:, 0, :]
            back_pose = self.plate_cfg.target_nearby_pose[:, 1, :]

            self.obs_buf[:, obs_counter:obs_counter + 21] = torch.cat([
                target_pose[:, :3] - self.target_init_top_edge, target_pose[:, 3:],  # target pose
                front_pose[:, :3] - self.target_init_top_edge, front_pose[:, 3:],  # front pose
                back_pose[:, :3] - self.target_init_top_edge, back_pose[:, 3:]  # back pose
            ], dim=-1)  # [num_envs, 3*7]

            obs_counter += 21

        if self.obs.use_ee_info:
            # add the target top edge and shift
            target_top_edge = self.plate_cfg.edges[torch.arange(self.num_envs), self.plate_cfg.target_indices, 0, :3] # [num_envs, 3]
            obs_size = 9 # 12

            self.obs_buf[:, obs_counter:obs_counter+obs_size] = torch.cat([
                # self.target_init_top_edge,
                self.allegro_ee_pos - self.target_init_top_edge,
                target_top_edge - self.target_init_top_edge,
                self.allegro_ee_pos - target_top_edge
            ], dim=-1)
            obs_counter += obs_size

        # fill actions
        if self.obs.use_last_action:
            perturbation_action = self.actions[:, :3]
            allegro_action = self.actions[:, self.num_arm_dof:self.num_allegro_hand_dofs]

            action_state = torch.cat([perturbation_action, allegro_action], dim=-1) # [num_envs, 10]
            obs_size = self.num_actions - self.num_arm_dof # + 3

            self.obs_buf[:, obs_counter:obs_counter+obs_size] = allegro_action
            obs_counter += obs_size

        # fill contacts
        if self.obs.use_tactile_sensor:
            binary_contacts = self.allegro_hand_contacts.reshape(self.num_envs, -1) # [num_envs, 4]
            obs_size = self.allegro_hand_contacts.shape[1] # num_contacts
            
            self.obs_buf[:, obs_counter:obs_counter+obs_size] = binary_contacts
            obs_counter += obs_size

        # fill progress percentage
        if self.obs.use_progress:
            binary_progress = torch.where(self.progress_buf < self.act.phase3_transition_time, 0.0, 1.0).unsqueeze(-1) # [num_envs, 1]
            obs_size = 1

            self.obs_buf[:, obs_counter:obs_counter+obs_size] = binary_progress
            obs_counter += obs_size

        # assure no extra observation space / overflow
        assert obs_counter == self.one_frame_num_obs

        # stack frames
        for i in range(len(self.obs_buf_stack_frames) - 1):
            self.obs_buf[:, (i+1) * self.one_frame_num_obs:(i+2) * self.one_frame_num_obs] = self.obs_buf_stack_frames[i]
            self.obs_buf_stack_frames[i] = self.obs_buf[:, (i) * self.one_frame_num_obs:(i+1) * self.one_frame_num_obs].clone()

        assert self.num_obs == self.one_frame_num_obs * len(self.obs_buf_stack_frames)

    def compute_sim2real_asymmetric_obs(self, rand_floats):
        state_counter = 0

        # fill hand dof state
        if self.state.use_allegro_dof_pos:
            self.states_buf[:, state_counter:state_counter+self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos, self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
            state_counter += self.num_allegro_hand_dofs

        if self.state.use_allegro_dof_vel:
            self.states_buf[:, state_counter:state_counter+self.num_allegro_hand_dofs] = self.allegro_hand_dof_vel * self.vel_obs_scale
            state_counter += self.num_allegro_hand_dofs

        # fill hand rigid body state
        if self.state.use_allegro_finger_tip_state:
            self.states_buf[:, state_counter:state_counter+4*13] = torch.cat([
                self.allegro_hand_ff_pos - self.target_init_top_edge, self.allegro_hand_ff_rot, self.allegro_hand_ff_linvel, self.allegro_hand_ff_angvel,
                self.allegro_hand_mf_pos - self.target_init_top_edge, self.allegro_hand_mf_rot, self.allegro_hand_mf_linvel, self.allegro_hand_mf_angvel,
                self.allegro_hand_rf_pos - self.target_init_top_edge, self.allegro_hand_rf_rot, self.allegro_hand_rf_linvel, self.allegro_hand_rf_angvel,
                self.allegro_hand_th_pos - self.target_init_top_edge, self.allegro_hand_th_rot, self.allegro_hand_th_linvel, self.allegro_hand_th_angvel
            ], dim=-1)
            state_counter += 4*13

        # fill plates state
        if self.state.use_target_plate_corner_dist:
            ## use target plate position & orientation
            poses = self.plate_cfg.pose[torch.arange(self.num_envs), self.plate_cfg.target_indices, :]
            pos = poses[:, :3] - self.target_init_top_edge
            quat = poses[:, 3:]
            linvels = self.plate_cfg.linvel[torch.arange(self.num_envs), self.plate_cfg.target_indices, :]
            angvels = self.plate_cfg.angvel[torch.arange(self.num_envs), self.plate_cfg.target_indices, :]
            concatenated = torch.cat([pos, quat, linvels, angvels], dim=-1)  # Shape: [num_envs, 13]
            self.states_buf[:, state_counter:state_counter+13] = concatenated
            state_counter += 13

            ## use target plate corners
            corners = self.plate_cfg.corners[torch.arange(self.num_envs), self.plate_cfg.target_indices, :, :] \
                - self.target_init_top_edge.unsqueeze(1) # [num_envs, 4, 3]
            self.states_buf[:, state_counter:state_counter+12] = corners.reshape(self.num_envs, -1) # [num_envs, 12]
            state_counter += 12

            self.states_buf[:, state_counter:state_counter+24] \
                = self.plate_cfg.target_nearby_vec.reshape(self.num_envs, -1)  # [num_envs, 24]
            state_counter += 24

        if self.state.use_ee_info:
            # add the target top edge and shift
            target_top_edge = self.plate_cfg.edges[torch.arange(self.num_envs), self.plate_cfg.target_indices, 0, :3] # [num_envs, 3]
            self.states_buf[:, state_counter:state_counter+12] = torch.cat([
                self.target_init_top_edge,
                self.allegro_ee_pos - self.target_init_top_edge,
                target_top_edge - self.target_init_top_edge,
                self.allegro_ee_pos - target_top_edge
            ], dim=-1)
            state_counter += 12

        # fill actions
        if self.state.use_last_action:
            self.states_buf[:, state_counter:state_counter+self.num_actions] = self.actions
            state_counter += self.num_actions

        # fill contacts
        if self.state.use_tactile_sensor:
            num_contacts = self.allegro_hand_contacts.shape[1]
            self.states_buf[:, state_counter:state_counter+num_contacts] = self.allegro_hand_contacts.reshape(self.num_envs, -1)
            state_counter += num_contacts

        # fill progress percentage
        if self.state.use_progress:
            self.states_buf[:, state_counter:state_counter+1] = self.progress_buf.unsqueeze(-1) / self.max_episode_length
            state_counter += 1

        # assure no extra observation space / overflow
        assert state_counter == self.one_frame_num_states

        # stack frames
        for i in range(len(self.state_buf_stack_frames) - 1):
            self.states_buf[:, (i+1) * self.one_frame_num_states:(i+2) * self.one_frame_num_states] = self.state_buf_stack_frames[i]
            self.state_buf_stack_frames[i] = self.states_buf[:, (i) * self.one_frame_num_states:(i+1) * self.one_frame_num_states].clone()

        assert self.num_states == self.one_frame_num_states * len(self.state_buf_stack_frames)

    def compute_reward(self, actions):
        box_thin = self.box_cfg.thickness
        base_pos = self.table_cfg.get_base_pose(self.device, box_thin).repeat(self.num_envs, 1)
        
        # Extract current, previous, and prev_prev allegro_dof_pos
        allegro_dof_num = self.num_allegro_hand_dofs - self.num_arm_dof # 16
        curr_allegro_dof_pos = self.obs_buf[:, :allegro_dof_num]
        prev_allegro_dof_pos = self.obs_buf[:, self.one_frame_num_obs:self.one_frame_num_obs + allegro_dof_num]
        prev_prev_allegro_dof_pos = self.obs_buf[:, self.one_frame_num_obs*2:self.one_frame_num_obs*2 + allegro_dof_num]

        # Calculate velocity and acceleration
        velocity = curr_allegro_dof_pos - prev_allegro_dof_pos
        acceleration = curr_allegro_dof_pos - 2 * prev_allegro_dof_pos + prev_prev_allegro_dof_pos

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], \
            self.successes[:], self.successes_counter[:], self.consecutive_successes[:], rw_status = \
        compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.successes_counter, self.consecutive_successes,
            self.rewards_1, self.rewards_2, self.rewards_3, self.max_episode_length, base_pos,
            self.plate_cfg.pose, self.plate_cfg.dims, self.plate_cfg.edges[:, :, 0], 
            self.plate_cfg.target_indices, self.plate_cfg.target_nearby_vec,
            self.allegro_hand_ff_pos, self.allegro_hand_mf_pos,
            self.allegro_hand_rf_pos, self.allegro_hand_th_pos, self.allegro_ee_pos,
            self.box_cfg.box_deg, self.plate_cfg.is_contact[torch.arange(self.num_envs), self.plate_cfg.target_indices, :],
            self.actions, self.prev_actions, velocity, acceleration, self.num_arm_dof, self.num_total_dof,
            self.penalty_system, self.av_factor, self.act.phase2_transition_time, self.act.phase3_transition_time
        )

        self.prev_actions[:, :] = self.actions[:, :]

        # log the reward status
        self.extras.update(rw_status)

        # log the success status
        self.extras['successes'] = self.successes
        self.extras['successes_counter'] = self.successes_counter
        self.extras['consecutive_successes'] = self.consecutive_successes
        self.extras['success_rate_last_100_eps'] = torch.where(self.total_eps == 0, 0.0, torch.sum(self.successes_last_100, dim=1) / torch.clamp(self.total_eps, max=100))
        self.extras['success_rate_all_eps'] = torch.where(self.total_eps == 0, 0.0, self.total_successes / self.total_eps)

        new_success_indicator = torch.where(self.successes_counter > self.eval_cfg.consecutive_time_steps, 
                                            torch.ones_like(self.successes_counter), 
                                            torch.zeros_like(self.successes_counter))
        self.successes_indicator = torch.max(self.successes_indicator, new_success_indicator)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug
        if self.viewer and self.debug.viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            if self.debug.box_corners:
                box_corners = self.box_cfg.get_corners()

            for i in range(self.num_envs):
                if self.debug.target_init_pos:
                    self.add_debug_lines(self.envs[i], self.target_init_top_edge[i], line_width=2)

                ### test hand center
                if self.debug.hand_center:
                    self.add_debug_lines(self.envs[i], self.allegro_hand_center_pos[i], self.allegro_hand_center_rot[i], line_width=2)

                if self.debug.allegro_ee_pos:
                    self.add_debug_lines(self.envs[i], self.allegro_ee_pos[i], self.allegro_hand_center_rot[i], line_width=2) # allegro ee pos
                    # self.add_debug_lines(self.envs[i], self.allegro_ee_pos[i] + to_torch([0, 0, 0.24 - 0.1]), self.allegro_hand_center_rot[i], line_width=2) # real franka ee pos (shifted)
                    # self.add_debug_lines(self.envs[i], self.allegro_ee_pos[i] + to_torch([0, 0, 0.24]), self.allegro_hand_center_rot[i], line_width=2) # sim franka ee pos (shifted)

                ### test hand tips
                if self.debug.hand_tips:
                    self.add_debug_lines(self.envs[i], self.allegro_hand_ff_pos[i], self.allegro_hand_ff_rot[i], line_width=2)
                    self.add_debug_lines(self.envs[i], self.allegro_hand_mf_pos[i], self.allegro_hand_mf_rot[i], line_width=2)
                    self.add_debug_lines(self.envs[i], self.allegro_hand_rf_pos[i], self.allegro_hand_rf_rot[i], line_width=2)
                    self.add_debug_lines(self.envs[i], self.allegro_hand_th_pos[i], self.allegro_hand_th_rot[i], line_width=2)

                ### test contact sensor
                if self.debug.contact_sensors:
                    for j, contact_idx in enumerate(self.finger_tip_indices):
                        if self.allegro_hand_contacts[i, j] > self.contact_thresh:
                            # !Bug: the color is not working
                            self.gym.set_rigid_body_color(self.envs[i], self.allegro_handles[i], contact_idx, 
                                                        gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 1.0, 0.0))
                            
                ### test is contact
                if self.debug.is_contact:
                    for j in range(self.plate_cfg.num_plates):
                        # !Bug: the color is not working
                        total_contact = self.plate_cfg.is_contact[i, j].sum()
                        if total_contact > 0:
                            print(f'contact {total_contact} times on plate {j}')
                        # for k in range(4):
                        #     if self.plate_cfg.is_contact[i, j, k]:
                        #         print('contact')
                        #         self.gym.set_rigid_body_color(self.envs[i], self.plate_cfg.plates_handles[i], j, 
                        #                                     gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 1.0, 0.0))
                        #     else:
                        #         self.gym.set_rigid_body_color(self.envs[i], self.plate_cfg.plates_handles[i], j, 
                        #                                     gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.0, 0.0))
                
                for j in range(self.plate_cfg.num_plates):
                    ### test plate center
                    if self.debug.plate_center:
                        self.add_debug_lines(self.envs[i], self.plate_cfg.pos[i, j], self.plate_cfg.rot[i, 0], line_width=2)

                    ### test plate corners & edges
                    for k in range(4):
                        if self.debug.plate_corners:
                            self.add_debug_lines(self.envs[i], self.plate_cfg.corners[i, j, k], line_width=2)
                        if self.debug.plate_edges:
                            self.add_debug_lines(self.envs[i], self.plate_cfg.edges[i, j, k], line_width=2)
                if self.debug.box_corners:
                    for j in range(8):
                        self.add_debug_lines(self.envs[i], box_corners[i, j], line_width=2)

                if self.debug.neighbor_pose:
                    self.add_debug_lines(self.envs[i], self.plate_cfg.target_nearby_pose[i, 0, :3], self.plate_cfg.target_nearby_pose[i, 0, 3:], line_width=2)
                    self.add_debug_lines(self.envs[i], self.plate_cfg.target_nearby_pose[i, 1, :3], self.plate_cfg.target_nearby_pose[i, 1, 3:], line_width=2)

    def add_debug_lines(self, env, pos, rot=None, line_width=1):
        if rot is None:
            rot = to_torch([1, 0, 0, 0], device=self.device)  # Identity quaternion
        
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, line_width, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])
        
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
class ProgressivePenaltySystem:
    def __init__(self, device: str, num_envs: int, num_plates: int, 
                 height_threshold: float, initial_penalty: float, penalty_growth_rate: float):
        self.device = device
        self.num_envs = num_envs
        self.num_plates = num_plates
        self.height_threshold = height_threshold
        self.initial_penalty = initial_penalty
        self.penalty_growth_rate = penalty_growth_rate
        self.counters = torch.zeros((num_envs, num_plates-1), 
                                               dtype=torch.int64, device=self.device)

    def update_counters_and_calculate_penalty(self, other_plates_heights: torch.Tensor):
        # plate_heights is expected to be a tensor of shape [num_envs, num_plates-1]
        wrong_plate_lifted = (other_plates_heights > self.height_threshold)
        self.counters[wrong_plate_lifted] += 1
        self.counters[~wrong_plate_lifted] = 0

        # Calculate penalty based on the counters
        penalties = self.initial_penalty + self.penalty_growth_rate * self.counters
        total_penalty = penalties.sum(dim=1)

        return total_penalty

@torch.jit.script
def is_between_plates(plate_left_pos, plate_right_pos, target_pos, \
                            y_threshold: float = 0.095, z_threshold: float = 0.095):
    x_condition = (target_pos[:, 0] > plate_left_pos[:, 0]) & (target_pos[:, 0] < plate_right_pos[:, 0])
    y_condition = (torch.abs(target_pos[:, 1] - plate_left_pos[:, 1]) < y_threshold) & \
                  (torch.abs(target_pos[:, 1] - plate_right_pos[:, 1]) < y_threshold)
    z_condition = (torch.abs(target_pos[:, 2] - plate_left_pos[:, 2]) < z_threshold) & \
                  (torch.abs(target_pos[:, 2] - plate_right_pos[:, 2]) < z_threshold)
    is_between = x_condition & y_condition & z_condition
    return is_between

@torch.jit.script
def calc_dist_along_angle(pos1, pos2, angle_deg: float):
    """
    Calculates the distance between two points along a specified direction in the XY plane.

    Args:
    - pos1 (torch.Tensor): The position of the first point, shape [num_envs, 3].
    - pos2 (torch.Tensor): The position of the second point, shape [num_envs, 3].
    - angle (float): The angle in degrees defining the direction.

    Returns:
    - torch.Tensor: The distance between the two points along the specified direction.
    """
    # Calculate the unit direction vector for the given angle
    angle_rad = torch.tensor(angle_deg * torch.pi / 180.0, dtype=torch.float, device=pos1.device)
    d_x = torch.cos(angle_rad)
    d_y = torch.sin(angle_rad)
    direction = torch.stack([d_x, d_y], dim=0)  # Shape: [2]

    # Project the position vectors onto the direction vector
    projection_pos1 = torch.matmul(pos1[:, :2], direction)
    projection_pos2 = torch.matmul(pos2[:, :2], direction)

    # Calculate the distance along the direction
    distance_along_direction = torch.abs(projection_pos1 - projection_pos2)

    return distance_along_direction

@torch.jit.script
def calc_weighted_dist_along_angle(pos1, pos2, angle_deg: float, weight: float):
    """
    Calculates the weighted distance between two points along a specified direction in the XY plane.

    Args:
    - pos1 (torch.Tensor): The position of the first point, shape [num_envs, 3].
    - pos2 (torch.Tensor): The position of the second point, shape [num_envs, 3].
    - angle (float): The angle in degrees defining the direction.
    - weight (float): The weights for the dimension

    Returns:
    - torch.Tensor: The weighted distance between the two points along the specified direction.
    """
    # Calculate the unit direction vector for the given angle
    angle_rad = torch.tensor(angle_deg * torch.pi / 180.0, dtype=torch.float, device=pos1.device)
    d_x = torch.cos(angle_rad)
    d_y = torch.sin(angle_rad)
    direction = torch.stack([d_x, d_y], dim=0)  # Shape: [2]

    diff_xy = pos2[:, :2] - pos1[:, :2] # Shape: [num_envs, 2]
    projection = (diff_xy * direction).sum(dim=-1) # Shape: [num_envs]

    weighted_projection = projection * weight
    perpendicular_component = torch.sqrt((diff_xy**2).sum(dim=-1) - projection**2)

    diff_z = pos2[:, 2] - pos1[:, 2]
    weighted_distance = torch.sqrt(weighted_projection**2 + perpendicular_component**2 + diff_z**2)

    return weighted_distance

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, successes_counter, consecutive_successes, 
    rw_scales_1: RewardComponents, rw_scales_2: RewardComponents, rw_scales_3: RewardComponents, max_episode_length: float, 
    base_pos, plate_pose, plate_dims, edge_top_pos, target_indices, target_nearby_vec, 
    allegro_hand_ff_pos, allegro_hand_mf_pos, allegro_hand_rf_pos, allegro_hand_th_pos, allegro_ee_pos,
    angle_deg: float, is_contacting_target,
    actions, prev_actions, dof_vel, dof_acc, num_arm_dof: int, num_total_dof: int,
    penalty_system: ProgressivePenaltySystem, av_factor, phase2_transition_time: int, phase3_transition_time: int):

    SUCCESS_THRESHOLD = 0.2

    device = base_pos.device
    num_envs = base_pos.size(0)
    num_plates = plate_pose.size(1)
    # nonzero_indices = is_target_idx.nonzero()
    # if len(nonzero_indices.shape) == 2 and nonzero_indices.shape[1] >= 2:
    #     # Assuming the second column contains the indices of interest
    #     target_plate_indices = nonzero_indices[:, 1]
    # else:
    #     # Handle unexpected shapes or raise an error
    #     raise ValueError("Unexpected shape for nonzero indices")

    reward_status = {}

    ### for target plate
    target_plate_pos = plate_pose[torch.arange(num_envs), target_indices, :3]  # [num_envs, 3]
    target_edge_top_pos = edge_top_pos[torch.arange(num_envs), target_indices]
    target_plate_height = target_plate_pos[:, 2] - plate_dims[2] * 0.5  # [num_envs]

    base_height = base_pos[:, 2]

    # calculate distance between the height of the target plate and the table
    target_height = target_plate_height - base_height  # [num_envs]
    target_dist = torch.norm(target_plate_pos - base_pos, p=2, dim=-1)  # [num_envs]

    # calculate accumulated distance between each finger tip and the plate
    xyz_weight = torch.tensor([1.0, 1.0, 1.0], device=device)
    norm_p = 2 # float('inf')
    d1 = torch.norm((target_plate_pos - allegro_hand_ff_pos) * xyz_weight, p=norm_p, dim=-1)
    d2 = torch.norm((target_plate_pos - allegro_hand_mf_pos) * xyz_weight, p=norm_p, dim=-1)
    d3 = torch.norm((target_plate_pos - allegro_hand_rf_pos) * xyz_weight, p=norm_p, dim=-1)
    d4 = torch.norm((target_plate_pos - allegro_hand_th_pos) * xyz_weight, p=norm_p, dim=-1)
    # d5 = torch.norm(target_plate_pos - allegro_ee_pos, p=2, dim=-1)
    finger_dist_sum = (d1 + d2 + d3 + d4) / 4 # [num_envs]
    
    # Compute Status
    is_dropped = target_height < -0.03
    is_fly_away = target_dist > 0.7
    is_time_out = progress_buf >= max_episode_length
    is_success = target_height > SUCCESS_THRESHOLD
    is_init_height = target_height > 0.01
    is_phase_1 = progress_buf <= phase2_transition_time
    is_phase_3 = progress_buf > phase3_transition_time
    is_phase_2 = ~is_phase_1 & ~is_phase_3

    init_height_bonus_scale = torch.where(is_phase_2, rw_scales_2.init_height_bonus, torch.where(is_phase_3, rw_scales_3.init_height_bonus, rw_scales_1.init_height_bonus))
    height_reward_scale = torch.where(is_phase_2, rw_scales_2.height_reward, torch.where(is_phase_3, rw_scales_3.height_reward, rw_scales_1.height_reward))
    success_bonus_scale = torch.where(is_phase_2, rw_scales_2.success_bonus, torch.where(is_phase_3, rw_scales_3.success_bonus, rw_scales_1.success_bonus))
    proximity_reward_scale = torch.where(is_phase_2, rw_scales_2.proximity_reward, torch.where(is_phase_3, rw_scales_3.proximity_reward, rw_scales_1.proximity_reward))
    grasp_reward_scale = torch.where(is_phase_2, rw_scales_2.grasp_reward, torch.where(is_phase_3, rw_scales_3.grasp_reward, rw_scales_1.grasp_reward))
    split_penalty_scale = torch.where(is_phase_2, rw_scales_2.split_penalty, torch.where(is_phase_3, rw_scales_3.split_penalty, rw_scales_1.split_penalty))
    split_reward_scale = torch.where(is_phase_2, rw_scales_2.split_reward, torch.where(is_phase_3, rw_scales_3.split_reward, rw_scales_1.split_reward))
    drop_penalty_scale = torch.where(is_phase_2, rw_scales_2.drop_penalty, torch.where(is_phase_3, rw_scales_3.drop_penalty, rw_scales_1.drop_penalty))
    action_penalty_scale = torch.where(is_phase_2, rw_scales_2.action_penalty, torch.where(is_phase_3, rw_scales_3.action_penalty, rw_scales_1.action_penalty))
    vel_penalty_scale = torch.where(is_phase_2, rw_scales_2.vel_penalty, torch.where(is_phase_3, rw_scales_3.vel_penalty, rw_scales_1.vel_penalty))
    acc_penalty_scale = torch.where(is_phase_2, rw_scales_2.acc_penalty, torch.where(is_phase_3, rw_scales_3.acc_penalty, rw_scales_1.acc_penalty))
    other_plate_penalty_scale = torch.where(is_phase_2, rw_scales_2.other_plate_penalty, torch.where(is_phase_3, rw_scales_3.other_plate_penalty, rw_scales_1.other_plate_penalty))
    
    # Compute Rewards
    height_reward = torch.where(
        target_height >= 0,
        height_reward_scale * torch.clamp(target_height, max=SUCCESS_THRESHOLD),
        0.1 * height_reward_scale * target_height
    ) + is_init_height * init_height_bonus_scale + is_success * success_bonus_scale
    
    proximity_reward = proximity_reward_scale \
        * torch.exp(-15 * torch.clamp(finger_dist_sum, 0.07, None))
    # proximity threshold -> adjusted based on the finger_dist_sum

    # grasp reward (thumb & 2+ fingers)
    # thumb_contact = is_contacting_target[:, 3] # [num_envs]
    # other_finger_contact = is_contacting_target[:, :3].sum(dim=1) # [num_envs]
    # grasp_reward = (thumb_contact + torch.clamp(other_finger_contact, max=2)) * grasp_reward_scale # [num_envs]

    # grasp reward (2+ fingers)
    all_finger_contact = is_contacting_target.sum(dim=1) # [num_envs]
    grasp_reward = torch.clamp(all_finger_contact, max=2) * grasp_reward_scale # [num_envs]

    # vecs = target_nearby_vec.reshape(num_envs, -1, 3) # [num_envs, 8, 3]
    top_target_nearby_vec = target_nearby_vec[:, :, 2:, :] # [num_envs, 2, 2, 3]
    vecs = top_target_nearby_vec.reshape(num_envs, -1, 3) # [num_envs, 4, 3]
    magnitudes = torch.norm(vecs, dim=2) # [num_envs, 4]
    min_magnitudes = torch.min(magnitudes, dim=1)[0] # [num_envs]
    split_penalty = torch.relu(0.04 - min_magnitudes) * split_penalty_scale

    split_reward = (min_magnitudes - 0.02) * split_reward_scale

    drop_penalty = is_dropped * drop_penalty_scale # [num_envs]

    action_penalty = action_penalty_scale * torch.sum(actions[:, 7:] ** 2, dim=-1)
        # 4 * action_penalty_scale * torch.sum(actions[:, [7, 15, 19]] ** 2, dim=-1)
        # 5 * action_penalty_scale * torch.sum(actions[:, :3] ** 2, dim=-1)

    vel_penalty = torch.sum(dof_vel[:, num_arm_dof:] ** 2, dim=-1) * vel_penalty_scale
    acc_penalty = torch.sum(dof_acc[:, num_arm_dof:] ** 2, dim=-1) * acc_penalty_scale
    
    # If other plates exist
    other_plates_penalty = torch.zeros((num_envs), device=device)
    hand_in_between_reward = torch.zeros((num_envs), device=device)

    if num_plates > 1:
        is_in_between = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)

        xyz_weight = torch.tensor([1.0, 1.0, 1.0], device=device)
        norm_p = 2 # float('inf')

        has_right = target_indices < num_plates - 1
        has_left = target_indices > 0   

        if has_right.any():
            right_indices = torch.where(has_right, target_indices + 1, target_indices)
            right_plate_pos = plate_pose[torch.arange(num_envs), right_indices, :3]
            is_in_between[:, 0] = ( is_between_plates(right_plate_pos, target_plate_pos, allegro_hand_ff_pos) | 
                                    is_between_plates(right_plate_pos, target_plate_pos, allegro_hand_mf_pos) | 
                                    is_between_plates(right_plate_pos, target_plate_pos, allegro_hand_rf_pos) | 
                                    is_between_plates(right_plate_pos, target_plate_pos, allegro_hand_th_pos) )

        if has_left.any():
            left_indices = torch.where(has_left, target_indices - 1, target_indices)
            left_plate_pos = plate_pose[torch.arange(num_envs), left_indices, :3]        
            is_in_between[:, 1] = ( is_between_plates(left_plate_pos, target_plate_pos, allegro_hand_ff_pos) | 
                                    is_between_plates(left_plate_pos, target_plate_pos, allegro_hand_mf_pos) | 
                                    is_between_plates(left_plate_pos, target_plate_pos, allegro_hand_rf_pos) | 
                                    is_between_plates(left_plate_pos, target_plate_pos, allegro_hand_th_pos) )
        
        all_indices = torch.arange(num_plates, device=device).expand(num_envs, -1) # [num_envs, num_plates]
        mask = all_indices != target_indices.unsqueeze(1) # [num_envs, num_plates]
        other_plates_pose = plate_pose[mask].view(num_envs, num_plates-1, -1) # [num_envs, num_plates-1, 7]
        other_plates_height = other_plates_pose[:, :, 2] - 0.5 * plate_dims[2].unsqueeze(-1) # [num_envs, num_plates-1]
        other_height = other_plates_height - base_height.unsqueeze(1) # [num_envs, num_plates-1]
        other_height = torch.max(other_height, torch.zeros_like(other_height)) # at least zero
        assert other_height.size(1) == num_plates - 1

        is_wrong_plate = torch.any(other_height > 0.05, dim=1)

        # other_max_height = torch.max(other_height, dim=1).values  # [num_envs]
        # other_max_height = torch.max(other_max_height, torch.zeros_like(other_max_height)) # at least zero
        # height_reward = is_split * ( # update height reward
        #     rw_scales.height_rewrad
        #     * torch.clamp(target_height - other_max_height, max=SUCCESS_THRESHOLD)
        #     + is_success * rw_scales.success_bonus )
                    
        other_plates_penalty = is_wrong_plate * other_plate_penalty_scale \
            * torch.sum(other_height, dim=1) # [num_envs]
        # other_plates_penalty = -penalty_system.update_counters_and_calculate_penalty(other_dist)

        hand_in_between_reward = is_in_between.sum(dim=1) * 0 # * rw_scales.hand_in_between_reward

    # calculate total rewards
    rewards = height_reward + proximity_reward + grasp_reward + split_reward \
        + drop_penalty + action_penalty + split_penalty + other_plates_penalty \
        + hand_in_between_reward + vel_penalty + acc_penalty
    
    # rewards = torch.where(is_phase_2, rewards * 2, rewards) 

    reward_status['height_reward'] = height_reward
    reward_status['proximity_reward'] = proximity_reward
    reward_status['split_penalty'] = split_penalty
    reward_status['grasp_reward'] = grasp_reward
    # reward_status['split_reward'] = split_reward
    # reward_status['drop_penalty'] = drop_penalty
    reward_status['action_penalty'] = action_penalty
    reward_status['vel_penalty'] = vel_penalty
    reward_status['acc_penalty'] = acc_penalty
    reward_status['other_plates_penalty'] = other_plates_penalty

    # Goal Reset
    goal_resets = torch.where(is_success, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes += goal_resets
    succ_counter = torch.where(is_success, successes_counter + 1, torch.zeros_like(successes_counter))
    is_consecutive_success = succ_counter > 60 # [num_envs]

    # Check reset conditions
    resets = torch.where(is_dropped, torch.ones_like(reset_buf), reset_buf) # reset if dropped
    resets = torch.where(is_fly_away, torch.ones_like(resets), resets) # reset if fly away
    # resets = torch.where(is_wrong_plate, torch.ones_like(resets), resets)
    # resets = torch.where(is_success, torch.ones_like(resets), resets)
    # resets = torch.where(is_consecutive_success, torch.ones_like(resets), resets) # reset if 60 continuous success
    resets = torch.where(is_time_out, torch.ones_like(resets), resets) # reset if time out

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return rewards, resets, reset_goal_buf, progress_buf, successes, succ_counter, cons_successes, reward_status

@torch.jit.script
def rotate_around_z_axis_with_offset(pos, quat, angle_deg: float, offset):
    """
    Rotate object around the Z-axis for multiple environments.
    pos: Tensor of shape [num_envs, 3], positions.
    quat: Tensor of shape [num_envs, 4], orientations as quaternions.
    angle_deg: float, rotation angle in degrees.
    offset: Tensor of shape [num_envs, 3] or [3], the offset from origin to rotate around.
    """
    angle_rad = torch.tensor(angle_deg * torch.pi / 180.0, dtype=torch.float, device=pos.device)
    axis = torch.tensor([0, 0, 1], dtype=torch.float, device=pos.device)

    # Create rotation quaternion from angle-axis representation
    rotation_quat = quat_from_angle_axis(angle_rad, axis)

    # Translate components to rotate around the point (0.5, 0, 0) instead of the origin
    translated_pos = pos - offset

    # Rotate translated positions using quaternion application
    rotated_translated_pos = quat_apply(rotation_quat, translated_pos)

    # Translate positions back
    rotated_pos = rotated_translated_pos + offset

    # Rotate orientations by quaternion multiplication
    rotated_quat = quat_mul(rotation_quat, quat)

    return rotated_pos, rotated_quat

def generate_exact_infinity_norm_noise(num_envs, max_norm, device):
    # num_envs: int, number of environments
    # max_norm: torch.Tensor[num_envs], maximum value of the noise for each environment
    max_component_is_x = torch.rand(num_envs, device=device) < 0.5
    signs = torch.randint(0, 2, (num_envs,), device=device) * 2 - 1
    max_values = max_norm * signs.float()  # Apply sign, resulting in +max_norm or -max_norm

    # Generating other values within [-max_norm, max_norm] for each environment
    other_values = (torch.rand(num_envs, device=device) * 2 - 1) * max_norm

    noise_x = torch.where(max_component_is_x, max_values, other_values)
    noise_y = torch.where(max_component_is_x, other_values, max_values)

    # Returning a stack of [noise_x, noise_y] pairs for each environment
    return torch.stack((noise_x, noise_y), dim=-1)  # Shape: [num_envs, 2]

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):
	# Set controller parameters
	# IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u