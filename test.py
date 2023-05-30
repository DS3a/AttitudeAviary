#!/usr/bin/env python3

import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary

from AttitudeAviary import AttitudeAviary

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 15
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

if __name__ == "__main__":
    env = AttitudeAviary(
        drone_model=DroneModel.CF2X,      # See DroneModel Enum class for other quadcopter models
        # num_drones=1,                     # Number of drones
        # neighbourhood_radius=np.inf,      # Distance at which drones are considered neighbors, only used for multiple drones
        initial_xyzs=None,                # Initial XYZ positions of the drones
        initial_rpys=None,                # Initial roll, pitch, and yaw of the drones in radians
        physics=Physics.PYB,     # Choice of (PyBullet) physics implementation
        freq=240,                         # Stepping frequency of the simulation
        aggregate_phy_steps=1,            # Number of physics updates within each call to BaseAviary.step()
        gui=True,                         # Whether to display PyBullet's GUI, only use this for debbuging
        record=False,                     # Whether to save a .mp4 video (if gui=True) or .png frames (if gui=False) in gym-pybullet-drones/files/, see script /files/videos/ffmpeg_png2mp4.sh for encoding
        # obstacles=False,                  # Whether to add obstacles to the environment
        # user_debug_gui=True
        )

    obs = env.reset()
    x = 0
    y = 0
    for i in range(10*24000):
        x = 0.01
        y = y+1e-7
        yawrate = 3.141591/20
        obs, reward, done, info = env.step(np.array([0, x, yawrate, 38727]))
        print(f"the reward is {reward}")
        print(f"the state is {obs}")
        if done:
            obs = env.reset()
        env.render()

    print(f"THE ACTION SPACE IS : {env.action_space}")
    env.close()
