#!/usr/bin/env python3

import numpy as np
from gym import spaces

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from AttitudeControl import AttitudeControl


class AttitudeAviary(CtrlAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'):
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder)


        self.ctrl = [AttitudeControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([-np.pi,           -np.pi,           -np.pi,           -2000.])
        act_upper_bound = np.array([np.pi, np.pi, np.pi, 200000])
        return spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
                                               high=act_upper_bound,
                                               dtype=np.float32
                                               ) for i in range(self.NUM_DRONES)})


    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
                                                                     high=obs_upper_bound,
                                                                     dtype=np.float32
                                                                     ),
                                                 "neighbors": spaces.MultiBinary(self.NUM_DRONES)
                                                 }) for i in range(self.NUM_DRONES)})

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        adjacency_mat = self._getAdjacencyMatrix()
        return {str(i): {"state": self._getDroneStateVector(i), "neighbors": adjacency_mat[i, :]} for i in range(self.NUM_DRONES)}

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Uses PID control to target a desired velocity vector.
        Converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The desired velocity input for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        print(action)
        # for k, v in action.items():
        #### Get the current state of the drone  ###################
        k = 0
        state = self._getDroneStateVector(int(k))
            #### Normalize the first 3 components of the target velocity
            #### We get target roll, target pitch, target yawrate target thrust
            # if np.linalg.norm(v[0:3]) != 0:
            #     v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
            # else:
            # v_unit_vector = np.zeros(3)
        temp = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP,
                                                       cur_pos=state[0:3],
                                                       cur_quat=state[3:7],
                                                       cur_vel=state[10:13],
                                                       cur_ang_vel=state[13:16],
                                                       target_pos=action[0:3], # same as the current position
                                                       #target_rpy=np.array([v[0], v[1],state[9]]),
                                                       #target_rpy_rates=np.array([0, 0, v[2]]),
                                                       # target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector # target the desired velocity vector
                                                       #thrust=v[3]
                                                )

        print("rpm is ", temp[0])
        rpm[int(k),:] = temp[0]
        print(f"calculated rpm for drone {k} is {temp}")
        return rpm



    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
