from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from warnings import WarningMessage
import numpy as np
from legged_gym.networks.base_wrapper import *
import gym
import copy
import torch




class NormObsWithImg(gym.ObservationWrapper, BaseWrapper):
  """
  Normalized Observation => Optional, Use Momentum
  """

  def __init__(self, env, epsilon=1e-4, clipob=10.):
    super(NormObsWithImg, self).__init__(env)
    self.count = epsilon
    self.clipob = clipob
    self._obs_normalizer = Normalizer(env.shape)
    self.state_shape = np.prod(env.shape)

  def copy_state(self, source_env):
    # self._obs_rms = copy.deepcopy(source_env._obs_rms)
    self._obs_var = copy.deepcopy(source_env._obs_var)
    self._obs_mean = copy.deepcopy(source_env._obs_mean)

  def observation(self, observation):
    if self.training:
      self._obs_normalizer.update_estimate(
        observation[..., :self.state_shape]
      )
    img_obs = observation[..., self.state_shape:]
    return np.hstack([
      self._obs_normalizer.filt(observation[..., :self.state_shape]),
      img_obs
    ])
