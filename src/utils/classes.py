from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sklearn.model_selection import train_test_split
import os
import sys
import copy

class HybridReplayBuffer(ReplayBuffer):
	def __init__(self, 
				 buffer_size,
				 observation_space,
				 action_space,
				 device="cpu",
				 n_envs=1,
				 alpha=0.3,  # proporzione offline
				 **kwargs):
		super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, **kwargs)
		self.alpha = alpha
		self.offline_buffer = ReplayBuffer(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, **kwargs)

		self._alpha_log = []

	def sample(self, batch_size: int, env=None, **kwargs):
		if self.offline_buffer is None:
			return super().sample(batch_size, env=env, **kwargs)

		num_samples = self.offline_buffer.pos if not self.offline_buffer.full else self.offline_buffer.buffer_size
		indices = np.arange(num_samples)
		n_offline = int(self.alpha * batch_size)
		n_online = batch_size - n_offline
		if n_offline > num_samples:
			n_offline = num_samples
			n_online = batch_size - n_offline

		offline_samples = None
		online_samples = None

		if n_offline > 0:
			offline_samples = self.offline_buffer.sample(n_offline, env=env, **kwargs)
		if n_online > 0:
			online_samples = super().sample(n_online, env=env, **kwargs)

		def _cat(a, b):
			if a is None:
				return b
			if b is None:
				return a
			return th.cat([a, b], dim=0)

		def _extract_obs_tensor(samples):
			if samples is None:
				return None
			obs = samples.observations
			if isinstance(obs, dict):
				if 'obs' in obs:
					return obs['obs']
				values = list(obs.values())
				return th.cat(values, dim=-1)
			return obs

		# estrai campioni in forma tensor
		obs_off = _extract_obs_tensor(offline_samples)
		obs_on = _extract_obs_tensor(online_samples)
		obs = _cat(obs_off, obs_on)

		next_off = None
		next_on = None
		if offline_samples is not None:
			no = getattr(offline_samples, 'next_observations', None)
			if isinstance(no, dict):
				next_off = no.get('obs') if 'obs' in no else th.cat(list(no.values()), dim=-1)
			else:
				next_off = no
		if online_samples is not None:
			no = getattr(online_samples, 'next_observations', None)
			if isinstance(no, dict):
				next_on = no.get('obs') if 'obs' in no else th.cat(list(no.values()), dim=-1)
			else:
				next_on = no
		next_obs = _cat(next_off, next_on)

		act = _cat(
			getattr(offline_samples, 'actions', None),
			getattr(online_samples, 'actions', None)
		)
		rew = _cat(
			getattr(offline_samples, 'rewards', None),
			getattr(online_samples, 'rewards', None)
		)
		don = _cat(
			getattr(offline_samples, 'dones', None),
			getattr(online_samples, 'dones', None)
		)

		return ReplayBufferSamples(
			observations=obs,
			actions=act,
			next_observations=next_obs,
			rewards=rew,
			dones=don,
		)
	
def _to_2d_array(arr, name):
	a = np.asarray(arr)
	return a.reshape(a.shape[0], -1)