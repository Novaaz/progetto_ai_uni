from datetime import datetime
import numpy as np
from src.utils.classes import FineTuningCallback
from stable_baselines3 import SAC
from ..utils.constants import *
from ..utils.core import *
from ..evaluation.evaluate_functions import evaluate_sac_performance
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.wrappers import TransformObservation
from functools import partial
import os
import csv

def train_sac(env, seed=42, sac_model=None, episodes=EPISODES, time_steps=None, track_rewards=False, eval_freq=1000, finetune=False):
	if sac_model is None:
		sac_model = SAC(
		policy='MlpPolicy',
		env=env,
		**SAC_KWARGS,
		seed=seed,
		verbose=1)

	if time_steps is None:
		time_steps = env.unwrapped.time_steps - 1
		
	callback = None
	
	if finetune:
		callback = FineTuningCallback(env)

	if track_rewards:
		reward_tracker = EpisodeRewardTracker(env=env, episode_length=time_steps, verbose=1)
		if callback is not None:
			callback = CallbackList([callback, reward_tracker])
		else:
			callback = reward_tracker
	print(callback)
	total_timesteps = episodes * time_steps
	
	env.reset()
	sac_model.learn(
		total_timesteps=total_timesteps,
		reset_num_timesteps=True,
		callback=callback,
		progress_bar=True
	)

	if track_rewards:
		return sac_model, reward_tracker.episode_rewards
	else:
		return sac_model

class EpisodeRewardTracker(BaseCallback):
	"""
	Callback semplice per tracciare le reward durante il training.
	Raccoglie le reward cumulative alla fine di ogni episodio completo.
	Parametri:
	- episode_length (int|None): lunghezza episodio in passi (se None usa fallback)
	"""
	def __init__(self, env, episode_length, verbose=0):
		super().__init__(verbose)
		self.episode_rewards = []
		self.env_timestep = 0
		self.env = env
		# memorizza la lunghezza episodio fornita dall'esterno
		self.episode_length = int(episode_length)
		
	def _on_step(self) -> bool:
		self.env_timestep += 1
		if self.env_timestep % self.episode_length == 0:
			ep = evaluate_sac_performance(self.env, self.model)['total_reward']
			self.episode_rewards.append(ep)

			if self.verbose > 0:
				episode_num = self.env_timestep // self.episode_length
				print(f"Fine episodio {episode_num} (timestep {self.env_timestep}): Reward episodica = {ep:.2f}")
					
		return True