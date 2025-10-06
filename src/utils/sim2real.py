import csv
import os
from pyexpat import model
import pandas as pd
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train_functions import train_sac
from src.utils.constants import *
from src.utils.core import *
from src.utils.classes import DynamicsModel

#Citylearn + noise
def city_noise(env,seed,noise):
	if os.path.exists(os.path.join(MODELS_DIR,'sim_to_real',f'sac_{noise}_{seed}.zip')):
		print(f"Model sac_{noise}_{seed}.zip already exists, skipping training.")
		model = SAC.load(os.path.join(MODELS_DIR,'sim_to_real',f'sac_{noise}_{seed}.zip'))
		try:
			episodes = load_from_csv(os.path.join(MODELS_DIR,'sim_to_real_results',f'sac_{noise}_{seed}_training.csv'))
		except FileNotFoundError:
			episodes = None
		return model, episodes
	env.reset()
	model, episodes = train_sac(env,seed=seed,track_rewards=True)
	model.save(os.path.join(MODELS_DIR,'sim_to_real',f'sac_{noise}_{seed}.zip'))
	to_csv(episodes, os.path.join(MODELS_DIR,'sim_to_real_results',f'sac_{noise}_{seed}_training.csv'))
	return model, episodes

def finetuning(env,episodes=35,seed=None,path=None,noise=None, model=None):
	if model is None:
		if path is None:
			path = os.path.join(MODELS_DIR,'sim_to_real',f'sac_{noise}_{seed}.zip')
		if os.path.exists(os.path.join(MODELS_DIR,'sim_to_real',f'sac_finetuned_{noise}_{seed}.zip')) and os.path.exists(os.path.join(MODELS_DIR,'sim_to_real_results',f'finetuned_{noise}_{seed}_finetune.csv')):
			print(f"Model sac_finetuned_{noise}_{seed}.zip already exists, skipping training.")
			model = SAC.load(os.path.join(MODELS_DIR,'sim_to_real',f'sac_finetuned_{noise}_{seed}.zip'))
			ep = load_from_csv(os.path.join(MODELS_DIR,'sim_to_real_results',f'finetuned_{noise}_{seed}_finetune.csv'))
			return model, ep
		model = SAC.load(path)	
	#Ambiente pulito
	env.reset()
	model2 = SAC("MlpPolicy", env, verbose=1, **FINETUNING_KWARGS)
	#aggiorno lr
	model2.policy.load_state_dict(model.policy.state_dict())
	model2, ep = train_sac(env,sac_model=model2,track_rewards=True,episodes=episodes)
	model2.save(os.path.join(MODELS_DIR,'sim_to_real',f'sac_finetuned_{noise}_{seed}.zip'))
	to_csv(ep, os.path.join(MODELS_DIR,'sim_to_real_results',f'finetuned_{noise}_{seed}_finetune.csv'))
	return model2, ep

def base(env):
	if os.path.exists(os.path.join(MODELS_DIR,'sim_to_real',f'sac_base_0.zip')) and os.path.exists(os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0.csv')):
		print(f"Model sac_base_0.zip already exists, skipping training.")
		model = SAC.load(os.path.join(MODELS_DIR,'sim_to_real',f'sac_base_0.zip'))
		episodes = load_from_csv(os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0.csv'))
		return model, episodes
	env.reset()
	model, episodes = train_sac(env,track_rewards=True)
	model.save(os.path.join(MODELS_DIR,'sim_to_real',f'sac_base_0.zip'))
	to_csv(episodes, os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0.csv'))
	return model, episodes

def to_csv(episodes, filename):
	with open(filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Episode-Step', 'Reward'])
		for i, reward in enumerate(episodes):
			writer.writerow([i + 1, reward])

def load_from_csv(filename):
	df = pd.read_csv(filename)
	return df['Reward'].tolist()