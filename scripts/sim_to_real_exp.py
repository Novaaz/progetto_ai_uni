import os
import numpy as np
import sys
import torch as th
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.sim2real import city_noise, finetuning, base, to_csv, load_from_csv
from src.utils.constants import *
from src.utils.core import *
from src.evaluation.evaluate_functions import evaluate_sac_performance
from src.utils.noise import create_intelligent_env

noise_seeds = [42, 300, 756]	# seed per il rumore
noise_value =  [0.3, 0.6, 1.0]	# valore del rumore
env_base = StableBaselines3Wrapper(NormalizedSpaceWrapper(CityLearnEnv(**ENV_CONFIG)))

colors = ['blue', 'red', 'green', 'orange', 'purple']
ep_train_mean = []
step_eval_mean = []
fine_mean = []
#genero un ambiente per ogni rumore
for noise in noise_value:
	episodes_train = []
	rwds_tot = []
	episodes_fine = []
	for seed in noise_seeds:
		#1.1
		env_noise = create_intelligent_env(seed, False, noise)
		model, episodes = city_noise(env_noise, seed, noise)
		episodes_train.append(episodes)
		if os.path.exists(os.path.join(MODELS_DIR,'sim_to_real_results',f'sac_{noise}_{seed}_training.csv')):
			print(f"Training for noise {noise} and seed {seed} already exists, skipping training save.")
		else:
			to_csv(episodes, os.path.join(MODELS_DIR,'sim_to_real_results',f'sac_{noise}_{seed}_training.csv'))
		#1.2 testo l'agente nell'ambiente pulito
		if os.path.exists(os.path.join(MODELS_DIR,'sim_to_real_results',f'eval_{noise}_{seed}_steps.csv')):
			print(f"Evaluation for noise {noise} and seed {seed} already exists, skipping evaluation.")
			rw = load_from_csv(os.path.join(MODELS_DIR,'sim_to_real_results',f'eval_{noise}_{seed}_steps.csv'))
			step_eval_mean.append(rw)
			rwds_tot.append(np.sum(rw))
		else:	
			rw = evaluate_sac_performance(env_base, model)
			step_eval_mean.append(rw['step_rewards'])
			rwds_tot.append(rw['total_reward'])
			to_csv(rw['step_rewards'], os.path.join(MODELS_DIR,'sim_to_real_results',f'eval_{noise}_{seed}_steps.csv'))
		#1.3 faccio finetuning dell'agente allenato con rumore su ambiente pulito
		model, episodes = finetuning(env_base,seed=seed,noise=noise)
		episodes_fine.append(episodes)
		to_csv(episodes, os.path.join(MODELS_DIR,'sim_to_real_results',f'finetuned_{noise}_{seed}_finetune.csv'))
	fine_mean.append(np.mean(episodes_fine, axis=0))
	ep_train_mean.append(np.mean(episodes_train, axis=0))
	rwds_tot_mean = np.mean(rwds_tot)
	print(f"Mean total reward after finetuning with noise {noise}: {rwds_tot_mean}")

model, episodes = base(env_base)
if os.path.exists(os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0_training.csv')):
	episodes = load_from_csv(os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0_training.csv'))
else:
	to_csv(episodes, os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0_training.csv'))
if os.path.exists(os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0_eval_steps.csv')):
	print(f"Evaluation for base model already exists, skipping evaluation.")
	rw = load_from_csv(os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0_eval_steps.csv'))
	step_eval_mean.append(rw)
else:
	rw = evaluate_sac_performance(env_base, model)['step_rewards']
	to_csv(rw, os.path.join(MODELS_DIR,'sim_to_real_results','sac_base_0_eval_steps.csv'))

ep_train_mean.append(episodes)
step_eval_mean.append(rw)

plot_configs = [
	{
		'title': 'Fine-tuning Episodes',
		'xlabel': 'Episode',
		'data_source': fine_mean,
		'show_base': False
	},
	{
		'title': 'Training Episodes', 
		'xlabel': 'Episode',
		'data_source': ep_train_mean,
		'show_base': True
	},
	{
		'title': 'Evaluation Steps',
		'xlabel': 'Step', 
		'data_source': step_eval_mean,
		'show_base': True
	}
]

for config in plot_configs:
	plt.figure(figsize=(10, 6))
	
	if config['show_base']:
		plt.plot(config['data_source'][-1], label='Base Model', color='black')
	
	for noise_idx in range(len(noise_value)):
		plt.plot(config['data_source'][noise_idx], 
				label=f"{config['title']} {noise_value[noise_idx]}", 
				color=colors[noise_idx])
		
		if config['title'] == 'Evaluation Steps':
			mean_reward = np.sum(config['data_source'][noise_idx])
		else:
			mean_reward = np.mean(config['data_source'][noise_idx])
		print(f"Mean reward for {config['title']} with noise {noise_value[noise_idx]}: {mean_reward}")
	
	plt.title(f"{config['title']} Rewards")
	plt.xlabel(config['xlabel'])
	plt.ylabel('Reward')
	plt.legend()
	plt.grid(True)
	plt.show()