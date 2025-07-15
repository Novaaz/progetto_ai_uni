import numpy as np
import pandas as pd
from ..utils.constants import *
from ..utils.core import *

def evaluate_sac_performance(env, sac_model, episode_name="Model"):
	"""
	Esegue un modello SAC allenato e raccoglie i risultati per ogni timestep.
	
	Parametri:
	env: CityLearnEnv - Ambiente su cui eseguire il modello
	sac_model: SAC - Modello SAC allenato
	episode_name: str - Nome da assegnare all'episodio per l'identificazione
	
	Returns:
	dict - Dizionario contenente i risultati e le informazioni sulle azioni
	"""
	print(f"Valutazione delle prestazioni di {episode_name}")
	observations, _ = env.reset()
	step_rewards = []
	sac_actions_list = []
	total_reward = 0
	
	try:
		step_count = 0
		
		while not env.unwrapped.terminated:
			actions, _ = sac_model.predict(observations, deterministic=True)
			observations, rewards, _, _, _ = env.step(actions)
			
			reward_val = float(rewards)
				
			step_rewards.append(reward_val)  # Salva reward del singolo step
			sac_actions_list.append(actions)
			total_reward += reward_val
			step_count += 1
			
	except Exception as e:
		print(f"Errore durante la valutazione di {episode_name}: {e}")
	
	print(f"Ricompensa totale per {episode_name}: {total_reward:.2f}")
	
	return {
		"name": episode_name,
		"total_reward": total_reward,
		"step_rewards": step_rewards,
		"actions": sac_actions_list
	}

def print_model_params(model, name):
	"""Stampa alcuni parametri del modello per verificare le differenze"""
	try:
		# Ottieni i primi 5 parametri del critic network
		critic_params = list(model.critic.parameters())[0].data.flatten()[:5]
		print(f"{name} - Primi 5 parametri critic: {critic_params}")
		
		# Ottieni i primi 5 parametri dell'actor network  
		actor_params = list(model.actor.parameters())[0].data.flatten()[:5]
		print(f"{name} - Primi 5 parametri actor: {actor_params}")
		
	except Exception as e:
		print(f"Errore nel debug del modello {name}: {e}")

def performance_evaluation(first_reward, second_reward, model_name):
	if abs(first_reward) > 0:
		improvement_ft = ((second_reward - first_reward) / abs(first_reward)) * 100
		print(f"\nIl modello {model_name} ha prodotto un cambiamento del {improvement_ft:.2f}% nella performance")

		if improvement_ft > 0:
			print(f"→ Il modello {model_name} ha MIGLIORATO le prestazioni del {abs(improvement_ft):.2f}%")
		else:
			print(f"→ Il modello {model_name} ha PEGGIORATO le prestazioni del {abs(improvement_ft):.2f}%")

def quick_evaluate(model, env, max_steps):
	"""
	Valuta rapidamente un singolo modello
	"""
	try:
		obs, _ = env.reset()
		total_reward = 0
		steps = 0
		
		while steps < max_steps:
			try:
				action, _ = model.predict(obs, deterministic=True)
				obs, reward, terminated, truncated, _ = env.step(action)
				
				reward_val = np.mean(reward) if hasattr(reward, '__iter__') else reward
				total_reward += reward_val
				steps += 1
				
				if terminated or truncated:
					break
					
			except Exception as e:
				print(f"    ⚠️  Errore durante valutazione step {steps}: {e}")
				break
		
		return total_reward
		
	except Exception as e:
		print(f"    ❌ Errore durante valutazione modello: {e}")
		return -999999