"""
Script per testare l'impatto del rumore su singole variabili delle osservazioni.
Genera modelli con rumore applicato a una sola variabile per volta,
permettendo di analizzare l'effetto specifico di ogni componente dell'osservazione.
"""

import os
import sys
import numpy as np
import gc
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import json
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_functions import (
	train_sac,
	generate_noise_levels
)
import src.utils.noise as noise_mod
from src.utils.constants import *
from src.utils.core import *
from src.evaluation.evaluate_functions import quick_evaluate
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.wrappers import TransformObservation
from functools import partial

def add_single_variable_noise(observations, variable_index, noise_level=0.15, noise_mean=0.0, 
							 noise_type='gaussian', name=None, seed=None):
	"""
	Aggiunge rumore a una singola variabile delle osservazioni.
	
	Parameters:
	observations: np.ndarray - Osservazioni originali
	variable_index: int - Indice della variabile da modificare
	noise_level: float - Livello di rumore da aggiungere
	noise_mean: float - Media del rumore
	noise_type: str - Tipo di rumore ('gaussian' o 'uniform')
	name: str - Nome univoco per identificare questa istanza di rumore
	
	Returns:
	np.ndarray - Osservazioni con rumore aggiunto solo alla variabile specificata
	"""
	obs = np.array(observations, dtype=float)

	orig_config = noise_mod._NOISE_CONFIG.copy()
	try:
		new_config = {k: v for k, v in orig_config.items() if k != variable_index}
		new_config[variable_index] = float(noise_level)
		noise_mod._NOISE_CONFIG = new_config

		noisy = noise_mod.add_noise_to_observations(obs, noise_type=noise_type, dinamic_noise=False, name=name, seed=seed)

	finally:
		noise_mod._NOISE_CONFIG = orig_config

	return noisy

def get_citylearn_variable_info():
	"""
	Restituisce informazioni sulle variabili disponibili in CityLearn.
	
	Returns:
	dict: Dizionario con indici e nomi delle variabili modificabili
	"""
	
	variable_info = {
		# Variabili ambientali e energetiche 
		3: "outdoor_dry_bulb_temperature",
		4: "outdoor_relative_humidity", 
		#5: "diffuse_solar_irradiance", #
		#6: "direct_solar_irradiance", #
		#7: "carbon_intensity",
		#8: "non_shiftable_load",
		9: "solar_generation",
		10: "electrical_storage_soc",
		#11: "net_electricity_consumption",
		#12: "electricity_pricing", #
		13: "cooling_storage_soc",
		14: "dhw_storage_soc",
		15: "indoor_dry_bulb_temperature",
		#16: "average_unmet_cooling_setpoint_difference",
		17: "indoor_relative_humidity",
		#18: "cooling_demand", #
		#19: "dhw_demand",
		20: "cooling_device_efficiency",
		21: "dhw_device_efficiency"
	}
	
	return variable_info

def create_single_variable_environment(base_env, variable_index, noise_level, noise_type='gaussian', 
									 noise_mean=0.0, env_name=None):
	"""
	Crea un ambiente con rumore applicato a una singola variabile usando un ambiente base.
	
	Parameters:
	base_env: CityLearnEnv - Ambiente base già creato
	variable_index: int - Indice della variabile da modificare
	noise_level: float - Livello di rumore
	noise_type: str - Tipo di rumore
	noise_mean: float - Media del rumore
	env_name: str - Nome dell'ambiente
	
	Returns:
	tuple: (ambiente_con_wrapper, nome_ambiente)
	"""
	if env_name is None:
		var_info = get_citylearn_variable_info()
		var_name = var_info.get(variable_index, f"var_{variable_index}")
		env_name = f"single_var_{var_name}_{noise_type}_{noise_level:.3f}"
	
	base_env.reset()
	env = NormalizedSpaceWrapper(base_env)
	env = StableBaselines3Wrapper(env)

	env = TransformObservation(
		env=env, 
		f=partial(
			add_single_variable_noise,
			variable_index=variable_index,
			noise_level=noise_level,
			noise_type=noise_type,
			noise_mean=noise_mean,
			name=env_name,
			seed=None
		)
	)

	return env, env_name

def train_model_with_single_variable_noise(base_env, variable_index, noise_std, seed, model_id):
	"""
	Allena un modello SAC con rumore su una singola variabile usando ambiente base.
	
	Parameters:
	base_env: CityLearnEnv - Ambiente base già creato
	variable_index: int - Indice della variabile da modificare
	noise_std: float - Deviazione standard del rumore
	seed: int - Seed per riproducibilità
	model_id: str - Identificativo del modello
	
	Returns:
	tuple: (modello_allenato, dati_training, metadati)
	"""
	var_info = get_citylearn_variable_info()
	var_name = var_info.get(variable_index, f"var_{variable_index}")
	
	print(f"📈 Training Modello {model_id} | Variabile: {var_name} (idx={variable_index}) | Noise STD: {noise_std:.4f}")
	
	try:
		train_env, env_name = create_single_variable_environment(
			base_env=base_env,
			variable_index=variable_index,
			noise_level=noise_std,
			noise_type='gaussian',
			noise_mean=0.0,
			env_name=f"train_{model_id}"
		)
		
		# Allena il modello
		_, trained_model, training_rewards, timesteps = train_sac(
			env=train_env,
			seed=seed,
			track_rewards=True,
			eval_freq=50,
			episodes=EPISODES
		)
		
		# Prepara dati di training
		training_data = {
			'rewards': training_rewards,
			'timesteps': timesteps,
			'noise_std': noise_std,
			'variable_index': variable_index,
			'variable_name': var_name,
			'seed': seed,
			'model_id': model_id,
			'env_name': env_name,
		}
		
		# Metadati per il salvataggio
		metadata = {
			'variable_index': variable_index,
			'variable_name': var_name,
			'noise_std': noise_std,
			'seed': seed
		}
		
		print(f"  ✅ Training completato | Variabile: {var_name} | Episodi: {len(training_rewards)}")
		return trained_model, training_data, metadata
		
	except Exception as e:
		print(f"  ❌ Errore durante training modello {model_id}: {e}")
		import traceback
		traceback.print_exc()
		return None, None, None

def train_test_and_save_model(base_env, clean_env, variable_index, noise_std, seed, model_id, results_dir):
	"""
	Allena, testa e salva immediatamente un modello per liberare la memoria.
	
	Parameters:
	base_env: CityLearnEnv - Ambiente base per training
	clean_env: Ambiente wrapper pulito per testing (condiviso)
	variable_index: int - Indice della variabile da modificare
	noise_std: float - Deviazione standard del rumore
	seed: int - Seed per riproducibilità
	model_id: str - Identificativo del modello
	results_dir: str - Directory per salvare i risultati
	
	Returns:
	dict: Risultati del modello o None se fallisce
	"""
	var_info = get_citylearn_variable_info()
	var_name = var_info.get(variable_index, f"var_{variable_index}")
	
	print(f"📈 Training+Test Modello {model_id} | Variabile: {var_name} (idx={variable_index}) | Noise STD: {noise_std:.4f}")
	
	try:
		trained_model, training_data, metadata = train_model_with_single_variable_noise(
			base_env=base_env,
			variable_index=variable_index,
			noise_std=noise_std,
			seed=seed,
			model_id=model_id
		)
		
		if trained_model is None:
			print(f"  ❌ Training fallito per {model_id}")
			return None
		
		print(f"  💾 Salvando modello...")
		try:
			filepath = save_single_variable_model(trained_model, metadata, results_dir)
		except Exception as save_error:
			print(f"  ❌ Errore durante salvataggio: {save_error}")
			filepath = "SAVE_FAILED"
		
		del trained_model
		if training_data:
			del training_data
		
		gc.collect()
		
	except Exception as e:
		print(f"  ❌ Errore durante training/test modello {model_id}: {e}")
		import traceback
		traceback.print_exc()
		return None

def initialize_single_variable_results_dir(base_dir="single_variable"):
	"""
	Inizializza le directory per salvare i risultati dell'analisi.
	
	Parameters:
	base_dir: str - Nome della directory base
	
	Returns:
	str: Percorso della directory creata
	"""
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	results_dir = os.path.join(RESULTS_DIR, base_dir, f"analysis_{timestamp}")
	
	# Crea directory principali
	os.makedirs(results_dir, exist_ok=True)
	os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
	os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)
	
	return results_dir

def save_single_variable_model(model, metadata, results_dir):
	"""
	Salva un modello con naming convention appropriato.
	
	Parameters:
	model: SAC model - Modello da salvare
	metadata: dict - Metadati del modello
	results_dir: str - Directory di salvataggio
	
	Returns:
	str: Percorso del file salvato
	"""
	var_name = metadata['variable_name']
	noise_std = metadata['noise_std']
	flag = "s"
	seed = metadata['seed']
	
	# Nome file: var_<nome_variabile>_<static/dynamic>_<noise_std>_seed<seed>.zip
	filename = f"var_{var_name}_{flag}_{noise_std:.3f}_seed{seed}.zip"
	filepath = os.path.join(results_dir, "models", filename)
	
	model.save(filepath)
	print(f"    💾 Salvato: {filename}")
	
	return filepath

def save_progress_checkpoint(results_dir, current_state):
	"""
	Salva checkpoint del progresso per recovery.
	"""
	checkpoint_data = {
		'timestamp': datetime.now().isoformat(),
		'current_variable_index': current_state.get('variable_index'),
		'current_noise_level_index': current_state.get('noise_index'),
		'completed_models': current_state.get('completed_models', []),
		'failed_models': current_state.get('failed_models', []),
		'total_results': len(current_state.get('all_results', [])),
		'last_model_id': current_state.get('last_model_id'),
		'noise_memory_keys': list(noise_mod._NOISE_MEMORY.keys()) if noise_mod._NOISE_MEMORY else []
	}
	
	checkpoint_path = os.path.join(results_dir, "recovery_checkpoint.json")
	with open(checkpoint_path, 'w') as f:
		json.dump(checkpoint_data, f, indent=2)
	
	print(f"💾 Checkpoint salvato: {os.path.basename(checkpoint_path)}")
	return checkpoint_path

def load_existing_results(results_dir):
	"""
	Carica risultati esistenti da recovery.
	"""
	existing_results = []
	
	# Cerca file di risultati parziali
	partial_files = glob.glob(os.path.join(results_dir, "data", "partial_results_*.csv"))
	partial_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
	
	if partial_files:
		latest_partial = partial_files[-1]
		print(f"📂 Caricando risultati parziali: {os.path.basename(latest_partial)}")
		df_existing = pd.read_csv(latest_partial)
		existing_results = df_existing.to_dict('records')
		print(f"✅ Caricati {len(existing_results)} risultati esistenti")
	
	# Cerca anche final_results se esiste
	final_path = os.path.join(results_dir, "data", "final_results.csv")
	if os.path.exists(final_path):
		print(f"📂 Caricando risultati finali esistenti...")
		df_final = pd.read_csv(final_path)
		existing_results = df_final.to_dict('records')
		print(f"✅ Caricati {len(existing_results)} risultati finali")
	
	return existing_results

def get_completed_models(results_dir):
	"""
	Identifica modelli già completati dai file salvati.
	"""
	models_dir = os.path.join(results_dir, "models")
	if not os.path.exists(models_dir):
		return set()
	
	model_files = glob.glob(os.path.join(models_dir, "*.zip"))
	completed_models = set()
	
	for model_file in model_files:
		filename = os.path.basename(model_file)
		# Estrai info dal filename: var_<nome>_<s/d>_<noise>_seed<seed>.zip
		try:
			parts = filename.replace('.zip', '').split('_')
			if len(parts) >= 5:
				var_name = '_'.join(parts[1:-3])  # Nome variabile può contenere _
				static_dynamic = parts[-3]  # 's' o 'd'
				noise_str = parts[-2]
				seed_str = parts[-1].replace('seed', '')
				
				model_key = f"{var_name}_{static_dynamic}_{noise_str}_{seed_str}"
				completed_models.add(model_key)
		except:
			continue
	
	print(f"🔍 Trovati {len(completed_models)} modelli già completati")
	return completed_models

def should_skip_model(var_name,  noise_std, seed, completed_models):
	"""
	Verifica se un modello dovrebbe essere saltato perché già completato.
	"""
	static_dynamic = 's'
	model_key = f"{var_name}_{static_dynamic}_{noise_std:.3f}_{seed}"
	return model_key in completed_models

def main_with_recovery():
	"""
	Versione principale con supporto per recovery e ambiente condiviso.
	"""
	print("🎯 ANALISI RUMORE SU SINGOLE VARIABILI (CON RECOVERY)")
	print("="*70)
	
	# 1. CONTROLLO RECOVERY
	recovery_mode = False
	existing_results_dir = None
	
	recovery_input = input("Riprendere esperimento esistente? (y/n/path): ").strip()
	
	if recovery_input.lower() in ['y', 'yes']:
		# Cerca directory recenti
		base_results = os.path.join(RESULTS_DIR, "single_variable")
		if os.path.exists(base_results):
			recent_dirs = sorted([d for d in os.listdir(base_results) 
								if d.startswith('analysis_')], reverse=True)
			if recent_dirs:
				existing_results_dir = os.path.join(base_results, recent_dirs[0])
				print(f"📂 Directory più recente: {existing_results_dir}")
				recovery_mode = True
	elif os.path.isdir(recovery_input):
		existing_results_dir = recovery_input
		recovery_mode = True
		print(f"📂 Directory specificata: {existing_results_dir}")
	
	# 2. CONFIGURAZIONE
	if recovery_mode and existing_results_dir:
		print(f"\n🔄 MODALITÀ RECOVERY")
		print("-" * 40)
		
		# Carica configurazione esistente
		config_path = os.path.join(existing_results_dir, "experiment_config.json")
		if os.path.exists(config_path):
			with open(config_path, 'r') as f:
				config = json.load(f)
			
			seed = config['seed']
			min_noise = config['min_noise']
			max_noise = config['max_noise']
			models_per_variable = config['models_per_variable']
			noise_levels = config['noise_levels']
			
			print(f"✅ Configurazione caricata: seed={seed}, range={min_noise}-{max_noise}")
		else:
			print("❌ Configurazione non trovata, inserire manualmente:")
			seed = int(input("Seed: "))
			min_noise = float(input("Rumore minimo: "))
			max_noise = float(input("Rumore massimo: "))
			models_per_variable = int(input("Modelli per variabile: "))
			noise_levels = generate_noise_levels(models_per_variable, min_noise, max_noise)
		
		# Carica risultati esistenti
		all_results = load_existing_results(existing_results_dir)
		completed_models = get_completed_models(existing_results_dir)
		results_dir = existing_results_dir
		
		print(f"📊 Risultati esistenti: {len(all_results)}")
		print(f"🎯 Modelli completati: {len(completed_models)}")
		
	else:
		print(f"\n🆕 NUOVA ANALISI")
		print("-" * 40)
		
		# Configurazione normale
		seed = int(input("Seed per l'esperimento (default: random): ") or str(np.random.randint(0, 10000)))
		min_noise = float(input("Rumore minimo (default 0.2): ") or "0.2")
		max_noise = float(input("Rumore massimo (default 0.8): ") or "0.8")
		models_per_variable = int(input("Modelli per variabile (default 3): ") or "3")
		noise_levels = generate_noise_levels(models_per_variable, min_noise, max_noise)
		
		all_results = []
		completed_models = set()
		results_dir = initialize_single_variable_results_dir()
	
	# 3. INFORMAZIONI ESPERIMENTO
	variable_info = get_citylearn_variable_info()
	
	print(f"\n📋 CONFIGURAZIONE ESPERIMENTO:")
	print(f"   • Seed: {seed}")
	print(f"   • Range rumore: {min_noise:.3f} - {max_noise:.3f}")
	print(f"   • Modelli per variabile: {models_per_variable}")
	print(f"   • Episodi training: {EPISODES}")
	print(f"   • Variabili da testare: {len(variable_info)}")
	print(f"   • Modelli completati: {len(completed_models)}")
	
	# Chiedi conferma
	action = "Continuare" if recovery_mode else "Procedere"
	confirm = input(f"\n{action} con l'analisi? (y/n): ").lower().strip()
	if confirm not in ['y', 'yes', 's', 'si']:
		print("❌ Operazione annullata")
		return
	
	# 4. CREAZIONE AMBIENTI CONDIVISI (UNA SOLA VOLTA!)
	print(f"\n🏗️  INIZIALIZZAZIONE AMBIENTI")
	print("-" * 40)
	
	print("🌍 Creando ambiente CityLearn base...")
	base_citylearn_env = CityLearnEnv(**ENV_CONFIG)
	print("✅ Ambiente CityLearn base creato")
	
	print("🧪 Creando ambiente pulito per test...")
	clean_test_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(base_citylearn_env))
	print("✅ Ambiente pulito per test creato")
	
	# 5. ESECUZIONE CON CHECKPOINT
	print(f"\n📁 Directory risultati: {results_dir}")
	print(f"\n🚀 {'RIPRESA' if recovery_mode else 'INIZIO'} TRAINING")
	print("="*60)
	
	start_time = time.time()
	model_counter = len(all_results)
	skipped_counter = 0
	
	try:
		# Per ogni variabile
		for var_idx, var_name in variable_info.items():
			print(f"\n📊 VARIABILE: {var_name} (indice {var_idx})")
			print("-" * 50)
			
			# Per ogni livello di rumore
			for i, noise_std in enumerate(noise_levels):
				model_counter += 1
				model_id = f"var{var_idx}_model{i+1}"
				
				# State per checkpoint
				current_state = {
					'variable_index': var_idx,
					'noise_index': i,
					'all_results': all_results,
					'completed_models': list(completed_models),
					'last_model_id': model_id
				}
				
				if should_skip_model(var_name, noise_std, seed, completed_models):
					print(f"  ⏭️  Skip ")
					skipped_counter += 1
				else:
					print(f"  🔸 Training...")
					result_static = train_test_and_save_model(
						base_env=base_citylearn_env,
						clean_env=clean_test_env,
						variable_index=var_idx,
						noise_std=noise_std,
						seed=seed,
						model_id=f"{model_id}_static",
						results_dir=results_dir
					)
					
					if result_static is not None:
						all_results.append(result_static)
						# Aggiorna completed_models
						model_key = f"{var_name}_s_{noise_std:.3f}_{seed}"
						completed_models.add(model_key)
				
				# CHECKPOINT ogni 5 modelli
				if (model_counter - skipped_counter) % 5 == 0:
					save_progress_checkpoint(results_dir, current_state)
					
					# Salva risultati parziali
					if all_results:
						df_partial = pd.DataFrame(all_results)
						partial_path = os.path.join(results_dir, "data", f"partial_results_{len(all_results)}.csv")
						df_partial.to_csv(partial_path, index=False)
						print(f"  💾 Checkpoint: {len(all_results)} risultati salvati")
		
		# 6. SALVATAGGIO FINALE
		print(f"\n💾 SALVATAGGIO FINALE")
		print("-" * 30)
		
		# Salva configurazione esperimento
		config_data = {
			'timestamp': datetime.now().isoformat(),
			'seed': seed,
			'min_noise': min_noise,
			'max_noise': max_noise,
			'models_per_variable': models_per_variable,
			'noise_levels': noise_levels,
			'episodes': EPISODES,
			'total_models_generated': len(all_results),
			'variables_tested': list(variable_info.keys())
		}
		
		config_path = os.path.join(results_dir, "experiment_config.json")
		with open(config_path, 'w') as f:
			json.dump(config_data, f, indent=2)
		
		# Salva risultati finali in CSV
		if all_results:
			df_results = pd.DataFrame(all_results)
			csv_path = os.path.join(results_dir, "data", "final_results.csv")
			df_results.to_csv(csv_path, index=False)
			
			print(f"✅ Configurazione salvata: {os.path.basename(config_path)}")
			print(f"✅ Risultati finali salvati: {os.path.basename(csv_path)}")
			print(f"📊 Modelli generati: {len(all_results)}")
			print(f"🏗️  Variabili testate: {len(variable_info)}")
			
			# Riepilogo per variabile con risultati test
			print(f"\n📈 RIEPILOGO PER VARIABILE (con test results):")
			for var_idx, var_name in variable_info.items():
				var_models = [r for r in all_results if r['variable_index'] == var_idx]
				if var_models:
					avg_test_reward = np.mean([r['test_mean_reward'] for r in var_models])
					best_test_reward = max([r['test_mean_reward'] for r in var_models])
					print(f"   {var_name}: {len(var_models)} modelli")
					print(f"     Test reward media: {avg_test_reward:.2f}, migliore: {best_test_reward:.2f}")
		
		else:
			print("❌ Nessun modello generato con successo!")
	
	except KeyboardInterrupt:
		print(f"\n⚠️ INTERRUZIONE UTENTE")
		print("💾 Salvando checkpoint finale...")
		save_progress_checkpoint(results_dir, current_state)
		if all_results:
			df_emergency = pd.DataFrame(all_results)
			emergency_path = os.path.join(results_dir, "data", "emergency_save.csv")
			df_emergency.to_csv(emergency_path, index=False)
			print(f"🚨 Salvataggio emergenza: {len(all_results)} risultati")
	
	except Exception as e:
		print(f"\n❌ ERRORE durante l'esperimento: {e}")
		save_progress_checkpoint(results_dir, current_state)
		import traceback
		traceback.print_exc()
	
	finally:
		# Pulizia finale
		print(f"\n🧹 PULIZIA FINALE")
		del base_citylearn_env, clean_test_env
		
		end_time = time.time()
		duration = end_time - start_time
		print(f"\n⏱️  Esperimento completato in {duration/60:.2f} minuti ({duration:.1f} secondi)")
		print(f"📁 Risultati salvati in: {results_dir}")

		import gc
		gc.collect()

if __name__ == "__main__":
	main_with_recovery()