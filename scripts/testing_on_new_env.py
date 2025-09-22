"""
Script per testare modelli rappresentativi + ensemble su N ambienti diversi
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_functions import add_noise_to_observations
from src.utils.constants import *
from src.utils.core import *
from src.models.ensemble import load_ensemble, list_saved_ensembles
from src.evaluation.evaluate_functions import evaluate_sac_performance
from stable_baselines3 import SAC
from gymnasium.wrappers import TransformObservation

def load_representative_models(analysis_dir, top_n=None):
	"""Carica modelli rappresentativi - riusa la logica esistente"""
	csv_files = [f for f in os.listdir(analysis_dir) if f.startswith('best_models_') and f.endswith('.csv')]
	if not csv_files:
		raise FileNotFoundError("Nessun file best_models_*.csv trovato!")
	
	latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(analysis_dir, x)))
	df = pd.read_csv(os.path.join(analysis_dir, latest_csv))
	
	# Se top_n non specificato, prendi un rappresentativo per ogni gruppo
	if top_n is None:
		# Un modello per ogni combinazione model_type + noise_level
		representative_models = []
		for group_key, group_df in df.groupby(['model_type', 'noise_level']):
			best_in_group = group_df.nlargest(1, 'median').iloc[0]
			if os.path.exists(best_in_group['path']):
				representative_models.append({
					'name': best_in_group['name'],
					'path': best_in_group['path'],
					'expected_performance': best_in_group['median'],
					'type': best_in_group['model_type'],
					'noise': best_in_group['noise_level']
				})
	else:
		# Top N modelli (logica esistente)
		top_models = df.nlargest(top_n, 'median')
		representative_models = []
		for _, row in top_models.iterrows():
			if os.path.exists(row['path']):
				representative_models.append({
					'name': row['name'],
					'path': row['path'],
					'expected_performance': row['median'],
					'type': row['model_type'],
					'noise': row['noise_level']
				})
	
	print(f"📊 Modelli selezionati: {len(representative_models)}")
	for i, model in enumerate(representative_models, 1):
		print(f"  {i}. {model['name']} (type: {model['type']}, noise: {model['noise']:.3f})")
	
	return representative_models

def create_random_test_env():
	"""Crea un ambiente di test con parametri casuali - riusa la logica esistente"""
	env = CityLearnEnv(**ENV_CONFIG)
	env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
	env = TransformObservation(env, partial(
		add_noise_to_observations, 
		noise_type='gaussian',
		dinamic_noise=False,
		name='random_env',
	))
	
	return env

def main():
	"""Test modelli rappresentativi su N ambienti casuali"""
	print("🧪 === MULTI-ENVIRONMENT TESTING ===")
	print("="*50)
	
	# Input utente
	n_environments = int(input("Numero ambienti di test (default 10): ") or "10")
	use_top_n = input("Usa top N modelli invece di rappresentativi? (y/n, default n): ").lower().strip()
	
	if use_top_n in ['y', 'yes', 's', 'si']:
		top_n = int(input("Quanti top modelli? (default 5): ") or "5")
	else:
		top_n = None
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	try:
		print("\n📊 Caricamento modelli...")
		analysis_dir = os.path.join(RESULTS_DIR, CSV_DIR)
		models = load_representative_models(analysis_dir, top_n)
		
		if not models:
			print("❌ Nessun modello trovato!")
			return
		
		print("\n🔧 Caricamento ensemble...")
		ensemble_info = None
		try:
			ensembles_dir = os.path.join(RESULTS_DIR, 'ensembles')
			ensembles = list_saved_ensembles(ensembles_dir)
			if ensembles:
				ensemble_info = max(ensembles, key=lambda x: x['creation_time'])
				print(f"  ✅ Ensemble: {ensemble_info['filename']}")
			else:
				print("  ⚠️  Nessun ensemble trovato")
		except:
			print("  ❌ Errore caricamento ensemble")
		
		print(f"\n🧪 Testing su {n_environments} ambienti...")
		all_results = []
		
		for env_idx in range(n_environments):
			print(f"\n🌍 Ambiente {env_idx+1}/{n_environments}")
			
			# Crea nuovo ambiente casuale
			test_env = create_random_test_env()
			
			# Test tutti i modelli su questo ambiente
			for model_info in models:
				try:
					print(f"  🔄 {model_info['name']}...", end=' ')
					
					model = SAC.load(model_info['path'], test_env)
					result = evaluate_sac_performance(test_env, model, 
													f"{model_info['name']}_env{env_idx+1:02d}")
					
					# Aggiungi metadati
					result.update({
						'env_id': f"env_{env_idx+1:02d}",
						'model_type': model_info['type'],
						'model_noise_level': model_info['noise'],
						'is_ensemble': False
					})
					
					all_results.append(result)
					print(f"Reward: {result['total_reward']:.2f}")
					
				except Exception as e:
					print(f"❌ {e}")
			
			# Test ensemble se disponibile
			if ensemble_info:
				try:
					print(f"  🔧 Ensemble...", end=' ')
					
					ensemble = load_ensemble(ensemble_info['filepath'], test_env, load_models=True)
					result = evaluate_sac_performance(test_env, ensemble, 
													f"Ensemble_env{env_idx+1:02d}")
					
					result.update({
						'env_id': f"env_{env_idx+1:02d}",
						'model_type': 'ensemble',
						'model_noise_level': -1,
						'is_ensemble': True
					})
					
					all_results.append(result)
					print(f"Reward: {result['total_reward']:.2f}")
					
				except Exception as e:
					print(f"❌ {e}")
		
		if all_results:
			print(f"\n💾 Salvando {len(all_results)} risultati...")
			
			csv_data = []
			for result in all_results:
				csv_data.append({
					'model_name': result['name'],
					'env_id': result['env_id'],
					'model_type': result['model_type'],
					'model_noise_level': result['model_noise_level'],
					'is_ensemble': result['is_ensemble'],
					'total_reward': result['total_reward']
				})
			
			csv_dir = os.path.join(RESULTS_DIR, CSV_DIR)
			os.makedirs(csv_dir, exist_ok=True)
			csv_path = os.path.join(csv_dir, f'multi_env_test_{timestamp}.csv')
			
			df = pd.DataFrame(csv_data)
			df.to_csv(csv_path, index=False)
			
			print(f"✅ Salvato: {os.path.basename(csv_path)}")
			print(f"📊 {len(csv_data)} risultati, {df['model_name'].nunique()} modelli, {df['env_id'].nunique()} ambienti")
			
			print(f"\n📈 Prestazioni medie:")
			avg_by_model = df.groupby('model_name')['total_reward'].mean().sort_values(ascending=False)
			for model, avg_reward in avg_by_model.head(5).items():
				print(f"  • {model}: {avg_reward:.2f}")
			
			print(f"\n💡 Per boxplot: python scripts/generate_boxplots.py")
			
		else:
			print("❌ Nessun risultato ottenuto!")
		
	except Exception as e:
		print(f"❌ Errore: {e}")

if __name__ == "__main__":
	main()