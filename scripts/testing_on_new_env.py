from functools import partial
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_functions import add_noise_to_observations
from src.utils.constants import *
from src.utils.core import *
from src.models.ensemble import load_ensemble, list_saved_ensembles
from src.evaluation.evaluate_functions import evaluate_sac_performance
from stable_baselines3 import SAC
from gymnasium.wrappers import TransformObservation

def load_top_models(analysis_dir, top_n=3):
	"""Carica i top N modelli dall'analisi, assicurandosi che ci sia sempre un modello con noise 0.0"""
	csv_files = [f for f in os.listdir(analysis_dir) if f.startswith('best_models_') and f.endswith('.csv')]
	if not csv_files:
		raise FileNotFoundError("Nessun file best_models_*.csv trovato!")
	
	latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(analysis_dir, x)))
	df = pd.read_csv(os.path.join(analysis_dir, latest_csv))
	
	baseline_models = df[df['noise_level'] == 0.0]
	top_models = df.nlargest(top_n, 'median')
	
	# Verifica se c'è già un modello con noise 0.0 nei top N
	has_baseline = any(model['noise_level'] == 0.0 for _, model in top_models.iterrows())
	
	if not has_baseline and not baseline_models.empty:
		print(f"  🔄 Nessun modello baseline (noise=0.0) nei top {top_n}, aggiungendolo...")
		
		best_baseline = baseline_models.nlargest(1, 'median').iloc[0]
		
		top_models = top_models.iloc[:-1]  # Rimuovi l'ultimo (peggiore)
		top_models = pd.concat([top_models, best_baseline.to_frame().T], ignore_index=True)
		
		# Riordina per mediana
		top_models = top_models.sort_values('median', ascending=False)
		
		print(f"    ✅ Sostituito modello peggiore con baseline: {best_baseline['name']} (median: {best_baseline['median']:.2f})")
	elif has_baseline:
		print(f"  ✅ Modello baseline già presente nei top {top_n}")
	else:
		print(f"  ⚠️  Nessun modello baseline (noise=0.0) trovato nel dataset")
	
	models = []
	for _, row in top_models.iterrows():
		if os.path.exists(row['path']):
			models.append({
				'name': row['name'],
				'path': row['path'],
				'expected_performance': row['median'],
				'type': row['model_type'],
				'noise': row['noise_level']
			})
		else:
			print(f"  ⚠️  Modello non trovato: {row['path']}")
	
	print(f"  📊 Modelli selezionati:")
	for i, model in enumerate(models, 1):
		noise_info = f"(noise: {model['noise']:.2f})" if model['noise'] > 0 else "(baseline)"
		print(f"    {i}. {model['name']} {noise_info} - Expected: {model['expected_performance']:.2f}")
	
	return models

def plot_comparison(results):
	"""Plotta confronto prestazioni"""
	_, ax = plt.subplots(1, 1, figsize=(12, 8))
	
	if 'step_rewards' in results[0]:
		colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
		
		for i, result in enumerate(results):
			if 'step_rewards' in result:
				color = colors[i % len(colors)]
				# Aggiungi reward finale nella legenda
				label = f"{result['name']} (Final: {result['total_reward']:.2f})"
				ax.plot(result['step_rewards'], 
					   label=label, 
					   linewidth=2, 
					   color=color,
					   alpha=0.8)
		
		ax.set_title('Confronto Reward per Step', fontsize=16, fontweight='bold')
		ax.set_xlabel('Step', fontsize=12)
		ax.set_ylabel('Reward', fontsize=12)
		ax.legend(loc='best', fontsize=10)
		ax.grid(True, alpha=0.3)

	else:
		names = [r['name'] for r in results]
		rewards = [r['total_reward'] for r in results]
		colors = ['blue' if 'Ensemble' in name else 'red' for name in names]
		
		ax.bar(names, rewards, color=colors, alpha=0.7)
		ax.set_title('Confronto Prestazioni Totali', fontsize=16, fontweight='bold')
		ax.set_ylabel('Total Reward', fontsize=12)
		ax.tick_params(axis='x', rotation=45)
		ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	save_path = os.path.join(RESULTS_DIR, f'comparison_{timestamp}.png')
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()
	
	print(f"📊 Grafico salvato: {save_path}")

def main():
	"""Test e confronto modelli"""
	print("🧪 === TEST & COMPARISON ===")
	
	env = CityLearnEnv(**ENV_CONFIG)
	env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
	env = TransformObservation(env, partial(
			add_noise_to_observations, 
			noise_type='gaussian',
			dinamic_noise=False,
			name='random_env',
			noise_level=0.75,
			noise_mean=0.25,
		))
	
	results = []
	
	print("\n🤖 Caricando top modelli...")
	try:
		analysis_dir = os.path.join(RESULTS_DIR, CSV_DIR)
		top_models = load_top_models(analysis_dir, top_n=5)
		
		for model_info in top_models:
			print(f"  🔄 Testando {model_info['name']}...")
			model = SAC.load(model_info['path'], env)
			result = evaluate_sac_performance(env, model, model_info['name'])
			results.append(result)
			print(f"    Reward: {result['total_reward']:.2f}")
			
	except Exception as e:
		print(f"❌ Errore caricamento modelli: {e}")
	
	print("\n🔧 Caricando ensemble...")
	try:
		ensembles_dir = os.path.join(RESULTS_DIR, 'ensembles')
		ensembles = list_saved_ensembles(ensembles_dir)
		
		if ensembles:
			latest_ensemble = max(ensembles, key=lambda x: x['creation_time'])
			print(f"  📂 Caricando: {latest_ensemble['filename']}")
			
			ensemble = load_ensemble(latest_ensemble['filepath'], env)
			result = evaluate_sac_performance(env, ensemble, f"Ensemble_{len(ensembles)}")
			results.append(result)
			print(f"    Reward: {result['total_reward']:.2f}")
		else:
			print("  ❌ Nessun ensemble trovato")
			
	except Exception as e:
		print(f"❌ Errore caricamento ensemble: {e}")
	
	if results:
		print(f"\n📊 Confrontando {len(results)} modelli...")
		
		print("\n🏆 === RISULTATI ===")
		for result in sorted(results, key=lambda x: x['total_reward'], reverse=True):
			print(f"  {result['name']}: {result['total_reward']:.2f}")
		
		plot_comparison(results)
		
	else:
		print("❌ Nessun risultato da confrontare")

if __name__ == "__main__":
	main()