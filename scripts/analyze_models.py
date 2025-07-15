"""Script per analizzare i modelli generati da 'generate_models.py'.
Questo script esegue l'analisi dei modelli generati, 
confrontando le loro performance e creando boxplot per visualizzare i risultati.
"""

import sys
import os
# Aggiungi la directory root al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.core import *
from src.utils.constants import *

from src.evaluation.evaluate_functions import evaluate_sac_performance
from src.visualization.boxplot import plot_boxplots

def get_models():
	"""
	Restituisce una lista di modelli disponibili nella directory dei risultati,
	organizzati per tipo e seed.
	
	Returns:
		dict: Dizionario con struttura:
		{
			"seed_0": [lista di modelli],
			"seed_1": [lista di modelli],
			...
		}
	"""

	path = os.path.join(RESULTS_DIR, MODELS_DIR)
	models = {}

	for seed_folder in os.listdir(path):
		seed_path = os.path.join(path, seed_folder)
		
		# Verifica che sia una directory
		if not os.path.isdir(seed_path):
			continue
			
		print(f"Scansionando seed: {seed_folder}")
		
		# Inizializza la lista per questo seed
		models[seed_folder] = []
		
		# Scansiona tutti i file .zip nella cartella del seed
		for model_file in os.listdir(seed_path):
			if not model_file.endswith('.zip'):
				continue
				
			model_path = os.path.join(seed_path, model_file)
			model_name = model_file.replace('.zip', '')
			
			# Aggiungi il modello alla lista del seed
			model_info = {
				'name': model_name,
				'path': model_path,
				'seed': seed_folder
			}
			
			models[seed_folder].append(model_info)
	
	for seed, model_list in models.items():
		print(f"Seed {seed}: {len(model_list)} modelli trovati")
		
	return models

def result_to_csv(result, filename):
	"""
	Salva i risultati dell'analisi in un file CSV.
	
	Args:
		result (dict): Dizionario contenente i risultati da salvare.
		filename (str): Nome del file CSV in cui salvare i risultati.
	"""
	path = os.path.join(RESULTS_DIR, CSV_DIR)
	if not os.path.exists(path):
		os.makedirs(path)
	df = pd.DataFrame(result)
	filepath = os.path.join(path, filename)
	df.to_csv(filepath, index=False)
	print(f"Risultati salvati in {filepath}")

def calculate_model_statistics(all_results):
	"""
	Calcola statistiche per ogni tipo di modello (noise level + type)
	e trova i modelli più vicini alla mediana
	
	Returns:
	dict: Statistiche per ogni combinazione model_type/noise_level
	list: Lista dei migliori modelli (più vicini alla mediana)
	"""
	# Organizza risultati per tipo di modello
	model_groups = {}
	
	for result in all_results:
		name = result['name']
		total_reward = result['total_reward']
		seed = result['seed']
		
		# Estrai info dal nome del modello
		if name.startswith('s_'):
			model_type = 'static'
			noise_level = float(name.split('_')[1])
		elif name.startswith('d_'):
			model_type = 'dynamic'
			noise_level = float(name.split('_')[1])
		else:
			continue  # Skip modelli con nome non riconosciuto
		
		# Crea chiave unica per gruppo
		group_key = f"{model_type}_{noise_level:.2f}"
		
		if group_key not in model_groups:
			model_groups[group_key] = {
				'model_type': model_type,
				'noise_level': noise_level,
				'rewards': [],
				'models': []
			}
		
		model_groups[group_key]['rewards'].append(total_reward)
		model_groups[group_key]['models'].append({
			'name': name,
			'reward': total_reward,
			'seed': seed,
			'path': None  # Sarà impostato dopo
		})
	
	# Calcola statistiche per ogni gruppo
	statistics = []
	best_models = []
	
	for group_key, group_data in model_groups.items():
		rewards = np.array(group_data['rewards'])
		
		if len(rewards) == 0:
			continue
			
		# Calcola statistiche
		stats = {
			'Model_Type': group_data['model_type'],
			'Noise_Level': group_data['noise_level'],
			'Min': np.min(rewards),
			'Q1': np.percentile(rewards, 25),
			'Median': np.median(rewards),
			'Q3': np.percentile(rewards, 75),
			'Max': np.max(rewards),
			'Count': len(rewards),
			'Mean': np.mean(rewards),
			'Std': np.std(rewards)
		}
		
		statistics.append(stats)
		
		# Trova il modello più vicino alla mediana
		median_value = stats['Median']
		closest_model = None
		min_distance = float('inf')
		
		for model in group_data['models']:
			distance = abs(model['reward'] - median_value)
			if distance < min_distance:
				min_distance = distance
				closest_model = model.copy()
		
		if closest_model:
			# Aggiungi statistiche al modello migliore
			closest_model.update({
				'model_type': group_data['model_type'],
				'noise_level': group_data['noise_level'],
				'median': median_value,
				'distance_from_median': min_distance,
				'group_count': len(rewards)
			})
			best_models.append(closest_model)
	
	return statistics, best_models

def find_model_paths(best_models):
	"""
	Trova i path fisici dei modelli migliori
	"""
	models_base_path = os.path.join(RESULTS_DIR, MODELS_DIR)
	
	for model in best_models:
		model_name = f"{model['name']}.zip"
		seed = model['seed']
		
		# Cerca nella cartella del seed
		model_path = os.path.join(models_base_path, seed, model_name)
		
		if os.path.exists(model_path):
			model['path'] = model_path
			print(f"  ✅ Trovato: {model['name']} in {seed}")
		else:
			model['path'] = None
			print(f"  ❌ Non trovato: {model_name} in {seed}")
	
	return best_models

def save_analysis_results(statistics, best_models, timestamp):
	"""
	Salva sia le statistiche che i migliori modelli
	"""
	# Salva statistiche
	stats_df = pd.DataFrame(statistics)
	stats_path = os.path.join(RESULTS_DIR, CSV_DIR, f"analysis_{timestamp}.csv")
	stats_df.to_csv(stats_path, index=False)
	print(f"📊 Statistiche salvate in: {stats_path}")
	
	# Salva migliori modelli
	best_models_df = pd.DataFrame(best_models)
	best_path = os.path.join(RESULTS_DIR, CSV_DIR, f"best_models_{timestamp}.csv")
	best_models_df.to_csv(best_path, index=False)
	print(f"🏆 Migliori modelli salvati in: {best_path}")
	
	return stats_path, best_path

def main():
	env = CityLearnEnv(**ENV_CONFIG)
	env = StableBaselines3Wrapper(NormalizedObservationWrapper(env))

	path = os.path.join(RESULTS_DIR, MODELS_DIR)
	if not os.path.exists(path):
		print(f"Directory {path} non trovata. Assicurati di aver eseguito 'generate_models.py' prima.")
		return
	
	# Caricamento dei modelli
	models = get_models()
	if not models:
		print("Nessun modello trovato")
		return
	
	# Fase di valutazione e salvataggio dei risultati
	all_results = []
	
	for seed, model_list in models.items():
		print(f"Analizzando i modelli per il seed: {seed}")
		
		for model_info in model_list:
			try:
				print(f"  Valutazione del modello: {model_info['name']}")
				sac_model = SAC.load(model_info['path'])
				
				result = evaluate_sac_performance(
					env, sac_model, 
					episode_name=model_info['name']
				)
				
				# Aggiungi info del seed
				result['seed'] = seed
				all_results.append(result)
				
				# Salva risultato individuale
				result_to_csv([result], f"{seed}_{model_info['name']}.csv")
				
				# Reset ambiente
				env.reset()
				
			except Exception as e:
				print(f"    ❌ Errore con {model_info['name']}: {e}")
	
	if not all_results:
		print("❌ Nessun risultato ottenuto!")
		return
	
	# Calcola statistiche e trova migliori modelli
	print(f"\n📊 Calcolando statistiche da {len(all_results)} risultati...")
	statistics, best_models = calculate_model_statistics(all_results)
	
	# Trova i path dei migliori modelli
	print(f"\n🔍 Cercando path per {len(best_models)} migliori modelli...")
	best_models = find_model_paths(best_models)
	
	# Salva risultati
	timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
	stats_path, best_path = save_analysis_results(statistics, best_models, timestamp)
	
	# Generazione dei boxplot
	print(f"\n📊 Generazione boxplot...")
	plot_path = os.path.join(RESULTS_DIR, PLOTS_DIR)
	os.makedirs(plot_path, exist_ok=True)
	
	plot_boxplots(plot_path, all_results)
	
	# Mostra riassunto
	print(f"\n🎯 === RIASSUNTO ===")
	print(f"   Modelli analizzati: {len(all_results)}")
	print(f"   Gruppi statistici: {len(statistics)}")
	print(f"   Migliori modelli trovati: {len([m for m in best_models if m['path']])}")
	
	# Mostra top 3 migliori modelli
	valid_models = [m for m in best_models if m['path']]
	if valid_models:
		sorted_models = sorted(valid_models, key=lambda x: x['median'], reverse=True)
		print(f"\n🏆 Top 3 migliori modelli:")
		for i, model in enumerate(sorted_models[:3], 1):
			print(f"   {i}. {model['name']} (reward: {model['reward']:.2f}, mediana gruppo: {model['median']:.2f})")
	
	print("\n✅ Analisi completata!")
	return stats_path, best_path

if __name__ == "__main__":
	main()