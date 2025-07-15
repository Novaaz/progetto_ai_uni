from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ensemble import WeightedEnsemble
from src.training.train_functions import train_ensemble_online_with_experience_replay
from src.evaluation.evaluate_functions import evaluate_sac_performance
from src.utils.constants import *
from src.utils.core import *

def load_best_models_from_analysis(analysis_dir, config=None):
	"""
	Carica i migliori modelli dal file di analisi usando configurazione
	
	Parameters:
	analysis_dir: str - Directory con i file di analisi
	config: dict - Configurazione (se None, usa ENSEMBLE_CONFIG)
	
	Returns:
	list: Lista di path ai modelli selezionati
	list: Info sui modelli selezionati
	"""
	if config is None:
		config = ENSEMBLE_CONFIG
	
	# Estrai parametri dalla configurazione
	top_n = config['top_n']
	noise_diversity = config['noise_diversity']
	min_performance_threshold = config['min_performance_threshold']
	
	print(f"📊 Configurazione selezione modelli:")
	print(f"   Top N: {top_n}")
	print(f"   Diversità rumore: {noise_diversity}")
	print(f"   Soglia minima: {min_performance_threshold}")
	
	# Trova l'ultimo file best_models
	csv_files = [f for f in os.listdir(analysis_dir) if f.startswith('best_models_') and f.endswith('.csv')]
	if not csv_files:
		raise FileNotFoundError("Nessun file best_models_*.csv trovato!")
	
	latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(analysis_dir, x)))
	csv_path = os.path.join(analysis_dir, latest_csv)
	
	print(f"📊 Caricando migliori modelli da: {latest_csv}")
	
	# Carica i dati
	df = pd.read_csv(csv_path)
	print(f"   Modelli disponibili: {len(df)}")
	
	# Filtra per performance minima
	if min_performance_threshold is not None:
		df = df[df['median'] >= min_performance_threshold]
		print(f"   Dopo filtro performance (>= {min_performance_threshold}): {len(df)}")
	
	# Filtra solo modelli con path valido
	df = df[df['path'].notna()]
	df = df[df['path'] != '']
	print(f"   Con path validi: {len(df)}")
	
	if len(df) == 0:
		raise ValueError("Nessun modello valido trovato!")
	
	# Selezione modelli
	if noise_diversity:
		# Prendi il migliore per ogni combinazione type/noise_level
		selected_models = []
		
		for _, group in df.groupby(['model_type', 'noise_level']):
			# Ordina per mediana e prendi il migliore
			best_in_group = group.loc[group['median'].idxmax()]
			selected_models.append(best_in_group)
		
		# Se abbiamo meno di top_n, aggiungi i migliori rimanenti
		if len(selected_models) < top_n:
			selected_indices = [model.name for model in selected_models]
			remaining_df = df[~df.index.isin(selected_indices)]
			additional = remaining_df.nlargest(top_n - len(selected_models), 'median')
			selected_models.extend([additional.iloc[i] for i in range(len(additional))])
	else:
		# Prendi semplicemente i top_n migliori per mediana
		selected_models = [df.nlargest(top_n, 'median').iloc[i] for i in range(min(top_n, len(df)))]
	
	# Prepara output
	model_paths = []
	model_infos = []
	
	for model_record in selected_models[:top_n]:  # Limita a top_n
		if os.path.exists(model_record['path']):
			model_paths.append(model_record['path'])
			model_infos.append({
				'name': model_record['name'],
				'noise_level': model_record['noise_level'],
				'type': model_record['model_type'],
				'expected_performance': model_record['median'],
				'actual_performance': model_record['reward'],
				'path': model_record['path'],
				'seed': model_record['seed']
			})
			print(f"  ✅ Selezionato: {model_record['name']} ({model_record['model_type']}, noise={model_record['noise_level']:.3f}, mediana={model_record['median']:.2f})")
		else:
			print(f"  ❌ Path non valido: {model_record['path']}")
	
	if not model_paths:
		raise ValueError("Nessun modello con path valido trovato!")
	
	print(f"🎯 Selezionati {len(model_paths)} modelli per ensemble")
	return model_paths, model_infos

def prepare_ensemble_metadata(config, model_infos, training_results=None):
	"""
	Prepara i metadati per l'ensemble
	
	Parameters:
	config: dict - Configurazione ensemble
	model_infos: list - Informazioni sui modelli
	training_results: dict - Risultati del training (opzionale)
	
	Returns:
	dict - Metadati completi
	"""
	# Copia template base
	metadata = ENSEMBLE_METADATA_TEMPLATE.copy()
	
	# Aggiungi configurazione training
	metadata['training_config'] = {
		'max_episodes': config['max_episodes'],
		'weight_update_freq': config['weight_update_freq'],
		'ensemble_method': config['ensemble_method'],
		'learning_rate': config['learning_rate'],
		'update_method': config['update_method'],
		'temperature': config.get('temperature', 0.5)
	}
	
	# Aggiungi info sui modelli
	metadata['model_selection'] = {
		'top_n': config['top_n'],
		'noise_diversity': config['noise_diversity'],
		'min_performance_threshold': config['min_performance_threshold'],
		'selected_models': len(model_infos)
	}
	
	# Aggiungi dettagli modelli
	metadata['models_info'] = []
	for i, info in enumerate(model_infos):
		metadata['models_info'].append({
			'index': i,
			'name': info['name'],
			'type': info['type'],
			'noise_level': info['noise_level'],
			'expected_performance': info['expected_performance'],
			'seed': info['seed']
		})
	
	# Aggiungi risultati training se disponibili
	if training_results:
		metadata['training_results'] = training_results
	
	# Aggiungi timestamp
	metadata['creation_timestamp'] = datetime.now().isoformat()
	
	return metadata

def main():
	print("🎯 === ENSEMBLE TRAINING ONLINE ===")
	
	# Mostra configurazione
	print(f"\n📋 Configurazione ensemble:")
	for key, value in ENSEMBLE_CONFIG.items():
		print(f"   {key}: {value}")
	
	env = CityLearnEnv(**ENV_CONFIG)
	env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
	
	results_path = os.path.join(RESULTS_DIR, CSV_DIR)
	print("\n🤖 Caricando migliori modelli dall'analisi...")
	
	try:
		model_paths, model_infos = load_best_models_from_analysis(results_path, ENSEMBLE_CONFIG)
	except Exception as e:
		print(f"❌ Errore nel caricamento modelli: {e}")
		return None
	
	print("\n🔧 Creando ensemble...")
	ensemble = WeightedEnsemble(model_paths, env)
	
	# Valutazione iniziale
	initial_results = None
	if ENSEMBLE_CONFIG['evaluate_initial']:
		print("\n📊 Valutando ensemble iniziale...")
		initial_results = evaluate_sac_performance(env, ensemble, "Ensemble_Initial")

	# Training con experience replay
	print("\n🔧 Training ensemble con experience replay...")
	trained_ensemble = train_ensemble_online_with_experience_replay(
		model_paths,
		env,
		max_episodes=int(ENSEMBLE_CONFIG['max_episodes']/3),
		weight_update_freq=ENSEMBLE_CONFIG['weight_update_freq'],
		ensemble_method=ENSEMBLE_CONFIG['ensemble_method']
	)
	
	# Valutazione finale
	final_results = None
	if ENSEMBLE_CONFIG['evaluate_final']:
		print("\n📊 Valutando ensemble finale...")
		final_results = evaluate_sac_performance(env, trained_ensemble, "Ensemble_Final")
	
	# Prepara risultati training
	training_results = {}
	if initial_results:
		training_results['initial_performance'] = initial_results['total_reward']
	if final_results:
		training_results['final_performance'] = final_results['total_reward']
	if initial_results and final_results:
		training_results['improvement'] = final_results['total_reward'] - initial_results['total_reward']
	
	# Mostra risultati
	print("\n🎯 === RISULTATI ===")
	if initial_results:
		print(f"Performance iniziale: {initial_results['total_reward']:.2f}")
	if final_results:
		print(f"Performance finale:   {final_results['total_reward']:.2f}")
	if initial_results and final_results:
		print(f"Miglioramento:        {final_results['total_reward'] - initial_results['total_reward']:.2f}")
	
	# Mostra modelli nell'ensemble
	print(f"\n📋 Modelli nell'ensemble:")
	for i, info in enumerate(model_infos):
		weight = trained_ensemble.weights[i]
		print(f"  {i+1}. {info['name']} (peso: {weight:.3f}, perf_attesa: {info['expected_performance']:.2f})")
	
	# Mostra modello dominante
	best_model, best_info = trained_ensemble.get_best_model()
	print(f"\nModello dominante finale:")
	print(f"  Tipo: {best_info['type']}")
	print(f"  Noise: {best_info['noise_level']:.3f}")
	print(f"  Peso: {trained_ensemble.weights[best_info['index']]:.3f}")
	
	# Salva ensemble se richiesto
	if ENSEMBLE_CONFIG['save_ensemble']:
		print(f"\n💾 Salvando ensemble...")
		
		# Prepara metadati
		metadata = prepare_ensemble_metadata(ENSEMBLE_CONFIG, model_infos, training_results)
		
		# Prepara path di salvataggio
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		save_dir = os.path.join(RESULTS_DIR, ENSEMBLE_CONFIG['save_directory'])
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f'ensemble_{timestamp}')
		
		# Salva ensemble
		try:
			saved_path = trained_ensemble.save_ensemble(
				save_path, 
				include_models=ENSEMBLE_CONFIG['include_models'], 
				metadata=metadata
			)
			
			# Mostra info salvaggio
			save_info = trained_ensemble.get_save_info()
			print(f"   Path: {saved_path}")
			print(f"   Modelli inclusi: {ENSEMBLE_CONFIG['include_models']}")
			print(f"   Modelli validi: {save_info['num_valid_models']}/{save_info['num_total_models']}")
			print(f"   Dimensione file: {os.path.getsize(saved_path)/1024/1024:.1f} MB")
			
		except Exception as e:
			print(f"   ❌ Errore durante salvataggio: {e}")
	
	return trained_ensemble

if __name__ == "__main__":
	ensemble = main()
