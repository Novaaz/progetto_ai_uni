"""
costanti condivise tra vari moduli.
"""

from citylearn.reward_function import SolarPenaltyAndComfortReward

EPISODES = 35
UPDATE_FREQ = 75
BATCH_SIZE = 256
LEARNING_STARTS = 150

ENV_CONFIG = {
	"schema": 'citylearn_challenge_2023_phase_1',
	"central_agent": True,
	'reward_function': SolarPenaltyAndComfortReward,
}

SAC_KWARGS = {
	"learning_rate": 0.0003,
	"tau": 0.005,
	"gamma": 0.99,
	"buffer_size": 10000,
	"batch_size": BATCH_SIZE,
	"learning_starts": LEARNING_STARTS,
}

FINETUNING_KWARGS = {
	"learning_rate": 0.00003, #un ordine inferiore
	"tau": 0.005,
	"gamma": 0.99,
	"buffer_size": 1000,
	"batch_size": 256,
}

#costanti degli script
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
CSV_DIR = 'csv'
PLOTS_DIR = 'plots'
SEEDS_FILE = 'seeds.txt'

# === ENSEMBLE CONFIGURATION ===
ENSEMBLE_CONFIG = {
	# Selezione modelli
	'top_n': 8,
	'noise_diversity': True,
	'min_performance_threshold': -50000,
	
	# Training parametri
	'max_episodes': EPISODES,  # Episodi per training ensemble
	'weight_update_freq': 5,           # Ogni quanti episodi aggiornare pesi
	'ensemble_method': 'weighted_average',  # 'best_only'
	
	# Aggiornamento pesi
	'learning_rate': 0.15,
	'update_method': 'softmax',  # 'exponential', 'softmax', 'linear'
	'temperature': 0.5,              # Per softmax
	
	# Salvataggio
	'save_ensemble': True,
	'include_models': True,
	'save_directory': 'ensembles',
	
	# Valutazione
	'evaluate_initial': True,
	'evaluate_final': True,
	'evaluation_max_steps': 100,
	
	# Training online parametri
	'online_training': {
		'experience_steps': 100,      # Ogni quanti step raccogliere esperienze
		'training_episodes': 5,       # Episodi di training per batch
		'final_training_episodes': 20, # Episodi di training finale
		'buffer_size': 10000,         # Dimensione buffer esperienze
	},
}

ENSEMBLE_METADATA_TEMPLATE = {
	'project': 'CityLearn AI Project',
	'algorithm': 'SAC Ensemble',
	'environment': 'CityLearn',
	'description': 'Weighted ensemble of SAC models with different noise levels',
	'author': 'Leonardo Novazzi',
	'version': '1.0.0'
}