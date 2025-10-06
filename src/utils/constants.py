"""
costanti condivise tra vari moduli.
"""

from citylearn.reward_function import SolarPenaltyAndComfortReward

EXCLUDED_NOISE_LEVELS = []  
EPISODES = 35
UPDATE_FREQ = 5
BATCH_SIZE = 512
LEARNING_STARTS = 256

ENV_CONFIG = {
	"schema": 'citylearn_challenge_2023_phase_2_online_evaluation_1',
	"central_agent": True,
	'reward_function': SolarPenaltyAndComfortReward,
}

SAC_KWARGS = {
	"learning_rate": 0.0003,
	"tau": 0.005,
	"gamma": 0.99,
	"buffer_size": 100000,
	"batch_size": BATCH_SIZE,
	"learning_starts": LEARNING_STARTS,
}

FINETUNING_KWARGS = {
	"learning_rate": 0.0001,
	"tau": 0.005,
	"gamma": 0.99,
	"buffer_size": 100000,
	"batch_size": BATCH_SIZE,
	"learning_starts": LEARNING_STARTS,
	"train_freq": UPDATE_FREQ,
}

RESULTS_DIR = 'results'
MODELS_DIR = 'models_dyn'
CSV_DIR = 'csv'
PLOTS_DIR = 'plots'
SEEDS_FILE = 'seeds.txt'