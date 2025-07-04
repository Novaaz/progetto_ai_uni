"""
costanti condivise tra vari moduli.
"""

# Configurazione dell'ambiente
from citylearn.reward_function import SolarPenaltyAndComfortReward

DATASET_NAME = 'citylearn_challenge_2023_phase_3_1'
DATASET_NAME_2 = 'citylearn_challenge_2023_phase_3_2'
BUILDINGS = ['Building_1', 'Building_2', 'Building_3']

ACTIVE_OBSERVATIONS = ['hour', 'electricity_pricing', 'solar_generation', 'electricity_pricing', 
                      'net_electricity_consumption','electrical_storage_soc',
                      'indoor_dry_bulb_temperature', 'carbon_intensity']
ACTIVE_ACTIONS = ['electrical_storage', 'cooling_storage_soc', 'heating_storage_soc', 'cooling_or_heating_device']
CENTRAL_AGENT = True

# Altre costanti condivise
EPISODES = 35
UPDATE_FREQ = 75
BATCH_SIZE = 256
LEARNING_STARTS = 150

ENV_CONFIG = {
    "schema": 'citylearn_challenge_2023_phase_1',
    "central_agent": CENTRAL_AGENT,
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