from citylearn.agents.rbc import RBC, HourRBC, OptimizedRBC
from citylearn.building import Building, LSTMDynamicsBuilding
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.reward_function import SolarPenaltyAndComfortReward
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper, NormalizedSpaceWrapper
from stable_baselines3 import SAC
from typing import Any, Mapping, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import os
from .constants import *

def clean_dead_dir():
	"""Rimuove le directory vuote nella cartella dei risultati."""
	path = os.path.join(RESULTS_DIR, MODELS_DIR)
	if not os.path.exists(path):
		return
	for seed_folder in os.listdir(path):
		seed_path = os.path.join(path, seed_folder)
		if not os.path.isdir(seed_path):
			continue
		if not os.listdir(seed_path):  # Se la cartella è vuota
			os.rmdir(seed_path)  # Rimuove la cartella vuota
			print(f"Rimossa cartella vuota: {seed_path}")
	# Rimuove i seed dal file seeds.txt se le loro cartelle sono state eliminate
	seeds_path = os.path.join(RESULTS_DIR, "seeds.txt")
	if os.path.exists(seeds_path):
		# Legge i seed esistenti
		with open(seeds_path, 'r') as f:
			seeds = f.read().splitlines()
		
		# Filtra i seed le cui cartelle esistono ancora
		remaining_seeds = []
		for seed in seeds:
			seed_dir = os.path.join(RESULTS_DIR, MODELS_DIR, f"seed_{seed}")
			if os.path.exists(seed_dir) and os.path.isdir(seed_dir):
				remaining_seeds.append(seed)
		
		# Scrive il nuovo elenco di seed
		with open(seeds_path, 'w') as f:
			f.write('\n'.join(remaining_seeds))
		
		print(f"Aggiornato il file seeds.txt: rimossi {len(seeds) - len(remaining_seeds)} seed")
