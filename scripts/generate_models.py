"""generazione di diiferenti modelli con seed e std differenti
 per testare quale di questi performa meglio e quanto dipendono dalla randomicità.
I dati vengono salvati su dei csv per una più facile elaborazione e visualizzazione.
Vengono generati dei boxplot per il confronto finale tra i modelli.
I vari modelli vengono anche salvati in una cartella per un uso futuro."""

import random
import os
import pandas as pd
from ..src.training.train_functions import *
from ..src.utils.constants import *

def create_seeds(n):
	"""Genera una lista di semi casuali per la riproducibilità degli esperimenti.
	Args:
		n (int): Numero di semi da generare.
	Returns:
		list: Lista di semi casuali."""
	seeds = []
	old_seeds = []
	path = os.path.join(RESULTS_DIR, MODELS_DIR, SEEDS_FILE)
	if os.path.exists(path):
		with open(path, 'r') as f:
			old_seeds = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
	for _ in range(n):
		seed = random.randint(0, 10000)
		if seed not in seeds and seed not in old_seeds:
			seeds.append(seed)
	return seeds

def initialize_res_dir(n):
	"""Crea le directory ed i file necessari per salvare i risultati.
	Args:
		n (int): Numero di semi da generare.
	Returns:
		list: Lista di semi casuali."""
	if not os.path.exists(RESULTS_DIR):
		os.makedirs(RESULTS_DIR)
	if not os.path.exists(os.path.join(RESULTS_DIR, MODELS_DIR)):
		os.makedirs(os.path.join(RESULTS_DIR, MODELS_DIR))
	path = os.path.join(RESULTS_DIR, MODELS_DIR, SEEDS_FILE)
	if not os.path.exists(path):
		with open(path, 'w') as f:
			f.write("# Seeds used for experiments\n")
	seeds = create_seeds(n)	
	for seed in seeds:
		with open(path, 'a') as f:
			f.write(f"{seed}\n")
		temp = os.path.join(RESULTS_DIR, MODELS_DIR, str(seed))
		if not os.path.exists(temp):
			os.makedirs(temp)
	return seeds

def save_data_to_csv(data, file_path):
	"""Salva i dati in un file CSV.
	Args:
		data (list): Lista di dizionari contenenti i dati da salvare.
		file_path (str): Percorso del file CSV dove salvare i dati."""
	df = pd.DataFrame(data)
	df.to_csv(file_path, index=False)
	print(f"Data saved to {file_path}")

def main():
	n = input("Quanti semi vuoi generare? (default 5)") or 5
	seeds = initialize_res_dir(n)
	m = input("Quante esecuzioni vuoi fare per ogni seed? (default 16 -> 8 normali e 8 dinamici)") or 8
	noise_values = generate_noise_levels(m,0.0,0.9)
	for seed in seeds:
		for i in range(m):
			noise_std = noise_values[i]
			model, _, _ = train_model_with_noise(noise_std=noise_std, dinamic_noise=False, seed=seed, model_id="model_"+str(i))
			model_path = os.path.join(RESULTS_DIR, MODELS_DIR, str(seed), f"s_{noise_std:.2f}.zip")
			model.save(model_path)
		for i in range(m):
			if i == 0:
				continue
			noise_std = noise_values[i]
			model, _, _ = train_model_with_noise(noise_std=noise_std, dinamic_noise=True, seed=seed, model_id="model_"+str(i+m))
			model_path = os.path.join(RESULTS_DIR, MODELS_DIR, str(seed), f"d_{noise_std:.2f}.zip")
			model.save(model_path)

if __name__ == "__main__":
	main()