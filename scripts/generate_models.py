"""generazione di diiferenti modelli con seed e std differenti
 per testare quale di questi performa meglio e quanto dipendono dalla randomicità.
I dati vengono salvati su dei csv per una più facile elaborazione e visualizzazione.
Vengono generati dei boxplot per il confronto finale tra i modelli.
I vari modelli vengono anche salvati in una cartella per un uso futuro."""

import random
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train_functions import *
from src.utils.constants import *

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
    n = input("Quanti semi vuoi generare? (default 5): ") or "5"
    n = int(n)
    seeds = initialize_res_dir(n)
    
    m = input("Quanti modelli per seed? (default 5 = 5s + 4d): ") or "5"
    m = int(m)
    
    min_noise = float(input("Rumore minimo (default 0.0): ") or "0.0")
    max_noise = float(input("Rumore massimo (default 0.8): ") or "0.8")
    
    # Genera livelli di rumore una sola volta
    noise_values = generate_noise_levels(m, min_noise, max_noise)
    
    print(f"\n🎯 Generando modelli:")
    print(f"   Seeds: {len(seeds)}")
    print(f"   Livelli di rumore: {[f'{n:.3f}' for n in noise_values]}")
    print(f"   Modelli per seed: {m} static + {m-1} dynamic")
    
    # Pre-crea tutti gli ambienti di rumore una volta sola
    print(f"\n🏗️  Pre-creando ambienti di rumore...")
    for noise_std in noise_values:
        if noise_std >= 0.0:  # Solo per static con rumore
            get_or_create_noise_environment(noise_std, 'gaussian', 0.0, False)
        if noise_std > 0.0:  # Solo per dynamic con rumore  
            get_or_create_noise_environment(noise_std, 'gaussian', 0.0, True)
    
    for seed in seeds:
        print(f"\n📦 Seed {seed}:")
        
        # Modelli static
        for i in range(m):
            noise_std = noise_values[i]
            model, _, _ = train_model_with_noise(
                noise_std=noise_std, 
                dinamic_noise=False, 
                seed=seed, 
                model_id=f"static_{i}"
            )
            
            if model is not None:
                model_path = os.path.join(RESULTS_DIR, MODELS_DIR, str(seed), f"s_{noise_std:.3f}.zip")
                model.save(model_path)
                print(f"    ✅ Static {noise_std:.3f}: {model_path}")
        
        # Modelli dynamic
        for i in range(m):
            if i == 0:  # Salta noise=0 per dynamic
                continue
            
            noise_std = noise_values[i]
            model, _, _ = train_model_with_noise(
                noise_std=noise_std, 
                dinamic_noise=True, 
                seed=seed, 
                model_id=f"dynamic_{i}"
            )
            
            if model is not None:
                model_path = os.path.join(RESULTS_DIR, MODELS_DIR, str(seed), f"d_{noise_std:.3f}.zip")
                model.save(model_path)
                print(f"    ✅ Dynamic {noise_std:.3f}: {model_path}")
    
    clean_dead_dir()
    print("\n🎉 Generazione completata!")

if __name__ == "__main__":
	main()