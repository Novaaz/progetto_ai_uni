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
    
    for seed, model_list in models.items():  # Fix: era 'for seed in models'
        print(f"Analizzando i modelli per il seed: {seed}")
        
        for model_info in model_list:  # Fix: era 'for model_info in seed'
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
    
    # Generazione dei boxplot
    print(f"\n📊 Generazione boxplot da {len(all_results)} risultati...")
    plot_path = os.path.join(RESULTS_DIR, PLOTS_DIR)
    os.makedirs(plot_path, exist_ok=True)
    
    plot_boxplots(plot_path, all_results)
    
    print("\n✅ Analisi completata!")

if __name__ == "__main__":
    main()