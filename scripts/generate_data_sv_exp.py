"""
Script semplice per testare modelli single-variable su un ambiente singolo.
Carica modelli direttamente dai file .zip e genera CSV con i risultati.
"""

import os
import sys
import pandas as pd
import numpy as np
import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.constants import *
from src.utils.core import *
from src.evaluation.evaluate_functions import evaluate_sac_performance
from stable_baselines3 import SAC

def load_single_variable_models_from_files(analysis_dir):
    """
    Carica i modelli single-variable direttamente dai file .zip nella directory models.
    """
    print("📂 Caricando modelli single-variable dai file...")
    
    models_dir = os.path.join(analysis_dir, "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Directory models non trovata in {analysis_dir}")
    
    # Trova tutti i file .zip nella directory models
    model_files = glob.glob(os.path.join(models_dir, "*.zip"))
    print(f"   Trovati {len(model_files)} file modello")
    
    models = []
    
    for model_path in model_files:
        filename = os.path.basename(model_path)
        
        try:
            # Estrai info dal filename: var_<nome>_s_<noise>_seed<seed>.zip
            parts = filename.replace('.zip', '').split('_')
            
            if len(parts) >= 5:
                var_name = '_'.join(parts[1:-3])  # Nome variabile (può contenere _)
                static_dynamic = parts[-3]  # 's' o 'd'
                noise_level = float(parts[-2])
                seed = int(parts[-1].replace('seed', ''))
                
                # Mappa nome variabile a indice (se necessario)
                variable_index = get_variable_index_from_name(var_name)
                
                models.append({
                    'name': filename.replace('.zip', ''),
                    'path': model_path,
                    'variable_name': var_name,
                    'variable_index': variable_index,
                    'noise_level': noise_level,
                    'seed': seed,
                    'model_type': 'static' if static_dynamic == 's' else 'dynamic'
                })
            else:
                print(f"   ⚠️ Filename non riconosciuto: {filename}")
                
        except Exception as e:
            print(f"   ⚠️ Errore parsing {filename}: {e}")
            continue
    
    print(f"   ✅ Caricati {len(models)} modelli validi")
    return models

def get_variable_index_from_name(var_name):
    """
    Mappa il nome della variabile al suo indice.
    Riusa la logica da get_citylearn_variable_info().
    """
    variable_info = {
        "outdoor_dry_bulb_temperature": 3,
        "outdoor_relative_humidity": 4, 
        "diffuse_solar_irradiance": 5,
        "direct_solar_irradiance": 6,
        "carbon_intensity": 7,
        "non_shiftable_load": 8,
        "solar_generation": 9,
        "electrical_storage_soc": 10,
        "net_electricity_consumption": 11,
        "electricity_pricing": 12,
        "cooling_storage_soc": 13,
        "dhw_storage_soc": 14,
        "indoor_dry_bulb_temperature": 15,
        "average_unmet_cooling_setpoint_difference": 16,
        "indoor_relative_humidity": 17,
        "cooling_demand": 18,
        "dhw_demand": 19,
        "cooling_device_efficiency": 20,
        "dhw_device_efficiency": 21
    }
    
    return variable_info.get(var_name, -1)

def create_test_environment():
    """
    Crea un ambiente di test pulito.
    """
    print("🌍 Creando ambiente di test pulito...")
    env = CityLearnEnv(**ENV_CONFIG)
    env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
    print("   ✅ Ambiente creato")
    return env

def test_models_on_environment(models, test_env):
    """
    Testa tutti i modelli sull'ambiente specificato.
    """
    print(f"\n🧪 Testing {len(models)} modelli...")
    
    results = []
    
    for i, model_info in enumerate(models, 1):
        model_name = model_info['name']
        print(f"  {i:2d}/{len(models)} • {model_name}...", end=' ')
        
        try:
            # Reset ambiente per ogni test
            test_env.reset()
            
            # Carica il modello
            model = SAC.load(model_info['path'], test_env)
            
            # Valuta usando la funzione esistente
            result = evaluate_sac_performance(
                test_env, 
                model, 
                episode_name=f"test_{model_name}"
            )
            
            # Aggiungi metadati del modello
            result.update({
                'model_name': model_name,
                'model_path': model_info['path'],
                'variable_name': model_info['variable_name'],
                'variable_index': model_info['variable_index'],
                'noise_level': model_info['noise_level'],
                'seed': model_info['seed'],
                'model_type': model_info['model_type'],
                'test_reward': result['total_reward']
            })
            
            results.append(result)
            print(f"✅ Reward: {result['total_reward']:.2f}")
            
        except Exception as e:
            print(f"❌ Errore: {str(e)}")
            # Aggiungi risultato con errore
            error_result = {
                'model_name': model_name,
                'model_path': model_info['path'],
                'variable_name': model_info['variable_name'],
                'variable_index': model_info['variable_index'],
                'noise_level': model_info['noise_level'],
                'seed': model_info['seed'],
                'model_type': model_info['model_type'],
                'total_reward': -999999,  # Indica errore
                'test_reward': -999999,
                'error': str(e)
            }
            results.append(error_result)
    
    return results

def save_results_to_csv(results, output_dir):
    """
    Salva i risultati in CSV.
    """
    print(f"\n💾 Salvando risultati...")
    
    # Prepara dati per CSV
    csv_data = []
    for result in results:
        csv_data.append({
            'model_name': result['model_name'],
            'variable_name': result['variable_name'],
            'variable_index': result['variable_index'],
            'noise_level': result['noise_level'],
            'seed': result['seed'],
            'model_type': result['model_type'],
            'total_reward': result['total_reward'],
            'test_reward': result.get('test_reward', result['total_reward']),
            'model_path': result['model_path'],
            'error': result.get('error', ''),
            # Aggiungi altre metriche se disponibili
            'name': result.get('name', result['model_name']),
            'episode_rewards': result.get('episode_rewards', []),
            'episode_lengths': result.get('episode_lengths', [])
        })
    
    # Salva CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'single_var_test_{timestamp}.csv')
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    print(f"   ✅ Salvato: {os.path.basename(csv_path)}")
    print(f"   📊 {len(csv_data)} risultati salvati")
    
    return csv_path

def print_summary(results):
    """
    Stampa un riassunto dei risultati.
    """
    print(f"\n📈 RIASSUNTO RISULTATI:")
    print("-" * 40)
    
    # Statistiche generali
    valid_results = [r for r in results if r['total_reward'] > -999999]
    error_count = len(results) - len(valid_results)
    
    print(f"   Modelli testati: {len(results)}")
    print(f"   Successi: {len(valid_results)}")
    print(f"   Errori: {error_count}")
    
    if valid_results:
        rewards = [r['total_reward'] for r in valid_results]
        print(f"   Reward medio: {np.mean(rewards):.2f}")
        print(f"   Reward migliore: {np.max(rewards):.2f}")
        print(f"   Reward peggiore: {np.min(rewards):.2f}")
        
        # Top 3 modelli
        sorted_results = sorted(valid_results, key=lambda x: x['total_reward'], reverse=True)
        print(f"\n🏆 Top 3 modelli:")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"   {i}. {result['variable_name']} (noise: {result['noise_level']:.3f}) → {result['total_reward']:.2f}")
        
        # Statistiche per variabile
        print(f"\n📊 Prestazioni per variabile:")
        var_stats = {}
        for result in valid_results:
            var_name = result['variable_name']
            if var_name not in var_stats:
                var_stats[var_name] = []
            var_stats[var_name].append(result['total_reward'])
        
        for var_name, rewards in var_stats.items():
            avg_reward = np.mean(rewards)
            print(f"   {var_name}: {avg_reward:.2f} (n={len(rewards)})")

def main():
    """
    Funzione principale - carica modelli dai file e genera dati.
    """
    print("🧪 === SINGLE VARIABLE MODELS TESTING ===")
    print("="*50)
    
    # Input utente - semplificato
    analysis_input = input("Directory analisi single-variable (o ENTER per cercare l'ultima): ").strip()
    
    try:
        # 1. TROVA DIRECTORY DI ANALISI
        if not analysis_input:
            # Cerca l'ultima directory di analisi
            base_results = os.path.join(RESULTS_DIR, "single_variable")
            if os.path.exists(base_results):
                recent_dirs = sorted([d for d in os.listdir(base_results) 
                                    if d.startswith('analysis_')], reverse=True)
                if recent_dirs:
                    analysis_dir = os.path.join(base_results, recent_dirs[0])
                    print(f"📂 Usando directory più recente: {os.path.basename(analysis_dir)}")
                else:
                    print("❌ Nessuna directory di analisi trovata!")
                    return
            else:
                print("❌ Directory base single_variable non trovata!")
                return
        else:
            analysis_dir = analysis_input
            if not os.path.exists(analysis_dir):
                print(f"❌ Directory {analysis_dir} non trovata!")
                return
        
        # 2. CARICA MODELLI DAI FILE .ZIP
        models = load_single_variable_models_from_files(analysis_dir)
        if not models:
            print("❌ Nessun modello trovato!")
            return

        env = create_test_environment()

        # 3. ESEGUI ANALISI SUI MODELLI
        results = test_models_on_environment(models, env)
        if not results:
            print("❌ Nessun risultato trovato!")
            return

        # 4. SALVA RISULTATI
        save_results_to_csv(results, analysis_dir)

    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()