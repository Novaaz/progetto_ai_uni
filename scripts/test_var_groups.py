"""
Script per testare modello con rumore intelligente: alto dove conviene, basso sulle altre.
"""

import os
import sys
import numpy as np
import pandas as pd
import glob
from datetime import datetime
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_functions import train_sac
from src.utils.constants import *
from src.utils.core import *
from src.evaluation.evaluate_functions import evaluate_sac_performance
from gymnasium.wrappers import TransformObservation

#-15259.69

def get_intelligent_noise_config():
    """Configurazione rumore intelligente basata su analisi."""
    config = {
        # ALTO RUMORE (0.8) - Variabili che beneficiano
        11: 0.8,  # net_electricity_consumption (+27%)
        8: 0.8,   # non_shiftable_load (+24%) 
        19: 0.8,  # dhw_demand (+22%)
        9: 0.8,   # solar_generation (+19%)
        3: 0.8,   # outdoor_dry_bulb_temperature (+15%)
                
        # BASSO RUMORE (0.15) - Variabili neutre/sensibili ma non dannose
        4: 0.15,   # outdoor_relative_humidity
        15: 0.15,  # indoor_dry_bulb_temperature
        16: 0.15,  # average_unmet_cooling_setpoint_difference
        20: 0.15,  # cooling_device_efficiency
        21: 0.15,  # dhw_device_efficiency
        7:  0.65,   # carbon_intensity (+18%) 0.5 è buono 0.65 meglio
        
    }
    
    # ESCLUSE (nessun rumore) - Variabili temporali e problematiche
    excluded = [
        0,   # day_type (temporale)
        1,   # hour (temporale)
        2,   # occupant_count (comportamentale)
        5,   # diffuse_solar_irradiance (problematica)
        6,   # direct_solar_irradiance (problematica)
        10,  # electrical_storage_soc
        12,  # electricity_pricing (molto dannosa)
        13,  # cooling_storage_soc
        14,  # dhw_storage_soc
        17,  # indoor_relative_humidity
        18,  # cooling_demand (molto dannosa)
        -4,  # power_outage (protetta)
        -3,  # cooling_set_point (protetta)
        -2,  # heating_set_point (protetta)
        -1,  # dhw_set_point (protetta)
    ]
    
    return config, excluded


def add_intelligent_noise(observations, noise_config, excluded_vars, name=None):
    """Applica rumore intelligente alle osservazioni."""
    noisy_obs = observations.copy()
    
    for var_index, noise_level in noise_config.items():
        # Salta se esclusa o fuori range
        if var_index in excluded_vars or var_index >= len(noisy_obs) or var_index < 0:
            continue
            
        # Applica rumore
        noise = np.random.normal(0, noise_level)
        original = noisy_obs[var_index]
        noisy_obs[var_index] = np.clip(original + noise, 0.0, 1.0)
        
        # Debug per prime iterazioni
        if name and False:  # Disabilitato per performance
            print(f"[{name}] Var {var_index}: {original:.3f} -> {noisy_obs[var_index]:.3f} (noise: {noise_level})")
    
    return noisy_obs


def create_intelligent_env(seed=100):
    """Crea ambiente con rumore intelligente."""
    print("🧠 Creando ambiente con rumore intelligente...")
    
    np.random.seed(seed)
    env = CityLearnEnv(**ENV_CONFIG)
    env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
    
    noise_config, excluded = get_intelligent_noise_config()
    
    # Conta configurazione
    high_noise = sum(1 for level in noise_config.values() if level >= 0.7)
    low_noise = sum(1 for level in noise_config.values() if level < 0.7)
    
    print(f"   📈 {high_noise} variabili con alto rumore (0.8)")
    print(f"   📉 {low_noise} variabili con basso rumore (0.15)")
    print(f"   🚫 {len(excluded)} variabili escluse")
    
    env = TransformObservation(
        env, 
        f=partial(add_intelligent_noise, noise_config=noise_config, excluded_vars=excluded)
    )
    env.reset(seed=seed)
    
    print("   ✅ Ambiente con rumore intelligente creato")
    return env


def find_latest_csv():
    """Trova ultimo CSV single-variable."""
    csv_dir = os.path.join(RESULTS_DIR, CSV_DIR)
    files = glob.glob(os.path.join(csv_dir, "single_var_test_*.csv"))
    return max(files, key=os.path.getctime)


def save_result_to_csv(test_result, model_path, seed=100):
    """Salva risultato nel CSV esistente."""
    result = {
        'model_name': 'intelligent_noise_model',
        'variable_name': 'INTELLIGENT_NOISE',
        'variable_index': -777,  # Codice speciale
        'noise_level': 0.45,  # Livello medio (mix 0.8 + 0.1)
        'seed': seed,
        'model_type': 'intelligent',
        'total_reward': test_result['total_reward'],
        'test_reward': test_result['total_reward'],
        'model_path': model_path,
        'error': '',
        'name': 'intelligent_noise_model',
        'episode_rewards': [],
        'episode_lengths': []
    }
    
    # Aggiungi al CSV
    csv_path = find_latest_csv()
    df = pd.read_csv(csv_path)
    
    # Rimuovi se già presente
    df = df[~df['variable_name'].str.contains('INTELLIGENT', na=False)]
    
    # Aggiungi nuovo
    df_new = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df_new.to_csv(csv_path, index=False)
    
    print(f"   ✅ Aggiunto al CSV: {os.path.basename(csv_path)}")


def main():
    """Funzione principale."""
    print("🧠 === INTELLIGENT NOISE MODEL ===")
    print("Alto rumore dove conviene, basso sulle altre")
    
    seed = int(input("Seed (default 100): ") or "100")
    episodes = int(input("Episodi (default 200): ") or "200")
    
    try:
        # Mostra configurazione
        noise_config, excluded = get_intelligent_noise_config()
        print(f"\n📋 CONFIGURAZIONE:")
        
        high_vars = [k for k, v in noise_config.items() if v >= 0.7]
        low_vars = [k for k, v in noise_config.items() if v < 0.7]
        
        print(f"   🔥 Alto rumore (0.8): variabili {high_vars}")
        print(f"   🔸 Basso rumore (0.15): variabili {low_vars}")
        print(f"   🚫 Escluse: {excluded}")
        
        confirm = input(f"\nProcedere? (y/n): ").lower()
        if confirm not in ['y', 'yes', 's', 'si']:
            print("❌ Annullato")
            return
        
        # 1. Training
        print(f"\n🏋️ Training con seed={seed}, episodi={episodes}")
        train_env = create_intelligent_env(seed)
        
        _, model, rewards, _ = train_sac(train_env, seed=seed, episodes=episodes, track_rewards=True)
        print(f"   ✅ Training completato: reward finale = {rewards[-1]:.2f}")
        
        # 2. Test su ambiente pulito
        print(f"\n🧪 Test su ambiente pulito")
        test_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(CityLearnEnv(**ENV_CONFIG)))
        test_env.reset(seed=seed)
        
        result = evaluate_sac_performance(test_env, model, episode_name="intelligent_noise_test")
        print(f"   ✅ Test reward: {result['total_reward']:.2f}")
        
        # 3. Salva modello
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(RESULTS_DIR, "models", f"intelligent_noise_{timestamp}.zip")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"   💾 Salvato: {os.path.basename(model_path)}")
        
        # 4. Aggiungi al CSV
        save_result_to_csv(result, model_path, seed)
        
        print(f"\n✅ COMPLETATO! Reward: {result['total_reward']:.2f}")
        print("📊 Ora confrontabile nei boxplot come 'INTELLIGENT_NOISE'!")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()