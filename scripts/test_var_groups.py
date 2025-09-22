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

from src.utils.noise import add_noise_to_observations, _NOISE_CONFIG as NOISE_CONFIG

def create_intelligent_env(seed=100, dinamic_noise=False):
    """Crea ambiente con rumore intelligente."""
    print("🧠 Creando ambiente con rumore intelligente...")
    
    env = CityLearnEnv(**ENV_CONFIG)
    env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
    
    def _obs_with_intelligent_noise(obs):
        arr = np.array(obs, dtype=float)
        return add_noise_to_observations(arr, noise_type='gaussian', dinamic_noise=dinamic_noise, name=f"intelligent_{seed}", seed=seed)

    env = TransformObservation(env, f=_obs_with_intelligent_noise)
    env.reset(seed=seed)
    
    print("   ✅ Ambiente con rumore intelligente creato")
    return env


def find_latest_csv():
    """Trova ultimo CSV single-variable."""
    csv_dir = os.path.join(RESULTS_DIR, CSV_DIR)
    files = glob.glob(os.path.join(csv_dir, "single_var_test_*.csv"))
    if not files:
        raise FileNotFoundError(f"Nessun CSV trovato in {csv_dir} matching single_var_test_*.csv")
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
    episodes = int(input("Episodi (default 35): ") or "35")

    try:
        # Mostra configurazione dai valori in NOISE_CONFIG
        noise_config = dict(NOISE_CONFIG)
        excluded = []
        print(f"\n📋 CONFIGURAZIONE:")
        
        high_vars = [k for k, v in noise_config.items() if v >= 0.7]
        low_vars = [k for k, v in noise_config.items() if v < 0.7]
        
        print(f"   🔥 Alto rumore (>=0.7): variabili {high_vars}")
        print(f"   🔸 Basso rumore (<0.7): variabili {low_vars}")
        
        confirm = input(f"\nProcedere? (Y/n): ").lower() or 'y'
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