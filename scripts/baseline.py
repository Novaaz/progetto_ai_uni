"""
Script per addestrare un modello su ambiente pulito (senza rumore), 
testarlo e aggiungere i risultati al CSV single-variable.
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
from src.training.train_functions import train_sac
from stable_baselines3 import SAC

def find_latest_csv():
    """Trova l'ultimo CSV single-variable test."""
    csv_dir = os.path.join(RESULTS_DIR, CSV_DIR)
    test_files = glob.glob(os.path.join(csv_dir, "single_var_test_*.csv"))
    if not test_files:
        raise FileNotFoundError("Nessun file single_var_test trovato!")
    return max(test_files, key=os.path.getctime)

def create_clean_environment(seed=100):
    """Crea ambiente pulito con seed fisso."""
    print(f"🌍 Creando ambiente pulito (seed={seed})...")
    # Imposta seed
    np.random.seed(seed)
    
    env = CityLearnEnv(**ENV_CONFIG)
    env = StableBaselines3Wrapper(NormalizedSpaceWrapper(env))
    env.reset(seed=seed)
    
    print("   ✅ Ambiente pulito creato")
    return env

def train_clean_baseline_model(seed=100, episodes=200):
    """Addestra un nuovo modello SAC su ambiente pulito."""
    print(f"\n🏋️ === TRAINING MODELLO BASELINE PULITO ===")
    print(f"   Seed: {seed}")
    print(f"   Episodi: {episodes}")
    
    # 1. Crea ambiente pulito per training
    train_env = create_clean_environment(seed=seed)
    
    print("🔥 Avviando training SAC...")
    
    try:
        # 2. Addestra il modello
        env, trained_model, training_rewards, timesteps = train_sac(
            env=train_env,
            seed=seed,
            track_rewards=True,
            eval_freq=50,
            episodes=episodes
        )
        
        print(f"   ✅ Training completato!")
        print(f"   📈 Episodi: {len(training_rewards)}")
        print(f"   🎯 Reward finale: {training_rewards[-1]:.2f}")
        
        return trained_model, training_rewards, timesteps
        
    except Exception as e:
        print(f"   ❌ Errore durante training: {e}")
        return None, None, None

def test_baseline_model(model, seed=100):
    """Testa il modello baseline su ambiente pulito."""
    print(f"\n🧪 === TEST MODELLO BASELINE ===")
    
    # Crea nuovo ambiente pulito per test
    test_env = create_clean_environment(seed=seed)
    
    print("🔍 Testing modello su ambiente pulito...")
    
    try:
        # Testa
        result = evaluate_sac_performance(
            test_env, 
            model, 
            episode_name="baseline_clean_trained"
        )
        
        print(f"   ✅ Test reward: {result['total_reward']:.2f}")
        return result
        
    except Exception as e:
        print(f"   ❌ Errore durante test: {e}")
        return None

def save_baseline_model(model, results_dir):
    """Salva il modello baseline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"baseline_clean_trained_{timestamp}.zip"
    model_path = os.path.join(results_dir, "models", model_filename)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    print(f"   💾 Modello salvato: {model_filename}")
    return model_path

def prepare_baseline_result(test_result, training_rewards, model_path, seed=100):
    """Prepara risultati in formato compatibile con CSV single-variable."""
    
    baseline_result = {
        'model_name': 'baseline_clean_trained',
        'variable_name': 'BASELINE_CLEAN',
        'variable_index': -999,  # Codice speciale per baseline
        'noise_level': 0.0,  # Nessun rumore
        'seed': seed,
        'model_type': 'baseline',
        'total_reward': test_result['total_reward'],
        'test_reward': test_result['total_reward'],
        'model_path': model_path,
        'error': '',
        'name': 'baseline_clean_trained',
        'episode_rewards': test_result.get('episode_rewards', []),
        'episode_lengths': test_result.get('episode_lengths', [])
    }
    
    return baseline_result

def append_to_csv(new_result, csv_path):
    """Aggiunge il risultato baseline al CSV esistente."""
    print(f"\n💾 Aggiungendo baseline al CSV...")
    
    # Carica CSV esistente
    df = pd.read_csv(csv_path)
    print(f"   📊 CSV attuale: {len(df)} righe")
    
    # Controlla se baseline già presente
    baseline_exists = df[df['variable_name'] == 'BASELINE_CLEAN'].shape[0] > 0
    if baseline_exists:
        print("   ⚠️ Baseline già presente, sostituendo...")
        df = df[df['variable_name'] != 'BASELINE_CLEAN']
    
    # Aggiungi nuova riga
    df_new = pd.DataFrame([new_result])
    df_combined = pd.concat([df, df_new], ignore_index=True)
    
    # Salva
    df_combined.to_csv(csv_path, index=False)
    print(f"   ✅ CSV aggiornato: {len(df_combined)} righe")
    print(f"   📁 File: {os.path.basename(csv_path)}")

def main():
    """Funzione principale."""
    print("🎯 === TRAIN & TEST BASELINE CLEAN ENVIRONMENT ===")
    print("="*60)
    
    # Input parametri
    seed = int(input("Seed per training (default 100): ") or "100")
    episodes = int(input("Episodi training (default 200): ") or "200")
    
    try:
        # 1. Trova ultimo CSV
        csv_path = find_latest_csv()
        print(f"📂 CSV trovato: {os.path.basename(csv_path)}")
        
        # 2. Addestra modello baseline
        trained_model, training_rewards, timesteps = train_clean_baseline_model(
            seed=seed, 
            episodes=episodes
        )
        
        if trained_model is None:
            print("❌ Training fallito!")
            return
        
        # 3. Testa modello
        test_result = test_baseline_model(trained_model, seed=seed)
        if test_result is None:
            print("❌ Test fallito!")
            return
        
        # 4. Salva modello
        model_path = save_baseline_model(trained_model, RESULTS_DIR)
        
        # 5. Prepara risultati
        baseline_result = prepare_baseline_result(
            test_result, 
            training_rewards, 
            model_path, 
            seed=seed
        )
        
        # 6. Aggiungi al CSV
        append_to_csv(baseline_result, csv_path)
        
        print(f"\n✅ BASELINE COMPLETATO!")
        print(f"🎯 Training reward finale: {training_rewards[-1]:.2f}")
        print(f"🏆 Test reward: {baseline_result['total_reward']:.2f}")
        print(f"📈 Ora puoi confrontare con i modelli single-variable!")
        
        # Cleanup memoria
        del trained_model
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()