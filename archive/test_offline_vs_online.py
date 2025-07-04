"""
Test comparativo tra training offline tradizionale e fine-tuning online.
Struttura più pulita e modulare rispetto a test_online_tuning.py
"""

from src.training.train_functions import *
import custom_plot as cp
import os
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from stable_baselines3 import SAC
import time
from datetime import datetime

seed = 0

def apply_online_finetuning(models, target_env):
    """
    Applica fine-tuning online ai modelli pre-allenati.
    
    Parametri:
    models: dict - Modelli pre-allenati
    target_env: CityLearnEnv - Ambiente target per il fine-tuning
    
    Returns:
    dict: Modelli con fine-tuning applicato
    """
    print("="*60)
    print("FASE 2: FINE-TUNING ONLINE")
    print("="*60)
    
    finetuned_models = {}
    online_training_data = {}
    
    for model_name, model in models.items():
        print(f"\n🔄 Fine-tuning del modello: {model_name.upper()}")
        with TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, f"model_{model_name}.zip")
            model.save(model_path)
            finetuned_model = SAC.load(model_path, **FINETUNING_KWARGS)
            finetuned_model.set_env(target_env)
            
            _, finetuned_model, training_rewards, timesteps = train_sac(
                env=target_env,
                sac_model=finetuned_model,
                episodes=1,
                track_rewards=True,
                eval_freq=50
            )

            finetuned_models[f"{model_name}_finetuned"] = finetuned_model

        total_reward = sum(training_rewards) if training_rewards else 0
        
        online_training_data[f"{model_name}_finetuned"] = {
            'step_rewards': training_rewards,
            'total_reward': total_reward,
            'final_observations': None,
            'steps_completed': len(training_rewards),
            'detailed_rewards': []
        }

        target_env.reset()
    
    print("\n✅ Fine-tuning online completato!")
    return finetuned_models, online_training_data

def evaluate_all_models(original_models, finetuned_models, test_env):
    """
    Valuta tutti i modelli (originali e con fine-tuning) sull'ambiente di test.
    
    Parametri:
    original_models: dict - Modelli originali
    finetuned_models: dict - Modelli con fine-tuning
    test_env: CityLearnEnv - Ambiente di test
    
    Returns:
    dict: Risultati di valutazione per tutti i modelli
    """
    print("="*60)
    print("FASE 3: VALUTAZIONE COMPARATIVA")
    print("="*60)
    
    evaluation_results = {}
    
    # Valuta modelli originali
    print("\n📊 Valutazione modelli ORIGINALI:")
    for model_name, model in original_models.items():
        print(f"\n🔍 Valutando {model_name.upper()}...")
        result = evaluate_sac_performance(test_env, model, f"Original-{model_name}")
        evaluation_results[f"Original-{model_name}"] = result
        test_env.reset()
    
    # Valuta modelli con fine-tuning
    print("\n📊 Valutazione modelli FINE-TUNED:")
    for model_name, model in finetuned_models.items():
        print(f"\n🔍 Valutando {model_name.upper()}...")
        result = evaluate_sac_performance(test_env, model, f"Finetuned-{model_name}")
        evaluation_results[f"Finetuned-{model_name}"] = result
        test_env.reset()
    
    return evaluation_results

def plot_models_training(models_data, model_type, save_dir):
    """
    Funzione generica per plottare un gruppo di modelli (originali o fine-tuned).
    
    Parametri:
    models_data: dict - Dati dei modelli da plottare
    model_type: str - Tipo di modelli ("Original" o "Finetuned")
    save_dir: str - Directory di salvataggio
    """
    if not models_data:
        print(f"⚠️ Nessun dato per modelli {model_type}")
        return
    
    print(f"📈 Generando grafico per modelli {model_type}...")
    
    # Genera il grafico usando la funzione esistente
    cp.plot_post_training_rewards(
        models_data, 
        save_dir=save_dir,
        title_suffix=f" - {model_type} Models"
    )
    
    # Salva con nome specifico
    filename = f"models_{model_type.lower()}_comparison.png"
    print(f"💾 Grafico salvato: {filename}")

def save_results(evaluation_results, training_data, online_training_data, save_dir):
    """
    Salva tutti i risultati e genera grafici separati.
    
    Parametri:
    evaluation_results: dict - Risultati di valutazione
    training_data: dict - Dati di training offline
    online_training_data: dict - Dati di training online
    save_dir: str - Directory di salvataggio
    """
    print("="*60)
    print("FASE 4: SALVATAGGIO RISULTATI")
    print("="*60)
    
    # Separa i risultati per tipo
    offline_results = {k: v for k, v in evaluation_results.items() if 'Original' in k}
    online_results = {k: v for k, v in evaluation_results.items() if 'Finetuned' in k}
    print(f"\n💾 Salvando risultati in: {save_dir}")
    
    try:
        training_curves_path = os.path.join(save_dir, "training_curves.png")
        create_training_curves_plot(training_data, training_curves_path)
        print(f"✅ Grafico curve di training salvato: {training_curves_path}")
    except Exception as e:
        print(f"⚠️ Errore nella creazione del grafico curve di training: {e}")

    plot_models_training(offline_results, "Original", save_dir)
    plot_models_training(online_results, "Finetuned", save_dir)
    
    print("📝 Generando riepilogo testuale...")
    summary_path = os.path.join(save_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("ESPERIMENTO: CONFRONTO OFFLINE vs ONLINE LEARNING\n")
        f.write("="*80 + "\n")
        f.write(f"Data esperimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"seed utilizzato: {seed}\n\n")
        
        f.write("RISULTATI MODELLI ORIGINALI (OFFLINE):\n")
        f.write("-" * 50 + "\n")
        for model_name, results in offline_results.items():
            f.write(f"• {model_name}: {results['total_reward']:.2f}\n")
        
        f.write("\nRISULTATI MODELLI FINE-TUNED (ONLINE):\n")
        f.write("-" * 50 + "\n")
        for model_name, results in online_results.items():
            f.write(f"• {model_name}: {results['total_reward']:.2f}\n")
        
        f.write("\nCONFRONTI MIGLIORAMENTO:\n")
        f.write("-" * 50 + "\n")
                # Confronta ogni modello originale con la sua versione fine-tuned
        for orig_key in offline_results.keys():
            base_name = orig_key.replace('Original-', '')
            finetuned_key = f"Finetuned-{base_name}_finetuned"
            
            if finetuned_key in online_results:
                orig_reward = offline_results[orig_key]['total_reward']
                finetuned_reward = online_results[finetuned_key]['total_reward']
                                
                if abs(orig_reward) > 0:
                    improvement = ((finetuned_reward - orig_reward) / abs(orig_reward)) * 100
                    f.write(f"• {base_name}: {improvement:+.2f}% "
                           f"({orig_reward:.2f} → {finetuned_reward:.2f})\n")
                    
    print(f"✅ Riepilogo salvato: {summary_path}")

def save_offline_rewards_to_csv(training_data, save_dir):
    """
    Salva i dati delle reward dei modelli offline in formato CSV con episodi come indice e modelli come colonne.
    
    Parametri:
    training_data: dict - Dati di training per tutti i modelli offline
    save_dir: str - Directory di salvataggio
    
    Returns:
    str: Path del file CSV salvato
    """
    # Prepara i dati per il DataFrame
    max_episodes = 0
    model_rewards = {}
    
    # Trova il numero massimo di episodi e prepara i dati
    for model_name, data in training_data.items():
        rewards = data['rewards']
        model_rewards[model_name] = rewards
        max_episodes = max(max_episodes, len(rewards))
    
    # Crea il DataFrame con episodi come indice
    df_data = {}
    for model_name, rewards in model_rewards.items():
        # Estendi con NaN se necessario per uniformare la lunghezza
        padded_rewards = rewards + [np.nan] * (max_episodes - len(rewards))
        df_data[model_name.replace('_', ' ').title()] = padded_rewards
    
    # Crea DataFrame
    df = pd.DataFrame(df_data)
    df.index.name = 'Episodio'
    df.index = df.index + 1  # Inizia episodi da 1 invece di 0
    
    # Crea riga di metadati come primo commento nel CSV
    metadata_line = f"# SEED: {seed}, EPISODI: {EPISODES}, DATA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Salva CSV
    csv_path = os.path.join(save_dir, "training_rewards.csv")
    
    # Scrivi prima i metadati come commento
    with open(csv_path, 'w') as f:
        f.write(metadata_line)
        # Aggiungi il DataFrame
        df.to_csv(f, index=True)
    
    print(f"📊 CSV modelli offline salvato: {csv_path}")
    return csv_path


def train_to_box_plot():
    
    print("🚀 AVVIO ESPERIMENTO: OFFLINE vs ONLINE LEARNING")
    print("="*80)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SAVE_DIR, f"box_plot_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Directory risultati: {run_dir}")
    
    start_time = time.time()
    seed = generate_seed()

    try:
        # 1. Training offline dei modelli
        original_models, training_data = train_offline_models(seed)
        
        # 2. Setup ambiente reale
        target_env = CityLearnEnv(**ENV_CONFIG)
        target_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(target_env))
        
        # 3. Fine-tuning online
        # finetuned_models, online_training_data = apply_online_finetuning(
        #     original_models, target_env
        # )
        
        # 4. Valutazione comparativa
        # evaluation_results = evaluate_all_models(
        #     original_models, finetuned_models, target_env
        # )
        
        # offline_results = {k: v for k, v in evaluation_results.items() if 'Original' in k}
        # online_results = {k: v for k, v in evaluation_results.items() if 'Finetuned' in k}
        
        # print("\n📊 CONFRONTO PERFORMANCE:")
        # print("-" * 40)
        # for orig_key in offline_results.keys():
        #     base_name = orig_key.replace('Original-', '')
        #     finetuned_key = f"Finetuned-{base_name}_finetuned"
            
        #     if finetuned_key in online_results:
        #         orig_reward = offline_results[orig_key]['total_reward']
        #         finetuned_reward = online_results[finetuned_key]['total_reward']
                
        #         print(f"\n{base_name.upper()}:")
        #         print(f"  Offline:  {orig_reward:.2f}")
        #         print(f"  Online:   {finetuned_reward:.2f}")
                
        #         if abs(orig_reward) > 0:
        #             improvement = ((finetuned_reward - orig_reward) / abs(orig_reward)) * 100
        #             direction = "↗️" if improvement > 0 else "↘️"
        #             print(f"  Cambio:   {improvement:+.2f}% {direction}")
        
        # # 6. Salvataggio risultati e grafici
        # save_results(evaluation_results, training_data, online_training_data, run_dir)
        
        # # 7. Debug parametri modelli
        # print("\n🔍 DEBUG PARAMETRI MODELLI:")
        # for name, model in original_models.items():
        #     print_model_params(model, f"Original-{name}")
        
        # for name, model in finetuned_models.items():
        #     print_model_params(model, f"Finetuned-{name}")
            
    except Exception as e:
        print(f"\n❌ ERRORE durante l'esperimento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️ Esperimento completato in {duration/60:.2f} minuti ({duration:.1f} secondi)")
        print(f"📁 Risultati salvati in: {run_dir}")


if __name__ == "__main__":
    main()