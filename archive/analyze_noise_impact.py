#!/usr/bin/env python3
"""
Analisi dell'impatto del rumore sulle performance dei modelli SAC.
Allena 10 modelli con livelli di rumore crescenti e li testa su ambiente pulito.
Genera grafico lineare che mostra la relazione tra rumore e performance finale.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
from pathlib import Path
from functools import partial

# Import delle funzioni custom
from src.training.train_functions import *
from constants import *



def test_all_models_on_clean_env(models_data, seed):
    """
    Testa tutti i modelli su un ambiente pulito
    
    Parameters:
    models_data: list - Lista di tuple (modello, dati_training, noise_std)
    seed: int - Seed per la riproducibilità del test
    
    Returns:
    list: Lista di risultati del test
    """
    print(f"\n🧪 TEST SU AMBIENTE PULITO")
    print("="*50)
    
    test_env = CityLearnEnv(**ENV_CONFIG)
    test_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(test_env))
    print(f"🔹 Ambiente di test: Pulito (no noise)")
    
    test_results = []
    
    for i, (model, training_data, noise_std) in enumerate(models_data):
        if model is None:
            print(f"🔸 Modello {i+1} | SALTATO (errore training)")
            continue
            
        model_id = training_data['model_id']
        print(f"🔸 Testing Modello {model_id:2d} | Trained con noise σ={noise_std:.4f}")
        
        try:
            result = evaluate_sac_performance(
                test_env, 
                model, 
                f"Model-{model_id:02d}-Noise{noise_std:.4f}"
            )
            
            test_result = {
                'model_id': model_id,
                'training_noise_std': noise_std,
                'test_final_reward': result['total_reward'],
                'test_step_rewards': result['step_rewards'],
                'test_steps_completed': len(result['step_rewards']),
                'training_episodes': len(training_data['rewards']),
                'training_final_episode_reward': training_data['rewards'][-1] if training_data['rewards'] else 0,
                'seed': seed
            }
            
            test_results.append(test_result)
            
            print(f"    ✅ Final Reward: {result['total_reward']:.2f}")
            
        except Exception as e:
            print(f"    ❌ Errore durante test: {e}")
            continue
        
        test_env.reset()
    
    print(f"\n🎯 Test completati: {len(test_results)} modelli")
    return test_results

def create_noise_impact_plot(test_results, save_path):
    """
    Crea grafico lineare dell'impatto del rumore sulle performance
    
    Parameters:
    test_results: list - Risultati dei test
    save_path: str - Percorso di salvataggio
    """
    print(f"\n📊 CREAZIONE GRAFICO IMPATTO RUMORE")
    print("-" * 40)
    
    if not test_results:
        print("❌ Nessun risultato per creare il grafico!")
        return
    
    # Ordina per livello di rumore crescente
    sorted_results = sorted(test_results, key=lambda x: x['training_noise_std'])
    
    # Estrai dati per il plot
    noise_levels = [r['training_noise_std'] for r in sorted_results]
    final_rewards = [r['test_final_reward'] for r in sorted_results]
    model_ids = [r['model_id'] for r in sorted_results]
    
    # Crea figura
    plt.figure(figsize=(12, 8))
      # Plot principale: linea + punti
    plt.plot(noise_levels, final_rewards, 
             'o-', linewidth=2.5, markersize=8, 
             color='#2E86AB', markerfacecolor='#F18F01', 
             markeredgecolor='#2E86AB', markeredgewidth=2,
             alpha=0.8, label='SAC Models Performance')
    
    # Aggiungi valori precisi per ogni punto
    for i, (noise, reward, model_id) in enumerate(zip(noise_levels, final_rewards, model_ids)):
        # Etichetta con ID modello (sopra)
        plt.annotate(f'M{model_id}', 
                    (noise, reward), 
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
        
        # Valore preciso del reward (sotto)
        plt.annotate(f'{reward:.1f}', 
                    (noise, reward), 
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', fontsize=8, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
      # Personalizzazione grafico
    plt.xlabel('Rumore Training (Standard Deviation)', fontsize=12, fontweight='bold')
    plt.ylabel('Final Reward su Ambiente Test', fontsize=12, fontweight='bold')
    plt.title('Impatto del Rumore di Training sulle Performance\n(Test su Ambiente Pulito)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Legenda in alto a destra
    plt.legend(loc='upper right', frameon=True, shadow=True, 
               fancybox=True, framealpha=0.9, fontsize=10)
    
    # Griglia
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Statistiche nel grafico (spostato più in basso per non sovrapporre legenda)
    if len(final_rewards) > 1:
        # Calcola trend
        correlation = np.corrcoef(noise_levels, final_rewards)[0, 1]
        slope = np.polyfit(noise_levels, final_rewards, 1)[0]
        
        stats_text = f"Statistiche:\n"
        stats_text += f"• Modelli testati: {len(test_results)}\n"
        stats_text += f"• Range rumore: {min(noise_levels):.3f} - {max(noise_levels):.3f}\n"
        stats_text += f"• Range reward: {min(final_rewards):.1f} - {max(final_rewards):.1f}\n"
        stats_text += f"• Correlazione: {correlation:.3f}\n"
        stats_text += f"• Trend: {'↘️ Negativo' if slope < 0 else '↗️ Positivo' if slope > 0 else '➡️ Neutro'}"
        
        plt.text(0.02, 0.70, stats_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Formattazione assi
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Layout e salvataggio
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grafico salvato: {save_path}")

def save_results_to_csv(test_results, save_path):
    """
    Salva i risultati in formato CSV
    
    Parameters:
    test_results: list - Risultati dei test
    save_path: str - Percorso di salvataggio
    """
    print(f"\n💾 SALVATAGGIO RISULTATI CSV")
    print("-" * 30)
    
    if not test_results:
        print("❌ Nessun risultato da salvare!")
        return
    
    # Crea DataFrame
    df = pd.DataFrame(test_results)
    
    # Ordina per livello di rumore
    df = df.sort_values('training_noise_std')
    
    # Riordina colonne per chiarezza
    column_order = [
        'model_id', 'training_noise_std', 'test_final_reward',
        'test_steps_completed', 'training_episodes', 
        'training_final_episode_reward', 'seed'
    ]
    df = df[column_order]
    
    # Aggiungi metadati come commenti
    metadata_lines = [
        f"# NOISE IMPACT ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# TOTAL MODELS: {len(df)}",
        f"# TRAINING EPISODES: {df['training_episodes'].iloc[0] if len(df) > 0 else 'N/A'}",
        f"# COMMON SEED: {df['seed'].iloc[0] if len(df) > 0 else 'N/A'}",
        f"# NOISE RANGE: {df['training_noise_std'].min():.4f} - {df['training_noise_std'].max():.4f}",
        "#",
        "# COLUMNS:",
        "# model_id: ID del modello (1-N)",
        "# training_noise_std: Rumore usato durante training (gaussian std)",
        "# test_final_reward: Reward finale su ambiente test pulito",
        "# test_steps_completed: Step completati durante test",
        "# training_episodes: Episodi di training completati",
        "# training_final_episode_reward: Reward ultimo episodio training",
        "# seed: Seed comune usato per tutti i modelli",
        "#"
    ]
    
    # Salva CSV con metadati
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        for line in metadata_lines:
            f.write(line + '\n')
        df.to_csv(f, index=False, float_format='%.6f')
    
    print(f"✅ CSV salvato: {save_path}")
    print(f"📊 Dati: {len(df)} modelli × {len(df.columns)} colonne")

def print_summary_analysis(test_results):
    """
    Stampa analisi riassuntiva dei risultati
    """
    print("\n📈 ANALISI RIASSUNTIVA")
    print("="*50)
    
    if not test_results:
        print("❌ Nessun risultato da analizzare!")
        return
    
    # Ordina per rumore crescente
    sorted_results = sorted(test_results, key=lambda x: x['training_noise_std'])
    
    noise_levels = [r['training_noise_std'] for r in sorted_results]
    final_rewards = [r['test_final_reward'] for r in sorted_results]
    
    print(f"🎯 Modelli analizzati: {len(test_results)}")
    print(f"🔢 Range rumore: {min(noise_levels):.4f} - {max(noise_levels):.4f}")
    print(f"🏆 Range performance: {min(final_rewards):.1f} - {max(final_rewards):.1f}")
    
    # Trova migliore e peggiore
    best_idx = np.argmax(final_rewards)
    worst_idx = np.argmin(final_rewards)
    
    best_result = sorted_results[best_idx]
    worst_result = sorted_results[worst_idx]
    
    print(f"\n🥇 MIGLIORE PERFORMANCE:")
    print(f"   Modello: {best_result['model_id']}")
    print(f"   Rumore training: {best_result['training_noise_std']:.4f}")
    print(f"   Reward finale: {best_result['test_final_reward']:.2f}")
    
    print(f"\n🥉 PEGGIORE PERFORMANCE:")
    print(f"   Modello: {worst_result['model_id']}")
    print(f"   Rumore training: {worst_result['training_noise_std']:.4f}")
    print(f"   Reward finale: {worst_result['test_final_reward']:.2f}")
    
    # Calcola correlazione
    if len(final_rewards) > 1:
        correlation = np.corrcoef(noise_levels, final_rewards)[0, 1]
        print(f"\n📊 CORRELAZIONE RUMORE-PERFORMANCE: {correlation:.3f}")
        if abs(correlation) > 0.7:
            trend = "forte"
        elif abs(correlation) > 0.3:
            trend = "moderata"
        else:
            trend = "debole"
        
        direction = "negativa" if correlation < 0 else "positiva"
        print(f"    Correlazione {trend} {direction}")
        
        if correlation < -0.3:
            print("    ➡️ Il rumore tende a PEGGIORARE le performance")
        elif correlation > 0.3:
            print("    ➡️ Il rumore tende a MIGLIORARE le performance")
        else:
            print("    ➡️ Il rumore ha IMPATTO LIMITATO sulle performance")

def main():
    """
    Funzione principale per l'analisi dell'impatto del rumore
    """
    print("🎯 ANALISI IMPATTO RUMORE su Modelli SAC")
    print("="*60)
    
    # Configurazione esperimento
    n_models = 10
    min_noise = 0.0
    max_noise = 1.0
    common_seed = generate_seed()
    
    print(f"📋 CONFIGURAZIONE ESPERIMENTO:")
    print(f"   • Numero modelli: {n_models}")
    print(f"   • Range rumore: {min_noise:.3f} - {max_noise:.3f}")
    print(f"   • Seed comune: {common_seed}")
    print(f"   • Episodi training: {EPISODES}")
    
    # Genera livelli di rumore
    noise_levels = generate_noise_levels(n_models, min_noise, max_noise)
    print(f"\n🔊 LIVELLI DI RUMORE:")
    for i, noise in enumerate(noise_levels):
        print(f"   Modello {i+1:2d}: σ = {noise:.4f}")
    
    # Setup directory di salvataggio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("plots", f"noise_analysis_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📁 Directory risultati: {save_dir}")
    
    start_time = time.time()
    
    try:
        # 1. TRAINING FASE
        print(f"\n🚀 FASE 1: TRAINING {n_models} MODELLI")
        print("="*50)
        
        models_data = []
        for i, noise_std in enumerate(noise_levels):
            model, training_data, noise_used = train_model_with_noise(
                noise_std, common_seed, i+1
            )
            models_data.append((model, training_data, noise_used))
        
        # Filtra modelli validi
        valid_models = [(m, t, n) for m, t, n in models_data if m is not None]
        print(f"\n✅ Training completato: {len(valid_models)}/{n_models} modelli validi")
        
        if not valid_models:
            print("❌ Nessun modello allenato con successo!")
            return
        
        # 2. TEST FASE
        print(f"\n🧪 FASE 2: TEST SU AMBIENTE PULITO")
        print("="*50)
        
        test_results = test_all_models_on_clean_env(valid_models, common_seed)
        
        if not test_results:
            print("❌ Nessun test completato con successo!")
            return
        
        # 3. VISUALIZZAZIONE E SALVATAGGIO
        print(f"\n📊 FASE 3: ANALISI E SALVATAGGIO")
        print("="*50)
        
        # Crea grafico impatto rumore
        plot_path = os.path.join(save_dir, "noise_impact_analysis.png")
        create_noise_impact_plot(test_results, plot_path)
        
        # Salva risultati CSV
        csv_path = os.path.join(save_dir, "noise_analysis_results.csv")
        save_results_to_csv(test_results, csv_path)
        
        # Analisi riassuntiva
        print_summary_analysis(test_results)
        
        # Salva summary testuale
        summary_path = os.path.join(save_dir, "analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("NOISE IMPACT ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelli testati: {len(test_results)}\n")
            f.write(f"Seed comune: {common_seed}\n")
            f.write(f"Episodi training: {EPISODES}\n")
            f.write(f"Range rumore: {min_noise:.4f} - {max_noise:.4f}\n\n")
            
            f.write("RISULTATI PER MODELLO:\n")
            f.write("-" * 30 + "\n")
            sorted_results = sorted(test_results, key=lambda x: x['training_noise_std'])
            for result in sorted_results:
                f.write(f"Modello {result['model_id']:2d} | ")
                f.write(f"Noise: {result['training_noise_std']:.4f} | ")
                f.write(f"Final Reward: {result['test_final_reward']:8.2f}\n")
            
            # Aggiungi correlazione
            if len(test_results) > 1:
                noise_vals = [r['training_noise_std'] for r in test_results]
                reward_vals = [r['test_final_reward'] for r in test_results]
                correlation = np.corrcoef(noise_vals, reward_vals)[0, 1]
                f.write(f"\nCorrelazione rumore-performance: {correlation:.3f}\n")
        
        print(f"✅ Summary salvato: {summary_path}")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante l'esperimento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️ Esperimento completato in {duration/60:.2f} minuti")
        print(f"📁 Tutti i risultati salvati in: {save_dir}")

if __name__ == "__main__":
    main()