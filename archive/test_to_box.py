#!/usr/bin/env python3
"""
Script per generare box plot comparativi di modelli SAC.
Esegue N training offline con seed diversi, poi testa tutti i modelli su un ambiente pulito.
Genera box plot per confrontare le performance e salva i risultati in CSV.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from pathlib import Path

# Import delle funzioni custom
from src.training.train_functions import *
from constants import *

def run_multiple_offline_training(n_runs):
    """
    Esegue N training offline con seed diversi
    
    Parameters:
    n_runs: int - Numero di training da eseguire
    
    Returns:
    list: Lista di dizionari contenenti modelli e dati per ogni run
    """
    print(f"🚀 AVVIO {n_runs} TRAINING OFFLINE")
    print("="*60)
    
    all_runs_data = []
    
    for i in range(n_runs):
        print(f"\n📈 Training {i+1}/{n_runs}")
        print("-" * 30)
        
        # Genera seed per questo run
        current_seed = generate_seed()
        print(f"🎲 SEED: {current_seed}")
        
        try:
            # Esegui training offline
            models, training_data = train_offline_models(current_seed)
            
            # Salva i dati di questo run
            run_data = {
                'run_id': i + 1,
                'seed': current_seed,
                'models': models,
                'training_data': training_data
            }
            all_runs_data.append(run_data)
            
            print(f"✅ Training {i+1} completato con seed {current_seed}")
            
        except Exception as e:
            print(f"❌ Errore nel training {i+1} (seed {current_seed}): {e}")
            continue
    
    print(f"\n🎯 Completati {len(all_runs_data)}/{n_runs} training")
    return all_runs_data

def evaluate_all_runs_on_clean_env(all_runs_data):
    """
    Testa tutti i modelli di tutti i run su un ambiente pulito
    
    Parameters:
    all_runs_data: list - Dati di tutti i run
    
    Returns:
    dict: Risultati di valutazione organizzati per modello
    """
    print("\n🧪 VALUTAZIONE SU AMBIENTE PULITO")
    print("="*60)
    
    # Crea ambiente pulito per il test
    test_env = CityLearnEnv(**ENV_CONFIG)
    test_env = StableBaselines3Wrapper(NormalizedSpaceWrapper(test_env))
    
    # Struttura per raccogliere i risultati
    # results[model_type][run_id] = reward_finale
    results = {}
    detailed_results = []
    
    for run_data in all_runs_data:
        run_id = run_data['run_id']
        seed = run_data['seed']
        models = run_data['models']
        
        print(f"\n🔍 Testando Run {run_id} (SEED: {seed})")
        
        for model_name, model in models.items():
            print(f"  • Testando {model_name}...")
            
            try:
                # Valuta il modello sull'ambiente pulito
                result = evaluate_sac_performance(
                    test_env, 
                    model, 
                    f"Run{run_id}-{model_name}"
                )
                
                # Organizza i risultati per tipo di modello
                if model_name not in results:
                    results[model_name] = []
                
                final_reward = result['total_reward']
                results[model_name].append(final_reward)
                
                # Salva dettagli per CSV
                detailed_results.append({
                    'run_id': run_id,
                    'seed': seed,
                    'model_type': model_name,
                    'final_reward': final_reward,
                    'steps_completed': result.get('steps_completed', 0),
                    'average_step_reward': result.get('total_reward', 0) / max(1, result.get('steps_completed', 1))
                })
                
                print(f"    ✅ {model_name}: {final_reward:.2f}")
                
            except Exception as e:
                print(f"    ❌ Errore nel test di {model_name}: {e}")
                continue
            
            # Reset ambiente per prossimo test
            test_env.reset()
    
    print(f"\n🎯 Valutazione completata!")
    print(f"📊 Raccolti dati per {len(results)} tipi di modelli")
    for model_type, rewards in results.items():
        print(f"  • {model_type}: {len(rewards)} test")
    
    return results, detailed_results

def create_box_plot(results, save_path):
    """
    Crea box plot comparativo dei risultati
    
    Parameters:
    results: dict - Risultati organizzati per tipo di modello
    save_path: str - Percorso di salvataggio
    """
    print(f"\n📊 CREAZIONE BOX PLOT")
    print("-" * 30)
    
    # Prepara i dati per il box plot
    plot_data = []
    for model_type, rewards in results.items():
        for reward in rewards:
            plot_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'Final Reward': reward
            })
    
    if not plot_data:
        print("❌ Nessun dato per creare il box plot!")
        return
    
    # Crea DataFrame
    df = pd.DataFrame(plot_data)
    
    # Crea figura
    plt.figure(figsize=(12, 8))
    
    # Box plot
    sns.boxplot(data=df, x='Model', y='Final Reward', palette='Set2')
    
    # Aggiungi punti individuali
    sns.stripplot(data=df, x='Model', y='Final Reward', 
                  color='black', alpha=0.6, size=4)
    
    # Personalizzazione
    plt.title('Confronto Performance Modelli SAC\n(Multiple Training Runs)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tipo di Modello', fontsize=12, fontweight='bold')
    plt.ylabel('Reward Finale su Ambiente Test', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Statistiche sul grafico
    stats_text = "Statistiche:\n"
    for model_type, rewards in results.items():
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        n_runs = len(rewards)
        stats_text += f"{model_type}: μ={mean_reward:.1f}±{std_reward:.1f} (N={n_runs})\n"
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Layout e salvataggio
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Box plot salvato: {save_path}")

def save_results_to_csv(detailed_results, save_path):
    """
    Salva i risultati dettagliati in CSV
    
    Parameters:
    detailed_results: list - Lista di risultati dettagliati
    save_path: str - Percorso di salvataggio CSV
    """
    print(f"\n💾 SALVATAGGIO RISULTATI CSV")
    print("-" * 30)
    
    if not detailed_results:
        print("❌ Nessun risultato da salvare!")
        return
    
    # Crea DataFrame
    df = pd.DataFrame(detailed_results)
    
    # Pivot per avere modelli come colonne
    pivot_df = df.pivot_table(
        index=['run_id', 'seed'], 
        columns='model_type', 
        values='final_reward', 
        aggfunc='first'
    )
    
    # Reset index per avere run_id e seed come colonne
    pivot_df = pivot_df.reset_index()
    
    # Riordina colonne: prima run_id e seed, poi i modelli
    model_columns = [col for col in pivot_df.columns if col not in ['run_id', 'seed']]
    column_order = ['run_id', 'seed'] + sorted(model_columns)
    pivot_df = pivot_df[column_order]
    
    # Aggiungi metadati come commento
    metadata_lines = [
        f"# BOX PLOT EXPERIMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# TOTAL RUNS: {len(pivot_df)}",
        f"# MODEL TYPES: {', '.join(sorted(model_columns))}",
        f"# EPISODES PER TRAINING: {EPISODES}",
        "#"
    ]
    
    # Salva CSV con metadati
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        for line in metadata_lines:
            f.write(line + '\n')
        pivot_df.to_csv(f, index=False)
    
    print(f"✅ CSV salvato: {save_path}")
    print(f"📊 Dimensioni: {len(pivot_df)} run × {len(model_columns)} modelli")

def print_summary_statistics(results):
    """
    Stampa statistiche riassuntive
    """
    print("\n📈 STATISTICHE RIASSUNTIVE")
    print("="*60)
    
    for model_type, rewards in results.items():
        if not rewards:
            continue
            
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        n_runs = len(rewards)
        
        print(f"\n🤖 {model_type.upper()}:")
        print(f"  📊 N° Run:           {n_runs}")
        print(f"  📈 Media:            {mean_reward:.3f}")
        print(f"  📉 Deviazione Std:   {std_reward:.3f}")
        print(f"  🔽 Minimo:           {min_reward:.3f}")
        print(f"  🔼 Massimo:          {max_reward:.3f}")
        print(f"  📏 Range:            {max_reward - min_reward:.3f}")

def main():
    """
    Funzione principale
    """
    print("🎯 BOX PLOT GENERATOR per Modelli SAC")
    print("="*80)
    
    # Input numero di run
    if len(sys.argv) > 1:
        try:
            n_runs = int(sys.argv[1])
        except ValueError:
            print("❌ Errore: Il parametro deve essere un numero intero!")
            sys.exit(1)
    else:
        try:
            n_runs = int(input("🔢 Inserisci il numero di training da eseguire: "))
        except ValueError:
            print("❌ Errore: Inserisci un numero valido!")
            sys.exit(1)
    
    if n_runs < 1:
        print("❌ Errore: Il numero di run deve essere almeno 1!")
        sys.exit(1)
    
    print(f"🎯 Configurazione: {n_runs} training offline")
    
    # Setup directory di salvataggio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("plots", f"box_plot_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Directory risultati: {save_dir}")
    
    start_time = time.time()
    
    try:
        # 1. Esegui N training offline
        all_runs_data = run_multiple_offline_training(n_runs)
        
        if not all_runs_data:
            print("❌ Nessun training completato con successo!")
            return
        
        # 2. Testa tutti i modelli su ambiente pulito
        results, detailed_results = evaluate_all_runs_on_clean_env(all_runs_data)
        
        if not results:
            print("❌ Nessun risultato di test ottenuto!")
            return
        
        # 3. Crea box plot
        box_plot_path = os.path.join(save_dir, "models_comparison_boxplot.png")
        create_box_plot(results, box_plot_path)
        
        # 4. Salva risultati in CSV
        csv_path = os.path.join(save_dir, "test_results.csv")
        save_results_to_csv(detailed_results, csv_path)
        
        # 5. Stampa statistiche
        print_summary_statistics(results)
        
        # 6. Salva summary testuale
        summary_path = os.path.join(save_dir, "experiment_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BOX PLOT EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Numero run: {len(all_runs_data)}\n")
            f.write(f"Episodi per training: {EPISODES}\n\n")
            
            f.write("SEEDS UTILIZZATI:\n")
            f.write("-" * 20 + "\n")
            for run_data in all_runs_data:
                f.write(f"Run {run_data['run_id']}: {run_data['seed']}\n")
            
            f.write("\nSTATISTICHE FINALI:\n")
            f.write("-" * 20 + "\n")
            for model_type, rewards in results.items():
                if rewards:
                    f.write(f"{model_type}:\n")
                    f.write(f"  Media: {np.mean(rewards):.3f}\n")
                    f.write(f"  Std: {np.std(rewards):.3f}\n")
                    f.write(f"  Min: {np.min(rewards):.3f}\n")
                    f.write(f"  Max: {np.max(rewards):.3f}\n\n")
        
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